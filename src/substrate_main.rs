//! Explicit-batch, single-shot substrate laboratory. Legacy CLI is unchanged.
//! Shared historical modules expose more APIs than this bounded runner uses.
#![allow(dead_code)]
mod ascii_corpus;
mod baselines;
mod checkpoint;
mod config;
mod dataset;
mod field;
mod instrumentation;
mod intervention;
mod latent;
mod nca;
mod substrate;
mod tasks;
mod vocab;

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use checkpoint::{CheckpointManager, ModelManifest, SCHEMA_VERSION};
use config::TitanConfig;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use substrate::{Lesions, State, Substrate, SubstrateConfig, SubstrateKind};
use tasks::{TaskBatch, TaskEngine, TaskKind};
use vocab::TokenInterface;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Campaign {
    name: String,
    task: String,
    seeds: Vec<u64>,
    arms: Vec<Arm>,
    length: usize,
    ticks: usize,
    steps: usize,
    batch_size: usize,
    fixed_tiny_batch: bool,
    eval_examples: usize,
    eval_lengths: Vec<usize>,
    eval_ticks: Vec<usize>,
    lr: f64,
    weight_decay: f64,
    channels: usize,
    nca_hidden: usize,
    transport: SubstrateConfig,
    stability_ticks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Nca,
    Directional,
    Multiscale,
}

impl Arm {
    fn name(&self) -> &'static str {
        match self {
            Self::Nca => "nca",
            Self::Directional => "directional",
            Self::Multiscale => "multiscale",
        }
    }
}

fn git(args: &[&str]) -> Result<String> {
    let out = std::process::Command::new("git").args(args).output()?;
    ensure!(out.status.success(), "git command failed");
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn take_rows(batch: &TaskBatch, indices: &[u32]) -> Result<TaskBatch> {
    let idx = Tensor::new(indices, batch.inputs.device())?;
    Ok(TaskBatch {
        inputs: batch.inputs.index_select(&idx, 0)?,
        targets: batch.targets.index_select(&idx, 0)?,
        loss_mask: batch.loss_mask.index_select(&idx, 0)?,
        prompt_text: String::new(),
        target_text: String::new(),
        carry_depths: None,
    })
}

/// Uses the existing generator and content-hash split unchanged, then deduplicates.
fn panel(engine: &TaskEngine, kind: TaskKind, length: usize, count: usize) -> Result<TaskBatch> {
    let raw =
        engine.generate_batch_seeded(kind, count * 32, length, true, 9_000_001, &Device::Cpu)?;
    let mut seen = HashSet::new();
    let indices: Vec<u32> = raw
        .inputs
        .to_vec2::<u32>()?
        .into_iter()
        .enumerate()
        .filter_map(|(i, x)| if seen.insert(x) { Some(i as u32) } else { None })
        .take(count)
        .collect();
    ensure!(indices.len() == count, "requested {count} unique heldout rows at L={length}, found {}; lower eval_examples explicitly", indices.len());
    take_rows(&raw, &indices)
}

fn forward(
    model: &Substrate,
    interface: &TokenInterface,
    inputs: &Tensor,
    ticks: usize,
    lesion: &Lesions,
) -> Result<(Tensor, State)> {
    let state = model.initialize(interface.embed_tokens(inputs)?)?;
    let state = model.evolve(&state, ticks, lesion)?;
    Ok((interface.logits(model.fine(&state))?, state))
}

// Evaluation never needs a BPTT tape. Detach each tick to bound memory independently
// of the evaluation horizon; this leaves all forward values unchanged.
fn detach_state(state: State) -> State {
    match state {
        State::Transport(scales) => State::Transport(scales.iter().map(Tensor::detach).collect()),
        State::Legacy(mut field) => {
            field.x = field.x.detach();
            field.slow_state = field.slow_state.map(|x| x.detach());
            field.seed = field.seed.map(|x| x.detach());
            State::Legacy(field)
        }
    }
}

fn evaluate(
    model: &Substrate,
    interface: &TokenInterface,
    inputs: &Tensor,
    ticks: usize,
    lesion: &Lesions,
) -> Result<(Tensor, State)> {
    let mut state = detach_state(model.initialize(interface.embed_tokens(inputs)?)?);
    for _ in 0..ticks {
        state = detach_state(model.step(&state, lesion)?);
    }
    Ok((interface.logits(model.fine(&state))?, state))
}

fn scores(logits: &Tensor, batch: &TaskBatch) -> Result<Value> {
    let pred = logits.argmax(2)?.to_vec2::<u32>()?;
    let targets = batch.targets.to_vec2::<u32>()?;
    let masks = batch.loss_mask.to_vec2::<f32>()?;
    let length = batch.inputs.dim(1)?;
    let mut hits = vec![0usize; length];
    let mut counts = vec![0usize; length];
    let mut exact = 0;
    let mut outcomes = Vec::new();
    for ((p, y), m) in pred.iter().zip(&targets).zip(&masks) {
        let mut ok = true;
        for j in 0..length {
            if m[j] > 0.5 {
                counts[j] += 1;
                hits[j] += usize::from(p[j] == y[j]);
                ok &= p[j] == y[j];
            }
        }
        exact += usize::from(ok);
        outcomes.push(ok);
    }
    let total: usize = counts.iter().sum();
    ensure!(total > 0, "empty query mask");
    Ok(
        json!({"query_accuracy": hits.iter().sum::<usize>() as f64 / total as f64,
        "exact_accuracy": exact as f64 / pred.len() as f64, "exact_correct":exact, "examples":pred.len(),
        "per_position": (0..length).filter(|&j|counts[j]>0).map(|j|json!({"position":j,"correct":hits[j],"total":counts[j],"accuracy":hits[j] as f64/counts[j] as f64})).collect::<Vec<_>>(),
        "exact_outcomes":outcomes, "predictions":pred}),
    )
}

fn diagnostics(model: &Substrate, state: &State, previous: Option<&State>) -> Result<Value> {
    let old = previous.map(|s| model.states(s));
    let mut entries = Vec::new();
    for (i, x) in model.states(state).iter().enumerate() {
        let v = x.flatten_all()?.to_vec1::<f32>()?;
        ensure!(
            v.iter().all(|x| x.is_finite()),
            "nonfinite state at scale {i}"
        );
        let rms = (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let delta = old
            .as_ref()
            .map(|o| -> Result<f32> {
                Ok((*x - o[i])?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
            })
            .transpose()?;
        entries.push(json!({"scale":i,"shape":x.dims(),"rms":rms,"max_abs":v.iter().map(|x|x.abs()).fold(0f32,f32::max),"fraction_abs_ge_1":v.iter().filter(|x|x.abs()>=1.0).count() as f64/v.len() as f64,"update_rms":delta}));
    }
    Ok(json!(entries))
}

/// Input-paired finite perturbation, measuring hidden states rather than decoded outputs.
fn impulse_probe(model: &Substrate, channels: usize, length: usize, ticks: usize) -> Result<Value> {
    let zero = Tensor::zeros((1, length, channels), DType::F32, &Device::Cpu)?;
    let mut values = vec![0f32; length * channels];
    // Perturb each family at source cell zero, retaining family-resolved measurements.
    for c in 0..channels {
        values[c] = 0.1;
    }
    let pulse = Tensor::from_vec(values, (1, length, channels), &Device::Cpu)?;
    let mut a = model.initialize(zero)?;
    let mut b = model.initialize(pulse)?;
    let mut rows = Vec::new();
    for tick in 0..=ticks {
        let aa = model.states(&a);
        let bb = model.states(&b);
        let mut scales = Vec::new();
        for (s, (x, y)) in aa.iter().zip(bb.iter()).enumerate() {
            let diff = (*y - *x)?.to_vec3::<f32>()?;
            let families: Vec<_> = (0..4)
                .map(|f| {
                    diff[0]
                        .iter()
                        .map(|cell| {
                            cell[f * channels / 4..(f + 1) * channels / 4]
                                .iter()
                                .map(|z| z.abs())
                                .fold(0f32, f32::max)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            scales.push(json!({"scale":s,"family_max_abs_by_position":families}));
        }
        rows.push(json!({"tick":tick,"scales":scales}));
        if tick < ticks {
            a = detach_state(model.step(&a, &Lesions::default())?);
            b = detach_state(model.step(&b, &Lesions::default())?);
        }
    }
    Ok(
        json!({"source":0,"amplitude":0.1,"measurement":"paired max absolute hidden difference; threshold 1e-7 is numerical, not useful information","length":length,"trace":rows}),
    )
}

fn config_for(c: &Campaign, arm: &Arm, seed: u64) -> (TitanConfig, SubstrateConfig) {
    let mut base = TitanConfig::default();
    base.field.seq_len = c.length;
    base.field.channels = c.channels;
    base.field.periodic_boundary = false;
    base.nca.hidden_dim = c.nca_hidden;
    base.nca.viscosity = 0.;
    base.nca.feedback_mode = "none".into();
    base.train.seed = seed;
    base.train.dev_steps = c.ticks;
    base.train.epochs = c.steps;
    base.train.batch_size = c.batch_size;
    base.train.lr = c.lr;
    base.train.weight_decay = c.weight_decay;
    let mut sub = c.transport.clone();
    sub.kind = if matches!(arm, Arm::Nca) {
        SubstrateKind::LegacyNca
    } else {
        SubstrateKind::Transport
    };
    sub.channels = c.channels;
    if !matches!(arm, Arm::Multiscale) {
        sub.scales = 1;
    }
    (base, sub)
}

fn budgets(c: &Campaign, sub: &SubstrateConfig, parameters: usize) -> Result<Value> {
    let lengths = sub.scale_lengths(c.length)?;
    let width = c.channels;
    let dormant = if sub.kind == SubstrateKind::Transport && lengths.len() == 1 {
        8 * width * width + 6 * width
    } else {
        0
    };
    let macs = if sub.kind == SubstrateKind::LegacyNca {
        c.length * 5 * width * c.nca_hidden
    } else {
        lengths.iter().sum::<usize>() * 3 * width * sub.hidden_dim
            + lengths.iter().skip(1).sum::<usize>() * 8 * width * width
    };
    Ok(
        json!({"allocated_parameters":parameters,"active_parameters":parameters-dormant,"dormant_exchange_parameters":dormant,
        "scale_lengths":lengths,"state_scalars_per_example":lengths.iter().sum::<usize>()*width,
        "channel_allocation":if sub.kind==SubstrateKind::Transport {Some(sub.channel_allocation())} else {None},
        "recurrent_dense_macs_per_tick_per_example":macs,"approx_recurrent_dense_flops_per_train_example":2*macs*c.ticks,
        "flop_scope":"multiply-add=2 FLOPs; excludes embedding, pointwise readout, nonlinearities, shifting, clamping, biases, backward and optimizer"}),
    )
}

fn counterfactual(
    model: &Substrate,
    interface: &TokenInterface,
    batch: &TaskBatch,
    ticks: usize,
    engine: &TaskEngine,
) -> Result<Value> {
    let rows = batch.inputs.to_vec2::<u32>()?;
    let targets = batch.targets.to_vec2::<u32>()?;
    let zero = engine.vocab.encode("0")[0] as u32;
    let one = engine.vocab.encode("1")[0] as u32;
    let lookup: std::collections::HashMap<_, _> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.clone(), i))
        .collect();
    let (logits, _) = evaluate(model, interface, &batch.inputs, ticks, &Lesions::default())?;
    let pred = logits.argmax(2)?.to_vec2::<u32>()?;
    let length = batch.inputs.dim(1)?;
    let mut results = Vec::new();
    for source in (0..length).filter(|j| j % 4 != 3) {
        let mut pairs = 0;
        let mut correct = vec![0usize; length / 4];
        let mut flips = vec![0usize; length / 4];
        for (i, row) in rows.iter().enumerate() {
            if row[source] != zero {
                continue;
            }
            let mut flipped = row.clone();
            flipped[source] = one;
            if let Some(&j) = lookup.get(&flipped) {
                pairs += 1;
                for slot in source / 4..length / 4 {
                    let q = slot * 4 + 3;
                    correct[slot] +=
                        usize::from(pred[i][q] == targets[i][q] && pred[j][q] == targets[j][q]);
                    flips[slot] += usize::from(pred[i][q] != pred[j][q]);
                }
            }
        }
        results.push(json!({"source":source,"heldout_pairs":pairs,"joint_correct_by_slot":correct,"prediction_flips_by_slot":flips}));
    }
    Ok(
        json!({"scope":"both members unique heldout panel; no train overlap; absent pairs unmeasured","sources":results}),
    )
}

fn run(
    c: &Campaign,
    arm: &Arm,
    seed: u64,
    root: &Path,
    engine: &TaskEngine,
    kind: TaskKind,
) -> Result<Value> {
    let dir = root.join(format!("{}_seed_{seed}", arm.name()));
    std::fs::create_dir(&dir)?;
    let (mut cfg, sub) = config_for(c, arm, seed);
    cfg.validate()?;
    let vm = VarMap::new();
    let device = Device::Cpu;
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let model = Substrate::new(vb.clone(), &sub, &cfg.nca, &cfg.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), engine.vocab.size(), c.channels)?;
    baselines::initialize_seeded(&vm, seed)?;
    let parameter_count = vm.all_vars().iter().map(|v| v.elem_count()).sum::<usize>();
    vm.save(dir.join("initial.safetensors"))?;
    let initial_hash =
        CheckpointManager::compute_file_hash(dir.join("initial.safetensors").to_str().unwrap());
    let started = Instant::now();
    let mut optimizer = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: c.lr,
            weight_decay: c.weight_decay,
            ..Default::default()
        },
    )?;
    let mut history = Vec::new();
    let mut observed = HashSet::new();
    let mut last_grad = 0.;
    for step in 0..c.steps {
        let data_seed = if c.fixed_tiny_batch {
            10_000
        } else {
            10_000 + step
        };
        let batch = engine.generate_batch_seeded(
            kind,
            c.batch_size,
            c.length,
            false,
            data_seed,
            &device,
        )?;
        observed.extend(batch.inputs.to_vec2::<u32>()?);
        let (logits, state) = forward(
            &model,
            &interface,
            &batch.inputs,
            c.ticks,
            &Lesions::default(),
        )?;
        let loss =
            interface.masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
        let last_loss = loss.to_scalar::<f32>()?;
        ensure!(last_loss.is_finite(), "nonfinite loss at {step}");
        let grads = loss.backward()?;
        let mut sum = 0f64;
        for var in vm.all_vars() {
            if let Some(g) = grads.get(&var) {
                for x in g.flatten_all()?.to_vec1::<f32>()? {
                    ensure!(x.is_finite(), "nonfinite gradient at {step}");
                    sum += (x as f64).powi(2);
                }
            }
        }
        last_grad = sum.sqrt() as f32;
        ensure!(last_grad.is_finite(), "nonfinite gradient norm");
        let last_acc = interface.masked_accuracy(&logits, &batch.targets, &batch.loss_mask)?;
        if step == 0 || (step + 1) % 25 == 0 || step + 1 == c.steps {
            history.push(json!({"step":step+1,"loss":last_loss,"query_accuracy":last_acc,"gradient_norm":last_grad,"states":diagnostics(&model,&state,None)?}));
            eprintln!(
                "{} seed{} step{} loss{:.4} accuracy{:.3}",
                arm.name(),
                seed,
                step + 1,
                last_loss,
                last_acc
            );
        }
        optimizer.step(&grads)?;
    }
    let train_seconds = started.elapsed().as_secs_f64();
    // Metrics are recomputed on final weights, distinct from the pre-update training trace.
    let train_batch =
        engine.generate_batch_seeded(kind, c.batch_size, c.length, false, 10_000, &device)?;
    let (train_logits, _) = evaluate(
        &model,
        &interface,
        &train_batch.inputs,
        c.ticks,
        &Lesions::default(),
    )?;
    let mut evals = Vec::new();
    for &length in &c.eval_lengths {
        let batch = panel(engine, kind, length, c.eval_examples)?;
        ensure!(
            batch
                .inputs
                .to_vec2::<u32>()?
                .iter()
                .all(|r| !observed.contains(r)),
            "train/eval overlap"
        );
        write_json(
            dir.join(format!("eval_inputs_L{length}.json")),
            &json!({"inputs":batch.inputs.to_vec2::<u32>()?,"targets":batch.targets.to_vec2::<u32>()?,"mask":batch.loss_mask.to_vec2::<f32>()?}),
        )?;
        for &ticks in &c.eval_ticks {
            let (logits, state) = evaluate(
                &model,
                &interface,
                &batch.inputs,
                ticks,
                &Lesions::default(),
            )?;
            let intact = scores(&logits, &batch)?;
            let b = batch.inputs.dim(0)?;
            let permutation: Vec<u32> = (0..b).map(|i| ((i + 1) % b) as u32).collect();
            let shuffled = logits.index_select(&Tensor::new(permutation, &device)?, 0)?;
            let shuffled_scores = scores(&shuffled, &batch)?;
            let loss = interface
                .masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?
                .to_scalar::<f32>()?;
            ensure!(loss.is_finite(), "nonfinite evaluation loss");
            let mut lesions = Vec::new();
            if ticks == c.ticks && !matches!(arm, Arm::Nca) {
                let tests = [
                    (
                        "no_transport",
                        Lesions {
                            disable_transport: true,
                            ..Default::default()
                        },
                    ),
                    (
                        "no_exchange",
                        Lesions {
                            disable_up: true,
                            disable_down: true,
                            ..Default::default()
                        },
                    ),
                    (
                        "no_up",
                        Lesions {
                            disable_up: true,
                            ..Default::default()
                        },
                    ),
                    (
                        "no_down",
                        Lesions {
                            disable_down: true,
                            ..Default::default()
                        },
                    ),
                    (
                        "no_reaction",
                        Lesions {
                            disable_reaction: true,
                            ..Default::default()
                        },
                    ),
                ];
                for (name, lesion) in tests {
                    let (l, _) = evaluate(&model, &interface, &batch.inputs, ticks, &lesion)?;
                    lesions.push(json!({"name":name,"scores":scores(&l,&batch)?}));
                }
                if matches!(arm, Arm::Multiscale) {
                    for scale in 1..model.states(&state).len() {
                        for shuffle in [false, true] {
                            let lesion = if shuffle {
                                Lesions {
                                    shuffle_scale: Some(scale),
                                    ..Default::default()
                                }
                            } else {
                                Lesions {
                                    reset_scale: Some(scale),
                                    ..Default::default()
                                }
                            };
                            let (l, _) =
                                evaluate(&model, &interface, &batch.inputs, ticks, &lesion)?;
                            lesions.push(json!({"name":if shuffle{"shuffle_scale"}else{"reset_scale"},"scale":scale,"scores":scores(&l,&batch)?}));
                        }
                    }
                }
            }
            let pairs = if kind == TaskKind::IteratedParity && ticks == c.ticks {
                Some(counterfactual(&model, &interface, &batch, ticks, engine)?)
            } else {
                None
            };
            evals.push(json!({"length":length,"ticks":ticks,"loss":loss,"identity_gap":intact["query_accuracy"].as_f64().unwrap()-shuffled_scores["query_accuracy"].as_f64().unwrap(),"intact":intact,"batch_derangement":shuffled_scores,"states":diagnostics(&model,&state,None)?,"lesions":lesions,"prefix_bit_counterfactuals":pairs}));
        }
    }
    let tiny = take_rows(&train_batch, &[0, 1])?;
    let mut state = model.initialize(interface.embed_tokens(&tiny.inputs)?)?;
    let mut stability = Vec::new();
    for t in 1..=c.stability_ticks {
        let next = detach_state(model.step(&state, &Lesions::default())?);
        if t.is_power_of_two() || t == c.stability_ticks {
            stability.push(json!({"tick":t,"scales":diagnostics(&model,&next,Some(&state))?}));
        }
        state = next;
    }
    write_json(dir.join("training.json"), &history)?;
    // Existing checkpoint schema, with explicit new substrate inside TitanConfig.
    cfg.substrate = Some(sub.clone());
    let final_train_loss = interface
        .masked_cross_entropy_loss(&train_logits, &train_batch.targets, &train_batch.loss_mask)?
        .to_scalar::<f32>()?;
    let final_train_accuracy =
        interface.masked_accuracy(&train_logits, &train_batch.targets, &train_batch.loss_mask)?;
    let mut manifest = ModelManifest {
        schema_version: SCHEMA_VERSION,
        git_commit: git(&["rev-parse", "HEAD"])?,
        random_seed: seed,
        cumulative_step: c.steps,
        dev_steps: c.ticks,
        train_loss: Some(final_train_loss),
        train_accuracy: Some(final_train_accuracy),
        val_loss: None,
        val_accuracy: None,
        grad_norm: Some(last_grad),
        state_energy: None,
        param_count: parameter_count,
        checkpoint_hash: String::new(),
        timestamp_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        config: cfg.clone(),
        task: c.task.clone(),
    };
    CheckpointManager::save(dir.to_str().unwrap(), &mut manifest, &vm)?;
    let loaded = CheckpointManager::load_manifest(dir.to_str().unwrap())?;
    let mut reload = VarMap::new();
    let vb = VarBuilder::from_varmap(&reload, DType::F32, &device);
    let restored = Substrate::new(
        vb.clone(),
        loaded
            .config
            .substrate
            .as_ref()
            .context("missing substrate")?,
        &loaded.config.nca,
        &loaded.config.field,
    )?;
    let restored_interface =
        TokenInterface::new(vb.pp("interface"), engine.vocab.size(), c.channels)?;
    CheckpointManager::load_weights(dir.to_str().unwrap(), &mut reload, &device)?;
    let (again, _) = evaluate(
        &restored,
        &restored_interface,
        &train_batch.inputs,
        c.ticks,
        &Lesions::default(),
    )?;
    let reload_error = (&again - &train_logits)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    ensure!(
        reload_error == 0.,
        "checkpoint changed outputs: {reload_error}"
    );
    let mut impulses = Vec::new();
    if seed == c.seeds[0] {
        for length in [16, 32, 64] {
            impulses.push(impulse_probe(&model, c.channels, length, 16)?);
        }
    }
    write_json(dir.join("transport.json"), &impulses)?;
    let result = json!({"arm":arm.name(),"seed":seed,"parameters":parameter_count,"budgets":budgets(c,&sub,parameter_count)?,"substrate":sub,"train_seconds":train_seconds,"total_seconds":started.elapsed().as_secs_f64(),"train_unique_inputs":observed.len(),"train_examples_seen":c.steps*c.batch_size,"initial_checkpoint_hash":initial_hash,"checkpoint_hash":manifest.checkpoint_hash,"roundtrip_max_abs_error":reload_error,"final_fixed_train_batch":scores(&train_logits,&train_batch)?,"evaluations":evals,"stability":stability});
    write_json(dir.join("results.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod lab_tests {
    use super::*;

    #[test]
    fn query_metrics_exclude_auxiliary_and_require_complete_answers() -> Result<()> {
        let batch = TaskBatch {
            inputs: Tensor::zeros((2, 2), DType::U32, &Device::Cpu)?,
            targets: Tensor::new(&[[1u32, 0], [0, 1]], &Device::Cpu)?,
            loss_mask: Tensor::new(&[[0.5f32, 1.0], [1.0, 1.0]], &Device::Cpu)?,
            prompt_text: String::new(),
            target_text: String::new(),
            carry_depths: None,
        };
        let logits = Tensor::new(
            &[[[2f32, 0.], [2., 0.]], [[2., 0.], [2., 0.]]],
            &Device::Cpu,
        )?;
        let score = scores(&logits, &batch)?;
        assert_eq!(score["query_accuracy"].as_f64().unwrap(), 2. / 3.);
        assert_eq!(score["exact_accuracy"], 0.5);
        Ok(())
    }

    #[test]
    fn inference_detachment_preserves_logits_and_config_rejects_legacy_loader() -> Result<()> {
        let mut cfg = TitanConfig::default();
        cfg.field.channels = 8;
        cfg.field.periodic_boundary = false;
        let sub = SubstrateConfig {
            channels: 8,
            hidden_dim: 12,
            ..Default::default()
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let model = Substrate::new(vb.clone(), &sub, &cfg.nca, &cfg.field)?;
        let interface = TokenInterface::new(vb.pp("interface"), 99, 8)?;
        baselines::initialize_seeded(&vm, 42)?;
        let input = Tensor::new(&[[1u32, 2, 3, 4, 5], [5, 4, 3, 2, 1]], &Device::Cpu)?;
        let a = forward(&model, &interface, &input, 4, &Lesions::default())?.0;
        let b = evaluate(&model, &interface, &input, 4, &Lesions::default())?.0;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        cfg.substrate = Some(sub);
        assert!(cfg.validate().is_err());
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && matches!(args[1].as_str(), "--help" | "-h") {
        println!("titan_substrate CONFIG.json OUTPUT_DIRECTORY\nFresh single-shot CPU campaign. Refuses an existing output directory; no optimizer resume.\nSee experiments/substrate_v1/*.json and docs/VNEXT_SUBSTRATE.md.");
        return Ok(());
    }
    ensure!(
        args.len() == 3,
        "usage: titan_substrate CONFIG.json OUTPUT_DIRECTORY"
    );
    let c: Campaign = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let kind = TaskKind::parse(&c.task).context("unknown task")?;
    ensure!(
        matches!(
            kind,
            TaskKind::IteratedParity | TaskKind::DelayedRecall | TaskKind::IteratedSumDense
        ),
        "unsupported research task"
    );
    ensure!(
        !c.seeds.is_empty()
            && !c.arms.is_empty()
            && c.steps > 0
            && c.batch_size >= 2
            && c.eval_examples >= 2
            && c.length >= 8
            && c.length % 4 == 0
            && c.ticks > 0
            && c.stability_ticks > 0,
        "invalid campaign dimensions"
    );
    ensure!(
        c.seeds.iter().collect::<BTreeSet<_>>().len() == c.seeds.len(),
        "duplicate seeds"
    );
    ensure!(
        c.arms
            .iter()
            .map(|a| a.name())
            .collect::<BTreeSet<_>>()
            .len()
            == c.arms.len(),
        "duplicate arms"
    );
    ensure!(
        !c.eval_lengths.is_empty()
            && c.eval_lengths.iter().all(|l| *l >= 8 && l % 4 == 0)
            && c.eval_ticks.contains(&c.ticks),
        "invalid evaluation grid"
    );
    ensure!(c.transport.channels == c.channels, "channel mismatch");
    for arm in &c.arms {
        let (cfg, s) = config_for(&c, arm, c.seeds[0]);
        cfg.validate()?;
        let vm = VarMap::new();
        Substrate::new(
            VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu),
            &s,
            &cfg.nca,
            &cfg.field,
        )?;
    }
    let root = Path::new(&args[2]);
    ensure!(
        !root.exists(),
        "output already exists; use a fresh directory"
    );
    let provenance = json!({"git_commit":git(&["rev-parse","HEAD"])?,"git_status":git(&["status","--porcelain=v1"])? ,"git_diff":git(&["diff","--no-ext-diff"])?,"backend":"Candle 0.4.1 CPU f32","hardware":std::process::Command::new("uname").arg("-a").output().ok().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_string()),"task_schema":tasks::TASK_SCHEMA_VERSION,"initialization":"baselines::initialize_seeded v1 sorted names; matched interface and B/C shared tensors","data":"existing TaskEngine content hash split; explicit fresh per-step seeds 10000+step, fixed seed10000 for tiny overfit; unique evaluation panel seed9000001","optimizer":"single uninterrupted AdamW default betas, explicit lr and weight_decay; no resume","args":args});
    std::fs::create_dir_all(root)?;
    write_json(root.join("config.json"), &c)?;
    write_json(root.join("provenance.json"), &provenance)?;
    let engine = TaskEngine::new();
    let mut results = Vec::new();
    for arm in &c.arms {
        for &seed in &c.seeds {
            match run(&c, arm, seed, root, &engine, kind) {
                Ok(r) => {
                    results.push(r);
                    write_json(root.join("summary.json"), &results)?;
                }
                Err(e) => {
                    write_json(
                        root.join("failure.json"),
                        &json!({"arm":arm.name(),"seed":seed,"error":format!("{e:#}")}),
                    )?;
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}
