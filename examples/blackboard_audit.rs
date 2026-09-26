//! Bounded CPU audit of the September 15 blackboard experiment.
//! Run: cargo run --release --locked --offline --example blackboard_audit -- OUTPUT.json
#![allow(dead_code)]
#[path = "../src/ascii_corpus.rs"]
pub mod ascii_corpus;
#[path = "../src/baselines.rs"]
mod baselines;
#[path = "../src/dataset.rs"]
mod dataset;
#[path = "../src/falsification.rs"]
mod falsification;
#[path = "../src/nca2d.rs"]
mod nca2d;
#[path = "../src/tasks.rs"]
mod tasks;
#[path = "../src/vocab.rs"]
mod vocab;

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use falsification::{ArmKind, DualSystemModel};
use nca2d::{
    DynamicsMode, ReadIntervention, SpatialCoordMode, TemporalIntervention, WriteIntervention,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    io::Write,
};
use tasks::{ArithmeticStratum, TaskBatch, TaskEngine};

fn rows(batch: &TaskBatch) -> Result<Vec<(Vec<u32>, Vec<u32>, Vec<f32>)>> {
    Ok(batch
        .inputs
        .to_vec2::<u32>()?
        .into_iter()
        .zip(batch.targets.to_vec2::<u32>()?)
        .zip(batch.loss_mask.to_vec2::<f32>()?)
        .map(|((x, y), m)| (x, y, m))
        .collect())
}

fn data_audit(engine: &TaskEngine, seed: usize) -> Result<Value> {
    let mut prior = vec![vec![0usize; engine.vocab.size()]; 32];
    let mut train_inputs = HashSet::new();
    for epoch in 0..8 {
        for step in 0..6 {
            let batch = engine.generate_stratified_arithmetic_batch(
                16,
                32,
                None,
                Some(3),
                seed * 1000 + epoch * 100 + step,
                &Device::Cpu,
            )?;
            for (x, y, m) in rows(&batch)? {
                train_inputs.insert(x);
                for j in 0..32 {
                    if m[j] == 1. {
                        prior[j][y[j] as usize] += 1;
                    }
                }
            }
        }
    }
    let predictions: Vec<_> = prior
        .iter()
        .map(|counts| {
            counts
                .iter()
                .enumerate()
                .max_by_key(|&(id, n)| (*n, std::cmp::Reverse(id)))
                .unwrap()
                .0 as u32
        })
        .collect();
    let zero = engine.vocab.encode("0")[0] as u32;
    let mut panels = Vec::new();
    for (label, stratum, max_depth, base_seed, n) in [
        ("validation", None, None, seed + 99999, 256),
        ("interventions", None, None, 12345, 256),
        ("in_distribution", None, Some(3), 54321, 128),
        (
            "long_partial",
            Some(ArithmeticStratum::LongPartial),
            None,
            54323,
            128,
        ),
        (
            "max_ripple",
            Some(ArithmeticStratum::MaxRipple),
            None,
            54322,
            128,
        ),
    ] {
        let mut unique = HashSet::new();
        let mut overlap = 0usize;
        let mut n_query = 0usize;
        let mut zero_correct = 0usize;
        let mut prior_correct = 0usize;
        let mut prior_exact = 0usize;
        let mut depths = BTreeMap::<usize, usize>::new();
        for offset in (0..n).step_by(32) {
            let batch = engine.generate_stratified_arithmetic_batch(
                32,
                32,
                stratum,
                max_depth,
                base_seed + offset,
                &Device::Cpu,
            )?;
            for d in batch.carry_depths.as_ref().unwrap() {
                *depths.entry(*d).or_default() += 1;
            }
            for (x, y, m) in rows(&batch)? {
                overlap += usize::from(train_inputs.contains(&x));
                unique.insert(x);
                let mut exact = true;
                for j in 0..32 {
                    if m[j] != 1. {
                        continue;
                    }
                    n_query += 1;
                    zero_correct += usize::from(y[j] == zero);
                    prior_correct += usize::from(y[j] == predictions[j]);
                    exact &= y[j] == predictions[j];
                }
                prior_exact += usize::from(exact);
            }
        }
        panels.push(
            json!({"panel":label,"n":n,"unique_observations":unique.len(),
            "train_overlap":overlap,"query_tokens":n_query,"carry_depth_counts":depths,
            "constant_zero_digit_acc":zero_correct as f64/n_query as f64,
            "train_fitted_position_prior_digit_acc":prior_correct as f64/n_query as f64,
            "train_fitted_position_prior_exact_acc":prior_exact as f64/n as f64}),
        );
    }
    Ok(json!({"seed":seed,"train_unique":train_inputs.len(),"panels":panels}))
}

fn model_audit(
    engine: &TaskEngine,
    arm: ArmKind,
    seed: u64,
    local_write_norm: bool,
) -> Result<Value> {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let mut model = DualSystemModel::new(
        VarBuilder::from_varmap(&vm, DType::F32, &dev),
        arm,
        engine.vocab.size(),
        32,
    )?;
    baselines::initialize_seeded(&vm, seed)?;
    if let Some(ref mut bb) = model.blackboard {
        bb.cfg.normalize_written_cell_only = local_write_norm;
    }
    let batch = engine.generate_stratified_arithmetic_batch(2, 32, None, Some(3), 123, &dev)?;
    let dynamics = if arm == ArmKind::Arm5StaticBuffer {
        DynamicsMode::StaticBuffer
    } else {
        DynamicsMode::FixedTicks(2)
    };
    let forward = |x: &Tensor, write| -> Result<Tensor> {
        Ok(model
            .forward(
                x,
                SpatialCoordMode::Hilbert,
                ReadIntervention::Active,
                write,
                TemporalIntervention::None,
                dynamics,
                false,
            )?
            .0)
    };
    let logits = forward(&batch.inputs, WriteIntervention::Active)?;
    let loss =
        model
            .interface
            .masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
    let gradients = loss.backward()?;
    let mut gradient_rows = Vec::new();
    for (name, var) in vm.data().lock().unwrap().iter() {
        if let Some(g) = gradients.get(var) {
            let values = g.flatten_all()?.to_vec1::<f32>()?;
            let finite = values.iter().all(|v| v.is_finite());
            gradient_rows.push(json!({"name":name,"finite":finite,
                "max_abs":if finite {Some(values.iter().fold(0f32,|a,v| a.max(v.abs())))} else {None}}));
        }
    }
    gradient_rows.sort_by_key(|v| v["name"].as_str().unwrap().to_string());
    let original = batch.inputs.narrow(0, 0, 1)?;
    let mut changed = original.to_vec2::<u32>()?;
    let seven = engine.vocab.encode("7")[0] as u32;
    changed[0][0] = if changed[0][0] == seven {
        seven + 1
    } else {
        seven
    };
    let changed = Tensor::new(changed, &dev)?;
    let original_logits = forward(&original, WriteIntervention::Active)?;
    let changed_logits = forward(&changed, WriteIntervention::Active)?;
    let max_abs =
        |x: Tensor| -> Result<f32> { Ok(x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?) };
    let answer_change =
        max_abs((original_logits.narrow(1, 20, 10)? - changed_logits.narrow(1, 20, 10)?)?)?;
    let disabled_change = max_abs(
        (forward(&original, WriteIntervention::Disabled)?.narrow(1, 20, 10)?
            - forward(&changed, WriteIntervention::Disabled)?.narrow(1, 20, 10)?)?,
    )?;
    let mut future = original.to_vec2::<u32>()?;
    future[0][25] = seven;
    let future_logits = forward(&Tensor::new(future, &dev)?, WriteIntervention::Active)?;
    let prefix_change =
        max_abs((original_logits.narrow(1, 0, 25)? - future_logits.narrow(1, 0, 25)?)?)?;
    ensure!(
        prefix_change == 0.,
        "future token influenced earlier logits"
    );
    Ok(
        json!({"arm":arm,"seed":seed,"local_write_norm":local_write_norm,"loss":loss.to_scalar::<f32>()?,
        "gradient_tensors":gradient_rows,"changed_operand_answer_logit_max_abs":answer_change,
        "write_disabled_operand_logit_max_abs":disabled_change,"future_prefix_logit_max_abs":prefix_change}),
    )
}

// Numerical/optimization gate only: repeatedly fit four examples, not a
// held-out benchmark. Explicitly compare a real cross-token GRU control.
fn optimization_gate(engine: &TaskEngine, seed: u64, label: &str) -> Result<Value> {
    use baselines::SequenceModel;
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let mut bb_model = None;
    let mut gru_model = None;
    if label == "gru47" {
        gru_model = Some(baselines::GruBaseline::new(vb, engine.vocab.size(), 47)?);
    } else {
        let arm = if label == "static_local_write" {
            ArmKind::Arm5StaticBuffer
        } else {
            ArmKind::Arm1BlackboardFixed
        };
        let mut model = DualSystemModel::new(vb, arm, engine.vocab.size(), 32)?;
        model
            .blackboard
            .as_mut()
            .unwrap()
            .cfg
            .normalize_written_cell_only = true;
        bb_model = Some(model);
    }
    baselines::initialize_seeded(&vm, seed)?;
    let n_params: usize = vm.all_vars().iter().map(|v| v.elem_count()).sum();
    let mut optimizer = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 0.003,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    let metric_vm = VarMap::new();
    let metrics = vocab::TokenInterface::new(
        VarBuilder::from_varmap(&metric_vm, DType::F32, &dev),
        engine.vocab.size(),
        1,
    )?;
    let batch = engine.generate_stratified_arithmetic_batch(4, 16, None, Some(3), 991, &dev)?;
    let started = std::time::Instant::now();
    let mut curve = Vec::new();
    let updates = if label == "gru47" { 256 } else { 32 };
    for step in 0..=updates {
        let logits = if let Some(ref gru) = gru_model {
            gru.forward(&batch.inputs)?
        } else {
            let model = bb_model.as_ref().unwrap();
            let mode = if label == "static_local_write" {
                DynamicsMode::StaticBuffer
            } else {
                DynamicsMode::FixedTicks(2)
            };
            model
                .forward(
                    &batch.inputs,
                    SpatialCoordMode::Hilbert,
                    ReadIntervention::Active,
                    WriteIntervention::Active,
                    TemporalIntervention::None,
                    mode,
                    true,
                )?
                .0
        };
        let loss = metrics.masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
        let loss_value = loss.to_scalar::<f32>()?;
        ensure!(loss_value.is_finite(), "non-finite optimization-gate loss");
        if [0, 1, 8, 16, 32, 64, 128, 256].contains(&step) {
            let (acc, exact) =
                falsification::evaluate_query_accuracy(&logits, &batch.targets, &batch.loss_mask)?;
            curve.push(json!({"updates":step,"loss":loss_value,"digit_acc":acc,
                "exact_acc":exact.iter().filter(|&&x|x).count() as f64 / exact.len() as f64}));
        }
        if step < updates {
            let gradients = loss.backward()?;
            falsification::ensure_finite_gradients(&vm, &gradients)?;
            optimizer.step(&gradients)?;
        }
    }
    let training_seconds = started.elapsed().as_secs_f64();
    let train_rows: HashSet<_> = batch.inputs.to_vec2::<u32>()?.into_iter().collect();
    let mut test_rows = HashSet::new();
    let mut correct = 0usize;
    let mut exact_correct = 0usize;
    let mut zero_correct = 0usize;
    let mut query_count = 0usize;
    let zero = engine.vocab.encode("0")[0] as u32;
    for offset in (0..128).step_by(4) {
        let held = engine.generate_stratified_arithmetic_batch(
            4,
            16,
            None,
            Some(3),
            4242 + offset,
            &dev,
        )?;
        let logits = if let Some(ref gru) = gru_model {
            gru.forward(&held.inputs)?
        } else {
            bb_model
                .as_ref()
                .unwrap()
                .forward(
                    &held.inputs,
                    SpatialCoordMode::Hilbert,
                    ReadIntervention::Active,
                    WriteIntervention::Active,
                    TemporalIntervention::None,
                    if label == "static_local_write" {
                        DynamicsMode::StaticBuffer
                    } else {
                        DynamicsMode::FixedTicks(2)
                    },
                    false,
                )?
                .0
        };
        let predictions = logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;
        for ((x, y, m), pred) in rows(&held)?.into_iter().zip(predictions) {
            ensure!(
                !train_rows.contains(&x),
                "held-out example overlaps training"
            );
            ensure!(test_rows.insert(x), "duplicate held-out example");
            let mut exact = true;
            for j in 0..16 {
                if m[j] != 1. {
                    continue;
                }
                correct += usize::from(y[j] == pred[j]);
                zero_correct += usize::from(y[j] == zero);
                query_count += 1;
                exact &= y[j] == pred[j];
            }
            exact_correct += usize::from(exact);
        }
    }
    Ok(
        json!({"model":label,"seed":seed,"parameters":n_params,"updates":updates,
        "held_out":{"samples":128,"unique":test_rows.len(),"train_overlap":0,
            "digit_acc":correct as f64/query_count as f64,"exact_acc":exact_correct as f64/128.,
            "constant_zero_digit_acc":zero_correct as f64/query_count as f64},
        "examples":4,"seq_len":16,"evaluation":"curve on four training examples; separate 128-example held-out panel",
        "all_gradients_finite":true,"wall_seconds":training_seconds,"curve":curve}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "expected fresh output JSON path");
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()?;
    let engine = TaskEngine::new();
    let mut data = Vec::new();
    for seed in [42, 101, 202] {
        data.push(data_audit(&engine, seed)?);
    }
    let mut models = Vec::new();
    for arm in [
        ArmKind::Arm1BlackboardFixed,
        ArmKind::Arm3UselessCompute,
        ArmKind::Arm5StaticBuffer,
    ] {
        eprintln!("Auditing {:?}", arm);
        models.push(model_audit(&engine, arm, 42, false)?);
        if arm != ArmKind::Arm3UselessCompute {
            models.push(model_audit(&engine, arm, 42, true)?);
        }
    }
    let mut optimization = Vec::new();
    for seed in [42, 101, 202] {
        for label in ["static_local_write", "fixed_local_write", "gru47"] {
            eprintln!("Optimization gate: {} seed {}", label, seed);
            optimization.push(optimization_gate(&engine, seed, label)?);
        }
    }
    let report = json!({"schema":"blackboard-audit-v1","device":"cpu","threads":2,
        "initialization":"seeded policy v1; fresh weights only; no checkpoint loading",
        "data":data,"models":models,"optimization_gate":optimization});
    writeln!(output, "{}", serde_json::to_string_pretty(&report)?)?;
    Ok(())
}
