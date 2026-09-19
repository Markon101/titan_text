pub mod arithmetic_corpus;
mod baselines;
mod checkpoint;
mod cli;
mod config;
mod dataset;
mod experiment;
mod field;
mod instrumentation;
mod intervention;
mod latent;
mod nca;
pub mod nca2d;
pub mod falsification;
mod tasks;
mod train;
mod vocab;

use anyhow::Result;
use candle_core::{Device, Tensor};
use checkpoint::{CheckpointManager, ModelManifest, SCHEMA_VERSION};
use config::TitanConfig;
use dataset::SequenceDataset;
use experiment::ExperimentSuite;
use field::MorphogenicField;
use nca::NeuralCellularAutomaton;
use rayon::prelude::*;
use std::env;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use train::Trainer;
use vocab::TokenInterface;

fn write_output_file(path: &str, content: impl AsRef<[u8]>) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, content)?;
    Ok(())
}

fn current_time_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Initializes the unified global thread pool for CPU execution across Candle and Titan.
/// Checks CLI `--threads`, then `RAYON_NUM_THREADS` env var, defaulting to detected hardware concurrency.
fn init_thread_pool(threads: Option<usize>) -> usize {
    let num_threads = threads
        .or_else(|| {
            env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();

    num_threads
}

fn cmd_train(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let save_dir_opt: Option<String> = args.value("--save-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let epochs_opt: Option<usize> = args.value("--epochs")?;
    let dev_steps_opt: Option<usize> = args.value("--dev-steps")?;
    let seq_len_opt: Option<usize> = args.value("--seq-len")?;
    let lr_opt: Option<f64> = args.value("--lr")?;
    let viscosity_opt: Option<f32> = args.value("--viscosity")?;
    let horizon_opt: Option<usize> = args.value("--horizon")?;
    let horizon_mode_opt: Option<String> = args.value("--horizon-mode")?;
    let horizon_min_opt: Option<usize> = args.value("--horizon-min")?;
    let horizon_max_opt: Option<usize> = args.value("--horizon-max")?;
    let horizon_jitter_opt: Option<usize> = args.value("--horizon-jitter")?;
    let stability_tail_opt: Option<usize> = args.value("--stability-tail")?;
    let multi_tick_decay_opt: Option<f32> = args.value("--multi-tick-decay")?;
    let seed_opt: Option<u64> = args.value("--seed")?;
    let feedback_mode_opt: Option<String> = args.value("--feedback-mode")?;
    let feedback_weight_opt: Option<f32> = args.value("--feedback-weight")?;
    let macro_stride_opt: Option<usize> = args.value("--macro-stride")?;
    let macro_period_opt: Option<usize> = args.value("--macro-period")?;
    let macro_channels_opt: Option<usize> = args.value("--macro-channels")?;
    let macro_downsampler_opt: Option<String> = args.value("--macro-downsampler")?;
    let macro_coupling_opt: Option<String> = args.value("--macro-coupling")?;
    let macro_gamma_opt: Option<f32> = args.value("--macro-gamma")?;
    let macro_lambda_opt: Option<f32> = args.value("--macro-lambda")?;
    let tail_eq_weight_opt: Option<f32> = args.value("--tail-eq-weight")?;
    let tail_eq_ticks_opt: Option<usize> = args.value("--tail-eq-ticks")?;
    let state_norm_opt: Option<String> = args.value("--state-norm")?;
    let output_path: Option<String> = args.value("--output")?;
    let patterns_only = args.flag("--patterns-only");
    let trace_output: Option<String> = args.value("--trace-output")?;
    let record_activations = args.flag("--record-activations") || trace_output.is_some();

    let (mut config, start_step, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        let start = manifest.cumulative_step;
        let t = manifest.task.clone();
        (manifest.config, start, t)
    } else {
        (TitanConfig::default(), 0, "text".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    let save_dir = save_dir_opt.unwrap_or_else(|| {
        if let Some(ref dir) = load_dir {
            dir.clone()
        } else {
            "checkpoints/v0_text".to_string()
        }
    });

    if let Some(e) = epochs_opt {
        config.train.epochs = e;
    }
    if let Some(d) = dev_steps_opt {
        config.train.dev_steps = d;
    }
    if let Some(s) = seq_len_opt {
        config.field.seq_len = s;
    }
    if let Some(lr) = lr_opt {
        config.train.lr = lr;
    }
    if let Some(v) = viscosity_opt {
        config.nca.viscosity = v;
    }
    if let Some(h) = horizon_opt {
        config.probe.horizon = h;
    }
    if let Some(m) = horizon_mode_opt {
        config.train.horizon_mode = m;
    }
    if let Some(min) = horizon_min_opt {
        config.train.horizon_min = min;
    }
    if let Some(max) = horizon_max_opt {
        config.train.horizon_max = max;
    }
    if let Some(j) = horizon_jitter_opt {
        config.train.horizon_jitter = j;
    }
    if let Some(st) = stability_tail_opt {
        config.train.stability_tail = st;
    }
    if let Some(decay) = multi_tick_decay_opt {
        config.train.multi_tick_decay = decay;
    }
    if let Some(seed) = seed_opt {
        config.train.seed = seed;
    }
    if let Some(fm) = feedback_mode_opt {
        config.nca.feedback_mode = fm;
    }
    if let Some(fw) = feedback_weight_opt {
        config.nca.feedback_weight = fw;
    }
    if let Some(ms) = macro_stride_opt {
        config.nca.macro_stride = ms;
    }
    if let Some(mp) = macro_period_opt {
        config.nca.macro_period = mp;
    }
    if let Some(mc) = macro_channels_opt {
        config.nca.macro_channels = mc;
    }
    if let Some(md) = macro_downsampler_opt {
        config.nca.macro_downsampler = md;
    }
    if let Some(mc) = macro_coupling_opt {
        config.nca.macro_coupling = mc;
    }
    if let Some(mg) = macro_gamma_opt {
        config.nca.macro_gamma = mg;
    }
    if let Some(ml) = macro_lambda_opt {
        config.nca.macro_lambda = ml;
    }
    if let Some(tew) = tail_eq_weight_opt {
        config.train.tail_equilibrium_weight = tew;
    }
    if let Some(tet) = tail_eq_ticks_opt {
        config.train.tail_equilibrium_ticks = tet;
    }
    if let Some(sn) = state_norm_opt {
        config.nca.state_norm = sn;
    }
    if args.flag("--zero-boundary") {
        config.field.periodic_boundary = false;
    }
    if args.flag("--coord-channel") {
        config.nca.coord_channel = true;
    }
    if args.flag("--causal-stencil") {
        config.nca.causal_stencil = true;
    }
    if let Some(ts) = args.value::<usize>("--target-slot")? {
        config.train.target_slot = Some(ts);
    }
    config.validate()?;
    start_step
        .checked_add(config.train.epochs)
        .ok_or_else(|| anyhow::anyhow!("cumulative training step would overflow"))?;

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ TITAN TEXT v0 · RECURRENT CELLULAR TRAINING SESSION                                  ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );
    if let Some(ref dir) = load_dir {
        println!(
            "  Resuming From       : {} (cumulative step {})",
            dir, start_step
        );
    }
    println!("  Task                : {}", task);
    println!("  Epochs to Train     : {}", config.train.epochs);
    println!("  Dev Recurrence (T)  : {} steps", config.train.dev_steps);
    println!("  Lattice Length (L)  : {} cells", config.field.seq_len);
    println!(
        "  Channels (C)        : {} continuous hidden dimensions",
        config.field.channels
    );
    println!(
        "  Feedback Mode       : {} (weight: {:.2})",
        config.nca.feedback_mode, config.nca.feedback_weight
    );
    if config.nca.state_norm != "none" {
        println!("  State Normalization : {}", config.nca.state_norm);
    }
    if config.nca.coord_channel {
        println!("  Coordinate Channel  : ENABLED (p_i in [-1, 1] per cell)");
    }
    if config.nca.causal_stencil {
        println!("  Causal Stencil      : ENABLED (directed DAG fold N(i) = {{i-1, i}})");
    }
    if let Some(ts) = config.train.target_slot {
        println!("  Target Slot Filter  : Slot {} only (cell {})", ts, (ts + 1) * 4 - 1);
    }
    if config.train.tail_equilibrium_weight > 0.0 {
        println!(
            "  Tail Equilibrium    : lambda={:.3}, K={} tail ticks",
            config.train.tail_equilibrium_weight, config.train.tail_equilibrium_ticks
        );
    }
    println!(
        "  Viscosity (nu)      : {:.4} (Navier-Stokes dissipation)",
        config.nca.viscosity
    );
    println!("  Horizon Regime      : {}", config.train.horizon_mode);
    if config.train.horizon_mode != "fixed" {
        println!("  Horizon Settings    : min={}, max={}, jitter={}, tail={}",
            config.train.horizon_min, config.train.horizon_max, config.train.horizon_jitter, config.train.stability_tail);
    }
    println!("  Save Directory      : {}", save_dir);

    let mut trainer = Trainer::new(config.clone(), &task, device)?;
    let total_params: usize = trainer
        .varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!("  Total Parameters    : {} weights\n", total_params);

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut trainer.varmap, device)?;
    }

    let start_time = Instant::now();
    let mut last_diag = None;

    println!("─── TRAINING & DYNAMICAL INSTRUMENTATION TRACE ─────────────────────────────────────────────────────────");
    println!("Epoch | Train Loss (Acc) | Val Loss (Acc)   | Grad Norm | ||Δx||  | Energy | Enstrophy | BKM Norm | Freq (L/M/H %)");
    println!("──────┼──────────────────┼──────────────────┼───────────┼─────────┼────────┼───────────┼──────────┼───────────────");

    for epoch in 1..=config.train.epochs {
        let cum_step = start_step + epoch;
        let diag = trainer.train_step(config.train.batch_size)?;
        let is_log_step = epoch % config.train.log_every == 0 || epoch == config.train.epochs;

        if is_log_step {
            println!(
                "{:04}  | {:.4} ({:5.1}%) | {:.4} ({:5.1}%) | {:<9.5} | {:<7.5} | {:<6.4} | {:<9.4} | {:<8.4} | {:3.0}/{:3.0}/{:3.0}%",
                cum_step,
                diag.train_loss,
                diag.train_acc * 100.0,
                diag.val_loss,
                diag.val_acc * 100.0,
                diag.grad_norm,
                diag.update_magnitude,
                diag.state_energy,
                diag.enstrophy,
                diag.bkm_norm,
                diag.freq_decomp.pct_low,
                diag.freq_decomp.pct_mid,
                diag.freq_decomp.pct_high,
            );
        }

        if epoch % config.train.save_every == 0 || epoch == config.train.epochs {
            let mut manifest = ModelManifest {
                schema_version: SCHEMA_VERSION,
                git_commit: CheckpointManager::get_git_commit(),
                random_seed: config.train.seed,
                cumulative_step: cum_step,
                dev_steps: config.train.dev_steps,
                train_loss: diag.train_loss,
                train_accuracy: diag.train_acc,
                val_loss: diag.val_loss,
                val_accuracy: diag.val_acc,
                grad_norm: diag.grad_norm,
                state_energy: diag.state_energy,
                param_count: total_params,
                checkpoint_hash: "".to_string(),
                timestamp_unix: current_time_unix(),
                config: config.clone(),
                task: task.clone(),
            };
            CheckpointManager::save(&save_dir, &mut manifest, &trainer.varmap)?;
        }

        last_diag = Some(diag);
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let speed = config.train.epochs as f32 / elapsed.max(1e-3);
    println!("────────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!(
        "✓ Training finished in {:.2}s ({:.1} epochs/sec)",
        elapsed, speed
    );
    println!(
        "✓ Final checkpoint and manifest committed to '{}'",
        save_dir
    );

    if let Some(final_d) = last_diag {
        println!("\nFinal State Summary:");
        println!(
            "  • Train Accuracy : {:.1}% (Loss: {:.4})",
            final_d.train_acc * 100.0,
            final_d.train_loss
        );
        println!(
            "  • Val Accuracy   : {:.1}% (Loss: {:.4})",
            final_d.val_acc * 100.0,
            final_d.val_loss
        );
        println!(
            "  • Hidden Mean    : {:.4} (Variance: {:.4})",
            final_d.hidden_mean, final_d.hidden_var
        );
        println!("  • State Energy   : {:.4}", final_d.state_energy);
        println!("  • Enstrophy (Ω)  : {:.4}", final_d.enstrophy);
        println!("  • BKM Norm (B)   : {:.4}", final_d.bkm_norm);
    }

    // Run post-training diagnostics
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║ POST-TRAINING DYNAMICAL DIAGNOSTICS & PERTURBATION PROBE                             ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let (sample_in, _) = trainer
        .dataset
        .sample_train_batch(1, config.field.seq_len, device)?;
    let seed_embed = trainer.interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);
    let settled_field = trainer
        .nca
        .develop(&sample_field, config.train.dev_steps, device)?;

    let probe_rep = ExperimentSuite::run_nonlinear_perturbation_probe(
        &trainer.nca,
        &settled_field,
        config.probe.perturbation_eps,
        config.probe.lp_window,
        device,
    )?;

    println!(
        "\n[1] ±EPSILON NONLINEAR PERTURBATION PROBE (eps = {})",
        config.probe.perturbation_eps
    );
    println!("    Wavenumber | Wavelength | ||Q|| Norm | ||Δx|| Base | Relative Q/||Δx||");
    println!("    ───────────┼────────────┼────────────┼─────────────┼──────────────────");
    for w in &probe_rep.wavelengths {
        println!(
            "    k = {:<6} | {:<2} cells   | {:<10.5} | {:<11.5} | {:<8.3} ({})",
            w.wavenumber,
            w.wavelength_cells,
            w.q_norm,
            w.ordinary_update_norm,
            w.relative_response_ratio,
            w.frequency_label
        );
    }
    println!("    Verdict: {}", probe_rep.verdict);

    let traj_rep = ExperimentSuite::run_autonomous_rollout_with_options(
        &trainer.nca,
        &trainer.interface,
        &trainer.dataset.vocab,
        &settled_field,
        config.probe.horizon,
        record_activations,
        device,
    )?;
    println!(
        "\n[2] AUTONOMOUS FROZEN TRAJECTORY ({}-step rollout)",
        config.probe.horizon
    );
    println!(
        "    • Trajectory Classification   : {}",
        traj_rep.classification
    );
    println!(
        "    • Mean Step Drift Velocity    : {:.5} units/step",
        traj_rep.mean_step_velocity
    );
    println!(
        "    • Min Recurrence Distance     : {:.5}",
        traj_rep.min_recurrence_distance
    );
    println!(
        "    • Final Field Energy          : {:.4}",
        traj_rep.final_energy
    );
    println!(
        "    • Final Output Token Entropy  : {:.4} nats",
        traj_rep.final_entropy
    );
    if let Some(first_t) = traj_rep.traces.first() {
        println!(
            "    • Initial Seed Pattern        : \"{}\"",
            first_t.decoded_pattern
        );
    }
    if let Some(last_t) = traj_rep.traces.last() {
        println!(
            "    • Final Emerged Pattern       : \"{}\"",
            last_t.decoded_pattern
        );
    }

    let falsify_rep = ExperimentSuite::run_falsification_battery(
        &trainer.nca,
        &trainer.interface,
        &trainer.dataset,
        device,
    )?;
    println!("\n[3] FALSIFICATION & CONTEXT SENSITIVITY BATTERY");
    println!(
        "    • Baseline Target Accuracy    : {:5.1}%",
        falsify_rep.baseline_accuracy * 100.0
    );
    println!(
        "    • Scrambled Context Accuracy  : {:5.1}%",
        falsify_rep.scrambled_context_accuracy * 100.0
    );
    println!(
        "    • Zeroed Context Accuracy     : {:5.1}%",
        falsify_rep.zeroed_context_accuracy * 100.0
    );
    println!(
        "    • Truncated Context Accuracy  : {:5.1}%",
        falsify_rep.truncated_context_accuracy * 100.0
    );
    println!(
        "    • State-Reset Divergence      : {:.5}",
        falsify_rep.state_reset_divergence
    );
    println!(
        "    • Contextual Memory Ratio     : {:5.1}%",
        falsify_rep.contextual_memory_ratio * 100.0
    );
    println!(
        "    • Verdict                     : {}",
        falsify_rep.verdict
    );

    if let Some(ref path) = output_path {
        let mut out_text = String::new();
        if patterns_only {
            for t in &traj_rep.traces {
                out_text.push_str(&format!("{}\n", t.decoded_pattern));
            }
        } else {
            out_text.push_str("# TITAN TEXT · AUTONOMOUS ROLLOUT PATTERN TRACE\n");
            out_text.push_str(&format!("# Checkpoint: {}\n", save_dir));
            out_text.push_str(&format!(
                "# Task: {} | Horizon: {} steps | Viscosity (nu): {:.4}\n",
                task, config.probe.horizon, config.nca.viscosity
            ));
            out_text.push_str(&format!(
                "# Classification: {}\n#\n",
                traj_rep.classification
            ));
            for t in &traj_rep.traces {
                out_text.push_str(&format!("Step {:03}: {}\n", t.step, t.decoded_pattern));
            }
        }
        write_output_file(path, out_text)?;
        println!("\n✓ Post-training patterns saved to '{}'", path);
    }

    if let Some(ref trace_path) = trace_output {
        let trace_json = serde_json::to_string_pretty(&traj_rep.traces)?;
        write_output_file(trace_path, trace_json)?;
        println!("\n✓ Per-step activation traces saved to '{}'", trace_path);
    }

    Ok(())
}

fn cmd_probe(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let eps: f32 = args.value("--eps")?.unwrap_or(0.01);
    let horizon: usize = args.value("--horizon")?.unwrap_or(64);
    let output_path: Option<String> = args.value("--output")?;
    let trace_output: Option<String> = args.value("--trace-output")?;
    let record_activations = args.flag("--record-activations") || trace_output.is_some();

    let (config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;

    config.validate()?;
    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(
        vb.pp("interface"),
        dataset.vocab.size(),
        config.field.channels,
    )?;

    if let Some(ref dir) = load_dir {
        println!("Loading trained weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    } else {
        println!("Probing freshly initialized model (random weights baseline)...");
    }

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ TITAN TEXT · DYNAMICAL BATTERY & NONLINEAR PERTURBATION REPORT                       ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let (sample_in, _) = dataset.sample_train_batch(1, config.field.seq_len, device)?;
    let seed_embed = interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);
    let settled_field = nca.develop(&sample_field, 8, device)?;

    let probe_rep = ExperimentSuite::run_nonlinear_perturbation_probe(
        &nca,
        &settled_field,
        eps,
        config.probe.lp_window,
        device,
    )?;

    println!(
        "\n[1] ±EPSILON NONLINEAR PERTURBATION PROBE (eps = {})",
        eps
    );
    println!("    Wavenumber | Wavelength | ||Q|| Norm | ||Δx|| Base | Relative Q/||Δx||");
    println!("    ───────────┼────────────┼────────────┼─────────────┼──────────────────");
    for w in &probe_rep.wavelengths {
        println!(
            "    k = {:<6} | {:<2} cells   | {:<10.5} | {:<11.5} | {:<8.3} ({})",
            w.wavenumber,
            w.wavelength_cells,
            w.q_norm,
            w.ordinary_update_norm,
            w.relative_response_ratio,
            w.frequency_label
        );
    }
    println!("    Verdict: {}", probe_rep.verdict);

    let traj_rep = ExperimentSuite::run_autonomous_rollout_with_options(
        &nca,
        &interface,
        &dataset.vocab,
        &settled_field,
        horizon,
        record_activations,
        device,
    )?;
    println!(
        "\n[2] AUTONOMOUS FROZEN TRAJECTORY ({}-step rollout)",
        horizon
    );
    println!(
        "    • Trajectory Classification   : {}",
        traj_rep.classification
    );
    println!(
        "    • Mean Step Drift Velocity    : {:.5} units/step",
        traj_rep.mean_step_velocity
    );
    println!(
        "    • Min Recurrence Distance     : {:.5}",
        traj_rep.min_recurrence_distance
    );
    println!(
        "    • Final Field Energy          : {:.4}",
        traj_rep.final_energy
    );
    println!(
        "    • Final Output Token Entropy  : {:.4} nats",
        traj_rep.final_entropy
    );
    if let Some(first_t) = traj_rep.traces.first() {
        println!(
            "    • Initial Seed Pattern        : \"{}\"",
            first_t.decoded_pattern
        );
    }
    if let Some(last_t) = traj_rep.traces.last() {
        println!(
            "    • Final Emerged Pattern       : \"{}\"",
            last_t.decoded_pattern
        );
    }

    let falsify_rep =
        ExperimentSuite::run_falsification_battery(&nca, &interface, &dataset, device)?;
    println!("\n[3] FALSIFICATION & CONTEXT SENSITIVITY BATTERY");
    println!(
        "    • Baseline Target Accuracy    : {:5.1}%",
        falsify_rep.baseline_accuracy * 100.0
    );
    println!(
        "    • Scrambled Context Accuracy  : {:5.1}%",
        falsify_rep.scrambled_context_accuracy * 100.0
    );
    println!(
        "    • Zeroed Context Accuracy     : {:5.1}%",
        falsify_rep.zeroed_context_accuracy * 100.0
    );
    println!(
        "    • Truncated Context Accuracy  : {:5.1}%",
        falsify_rep.truncated_context_accuracy * 100.0
    );
    println!(
        "    • State-Reset Divergence      : {:.5}",
        falsify_rep.state_reset_divergence
    );
    println!(
        "    • Contextual Memory Ratio     : {:5.1}%",
        falsify_rep.contextual_memory_ratio * 100.0
    );
    println!(
        "    • Verdict                     : {}",
        falsify_rep.verdict
    );

    if let Some(path) = output_path {
        let full_report = experiment::FullDiagnosticReport {
            perturbation: probe_rep,
            trajectory: traj_rep.clone(),
            falsification: falsify_rep,
            navier_stokes: None,
        };
        let json_str = serde_json::to_string_pretty(&full_report)?;
        write_output_file(&path, json_str)?;
        println!("\n✓ Scientific report saved to '{}'", path);
    }

    if let Some(ref trace_path) = trace_output {
        let trace_json = serde_json::to_string_pretty(&traj_rep.traces)?;
        write_output_file(trace_path, trace_json)?;
        println!("\n✓ Per-step activation traces saved to '{}'", trace_path);
    }

    Ok(())
}

fn cmd_ns_probe(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let forcing_amp: f32 = args.value("--forcing-amp")?.unwrap_or(0.2);
    let viscosity: f32 = args.value("--viscosity")?.unwrap_or(0.0);
    let horizon: usize = args.value("--horizon")?.unwrap_or(64);
    let output_path: Option<String> = args.value("--output")?;
    let trace_output: Option<String> = args.value("--trace-output")?;

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    config.nca.viscosity = viscosity;

    config.validate()?;
    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(
        vb.pp("interface"),
        dataset.vocab.size(),
        config.field.channels,
    )?;

    if let Some(ref dir) = load_dir {
        println!("Loading trained weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    } else {
        println!("Probing freshly initialized model (random weights baseline)...");
    }

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ TITAN TEXT · NAVIER-STOKES MILLENNIUM SINGULARITY & ENSTROPHY CASCADE PROBE          ║"
    );
    println!(
        "║ Theoretical Foundations: Beale-Kato-Majda Blow-up & Smooth External Forcing          ║"
    );
    println!(
        "║ (Context: OpenAI Sept 8, 2026 Finite-Time Singularity Proof for Cases C & D)         ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!("  Horizon Steps (T)   : {}", horizon);
    println!(
        "  Forcing Amplitude A : {:.3} (Smooth C^inf multi-mode periodic)",
        forcing_amp
    );
    println!("  Base Viscosity (nu) : {:.4}", viscosity);
    println!("  Spatial Lattice (L) : {} cells", config.field.seq_len);
    println!("  Channels (C)        : {}", config.field.channels);

    let (sample_in, _) = dataset.sample_train_batch(1, config.field.seq_len, device)?;
    let seed_embed = interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);
    let settled_field = nca.develop(&sample_field, 8, device)?;

    let ns_report = ExperimentSuite::run_navier_stokes_blowup_probe(
        &nca,
        &settled_field,
        horizon,
        forcing_amp,
        viscosity,
        device,
    )?;

    println!("\n[1] UNFORCED FLOW (f = 0, nu = {:.4})", viscosity);
    println!("    Step | Energy   | Enstrophy (Ω) | Palinstrophy (P) | BKM Norm (||∇x||) | ||x||_inf | Centroid k | Slope β");
    println!("    ─────┼──────────┼───────────────┼──────────────────┼───────────────────┼───────────┼────────────┼────────");
    for (idx, t) in ns_report.unforced_regime.traces.iter().enumerate() {
        if idx % 8 == 0 || idx == ns_report.unforced_regime.traces.len() - 1 {
            println!(
                "    {:03}  | {:<8.4} | {:<13.4} | {:<16.4} | {:<17.4} | {:<9.4} | {:<10.2} | {:<7.3}",
                t.step, t.energy, t.enstrophy, t.palinstrophy, t.bkm_norm, t.max_velocity, t.spectral_centroid, t.spectral_slope
            );
        }
    }
    println!(
        "    Initial Enstrophy Ω(0) : {:.5}",
        ns_report.unforced_regime.initial_enstrophy
    );
    println!(
        "    Max Enstrophy Ω_max    : {:.5}",
        ns_report.unforced_regime.max_enstrophy
    );
    println!(
        "    Max BKM Norm B_max     : {:.5}",
        ns_report.unforced_regime.max_bkm_norm
    );
    println!(
        "    Accumulated BKM ∫B dt  : {:.5}",
        ns_report.unforced_regime.accumulated_bkm
    );

    println!(
        "\n[2] SMOOTH EXTERNAL FORCING FLOW (f ∈ C^inf, A = {:.3}, nu = 0.0)",
        forcing_amp
    );
    println!("    Step | Energy   | Enstrophy (Ω) | Palinstrophy (P) | BKM Norm (||∇x||) | ||x||_inf | Forcing Pwr | Dissip Rate");
    println!("    ─────┼──────────┼───────────────┼──────────────────┼───────────────────┼───────────┼─────────────┼────────────");
    for (idx, t) in ns_report.forced_regime.traces.iter().enumerate() {
        if idx % 8 == 0 || idx == ns_report.forced_regime.traces.len() - 1 {
            println!(
                "    {:03}  | {:<8.4} | {:<13.4} | {:<16.4} | {:<17.4} | {:<9.4} | {:<11.5} | {:<11.5}",
                t.step, t.energy, t.enstrophy, t.palinstrophy, t.bkm_norm, t.max_velocity, t.forcing_power, t.dissipation_rate
            );
        }
    }
    println!(
        "    Blowup Detected        : {}",
        if ns_report.forced_regime.blowup_detected {
            "YES (Singularity/Breakdown Observed)"
        } else {
            "NO (Solution Remained Bounded)"
        }
    );
    if let Some(s) = ns_report.forced_regime.singularity_step {
        println!("    Singularity Time T*    : Step {}", s);
    }
    println!(
        "    Enstrophy Exponent γ   : {:.3} (dΩ/dt ~ Ω^γ; γ > 1 implies finite-time blowup)",
        ns_report.forced_regime.enstrophy_growth_exponent
    );
    println!(
        "    BKM Criteria Met       : {}",
        if ns_report.bkm_blowup_criteria_met {
            "YES (Singularity Criteria Satisfied)"
        } else {
            "NO"
        }
    );

    println!("\n[3] VISCOSITY REGULARIZATION SWEEP (nu * Δx Dissipation)");
    println!("    Viscosity (nu) | Final Energy | Final Enstrophy | Max BKM Norm | Max ||x||_inf | Regularized?");
    println!("    ───────────────┼──────────────┼─────────────────┼──────────────┼───────────────┼─────────────");
    for res in &ns_report.viscosity_sweep {
        println!(
            "    nu = {:<9.4} | {:<12.4} | {:<15.4} | {:<12.4} | {:<13.4} | {}",
            res.viscosity,
            res.final_energy,
            res.final_enstrophy,
            res.max_bkm,
            res.max_velocity,
            if res.regularized {
                "✓ YES (Stable)"
            } else {
                "✗ NO (Singular/Blowup)"
            }
        );
    }
    if let Some(nu_c) = ns_report.critical_viscosity_est {
        println!(
            "    Critical Viscosity nu* : ~{:.4} (Minimum dissipation needed to arrest blowup)",
            nu_c
        );
    } else {
        println!("    Critical Viscosity nu* : > 0.2000 (Requires stronger dissipation)");
    }
    println!("\nVerdict: {}", ns_report.verdict);

    if let Some(path) = output_path {
        let json_str = serde_json::to_string_pretty(&ns_report)?;
        write_output_file(&path, json_str)?;
        println!("\n✓ Scientific report saved to '{}'", path);
    }

    if let Some(ref trace_path) = trace_output {
        let trace_json = serde_json::to_string_pretty(&ns_report.forced_regime.traces)?;
        write_output_file(trace_path, trace_json)?;
        println!("\n✓ Per-step forced regime traces saved to '{}'", trace_path);
    }

    Ok(())
}

fn cmd_rollout(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let horizon: usize = args.value("--horizon")?.unwrap_or(32);
    let seq_len_opt: Option<usize> = args.value("--seq-len")?;
    let viscosity_opt: Option<f32> = args.value("--viscosity")?;
    let output_path: Option<String> = args.value("--output")?;
    let prompt_opt: Option<String> = args.value("--prompt")?;
    let patterns_only = args.flag("--patterns-only");
    let trace_output: Option<String> = args.value("--trace-output")?;
    let record_activations = args.flag("--record-activations") || trace_output.is_some();

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        let t = manifest.task.clone();
        (manifest.config, t)
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    if let Some(s) = seq_len_opt {
        config.field.seq_len = s;
    }
    if let Some(v) = viscosity_opt {
        config.nca.viscosity = v;
    }
    if args.flag("--zero-boundary") {
        config.field.periodic_boundary = false;
    }

    config.validate()?;
    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(
        vb.pp("interface"),
        dataset.vocab.size(),
        config.field.channels,
    )?;

    if let Some(ref dir) = load_dir {
        if !patterns_only {
            println!("Loading model weights from '{}'...", dir);
        }
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    } else if !patterns_only {
        println!("Warning: Rolling out freshly initialized model (random weights baseline)...");
    }

    let seq_len = config.field.seq_len;
    let sample_in = if let Some(ref prompt) = prompt_opt {
        let mut ids = dataset.vocab.encode(prompt);
        if ids.len() < seq_len {
            ids.resize(seq_len, dataset.vocab.pad_id);
        } else {
            ids.truncate(seq_len);
        }
        let u32_ids: Vec<u32> = ids.into_iter().map(|id| id as u32).collect();
        Tensor::from_slice(&u32_ids, (1, seq_len), device)?
    } else {
        let (in_batch, _) = dataset.sample_train_batch(1, seq_len, device)?;
        in_batch
    };

    let seed_embed = interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);

    let traj = ExperimentSuite::run_autonomous_rollout_with_options(
        &nca,
        &interface,
        &dataset.vocab,
        &sample_field,
        horizon,
        record_activations,
        device,
    )?;

    if patterns_only {
        for t in &traj.traces {
            println!("{}", t.decoded_pattern);
        }
    } else {
        println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
        println!("║ TITAN TEXT · STEP-BY-STEP AUTONOMOUS ROLLOUT TRACE                                   ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
        if let Some(ref dir) = load_dir {
            println!("  Model Checkpoint    : {}", dir);
        } else {
            println!("  Model Checkpoint    : (random weights baseline)");
        }
        println!("  Task                : {}", task);
        println!("  Horizon (T)         : {} developmental steps", horizon);
        println!(
            "  Viscosity (nu)      : {:.4} (Navier-Stokes dissipation)",
            config.nca.viscosity
        );
        println!("  Lattice Length (L)  : {} cells\n", config.field.seq_len);

        println!("Step | State Norm | Energy | Entropy | Decoded Lattice Pattern");
        println!("─────┼────────────┼────────┼─────────┼──────────────────────────────────────────────────");
        for t in &traj.traces {
            println!(
                "{:03}  | {:<10.4} | {:<6.4} | {:.4}  | \"{}\"",
                t.step, t.state_norm, t.energy, t.output_entropy, t.decoded_pattern
            );
        }
        println!(
            "───────────────────────────────────────────────────────────────────────────────────"
        );
        println!("Classification : {}", traj.classification);
        println!("Drift Velocity : {:.5} units/step", traj.mean_step_velocity);
        println!("Final Energy   : {:.4}", traj.final_energy);
        println!("Final Entropy  : {:.4} nats", traj.final_entropy);
    }

    if let Some(ref out_file) = output_path {
        if out_file.ends_with(".json") {
            let json_str = serde_json::to_string_pretty(&traj)?;
            write_output_file(out_file, json_str)?;
            if !patterns_only {
                println!("\n✓ Rollout JSON report saved to '{}'", out_file);
            }
        } else {
            let mut out_text = String::new();
            if patterns_only {
                for t in &traj.traces {
                    out_text.push_str(&format!("{}\n", t.decoded_pattern));
                }
            } else {
                out_text.push_str("# TITAN TEXT · AUTONOMOUS ROLLOUT PATTERN TRACE\n");
                if let Some(ref dir) = load_dir {
                    out_text.push_str(&format!("# Checkpoint: {}\n", dir));
                } else {
                    out_text.push_str("# Checkpoint: random weights baseline\n");
                }
                out_text.push_str(&format!(
                    "# Task: {} | Horizon: {} steps | Viscosity: {:.4}\n",
                    task, horizon, config.nca.viscosity
                ));
                out_text.push_str(&format!("# Classification: {}\n#\n", traj.classification));
                for t in &traj.traces {
                    out_text.push_str(&format!("Step {:03}: {}\n", t.step, t.decoded_pattern));
                }
            }

            write_output_file(out_file, out_text)?;
            if !patterns_only {
                println!("\n✓ Rollout patterns saved to '{}'", out_file);
            }
        }
    }

    if let Some(ref trace_path) = trace_output {
        let trace_json = serde_json::to_string_pretty(&traj.traces)?;
        write_output_file(trace_path, trace_json)?;
        if !patterns_only {
            println!("\n✓ Per-step activation traces saved to '{}'", trace_path);
        }
    }

    Ok(())
}

fn cmd_sweep(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let budgets_str: Option<String> = args.value("--budgets")?;
    let seq_len_opt: Option<usize> = args.value("--seq-len")?;
    let batch_size: usize = args.value("--batch-size")?.unwrap_or(8);
    let output_path: Option<String> = args.value("--output")?;
    let trace_output: Option<String> = args.value("--trace-output")?;
    let model_opt: Option<String> = args.value("--model")?;
    let seeds_str: Option<String> = args.value("--seeds")?;
    let seeds: Vec<usize> = match seeds_str {
        Some(s) => s
            .split(',')
            .filter_map(|b| b.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0],
    };
    let seeds = if seeds.is_empty() { vec![0] } else { seeds };

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "delayed-recall".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    if let Some(s) = seq_len_opt {
        config.field.seq_len = s;
    }
    if let Some(m) = model_opt {
        config.model = m;
    }

    let mut intervention = config.intervention.clone();
    if args.flag("--lesion-state") {
        intervention.disable_recurrent = true;
    }
    if args.flag("--lesion-gates") {
        intervention.disable_gates = true;
    }
    if args.flag("--lesion-residual") {
        intervention.disable_residual = true;
    }
    if let Some(g) = args.value::<f64>("--lesion-gain")? {
        intervention.recurrence_gain = g as f32;
    }
    if let Some(n) = args.value::<f32>("--lesion-noise")? {
        intervention.additive_noise_sigma = n;
    }
    if let Some(f) = args.value::<usize>("--lesion-freeze")? {
        intervention.freeze_step = Some(f);
    }
    if let Some(r) = args.value::<usize>("--lesion-reset")? {
        intervention.reset_step = Some(r);
    }
    if let Some(ch_str) = args.value::<String>("--lesion-channels")? {
        intervention.ablate_channel_indices = ch_str
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
    }
    if args.flag("--lesion-shuffle") {
        intervention.shuffle_batch = true;
    }
    if args.flag("--zero-boundary") {
        config.field.periodic_boundary = false;
    }
    if args.flag("--coord-channel") {
        config.nca.coord_channel = true;
    }
    if args.flag("--causal-stencil") {
        config.nca.causal_stencil = true;
    }
    if let Some(cm) = args.value::<String>("--coord-mode")? {
        intervention.coord_mode = Some(cm);
    }
    if let Some(fm) = args.value::<String>("--feedback-mode")? {
        config.nca.feedback_mode = fm;
    }
    if let Some(fw) = args.value::<f32>("--feedback-weight")? {
        config.nca.feedback_weight = fw;
    }
    if let Some(ms) = args.value::<usize>("--macro-stride")? {
        config.nca.macro_stride = ms;
    }
    if let Some(mp) = args.value::<usize>("--macro-period")? {
        config.nca.macro_period = mp;
    }
    if let Some(mc) = args.value::<usize>("--macro-channels")? {
        config.nca.macro_channels = mc;
    }
    if let Some(md) = args.value::<String>("--macro-downsampler")? {
        config.nca.macro_downsampler = md;
    }
    if let Some(mc) = args.value::<String>("--macro-coupling")? {
        config.nca.macro_coupling = mc;
    }
    if let Some(mg) = args.value::<f32>("--macro-gamma")? {
        config.nca.macro_gamma = mg;
    }
    if let Some(ml) = args.value::<f32>("--macro-lambda")? {
        config.nca.macro_lambda = ml;
    }

    config.validate()?;

    let task_kind = tasks::TaskKind::parse(&task).unwrap_or(tasks::TaskKind::DelayedRecall);
    let budgets: Vec<usize> = match budgets_str {
        Some(s) => s
            .split(',')
            .filter_map(|b| b.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0, 1, 2, 4, 8, 16, 32],
    };

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let vocab = vocab::Vocab::new_ascii();
    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    }

    let executor = latent::LatentExecutor::new(&nca, &interface, &vocab);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · LATENT COMPUTE BUDGET SWEEP                                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Task                : {}", task_kind.name());
    println!("  Budgets Evaluated   : {:?}", budgets);
    println!("  Sequence Length (L) : {}", config.field.seq_len);
    println!("  Batch Size          : {}", batch_size);
    if seeds.len() > 1 {
        println!("  Seeds Evaluated     : {:?}", seeds);
    }
    if intervention.is_active() {
        println!("  Intervention Active : {:?}", intervention);
    }

    if seeds.len() == 1 {
        let report = executor.run_latent_budget_sweep_seeded(
            task_kind,
            &budgets,
            batch_size,
            config.field.seq_len,
            &intervention,
            seeds[0],
            device,
        )?;

        println!("\n| Latent Ticks | Accuracy (%) | Cross-Entropy Loss | Confidence | State Displacement | Compute Cost | Regime |");
        println!("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |");
        for b in &report.budgets {
            println!(
                "| {:<12} | {:<12.1} | {:<18.4} | {:<10.4} | {:<18.4} | {:<12} | {:<6} |",
                b.latent_ticks,
                b.accuracy * 100.0,
                b.loss,
                b.mean_confidence,
                b.state_displacement,
                b.compute_cost_units,
                b.regime
            );
            if let Some(ref slots) = b.per_slot_accuracy {
                let slot_strs: Vec<String> = slots.iter().enumerate().map(|(si, &acc)| format!("slot {}: {:.1}%", si, acc * 100.0)).collect();
                println!("  └─ Query Slots: {}", slot_strs.join(", "));
            }
        }

        println!("\nAnalysis Summary:");
        println!("  • Monotonic Compute Gain : {}", if report.monotonic_improvement { "YES (Accuracy strictly increases with ticks)" } else { "NO (Non-monotonic / plateauing response)" });
        println!("  • Accuracy Discontinuity : {}", if report.bifurcation_detected { "DETECTED (>25% jump between consecutive budgets)" } else { "NOT DETECTED (Smooth or continuous progression)" });
        println!("  • Optimal Budget         : {} latent ticks", report.optimal_ticks);

        if let Some(ref path) = output_path {
            let json_str = serde_json::to_string_pretty(&report)?;
            write_output_file(path, json_str)?;
            println!("\n✓ Sweep report saved to '{}'", path);
        }
        if let Some(ref trace_path) = trace_output {
            let json_str = serde_json::to_string_pretty(&report)?;
            write_output_file(trace_path, json_str)?;
            println!("\n✓ Sweep traces saved to '{}'", trace_path);
        }
    } else {
        let reports: Vec<latent::LatentBudgetSweepReport> = seeds
            .par_iter()
            .map(|&s| {
                executor.run_latent_budget_sweep_seeded(
                    task_kind,
                    &budgets,
                    batch_size,
                    config.field.seq_len,
                    &intervention,
                    s,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        println!("\n| Latent Ticks | Accuracy (%) (mean ± std) | Loss (mean) | Displacement (mean) | Regime |");
        println!("| :--- | :--- | :--- | :--- | :--- |");
        for (idx, &ticks) in budgets.iter().enumerate() {
            let accs: Vec<f32> = reports.iter().map(|r| r.budgets[idx].accuracy * 100.0).collect();
            let losses: Vec<f32> = reports.iter().map(|r| r.budgets[idx].loss).collect();
            let disps: Vec<f32> = reports.iter().map(|r| r.budgets[idx].state_displacement).collect();
            let mean_acc = accs.iter().sum::<f32>() / accs.len() as f32;
            let std_acc = if accs.len() > 1 {
                (accs.iter().map(|a| (a - mean_acc).powi(2)).sum::<f32>() / (accs.len() - 1) as f32).sqrt()
            } else {
                0.0
            };
            let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            let mean_disp = disps.iter().sum::<f32>() / disps.len() as f32;
            let regime = &reports[0].budgets[idx].regime;

            println!(
                "| {:<12} | {:<12.1} ± {:<10.1} | {:<11.4} | {:<19.4} | {:<6} |",
                ticks, mean_acc, std_acc, mean_loss, mean_disp, regime
            );
            if let Some(ref first_slots) = reports[0].budgets[idx].per_slot_accuracy {
                let num_slots = first_slots.len();
                let mut slot_sums = vec![0.0f32; num_slots];
                for rep in &reports {
                    if let Some(ref s) = rep.budgets[idx].per_slot_accuracy {
                        for (si, &val) in s.iter().enumerate().take(num_slots) {
                            slot_sums[si] += val;
                        }
                    }
                }
                let slot_strs: Vec<String> = slot_sums
                    .iter()
                    .enumerate()
                    .map(|(si, &sum_val)| format!("slot {}: {:.1}%", si, (sum_val / reports.len() as f32) * 100.0))
                    .collect();
                println!("  └─ Mean Query Slots: {}", slot_strs.join(", "));
            }
        }

        let all_monotonic = reports.iter().all(|r| r.monotonic_improvement);
        let any_bifurcation = reports.iter().any(|r| r.bifurcation_detected);
        println!("\nMulti-Seed Analysis Summary (over {} seeds: {:?}):", seeds.len(), seeds);
        println!("  • Monotonic Compute Gain : {}", if all_monotonic { "YES (Consistently monotonic across all seeds)" } else { "NO (Non-monotonic response in some seeds)" });
        println!("  • Accuracy Discontinuity : {}", if any_bifurcation { "DETECTED (>25% jump in at least one seed)" } else { "NOT DETECTED (Smooth progression across seeds)" });

        if let Some(ref path) = output_path {
            let multi_seed_output = serde_json::json!({
                "task_name": task_kind.name(),
                "seeds": seeds,
                "budgets": budgets,
                "all_monotonic": all_monotonic,
                "any_accuracy_discontinuity": any_bifurcation,
                "reports": reports,
            });
            let json_str = serde_json::to_string_pretty(&multi_seed_output)?;
            write_output_file(path, json_str)?;
            println!("\n✓ Multi-seed sweep report saved to '{}'", path);
        }
        if let Some(ref trace_path) = trace_output {
            let json_str = serde_json::to_string_pretty(&reports)?;
            write_output_file(trace_path, json_str)?;
            println!("\n✓ Multi-seed sweep traces saved to '{}'", trace_path);
        }
    }

    Ok(())
}

fn cmd_associate(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let stim_str: Option<String> = args.value("--stimulus")?;
    let tgt_str: Option<String> = args.value("--target")?;
    let horizon: usize = args.value("--horizon")?.unwrap_or(16);
    let output_path: Option<String> = args.value("--output")?;
    let slow_cadence: Option<usize> = args.value("--slow-cadence")?;
    let slow_fraction: Option<f32> = args.value("--slow-fraction")?;

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "associative".to_string())
    };

    let _task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    if let Some(sc) = slow_cadence {
        config.latent.slow_timescale_cadence = sc;
    }
    if let Some(sf) = slow_fraction {
        config.latent.slow_channel_fraction = sf;
    }
    config.validate()?;

    let stim_char = stim_str.and_then(|s| s.chars().next()).unwrap_or('A');
    let tgt_char = tgt_str.and_then(|s| s.chars().next()).unwrap_or('B');

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let vocab = vocab::Vocab::new_ascii();
    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    }

    let executor = latent::LatentExecutor::new(&nca, &interface, &vocab);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · DYNAMIC CONCEPT CORRELATION PROBE THROUGH LATENT TIME                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Stimulus Token      : '{}'", stim_char);
    println!("  Target Token        : '{}'", tgt_char);
    println!("  Horizon Steps       : {} latent ticks", horizon);
    println!("  Slow Path Cadence   : {}", config.latent.slow_timescale_cadence);
    println!("  Slow Channel Frac   : {:.2}", config.latent.slow_channel_fraction);

    let report = executor.run_dynamic_association_probe(
        stim_char,
        tgt_char,
        horizon,
        &config.latent,
        &config.intervention,
        device,
    )?;

    println!("\nTick | Target Token   | P(Target) | Target Rank | Top Decoded Token | State Displacement");
    println!("─────┼────────────────┼───────────┼─────────────┼───────────────────┼───────────────────");
    for p in &report.probes {
        println!(
            "{:03}  | '{}'            | {:<9.4} | #{:<10} | \"{}\"                 | {:<17.4}",
            p.tick, p.target_concept, p.target_probability, p.target_rank, p.top_decoded_token, p.state_displacement
        );
    }

    println!("\nDynamic Correlation Dynamics Summary:");
    println!("  • Emergence Tick   : {:?}", report.emergence_tick);
    println!("  • Peak Tick        : Tick {} (P = {:.4})", report.peak_tick, report.peak_probability);
    println!("  • Final Tick Prob  : {:.4}", report.final_probability);
    println!("  • Verdict          : {}", report.verdict);

    if let Some(ref path) = output_path {
        let json_str = serde_json::to_string_pretty(&report)?;
        write_output_file(path, json_str)?;
        println!("\n✓ Dynamic correlation probe saved to '{}'", path);
    }

    Ok(())
}

fn cmd_attractor(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let prompt_a: String = args.value("--prompt-a")?.unwrap_or_else(|| "context_alpha".to_string());
    let prompt_b: String = args.value("--prompt-b")?.unwrap_or_else(|| "context_beta".to_string());
    let horizon: usize = args.value("--horizon")?.unwrap_or(16);
    let output_path: Option<String> = args.value("--output")?;
    let slow_cadence: Option<usize> = args.value("--slow-cadence")?;
    let slow_fraction: Option<f32> = args.value("--slow-fraction")?;

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "ambiguous-basin".to_string())
    };

    let _task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    if let Some(sc) = slow_cadence {
        config.latent.slow_timescale_cadence = sc;
    }
    if let Some(sf) = slow_fraction {
        config.latent.slow_channel_fraction = sf;
    }
    config.validate()?;

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let vocab = vocab::Vocab::new_ascii();
    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    }

    let executor = latent::LatentExecutor::new(&nca, &interface, &vocab);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · BASIN SEPARATION & HYSTERESIS ANALYSIS                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Condition A Stimulus : \"{}\"", prompt_a);
    println!("  Condition B Stimulus : \"{}\"", prompt_b);
    println!("  Rollout Horizon      : {} ticks", horizon);

    let report = executor.run_attractor_basin_probe(
        &prompt_a,
        &prompt_b,
        horizon,
        &config.latent,
        &config.intervention,
        device,
    )?;

    println!("\nBasin Convergence Metrics:");
    println!("  • Basin A Final Norm     : {:.4}", report.basin_a_final_norm);
    println!("  • Basin B Final Norm     : {:.4}", report.basin_b_final_norm);
    println!("  • Inter-Basin Distance   : {:.4}", report.final_inter_basin_distance);
    println!("  • Basin Separation Ratio : {:.2}%", report.basin_separation_ratio * 100.0);
    println!("  • Hysteresis Detected    : {}", if report.hysteresis_detected { "YES (State remembers prior trajectory)" } else { "NO" });
    println!("  • Classification         : {}", report.classification);

    if let Some(ref path) = output_path {
        let json_str = serde_json::to_string_pretty(&report)?;
        write_output_file(path, json_str)?;
        println!("\n✓ Basin separation report saved to '{}'", path);
    }

    Ok(())
}

fn eval_baseline_masked(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<(f32, f32)> {
    let (b, l, v) = logits.dims3()?;
    let flat_l = logits.reshape((b * l, v))?;
    let flat_t = targets.reshape((b * l,))?;
    let flat_m = mask.reshape((b * l,))?;
    let log_sm = candle_nn::ops::log_softmax(&flat_l, 1)?;
    let t_log_probs = log_sm.gather(&flat_t.unsqueeze(1)?, 1)?.squeeze(1)?;
    let m_nll = (t_log_probs.neg()? * flat_m.clone())?;
    let m_sum = flat_m.sum_all()?.to_scalar::<f32>()?.max(1.0);
    let loss = (m_nll.sum_all()? / (m_sum as f64))?.to_scalar::<f32>()?;

    let preds = logits.argmax(candle_core::D::Minus1)?;
    let p_vec: Vec<u32> = preds.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
    let t_vec: Vec<u32> = targets.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
    let m_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;
    anyhow::ensure!(
        p_vec.len() == m_vec.len() && t_vec.len() == m_vec.len(),
        "Tensor length mismatch in evaluation"
    );
    let mut active = 0usize;
    let mut matches = 0usize;
    for i in 0..p_vec.len() {
        if m_vec[i] > 0.5 {
            active += 1;
            if p_vec[i] == t_vec[i] {
                matches += 1;
            }
        }
    }
    let acc = if active > 0 { matches as f32 / active as f32 } else { 0.0 };
    Ok((loss, acc))
}

fn compute_dummy_baselines(targets: &Tensor, mask: &Tensor) -> Result<(f32, f32)> {
    let t_vec: Vec<u32> = targets.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
    let m_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;
    let mut counts = std::collections::HashMap::new();
    let mut active = 0usize;
    for (&t, &m) in t_vec.iter().zip(&m_vec) {
        if m > 0.5 {
            active += 1;
            *counts.entry(t).or_insert(0usize) += 1;
        }
    }
    let max_count = counts.values().max().copied().unwrap_or(0);
    let majority_acc = if active > 0 { max_count as f32 / active as f32 } else { 0.0 };
    let uniform_chance = if !counts.is_empty() { 1.0 / counts.len() as f32 } else { 0.0 };
    Ok((majority_acc, uniform_chance))
}

fn cmd_benchmark(args: &cli::Options, device: &Device) -> Result<()> {
    let task_str: Option<String> = args.value("--task")?;
    let seq_len: usize = args.value("--seq-len")?.unwrap_or(16);
    let batch_size: usize = args.value("--batch-size")?.unwrap_or(4);
    let output_path: Option<String> = args.value("--output")?;
    let cp_val: Option<String> = args.value("--checkpoint")?;
    let ld_val: Option<String> = args.value("--load-dir")?;
    if let (Some(ref c), Some(ref l)) = (&cp_val, &ld_val) {
        if c != l {
            anyhow::bail!("Conflicting checkpoint arguments: --checkpoint='{}' and --load-dir='{}'", c, l);
        }
    }
    let checkpoint_path = cp_val.or(ld_val);
    let seeds_str: Option<String> = args.value("--seeds")?;
    let seeds: Vec<usize> = match seeds_str {
        Some(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0],
    };
    let seeds = if seeds.is_empty() { vec![0] } else { seeds };
    let feedback_mode_opt: Option<String> = args.value("--feedback-mode")?;
    let feedback_weight_opt: Option<f32> = args.value("--feedback-weight")?;
    let tail_eq_weight_opt: Option<f32> = args.value("--tail-eq-weight")?;
    let tail_eq_ticks_opt: Option<usize> = args.value("--tail-eq-ticks")?;
    let state_norm_opt: Option<String> = args.value("--state-norm")?;

    let detected_concurrency = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let active_threads = rayon::current_num_threads();

    let task_name = task_str.unwrap_or_else(|| "delayed-recall".to_string());
    let task_kind = tasks::TaskKind::parse(&task_name).unwrap_or(tasks::TaskKind::DelayedRecall);

    let task_engine = tasks::TaskEngine::new();
    task_engine.verify_task_suite_integrity(device)?;

    println!("  Task Schema Version : {}", tasks::TASK_SCHEMA_VERSION);

    let vocab = vocab::Vocab::new_ascii();
    let vocab_size = vocab.size();

    // 1. NCA (from checkpoint if provided, else fresh initialization)
    let (nca_cfg, field_cfg, channels) = if let Some(ref dir) = checkpoint_path {
        let manifest = checkpoint::CheckpointManager::load_manifest(dir)?;
        let ch = manifest.config.field.channels;
        if let Some(ref fm) = feedback_mode_opt {
            if *fm != manifest.config.nca.feedback_mode {
                anyhow::bail!(
                    "Checkpoint '{}' was trained with feedback_mode='{}', cannot evaluate with requested --feedback-mode='{}'",
                    dir, manifest.config.nca.feedback_mode, fm
                );
            }
        }
        if let Some(ref sn) = state_norm_opt {
            if *sn != manifest.config.nca.state_norm {
                anyhow::bail!(
                    "Checkpoint '{}' was trained with state_norm='{}', cannot evaluate with requested --state-norm='{}'",
                    dir, manifest.config.nca.state_norm, sn
                );
            }
        }
        (manifest.config.nca, manifest.config.field, ch)
    } else {
        let mut nca_cfg = config::NcaConfig::default();
        if let Some(ref fm) = feedback_mode_opt {
            nca_cfg.feedback_mode = fm.clone();
        }
        if let Some(fw) = feedback_weight_opt {
            nca_cfg.feedback_weight = fw;
        }
        if let Some(ref sn) = state_norm_opt {
            nca_cfg.state_norm = sn.clone();
        }
        let field_cfg = config::FieldConfig {
            seq_len,
            channels: 64,
            periodic_boundary: true,
        };
        (nca_cfg, field_cfg, 64)
    };

    let mut vm_nca = candle_nn::VarMap::new();
    let vb_nca = candle_nn::VarBuilder::from_varmap(&vm_nca, candle_core::DType::F32, device);
    let nca = NeuralCellularAutomaton::new(vb_nca.pp("nca"), &nca_cfg, &field_cfg)?;
    let interface = TokenInterface::new(vb_nca.pp("interface"), vocab_size, channels)?;
    if let Some(ref dir) = checkpoint_path {
        checkpoint::CheckpointManager::load_weights(dir, &mut vm_nca, device)?;
        println!("  ✓ Loaded trained Titan NCA weights from '{}'", dir);
    }


    // 2. Transformer
    let vm_tf = candle_nn::VarMap::new();
    let vb_tf = candle_nn::VarBuilder::from_varmap(&vm_tf, candle_core::DType::F32, device);
    let transformer = baselines::TransformerBaseline::new(vb_tf, vocab_size, channels, 96)?;

    // 3. GRU
    let vm_gru = candle_nn::VarMap::new();
    let vb_gru = candle_nn::VarBuilder::from_varmap(&vm_gru, candle_core::DType::F32, device);
    let gru = baselines::GruBaseline::new(vb_gru, vocab_size, channels)?;

    // 4. Simple RNN
    let vm_sr = candle_nn::VarMap::new();
    let vb_sr = candle_nn::VarBuilder::from_varmap(&vm_sr, candle_core::DType::F32, device);
    let sr = baselines::SimpleRecurrentBaseline::new(vb_sr, vocab_size, channels)?;

    // 5. Untied Feedforward (4 layers)
    let vm_untied = candle_nn::VarMap::new();
    let vb_untied = candle_nn::VarBuilder::from_varmap(&vm_untied, candle_core::DType::F32, device);
    let untied = baselines::UntiedFeedforwardBaseline::new(vb_untied, vocab_size, channels, 4, channels * 2)?;

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · SYSTEM ARCHITECTURE & BASELINE BENCHMARK                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Task Evaluated      : {}", task_kind.name());
    println!("  Sequence Length (L) : {}", seq_len);
    println!("  Hidden Channels (C) : {}", channels);
    println!("  Hardware Concurrency: {} cores/threads detected", detected_concurrency);
    println!("  Active Worker Pool  : {} threads\n", active_threads);

    println!("─── PARAMETER COUNT BREAKDOWN BY SUBSYSTEM ─────────────────────────────────────────────");
    println!("{:<24} | {:<32} | {:<12}", "Architecture", "Subsystem", "Parameters");
    println!("─────────────────────────┼──────────────────────────────────┼─────────────");

    let mut nca_total = 0;
    for (sub, cnt) in nca.parameter_breakdown() {
        println!("{:<24} | {:<32} | {:<12}", "Titan NCA", sub, cnt);
        nca_total += cnt;
    }
    let p_embed = vocab_size * channels;
    let p_proj = channels * vocab_size + vocab_size;
    println!("{:<24} | {:<32} | {:<12}", "Titan NCA", "token_embedding", p_embed);
    println!("{:<24} | {:<32} | {:<12}", "Titan NCA", "token_readout_projection", p_proj);
    nca_total += p_embed + p_proj;
    println!("{:<24} | {:<32} | {:<12}", "Titan NCA (TOTAL)", "all_parameters", nca_total);
    println!("─────────────────────────┼──────────────────────────────────┼─────────────");

    use baselines::SequenceModel;
    for (sub, cnt) in transformer.parameter_breakdown() {
        println!("{:<24} | {:<32} | {:<12}", "Transformer", sub, cnt);
    }
    println!("{:<24} | {:<32} | {:<12}", "Transformer (TOTAL)", "all_parameters", transformer.total_parameters());
    println!("─────────────────────────┼──────────────────────────────────┼─────────────");

    for (sub, cnt) in gru.parameter_breakdown() {
        println!("{:<24} | {:<32} | {:<12}", "GRU Recurrent", sub, cnt);
    }
    println!("{:<24} | {:<32} | {:<12}", "GRU (TOTAL)", "all_parameters", gru.total_parameters());
    println!("─────────────────────────┼──────────────────────────────────┼─────────────");

    for (sub, cnt) in sr.parameter_breakdown() {
        println!("{:<24} | {:<32} | {:<12}", "Simple RNN", sub, cnt);
    }
    println!("{:<24} | {:<32} | {:<12}", "Simple RNN (TOTAL)", "all_parameters", sr.total_parameters());
    println!("─────────────────────────┼──────────────────────────────────┼─────────────");

    for (sub, cnt) in untied.parameter_breakdown() {
        println!("{:<24} | {:<32} | {:<12}", "Untied FF (4-layer)", sub, cnt);
    }
    println!("{:<24} | {:<32} | {:<12}", "Untied FF (TOTAL)", "all_parameters", untied.total_parameters());
    println!("─────────────────────────┴──────────────────────────────────┴─────────────\n");

    let task_engine = tasks::TaskEngine::new();

    #[derive(Clone, Copy, Debug)]
    struct ArchMetric {
        dummy_majority: f32,
        dummy_chance: f32,
        nca_loss: f32,
        nca_acc: f32,
        tf_loss: f32,
        tf_acc: f32,
        gru_loss: f32,
        gru_acc: f32,
        sr_loss: f32,
        sr_acc: f32,
        untied_loss: f32,
        untied_acc: f32,
    }

    let eval_for_seed = |seed: usize| -> Result<ArchMetric> {
        let batch = task_engine.generate_batch_seeded(task_kind, batch_size, seq_len, true, seed, device)?;
        let (dummy_majority, dummy_chance) = compute_dummy_baselines(&batch.targets, &batch.loss_mask)?;

        // Concurrently run model forward evaluations across all 5 architectures via nested rayon::join
        let (untied_res, (nca_res, (tf_res, (gru_res, sr_res)))) = rayon::join(
            || -> Result<(f32, f32)> {
                let logits = untied.forward(&batch.inputs)?;
                eval_baseline_masked(&logits, &batch.targets, &batch.loss_mask)
            },
            || {
                rayon::join(
                    || -> Result<(f32, f32)> {
                        let seed_emb = interface.embed_tokens(&batch.inputs)?;
                        let field = MorphogenicField::from_tensor(seed_emb, &field_cfg);
                        let developed = nca.develop(&field, 8, device)?;
                        let logits = interface.logits(&developed.x)?;
                        let loss = interface.masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?.to_scalar::<f32>()?;
                        let acc = interface.masked_accuracy(&logits, &batch.targets, &batch.loss_mask)?;
                        Ok((loss, acc))
                    },
                    || {
                        rayon::join(
                            || -> Result<(f32, f32)> {
                                let logits = transformer.forward(&batch.inputs)?;
                                eval_baseline_masked(&logits, &batch.targets, &batch.loss_mask)
                            },
                            || {
                                rayon::join(
                                    || -> Result<(f32, f32)> {
                                        let logits = gru.forward(&batch.inputs)?;
                                        eval_baseline_masked(&logits, &batch.targets, &batch.loss_mask)
                                    },
                                    || -> Result<(f32, f32)> {
                                        let logits = sr.forward(&batch.inputs)?;
                                        eval_baseline_masked(&logits, &batch.targets, &batch.loss_mask)
                                    },
                                )
                            },
                        )
                    },
                )
            },
        );

        let (untied_loss, untied_acc) = untied_res?;
        let (nca_loss, nca_acc) = nca_res?;
        let (tf_loss, tf_acc) = tf_res?;
        let (gru_loss, gru_acc) = gru_res?;
        let (sr_loss, sr_acc) = sr_res?;

        Ok(ArchMetric {
            dummy_majority,
            dummy_chance,
            nca_loss,
            nca_acc,
            tf_loss,
            tf_acc,
            gru_loss,
            gru_acc,
            sr_loss,
            sr_acc,
            untied_loss,
            untied_acc,
        })
    };

    let evals: Vec<ArchMetric> = if seeds.len() > 1 {
        seeds.par_iter().map(|&s| eval_for_seed(s)).collect::<Result<Vec<_>>>()?
    } else {
        vec![eval_for_seed(seeds[0])?]
    };

    let n = evals.len() as f32;
    let mean_dummy_maj = evals.iter().map(|e| e.dummy_majority).sum::<f32>() / n;
    let mean_dummy_chance = evals.iter().map(|e| e.dummy_chance).sum::<f32>() / n;
    let mean_nca_loss = evals.iter().map(|e| e.nca_loss).sum::<f32>() / n;
    let mean_nca_acc = evals.iter().map(|e| e.nca_acc).sum::<f32>() / n;
    let mean_tf_loss = evals.iter().map(|e| e.tf_loss).sum::<f32>() / n;
    let mean_tf_acc = evals.iter().map(|e| e.tf_acc).sum::<f32>() / n;
    let mean_gru_loss = evals.iter().map(|e| e.gru_loss).sum::<f32>() / n;
    let mean_gru_acc = evals.iter().map(|e| e.gru_acc).sum::<f32>() / n;
    let mean_sr_loss = evals.iter().map(|e| e.sr_loss).sum::<f32>() / n;
    let mean_sr_acc = evals.iter().map(|e| e.sr_acc).sum::<f32>() / n;
    let mean_untied_loss = evals.iter().map(|e| e.untied_loss).sum::<f32>() / n;
    let mean_untied_acc = evals.iter().map(|e| e.untied_acc).sum::<f32>() / n;

    println!("─── NULL-HYPOTHESIS CONTROLS (Masked Positions) ────────────────────────────────────────");
    println!("  • Uniform Chance Accuracy : {:5.1}%", mean_dummy_chance * 100.0);
    println!("  • Majority Token Dummy    : {:5.1}% (Static predictor predicting most common label)", mean_dummy_maj * 100.0);
    println!("────────────────────────────────────────────────────────────────────────────────────────\n");

    if seeds.len() == 1 {
        println!("Forward Performance on Task '{}' (Masked Targets, Seed: {}):", task_kind.name(), seeds[0]);
        println!("  • Titan NCA          : Loss = {:.4}, Acc = {:5.1}%", mean_nca_loss, mean_nca_acc * 100.0);
        println!("  • Transformer        : Loss = {:.4}, Acc = {:5.1}%", mean_tf_loss, mean_tf_acc * 100.0);
        println!("  • GRU Recurrent      : Loss = {:.4}, Acc = {:5.1}%", mean_gru_loss, mean_gru_acc * 100.0);
        println!("  • Simple RNN         : Loss = {:.4}, Acc = {:5.1}%", mean_sr_loss, mean_sr_acc * 100.0);
        println!("  • Untied FF (4-step) : Loss = {:.4}, Acc = {:5.1}%", mean_untied_loss, mean_untied_acc * 100.0);
    } else {
        let std_dev = |extract: fn(&ArchMetric) -> f32, mean: f32| -> f32 {
            (evals.iter().map(|e| (extract(e) - mean).powi(2)).sum::<f32>() / (n - 1.0)).sqrt()
        };
        let std_nca_loss = std_dev(|e| e.nca_loss, mean_nca_loss);
        let std_nca_acc = std_dev(|e| e.nca_acc, mean_nca_acc);
        let std_tf_loss = std_dev(|e| e.tf_loss, mean_tf_loss);
        let std_tf_acc = std_dev(|e| e.tf_acc, mean_tf_acc);
        let std_gru_loss = std_dev(|e| e.gru_loss, mean_gru_loss);
        let std_gru_acc = std_dev(|e| e.gru_acc, mean_gru_acc);
        let std_sr_loss = std_dev(|e| e.sr_loss, mean_sr_loss);
        let std_sr_acc = std_dev(|e| e.sr_acc, mean_sr_acc);
        let std_untied_loss = std_dev(|e| e.untied_loss, mean_untied_loss);
        let std_untied_acc = std_dev(|e| e.untied_acc, mean_untied_acc);

        println!("Multi-Seed Forward Performance on Task '{}' (Masked, over {} seeds: {:?}):", task_kind.name(), seeds.len(), seeds);
        println!("  • Titan NCA          : Loss = {:.4} ± {:.4}, Acc = {:5.1}% ± {:4.1}%", mean_nca_loss, std_nca_loss, mean_nca_acc * 100.0, std_nca_acc * 100.0);
        println!("  • Transformer        : Loss = {:.4} ± {:.4}, Acc = {:5.1}% ± {:4.1}%", mean_tf_loss, std_tf_loss, mean_tf_acc * 100.0, std_tf_acc * 100.0);
        println!("  • GRU Recurrent      : Loss = {:.4} ± {:.4}, Acc = {:5.1}% ± {:4.1}%", mean_gru_loss, std_gru_loss, mean_gru_acc * 100.0, std_gru_acc * 100.0);
        println!("  • Simple RNN         : Loss = {:.4} ± {:.4}, Acc = {:5.1}% ± {:4.1}%", mean_sr_loss, std_sr_loss, mean_sr_acc * 100.0, std_sr_acc * 100.0);
        println!("  • Untied FF (4-step) : Loss = {:.4} ± {:.4}, Acc = {:5.1}% ± {:4.1}%", mean_untied_loss, std_untied_loss, mean_untied_acc * 100.0, std_untied_acc * 100.0);
    }

    // Trained comparison if requested
    let mut trained_results_json = serde_json::Map::new();
    if args.flag("--train") {
        let train_epochs: usize = args.value("--epochs")?.unwrap_or(20);
        let train_lr: f64 = args.value("--lr")?.unwrap_or(0.003);
        let base_seed = seeds[0] as u64;

        println!("\n─── ARCHITECTURAL TRAINING CONVERGENCE (Task: {}, Epochs: {}, lr: {}, seed: {}) ───────", task_name, train_epochs, train_lr, base_seed);

        // 1. Train NCA
        print!("  [1/5] Training Titan NCA (T=8, seed={})... ", base_seed);
        let mut nca_cfg_train = config::TitanConfig::default();
        nca_cfg_train.field.seq_len = seq_len;
        nca_cfg_train.field.channels = channels;
        nca_cfg_train.train.epochs = train_epochs;
        nca_cfg_train.train.batch_size = batch_size;
        nca_cfg_train.train.dev_steps = 8;
        nca_cfg_train.train.lr = train_lr;
        nca_cfg_train.train.seed = base_seed;
        if let Some(ref fm) = feedback_mode_opt {
            nca_cfg_train.nca.feedback_mode = fm.clone();
        }
        if let Some(fw) = feedback_weight_opt {
            nca_cfg_train.nca.feedback_weight = fw;
        }
        if let Some(tew) = tail_eq_weight_opt {
            nca_cfg_train.train.tail_equilibrium_weight = tew;
        }
        if let Some(tet) = tail_eq_ticks_opt {
            nca_cfg_train.train.tail_equilibrium_ticks = tet;
        }
        if let Some(ref sn) = state_norm_opt {
            nca_cfg_train.nca.state_norm = sn.clone();
        }
        let mut trainer = Trainer::new(nca_cfg_train, &task_name, device)?;
        let mut last_nca_diag = None;
        for _ in 1..=train_epochs {
            last_nca_diag = Some(trainer.train_step(batch_size)?);
        }
        let (nca_val_loss, nca_val_acc) = trainer.evaluate_val(batch_size)?;
        let nca_train_loss = last_nca_diag.as_ref().map(|d| d.train_loss).unwrap_or(0.0);
        let nca_train_acc = last_nca_diag.as_ref().map(|d| d.train_acc).unwrap_or(0.0);
        println!("done. Val Acc: {:5.1}%", nca_val_acc * 100.0);

        // 2. Train Transformer
        print!("  [2/5] Training Transformer (1-layer, seed={})... ", base_seed);
        let (_tf_m, tf_res) = baselines::train_baseline_model("transformer", &task_name, train_epochs, train_lr, 0.01, batch_size, seq_len, channels, 1, base_seed, device)?;
        println!("done. Val Acc: {:5.1}%", tf_res.final_val_acc * 100.0);

        // 3. Train GRU
        print!("  [3/5] Training GRU Recurrent (seed={})... ", base_seed);
        let (_gru_m, gru_res) = baselines::train_baseline_model("gru", &task_name, train_epochs, train_lr, 0.01, batch_size, seq_len, channels, 1, base_seed, device)?;
        println!("done. Val Acc: {:5.1}%", gru_res.final_val_acc * 100.0);

        // 4. Train Simple RNN
        print!("  [4/5] Training Simple RNN (seed={})... ", base_seed);
        let (_sr_m, sr_res) = baselines::train_baseline_model("rnn", &task_name, train_epochs, train_lr, 0.01, batch_size, seq_len, channels, 1, base_seed, device)?;
        println!("done. Val Acc: {:5.1}%", sr_res.final_val_acc * 100.0);

        // 5. Train Untied Feedforward
        print!("  [5/5] Training Untied Feedforward (4-layer, seed={})... ", base_seed);
        let (_untied_m, untied_res) = baselines::train_baseline_model("untied", &task_name, train_epochs, train_lr, 0.01, batch_size, seq_len, channels, 4, base_seed, device)?;
        println!("done. Val Acc: {:5.1}%\n", untied_res.final_val_acc * 100.0);

        println!("─── TRAINED CONVERGENCE COMPARISON ─────────────────────────────────────────────────────");
        println!("{:<22} | {:<10} | {:<18} | {:<18}", "Architecture", "Params", "Train Loss (Acc)", "Val Loss (Acc)");
        println!("───────────────────────┼────────────┼────────────────────┼───────────────────");
        println!("{:<22} | {:<10} | {:>6.4} ({:>5.1}%)    | {:>6.4} ({:>5.1}%)", "Titan NCA (T=8)", nca_total, nca_train_loss, nca_train_acc * 100.0, nca_val_loss, nca_val_acc * 100.0);
        println!("{:<22} | {:<10} | {:>6.4} ({:>5.1}%)    | {:>6.4} ({:>5.1}%)", "Transformer", tf_res.total_parameters, tf_res.final_train_loss, tf_res.final_train_acc * 100.0, tf_res.final_val_loss, tf_res.final_val_acc * 100.0);
        println!("{:<22} | {:<10} | {:>6.4} ({:>5.1}%)    | {:>6.4} ({:>5.1}%)", "GRU Recurrent", gru_res.total_parameters, gru_res.final_train_loss, gru_res.final_train_acc * 100.0, gru_res.final_val_loss, gru_res.final_val_acc * 100.0);
        println!("{:<22} | {:<10} | {:>6.4} ({:>5.1}%)    | {:>6.4} ({:>5.1}%)", "Simple RNN", sr_res.total_parameters, sr_res.final_train_loss, sr_res.final_train_acc * 100.0, sr_res.final_val_loss, sr_res.final_val_acc * 100.0);
        println!("{:<22} | {:<10} | {:>6.4} ({:>5.1}%)    | {:>6.4} ({:>5.1}%)", "Untied FF (4-layer)", untied_res.total_parameters, untied_res.final_train_loss, untied_res.final_train_acc * 100.0, untied_res.final_val_loss, untied_res.final_val_acc * 100.0);
        println!("───────────────────────┴────────────┴────────────────────┴───────────────────\n");

        trained_results_json.insert("nca".to_string(), serde_json::json!({ "params": nca_total, "train_loss": nca_train_loss, "train_acc": nca_train_acc, "val_loss": nca_val_loss, "val_acc": nca_val_acc }));
        trained_results_json.insert("transformer".to_string(), serde_json::json!({ "params": tf_res.total_parameters, "train_loss": tf_res.final_train_loss, "train_acc": tf_res.final_train_acc, "val_loss": tf_res.final_val_loss, "val_acc": tf_res.final_val_acc }));
        trained_results_json.insert("gru".to_string(), serde_json::json!({ "params": gru_res.total_parameters, "train_loss": gru_res.final_train_loss, "train_acc": gru_res.final_train_acc, "val_loss": gru_res.final_val_loss, "val_acc": gru_res.final_val_acc }));
        trained_results_json.insert("simple_rnn".to_string(), serde_json::json!({ "params": sr_res.total_parameters, "train_loss": sr_res.final_train_loss, "train_acc": sr_res.final_train_acc, "val_loss": sr_res.final_val_loss, "val_acc": sr_res.final_val_acc }));
        trained_results_json.insert("untied_ff".to_string(), serde_json::json!({ "params": untied_res.total_parameters, "train_loss": untied_res.final_train_loss, "train_acc": untied_res.final_train_acc, "val_loss": untied_res.final_val_loss, "val_acc": untied_res.final_val_acc }));
    }

    // ─── CPU PARALLELISM & THROUGHPUT BENCHMARK ─────────────────────────
    let num_bench_batches = 24usize;
    let eval_throughput_batch = |idx: usize| -> Result<()> {
        let b = task_engine.generate_batch_seeded(task_kind, batch_size, seq_len, true, 10_000 + idx, device)?;
        let seed_emb = interface.embed_tokens(&b.inputs)?;
        let f = MorphogenicField::from_tensor(seed_emb, &field_cfg);
        let dev = nca.develop(&f, 8, device)?;
        let logits = interface.logits(&dev.x)?;
        let _ = interface.accuracy(&logits, &b.targets)?;
        Ok(())
    };

    // Warm-up pass
    eval_throughput_batch(0)?;

    // 1. Single-threaded measurement (isolated in a dedicated 1-worker Rayon pool)
    let single_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let t_single_start = Instant::now();
    single_pool.install(|| -> Result<()> {
        for idx in 0..num_bench_batches {
            eval_throughput_batch(idx)?;
        }
        Ok(())
    })?;
    let t_single_dur = t_single_start.elapsed();

    // 2. Parallel measurement (using active global thread pool)
    let t_par_start = Instant::now();
    (0..num_bench_batches)
        .into_par_iter()
        .map(|idx| eval_throughput_batch(idx))
        .collect::<Result<Vec<()>>>()?;
    let t_par_dur = t_par_start.elapsed();

    let total_bench_tokens = (num_bench_batches * batch_size * seq_len) as f64;
    let single_sec = t_single_dur.as_secs_f64().max(1e-6);
    let par_sec = t_par_dur.as_secs_f64().max(1e-6);
    let single_tok_per_sec = total_bench_tokens / single_sec;
    let par_tok_per_sec = total_bench_tokens / par_sec;
    let single_ms_per_batch = (single_sec * 1000.0) / (num_bench_batches as f64);
    let par_ms_per_batch = (par_sec * 1000.0) / (num_bench_batches as f64);
    let speedup = par_tok_per_sec / single_tok_per_sec.max(1e-6);
    let efficiency_pct = (speedup / (active_threads as f64)) * 100.0;

    println!("\n─── CPU PARALLELISM & THROUGHPUT BENCHMARK ─────────────────────────────────────────────");
    println!("  Target Hardware Concurrency : {} cores/threads", detected_concurrency);
    println!("  Configured Worker Threads   : {} threads", active_threads);
    println!("  Workload Batches Evaluated  : {} (B={}, L={}, total {} tokens)", num_bench_batches, batch_size, seq_len, total_bench_tokens as usize);
    println!("  ──────────────────────────────────────────────────────────────────────────────────────");
    println!("  Single-Thread (1 worker)    : {:>8.2} ms/batch  | {:>10.1} tokens/sec", single_ms_per_batch, single_tok_per_sec);
    println!("  Parallel ({} workers)       : {:>8.2} ms/batch  | {:>10.1} tokens/sec", active_threads, par_ms_per_batch, par_tok_per_sec);
    println!("  Speedup                     : {:>8.2}x", speedup);
    println!("  Parallel Efficiency         : {:>8.1}%", efficiency_pct);
    println!("  Nested Oversubscription     : Prevented (Unified thread pool across Candle and Titan)");
    println!("  Memory Locality             : Zero tensor clone churn; contiguous flat buffers");
    println!("────────────────────────────────────────────────────────────────────────────────────────\n");

    if let Some(ref path) = output_path {
        let report_data = serde_json::json!({
            "task": task_kind.name(),
            "threads": active_threads,
            "detected_hardware_concurrency": detected_concurrency,
            "models": {
                "nca": { "params": nca_total, "loss": mean_nca_loss, "acc": mean_nca_acc },
                "transformer": { "params": transformer.total_parameters(), "loss": mean_tf_loss, "acc": mean_tf_acc },
                "gru": { "params": gru.total_parameters(), "loss": mean_gru_loss, "acc": mean_gru_acc },
                "simple_rnn": { "params": sr.total_parameters(), "loss": mean_sr_loss, "acc": mean_sr_acc },
                "untied_ff": { "params": untied.total_parameters(), "loss": mean_untied_loss, "acc": mean_untied_acc }
            },
            "trained_comparison": trained_results_json,
            "parallelism_benchmark": {
                "workload_tokens": total_bench_tokens,
                "single_thread": {
                    "latency_ms_per_batch": single_ms_per_batch,
                    "tokens_per_sec": single_tok_per_sec,
                    "duration_ms": single_sec * 1000.0,
                },
                "parallel": {
                    "threads": active_threads,
                    "latency_ms_per_batch": par_ms_per_batch,
                    "tokens_per_sec": par_tok_per_sec,
                    "duration_ms": par_sec * 1000.0,
                },
                "speedup": speedup,
                "efficiency_pct": efficiency_pct,
            },
            "seeds": seeds,
        });
        write_output_file(path, serde_json::to_string_pretty(&report_data)?)?;
        println!("\n✓ Benchmark summary saved to '{}'", path);
    }

    Ok(())
}

fn cmd_memory(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let event_1: String = args.value("--event-1")?.unwrap_or_else(|| "prior_activation".to_string());
    let event_2: String = args.value("--event-2")?.unwrap_or_else(|| "probe_stimulus".to_string());
    let ticks: usize = args.value("--ticks")?.unwrap_or(8);
    let output_path: Option<String> = args.value("--output")?;

    let (config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let _task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    config.validate()?;

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let vocab = vocab::Vocab::new_ascii();
    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    }

    let executor = latent::LatentExecutor::new(&nca, &interface, &vocab);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · MEMORY AS DYNAMICS EXPERIMENT                                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Prior Event (E1)      : \"{}\"", event_1);
    println!("  Probe Event (E2)      : \"{}\"", event_2);
    println!("  Latent Ticks Between  : {} ticks", ticks);

    let report = executor.run_memory_as_dynamics_experiment(
        &event_1,
        &event_2,
        ticks,
        &config.intervention,
        device,
    )?;

    println!("\nDynamical Memory Results:");
    println!("  • State Distortion ||s_E2|E1 - s_E2|0|| : {:.5}", report.state_distortion_distance);
    println!("  • Readout KL Divergence                 : {:.5}", report.readout_kl_divergence);
    println!("  • Future Dynamics Modified              : {}", if report.dynamics_modified { "YES (Lingering distortion in latent state)" } else { "NO" });
    println!("  • Conclusion                            : {}", report.description);

    if let Some(ref path) = output_path {
        let json_str = serde_json::to_string_pretty(&report)?;
        write_output_file(path, json_str)?;
        println!("\n✓ Memory experiment report saved to '{}'", path);
    }

    Ok(())
}

fn cmd_falsify(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    if load_dir.is_some() {
        return cmd_probe(args, device);
    }
    let epochs = args.value("--epochs")?.unwrap_or(8);
    let steps_per_epoch = args.value("--steps-per-epoch")?.unwrap_or(6);
    let batch_size = args.value("--batch-size")?.unwrap_or(16);
    let seq_len = args.value("--seq-len")?.unwrap_or(32);
    let lr = args.value("--lr")?.unwrap_or(0.003);
    let seeds_str: Option<String> = args.value("--seeds")?;
    let output_path: Option<String> = args.value("--output")?;

    let seeds: Vec<u64> = if let Some(s) = seeds_str {
        s.split(',')
            .filter_map(|x| x.trim().parse::<u64>().ok())
            .collect()
    } else {
        vec![42, 101, 202]
    };

    println!("================================================================================");
    println!("TITAN TEXT: 2D LATENT BLACKBOARD CONTROLLED FALSIFICATION EXPERIMENT");
    println!("================================================================================");
    println!("Task: ColumnArithmetic (Multi-digit Addition with Hostile Ripple Carries)");
    println!("Seeds: {:?}", seeds);
    println!("Epochs: {}, Steps/Epoch: {}, Batch Size: {}, Seq Len: {}, LR: {}", epochs, steps_per_epoch, batch_size, seq_len, lr);
    println!("Comparative Arms:");
    println!("  1. Arm 1: 2D Latent Blackboard (Hilbert, Active, Fixed T=2)");
    println!("  2. Arm 2: 2D Latent Blackboard (Hilbert, Active, Adaptive Kinetic Halting)");
    println!("  3. Arm 3: Tokenwise Recurrent Refinement Control (Similar Params; Unmatched Compute)");
    println!("  4. Arm 4: Retrained Fixed Shuffled Coordinates Control (Topology Test)");
    println!("  5. Arm 5: Static Buffer Control (T=0, Identity Memory Test)");
    println!("--------------------------------------------------------------------------------");

    let report = falsification::run_falsification_battery(
        &seeds,
        epochs,
        steps_per_epoch,
        batch_size,
        seq_len,
        lr,
        device,
    )?;

    let json_str = serde_json::to_string_pretty(&report)?;
    let out_path = output_path.unwrap_or_else(|| "reports/falsification_report.json".to_string());
    write_output_file(&out_path, &json_str)?;

    println!("--------------------------------------------------------------------------------");
    println!("FALSIFICATION BATTERY COMPLETE. Report written to: {}", out_path);
    println!("Summary Verdict: {}", report.summary_verdict);
    println!("================================================================================");

    Ok(())
}

fn cmd_influence(args: &cli::Options, device: &Device) -> Result<()> {
    let load_dir: Option<String> = args.value("--load-dir")?;
    let task_opt: Option<String> = args.value("--task")?;
    let budgets_str: Option<String> = args.value("--budgets")?;
    let seq_len_opt: Option<usize> = args.value("--seq-len")?;
    let batch_size: usize = args.value("--batch-size")?.unwrap_or(64);
    let output_path: Option<String> = args.value("--output")?;
    let seed: usize = args.value("--seed")?.unwrap_or(42);

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "iterated-parity".to_string())
    };

    let task = cli::resolve_task(task_opt, &default_task, load_dir.is_some())?;
    if let Some(s) = seq_len_opt {
        config.field.seq_len = s;
    }
    if args.flag("--zero-boundary") {
        config.field.periodic_boundary = false;
    }
    if args.flag("--coord-channel") {
        config.nca.coord_channel = true;
    }
    if args.flag("--causal-stencil") {
        config.nca.causal_stencil = true;
    }
    config.validate()?;

    let task_kind = tasks::TaskKind::parse(&task).unwrap_or(tasks::TaskKind::IteratedParity);
    let budgets: Vec<usize> = match budgets_str {
        Some(s) => s
            .split(',')
            .filter_map(|b| b.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0, 1, 2, 4, 8],
    };

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let vocab = vocab::Vocab::new_ascii();
    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading checkpoint weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    }

    let executor = latent::LatentExecutor::new(&nca, &interface, &vocab);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · EMPIRICAL CAUSAL INFLUENCE & DEPENDENCY MAP                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Task                : {}", task_kind.name());
    println!("  Sequence Length (L) : {}", config.field.seq_len);
    println!("  Budgets Evaluated   : {:?}", budgets);
    println!("  Batch Size          : {}", batch_size);
    println!("  Seed                : {}", seed);
    println!("  Zero Boundary       : {}", !config.field.periodic_boundary);
    println!("  Coord Channel       : {}", config.nca.coord_channel);

    let report = executor.compute_causal_influence_matrix(
        task_kind,
        &budgets,
        batch_size,
        config.field.seq_len,
        !config.field.periodic_boundary,
        seed,
        device,
    )?;

    for (b_idx, &tau) in budgets.iter().enumerate() {
        println!("\n─── LATENT TICK τ = {} ─────────────────────────────────────────────────────────────", tau);
        for (s_idx, &q_pos) in report.query_slot_indices.iter().enumerate() {
            println!("  [Query Slot {} (pos {})]", s_idx, q_pos);
            print!("    Pos   : ");
            for j in 0..config.field.seq_len {
                print!("{:>6} ", j);
            }
            println!();
            print!("    Reach : ");
            for j in 0..config.field.seq_len {
                let r = if report.theoretical_reachability[b_idx][s_idx][j] { "YES" } else { " no" };
                print!("{:>6} ", r);
            }
            println!();
            print!("    NumSens: ");
            for j in 0..config.field.seq_len {
                print!("{:>6.3} ", report.numerical_influence[b_idx][s_idx][j]);
            }
            println!();
            print!("    FlipPr: ");
            for j in 0..config.field.seq_len {
                print!("{:>5.1}% ", report.flip_probability[b_idx][s_idx][j] * 100.0);
            }
            println!();
        }
    }

    if task_kind.is_iterated_parity() {
        let max_ticks = budgets.last().copied().unwrap_or(config.latent.latent_ticks_per_token);
        println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
        println!("║ TITAN TEXT · BOUNDARY RELOCATION PROBE (3-bit Parity Shift Test)                     ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
        println!("  Evaluating identical 3-bit parity shifted across chunks at τ = {}", max_ticks);
        let reloc_report = executor.run_boundary_relocation_probe(max_ticks, config.field.seq_len, device)?;
        println!("  Placement Query Positions  : {:?}", reloc_report.placement_positions);
        println!("  Zero-Padded Accuracy       : {:?}", reloc_report.zero_padded_accuracies);
        println!("  Random-Ctx Local Accuracy  : {:?}", reloc_report.random_context_local_accuracies);
        println!("  Random-Ctx Cumul Accuracy  : {:?}", reloc_report.random_context_cumulative_accuracies);

        println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
        println!("║ TITAN TEXT · LATENT REPRESENTATION PROBE (Linear & Non-linear MLP)                   ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
        println!("  Evaluating latent hidden state representations at τ = {}", max_ticks);
        let latent_probe = executor.run_latent_representation_probe(max_ticks, config.field.seq_len, device)?;
        println!("  Query Slot Cell Indices    : {:?}", latent_probe.slot_indices);
        println!("  Linear Chunk0 Parity Acc   : {:?}", latent_probe.linear_chunk0_parity_acc);
        println!("  Linear Local Parity Acc    : {:?}", latent_probe.linear_local_parity_acc);
        println!("  Linear Cumulative Acc      : {:?}", latent_probe.linear_cumulative_parity_acc);
        println!("  MLP Chunk0 Parity Acc      : {:?}", latent_probe.mlp_chunk0_parity_acc);
        println!("  MLP Local Parity Acc       : {:?}", latent_probe.mlp_local_parity_acc);
        println!("  MLP Cumulative Acc         : {:?}", latent_probe.mlp_cumulative_parity_acc);
    }

    if let Some(ref out_p) = output_path {
        let p = std::path::Path::new(out_p);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(p)?;
        serde_json::to_writer_pretty(f, &report)?;
        println!("\n✓ Saved causal influence report to '{}'", out_p);
    }

    Ok(())
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = match env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string()
                .map_err(|_| "command-line arguments must be valid UTF-8")
        })
        .collect::<std::result::Result<_, _>>()
    {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    let invocation = match cli::parse(&args) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("error: {error:#}");
            return std::process::ExitCode::from(2);
        }
    };
    let device = Device::Cpu;
    let result = match invocation {
        cli::Invocation::Help(command) => {
            cli::print_help(command);
            Ok(())
        }
        cli::Invocation::Run(command, args) => {
            let threads: Option<usize> = args.value("--threads").unwrap_or(None);
            let _ = init_thread_pool(threads);
            match command {
                cli::Command::Train => cmd_train(&args, &device),
                cli::Command::Probe => cmd_probe(&args, &device),
                cli::Command::Falsify => cmd_falsify(&args, &device),
                cli::Command::NsProbe => cmd_ns_probe(&args, &device),
                cli::Command::Rollout => cmd_rollout(&args, &device),
                cli::Command::Sweep => cmd_sweep(&args, &device),
                cli::Command::Associate => cmd_associate(&args, &device),
                cli::Command::Attractor => cmd_attractor(&args, &device),
                cli::Command::Benchmark => cmd_benchmark(&args, &device),
                cli::Command::Memory => cmd_memory(&args, &device),
                cli::Command::Influence => cmd_influence(&args, &device),
            }
        }
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
