mod checkpoint;
mod config;
mod dataset;
mod experiment;
mod field;
mod nca;
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

fn print_help() {
    println!("TITAN TEXT v0.1.0 · Morphogenic Neural-Cellular Sequence Laboratory");
    println!("Lean experimental sequence morphogenesis exploring nonlinear cellular dynamics");
    println!();
    println!("USAGE:");
    println!("  titan_text <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("  train       Train neural-cellular sequence field with full diagnostics");
    println!("  probe       Run ±epsilon nonlinear perturbation probe & dynamical battery");
    println!("  ns-probe    Run Navier-Stokes Millennium C&D blowup probe (forcing & dissipation)");
    println!("  rollout     Evolve internal field autonomously with frozen weights");
    println!("  falsify     Run falsification hooks (context scrambling, zeroing, state reset)");
    println!("  help, -h    Show this help message");
    println!();
    println!("OPTIONS FOR 'train':
  --epochs <NUM>       Training iterations (default: 200)
  --dev-steps <NUM>    Developmental recurrence steps per iteration (default: 8)
  --task <NAME>        One of: 'text', 'dyck' (default: 'text')
  --save-dir <DIR>     Checkpoint directory (default: 'checkpoints/v0_text')
  --load-dir <DIR>     Resume training from existing checkpoint directory
  --seq-len <NUM>      Spatial sequence length (default: 32)
  --lr <FLOAT>         Learning rate (default: 0.003)
  --viscosity <FLOAT>  Navier-Stokes viscous dissipation nu (default: 0.0)
  --horizon <NUM>      Post-training rollout horizon steps (default: 64)
  --output <FILE>      Save post-training rollout patterns to text file
  --patterns-only      Format post-training pattern output as raw strings

OPTIONS FOR 'rollout':
  --load-dir <DIR>     Load trained model weights from checkpoint directory
  --task <NAME>        Task vocabulary ('text' or 'dyck', default: from checkpoint or 'text')
  --horizon <NUM>      Autonomous rollout horizon steps (default: 32)
  --seq-len <NUM>      Spatial sequence length (default: from checkpoint or 32)
  --viscosity <FLOAT>  Navier-Stokes dissipation nu (default: from checkpoint or 0.0)
  --prompt <TEXT>      Optional seed prompt string to initialize lattice
  --output <FILE>      Save step-by-step decoded patterns to text file
  --patterns-only      Output only raw pattern strings (one per line) for piping

OPTIONS FOR 'probe':
  --load-dir <DIR>     Load existing checkpoint directory
  --task <NAME>        Task vocabulary ('text' or 'dyck')
  --eps <FLOAT>        Perturbation epsilon (default: 0.01)
  --horizon <NUM>      Autonomous rollout horizon (default: 64)
  --output <FILE>      Save scientific JSON report to file

OPTIONS FOR 'ns-probe':
  --load-dir <DIR>     Load existing checkpoint directory
  --task <NAME>        Task vocabulary ('text' or 'dyck')
  --forcing-amp <FLOAT> Smooth forcing amplitude A (default: 0.2)
  --viscosity <FLOAT>  Base Navier-Stokes viscosity nu (default: 0.0)
  --horizon <NUM>      Rollout horizon steps (default: 64)
  --output <FILE>      Save scientific JSON report to file");
}

fn current_time_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn cmd_train(args: &[String], device: &Device) -> Result<()> {
    let mut load_dir: Option<String> = None;
    let mut save_dir_opt: Option<String> = None;
    let mut task_opt: Option<String> = None;
    let mut epochs_opt: Option<usize> = None;
    let mut dev_steps_opt: Option<usize> = None;
    let mut seq_len_opt: Option<usize> = None;
    let mut lr_opt: Option<f64> = None;
    let mut viscosity_opt: Option<f32> = None;
    let mut horizon_opt: Option<usize> = None;
    let mut patterns_only = false;
    let mut output_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--load-dir" => {
                if i + 1 < args.len() {
                    load_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--save-dir" => {
                if i + 1 < args.len() {
                    save_dir_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--task" => {
                if i + 1 < args.len() {
                    task_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    epochs_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--dev-steps" => {
                if i + 1 < args.len() {
                    dev_steps_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--seq-len" => {
                if i + 1 < args.len() {
                    seq_len_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    lr_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--viscosity" => {
                if i + 1 < args.len() {
                    let v: f32 = args[i + 1].parse()?;
                    if v < 0.0 {
                        anyhow::bail!("Viscosity nu must be non-negative, got {}", v);
                    }
                    viscosity_opt = Some(v);
                    i += 1;
                }
            }
            "--horizon" => {
                if i + 1 < args.len() {
                    horizon_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--patterns-only" | "--raw" => {
                patterns_only = true;
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let (mut config, start_step, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        let start = manifest.cumulative_step;
        let t = manifest.task.clone();
        (manifest.config, start, t)
    } else {
        (TitanConfig::default(), 0, "text".to_string())
    };

    let task = task_opt.unwrap_or(if default_task.is_empty() { "text".to_string() } else { default_task });
    let save_dir = save_dir_opt.unwrap_or_else(|| {
        if let Some(ref dir) = load_dir {
            dir.clone()
        } else {
            "checkpoints/v0_text".to_string()
        }
    });

    if let Some(e) = epochs_opt { config.train.epochs = e; }
    if let Some(d) = dev_steps_opt { config.train.dev_steps = d; }
    if let Some(s) = seq_len_opt { config.field.seq_len = s; }
    if let Some(lr) = lr_opt { config.train.lr = lr; }
    if let Some(v) = viscosity_opt { config.nca.viscosity = v; }
    if let Some(h) = horizon_opt { config.probe.horizon = h; }

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT v0 · RECURRENT CELLULAR TRAINING SESSION                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    if let Some(ref dir) = load_dir {
        println!("  Resuming From       : {} (cumulative step {})", dir, start_step);
    }
    println!("  Task                : {}", task);
    println!("  Epochs to Train     : {}", config.train.epochs);
    println!("  Dev Recurrence (T)  : {} steps", config.train.dev_steps);
    println!("  Lattice Length (L)  : {} cells", config.field.seq_len);
    println!("  Channels (C)        : {} continuous hidden dimensions", config.field.channels);
    println!("  Viscosity (nu)      : {:.4} (Navier-Stokes dissipation)", config.nca.viscosity);
    println!("  Save Directory      : {}", save_dir);

    let mut trainer = Trainer::new(config.clone(), &task, device)?;
    let total_params: usize = trainer.varmap.all_vars().iter().map(|v| v.elem_count()).sum();
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
    println!("✓ Training finished in {:.2}s ({:.1} epochs/sec)", elapsed, speed);
    println!("✓ Final checkpoint and manifest committed to '{}'", save_dir);

    if let Some(final_d) = last_diag {
        println!("\nFinal State Summary:");
        println!("  • Train Accuracy : {:.1}% (Loss: {:.4})", final_d.train_acc * 100.0, final_d.train_loss);
        println!("  • Val Accuracy   : {:.1}% (Loss: {:.4})", final_d.val_acc * 100.0, final_d.val_loss);
        println!("  • Hidden Mean    : {:.4} (Variance: {:.4})", final_d.hidden_mean, final_d.hidden_var);
        println!("  • State Energy   : {:.4}", final_d.state_energy);
        println!("  • Enstrophy (Ω)  : {:.4}", final_d.enstrophy);
        println!("  • BKM Norm (B)   : {:.4}", final_d.bkm_norm);
    }

    // Run post-training diagnostics
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ POST-TRAINING DYNAMICAL DIAGNOSTICS & PERTURBATION PROBE                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    
    let (sample_in, _) = trainer.dataset.sample_train_batch(1, config.field.seq_len, device)?;
    let seed_embed = trainer.interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);
    let settled_field = trainer.nca.develop(&sample_field, config.train.dev_steps, device)?;

    let probe_rep = ExperimentSuite::run_nonlinear_perturbation_probe(
        &trainer.nca,
        &settled_field,
        config.probe.perturbation_eps,
        config.probe.lp_window,
        device,
    )?;

    println!("\n[1] ±EPSILON NONLINEAR PERTURBATION PROBE (eps = {})", config.probe.perturbation_eps);
    println!("    Wavenumber | Wavelength | ||Q|| Norm | ||Δx|| Base | Relative Q/||Δx||");
    println!("    ───────────┼────────────┼────────────┼─────────────┼──────────────────");
    for w in &probe_rep.wavelengths {
        println!(
            "    k = {:<6} | {:<2} cells   | {:<10.5} | {:<11.5} | {:<8.3} ({})",
            w.wavenumber, w.wavelength_cells, w.q_norm, w.ordinary_update_norm, w.relative_response_ratio, w.frequency_label
        );
    }
    println!("    Verdict: {}", probe_rep.verdict);

    let traj_rep = ExperimentSuite::run_autonomous_rollout(
        &trainer.nca,
        &trainer.interface,
        &trainer.dataset.vocab,
        &settled_field,
        config.probe.horizon,
        device,
    )?;
    println!("\n[2] AUTONOMOUS FROZEN TRAJECTORY ({}-step rollout)", config.probe.horizon);
    println!("    • Trajectory Classification   : {}", traj_rep.classification);
    println!("    • Mean Step Drift Velocity    : {:.5} units/step", traj_rep.mean_step_velocity);
    println!("    • Min Recurrence Distance     : {:.5}", traj_rep.min_recurrence_distance);
    println!("    • Final Field Energy          : {:.4}", traj_rep.final_energy);
    println!("    • Final Output Token Entropy  : {:.4} nats", traj_rep.final_entropy);
    if let Some(first_t) = traj_rep.traces.first() {
        println!("    • Initial Seed Pattern        : \"{}\"", first_t.decoded_pattern);
    }
    if let Some(last_t) = traj_rep.traces.last() {
        println!("    • Final Emerged Pattern       : \"{}\"", last_t.decoded_pattern);
    }

    let falsify_rep = ExperimentSuite::run_falsification_battery(
        &trainer.nca,
        &trainer.interface,
        &trainer.dataset,
        device,
    )?;
    println!("\n[3] FALSIFICATION & CONTEXT SENSITIVITY BATTERY");
    println!("    • Baseline Target Accuracy    : {:5.1}%", falsify_rep.baseline_accuracy * 100.0);
    println!("    • Scrambled Context Accuracy  : {:5.1}%", falsify_rep.scrambled_context_accuracy * 100.0);
    println!("    • Zeroed Context Accuracy     : {:5.1}%", falsify_rep.zeroed_context_accuracy * 100.0);
    println!("    • Truncated Context Accuracy  : {:5.1}%", falsify_rep.truncated_context_accuracy * 100.0);
    println!("    • State-Reset Divergence      : {:.5}", falsify_rep.state_reset_divergence);
    println!("    • Contextual Memory Ratio     : {:5.1}%", falsify_rep.contextual_memory_ratio * 100.0);
    println!("    • Verdict                     : {}", falsify_rep.verdict);

    if let Some(ref path) = output_path {
        let mut out_text = String::new();
        if patterns_only {
            for t in &traj_rep.traces {
                out_text.push_str(&format!("{}\n", t.decoded_pattern));
            }
        } else {
            out_text.push_str("# TITAN TEXT · AUTONOMOUS ROLLOUT PATTERN TRACE\n");
            out_text.push_str(&format!("# Checkpoint: {}\n", save_dir));
            out_text.push_str(&format!("# Task: {} | Horizon: {} steps | Viscosity (nu): {:.4}\n", task, config.probe.horizon, config.nca.viscosity));
            out_text.push_str(&format!("# Classification: {}\n#\n", traj_rep.classification));
            for t in &traj_rep.traces {
                out_text.push_str(&format!("Step {:03}: {}\n", t.step, t.decoded_pattern));
            }
        }
        write_output_file(path, out_text)?;
        println!("\n✓ Post-training patterns saved to '{}'", path);
    }

    Ok(())
}

fn cmd_probe(args: &[String], device: &Device) -> Result<()> {
    let mut load_dir: Option<String> = None;
    let mut task_opt: Option<String> = None;
    let mut eps = 0.01f32;
    let mut horizon = 64usize;
    let mut output_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--load-dir" => {
                if i + 1 < args.len() {
                    load_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--task" => {
                if i + 1 < args.len() {
                    task_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--eps" => {
                if i + 1 < args.len() {
                    eps = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--horizon" => {
                if i + 1 < args.len() {
                    horizon = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let (config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = task_opt.unwrap_or(if default_task.is_empty() { "text".to_string() } else { default_task });

    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), dataset.vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading trained weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    } else {
        println!("Probing freshly initialized model (random weights baseline)...");
    }

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · DYNAMICAL BATTERY & NONLINEAR PERTURBATION REPORT                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");

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

    println!("\n[1] ±EPSILON NONLINEAR PERTURBATION PROBE (eps = {})", eps);
    println!("    Wavenumber | Wavelength | ||Q|| Norm | ||Δx|| Base | Relative Q/||Δx||");
    println!("    ───────────┼────────────┼────────────┼─────────────┼──────────────────");
    for w in &probe_rep.wavelengths {
        println!(
            "    k = {:<6} | {:<2} cells   | {:<10.5} | {:<11.5} | {:<8.3} ({})",
            w.wavenumber, w.wavelength_cells, w.q_norm, w.ordinary_update_norm, w.relative_response_ratio, w.frequency_label
        );
    }
    println!("    Verdict: {}", probe_rep.verdict);

    let traj_rep = ExperimentSuite::run_autonomous_rollout(
        &nca,
        &interface,
        &dataset.vocab,
        &settled_field,
        horizon,
        device,
    )?;
    println!("\n[2] AUTONOMOUS FROZEN TRAJECTORY ({}-step rollout)", horizon);
    println!("    • Trajectory Classification   : {}", traj_rep.classification);
    println!("    • Mean Step Drift Velocity    : {:.5} units/step", traj_rep.mean_step_velocity);
    println!("    • Min Recurrence Distance     : {:.5}", traj_rep.min_recurrence_distance);
    println!("    • Final Field Energy          : {:.4}", traj_rep.final_energy);
    println!("    • Final Output Token Entropy  : {:.4} nats", traj_rep.final_entropy);
    if let Some(first_t) = traj_rep.traces.first() {
        println!("    • Initial Seed Pattern        : \"{}\"", first_t.decoded_pattern);
    }
    if let Some(last_t) = traj_rep.traces.last() {
        println!("    • Final Emerged Pattern       : \"{}\"", last_t.decoded_pattern);
    }

    let falsify_rep = ExperimentSuite::run_falsification_battery(
        &nca,
        &interface,
        &dataset,
        device,
    )?;
    println!("\n[3] FALSIFICATION & CONTEXT SENSITIVITY BATTERY");
    println!("    • Baseline Target Accuracy    : {:5.1}%", falsify_rep.baseline_accuracy * 100.0);
    println!("    • Scrambled Context Accuracy  : {:5.1}%", falsify_rep.scrambled_context_accuracy * 100.0);
    println!("    • Zeroed Context Accuracy     : {:5.1}%", falsify_rep.zeroed_context_accuracy * 100.0);
    println!("    • Truncated Context Accuracy  : {:5.1}%", falsify_rep.truncated_context_accuracy * 100.0);
    println!("    • State-Reset Divergence      : {:.5}", falsify_rep.state_reset_divergence);
    println!("    • Contextual Memory Ratio     : {:5.1}%", falsify_rep.contextual_memory_ratio * 100.0);
    println!("    • Verdict                     : {}", falsify_rep.verdict);

    if let Some(path) = output_path {
        let full_report = experiment::FullDiagnosticReport {
            perturbation: probe_rep,
            trajectory: traj_rep,
            falsification: falsify_rep,
            navier_stokes: None,
        };
        let json_str = serde_json::to_string_pretty(&full_report)?;
        write_output_file(&path, json_str)?;
        println!("\n✓ Scientific report saved to '{}'", path);
    }

    Ok(())
}

fn cmd_ns_probe(args: &[String], device: &Device) -> Result<()> {
    let mut load_dir: Option<String> = None;
    let mut task_opt: Option<String> = None;
    let mut forcing_amp = 0.2f32;
    let mut viscosity = 0.0f32;
    let mut horizon = 64usize;
    let mut output_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--load-dir" => {
                if i + 1 < args.len() {
                    load_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--task" => {
                if i + 1 < args.len() {
                    task_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--forcing-amp" => {
                if i + 1 < args.len() {
                    let a: f32 = args[i + 1].parse()?;
                    if a < 0.0 {
                        anyhow::bail!("Forcing amplitude A must be non-negative, got {}", a);
                    }
                    forcing_amp = a;
                    i += 1;
                }
            }
            "--viscosity" => {
                if i + 1 < args.len() {
                    let v: f32 = args[i + 1].parse()?;
                    if v < 0.0 {
                        anyhow::bail!("Viscosity nu must be non-negative, got {}", v);
                    }
                    viscosity = v;
                    i += 1;
                }
            }
            "--horizon" => {
                if i + 1 < args.len() {
                    horizon = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        (manifest.config, manifest.task.clone())
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = task_opt.unwrap_or(if default_task.is_empty() { "text".to_string() } else { default_task });
    config.nca.viscosity = viscosity;

    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), dataset.vocab.size(), config.field.channels)?;

    if let Some(ref dir) = load_dir {
        println!("Loading trained weights from '{}'...", dir);
        CheckpointManager::load_weights(dir, &mut varmap, device)?;
    } else {
        println!("Probing freshly initialized model (random weights baseline)...");
    }

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · NAVIER-STOKES MILLENNIUM SINGULARITY & ENSTROPHY CASCADE PROBE          ║");
    println!("║ Theoretical Foundations: Beale-Kato-Majda Blow-up & Smooth External Forcing          ║");
    println!("║ (Context: OpenAI Sept 8, 2026 Finite-Time Singularity Proof for Cases C & D)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Horizon Steps (T)   : {}", horizon);
    println!("  Forcing Amplitude A : {:.3} (Smooth C^inf multi-mode periodic)", forcing_amp);
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
    println!("    Initial Enstrophy Ω(0) : {:.5}", ns_report.unforced_regime.initial_enstrophy);
    println!("    Max Enstrophy Ω_max    : {:.5}", ns_report.unforced_regime.max_enstrophy);
    println!("    Max BKM Norm B_max     : {:.5}", ns_report.unforced_regime.max_bkm_norm);
    println!("    Accumulated BKM ∫B dt  : {:.5}", ns_report.unforced_regime.accumulated_bkm);

    println!("\n[2] SMOOTH EXTERNAL FORCING FLOW (f ∈ C^inf, A = {:.3}, nu = 0.0)", forcing_amp);
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
    println!("    Blowup Detected        : {}", if ns_report.forced_regime.blowup_detected { "YES (Singularity/Breakdown Observed)" } else { "NO (Solution Remained Bounded)" });
    if let Some(s) = ns_report.forced_regime.singularity_step {
        println!("    Singularity Time T*    : Step {}", s);
    }
    println!("    Enstrophy Exponent γ   : {:.3} (dΩ/dt ~ Ω^γ; γ > 1 implies finite-time blowup)", ns_report.forced_regime.enstrophy_growth_exponent);
    println!("    BKM Criteria Met       : {}", if ns_report.bkm_blowup_criteria_met { "YES (Singularity Criteria Satisfied)" } else { "NO" });

    println!("\n[3] VISCOSITY REGULARIZATION SWEEP (nu * Δx Dissipation)");
    println!("    Viscosity (nu) | Final Energy | Final Enstrophy | Max BKM Norm | Max ||x||_inf | Regularized?");
    println!("    ───────────────┼──────────────┼─────────────────┼──────────────┼───────────────┼─────────────");
    for res in &ns_report.viscosity_sweep {
        println!(
            "    nu = {:<9.4} | {:<12.4} | {:<15.4} | {:<12.4} | {:<13.4} | {}",
            res.viscosity, res.final_energy, res.final_enstrophy, res.max_bkm, res.max_velocity,
            if res.regularized { "✓ YES (Stable)" } else { "✗ NO (Singular/Blowup)" }
        );
    }
    if let Some(nu_c) = ns_report.critical_viscosity_est {
        println!("    Critical Viscosity nu* : ~{:.4} (Minimum dissipation needed to arrest blowup)", nu_c);
    } else {
        println!("    Critical Viscosity nu* : > 0.2000 (Requires stronger dissipation)");
    }
    println!("\nVerdict: {}", ns_report.verdict);

    if let Some(path) = output_path {
        let json_str = serde_json::to_string_pretty(&ns_report)?;
        write_output_file(&path, json_str)?;
        println!("\n✓ Scientific report saved to '{}'", path);
    }

    Ok(())
}

fn cmd_rollout(args: &[String], device: &Device) -> Result<()> {
    let mut load_dir: Option<String> = None;
    let mut task_opt: Option<String> = None;
    let mut horizon = 32usize;
    let mut seq_len_opt: Option<usize> = None;
    let mut viscosity_opt: Option<f32> = None;
    let mut output_path: Option<String> = None;
    let mut prompt_opt: Option<String> = None;
    let mut patterns_only = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--load-dir" => {
                if i + 1 < args.len() {
                    load_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--task" => {
                if i + 1 < args.len() {
                    task_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--horizon" | "--steps" | "--dev-steps" => {
                if i + 1 < args.len() {
                    horizon = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--seq-len" => {
                if i + 1 < args.len() {
                    seq_len_opt = Some(args[i + 1].parse()?);
                    i += 1;
                }
            }
            "--viscosity" => {
                if i + 1 < args.len() {
                    let v: f32 = args[i + 1].parse()?;
                    if v < 0.0 {
                        anyhow::bail!("Viscosity nu must be non-negative, got {}", v);
                    }
                    viscosity_opt = Some(v);
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt_opt = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--patterns-only" | "--raw" => {
                patterns_only = true;
            }
            _ => {}
        }
        i += 1;
    }

    let (mut config, default_task) = if let Some(ref dir) = load_dir {
        let manifest = CheckpointManager::load_manifest(dir)?;
        let t = manifest.task.clone();
        (manifest.config, t)
    } else {
        (TitanConfig::default(), "text".to_string())
    };

    let task = task_opt.unwrap_or(if default_task.is_empty() { "text".to_string() } else { default_task });
    if let Some(s) = seq_len_opt {
        config.field.seq_len = s;
    }
    if let Some(v) = viscosity_opt {
        config.nca.viscosity = v;
    }

    let dataset = SequenceDataset::new(&task);
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), dataset.vocab.size(), config.field.channels)?;

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

    let traj = ExperimentSuite::run_autonomous_rollout(&nca, &interface, &dataset.vocab, &sample_field, horizon, device)?;

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
        println!("  Viscosity (nu)      : {:.4} (Navier-Stokes dissipation)", config.nca.viscosity);
        println!("  Lattice Length (L)  : {} cells\n", config.field.seq_len);

        println!("Step | State Norm | Energy | Entropy | Decoded Lattice Pattern");
        println!("─────┼────────────┼────────┼─────────┼──────────────────────────────────────────────────");
        for t in &traj.traces {
            println!(
                "{:03}  | {:<10.4} | {:<6.4} | {:.4}  | \"{}\"",
                t.step, t.state_norm, t.energy, t.output_entropy, t.decoded_pattern
            );
        }
        println!("───────────────────────────────────────────────────────────────────────────────────");
        println!("Classification : {}", traj.classification);
        println!("Drift Velocity : {:.5} units/step", traj.mean_step_velocity);
        println!("Final Energy   : {:.4}", traj.final_energy);
        println!("Final Entropy  : {:.4} nats", traj.final_entropy);
    }

    if let Some(ref out_file) = output_path {
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
            out_text.push_str(&format!("# Task: {} | Horizon: {} steps | Viscosity: {:.4}\n", task, horizon, config.nca.viscosity));
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

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    let device = Device::Cpu;

    match args[1].as_str() {
        "train" => cmd_train(&args[2..], &device)?,
        "probe" | "diagnostics" => cmd_probe(&args[2..], &device)?,
        "ns-probe" | "navier-stokes" | "blowup" | "fluid-probe" => cmd_ns_probe(&args[2..], &device)?,
        "rollout" => cmd_rollout(&args[2..], &device)?,
        "falsify" => cmd_probe(&args[2..], &device)?,
        "-h" | "--help" | "help" => print_help(),
        unknown => {
            eprintln!("Unknown command: '{}'. Use --help for usage.", unknown);
            std::process::exit(1);
        }
    }

    Ok(())
}
