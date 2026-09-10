mod checkpoint;
mod config;
mod dataset;
mod experiment;
mod field;
mod nca;
mod train;
mod vocab;

use anyhow::Result;
use candle_core::Device;
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
    println!("  rollout     Evolve internal field autonomously with frozen weights");
    println!("  falsify     Run falsification hooks (context scrambling, zeroing, state reset)");
    println!("  help, -h    Show this help message");
    println!();
    println!("OPTIONS FOR 'train':");
    println!("  --epochs <NUM>       Training iterations (default: 200)");
    println!("  --dev-steps <NUM>    Developmental recurrence steps per iteration (default: 8)");
    println!("  --task <NAME>        One of: 'text', 'dyck' (default: 'text')");
    println!("  --save-dir <DIR>     Checkpoint directory (default: 'checkpoints/v0_text')");
    println!("  --seq-len <NUM>      Spatial sequence length (default: 32)");
    println!("  --lr <FLOAT>         Learning rate (default: 0.003)");
    println!();
    println!("OPTIONS FOR 'probe':");
    println!("  --load-dir <DIR>     Load existing checkpoint directory");
    println!("  --task <NAME>        Task vocabulary ('text' or 'dyck')");
    println!("  --eps <FLOAT>        Perturbation epsilon (default: 0.01)");
    println!("  --horizon <NUM>      Autonomous rollout horizon (default: 64)");
    println!("  --output <FILE>      Save scientific JSON report to file");
}

fn current_time_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn cmd_train(args: &[String], device: &Device) -> Result<()> {
    let mut config = TitanConfig::default();
    let mut task = "text".to_string();
    let mut save_dir = "checkpoints/v0_text".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" => {
                if i + 1 < args.len() {
                    config.train.epochs = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--dev-steps" => {
                if i + 1 < args.len() {
                    config.train.dev_steps = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--task" => {
                if i + 1 < args.len() {
                    task = args[i + 1].clone();
                    i += 1;
                }
            }
            "--save-dir" => {
                if i + 1 < args.len() {
                    save_dir = args[i + 1].clone();
                    i += 1;
                }
            }
            "--seq-len" => {
                if i + 1 < args.len() {
                    config.field.seq_len = args[i + 1].parse()?;
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    config.train.lr = args[i + 1].parse()?;
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT v0 · RECURRENT CELLULAR TRAINING SESSION                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Task                : {}", task);
    println!("  Epochs              : {}", config.train.epochs);
    println!("  Dev Recurrence (T)  : {} steps", config.train.dev_steps);
    println!("  Lattice Length (L)  : {} cells", config.field.seq_len);
    println!("  Channels (C)        : {} continuous hidden dimensions", config.field.channels);
    println!("  Save Directory      : {}", save_dir);

    let mut trainer = Trainer::new(config.clone(), &task, device)?;
    let total_params: usize = trainer.varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("  Total Parameters    : {} weights\n", total_params);

    let start_time = Instant::now();
    let mut last_diag = None;

    println!("─── TRAINING & DYNAMICAL INSTRUMENTATION TRACE ─────────────────────────────────────────");
    println!("Epoch | Train Loss (Acc) | Val Loss (Acc)   | Grad Norm | ||Δx||  | Energy | Freq (L/M/H %)");
    println!("──────┼──────────────────┼──────────────────┼───────────┼─────────┼────────┼───────────────");

    for epoch in 1..=config.train.epochs {
        let diag = trainer.train_step(config.train.batch_size)?;
        let is_log_step = epoch % config.train.log_every == 0 || epoch == config.train.epochs;

        if is_log_step {
            println!(
                "{:04}  | {:.4} ({:5.1}%) | {:.4} ({:5.1}%) | {:<9.5} | {:<7.5} | {:<6.4} | {:3.0}/{:3.0}/{:3.0}%",
                epoch,
                diag.train_loss,
                diag.train_acc * 100.0,
                diag.val_loss,
                diag.val_acc * 100.0,
                diag.grad_norm,
                diag.update_magnitude,
                diag.state_energy,
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
                cumulative_step: epoch,
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
            };
            CheckpointManager::save(&save_dir, &mut manifest, &trainer.varmap)?;
        }

        last_diag = Some(diag);
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let speed = config.train.epochs as f32 / elapsed.max(1e-3);
    println!("────────────────────────────────────────────────────────────────────────────────────────");
    println!("✓ Training finished in {:.2}s ({:.1} epochs/sec)", elapsed, speed);
    println!("✓ Final checkpoint and manifest committed to '{}'", save_dir);

    if let Some(final_d) = last_diag {
        println!("\nFinal State Summary:");
        println!("  • Train Accuracy : {:.1}% (Loss: {:.4})", final_d.train_acc * 100.0, final_d.train_loss);
        println!("  • Val Accuracy   : {:.1}% (Loss: {:.4})", final_d.val_acc * 100.0, final_d.val_loss);
        println!("  • Hidden Mean    : {:.4} (Variance: {:.4})", final_d.hidden_mean, final_d.hidden_var);
        println!("  • Final Energy   : {:.4}", final_d.state_energy);
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

    Ok(())
}

fn cmd_probe(args: &[String], device: &Device) -> Result<()> {
    let mut load_dir: Option<String> = None;
    let mut task = "text".to_string();
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
                    task = args[i + 1].clone();
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

    let config = if let Some(ref dir) = load_dir {
        CheckpointManager::load_manifest(dir)?.config
    } else {
        TitanConfig::default()
    };

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
        };
        let json_str = serde_json::to_string_pretty(&full_report)?;
        std::fs::write(&path, json_str)?;
        println!("\n✓ Scientific report saved to '{}'", path);
    }

    Ok(())
}

fn cmd_rollout(args: &[String], device: &Device) -> Result<()> {
    let mut horizon = 32usize;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--horizon" && i + 1 < args.len() {
            horizon = args[i + 1].parse()?;
            i += 1;
        }
        i += 1;
    }

    let config = TitanConfig::default();
    let dataset = SequenceDataset::new("text");
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

    let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
    let interface = TokenInterface::new(vb.pp("interface"), dataset.vocab.size(), config.field.channels)?;

    let (sample_in, _) = dataset.sample_train_batch(1, config.field.seq_len, device)?;
    let seed_embed = interface.embed_tokens(&sample_in)?;
    let sample_field = MorphogenicField::from_tensor(seed_embed, &config.field);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TITAN TEXT · STEP-BY-STEP AUTONOMOUS ROLLOUT TRACE (Frozen Model)                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("Step | State Norm | ||Δx||  | Energy | Low % | Mid % | High % | Output Entropy");
    println!("─────┼────────────┼─────────┼────────┼───────┼───────┼────────┼───────────────");

    let traj = ExperimentSuite::run_autonomous_rollout(&nca, &interface, &sample_field, horizon, device)?;
    for t in &traj.traces {
        println!(
            "{:03}  | {:<10.4} | {:<7.4} | {:<6.4} | {:5.1}% | {:5.1}% | {:6.1}% | {:.4} nats",
            t.step, t.state_norm, t.update_magnitude, t.energy, t.pct_low, t.pct_mid, t.pct_high, t.output_entropy
        );
    }
    println!("───────────────────────────────────────────────────────────────────────────────────────");
    println!("Classification: {}", traj.classification);

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
