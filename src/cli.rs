use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Train,
    Probe,
    NsProbe,
    Rollout,
    Falsify,
    Sweep,
    Associate,
    Attractor,
    Benchmark,
    Memory,
}

impl Command {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "train" => Ok(Self::Train),
            "probe" | "diagnostics" => Ok(Self::Probe),
            "ns-probe" | "navier-stokes" | "blowup" | "fluid-probe" => Ok(Self::NsProbe),
            "rollout" => Ok(Self::Rollout),
            "falsify" => Ok(Self::Falsify),
            "sweep" | "latent-sweep" | "budget-sweep" => Ok(Self::Sweep),
            "associate" | "association-probe" => Ok(Self::Associate),
            "attractor" | "attractor-probe" | "basin" => Ok(Self::Attractor),
            "benchmark" | "compare" | "baselines" => Ok(Self::Benchmark),
            "memory" | "memory-dynamics" => Ok(Self::Memory),
            _ => bail!("unknown command '{name}'; use --help for usage"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Probe => "probe",
            Self::NsProbe => "ns-probe",
            Self::Rollout => "rollout",
            Self::Falsify => "falsify",
            Self::Sweep => "sweep",
            Self::Associate => "associate",
            Self::Attractor => "attractor",
            Self::Benchmark => "benchmark",
            Self::Memory => "memory",
        }
    }

    fn options(self) -> &'static [&'static str] {
        match self {
            Self::Train => &[
                "--load-dir",
                "--save-dir",
                "--task",
                "--epochs",
                "--dev-steps",
                "--seq-len",
                "--lr",
                "--viscosity",
                "--horizon",
                "--output",
                "--patterns-only",
                "--threads",
                "--horizon-mode",
                "--horizon-min",
                "--horizon-max",
                "--horizon-jitter",
                "--stability-tail",
                "--multi-tick-decay",
                "--seed",
                "--feedback-mode",
                "--feedback-weight",
                "--tail-eq-weight",
                "--tail-eq-ticks",
                "--state-norm",
                "--trace-output",
                "--record-activations",
            ],
            Self::Rollout => &[
                "--load-dir",
                "--task",
                "--horizon",
                "--seq-len",
                "--viscosity",
                "--prompt",
                "--output",
                "--patterns-only",
                "--latent-ticks",
                "--pause-ticks",
                "--slow-cadence",
                "--slow-fraction",
                "--lesion-state",
                "--lesion-gates",
                "--lesion-residual",
                "--lesion-gain",
                "--lesion-noise",
                "--lesion-freeze",
                "--lesion-reset",
                "--lesion-channels",
                "--threads",
                "--trace-output",
                "--record-activations",
            ],
            Self::Probe => &[
                "--load-dir",
                "--task",
                "--eps",
                "--horizon",
                "--output",
                "--threads",
                "--trace-output",
                "--record-activations",
            ],
            Self::Falsify => &[
                "--epochs",
                "--steps-per-epoch",
                "--batch-size",
                "--seq-len",
                "--lr",
                "--seeds",
                "--load-dir",
                "--task",
                "--eps",
                "--horizon",
                "--output",
                "--threads",
                "--trace-output",
                "--record-activations",
            ],
            Self::NsProbe => &[
                "--load-dir",
                "--task",
                "--forcing-amp",
                "--viscosity",
                "--horizon",
                "--output",
                "--threads",
                "--trace-output",
                "--record-activations",
            ],
            Self::Sweep => &[
                "--load-dir",
                "--task",
                "--budgets",
                "--seq-len",
                "--batch-size",
                "--output",
                "--model",
                "--seeds",
                "--threads",
                "--lesion-state",
                "--lesion-gates",
                "--lesion-residual",
                "--lesion-gain",
                "--lesion-noise",
                "--lesion-freeze",
                "--lesion-reset",
                "--lesion-channels",
                "--trace-output",
                "--record-activations",
            ],
            Self::Associate => &[
                "--load-dir",
                "--task",
                "--stimulus",
                "--target",
                "--horizon",
                "--output",
                "--slow-cadence",
                "--slow-fraction",
                "--threads",
            ],
            Self::Attractor => &[
                "--load-dir",
                "--task",
                "--prompt-a",
                "--prompt-b",
                "--horizon",
                "--output",
                "--slow-cadence",
                "--slow-fraction",
                "--threads",
            ],
            Self::Benchmark => &[
                "--task",
                "--seq-len",
                "--batch-size",
                "--epochs",
                "--lr",
                "--seeds",
                "--threads",
                "--train",
                "--checkpoint",
                "--load-dir",
                "--output",
                "--feedback-mode",
                "--feedback-weight",
                "--tail-eq-weight",
                "--tail-eq-ticks",
                "--state-norm",
            ],
            Self::Memory => &[
                "--load-dir",
                "--task",
                "--event-1",
                "--event-2",
                "--ticks",
                "--output",
                "--threads",
            ],
        }
    }
}

#[derive(Debug, Default)]
pub struct Options(BTreeMap<String, String>);

impl Options {
    pub fn value<T: FromStr>(&self, name: &str) -> Result<Option<T>> {
        self.0
            .get(name)
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid value '{value}' for {name}"))
            })
            .transpose()
    }

    pub fn flag(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

#[derive(Debug)]
pub enum Invocation {
    Help(Option<Command>),
    Run(Command, Options),
}

pub fn canonical_task(task: &str) -> Result<&'static str> {
    match task {
        "text" => Ok("text"),
        "dyck" | "paren" => Ok("dyck"),
        "delayed-recall" | "delayed-copy" | "delay" => Ok("delayed-recall"),
        "bracket" | "bracket-depth" => Ok("bracket-depth"),
        "parity" | "counting" => Ok("parity"),
        "reverse" | "reversal" => Ok("reverse"),
        "hidden-rule" | "rule" => Ok("hidden-rule"),
        "associative" | "assoc" => Ok("associative"),
        "ambiguous" | "ambiguous-basin" | "attractor" | "basin" => Ok("ambiguous-basin"),
        "column-arithmetic" | "arithmetic" | "addition" | "carry" => Ok("column-arithmetic"),
        "iterated-parity" | "ippr" => Ok("iterated-parity"),
        _ => bail!("invalid --task '{task}'; expected text or dyck (alias: paren)"),
    }
}

pub fn resolve_task(requested: Option<String>, saved: &str, loading: bool) -> Result<String> {
    let has_saved_task = !saved.is_empty();
    let saved = canonical_task(if saved.is_empty() { "text" } else { saved })?;
    let task = requested
        .as_deref()
        .map(canonical_task)
        .transpose()?
        .unwrap_or(saved);
    ensure!(
        !loading || !has_saved_task || task == saved,
        "--task '{task}' conflicts with checkpoint task '{saved}'"
    );
    Ok(task.to_string())
}

fn is_flag_option(name: &str) -> bool {
    matches!(
        name,
        "--patterns-only"
            | "--lesion-state"
            | "--lesion-gates"
            | "--lesion-residual"
            | "--train"
            | "--record-activations"
    )
}

fn validate_value(name: &str, value: &str) -> Result<()> {
    let invalid = || format!("invalid value '{value}' for {name}");
    match name {
        "--epochs" | "--dev-steps" | "--seq-len" | "--horizon" | "--latent-ticks" | "--slow-cadence" | "--ticks" | "--batch-size" | "--threads" | "--horizon-min" | "--horizon-max" | "--stability-tail" => {
            let number: usize = value.parse().with_context(invalid)?;
            ensure!(number > 0, "{name} must be a positive integer");
            if name == "--seq-len" {
                ensure!(
                    number <= i32::MAX as usize,
                    "--seq-len exceeds supported spatial indexing range"
                );
            }
        }
        "--pause-ticks" | "--lesion-freeze" | "--lesion-reset" | "--horizon-jitter" => {
            let _number: usize = value.parse().with_context(invalid)?;
        }
        "--seed" => {
            let _number: u64 = value.parse().with_context(invalid)?;
        }
        "--lr" | "--lesion-gain" => {
            let number: f64 = value.parse().with_context(invalid)?;
            ensure!(
                number.is_finite() && number > 0.0,
                "{name} must be finite and greater than zero"
            );
        }
        "--eps" | "--viscosity" | "--forcing-amp" | "--slow-fraction" | "--lesion-noise" => {
            let number: f32 = value.parse().with_context(invalid)?;
            if name == "--eps" {
                ensure!(
                    number.is_finite() && number > 0.0,
                    "{name} must be finite and greater than zero"
                );
                let denominator = 2.0 * number * number;
                ensure!(
                    denominator.is_finite() && denominator > 0.0,
                    "--eps is outside the supported f32 squared range"
                );
            } else if name == "--slow-fraction" {
                ensure!(
                    number.is_finite() && (0.0..=1.0).contains(&number),
                    "{name} must be in [0, 1]"
                );
            } else {
                ensure!(
                    number.is_finite() && number >= 0.0,
                    "{name} must be finite and non-negative"
                );
            }
        }
        "--multi-tick-decay" => {
            let number: f32 = value.parse().with_context(invalid)?;
            ensure!(
                number.is_finite() && number > 0.0 && number <= 1.0,
                "{name} must be finite and in (0, 1]"
            );
        }
        "--horizon-mode" => {
            ensure!(
                matches!(value, "fixed" | "randomized" | "multi_tick" | "jitter" | "stability_tail" | "early_exit"),
                "invalid --horizon-mode '{value}'; expected fixed, randomized, multi_tick, jitter, stability_tail, or early_exit"
            );
        }
        "--task" => {
            canonical_task(value)?;
        }
        "--prompt" | "--prompt-a" | "--prompt-b" | "--stimulus" | "--target" | "--event-1" | "--event-2" | "--budgets" | "--lesion-channels" | "--model" | "--seeds" => {}
        _ => ensure!(!value.is_empty(), "{name} must not be empty"),
    }
    Ok(())
}

/// Parse the entire invocation before accessing checkpoints or starting computation.
pub fn parse(args: &[String]) -> Result<Invocation> {
    let Some(first) = args.first() else {
        return Ok(Invocation::Help(None));
    };
    if matches!(first.as_str(), "--help" | "-h" | "help") {
        return match args.get(1) {
            None => Ok(Invocation::Help(None)),
            Some(name) if first == "help" && args.len() == 2 => {
                Ok(Invocation::Help(Some(Command::parse(name)?)))
            }
            _ => bail!("unexpected arguments after '{first}'; use help <COMMAND>"),
        };
    }
    let command = Command::parse(first)?;
    let mut options = Options::default();
    let mut help = false;
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        let (name, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| (name, Some(value)));
        let name = match name {
            "--raw" => "--patterns-only",
            "--steps" | "--dev-steps" if command == Command::Rollout => "--horizon",
            "-h" => "--help",
            name => name,
        };
        if name == "--help" {
            ensure!(inline.is_none(), "--help does not take a value");
            ensure!(!help, "duplicate option --help");
            help = true;
            i += 1;
            continue;
        }
        ensure!(
            command.options().contains(&name),
            "unexpected argument '{arg}' for {}; use {} --help",
            command.name(),
            command.name()
        );
        ensure!(!options.flag(name), "duplicate option {name}");
        let value = if is_flag_option(name) {
            ensure!(inline.is_none(), "{name} does not take a value");
            String::new()
        } else {
            let value = if let Some(value) = inline {
                value
            } else {
                i += 1;
                let value = args
                    .get(i)
                    .with_context(|| format!("missing value for {name}"))?;
                ensure!(!value.starts_with('-') || value.parse::<f64>().is_ok(),
                    "missing value for {name} before '{value}'; use {name}=<VALUE> for a value starting with '-'");
                value
            };
            validate_value(name, value)?;
            value.to_string()
        };
        options.0.insert(name.to_string(), value);
        i += 1;
    }
    if help {
        return Ok(Invocation::Help(Some(command)));
    }
    ensure!(
        command != Command::Train || !options.flag("--patterns-only") || options.flag("--output"),
        "train --patterns-only requires --output <FILE>"
    );
    Ok(Invocation::Run(command, options))
}

pub fn print_help(command: Option<Command>) {
    println!(
        "TITAN TEXT {} · Recurrent Neural-Cellular Sequence Laboratory\n",
        env!("CARGO_PKG_VERSION")
    );
    let Some(command) = command else {
        println!("USAGE:\n  titan_text <COMMAND> [OPTIONS]\n  titan_text help <COMMAND>\n\nCOMMANDS:\n  train       Train or continue training from saved weights\n  probe       Run perturbation, trajectory, and falsification diagnostics\n  ns-probe    Run forcing and viscosity diagnostics\n  rollout     Evolve a field with frozen weights\n  falsify     Run the full diagnostic battery (same as probe)\n  sweep       Run latent-budget sweeps across tick allocations (0, 1, 2, 4, 8, 16...)\n  associate   Probe concept correlation emergence through latent recurrence time\n  attractor   Test trajectory separation across competing initial states and hysteresis\n  benchmark   Report parameter counts and compare NCA vs Transformer / GRU / RNN\n  memory      Test memory as dynamics (earlier events persistently modifying state)\n  help        Show global or command-specific help\n\nOPTIONS:\n  -h, --help  Show help\n\nALIASES:\n  diagnostics = probe\n  navier-stokes, blowup, fluid-probe = ns-probe\n  latent-sweep, budget-sweep = sweep\n  association-probe = associate\n  attractor-probe, basin = attractor\n  compare, baselines = benchmark\n  memory-dynamics = memory\n\nUse titan_text <COMMAND> --help for options and defaults.");
        return;
    };
    println!("USAGE:\n  titan_text {} [OPTIONS]\n\nOPTIONS:\n  -h, --help          Show this command's help", command.name());
    for name in command.options() {
        let description = match *name {
            "--load-dir" => "<DIR>    Load checkpoint config, task, and model weights",
            "--save-dir" => "<DIR>    Save checkpoint (default: load-dir or checkpoints/v0_text)",
            "--task" => "<NAME>       Task (text, dyck, delayed-recall, bracket, parity, reverse, hidden-rule, associative, ambiguous, column-arithmetic)",
            "--epochs" => "<NUM>      Additional training iterations (default: checkpoint or 200)",
            "--steps-per-epoch" => "<NUM> Steps per training iteration (default: 8)",
            "--dev-steps" => "<NUM>   Recurrence steps per iteration (default: checkpoint or 8)",
            "--seq-len" => "<NUM>     Spatial sequence length (default: checkpoint or 32)",
            "--lr" => "<FLOAT>        Learning rate (default: checkpoint or 0.003)",
            "--eps" => "<FLOAT>       Perturbation epsilon (default: 0.01)",
            "--forcing-amp" => "<FLOAT> Smooth forcing amplitude (default: 0.2)",
            "--viscosity" if command == Command::NsProbe => "<FLOAT> Base viscosity (default: 0.0)",
            "--viscosity" => "<FLOAT>  Viscous dissipation (default: checkpoint or 0.0)",
            "--horizon" if command == Command::Train => {
                "<NUM>     Post-training rollout steps (default: checkpoint or 64)"
            }
            "--horizon" if command == Command::Rollout => {
                "<NUM>     Rollout steps (default: 32; aliases: --steps, --dev-steps)"
            }
            "--horizon" => "<NUM>     Rollout steps (default: 64)",
            "--prompt" => {
                "<TEXT>     Seed prompt, padded/truncated to seq-len (default: task sample)"
            }
            "--prompt-a" => "<TEXT>   First prompt for competing basin trajectory probe",
            "--prompt-b" => "<TEXT>   Second prompt for competing basin trajectory probe",
            "--stimulus" => "<CHAR>   Stimulus concept character for association probe",
            "--target" => "<CHAR>     Target concept character for association probe",
            "--event-1" => "<TEXT>    Prior event stimulus for memory-as-dynamics test",
            "--event-2" => "<TEXT>    Probe event stimulus for memory-as-dynamics test",
            "--ticks" => "<NUM>       Latent ticks between events (default: 8)",
            "--budgets" => "<LIST>    Comma-separated latent budgets (default: 0,1,2,4,8,16,32)",
            "--batch-size" => "<NUM>  Evaluation batch size (default: 8)",
            "--latent-ticks" => "<NUM> Internal recurrence ticks per token (default: 1)",
            "--pause-ticks" => "<NUM> Post-input deliberation ticks before readout (default: 0)",
            "--slow-cadence" => "<NUM> Cadence for delayed slow path (default: 1)",
            "--slow-fraction" => "<FLOAT> Fraction of channels in slow path (default: 0.0)",
            "--model" => "<NAME>      Model architecture (nca, transformer, gru, rnn, simple-recurrent)",
            "--threads" => "<NUM>      Number of worker threads (default: hardware concurrency or RAYON_NUM_THREADS)",
            "--seeds" => "<LIST>    Evaluation seeds (default: 0; e.g. 1,2,3 for multi-seed sweep)",
            "--lesion-state" => "      Lesion: zero out recurrent state updates",
            "--lesion-gates" => "      Lesion: force update gates open (1.0)",
            "--lesion-residual" => "   Lesion: bypass residual connection",
            "--lesion-gain" => "<FLOAT> Scale update delta gain factor",
            "--lesion-noise" => "<FLOAT> Inject additive Gaussian noise to state",
            "--lesion-freeze" => "<NUM> Freeze state updates at and after step K",
            "--lesion-reset" => "<NUM> Reset state to zero at step K",
            "--lesion-channels" => "<LIST> Comma-separated channel indices to ablate",
            "--output" if command == Command::Train || command == Command::Rollout => {
                "<FILE>     Write decoded patterns; create parent directories as needed"
            }
            "--output" => "<FILE>     Write report JSON; create parent directories as needed",
            "--patterns-only" if command == Command::Train => {
                "       Raw pattern file; requires --output (alias: --raw)"
            }
            "--patterns-only" => "       Raw patterns on stdout and in output file (alias: --raw)",
            "--horizon-mode" => "<NAME>   Horizon training regime (fixed, randomized, multi_tick, jitter, stability_tail, early_exit)",
            "--horizon-min" => "<NUM>    Minimum recurrence steps for randomized horizon training (default: 2)",
            "--horizon-max" => "<NUM>    Maximum recurrence steps for randomized horizon training (default: 16)",
            "--horizon-jitter" => "<NUM> Max jitter steps added to dev-steps (default: 2)",
            "--stability-tail" => "<NUM> Steps to supervise after first correct step (default: 4)",
            "--multi-tick-decay" => "<FLOAT> Exponential decay for multi-tick supervision (default: 1.0)",
            "--seed" => "<NUM>          Random seed for training/evaluation (default: 42)",
            "--train" => "              Train baseline sequence models and compare convergence",
            "--feedback-mode" => "<NAME>   Macro recursive feedback mode (none, global_pool, dual_timescale; default: none)",
            "--feedback-weight" => "<FLOAT> Feedback coupling rate beta for dual timescale integration (default: 0.2)",
            "--tail-eq-weight" => "<FLOAT>  Tail equilibrium contraction loss weight lambda_eq (default: 0.0)",
            "--tail-eq-ticks" => "<NUM>    Number of tail ticks to supervise for equilibrium (default: 2)",
            "--state-norm" => "<NAME>      State normalization on intermediate fields (none, rms, bounded, layer_norm; default: none)",
            "--trace-output" => "<FILE>    Write detailed per-step activation traces to JSON file",
            "--record-activations" => "    Record full spatial activation matrices across rollout steps",
            _ => unreachable!(),
        };
        println!("  {name} {description}");
    }
    println!("\nCounts must be positive integers. Floats must be finite; lr and eps > 0,\nviscosity and forcing-amp >= 0. Unknown, duplicate, and misplaced arguments\nare errors. Both --option value and --option=value are accepted.");
    if command == Command::Train {
        println!("\nLoading continues weights and cumulative steps with a fresh AdamW optimizer.\nAn explicit task must match the checkpoint vocabulary.");
    }
}
