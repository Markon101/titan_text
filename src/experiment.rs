use crate::field::MorphogenicField;
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::TokenInterface;
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerturbationWavelengthResult {
    pub frequency_label: String,
    pub wavenumber: usize,
    pub wavelength_cells: usize,
    pub q_norm: f32,
    pub ordinary_update_norm: f32,
    pub relative_response_ratio: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NonlinearPerturbationReport {
    pub epsilon: f32,
    pub lp_window_cells: usize,
    pub wavelengths: Vec<PerturbationWavelengthResult>,
    pub verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryStepTrace {
    pub step: usize,
    pub state_norm: f32,
    pub update_magnitude: f32,
    pub energy: f32,
    pub pct_low: f32,
    pub pct_mid: f32,
    pub pct_high: f32,
    pub output_entropy: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousTrajectoryReport {
    pub horizon: usize,
    pub classification: String,
    pub min_recurrence_distance: f32,
    pub detected_cycle_period: Option<usize>,
    pub mean_step_velocity: f32,
    pub final_energy: f32,
    pub final_entropy: f32,
    pub traces: Vec<TrajectoryStepTrace>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FalsificationReport {
    pub baseline_accuracy: f32,
    pub scrambled_context_accuracy: f32,
    pub zeroed_context_accuracy: f32,
    pub truncated_context_accuracy: f32,
    pub state_reset_divergence: f32,
    pub contextual_memory_ratio: f32,
    pub verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullDiagnosticReport {
    pub perturbation: NonlinearPerturbationReport,
    pub trajectory: AutonomousTrajectoryReport,
    pub falsification: FalsificationReport,
}

pub struct ExperimentSuite;

impl ExperimentSuite {
    /// Low-pass spatial filter via moving average window
    fn low_pass_filter(tensor: &Tensor, window: usize) -> Result<Tensor> {
        let (b, l, c) = tensor.dims3()?;
        let mut data = tensor.to_vec3::<f32>()?;
        let half_w = (window / 2).max(1);

        for batch_idx in 0..b {
            for ch in 0..c {
                let original_row: Vec<f32> = (0..l).map(|i| data[batch_idx][i][ch]).collect();
                for i in 0..l {
                    let mut sum = 0.0f32;
                    let count = (2 * half_w) as f32;
                    for offset in -(half_w as i32)..(half_w as i32) {
                        let idx = ((i as i32 + offset) % l as i32 + l as i32) as usize % l;
                        sum += original_row[idx];
                    }
                    data[batch_idx][i][ch] = sum / count;
                }
            }
        }

        Tensor::new(data, tensor.device()).map_err(|e| e.into())
    }

    /// Generates a normalized zero-mean spatial perturbation tensor [1, seq_len, channels]
    fn make_zero_mean_perturbation(seq_len: usize, channels: usize, wavenumber: usize, device: &Device) -> Result<Tensor> {
        let mut data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;

        for i in 0..seq_len {
            let val = if wavenumber == seq_len / 2 {
                // High-frequency alternating (+1, -1, +1, -1)
                if i % 2 == 0 { 1.0f32 } else { -1.0f32 }
            } else {
                // Sinusoidal spatial mode
                (two_pi * (wavenumber as f32) * (i as f32) / (seq_len as f32)).sin()
            };
            for ch in 0..channels {
                data[0][i][ch] = val;
            }
        }

        Tensor::new(data, device).map_err(|e| e.into())
    }

    /// Evaluates the second-order nonlinear response probe:
    /// Q = LP( G(x + eps*p) + G(x - eps*p) - 2*G(x) ) / (2 * eps^2)
    pub fn run_nonlinear_perturbation_probe(
        nca: &NeuralCellularAutomaton,
        field: &MorphogenicField,
        eps: f32,
        lp_window: usize,
        device: &Device,
    ) -> Result<NonlinearPerturbationReport> {
        let seq_len = field.config.seq_len;
        let channels = field.config.channels;

        // Base autonomous rollout G(x)
        let (g_0_field, ordinary_update_norm) = nca.step(field, device)?;
        let g_0 = &g_0_field.x;

        // Test frequencies: High (k = L/2), Mid (k = L/4), Low (k = L/8)
        let test_cases = vec![
            ("High-Frequency (Nyquist alternating)", seq_len / 2, 2),
            ("Mid-Frequency (Mesoscale wave)", (seq_len / 4).max(1), 4),
            ("Low-Frequency (Macro wave)", (seq_len / 8).max(1), 8),
        ];

        let mut results = Vec::new();
        let two_eps_sq = 2.0 * eps * eps;

        for (label, k, wavelength) in test_cases {
            let p = Self::make_zero_mean_perturbation(seq_len, channels, k, device)?;

            // x + eps*p and x - eps*p
            let field_plus = field.inject_perturbation(&p, eps)?;
            let field_minus = field.inject_perturbation(&p, -eps)?;

            let (g_plus_field, _) = nca.step(&field_plus, device)?;
            let (g_minus_field, _) = nca.step(&field_minus, device)?;

            let g_plus = &g_plus_field.x;
            let g_minus = &g_minus_field.x;

            // Second difference: G(x + eps*p) + G(x - eps*p) - 2 * G(x)
            let diff_sum = (g_plus + g_minus)?;
            let two_g0 = (g_0 * 2.0)?;
            let second_diff = (&diff_sum - &two_g0)?;

            // Apply Low-Pass projection LP
            let lp_diff = Self::low_pass_filter(&second_diff, lp_window)?;

            // Scale by 1 / (2 * eps^2)
            let q_tensor = (&lp_diff / (two_eps_sq as f64))?;
            let q_norm = q_tensor.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

            let relative_ratio = q_norm / (ordinary_update_norm + 1e-6);

            results.push(PerturbationWavelengthResult {
                frequency_label: label.to_string(),
                wavenumber: k,
                wavelength_cells: wavelength,
                q_norm,
                ordinary_update_norm,
                relative_response_ratio: relative_ratio,
            });
        }

        let max_ratio = results.iter().map(|r| r.relative_response_ratio).fold(0.0f32, f32::max);
        let verdict = if max_ratio > 0.5 {
            "STRONG_NONLINEAR_COUPLING (Microscopic perturbations induce significant macro response)".to_string()
        } else if max_ratio > 0.05 {
            "WEAK_NONLINEAR_COUPLING (Measurable coarse-scale coupling observed)".to_string()
        } else {
            "LINEAR_DOMINATED (System acts locally linear; coarse modes uncoupled)".to_string()
        };

        Ok(NonlinearPerturbationReport {
            epsilon: eps,
            lp_window_cells: lp_window,
            wavelengths: results,
            verdict,
        })
    }

    /// Evaluates autonomous frozen rollout over H steps and records field trajectory
    pub fn run_autonomous_rollout(
        nca: &NeuralCellularAutomaton,
        interface: &TokenInterface,
        initial_field: &MorphogenicField,
        horizon: usize,
        device: &Device,
    ) -> Result<AutonomousTrajectoryReport> {
        let mut history: Vec<MorphogenicField> = Vec::with_capacity(horizon);
        let mut traces = Vec::with_capacity(horizon);
        let mut current = initial_field.clone();

        for step in 0..horizon {
            let (next_field, update_mag) = nca.step(&current, device)?;
            let state_norm = current.x.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            let energy = current.energy()?;
            let decomp = current.spatial_frequency_decomposition()?;

            // Compute output logits entropy: H = - sum(p * log(p))
            let logits = interface.logits(&current.x)?;
            let (b, l, v) = logits.dims3()?;
            let flat_logits = logits.reshape((b * l, v))?;
            let probs = candle_nn::ops::softmax(&flat_logits, 1)?;
            let probs_vec = probs.to_vec2::<f32>()?;
            let mut total_entropy = 0.0f32;
            for row in &probs_vec {
                let mut ent = 0.0f32;
                for &p in row {
                    if p > 1e-8 {
                        ent -= p * p.ln();
                    }
                }
                total_entropy += ent;
            }
            let mean_entropy = total_entropy / probs_vec.len().max(1) as f32;

            traces.push(TrajectoryStepTrace {
                step,
                state_norm,
                update_magnitude: update_mag,
                energy,
                pct_low: decomp.pct_low,
                pct_mid: decomp.pct_mid,
                pct_high: decomp.pct_high,
                output_entropy: mean_entropy,
            });

            history.push(current.clone());
            current = next_field;
        }

        // Analyze trajectory classification
        let mean_velocity: f32 = traces.iter().map(|t| t.update_magnitude).sum::<f32>() / traces.len().max(1) as f32;
        let final_norm = traces.last().map(|t| t.state_norm).unwrap_or(0.0);
        let final_energy = traces.last().map(|t| t.energy).unwrap_or(0.0);
        let final_entropy = traces.last().map(|t| t.output_entropy).unwrap_or(0.0);

        // Search for periodic recurrence
        let mut min_rec = f32::MAX;
        let mut detected_period = None;
        for i in 8..horizon {
            for j in 0..(i.saturating_sub(2)) {
                let d = history[i].l2_distance(&history[j])?;
                if d < min_rec {
                    min_rec = d;
                    if d < 0.04 && detected_period.is_none() {
                        detected_period = Some(i - j);
                    }
                }
            }
        }

        let classification = if final_norm > 50.0 || final_norm.is_nan() {
            "DIVERGENT (Field exploded without stability)".to_string()
        } else if mean_velocity < 0.001 {
            "CONVERGED_FIXED_POINT (Field decayed to static attractor)".to_string()
        } else if let Some(p) = detected_period {
            format!("PERIODIC_LIMIT_CYCLE (Period T = {} developmental steps)", p)
        } else if min_rec < 0.15 {
            "ACTIVE_BOUNDED_ATTRACTOR (Nontrivial ergodic trajectory)".to_string()
        } else {
            "OPEN_PHASE_DRIFT (Bounded wandering trajectory)".to_string()
        };

        Ok(AutonomousTrajectoryReport {
            horizon,
            classification,
            min_recurrence_distance: min_rec,
            detected_cycle_period: detected_period,
            mean_step_velocity: mean_velocity,
            final_energy,
            final_entropy,
            traces,
        })
    }

    /// Evaluates critical falsification battery:
    /// - Context scrambling
    /// - Context zeroing
    /// - Context truncation
    /// - Hidden-state reset
    /// - Replaying identical input from different internal states
    pub fn run_falsification_battery(
        nca: &NeuralCellularAutomaton,
        interface: &TokenInterface,
        dataset: &crate::dataset::SequenceDataset,
        device: &Device,
    ) -> Result<FalsificationReport> {
        let (inputs, targets) = dataset.sample_train_batch(8, nca.field_cfg.seq_len, device)?;

        // 1. Baseline accuracy
        let base_seed = interface.embed_tokens(&inputs)?;
        let base_field = MorphogenicField::from_tensor(base_seed, &nca.field_cfg);
        let settled_base = nca.develop(&base_field, 8, device)?;
        let base_logits = interface.logits(&settled_base.x)?;
        let base_acc = interface.accuracy(&base_logits, &targets)?;

        // 2. Context scrambling (spatial shuffle)
        let (b, l) = inputs.dims2()?;
        let mut scrambled_indices: Vec<usize> = (0..l).collect();
        // Deterministic shuffle
        for i in (1..l).rev() {
            let j = (i * 7) % (i + 1);
            scrambled_indices.swap(i, j);
        }
        let in_vec = inputs.to_vec2::<u32>()?;
        let mut scrambled_in = in_vec.clone();
        for batch_idx in 0..b {
            for i in 0..l {
                scrambled_in[batch_idx][i] = in_vec[batch_idx][scrambled_indices[i]];
            }
        }
        let scrambled_tensor = Tensor::new(scrambled_in, device)?;
        let scr_seed = interface.embed_tokens(&scrambled_tensor)?;
        let scr_field = MorphogenicField::from_tensor(scr_seed, &nca.field_cfg);
        let settled_scr = nca.develop(&scr_field, 8, device)?;
        let scr_logits = interface.logits(&settled_scr.x)?;
        let scr_acc = interface.accuracy(&scr_logits, &targets)?;

        // 3. Context zeroing
        let zero_seed = Tensor::zeros((b, l, nca.field_cfg.channels), candle_core::DType::F32, device)?;
        let zero_field = MorphogenicField::from_tensor(zero_seed, &nca.field_cfg);
        let settled_zero = nca.develop(&zero_field, 8, device)?;
        let zero_logits = interface.logits(&settled_zero.x)?;
        let zero_acc = interface.accuracy(&zero_logits, &targets)?;

        // 4. Context truncation (zero out second half)
        let mut trunc_in = in_vec.clone();
        for batch_idx in 0..b {
            for i in (l / 2)..l {
                trunc_in[batch_idx][i] = dataset.vocab.pad_id as u32;
            }
        }
        let trunc_tensor = Tensor::new(trunc_in, device)?;
        let trunc_seed = interface.embed_tokens(&trunc_tensor)?;
        let trunc_field = MorphogenicField::from_tensor(trunc_seed, &nca.field_cfg);
        let settled_trunc = nca.develop(&trunc_field, 8, device)?;
        let trunc_logits = interface.logits(&settled_trunc.x)?;
        let trunc_acc = interface.accuracy(&trunc_logits, &targets)?;

        // 5. State dependence (replaying same input from different internal states)
        let state_a = MorphogenicField::zeros(1, &nca.field_cfg, device)?;
        let noise = Tensor::randn(0.0f32, 0.2f32, (1, nca.field_cfg.seq_len, nca.field_cfg.channels), device)?;
        let mut state_b = MorphogenicField::from_tensor(noise.tanh()?, &nca.field_cfg);
        state_b = nca.develop(&state_b, 6, device)?;
        let init_dist = state_a.l2_distance(&state_b)?;

        // Stimulate both with identical token embedding
        let single_input = inputs.narrow(0, 0, 1)?;
        let stim_embed = interface.embed_tokens(&single_input)?;
        let field_a_stim = MorphogenicField::from_tensor((&state_a.x + &stim_embed)?, &nca.field_cfg);
        let field_b_stim = MorphogenicField::from_tensor((&state_b.x + &stim_embed)?, &nca.field_cfg);

        let post_a = nca.develop(&field_a_stim, 8, device)?;
        let post_b = nca.develop(&field_b_stim, 8, device)?;
        let final_dist = post_a.l2_distance(&post_b)?;
        let context_ratio = final_dist / init_dist.max(1e-5);

        let verdict = if scr_acc < base_acc * 0.7 && base_acc > 0.4 {
            "POSITION_AND_CONTEXT_DEPENDENT (Model relies on sequence structure rather than position shortcut)".to_string()
        } else if zero_acc > 0.5 {
            "UNCONDITIONAL_BIAS (Model reproduces sequence without input context)".to_string()
        } else {
            "INITIALIZING_STATE (Model learning underway)".to_string()
        };

        Ok(FalsificationReport {
            baseline_accuracy: base_acc,
            scrambled_context_accuracy: scr_acc,
            zeroed_context_accuracy: zero_acc,
            truncated_context_accuracy: trunc_acc,
            state_reset_divergence: final_dist,
            contextual_memory_ratio: context_ratio,
            verdict,
        })
    }
}
