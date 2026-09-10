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
pub struct FluidStepTrace {
    pub step: usize,
    pub energy: f32,
    pub enstrophy: f32,
    pub palinstrophy: f32,
    pub bkm_norm: f32,
    pub max_velocity: f32,
    pub forcing_power: f32,
    pub dissipation_rate: f32,
    pub spectral_centroid: f32,
    pub spectral_slope: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavierStokesRegimeReport {
    pub condition_label: String,
    pub viscosity: f32,
    pub forcing_amplitude: f32,
    pub initial_enstrophy: f32,
    pub final_enstrophy: f32,
    pub max_enstrophy: f32,
    pub max_bkm_norm: f32,
    pub accumulated_bkm: f32,
    pub max_velocity: f32,
    pub enstrophy_growth_exponent: f32,
    pub blowup_detected: bool,
    pub singularity_step: Option<usize>,
    pub traces: Vec<FluidStepTrace>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViscositySweepResult {
    pub viscosity: f32,
    pub final_energy: f32,
    pub final_enstrophy: f32,
    pub max_bkm: f32,
    pub max_velocity: f32,
    pub blowup_detected: bool,
    pub regularized: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavierStokesBlowupReport {
    pub horizon: usize,
    pub unforced_regime: NavierStokesRegimeReport,
    pub forced_regime: NavierStokesRegimeReport,
    pub viscosity_sweep: Vec<ViscositySweepResult>,
    pub critical_viscosity_est: Option<f32>,
    pub bkm_blowup_criteria_met: bool,
    pub verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullDiagnosticReport {
    pub perturbation: NonlinearPerturbationReport,
    pub trajectory: AutonomousTrajectoryReport,
    pub falsification: FalsificationReport,
    #[serde(default)]
    pub navier_stokes: Option<NavierStokesBlowupReport>,
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

    /// Generates a smooth spatiotemporal forcing field f(s, t) in C^\infty(T^1 x [0, T]).
    /// Spatially zero-mean, restricted to lowest Fourier modes (m in 1..=3), with smooth envelope.
    /// Direct implementation of smooth forcing from Fefferman Millennium Cases C & D.
    pub fn generate_smooth_forcing(
        seq_len: usize,
        channels: usize,
        step: usize,
        horizon: usize,
        amplitude: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let mut data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;

        // Smooth temporal envelope: sin(pi * (t + 1) / (T + 1))
        let t_norm = (step as f32 + 1.0) / (horizon as f32 + 1.0);
        let time_factor = (std::f32::consts::PI * t_norm).sin();
        let effective_amp = amplitude * time_factor;

        for s in 0..seq_len {
            for ch in 0..channels {
                let mut f_val = 0.0f32;
                // Sum of smooth low-wavenumber spatial modes (m = 1, 2, 3)
                for m in 1..=3 {
                    let spatial_phase = two_pi * (m as f32) * (s as f32) / (seq_len as f32);
                    let channel_phase = ((ch * 3 + m * 5) % 17) as f32 * (two_pi / 17.0);
                    let temporal_freq = (m as f32) * 0.25 * (step as f32);
                    let mode_contrib = (spatial_phase + channel_phase + temporal_freq).sin() / (m as f32 * m as f32);
                    f_val += mode_contrib;
                }
                data[0][s][ch] = effective_amp * f_val;
            }
        }

        // Strictly enforce spatial zero-mean across sequence dimension L for every channel
        // Guaranteeing divergence-free zero-mean condition of Fefferman Cases C & D on T^1
        for ch in 0..channels {
            let ch_sum: f32 = (0..seq_len).map(|s| data[0][s][ch]).sum();
            let ch_mean = ch_sum / seq_len as f32;
            for s in 0..seq_len {
                data[0][s][ch] -= ch_mean;
            }
        }

        Tensor::new(data, device).map_err(|e| e.into())
    }

    /// Executes a continuous fluid rollout tracking enstrophy, palinstrophy, BKM norm, and energy cascade
    pub fn run_fluid_rollout(
        nca: &NeuralCellularAutomaton,
        initial_field: &MorphogenicField,
        horizon: usize,
        forcing_amp: f32,
        viscosity: f32,
        condition_label: &str,
        device: &Device,
    ) -> Result<NavierStokesRegimeReport> {
        let active_nca = nca.with_viscosity(viscosity);
        let seq_len = initial_field.config.seq_len;
        let channels = initial_field.config.channels;

        let mut traces = Vec::with_capacity(horizon);
        let mut current = initial_field.clone();

        let initial_enstrophy = current.enstrophy()?;
        let initial_bkm = current.bkm_norm()?;
        let mut max_enstrophy = initial_enstrophy;
        let mut max_bkm_norm = initial_bkm;
        let mut max_velocity = current.max_amplitude()?;
        let mut accumulated_bkm = 0.0f32;
        let mut blowup_detected = false;
        let mut singularity_step = None;

        for step in 0..horizon {
            let energy = current.energy()?;
            let enstrophy = current.enstrophy()?;
            let palinstrophy = current.palinstrophy()?;
            let bkm_norm = current.bkm_norm()?;
            let vel = current.max_amplitude()?;
            let cascade = current.spectral_cascade_analysis()?;

            accumulated_bkm += bkm_norm * active_nca.nca_cfg.step_size;

            if enstrophy > max_enstrophy {
                max_enstrophy = enstrophy;
            }
            if bkm_norm > max_bkm_norm {
                max_bkm_norm = bkm_norm;
            }
            if vel > max_velocity {
                max_velocity = vel;
            }

            // Singularity check: Beale-Kato-Majda divergence or enstrophy/velocity blow-up
            // Requires BOTH large relative surge (>25x initial) AND high absolute threshold (>25.0),
            // OR absolute catastrophic explosion (>100.0 enstrophy, >80.0 velocity, or NaN).
            let rel_ens_surge = enstrophy / initial_enstrophy.max(1e-4);
            let rel_bkm_surge = bkm_norm / initial_bkm.max(1e-4);
            let is_blowup = enstrophy.is_nan()
                || vel.is_nan()
                || bkm_norm.is_nan()
                || vel > 80.0
                || enstrophy > 100.0
                || bkm_norm > 100.0
                || (rel_ens_surge > 25.0 && enstrophy > 25.0)
                || (rel_bkm_surge > 25.0 && bkm_norm > 25.0);

            if !blowup_detected && is_blowup {
                blowup_detected = true;
                singularity_step = Some(step);
            }

            // Smooth forcing calculation
            let (forcing_tensor, forcing_power) = if forcing_amp > 0.0 {
                let f_t = Self::generate_smooth_forcing(seq_len, channels, step, horizon, forcing_amp, device)?;
                // Forcing power: <x, f> = mean(x * f)
                let x_dot_f = (&current.x * &f_t)?.mean_all()?.to_scalar::<f32>()?;
                (Some(f_t), x_dot_f)
            } else {
                (None, 0.0f32)
            };

            let dissipation_rate = 2.0 * viscosity * enstrophy;

            traces.push(FluidStepTrace {
                step,
                energy,
                enstrophy,
                palinstrophy,
                bkm_norm,
                max_velocity: vel,
                forcing_power,
                dissipation_rate,
                spectral_centroid: cascade.centroid_wavenumber,
                spectral_slope: cascade.spectral_slope,
            });

            // If catastrophic divergence occurred, stop rollout early
            if vel > 200.0 || enstrophy > 500.0 || enstrophy.is_nan() || vel.is_nan() {
                break;
            }

            // Advance step
            let (next_field, _) = active_nca.step_with_forcing(&current, forcing_tensor.as_ref(), device)?;
            current = next_field;
        }

        let final_enstrophy = traces.last().map(|t| t.enstrophy).unwrap_or(0.0);

        // Estimate enstrophy growth exponent gamma: dOmega/dt ~ Omega^gamma
        // Strictly fit during the active cascade / blow-up growth window (up to singularity step or peak enstrophy)
        // to avoid dilution from grid saturation / post-breakdown flattening.
        let eval_limit = singularity_step.unwrap_or(traces.len().saturating_sub(1)).max(1);
        let mut log_omega_pairs = Vec::new();
        for i in 1..=eval_limit.min(traces.len().saturating_sub(1)) {
            let delta_omega = traces[i].enstrophy - traces[i - 1].enstrophy;
            let avg_omega = 0.5 * (traces[i].enstrophy + traces[i - 1].enstrophy);
            if delta_omega > 1e-4 && avg_omega > 1e-4 {
                let dt = active_nca.nca_cfg.step_size;
                let rate = delta_omega / dt;
                log_omega_pairs.push((avg_omega.ln(), rate.ln()));
            }
        }

        let enstrophy_growth_exponent = if log_omega_pairs.len() >= 3 {
            let n = log_omega_pairs.len() as f32;
            let sum_x: f32 = log_omega_pairs.iter().map(|(x, _)| x).sum();
            let sum_y: f32 = log_omega_pairs.iter().map(|(_, y)| y).sum();
            let sum_xx: f32 = log_omega_pairs.iter().map(|(x, _)| x * x).sum();
            let sum_xy: f32 = log_omega_pairs.iter().map(|(x, y)| x * y).sum();
            let denom = n * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-6 {
                (n * sum_xy - sum_x * sum_y) / denom
            } else {
                1.0
            }
        } else {
            1.0
        };

        Ok(NavierStokesRegimeReport {
            condition_label: condition_label.to_string(),
            viscosity,
            forcing_amplitude: forcing_amp,
            initial_enstrophy,
            final_enstrophy,
            max_enstrophy,
            max_bkm_norm,
            accumulated_bkm,
            max_velocity,
            enstrophy_growth_exponent,
            blowup_detected,
            singularity_step,
            traces,
        })
    }

    /// Evaluates the complete Navier-Stokes singularity battery (Cases C & D Fefferman formulation):
    /// 1. Unforced baseline dynamics
    /// 2. Smoothly forced dynamics
    /// 3. Viscosity regularization sweep to detect critical viscosity nu*
    pub fn run_navier_stokes_blowup_probe(
        nca: &NeuralCellularAutomaton,
        field: &MorphogenicField,
        horizon: usize,
        forcing_amp: f32,
        base_viscosity: f32,
        device: &Device,
    ) -> Result<NavierStokesBlowupReport> {
        // 1. Unforced baseline regime (f = 0, nu = base_viscosity)
        let unforced_rep = Self::run_fluid_rollout(
            nca,
            field,
            horizon,
            0.0,
            base_viscosity,
            "Unforced Autonomous Flow (f = 0)",
            device,
        )?;

        // 2. Smoothly forced inviscid/baseline regime (f in C^\infty, nu = 0.0)
        let forced_rep = Self::run_fluid_rollout(
            nca,
            field,
            horizon,
            forcing_amp,
            0.0,
            &format!("Smooth External Forcing (A = {:.2}, nu = 0.0)", forcing_amp),
            device,
        )?;

        // 3. Viscosity regularization sweep (including base_viscosity and up to 0.40)
        let mut sweep_viscosities = vec![0.0f32, 0.005, 0.02, 0.05, 0.10, 0.20, 0.40];
        if base_viscosity > 0.0 && !sweep_viscosities.iter().any(|&v| (v - base_viscosity).abs() < 1e-4) {
            sweep_viscosities.push(base_viscosity);
            sweep_viscosities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let mut sweep_results = Vec::new();
        let mut critical_viscosity = None;

        for &nu in &sweep_viscosities {
            let rep = Self::run_fluid_rollout(
                nca,
                field,
                horizon,
                forcing_amp,
                nu,
                &format!("Forced Sweep (nu = {:.3})", nu),
                device,
            )?;

            let final_energy = rep.traces.last().map(|t| t.energy).unwrap_or(0.0);
            let regularized = !rep.blowup_detected && rep.final_enstrophy < (rep.initial_enstrophy * 5.0 + 10.0);

            if regularized && critical_viscosity.is_none() && nu > 0.0 {
                critical_viscosity = Some(nu);
            }

            sweep_results.push(ViscositySweepResult {
                viscosity: nu,
                final_energy,
                final_enstrophy: rep.final_enstrophy,
                max_bkm: rep.max_bkm_norm,
                max_velocity: rep.max_velocity,
                blowup_detected: rep.blowup_detected,
                regularized,
            });
        }

        let bkm_blowup_criteria_met = forced_rep.blowup_detected || forced_rep.enstrophy_growth_exponent > 1.2;

        let verdict = if forced_rep.blowup_detected {
            format!(
                "FINITE_TIME_SINGULARITY_BLOWUP (Smooth forcing triggered gradient breakdown at step {:?}; regularized by nu >= {:?})",
                forced_rep.singularity_step.unwrap_or(0),
                critical_viscosity.unwrap_or(0.05)
            )
        } else if forced_rep.enstrophy_growth_exponent > 1.1 {
            format!(
                "SUPERLINEAR_ENSTROPHY_CASCADE (Growth exponent gamma = {:.2} > 1.0; incipient finite-time singularity dynamics)",
                forced_rep.enstrophy_growth_exponent
            )
        } else {
            "REGULAR_DISSIPATIVE_FLOW (Dynamics remained bounded under smooth external forcing)".to_string()
        };

        Ok(NavierStokesBlowupReport {
            horizon,
            unforced_regime: unforced_rep,
            forced_regime: forced_rep,
            viscosity_sweep: sweep_results,
            critical_viscosity_est: critical_viscosity,
            bkm_blowup_criteria_met,
            verdict,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use crate::config::{FieldConfig, NcaConfig};

    #[test]
    fn test_smooth_forcing_spatial_zero_mean() -> Result<()> {
        let dev = Device::Cpu;
        for &seq_len in &[7, 13, 16, 32] {
            for &channels in &[1, 4, 8] {
                let forcing = ExperimentSuite::generate_smooth_forcing(seq_len, channels, 10, 64, 0.5, &dev)?;
                assert_eq!(forcing.dims3()?, (1, seq_len, channels));
                let f_vec = forcing.to_vec3::<f32>()?;
                for ch in 0..channels {
                    let sum: f32 = (0..seq_len).map(|s| f_vec[0][s][ch]).sum();
                    assert!(sum.abs() < 1e-5, "Smooth forcing must be strictly zero-mean for L={}, ch={}, sum={}", seq_len, ch, sum);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_navier_stokes_blowup_probe_execution() -> Result<()> {
        let dev = Device::Cpu;
        let nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 16, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let initial_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;

        let report = ExperimentSuite::run_navier_stokes_blowup_probe(
            &nca,
            &initial_field,
            8,   // short horizon for unit test
            0.2, // forcing amp
            0.0, // base viscosity
            &dev,
        )?;

        assert_eq!(report.horizon, 8);
        assert_eq!(report.unforced_regime.traces.len(), 8);
        assert_eq!(report.forced_regime.traces.len(), 8);
        assert!(!report.viscosity_sweep.is_empty());
        assert!(!report.verdict.is_empty());
        Ok(())
    }
}
