use crate::field::MorphogenicField;
use crate::instrumentation::{
    compute_cosine_similarity, compute_effective_dimension, compute_kl_divergence,
    InstrumentationTrace, TickMetrics, TraceSummary,
};
use crate::intervention::InterventionConfig;
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::{TokenInterface, Vocab};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use serde::{Deserialize, Serialize};

/// Configuration controlling independent sequence and latent compute depth,
/// mixed-timescale cadences, and readout intervals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentConfig {
    /// Number of internal recurrence ticks per token input step (tau in 0, 1, 2, 4, 8, 16...)
    pub latent_ticks_per_token: usize,
    /// Number of post-input autonomous deliberation/settling ticks before final readout
    pub post_input_ticks: usize,
    /// Cadence for slow delayed pathway (e.g. 2, 4; slow channels update every M ticks)
    #[serde(default = "default_cadence")]
    pub slow_timescale_cadence: usize,
    /// Fraction of state channels assigned to the slow delayed pathway (0.0 to 1.0)
    #[serde(default)]
    pub slow_channel_fraction: f32,
    /// Interval for non-invasive passive observation (1 = every tick)
    #[serde(default = "default_cadence")]
    pub observation_interval: usize,
}

fn default_cadence() -> usize {
    1
}

impl Default for LatentConfig {
    fn default() -> Self {
        Self {
            latent_ticks_per_token: 1,
            post_input_ticks: 0,
            slow_timescale_cadence: 1,
            slow_channel_fraction: 0.0,
            observation_interval: 1,
        }
    }
}

/// Result of executing a sequence with latent ticks and instrumentation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRunResult {
    pub total_latent_ticks: usize,
    pub final_logits: Vec<Vec<f32>>,
    pub final_predicted_tokens: Vec<usize>,
    pub decoded_output: String,
    pub trace: InstrumentationTrace,
    pub summary: TraceSummary,
    #[serde(default)]
    pub final_state_vec: Vec<f32>,
}

/// Report for dynamic association emergence across latent time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssociationTickProbe {
    pub tick: usize,
    pub target_concept: String,
    pub target_probability: f32,
    pub target_rank: usize,
    pub top_decoded_token: String,
    pub state_displacement: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicAssociationReport {
    pub stimulus_concept: String,
    pub target_concept: String,
    pub horizon: usize,
    pub emergence_tick: Option<usize>,
    pub peak_tick: usize,
    pub peak_probability: f32,
    pub final_probability: f32,
    pub probes: Vec<AssociationTickProbe>,
    pub verdict: String,
}

/// Report for attractor basin and hysteresis dynamics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttractorBasinReport {
    pub stimulus_label: String,
    pub basin_a_final_norm: f32,
    pub basin_b_final_norm: f32,
    pub final_inter_basin_distance: f32,
    pub basin_separation_ratio: f32,
    pub hysteresis_detected: bool,
    pub state_trajectories_basin_a: Vec<Vec<f32>>, // sample trajectory vectors for PCA/UMAP
    pub state_trajectories_basin_b: Vec<Vec<f32>>,
    pub classification: String,
}

/// Report for memory-as-dynamics experiment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryDynamicsReport {
    pub event_a_label: String,
    pub baseline_event_label: String,
    pub probe_event_label: String,
    pub latent_ticks_between: usize,
    pub state_distortion_distance: f32,
    pub readout_kl_divergence: f32,
    pub dynamics_modified: bool,
    pub description: String,
}

/// Latent budget point result for sweep comparison tables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentBudgetPoint {
    pub latent_ticks: usize,
    pub accuracy: f32,
    pub loss: f32,
    pub mean_confidence: f32,
    pub state_displacement: f32,
    pub compute_cost_units: usize,
    pub regime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_slot_accuracy: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentBudgetSweepReport {
    pub task_name: String,
    pub budgets: Vec<LatentBudgetPoint>,
    pub monotonic_improvement: bool,
    pub bifurcation_detected: bool,
    pub optimal_ticks: usize,
}

pub struct LatentExecutor<'a> {
    pub nca: &'a NeuralCellularAutomaton,
    pub interface: &'a TokenInterface,
    pub vocab: &'a Vocab,
}

impl<'a> LatentExecutor<'a> {
    pub fn new(
        nca: &'a NeuralCellularAutomaton,
        interface: &'a TokenInterface,
        vocab: &'a Vocab,
    ) -> Self {
        Self {
            nca,
            interface,
            vocab,
        }
    }

    /// Single latent recurrence tick with explicit metrics collection and intervention hooks.
    /// Returns (next_field, tick_metrics, current_probs).
    pub fn step_latent(
        &self,
        field: &MorphogenicField,
        tick: usize,
        token_step: Option<usize>,
        latent_cfg: &LatentConfig,
        intervention: &InterventionConfig,
        origin_tensor: &Tensor,
        last_probs: Option<&[f32]>,
        device: &Device,
    ) -> Result<(MorphogenicField, TickMetrics, Vec<f32>)> {
        let x = &field.x;

        // Apply pre-step state intervention (e.g. reset, noise, ablation)
        let modified_x = intervention.apply_to_state(tick, x, device)?;
        let is_frozen = intervention.is_frozen(tick);

        if is_frozen {
            let state_norm = modified_x.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            let logits = self.interface.logits(&modified_x)?;
            let first_logits = logits.narrow(0, 0, 1)?.squeeze(0)?.narrow(0, 0, 1)?.squeeze(0)?;
            let probs = candle_nn::ops::softmax(&first_logits, 0)?.to_vec1::<f32>()?;
            let top_id = first_logits.argmax(0)?.to_scalar::<u32>()? as usize;
            let conf = probs[top_id.min(probs.len().saturating_sub(1))];

            let metrics = TickMetrics {
                tick,
                token_step,
                state_norm,
                update_magnitude: 0.0,
                cosine_to_prev: 1.0,
                cosine_to_origin: compute_cosine_similarity(&modified_x, origin_tensor)?,
                gate_mean: 0.0,
                gate_std: 0.0,
                gate_sat_closed: 1.0,
                gate_sat_open: 0.0,
                delta_sat: 0.0,
                effective_dimension: compute_effective_dimension(&modified_x)?,
                mlp_norm: 0.0,
                raw_delta_norm: 0.0,
                gated_delta_norm: 0.0,
                dissipation_norm: 0.0,
                top_token_id: top_id,
                top_token_char: self.vocab.decode(&[top_id]),
                confidence: conf,
                output_entropy: 0.0,
                readout_kl_prev: 0.0,
            };

            return Ok((
                MorphogenicField {
                    x: modified_x,
                    config: field.config.clone(),
                    slow_state: field.slow_state.clone(),
                },
                metrics,
                probs,
            ));
        }

        // 1. Perception
        let raw_perception = self.nca.perceive(&modified_x)?;
        let perception = intervention.apply_to_perception(&raw_perception, &modified_x)?;

        // 2. Hidden feature extraction
        let h1 = self.nca.dense1.forward(&perception)?;
        let h1_act = match self.nca.nca_cfg.activation.as_str() {
            "tanh" => h1.tanh()?,
            _ => candle_nn::Activation::Gelu.forward(&h1)?,
        };
        let mlp_norm = h1_act.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

        // 3. Raw directional delta and gate
        let raw_delta = self.nca.dense_delta.forward(&h1_act)?.tanh()?;
        let raw_gate = candle_nn::ops::sigmoid(&self.nca.dense_gate.forward(&h1_act)?)?;

        // Interventions on gate and delta
        let active_gate = intervention.apply_to_gate(&raw_gate)?;
        let active_delta = intervention.apply_to_delta(&raw_delta)?;

        let gated_delta = active_delta.mul(&active_gate)?;
        let raw_delta_norm = active_delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let gated_delta_norm = gated_delta.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();

        // Mixed-timescale handling (Requirement 10)
        let effective_delta = if latent_cfg.slow_channel_fraction > 0.0
            && latent_cfg.slow_timescale_cadence > 1
        {
            let (_, _, c) = modified_x.dims3()?;
            let num_slow = ((c as f32) * latent_cfg.slow_channel_fraction.clamp(0.0, 1.0)).round() as usize;
            let is_slow_active = tick % latent_cfg.slow_timescale_cadence == 0;

            if !is_slow_active && num_slow > 0 {
                // Zero out delta for the slow channels on non-cadence ticks
                let mut mask_vec = vec![1.0f32; c];
                let slow_start = c.saturating_sub(num_slow);
                for ch in slow_start..c {
                    mask_vec[ch] = 0.0;
                }
                let mask = Tensor::from_slice(&mask_vec, (1, 1, c), device)?;
                gated_delta.broadcast_mul(&mask)?
            } else {
                gated_delta
            }
        } else {
            gated_delta
        };

        // 4. Residual integration: x_{t+1} = x_t + alpha * delta (or bypass residual)
        let alpha = self.nca.nca_cfg.step_size as f64;
        let scaled_delta = (effective_delta * alpha)?;
        let interim_x = if intervention.disable_residual {
            scaled_delta
        } else {
            (&modified_x + &scaled_delta)?
        };

        // 5. Viscous dissipation
        let mut dissipation_norm = 0.0f32;
        let new_x = if self.nca.nca_cfg.viscosity > 0.0 {
            let diff_amount = self.nca.nca_cfg.viscosity.max(0.0) * self.nca.nca_cfg.step_size;
            let dissipated = self.nca.dissipate(&interim_x, diff_amount)?;
            let diss_diff = (&dissipated - &interim_x)?;
            dissipation_norm = diss_diff.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
            dissipated
        } else {
            interim_x
        };

        // Metrics computation
        let state_norm = new_x.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let disp = (&new_x - x)?;
        let update_magnitude = disp.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let cosine_to_prev = compute_cosine_similarity(&new_x, x)?;
        let cosine_to_origin = compute_cosine_similarity(&new_x, origin_tensor)?;

        // Gate statistics
        let gate_vec = active_gate.flatten_all()?.to_vec1::<f32>()?;
        let mut sum_g = 0.0f32;
        let mut sum_g_sq = 0.0f32;
        let mut count_closed = 0usize;
        let mut count_open = 0usize;
        for &g in &gate_vec {
            sum_g += g;
            sum_g_sq += g * g;
            if g < 0.10 {
                count_closed += 1;
            }
            if g > 0.90 {
                count_open += 1;
            }
        }
        let total_g = gate_vec.len().max(1) as f32;
        let gate_mean = sum_g / total_g;
        let gate_var = (sum_g_sq / total_g - gate_mean * gate_mean).max(0.0);
        let gate_std = gate_var.sqrt();
        let gate_sat_closed = count_closed as f32 / total_g;
        let gate_sat_open = count_open as f32 / total_g;

        // Delta saturation (|delta| > 0.95)
        let delta_vec = active_delta.flatten_all()?.to_vec1::<f32>()?;
        let count_sat_delta = delta_vec.iter().filter(|&&d| d.abs() > 0.95).count();
        let delta_sat = count_sat_delta as f32 / delta_vec.len().max(1) as f32;

        let effective_dimension = compute_effective_dimension(&new_x)?;

        // Readout metrics
        let logits = self.interface.logits(&new_x)?;
        let (_b, _l, _v) = logits.dims3()?;
        let first_logits = logits.narrow(0, 0, 1)?.squeeze(0)?.narrow(0, 0, 1)?.squeeze(0)?;
        let current_probs = candle_nn::ops::softmax(&first_logits, 0)?.to_vec1::<f32>()?;

        let top_id = first_logits.argmax(0)?.to_scalar::<u32>()? as usize;
        let confidence = current_probs[top_id.min(current_probs.len().saturating_sub(1))];

        let mut output_entropy = 0.0f32;
        for &p in &current_probs {
            if p > 1e-8 {
                output_entropy -= p * p.ln();
            }
        }

        let readout_kl_prev = match last_probs {
            Some(prev) => compute_kl_divergence(&current_probs, prev),
            None => 0.0,
        };

        let metrics = TickMetrics {
            tick,
            token_step,
            state_norm,
            update_magnitude,
            cosine_to_prev,
            cosine_to_origin,
            gate_mean,
            gate_std,
            gate_sat_closed,
            gate_sat_open,
            delta_sat,
            effective_dimension,
            mlp_norm,
            raw_delta_norm,
            gated_delta_norm,
            dissipation_norm,
            top_token_id: top_id,
            top_token_char: self.vocab.decode(&[top_id]),
            confidence,
            output_entropy,
            readout_kl_prev,
        };

        let next_field = MorphogenicField {
            x: new_x,
            config: field.config.clone(),
            slow_state: field.slow_state.clone(),
        };

        Ok((next_field, metrics, current_probs))
    }

    /// Executes sequence with independent token steps and latent ticks.
    /// Clearly separates:
    /// - Token/input step: embeds token into state
    /// - Latent recurrence step: runs K ticks between tokens
    /// - Post-input deliberation: runs N ticks before readout
    /// - Readout step: observes output without feedback
    pub fn execute_sequence(
        &self,
        token_ids: &[usize],
        latent_cfg: &LatentConfig,
        intervention: &InterventionConfig,
        device: &Device,
    ) -> Result<LatentRunResult> {
        let seq_len = self.nca.field_cfg.seq_len;
        let channels = self.nca.field_cfg.channels;
        let mut field = MorphogenicField::zeros(1, &self.nca.field_cfg, device)?;
        let origin_tensor = field.x.clone();

        let mut trace = InstrumentationTrace::new("sequence_execution");
        let mut global_tick = 0;
        let mut last_probs = None;

        // Sequence token processing phase
        for (step_idx, &tok) in token_ids.iter().enumerate() {
            // Token input step: inject token embedding into the field
            let tok_tensor = Tensor::from_slice(&[tok as u32], (1, 1), device)?;
            let tok_embed = self.interface.embed_tokens(&tok_tensor)?; // [1, 1, C]

            // Inject at position (or broadcast across spatial lattice)
            let cell_pos = step_idx % seq_len;
            let mut field_data = field.x.flatten_all()?.to_vec1::<f32>()?;
            let embed_data = tok_embed.flatten_all()?.to_vec1::<f32>()?;
            let offset = cell_pos * channels;
            for ch in 0..channels {
                field_data[offset + ch] += embed_data[ch];
            }
            field.x = Tensor::from_slice(&field_data, (1, seq_len, channels), device)?;

            // Latent recurrence ticks for this token step
            for _ in 0..latent_cfg.latent_ticks_per_token {
                let (next_f, metric, probs) = self.step_latent(
                    &field,
                    global_tick,
                    Some(step_idx),
                    latent_cfg,
                    intervention,
                    &origin_tensor,
                    last_probs.as_deref(),
                    device,
                )?;
                trace.record(metric);
                last_probs = Some(probs);
                field = next_f;
                global_tick += 1;
            }
        }

        // Post-input autonomous deliberation phase (Requirement 3: N latent ticks before readout)
        for _ in 0..latent_cfg.post_input_ticks {
            let (next_f, metric, probs) = self.step_latent(
                &field,
                global_tick,
                None,
                latent_cfg,
                intervention,
                &origin_tensor,
                last_probs.as_deref(),
                device,
            )?;
            trace.record(metric);
            last_probs = Some(probs);
            field = next_f;
            global_tick += 1;
        }

        // Final readout step (non-invasive observation)
        let logits = self.interface.logits(&field.x)?;
        let (_, _, v) = logits.dims3()?;
        let flat_logits = logits.reshape((seq_len, v))?.to_vec2::<f32>()?;

        let preds = logits.argmax(candle_core::D::Minus1)?;
        let pred_ids: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let usize_ids: Vec<usize> = pred_ids.iter().map(|&id| id as usize).collect();
        let decoded = self.vocab.decode(&usize_ids);
        let summary = trace.summary();
        let final_state_vec = field.x.flatten_all()?.to_vec1::<f32>()?;

        Ok(LatentRunResult {
            total_latent_ticks: global_tick,
            final_logits: flat_logits,
            final_predicted_tokens: usize_ids,
            decoded_output: decoded,
            trace,
            summary,
            final_state_vec,
        })
    }

    /// Dynamic association experiment (Requirement 8):
    /// Given stimulus concept A:
    /// - Inject into state
    /// - Evolve state over latent ticks without new input
    /// - Probe at each tick which concepts/tokens are decodable and their rank/probability
    pub fn run_dynamic_association_probe(
        &self,
        stimulus_char: char,
        target_concept_char: char,
        horizon: usize,
        latent_cfg: &LatentConfig,
        intervention: &InterventionConfig,
        device: &Device,
    ) -> Result<DynamicAssociationReport> {
        let stim_ids = self.vocab.encode(&stimulus_char.to_string());
        let tgt_ids = self.vocab.encode(&target_concept_char.to_string());
        let stim_id = stim_ids.first().cloned().unwrap_or(self.vocab.unk_id);
        let tgt_id = tgt_ids.first().cloned().unwrap_or(self.vocab.unk_id);

        let mut field = MorphogenicField::zeros(1, &self.nca.field_cfg, device)?;
        let stim_tensor = Tensor::from_slice(&[stim_id as u32], (1, 1), device)?;
        let stim_embed = self.interface.embed_tokens(&stim_tensor)?;

        let mut data = field.x.to_vec3::<f32>()?;
        let embed_vec = stim_embed.to_vec3::<f32>()?;
        for ch in 0..self.nca.field_cfg.channels {
            data[0][0][ch] = embed_vec[0][0][ch];
        }
        field.x = Tensor::new(data, device)?;
        let origin_tensor = field.x.clone();

        let mut probes = Vec::with_capacity(horizon);
        let mut last_probs = None;
        let mut peak_prob = 0.0f32;
        let mut peak_tick = 0;
        let mut emergence_tick = None;

        for tick in 0..horizon {
            let (next_f, metric, probs) = self.step_latent(
                &field,
                tick,
                None,
                latent_cfg,
                intervention,
                &origin_tensor,
                last_probs.as_deref(),
                device,
            )?;

            let tgt_prob = probs.get(tgt_id).cloned().unwrap_or(0.0);
            if tgt_prob > 0.10 && emergence_tick.is_none() {
                emergence_tick = Some(tick);
            }
            if tgt_prob > peak_prob {
                peak_prob = tgt_prob;
                peak_tick = tick;
            }

            // Target rank
            let mut rank = 1;
            for &p in &probs {
                if p > tgt_prob {
                    rank += 1;
                }
            }

            probes.push(AssociationTickProbe {
                tick,
                target_concept: target_concept_char.to_string(),
                target_probability: tgt_prob,
                target_rank: rank,
                top_decoded_token: metric.top_token_char.clone(),
                state_displacement: metric.update_magnitude,
            });

            last_probs = Some(probs);
            field = next_f;
        }

        let final_prob = probes.last().map(|p| p.target_probability).unwrap_or(0.0);
        let verdict = if let Some(em_t) = emergence_tick {
            format!(
                "DYNAMIC_PROPAGATION (Association emerged at tick {}, peaked at tick {} with P = {:.3})",
                em_t, peak_tick, peak_prob
            )
        } else {
            "WEAK_ASSOCIATION (Target concept probability remained below detection threshold)".to_string()
        };

        Ok(DynamicAssociationReport {
            stimulus_concept: stimulus_char.to_string(),
            target_concept: target_concept_char.to_string(),
            horizon,
            emergence_tick,
            peak_tick,
            peak_probability: peak_prob,
            final_probability: final_prob,
            probes,
            verdict,
        })
    }

    /// Attractor basin and hysteresis experiment (Requirement 9):
    /// Tests convergence to competing state basins and hysteresis.
    pub fn run_attractor_basin_probe(
        &self,
        stimulus_a: &str,
        stimulus_b: &str,
        horizon: usize,
        latent_cfg: &LatentConfig,
        intervention: &InterventionConfig,
        device: &Device,
    ) -> Result<AttractorBasinReport> {
        let seq_a = self.vocab.encode(stimulus_a);
        let seq_b = self.vocab.encode(stimulus_b);

        let mut cfg = latent_cfg.clone();
        cfg.post_input_ticks = horizon;

        let (res_a, res_b) = rayon::join(
            || self.execute_sequence(&seq_a, &cfg, intervention, device),
            || self.execute_sequence(&seq_b, &cfg, intervention, device),
        );
        let res_a = res_a?;
        let res_b = res_b?;

        let mut traj_a = Vec::new();
        let mut traj_b = Vec::new();

        for t in &res_a.trace.ticks {
            traj_a.push(vec![t.state_norm, t.update_magnitude, t.confidence, t.output_entropy]);
        }
        for t in &res_b.trace.ticks {
            traj_b.push(vec![t.state_norm, t.update_magnitude, t.confidence, t.output_entropy]);
        }

        let norm_a = res_a.summary.final_norm;
        let norm_b = res_b.summary.final_norm;
        let dist = if res_a.final_state_vec.len() == res_b.final_state_vec.len() && !res_a.final_state_vec.is_empty() {
            let sum_sq: f32 = res_a.final_state_vec.iter().zip(&res_b.final_state_vec)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            (sum_sq / res_a.final_state_vec.len() as f32).sqrt()
        } else {
            (norm_a - norm_b).abs()
        };
        let sep_ratio = dist / norm_a.max(norm_b).max(1e-5);

        let hysteresis_detected = sep_ratio > 0.15;
        let classification = if hysteresis_detected {
            "SEPARATED_STABLE_REGIMES (Trajectories remain separated in distinct state basins; path dependence confirmed)".to_string()
        } else {
            "MONO_BASIN_CONVERGENCE (Both stimuli settle into overlapping state regions)".to_string()
        };

        Ok(AttractorBasinReport {
            stimulus_label: format!("{} vs {}", stimulus_a, stimulus_b),
            basin_a_final_norm: norm_a,
            basin_b_final_norm: norm_b,
            final_inter_basin_distance: dist,
            basin_separation_ratio: sep_ratio,
            hysteresis_detected,
            state_trajectories_basin_a: traj_a,
            state_trajectories_basin_b: traj_b,
            classification,
        })
    }

    /// Memory as dynamics experiment (Requirement 12):
    /// Earlier event E1 modifies state. Later event E2 is processed.
    /// Measure whether E1 persistently modifies the dynamic response to E2
    /// without directly replaying E1.
    pub fn run_memory_as_dynamics_experiment(
        &self,
        event_1: &str,
        event_2: &str,
        latent_ticks_between: usize,
        intervention: &InterventionConfig,
        device: &Device,
    ) -> Result<MemoryDynamicsReport> {
        let e1_ids = self.vocab.encode(event_1);
        let e2_ids = self.vocab.encode(event_2);

        let mut cfg = LatentConfig::default();
        cfg.latent_ticks_per_token = 1;
        cfg.post_input_ticks = latent_ticks_between;

        // Condition A: Process E1, wait ticks_between, then process E2
        let mut full_seq = e1_ids.clone();
        full_seq.extend(&e2_ids);
        let (res_conditioned, res_unconditioned) = rayon::join(
            || self.execute_sequence(&full_seq, &cfg, intervention, device),
            || self.execute_sequence(&e2_ids, &cfg, intervention, device),
        );
        let res_conditioned = res_conditioned?;
        let res_unconditioned = res_unconditioned?;

        let final_norm_a = res_conditioned.summary.final_norm;
        let final_norm_b = res_unconditioned.summary.final_norm;
        let state_dist = (final_norm_a - final_norm_b).abs();

        // Compare readout distribution
        let kl = match (res_conditioned.final_logits.first(), res_unconditioned.final_logits.first()) {
            (Some(l1), Some(l2)) => {
                let p1 = softmax_slice(l1);
                let p2 = softmax_slice(l2);
                compute_kl_divergence(&p1, &p2)
            }
            _ => 0.0,
        };

        let dynamics_modified = state_dist > 0.05 || kl > 0.05;
        let description = if dynamics_modified {
            format!(
                "PERSISTENT_DYNAMIC_MEMORY (Prior event '{}' shifted subsequent trajectory with KL = {:.4}, displacement = {:.4})",
                event_1, kl, state_dist
            )
        } else {
            "DECAYED_INFLUENCE (Prior event relaxed to baseline state before probe)".to_string()
        };

        Ok(MemoryDynamicsReport {
            event_a_label: event_1.to_string(),
            baseline_event_label: "(baseline)".to_string(),
            probe_event_label: event_2.to_string(),
            latent_ticks_between,
            state_distortion_distance: state_dist,
            readout_kl_divergence: kl,
            dynamics_modified,
            description,
        })
    }

    /// Latent-budget sweep (Requirement 11):
    /// Evaluates model performance across a range of latent compute budgets (0, 1, 2, 4, 8, 16, 32...).
    #[allow(dead_code)]
    pub fn run_latent_budget_sweep(
        &self,
        task_kind: crate::tasks::TaskKind,
        latent_budgets: &[usize],
        batch_size: usize,
        seq_len: usize,
        intervention: &InterventionConfig,
        device: &Device,
    ) -> Result<LatentBudgetSweepReport> {
        self.run_latent_budget_sweep_seeded(
            task_kind,
            latent_budgets,
            batch_size,
            seq_len,
            intervention,
            0,
            device,
        )
    }

    /// Latent-budget sweep with explicit seed support for multi-seed evaluations.
    pub fn run_latent_budget_sweep_seeded(
        &self,
        task_kind: crate::tasks::TaskKind,
        latent_budgets: &[usize],
        batch_size: usize,
        seq_len: usize,
        intervention: &InterventionConfig,
        seed: usize,
        device: &Device,
    ) -> Result<LatentBudgetSweepReport> {
        let task_engine = crate::tasks::TaskEngine::new();
        let batch = task_engine.generate_batch_seeded(task_kind, batch_size, seq_len, true, seed, device)?;

        use rayon::prelude::*;
        let raw_points: Vec<(usize, f32, f32, f32, f32, Option<Vec<f32>>)> = latent_budgets
            .par_iter()
            .map(|&ticks| -> Result<(usize, f32, f32, f32, f32, Option<Vec<f32>>)> {
                let mut cfg = LatentConfig::default();
                cfg.latent_ticks_per_token = ticks.max(1);
                cfg.post_input_ticks = if ticks == 0 { 0 } else { ticks };

                let seed = self.interface.embed_tokens(&batch.inputs)?;
                let mut field = MorphogenicField::from_tensor(seed, &self.nca.field_cfg);
                let origin = field.x.clone();

                let mut total_disp = 0.0f32;
                let mut last_probs = None;

                for t in 0..ticks {
                    let (next_f, metric, probs) = self.step_latent(
                        &field,
                        t,
                        None,
                        &cfg,
                        intervention,
                        &origin,
                        last_probs.as_deref(),
                        device,
                    )?;
                    total_disp += metric.update_magnitude;
                    last_probs = Some(probs);
                    field = next_f;
                }

                let logits = self.interface.logits(&field.x)?;
                let mask_sum = batch.loss_mask.sum_all()?.to_scalar::<f32>()?;
                let (loss, acc, per_slot) = if mask_sum > 0.0 {
                    let loss_t = self.interface.masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
                    let loss = loss_t.to_scalar::<f32>()?;
                    let acc = self.interface.masked_accuracy(&logits, &batch.targets, &batch.loss_mask)?;

                    let preds = logits.argmax(candle_core::D::Minus1)?;
                    let preds_vec = preds.to_vec2::<u32>()?;
                    let targets_vec = batch.targets.to_vec2::<u32>()?;
                    let mask_vec = batch.loss_mask.to_vec2::<f32>()?;
                    let b = preds_vec.len();
                    let l = if b > 0 { preds_vec[0].len() } else { 0 };

                    let mut slot_counts: Vec<(usize, usize)> = Vec::new();
                    for i in 0..b {
                        let mut slot_idx = 0;
                        for j in 0..l {
                            if mask_vec[i][j] > 0.5 {
                                if slot_idx >= slot_counts.len() {
                                    slot_counts.push((0, 0));
                                }
                                slot_counts[slot_idx].1 += 1;
                                if preds_vec[i][j] == targets_vec[i][j] {
                                    slot_counts[slot_idx].0 += 1;
                                }
                                slot_idx += 1;
                            }
                        }
                    }
                    let slot_accs = if !slot_counts.is_empty() {
                        Some(slot_counts.into_iter().map(|(c, t)| if t > 0 { c as f32 / t as f32 } else { 0.0 }).collect())
                    } else {
                        None
                    };

                    (loss, acc, slot_accs)
                } else {
                    let loss_t = self.interface.cross_entropy_loss(&logits, &batch.targets)?;
                    let loss = loss_t.to_scalar::<f32>()?;
                    let acc = self.interface.accuracy(&logits, &batch.targets)?;
                    (loss, acc, None)
                };

                let flat_logits = logits.flatten_all()?;
                let probs = candle_nn::ops::softmax(&flat_logits, 0)?.to_vec1::<f32>()?;
                let mean_conf = probs.iter().fold(0.0f32, |m, &p| m.max(p));

                Ok((ticks, acc, loss, mean_conf, total_disp, per_slot))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut points = Vec::with_capacity(raw_points.len());
        let mut best_acc = -1.0f32;
        let mut best_ticks = 0;

        for (ticks, acc, loss, mean_conf, total_disp, per_slot) in raw_points {
            let regime = if acc > best_acc && best_acc >= 0.0 {
                "COMPUTE_GAIN".to_string()
            } else if (acc - best_acc).abs() < 1e-4 {
                "SATURATED".to_string()
            } else {
                "DEGRADATION".to_string()
            };

            if acc > best_acc {
                best_acc = acc;
                best_ticks = ticks;
            }

            points.push(LatentBudgetPoint {
                latent_ticks: ticks,
                accuracy: acc,
                loss,
                mean_confidence: mean_conf,
                state_displacement: total_disp,
                compute_cost_units: ticks * batch_size * seq_len,
                regime,
                per_slot_accuracy: per_slot,
            });
        }

        let monotonic = points.windows(2).all(|w| w[1].accuracy >= w[0].accuracy)
            && points.iter().any(|w| w.accuracy > points[0].accuracy);
        let bifurcation = points.windows(2).any(|w| (w[1].accuracy - w[0].accuracy).abs() > 0.25);

        Ok(LatentBudgetSweepReport {
            task_name: task_kind.name().to_string(),
            budgets: points,
            monotonic_improvement: monotonic,
            bifurcation_detected: bifurcation,
            optimal_ticks: best_ticks,
        })
    }

    /// Computes the empirical causal dependency map: input position -> output query slot across latent ticks.
    pub fn compute_causal_influence_matrix(
        &self,
        task_kind: crate::tasks::TaskKind,
        budgets: &[usize],
        batch_size: usize,
        seq_len: usize,
        zero_boundary: bool,
        seed: usize,
        device: &Device,
    ) -> Result<CausalInfluenceReport> {
        let task_engine = crate::tasks::TaskEngine::new();
        let batch = task_engine.generate_batch_seeded(task_kind, batch_size, seq_len, true, seed, device)?;
        let mask_vec = batch.loss_mask.to_vec2::<f32>()?;
        let inputs_vec = batch.inputs.to_vec2::<u32>()?;
        let b = inputs_vec.len();
        let l = if b > 0 { inputs_vec[0].len() } else { 0 };

        let mut query_positions = Vec::new();
        if b > 0 {
            for j in 0..l {
                if mask_vec[0][j] > 0.5 {
                    query_positions.push(j);
                }
            }
        }
        let num_slots = query_positions.len();

        let zero_id = self.vocab.encode("0")[0] as u32;
        let one_id = self.vocab.encode("1")[0] as u32;

        let mut num_inf = Vec::with_capacity(budgets.len());
        let mut flip_prob = Vec::with_capacity(budgets.len());
        let mut theo_reach = Vec::with_capacity(budgets.len());

        let cfg = LatentConfig::default();
        let intervention = InterventionConfig::default();

        for &ticks in budgets {
            let clean_seed = self.interface.embed_tokens(&batch.inputs)?;
            let mut clean_field = MorphogenicField::from_tensor(clean_seed, &self.nca.field_cfg);
            let clean_origin = clean_field.x.clone();
            for t in 0..ticks {
                let (next_f, _, _) = self.step_latent(&clean_field, t, None, &cfg, &intervention, &clean_origin, None, device)?;
                clean_field = next_f;
            }
            let clean_logits = self.interface.logits(&clean_field.x)?;
            let clean_preds = clean_logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;
            let clean_logits_vec = clean_logits.to_vec3::<f32>()?;

            let mut budget_num = vec![vec![0.0f32; l]; num_slots];
            let mut budget_flip = vec![vec![0.0f32; l]; num_slots];
            let mut budget_theo = vec![vec![false; l]; num_slots];

            for (s_idx, &q_pos) in query_positions.iter().enumerate() {
                for j in 0..l {
                    let dist = if zero_boundary {
                        (q_pos as isize - j as isize).abs() as usize
                    } else {
                        let d1 = (q_pos as isize - j as isize).abs() as usize;
                        let d2 = l - d1;
                        d1.min(d2)
                    };
                    budget_theo[s_idx][j] = dist <= ticks;

                    let mut flipped_inputs_vec = inputs_vec.clone();
                    for row in 0..b {
                        let cur = flipped_inputs_vec[row][j];
                        if cur == zero_id {
                            flipped_inputs_vec[row][j] = one_id;
                        } else if cur == one_id {
                            flipped_inputs_vec[row][j] = zero_id;
                        }
                    }
                    let flipped_tokens = Tensor::from_vec(
                        flipped_inputs_vec.into_iter().flatten().collect(),
                        (b, l),
                        device,
                    )?;

                    let flipped_seed = self.interface.embed_tokens(&flipped_tokens)?;
                    let mut flipped_field = MorphogenicField::from_tensor(flipped_seed, &self.nca.field_cfg);
                    let flipped_origin = flipped_field.x.clone();
                    for t in 0..ticks {
                        let (next_f, _, _) = self.step_latent(&flipped_field, t, None, &cfg, &intervention, &flipped_origin, None, device)?;
                        flipped_field = next_f;
                    }
                    let flipped_logits = self.interface.logits(&flipped_field.x)?;
                    let flipped_preds = flipped_logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;
                    let flipped_logits_vec = flipped_logits.to_vec3::<f32>()?;

                    let mut sum_l2 = 0.0f32;
                    let mut flip_count = 0usize;
                    for row in 0..b {
                        let mut diff_sq = 0.0f32;
                        for v in 0..clean_logits_vec[row][q_pos].len() {
                            let diff = flipped_logits_vec[row][q_pos][v] - clean_logits_vec[row][q_pos][v];
                            diff_sq += diff * diff;
                        }
                        sum_l2 += diff_sq.sqrt();
                        if flipped_preds[row][q_pos] != clean_preds[row][q_pos] {
                            flip_count += 1;
                        }
                    }
                    budget_num[s_idx][j] = sum_l2 / b as f32;
                    budget_flip[s_idx][j] = flip_count as f32 / b as f32;
                }
            }

            num_inf.push(budget_num);
            flip_prob.push(budget_flip);
            theo_reach.push(budget_theo);
        }

        Ok(CausalInfluenceReport {
            task_name: task_kind.name().to_string(),
            seq_len,
            budgets: budgets.to_vec(),
            query_slots: (0..num_slots).collect(),
            query_slot_indices: query_positions,
            numerical_influence: num_inf,
            flip_probability: flip_prob,
            theoretical_reachability: theo_reach,
        })
    }

    /// Boundary Relocation Probe (Phase 5):
    /// Tests whether competence on 3-bit parity tracks physical distance to a spatial boundary
    /// rather than logical task depth by evaluating an identical 3-bit parity problem shifted
    /// across all chunk positions within the sequence length.
    pub fn run_boundary_relocation_probe(
        &self,
        ticks: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<BoundaryRelocationReport> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let chunk_size = 4;
        let num_placements = seq_len / chunk_size;
        let mut query_positions = Vec::with_capacity(num_placements);
        let mut zero_padded_accuracies = Vec::with_capacity(num_placements);

        let zero_id = self.vocab.encode("0")[0] as u32;
        let one_id = self.vocab.encode("1")[0] as u32;
        let query_id = self.vocab.encode("?")[0] as u32;

        let cfg = LatentConfig::default();
        let intervention = InterventionConfig::default();

        let all_bits = [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ];
        let repeat = 16;
        let b_zp = all_bits.len() * repeat;

        // Condition 1: Zero-padded background (all non-target chunks are '000?')
        for p in 0..num_placements {
            let base = p * chunk_size;
            let q_pos = base + 3;
            query_positions.push(q_pos);

            let mut batch_inputs = vec![vec![zero_id; seq_len]; b_zp];
            for row in 0..b_zp {
                for c in 0..num_placements {
                    batch_inputs[row][c * chunk_size + 3] = query_id;
                }
            }
            let mut batch_targets = vec![0u32; b_zp];

            let mut row = 0;
            for _ in 0..repeat {
                for bits in &all_bits {
                    let parity = bits[0] ^ bits[1] ^ bits[2];
                    for j in 0..3 {
                        batch_inputs[row][base + j] = if bits[j] == 1 { one_id } else { zero_id };
                    }
                    batch_targets[row] = if parity == 1 { one_id } else { zero_id };
                    row += 1;
                }
            }

            let input_tensor = Tensor::from_vec(
                batch_inputs.into_iter().flatten().collect(),
                (b_zp, seq_len),
                device,
            )?;

            let seed = self.interface.embed_tokens(&input_tensor)?;
            let mut field = MorphogenicField::from_tensor(seed, &self.nca.field_cfg);
            let origin = field.x.clone();

            for t in 0..ticks {
                let (next_f, _, _) = self.step_latent(&field, t, None, &cfg, &intervention, &origin, None, device)?;
                field = next_f;
            }

            let logits = self.interface.logits(&field.x)?;
            let preds = logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;

            let mut correct = 0;
            for i in 0..b_zp {
                if preds[i][q_pos] == batch_targets[i] {
                    correct += 1;
                }
            }
            zero_padded_accuracies.push(correct as f32 / b_zp as f32);
        }

        // Condition 2 & 3: Random context background (all chunks have uniform random bits and '?')
        let b_rand = 256;
        let mut rng = StdRng::seed_from_u64(42);
        let mut rand_inputs = vec![vec![zero_id; seq_len]; b_rand];
        let mut local_targets = vec![vec![0u32; num_placements]; b_rand];
        let mut cum_targets = vec![vec![0u32; num_placements]; b_rand];

        for row in 0..b_rand {
            let mut cum_p = 0u32;
            for c in 0..num_placements {
                let c_base = c * chunk_size;
                let mut loc_p = 0u32;
                for j in 0..3 {
                    let bit = rng.gen_range(0..2);
                    rand_inputs[row][c_base + j] = if bit == 1 { one_id } else { zero_id };
                    loc_p ^= bit;
                    cum_p ^= bit;
                }
                rand_inputs[row][c_base + 3] = query_id;
                local_targets[row][c] = if loc_p == 1 { one_id } else { zero_id };
                cum_targets[row][c] = if cum_p == 1 { one_id } else { zero_id };
            }
        }

        let rand_tensor = Tensor::from_vec(
            rand_inputs.into_iter().flatten().collect(),
            (b_rand, seq_len),
            device,
        )?;
        let rand_seed = self.interface.embed_tokens(&rand_tensor)?;
        let mut rand_field = MorphogenicField::from_tensor(rand_seed, &self.nca.field_cfg);
        let rand_origin = rand_field.x.clone();

        for t in 0..ticks {
            let (next_f, _, _) = self.step_latent(&rand_field, t, None, &cfg, &intervention, &rand_origin, None, device)?;
            rand_field = next_f;
        }

        let rand_logits = self.interface.logits(&rand_field.x)?;
        let rand_preds = rand_logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;

        let mut random_context_local_accuracies = Vec::with_capacity(num_placements);
        let mut random_context_cumulative_accuracies = Vec::with_capacity(num_placements);

        for c in 0..num_placements {
            let q_pos = c * chunk_size + 3;
            let mut loc_corr = 0;
            let mut cum_corr = 0;
            for row in 0..b_rand {
                if rand_preds[row][q_pos] == local_targets[row][c] {
                    loc_corr += 1;
                }
                if rand_preds[row][q_pos] == cum_targets[row][c] {
                    cum_corr += 1;
                }
            }
            random_context_local_accuracies.push(loc_corr as f32 / b_rand as f32);
            random_context_cumulative_accuracies.push(cum_corr as f32 / b_rand as f32);
        }

        Ok(BoundaryRelocationReport {
            task_name: "boundary-relocation-3bit-parity".to_string(),
            seq_len,
            ticks,
            placement_positions: query_positions,
            zero_padded_accuracies,
            random_context_local_accuracies,
            random_context_cumulative_accuracies,
        })
    }

    /// Latent Representation Probe:
    /// Trains linear and non-linear (2-layer MLP) probes on the 64-dimensional latent state
    /// at each query cell q_k to determine whether prefix parity is encoded in the latent space
    /// but unread by the projection head (readout failure), or completely absent (transport failure).
    pub fn run_latent_representation_probe(
        &self,
        ticks: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<LatentRepresentationProbeReport> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let chunk_size = 4;
        let num_slots = seq_len / chunk_size;
        let channels = self.nca.field_cfg.channels;

        let zero_id = self.vocab.encode("0")[0] as u32;
        let one_id = self.vocab.encode("1")[0] as u32;
        let query_id = self.vocab.encode("?")[0] as u32;

        let cfg = LatentConfig::default();
        let intervention = InterventionConfig::default();

        let generate_data = |b: usize, seed: u64| -> Result<(Tensor, Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<u32>)> {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut inputs = vec![vec![zero_id; seq_len]; b];
            let mut c0_parity = vec![0u32; b];
            let mut local_parity = vec![vec![0u32; num_slots]; b];
            let mut cumul_parity = vec![vec![0u32; num_slots]; b];

            for row in 0..b {
                let mut cum_p = 0u32;
                for c in 0..num_slots {
                    let c_base = c * chunk_size;
                    let mut loc_p = 0u32;
                    for j in 0..3 {
                        let bit = rng.gen_range(0..2);
                        inputs[row][c_base + j] = if bit == 1 { one_id } else { zero_id };
                        loc_p ^= bit;
                        cum_p ^= bit;
                    }
                    inputs[row][c_base + 3] = query_id;
                    local_parity[row][c] = loc_p;
                    cumul_parity[row][c] = cum_p;
                    if c == 0 {
                        c0_parity[row] = loc_p;
                    }
                }
            }

            let input_tensor = Tensor::from_vec(
                inputs.into_iter().flatten().collect(),
                (b, seq_len),
                device,
            )?;
            Ok((input_tensor, local_parity, cumul_parity, c0_parity))
        };

        let (train_inputs, train_loc, train_cum, train_c0) = generate_data(256, 42)?;
        let (val_inputs, val_loc, val_cum, val_c0) = generate_data(128, 999)?;

        let forward_to_states = |inputs: &Tensor| -> Result<Tensor> {
            let seed = self.interface.embed_tokens(inputs)?;
            let mut field = MorphogenicField::from_tensor(seed, &self.nca.field_cfg);
            let origin = field.x.clone();
            for t in 0..ticks {
                let (next_f, _, _) = self.step_latent(&field, t, None, &cfg, &intervention, &origin, None, device)?;
                field = next_f;
            }
            Ok(field.x)
        };

        let train_x = forward_to_states(&train_inputs)?; // [256, L, C]
        let val_x = forward_to_states(&val_inputs)?;     // [128, L, C]

        let mut slot_indices = Vec::with_capacity(num_slots);
        let mut lin_c0_acc = Vec::with_capacity(num_slots);
        let mut lin_loc_acc = Vec::with_capacity(num_slots);
        let mut lin_cum_acc = Vec::with_capacity(num_slots);
        let mut mlp_c0_acc = Vec::with_capacity(num_slots);
        let mut mlp_loc_acc = Vec::with_capacity(num_slots);
        let mut mlp_cum_acc = Vec::with_capacity(num_slots);

        let train_x_vec = train_x.to_vec3::<f32>()?;
        let val_x_vec = val_x.to_vec3::<f32>()?;

        // Closed-form Ridge linear regression: (X^T * X + lambda * I) w = X^T * y
        let solve_ridge_probe = |x_tr: &[Vec<f32>], y_tr: &[u32], x_va: &[Vec<f32>], y_va: &[u32], d_in: usize| -> f32 {
            let d = d_in + 1; // + 1 for bias
            let n_tr = x_tr.len();
            let n_va = y_va.len();
            let lambda = 1e-2f32;

            let mut a = vec![vec![0.0f32; d]; d];
            let mut b = vec![0.0f32; d];

            for row in 0..n_tr {
                let y_val = if y_tr[row] == 1 { 1.0f32 } else { -1.0f32 };
                let feat = &x_tr[row];
                for i in 0..d {
                    let xi = if i < d_in { feat[i] } else { 1.0 };
                    b[i] += xi * y_val;
                    for j in 0..d {
                        let xj = if j < d_in { feat[j] } else { 1.0 };
                        a[i][j] += xi * xj;
                    }
                }
            }

            for i in 0..d {
                a[i][i] += lambda;
            }

            for i in 0..d {
                let mut pivot = i;
                for k in (i + 1)..d {
                    if a[k][i].abs() > a[pivot][i].abs() {
                        pivot = k;
                    }
                }
                a.swap(i, pivot);
                b.swap(i, pivot);

                let diag = a[i][i];
                if diag.abs() > 1e-9 {
                    for k in (i + 1)..d {
                        let factor = a[k][i] / diag;
                        for j in i..d {
                            a[k][j] -= factor * a[i][j];
                        }
                        b[k] -= factor * b[i];
                    }
                }
            }

            let mut w = vec![0.0f32; d];
            for i in (0..d).rev() {
                let mut sum = b[i];
                for j in (i + 1)..d {
                    sum -= a[i][j] * w[j];
                }
                let diag = a[i][i];
                w[i] = if diag.abs() > 1e-9 { sum / diag } else { 0.0 };
            }

            let mut correct = 0usize;
            for row in 0..n_va {
                let feat = &x_va[row];
                let mut dot = w[d_in]; // bias
                for i in 0..d_in {
                    dot += w[i] * feat[i];
                }
                let pred = if dot >= 0.0 { 1u32 } else { 0u32 };
                if pred == y_va[row] {
                    correct += 1;
                }
            }

            correct as f32 / n_va as f32
        };

        // Fixed random projection matrix for non-linear extreme learning machine (ELM) probe
        let d_rf = 64;
        let mut rng_rf = StdRng::seed_from_u64(1337);
        let mut w_rf = vec![vec![0.0f32; d_rf]; channels];
        for i in 0..channels {
            for j in 0..d_rf {
                w_rf[i][j] = rng_rf.gen_range(-1.0f32..1.0f32) / (channels as f32).sqrt();
            }
        }
        let project_rf = |x_raw: &[Vec<f32>]| -> Vec<Vec<f32>> {
            x_raw.iter().map(|row| {
                let mut h = vec![0.0f32; d_rf];
                for j in 0..d_rf {
                    let mut dot = 0.0f32;
                    for i in 0..channels {
                        dot += row[i] * w_rf[i][j];
                    }
                    let x = dot;
                    let inner = 0.7978846 * (x + 0.044715 * x * x * x);
                    h[j] = 0.5 * x * (1.0 + inner.tanh());
                }
                h
            }).collect()
        };

        for c in 0..num_slots {
            let q_pos = c * chunk_size + 3;
            slot_indices.push(q_pos);

            let x_tr_slot: Vec<Vec<f32>> = train_x_vec.iter().map(|row| row[q_pos].clone()).collect();
            let x_va_slot: Vec<Vec<f32>> = val_x_vec.iter().map(|row| row[q_pos].clone()).collect();

            let y_tr_loc: Vec<u32> = train_loc.iter().map(|row| row[c]).collect();
            let y_va_loc: Vec<u32> = val_loc.iter().map(|row| row[c]).collect();

            let y_tr_cum: Vec<u32> = train_cum.iter().map(|row| row[c]).collect();
            let y_va_cum: Vec<u32> = val_cum.iter().map(|row| row[c]).collect();

            let y_tr_c0 = &train_c0;
            let y_va_c0 = &val_c0;

            let x_tr_rf = project_rf(&x_tr_slot);
            let x_va_rf = project_rf(&x_va_slot);

            lin_c0_acc.push(solve_ridge_probe(&x_tr_slot, y_tr_c0, &x_va_slot, y_va_c0, channels));
            lin_loc_acc.push(solve_ridge_probe(&x_tr_slot, &y_tr_loc, &x_va_slot, &y_va_loc, channels));
            lin_cum_acc.push(solve_ridge_probe(&x_tr_slot, &y_tr_cum, &x_va_slot, &y_va_cum, channels));

            mlp_c0_acc.push(solve_ridge_probe(&x_tr_rf, y_tr_c0, &x_va_rf, y_va_c0, d_rf));
            mlp_loc_acc.push(solve_ridge_probe(&x_tr_rf, &y_tr_loc, &x_va_rf, &y_va_loc, d_rf));
            mlp_cum_acc.push(solve_ridge_probe(&x_tr_rf, &y_tr_cum, &x_va_rf, &y_va_cum, d_rf));
        }

        Ok(LatentRepresentationProbeReport {
            slot_indices,
            linear_chunk0_parity_acc: lin_c0_acc,
            linear_local_parity_acc: lin_loc_acc,
            linear_cumulative_parity_acc: lin_cum_acc,
            mlp_chunk0_parity_acc: mlp_c0_acc,
            mlp_local_parity_acc: mlp_loc_acc,
            mlp_cumulative_parity_acc: mlp_cum_acc,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalInfluenceReport {
    pub task_name: String,
    pub seq_len: usize,
    pub budgets: Vec<usize>,
    pub query_slots: Vec<usize>,
    pub query_slot_indices: Vec<usize>,
    pub numerical_influence: Vec<Vec<Vec<f32>>>,
    pub flip_probability: Vec<Vec<Vec<f32>>>,
    pub theoretical_reachability: Vec<Vec<Vec<bool>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoundaryRelocationReport {
    pub task_name: String,
    pub seq_len: usize,
    pub ticks: usize,
    pub placement_positions: Vec<usize>,
    pub zero_padded_accuracies: Vec<f32>,
    pub random_context_local_accuracies: Vec<f32>,
    pub random_context_cumulative_accuracies: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatentRepresentationProbeReport {
    pub slot_indices: Vec<usize>,
    pub linear_chunk0_parity_acc: Vec<f32>,
    pub linear_local_parity_acc: Vec<f32>,
    pub linear_cumulative_parity_acc: Vec<f32>,
    pub mlp_chunk0_parity_acc: Vec<f32>,
    pub mlp_local_parity_acc: Vec<f32>,
    pub mlp_cumulative_parity_acc: Vec<f32>,
}

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum.max(1e-12)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use crate::config::{FieldConfig, NcaConfig};

    #[test]
    fn test_latent_executor_independent_depth_and_association() -> Result<()> {
        let dev = Device::Cpu;
        let nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 16, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let vocab = Vocab::new_ascii();
        let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let interface = TokenInterface::new(vb.pp("interface"), vocab.size(), field_cfg.channels)?;
        let executor = LatentExecutor::new(&nca, &interface, &vocab);

        // 1. Independent depth controls: 4 tokens x 3 latent ticks
        let mut latent_cfg = LatentConfig::default();
        latent_cfg.latent_ticks_per_token = 3;
        latent_cfg.post_input_ticks = 2;

        let intervention = InterventionConfig::default();
        let tokens = vec![vocab.encode("test")[0], vocab.encode("test")[1]];

        let res = executor.execute_sequence(&tokens, &latent_cfg, &intervention, &dev)?;
        // 2 tokens * 3 ticks + 2 post-input ticks = 8 ticks
        assert_eq!(res.total_latent_ticks, 8);
        assert_eq!(res.trace.ticks.len(), 8);
        assert_eq!(res.final_predicted_tokens.len(), field_cfg.seq_len);

        // 2. Dynamic association probe
        let assoc_rep = executor.run_dynamic_association_probe(
            'A',
            'B',
            6,
            &latent_cfg,
            &intervention,
            &dev,
        )?;
        assert_eq!(assoc_rep.probes.len(), 6);
        assert_eq!(assoc_rep.stimulus_concept, "A");
        assert_eq!(assoc_rep.target_concept, "B");

        // 3. Attractor basin probe
        let basin_rep = executor.run_attractor_basin_probe(
            "context_A",
            "context_B",
            4,
            &latent_cfg,
            &intervention,
            &dev,
        )?;
        assert!(!basin_rep.state_trajectories_basin_a.is_empty());
        assert!(!basin_rep.state_trajectories_basin_b.is_empty());

        // 4. Memory as dynamics experiment
        let mem_rep = executor.run_memory_as_dynamics_experiment(
            "event_alpha",
            "probe_beta",
            4,
            &intervention,
            &dev,
        )?;
        assert!(!mem_rep.description.is_empty());

        Ok(())
    }

    #[test]
    fn test_scientific_hygiene_gradients_through_recurrent_paths() -> Result<()> {
        let dev = Device::Cpu;
        let nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 8, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let interface = TokenInterface::new(vb.pp("interface"), 20, field_cfg.channels)?;

        let tokens = Tensor::zeros((1, 8), DType::U32, &dev)?;
        let seed = interface.embed_tokens(&tokens)?;
        let mut field = MorphogenicField::from_tensor(seed, &field_cfg);
        for _ in 0..4 {
            let (next_f, _) = nca.step(&field, &dev)?;
            field = next_f;
        }

        let logits = interface.logits(&field.x)?;
        let targets = Tensor::zeros((1, 8), DType::U32, &dev)?;
        let loss = interface.cross_entropy_loss(&logits, &targets)?;

        let grads = loss.backward()?;
        // Verify EVERY variable has non-zero gradient
        for var in varmap.all_vars() {
            let grad = grads.get(&var).expect("Variable must have gradient through recurrence");
            let norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            assert!(norm > 0.0, "Gradient norm for {:?} must be non-zero", var);
        }

        Ok(())
    }

    #[test]
    fn test_scientific_hygiene_recurrent_depth_changes_computation() -> Result<()> {
        let dev = Device::Cpu;
        let nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 16, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let initial = MorphogenicField::from_tensor(
            Tensor::randn(0.0f32, 1.0f32, (1, 16, 8), &dev)?,
            &field_cfg,
        );

        let state_1 = nca.develop(&initial, 1, &dev)?;
        let state_2 = nca.develop(&initial, 2, &dev)?;
        let state_8 = nca.develop(&initial, 8, &dev)?;

        let dist_1_2 = state_1.l2_distance(&state_2)?;
        let dist_2_8 = state_2.l2_distance(&state_8)?;

        assert!(dist_1_2 > 1e-4, "Depth 2 must differ from depth 1");
        assert!(dist_2_8 > 1e-4, "Depth 8 must differ from depth 2");

        Ok(())
    }

    #[test]
    fn test_scientific_hygiene_observation_does_not_mutate_state() -> Result<()> {
        let dev = Device::Cpu;
        let field_cfg = FieldConfig { seq_len: 8, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let interface = TokenInterface::new(vb.pp("interface"), 10, field_cfg.channels)?;

        let field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let initial_vec = field.x.to_vec3::<f32>()?;

        // Perform multiple observations / readouts
        let _l1 = interface.logits(&field.x)?;
        let _l2 = interface.logits(&field.x)?;

        let final_vec = field.x.to_vec3::<f32>()?;
        assert_eq!(initial_vec, final_vec, "Observation must not mutate field state");

        Ok(())
    }
}
