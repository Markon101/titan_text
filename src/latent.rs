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
