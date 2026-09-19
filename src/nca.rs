use crate::config::{FieldConfig, NcaConfig};
use crate::field::MorphogenicField;
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Clone)]
pub struct NeuralCellularAutomaton {
    pub dense1: Linear,
    pub dense_delta: Linear,
    pub dense_gate: Linear,
    pub macro_dense1: Option<Linear>,
    pub macro_dense_delta: Option<Linear>,
    pub macro_inject: Option<Linear>,
    pub nca_cfg: NcaConfig,
    pub field_cfg: FieldConfig,
}

impl NeuralCellularAutomaton {
    /// Creates a clone of this NCA with an altered Navier-Stokes viscosity parameter
    pub fn with_viscosity(&self, viscosity: f32) -> Self {
        let mut cfg = self.nca_cfg.clone();
        cfg.viscosity = viscosity;
        Self {
            dense1: self.dense1.clone(),
            dense_delta: self.dense_delta.clone(),
            dense_gate: self.dense_gate.clone(),
            macro_dense1: self.macro_dense1.clone(),
            macro_dense_delta: self.macro_dense_delta.clone(),
            macro_inject: self.macro_inject.clone(),
            nca_cfg: cfg,
            field_cfg: self.field_cfg.clone(),
        }
    }
    pub fn new(vb: VarBuilder, nca_cfg: &NcaConfig, field_cfg: &FieldConfig) -> Result<Self> {
        // Local 1D perception: [Identity (C), Gradient (C), Laplacian (C)] = 3 * C
        // Macro hierarchy adds local coarsened modulation channel only under legacy perception coupling: + C_M
        // Macro recursive feedback adds global collective state: + C = 4 * C
        let is_hierarchy = nca_cfg.is_hierarchy();
        let macro_channels = if is_hierarchy { nca_cfg.macro_channels.max(1) } else { 0 };
        let is_perception_coupling = is_hierarchy && nca_cfg.macro_coupling == "perception";
        let mut in_channels = if is_hierarchy {
            if is_perception_coupling {
                field_cfg.channels * 3 + macro_channels
            } else {
                field_cfg.channels * 3
            }
        } else if nca_cfg.has_feedback() {
            field_cfg.channels * 4
        } else {
            field_cfg.channels * 3
        };
        if nca_cfg.coord_channel {
            in_channels += 1;
        }

        let dense1 = linear(in_channels, nca_cfg.hidden_dim, vb.pp("dense1"))?;
        let dense_delta = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_delta"))?;
        let dense_gate = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_gate"))?;

        let (macro_dense1, macro_dense_delta, macro_inject) = if is_hierarchy {
            let micro_pool_dim = if nca_cfg.macro_downsampler == "walsh" && nca_cfg.macro_stride == 2 {
                field_cfg.channels * 2
            } else {
                field_cfg.channels
            };
            let m_in_channels = macro_channels * 3 + micro_pool_dim;
            let m_hidden = (nca_cfg.hidden_dim / 2).max(16);
            let md1 = linear(m_in_channels, m_hidden, vb.pp("macro_dense1"))?;
            let mdd = linear(m_hidden, macro_channels, vb.pp("macro_dense_delta"))?;
            let minj = if nca_cfg.macro_coupling == "state_derivative" {
                Some(linear(macro_channels, field_cfg.channels, vb.pp("macro_inject"))?)
            } else {
                None
            };
            (Some(md1), Some(mdd), minj)
        } else {
            (None, None, None)
        };

        Ok(Self {
            dense1,
            dense_delta,
            dense_gate,
            macro_dense1,
            macro_dense_delta,
            macro_inject,
            nca_cfg: nca_cfg.clone(),
            field_cfg: field_cfg.clone(),
        })
    }

    /// Parameter count breakdown across recurrent subsystems
    pub fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let p_dense1 = self.dense1.weight().elem_count() + self.dense1.bias().map_or(0, |b| b.elem_count());
        let p_delta = self.dense_delta.weight().elem_count() + self.dense_delta.bias().map_or(0, |b| b.elem_count());
        let p_gate = self.dense_gate.weight().elem_count() + self.dense_gate.bias().map_or(0, |b| b.elem_count());

        let mut breakdown = vec![
            ("spatial_perception_stencil".to_string(), 0), // fixed differential operator
            ("hidden_feature_mlp_dense1".to_string(), p_dense1),
            ("directional_update_dense_delta".to_string(), p_delta),
            ("adaptive_gate_dense_gate".to_string(), p_gate),
            ("viscous_dissipation_damping".to_string(), 0), // physical continuum operator
        ];
        if self.nca_cfg.is_hierarchy() {
            if let Some(ref md1) = self.macro_dense1 {
                let p = md1.weight().elem_count() + md1.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_hierarchy_mlp_dense1".to_string(), p));
            }
            if let Some(ref mdd) = self.macro_dense_delta {
                let p = mdd.weight().elem_count() + mdd.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_hierarchy_directional_delta".to_string(), p));
            }
            if let Some(ref minj) = self.macro_inject {
                let p = minj.weight().elem_count() + minj.bias().map_or(0, |b| b.elem_count());
                breakdown.push(("macro_state_derivative_inject".to_string(), p));
            }
            breakdown.push(("macro_local_highpass_modulation".to_string(), 0));
        } else if self.nca_cfg.has_feedback() {
            breakdown.push(("macro_recursive_feedback_loop".to_string(), 0));
        }
        if self.nca_cfg.coord_channel {
            breakdown.push(("spatial_coordinate_channel".to_string(), 0));
        }
        breakdown
    }

    /// Generates static 1D spatial coordinate channel p_i in [-1, 1] for B sequences of length L
    pub fn generate_coordinates(b: usize, l: usize, mode: Option<&str>, device: &Device) -> Result<Tensor> {
        let mut coords = Vec::with_capacity(l);
        if l <= 1 {
            coords.push(0.0f32);
        } else {
            for i in 0..l {
                let p = 2.0 * (i as f32) / ((l - 1) as f32) - 1.0;
                coords.push(p);
            }
        }

        match mode.unwrap_or("intact") {
            "zeroed" => {
                for c in coords.iter_mut() {
                    *c = 0.0;
                }
            }
            "constant" => {
                for c in coords.iter_mut() {
                    *c = 0.5;
                }
            }
            "reversed" => {
                coords.reverse();
            }
            "shuffled" => {
                use rand::seq::SliceRandom;
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                coords.shuffle(&mut rng);
            }
            "intact" | _ => {}
        }

        let mut batch_coords = Vec::with_capacity(b * l);
        for _ in 0..b {
            batch_coords.extend_from_slice(&coords);
        }
        Ok(Tensor::from_vec(batch_coords, (b, l, 1), device)?)
    }

    /// Circular spatial shift: rolls tensor [B, L, C] along spatial dimension L
    fn roll_spatial(tensor: &Tensor, shift: i32) -> Result<Tensor> {
        let (_, l, _) = tensor.dims3()?;
        let shift = ((shift % l as i32) + l as i32) as usize % l;
        if shift == 0 {
            return Ok(tensor.clone());
        }

        let left_part = tensor.narrow(1, l - shift, shift)?;
        let right_part = tensor.narrow(1, 0, l - shift)?;
        Ok(Tensor::cat(&[&left_part, &right_part], 1)?)
    }

    /// Computes spatial perception: [x, grad, laplacian] and optional macroscopic slow feedback [s]
    pub fn perceive_with_feedback(&self, x: &Tensor, slow_state: Option<&Tensor>) -> Result<Tensor> {
        let (left, right) = self.neighbors(x)?;

        // 1st spatial derivative: (right - left) / 2
        let grad = ((&right - &left)? / 2.0)?;

        // 2nd spatial derivative (Laplacian): right - 2 * x + left
        let double_x = (x * 2.0)?;
        let diff = (&right - &double_x)?;
        let laplacian = (&diff + &left)?;

        let base_perc = if self.nca_cfg.is_hierarchy() {
            if self.nca_cfg.macro_coupling == "perception" {
                let (b, l, _) = x.dims3()?;
                let s = self.nca_cfg.macro_stride.max(1);
                let lm = (l / s).max(1);
                let cm = self.nca_cfg.macro_channels.max(1);

                let m = match slow_state {
                    Some(s_ten) if s_ten.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => s_ten.clone(),
                    _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, x.device())?,
                };

                // Local discrete high-pass / difference stencil on macro grid:
                // w_macro_j = M_j - 0.5 * (M_{j-1} + M_{j+1})
                // Zero DC response guaranteed by construction!
                let (m_left, m_right) = self.neighbors(&m)?;
                let lap_avg = ((&m_left + &m_right)? * 0.5)?;
                let w_macro = (&m - &lap_avg)?;

                // Nearest-neighbor upsampling to micro resolution:
                let w = if s > 1 {
                    w_macro.unsqueeze(2)?.repeat((1, 1, s, 1))?.reshape((b, lm * s, cm))?
                } else {
                    w_macro.clone()
                };
                let w = if w.dim(1)? > l {
                    w.narrow(1, 0, l)?
                } else {
                    w
                };

                // Hard-bounded coupling gain: gamma in [0.001, 0.10]
                let gamma = (self.nca_cfg.feedback_weight as f64).clamp(0.001, 0.10);
                let scaled_w = (w * gamma)?;

                Tensor::cat(&[x, &grad, &laplacian, &scaled_w], 2)?
            } else {
                // In state_derivative coupling mode, micro perception is strictly local
                Tensor::cat(&[x, &grad, &laplacian], 2)?
            }
        } else if self.nca_cfg.has_feedback() {
            let (_, l, _) = x.dims3()?;
            let s = match slow_state {
                Some(s_ten) => s_ten.clone(),
                None => x.mean(1)?.unsqueeze(1)?,
            };
            let s_broadcast = s.repeat((1, l, 1))?;
            Tensor::cat(&[x, &grad, &laplacian, &s_broadcast], 2)?
        } else {
            // Concatenate along channels: [B, L, 3 * C]
            Tensor::cat(&[x, &grad, &laplacian], 2)?
        };

        if self.nca_cfg.coord_channel {
            let (b, l, _) = x.dims3()?;
            let coords = Self::generate_coordinates(b, l, None, x.device())?;
            Ok(Tensor::cat(&[&base_perc, &coords], 2)?)
        } else {
            Ok(base_perc)
        }
    }

    /// Computes local 1D perception: Identity, 1st Derivative (Gradient), 2nd Derivative (Laplacian)
    pub fn perceive(&self, x: &Tensor) -> Result<Tensor> {
        self.perceive_with_feedback(x, None)
    }

    /// Zero boundaries prevent end-to-start shortcuts when requested.
    fn neighbors(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.field_cfg.periodic_boundary {
            return Ok((Self::roll_spatial(x, 1)?, Self::roll_spatial(x, -1)?));
        }
        let (b, l, c) = x.dims3()?;
        let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        if l == 1 { return Ok((zero.clone(), zero)); }
        Ok((Tensor::cat(&[&zero, &x.narrow(1, 0, l - 1)?], 1)?,
            Tensor::cat(&[&x.narrow(1, 1, l - 1)?, &zero], 1)?))
    }

    pub fn dissipate(&self, tensor: &Tensor, effective_diff: f32) -> Result<Tensor> {
        if self.field_cfg.periodic_boundary {
            return Self::apply_viscous_dissipation(tensor, effective_diff);
        }
        if effective_diff <= 0.0 { return Ok(tensor.clone()); }
        let n = (effective_diff / 0.25).ceil().max(1.0) as usize;
        let mut x = tensor.clone();
        for _ in 0..n {
            let (left, right) = self.neighbors(&x)?;
            let lap = ((left + right)? - (&x * 2.0)?)?;
            x = (&x + (lap * (effective_diff as f64 / n as f64))?)?;
        }
        Ok(x)
    }

    /// Applies physical Navier-Stokes viscous dissipation (\nu * \Delta x * \Delta t)
    /// with adaptive substepping to guarantee unconditional von Neumann stability:
    /// For 1D central Laplacian stencil [1, -2, 1], maximum stable substep is D_max = 0.25.
    /// If total diffusion weight D = \nu * \alpha > 0.25, substepping partitions D into N = ceil(D / 0.25)
    /// substeps of size D / N <= 0.25. This ensures strictly dissipative, monotonic damping
    /// without artificial high-frequency numerical oscillations or explosions for any \nu >= 0.
    pub fn apply_viscous_dissipation(tensor: &Tensor, effective_diff: f32) -> Result<Tensor> {
        if effective_diff <= 0.0 {
            return Ok(tensor.clone());
        }
        let max_substep = 0.25f32;
        let num_substeps = (effective_diff / max_substep).ceil().max(1.0) as usize;
        let dt_sub = (effective_diff / num_substeps as f32) as f64;

        let mut current = tensor.clone();
        for _ in 0..num_substeps {
            let left = MorphogenicField::roll_spatial(&current, 1)?;
            let right = MorphogenicField::roll_spatial(&current, -1)?;
            let double = (&current * 2.0)?;
            let lap = (&right - &double)?;
            let lap = (&lap + &left)?;
            let damping = (lap * dt_sub)?;
            current = (&current + &damping)?;
        }
        Ok(current)
    }

    /// Projects cell states onto a compact invariant manifold:
    /// - "rms": Zero-parameter RMS normalization along channel dimension
    /// - "bounded": Smoothly clamps RMS amplitude above bound_threshold while leaving normal dynamics untouched
    /// - "layer_norm": Zero-mean, unit-variance LayerNorm along channel dimension
    #[allow(dead_code)]
    pub fn apply_state_norm(tensor: &Tensor, mode: &str) -> Result<Tensor> {
        Self::apply_state_norm_with_threshold(tensor, mode, 1.5)
    }

    pub fn apply_state_norm_with_threshold(tensor: &Tensor, mode: &str, threshold: f32) -> Result<Tensor> {
        match mode {
            "rms" => {
                let sq = tensor.sqr()?;
                let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                Ok(tensor.broadcast_div(&rms)?)
            }
            "bounded" => {
                let sq = tensor.sqr()?;
                let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                let th = (threshold as f64).max(0.1);
                let ratio = (rms / th)?;
                let ones = ratio.ones_like()?;
                let scale = ratio.maximum(&ones)?;
                Ok(tensor.broadcast_div(&scale)?)
            }
            "layer_norm" => {
                let mean = tensor.mean_keepdim(candle_core::D::Minus1)?;
                let centered = tensor.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let std = (var + 1e-5)?.sqrt()?;
                Ok(centered.broadcast_div(&std)?)
            }
            _ => Ok(tensor.clone()),
        }
    }

    /// Single developmental step with optional external forcing:
    /// \partial_t x = N_\theta(x) + \nu \Delta x + f(s, t)
    /// Discretized with time step \Delta t = \alpha = step_size
    /// Returns (new_field, update_norm)
    pub fn step_with_forcing(
        &self,
        field: &MorphogenicField,
        forcing: Option<&Tensor>,
        device: &Device,
    ) -> Result<(MorphogenicField, f32)> {
        let x = &field.x;

        // 1. Spatial perception with optional recursive macro feedback
        let perception = self.perceive_with_feedback(x, field.slow_state.as_ref())?;

        // 2. Hidden feature extraction
        let h1 = self.dense1.forward(&perception)?;
        let h1_act = match self.nca_cfg.activation.as_str() {
            "tanh" => h1.tanh()?,
            _ => candle_nn::Activation::Gelu.forward(&h1)?,
        };

        // 3. Gated residual update: delta in (-1, 1), gate in (0, 1)
        let delta = self.dense_delta.forward(&h1_act)?.tanh()?;
        let gate = candle_nn::ops::sigmoid(&self.dense_gate.forward(&h1_act)?)?;
        let gated_delta = delta.mul(&gate)?;

        // 4. Optional stochastic update mask (for asynchronous cellular dynamics)
        let effective_delta = if self.nca_cfg.update_rate < 0.999 {
            let (b, l, c) = x.dims3()?;
            let mask_vals: Vec<f32> = (0..(b * l))
                .map(|_| if rand::random::<f32>() < self.nca_cfg.update_rate { 1.0 } else { 0.0 })
                .collect();
            let mask = Tensor::from_slice(&mask_vals, (b, l, 1), device)?
                .repeat((1, 1, c))?;
            gated_delta.mul(&mask)?
        } else {
            gated_delta
        };

        // 5. Residual active drift integration with optional damping and leaky contraction:
        // x_(t+1/2) = (1 - lambda) * x_t + alpha * damping * (\delta + f)
        let alpha = (self.nca_cfg.step_size * self.nca_cfg.damping_alpha) as f64;
        let scaled_delta = (effective_delta * alpha)?;
        let base_x = if self.nca_cfg.leaky_lambda > 0.0 {
            (x * (1.0 - self.nca_cfg.leaky_lambda.clamp(0.0, 0.99) as f64))?
        } else {
            x.clone()
        };
        let mut interim_x = (&base_x + &scaled_delta)?;

        // State-derivative macro injection (Track 1 / Path A):
        if self.nca_cfg.is_hierarchy() && self.nca_cfg.macro_coupling == "state_derivative" {
            if let Some(ref minj) = self.macro_inject {
                let (b, l, _) = x.dims3()?;
                let s = self.nca_cfg.macro_stride.max(1);
                let lm = (l / s).max(1);
                let cm = self.nca_cfg.macro_channels.max(1);

                let m = match &field.slow_state {
                    Some(s_ten) if s_ten.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => s_ten.clone(),
                    _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, device)?,
                };

                let (m_left, m_right) = self.neighbors(&m)?;
                let lap_avg = ((&m_left + &m_right)? * 0.5)?;
                let w_macro = (&m - &lap_avg)?;

                let w = if s > 1 {
                    w_macro.unsqueeze(2)?.repeat((1, 1, s, 1))?.reshape((b, lm * s, cm))?
                } else {
                    w_macro
                };
                let w = if w.dim(1)? > l {
                    w.narrow(1, 0, l)?
                } else {
                    w
                };

                let gamma = (self.nca_cfg.macro_gamma as f64).clamp(0.001, 1.0);
                let inj_h = minj.forward(&w)?.tanh()?;
                let one_minus_x2 = (Tensor::ones_like(x)? - x.sqr()?)?.clamp(0.0, 1.0)?;
                let gated_inj = (inj_h * gamma)?.mul(&one_minus_x2)?;
                let scaled_inj = (gated_inj * alpha)?;
                interim_x = (&interim_x + &scaled_inj)?;
            }
        }

        if let Some(f) = forcing {
            let scaled_f = (f * alpha)?;
            interim_x = (&interim_x + &scaled_f)?;
        }

        // 6. State normalization: projects onto compact invariant manifold (prevents open-phase energy explosion)
        let interim_x = if self.nca_cfg.state_norm != "none" {
            Self::apply_state_norm_with_threshold(&interim_x, &self.nca_cfg.state_norm, self.nca_cfg.bound_threshold)?
        } else {
            interim_x
        };

        // 7. Navier-Stokes physical viscous dissipation: \nu * \Delta x * \alpha
        // Scales consistently with time step alpha and employs adaptive substepping
        let new_x = if self.nca_cfg.viscosity > 0.0 {
            let effective_diff = self.nca_cfg.viscosity.max(0.0) * self.nca_cfg.step_size;
            self.dissipate(&interim_x, effective_diff)?
        } else {
            interim_x
        };

        // 8. Macro hierarchy or recursive feedback loop:
        let new_slow_state = if self.nca_cfg.is_hierarchy() {
            let (b, l, c) = new_x.dims3()?;
            let s = self.nca_cfg.macro_stride.max(1);
            let lm = (l / s).max(1);
            let cm = self.nca_cfg.macro_channels.max(1);

            let prev_m = match &field.slow_state {
                Some(st) if st.dims3().map(|d| d.1 == lm && d.2 == cm).unwrap_or(false) => st.clone(),
                _ => Tensor::zeros((b, lm, cm), candle_core::DType::F32, device)?,
            };

            let is_macro_tick = (field.tick + 1) % self.nca_cfg.macro_period.max(1) == 0;
            if is_macro_tick {
                let downsampled_micro = if self.nca_cfg.macro_downsampler == "walsh" && s == 2 && l >= lm * 2 {
                    let truncated_x = new_x.narrow(1, 0, lm * 2)?;
                    let reshaped = truncated_x.reshape((b, lm, 2, c))?;
                    let x_even = reshaped.narrow(2, 0, 1)?.squeeze(2)?;
                    let x_odd = reshaped.narrow(2, 1, 1)?.squeeze(2)?;
                    let p_sum = (&x_even + &x_odd)?;
                    let p_diff = (&x_even - &x_odd)?;
                    Tensor::cat(&[&p_sum, &p_diff], 2)?
                } else if s > 1 && l >= lm * s {
                    let truncated_x = new_x.narrow(1, 0, lm * s)?;
                    truncated_x.reshape((b, lm, s, c))?.mean(candle_core::D::Minus2)?
                } else {
                    new_x.narrow(1, 0, lm)?
                };

                if let (Some(md1), Some(mdd)) = (&self.macro_dense1, &self.macro_dense_delta) {
                    let (m_left, m_right) = self.neighbors(&prev_m)?;
                    let macro_perc = Tensor::cat(&[&m_left, &prev_m, &m_right, &downsampled_micro], 2)?;
                    let mh1 = md1.forward(&macro_perc)?.tanh()?;
                    let m_delta = mdd.forward(&mh1)?.tanh()?;
                    let m_alpha = (self.nca_cfg.step_size * 0.5) as f64;
                    let scaled_md = (m_delta * m_alpha)?;
                    let mut updated_m = (&prev_m + &scaled_md)?;

                    // Emergent bistable potential: V(w) = (lambda / 4) * (w^2 - 1)^2
                    // Restoring force: -dV/dw = lambda * w * (1 - w^2)
                    if self.nca_cfg.macro_lambda > 0.0 {
                        let lambda = (self.nca_cfg.macro_lambda as f64).clamp(0.0, 1.0);
                        let one_minus_m2 = (Tensor::ones_like(&prev_m)? - prev_m.sqr()?)?;
                        let bistable_drift = prev_m.mul(&one_minus_m2)?;
                        let scaled_bistable = (bistable_drift * (lambda * m_alpha))?;
                        updated_m = (&updated_m + &scaled_bistable)?;
                    }

                    Some(updated_m)
                } else {
                    Some(prev_m)
                }
            } else {
                Some(prev_m)
            }
        } else if self.nca_cfg.has_feedback() {
            let pool_x = new_x.mean(1)?.unsqueeze(1)?; // [B, 1, C]
            if self.nca_cfg.feedback_mode == "dual_timescale" {
                let beta = (self.nca_cfg.feedback_weight as f64).clamp(0.01, 1.0);
                match &field.slow_state {
                    Some(prev_s) => {
                        let decay = (prev_s * (1.0 - beta))?;
                        let input = (&pool_x * beta)?;
                        Some((&decay + &input)?)
                    }
                    None => Some(pool_x),
                }
            } else {
                // "global_pool" mode: instantaneous spatial mean
                Some(pool_x)
            }
        } else {
            None
        };

        // Compute displacement norm ||x_(t+1) - x_t||
        let total_displacement = (&new_x - x)?;
        let update_sq = total_displacement.sqr()?;
        let update_norm = update_sq.mean_all()?.to_scalar::<f32>()?.sqrt();

        Ok((
            MorphogenicField {
                x: new_x,
                config: self.field_cfg.clone(),
                slow_state: new_slow_state,
                tick: field.tick + 1,
            },
            update_norm,
        ))
    }

    /// Single autonomous developmental step (unforced)
    pub fn step(&self, field: &MorphogenicField, device: &Device) -> Result<(MorphogenicField, f32)> {
        self.step_with_forcing(field, None, device)
    }

    /// Single step returning just field
    pub fn step_field(&self, field: &MorphogenicField, device: &Device) -> Result<MorphogenicField> {
        let (next_field, _) = self.step(field, device)?;
        Ok(next_field)
    }

    /// Unrolls development autonomously for T steps
    pub fn develop(&self, initial: &MorphogenicField, steps: usize, device: &Device) -> Result<MorphogenicField> {
        let mut current = initial.clone();
        for _ in 0..steps {
            current = self.step_field(&current, device)?;
        }
        Ok(current)
    }

    /// Unrolls development with a sequence of external forcing tensors
    #[allow(dead_code)]
    pub fn develop_with_forcing(
        &self,
        initial: &MorphogenicField,
        forcings: &[Tensor],
        device: &Device,
    ) -> Result<MorphogenicField> {
        let mut current = initial.clone();
        for f in forcings {
            let (next, _) = self.step_with_forcing(&current, Some(f), device)?;
            current = next;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_nca_step_and_viscosity() -> Result<()> {
        let dev = Device::Cpu;
        let mut nca_cfg = NcaConfig::default();
        let field_cfg = FieldConfig { seq_len: 16, channels: 8, periodic_boundary: true };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        // 1. Without viscosity
        let nca_inviscid = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let initial = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (next_field, norm) = nca_inviscid.step(&initial, &dev)?;
        assert_eq!(next_field.x.dims3()?, (1, 16, 8));
        assert!(norm >= 0.0);

        // 2. With Navier-Stokes viscosity
        nca_cfg.viscosity = 0.05;
        let nca_viscous = NeuralCellularAutomaton::new(vb.pp("nca"), &nca_cfg, &field_cfg)?;
        let (next_viscous, _) = nca_viscous.step(&initial, &dev)?;
        assert_eq!(next_viscous.x.dims3()?, (1, 16, 8));

        // 3. Forcing test
        let forcing_t = Tensor::zeros((1, 16, 8), candle_core::DType::F32, &dev)?;
        let (forced_field, _) = nca_viscous.step_with_forcing(&initial, Some(&forcing_t), &dev)?;
        assert_eq!(forced_field.x.dims3()?, (1, 16, 8));

        Ok(())
    }

    #[test]
    fn test_viscous_damping_attenuates_high_frequencies() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 4;
        let cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // 1. Test k = L/4 mode where both central gradient and Laplacian are active
        let mut data_wave = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..seq_len {
            let val = (two_pi * 4.0 * (i as f32) / (seq_len as f32)).sin();
            for c in 0..channels {
                data_wave[0][i][c] = val;
            }
        }
        let t_wave = Tensor::new(data_wave, &dev)?;
        let field_wave = MorphogenicField::from_tensor(t_wave, &cfg);

        let init_ens = field_wave.enstrophy()?;
        let init_palin = field_wave.palinstrophy()?;
        assert!(init_ens > 0.0);
        assert!(init_palin > 0.0);

        // Damping with moderate viscosity (effective diffusion D = 0.05 * 0.5 = 0.025)
        let damped_wave = NeuralCellularAutomaton::apply_viscous_dissipation(&field_wave.x, 0.025)?;
        let field_damped = MorphogenicField::from_tensor(damped_wave, &cfg);
        assert!(field_damped.enstrophy()? < init_ens, "Viscosity must attenuate enstrophy");
        assert!(field_damped.palinstrophy()? < init_palin, "Viscosity must attenuate palinstrophy");

        // 2. Test Nyquist mode (k = L/2: +1, -1, +1, -1) under high viscosity (D = 0.30 and D = 1.0)
        // In the prior attempt without adaptive substepping, D = 0.60 exploded to 1889.5!
        let mut data_nyquist = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        for i in 0..seq_len {
            let sign = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
            for c in 0..channels {
                data_nyquist[0][i][c] = sign;
            }
        }
        let t_nyq = Tensor::new(data_nyquist, &dev)?;
        let field_nyq = MorphogenicField::from_tensor(t_nyq, &cfg);
        let init_palin_nyq = field_nyq.palinstrophy()?;
        assert!(init_palin_nyq > 0.0);

        // High viscosity D = 0.30 (nu = 0.60, alpha = 0.5)
        let damped_high = NeuralCellularAutomaton::apply_viscous_dissipation(&field_nyq.x, 0.30)?;
        let field_high = MorphogenicField::from_tensor(damped_high, &cfg);
        let palin_high = field_high.palinstrophy()?;
        assert!(palin_high < init_palin_nyq, "Higher viscosity must damp palinstrophy");
        assert!(field_high.max_amplitude()? < field_nyq.max_amplitude()?);
        assert!(!palin_high.is_nan());

        // Extreme viscosity D = 1.00 (strongly extinguished by substepping)
        let damped_extreme = NeuralCellularAutomaton::apply_viscous_dissipation(&field_nyq.x, 1.00)?;
        let field_extreme = MorphogenicField::from_tensor(damped_extreme, &cfg);
        assert!(field_extreme.palinstrophy()? < palin_high);
        assert!(field_extreme.max_amplitude()? < 0.05, "Extreme viscosity must strongly extinguish Nyquist mode");

        Ok(())
    }

    #[test]
    fn test_macro_recursive_feedback_loop_lightcone_expansion() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: false };

        // 1. Test Baseline Local NCA (no feedback)
        let nca_cfg_local = NcaConfig {
            feedback_mode: "none".to_string(),
            ..NcaConfig::default()
        };
        let varmap_local = VarMap::new();
        let vb_local = VarBuilder::from_varmap(&varmap_local, candle_core::DType::F32, &dev);
        let nca_local = NeuralCellularAutomaton::new(vb_local, &nca_cfg_local, &field_cfg)?;

        let zeros_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (step_local_zeros, _) = nca_local.step(&zeros_field, &dev)?;

        // Impulse at position 0 only: [1, 16, 8]
        let mut impulse_data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        for c in 0..channels {
            impulse_data[0][0][c] = 5.0;
        }
        let impulse_field = MorphogenicField::from_tensor(Tensor::new(impulse_data, &dev)?, &field_cfg);

        // Run 1 step of local-only NCA
        let (step_local_impulse, _) = nca_local.step(&impulse_field, &dev)?;
        // Cell 8 is distance 8 away: with zero boundary and radius 1, cell 8's causal difference MUST be exactly 0.0!
        let diff_local = (&step_local_impulse.x.narrow(1, 8, 1)? - &step_local_zeros.x.narrow(1, 8, 1)?)?;
        let causal_effect_local = diff_local.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(causal_effect_local, 0.0, "Local-only NCA cannot causally influence distant cell in 1 step");

        // 2. Test Macro Recursive Feedback NCA
        let nca_cfg_feedback = NcaConfig {
            feedback_mode: "global_pool".to_string(),
            ..NcaConfig::default()
        };
        let varmap_fb = VarMap::new();
        let vb_fb = VarBuilder::from_varmap(&varmap_fb, candle_core::DType::F32, &dev);
        let nca_fb = NeuralCellularAutomaton::new(vb_fb, &nca_cfg_feedback, &field_cfg)?;

        let (step_fb_zeros, _) = nca_fb.step(&zeros_field, &dev)?;
        let (step_fb_impulse, _) = nca_fb.step(&impulse_field, &dev)?;
        assert!(step_fb_impulse.slow_state.is_some(), "Feedback NCA must maintain macroscopic slow state");

        // In feedback NCA, position 0's impulse is pooled to macro state s_1, and perceived by all cells
        let diff_fb = (&step_fb_impulse.x.narrow(1, 8, 1)? - &step_fb_zeros.x.narrow(1, 8, 1)?)?;
        let causal_effect_fb = diff_fb.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(causal_effect_fb > 1e-6, "Feedback NCA enables instantaneous O(1) global causal communication");

        Ok(())
    }

    #[test]
    fn test_dual_timescale_leaky_contraction() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "dual_timescale".to_string(),
            feedback_weight: 0.2,
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let developed = nca.develop(&initial, 8, &dev)?;
        assert!(developed.slow_state.is_some());
        let norm = developed.x.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        assert!(norm.is_finite());

        Ok(())
    }

    #[test]
    fn test_state_norm_compact_manifold() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            state_norm: "rms".to_string(),
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        // Seed with random large tensor
        let rand_tensor = Tensor::randn(0.0f32, 10.0f32, (1, seq_len, channels), &dev)?;
        let initial = MorphogenicField::from_tensor(rand_tensor, &field_cfg);

        // Unroll 64 autonomous steps
        let rolled = nca.develop(&initial, 64, &dev)?;

        // Under RMS normalization, each cell's mean square over channels must be ~1.0
        let cell_rms = rolled.x.sqr()?.mean_keepdim(candle_core::D::Minus1)?.to_vec3::<f32>()?;
        for b in 0..1 {
            for l in 0..seq_len {
                let rms_sq = cell_rms[b][l][0];
                assert!((rms_sq - 1.0).abs() < 1e-3, "Each cell vector must lie on the RMS unit sphere");
            }
        }

        // Energy must be exactly 0.5 * mean(sq) = 0.5 * 1.0 = 0.5
        let energy = rolled.energy()?;
        assert!((energy - 0.5).abs() < 1e-3, "Field energy on compact manifold must be bounded to 0.5");

        Ok(())
    }

    #[test]
    fn test_coordinate_channel_and_counterfactuals() -> Result<()> {
        let dev = Device::Cpu;
        let b = 2;
        let l = 5;

        // 1. Coordinate generation
        let intact = NeuralCellularAutomaton::generate_coordinates(b, l, None, &dev)?;
        let intact_v = intact.to_vec3::<f32>()?;
        assert_eq!(intact_v[0][0][0], -1.0);
        assert_eq!(intact_v[0][2][0], 0.0);
        assert_eq!(intact_v[0][4][0], 1.0);

        let zeroed = NeuralCellularAutomaton::generate_coordinates(b, l, Some("zeroed"), &dev)?;
        let zeroed_v = zeroed.to_vec3::<f32>()?;
        for cell in 0..l {
            assert_eq!(zeroed_v[0][cell][0], 0.0);
        }

        let constant = NeuralCellularAutomaton::generate_coordinates(b, l, Some("constant"), &dev)?;
        let constant_v = constant.to_vec3::<f32>()?;
        for cell in 0..l {
            assert_eq!(constant_v[0][cell][0], 0.5);
        }

        let reversed = NeuralCellularAutomaton::generate_coordinates(b, l, Some("reversed"), &dev)?;
        let reversed_v = reversed.to_vec3::<f32>()?;
        assert_eq!(reversed_v[0][0][0], 1.0);
        assert_eq!(reversed_v[0][4][0], -1.0);

        // 2. NCA Perception dimension
        let channels = 8;
        let field_cfg = FieldConfig { seq_len: l, channels, periodic_boundary: true };
        let nca_cfg = NcaConfig {
            coord_channel: true,
            ..NcaConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let x = Tensor::zeros((b, l, channels), candle_core::DType::F32, &dev)?;
        let perc = nca.perceive(&x)?;
        assert_eq!(perc.dims3()?, (b, l, 3 * channels + 1));

        Ok(())
    }

    #[test]
    fn test_hierarchy_zero_dc_response() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_period: 2,
            feedback_weight: 0.1,
            macro_coupling: "perception".to_string(),
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        // Construct a constant macro field: M_j = 3.5 for all j in 0..8
        let macro_len = 8;
        let const_macro = (Tensor::ones((1, macro_len, 4), candle_core::DType::F32, &dev)? * 3.5)?;
        let zero_micro = Tensor::zeros((1, seq_len, channels), candle_core::DType::F32, &dev)?;

        // Perceive with the constant macro state
        let perc = nca.perceive_with_feedback(&zero_micro, Some(&const_macro))?;
        // Micro perception has channels: 3 * C + C_M = 3 * 8 + 4 = 28
        assert_eq!(perc.dims3()?, (1, seq_len, 28));

        // The last 4 channels correspond to scaled_w.
        // Under a spatially constant macro field, the local difference stencil
        // w_j = M_j - 0.5 * (M_{j-1} + M_{j+1}) = 3.5 - 3.5 = 0.0!
        let w_channels = perc.narrow(2, 24, 4)?;
        let w_rms = w_channels.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        assert!(
            w_rms < 1e-6,
            "Local difference stencil must produce EXACT zero response to constant DC field, got RMS {}",
            w_rms
        );

        Ok(())
    }

    #[test]
    fn test_hierarchy_step_and_slow_clock() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_period: 2, // macro updates every 2 ticks
            step_size: 0.5,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        assert_eq!(initial_field.tick, 0);

        // Step 1: field.tick goes from 0 -> 1.
        // (0 + 1) % 2 == 1 != 0, so macro clock is not active; macro state remains all zeros
        let (field_step1, _) = nca.step(&initial_field, &dev)?;
        assert_eq!(field_step1.tick, 1);
        let m1 = field_step1.slow_state.as_ref().expect("macro state should exist");
        assert_eq!(m1.dims3()?, (1, 8, 4));
        let m1_norm = m1.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(m1_norm, 0.0, "Macro state should remain frozen on non-macro ticks");

        // Step 2: field.tick goes from 1 -> 2.
        // (1 + 1) % 2 == 0, so macro clock fires! Macro field updates
        let (field_step2, _) = nca.step(&field_step1, &dev)?;
        assert_eq!(field_step2.tick, 2);
        assert!(field_step2.slow_state.is_some());

        Ok(())
    }

    #[test]
    fn test_hierarchy_degenerate_stride_control() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // Degenerate control: stride s = 1 (no spatial coarsening)
        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 1,
            macro_channels: 4,
            macro_period: 2,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let initial_field = MorphogenicField::zeros(1, &field_cfg, &dev)?;
        let (field_step1, _) = nca.step(&initial_field, &dev)?;
        let m1 = field_step1.slow_state.as_ref().expect("macro state should exist");
        // With stride 1, macro grid length is L_M = L = 16
        assert_eq!(m1.dims3()?, (1, 16, 4));

        Ok(())
    }

    #[test]
    fn test_hierarchy_parameter_breakdown() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_coupling: "state_derivative".to_string(),
            hidden_dim: 32,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let breakdown = nca.parameter_breakdown();
        let names: Vec<&str> = breakdown.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"macro_hierarchy_mlp_dense1"));
        assert!(names.contains(&"macro_hierarchy_directional_delta"));
        assert!(names.contains(&"macro_state_derivative_inject"));
        assert!(names.contains(&"macro_local_highpass_modulation"));

        Ok(())
    }

    #[test]
    fn test_hierarchy_walsh_and_state_derivative() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 16;
        let channels = 8;
        let field_cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // Test with Walsh downsampling and state-derivative coupling
        let nca_cfg = NcaConfig {
            feedback_mode: "hierarchy".to_string(),
            macro_stride: 2,
            macro_channels: 4,
            macro_downsampler: "walsh".to_string(),
            macro_coupling: "state_derivative".to_string(),
            macro_gamma: 0.2,
            macro_lambda: 0.1,
            ..NcaConfig::default()
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let nca = NeuralCellularAutomaton::new(vb, &nca_cfg, &field_cfg)?;

        let mut field = MorphogenicField::zeros(2, &field_cfg, &dev)?;
        // Run 4 steps (two macro update cycles)
        for _ in 0..4 {
            let (next_f, _) = nca.step(&field, &dev)?;
            field = next_f;
        }

        // Verify state is bounded in [-1, 1] (barrier function)
        let max_abs = field.x.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_abs <= 1.05, "State must remain bounded within barrier, got {}", max_abs);

        // Verify macro state exists and has correct dimensions [B, L/2, C_M]
        let m = field.slow_state.expect("Macro state should exist");
        assert_eq!(m.dims3()?, (2, 8, 4));

        Ok(())
    }
}
