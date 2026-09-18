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
            nca_cfg: cfg,
            field_cfg: self.field_cfg.clone(),
        }
    }
    pub fn new(vb: VarBuilder, nca_cfg: &NcaConfig, field_cfg: &FieldConfig) -> Result<Self> {
        // Local 1D perception: [Identity (C), Gradient (C), Laplacian (C)] = 3 * C
        // Macro recursive feedback adds global collective state: + C = 4 * C
        let in_channels = if nca_cfg.has_feedback() {
            field_cfg.channels * 4
        } else {
            field_cfg.channels * 3
        };

        let dense1 = linear(in_channels, nca_cfg.hidden_dim, vb.pp("dense1"))?;
        let dense_delta = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_delta"))?;
        let dense_gate = linear(nca_cfg.hidden_dim, field_cfg.channels, vb.pp("dense_gate"))?;

        Ok(Self {
            dense1,
            dense_delta,
            dense_gate,
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
        if self.nca_cfg.has_feedback() {
            breakdown.push(("macro_recursive_feedback_loop".to_string(), 0));
        }
        breakdown
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

        if self.nca_cfg.has_feedback() {
            let (_, l, _) = x.dims3()?;
            let s = match slow_state {
                Some(s_ten) => s_ten.clone(),
                None => x.mean(1)?.unsqueeze(1)?,
            };
            let s_broadcast = s.repeat((1, l, 1))?;
            Ok(Tensor::cat(&[x, &grad, &laplacian, &s_broadcast], 2)?)
        } else {
            // Concatenate along channels: [B, L, 3 * C]
            Ok(Tensor::cat(&[x, &grad, &laplacian], 2)?)
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
    /// - "layer_norm": Zero-mean, unit-variance LayerNorm along channel dimension
    pub fn apply_state_norm(tensor: &Tensor, mode: &str) -> Result<Tensor> {
        match mode {
            "rms" => {
                let sq = tensor.sqr()?;
                let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                Ok(tensor.broadcast_div(&rms)?)
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

        // 5. Residual active drift integration: x_(t+1/2) = x_t + \alpha * (\delta + f)
        let alpha = self.nca_cfg.step_size as f64;
        let scaled_delta = (effective_delta * alpha)?;
        let mut interim_x = (x + &scaled_delta)?;

        if let Some(f) = forcing {
            let scaled_f = (f * alpha)?;
            interim_x = (&interim_x + &scaled_f)?;
        }

        // 6. State normalization: projects onto compact invariant manifold (prevents open-phase energy explosion)
        let interim_x = if self.nca_cfg.state_norm != "none" {
            Self::apply_state_norm(&interim_x, &self.nca_cfg.state_norm)?
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

        // 8. Recursive feedback loop:
        // Update macroscopic slow state s_{t+1} = (1 - beta) s_t + beta * pool(new_x)
        let new_slow_state = if self.nca_cfg.has_feedback() {
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
}
