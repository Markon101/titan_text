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
        let in_channels = field_cfg.channels * 3;

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

    /// Computes local 1D perception: Identity, 1st Derivative (Gradient), 2nd Derivative (Laplacian)
    pub fn perceive(&self, x: &Tensor) -> Result<Tensor> {
        let left = Self::roll_spatial(x, 1)?;   // cell i-1
        let right = Self::roll_spatial(x, -1)?; // cell i+1

        // 1st spatial derivative: (right - left) / 2
        let grad = ((&right - &left)? / 2.0)?;

        // 2nd spatial derivative (Laplacian): right - 2 * x + left
        let double_x = (x * 2.0)?;
        let diff = (&right - &double_x)?;
        let laplacian = (&diff + &left)?;

        // Concatenate along channels: [B, L, 3 * C]
        Ok(Tensor::cat(&[x, &grad, &laplacian], 2)?)
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

        // 1. Local spatial perception [B, L, 3C]
        let perception = self.perceive(x)?;

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

        // 6. Navier-Stokes physical viscous dissipation: \nu * \Delta x * \alpha
        // Scales consistently with time step alpha and employs adaptive substepping
        let new_x = if self.nca_cfg.viscosity > 0.0 {
            let effective_diff = self.nca_cfg.viscosity.max(0.0) * self.nca_cfg.step_size;
            Self::apply_viscous_dissipation(&interim_x, effective_diff)?
        } else {
            interim_x
        };

        // Compute displacement norm ||x_(t+1) - x_t||
        let total_displacement = (&new_x - x)?;
        let update_sq = total_displacement.sqr()?;
        let update_norm = update_sq.mean_all()?.to_scalar::<f32>()?.sqrt();

        Ok((
            MorphogenicField {
                x: new_x,
                config: self.field_cfg.clone(),
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
}
