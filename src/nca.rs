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

    /// Single developmental step with optional external forcing:
    /// x_(t+1) = x_t + alpha * (gate * delta) + nu * Delta x + alpha * f
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

        // 5. Residual integration: x_(t+1) = x_t + alpha * delta
        let scaled_delta = (effective_delta * (self.nca_cfg.step_size as f64))?;
        let mut total_delta = scaled_delta;

        // 6. Navier-Stokes physical viscous dissipation: nu * Delta x
        if self.nca_cfg.viscosity > 0.0 {
            let laplacian = field.spatial_laplacian()?;
            let viscous_damping = (laplacian * (self.nca_cfg.viscosity as f64))?;
            total_delta = (&total_delta + &viscous_damping)?;
        }

        // 7. Smooth external forcing: f(s, t) * alpha
        if let Some(f) = forcing {
            let scaled_f = (f * (self.nca_cfg.step_size as f64))?;
            total_delta = (&total_delta + &scaled_f)?;
        }

        let new_x = (x + &total_delta)?;

        // Compute update norm ||x_(t+1) - x_t||
        let update_sq = total_delta.sqr()?;
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
}
