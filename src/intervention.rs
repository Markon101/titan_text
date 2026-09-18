use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for lesion and intervention experiments.
/// Allows independently disabling, clamping, ablating, or perturbing pathways.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterventionConfig {
    /// If true, disables recurrent state transitions (recurrent update is forced to zero).
    #[serde(default)]
    pub disable_recurrent: bool,

    /// If true, forces gate values to 1.0 (fully open, un-gated update).
    #[serde(default)]
    pub disable_gates: bool,

    /// If set, clamps all gate activations to a fixed constant in [0.0, 1.0].
    #[serde(default)]
    pub gate_clamp: Option<f32>,

    /// If true, bypasses residual connection: x_{t+1} = delta instead of x_t + alpha * delta.
    #[serde(default)]
    pub disable_residual: bool,

    /// Scaling factor for recurrent update delta (default: 1.0).
    #[serde(default = "default_gain")]
    pub recurrence_gain: f32,

    /// Fraction of latent channels to randomly zero out (0.0 to 1.0).
    #[serde(default)]
    pub ablate_channel_pct: f32,

    /// Specific channel indices to ablate (set to 0.0).
    #[serde(default)]
    pub ablate_channel_indices: Vec<usize>,

    /// Standard deviation of zero-mean additive Gaussian noise injected at each tick.
    #[serde(default)]
    pub additive_noise_sigma: f32,

    /// If set, state is frozen (no updates) at and after this tick index.
    #[serde(default)]
    pub freeze_step: Option<usize>,

    /// If set, state is reset to zero at this exact tick index.
    #[serde(default)]
    pub reset_step: Option<usize>,

    /// If true, bypasses spatial perception derivatives (gradient & laplacian set to zero).
    #[serde(default)]
    pub bypass_perception: bool,

    /// If true, forces perception to be identity-only (no spatial derivatives).
    #[serde(default)]
    pub identity_only_perception: bool,

    /// If true, permutes/rotates state across batch dimension at each tick (shuffled-recurrence control).
    #[serde(default)]
    pub shuffle_batch: bool,

    /// Coordinate channel counterfactual mode: None ("intact"), "zeroed", "shuffled", "reversed", "constant"
    #[serde(default)]
    pub coord_mode: Option<String>,
}

fn default_gain() -> f32 {
    1.0
}

impl Default for InterventionConfig {
    fn default() -> Self {
        Self {
            disable_recurrent: false,
            disable_gates: false,
            gate_clamp: None,
            disable_residual: false,
            recurrence_gain: 1.0,
            ablate_channel_pct: 0.0,
            ablate_channel_indices: Vec::new(),
            additive_noise_sigma: 0.0,
            freeze_step: None,
            reset_step: None,
            bypass_perception: false,
            identity_only_perception: false,
            shuffle_batch: false,
            coord_mode: None,
        }
    }
}

impl InterventionConfig {
    /// Returns true if any intervention is active.
    pub fn is_active(&self) -> bool {
        self.disable_recurrent
            || self.disable_gates
            || self.gate_clamp.is_some()
            || self.disable_residual
            || (self.recurrence_gain - 1.0).abs() > 1e-6
            || self.ablate_channel_pct > 0.0
            || !self.ablate_channel_indices.is_empty()
            || self.additive_noise_sigma > 0.0
            || self.freeze_step.is_some()
            || self.reset_step.is_some()
            || self.bypass_perception
            || self.identity_only_perception
            || self.shuffle_batch
            || self.coord_mode.is_some()
    }

    /// Applies state-level interventions (reset, noise, freeze check) before/after tick.
    pub fn apply_to_state(&self, step: usize, state: &Tensor, device: &Device) -> Result<Tensor> {
        let mut x = state.clone();

        // 1. Reset check: reset to all zeros at specified step
        if let Some(r_step) = self.reset_step {
            if step == r_step {
                x = x.zeros_like()?;
            }
        }

        // 2. Channel ablation
        if !self.ablate_channel_indices.is_empty() {
            let (_, _, c) = x.dims3()?;
            let mut mask_vec = vec![1.0f32; c];
            for &idx in &self.ablate_channel_indices {
                if idx < c {
                    mask_vec[idx] = 0.0;
                }
            }
            let mask = Tensor::from_slice(&mask_vec, (1, 1, c), device)?;
            x = x.broadcast_mul(&mask)?;
        } else if self.ablate_channel_pct > 0.0 {
            let (_, _, c) = x.dims3()?;
            let num_to_ablate = ((c as f32) * self.ablate_channel_pct.clamp(0.0, 1.0)).round() as usize;
            let mut mask_vec = vec![1.0f32; c];
            for i in 0..num_to_ablate.min(c) {
                mask_vec[i] = 0.0;
            }
            let mask = Tensor::from_slice(&mask_vec, (1, 1, c), device)?;
            x = x.broadcast_mul(&mask)?;
        }

        // 3. Additive state noise
        if self.additive_noise_sigma > 0.0 {
            let shape = x.shape();
            let noise = Tensor::randn(0.0f32, self.additive_noise_sigma, shape, device)?;
            x = (&x + &noise)?;
        }

        // 4. Batch permutation (shuffled recurrence control)
        if self.shuffle_batch {
            let (b, _, _) = x.dims3()?;
            if b > 1 {
                let mut indices: Vec<u32> = (1..b as u32).collect();
                indices.push(0);
                let idx_tensor = Tensor::from_slice(&indices, b, device)?;
                x = x.index_select(&idx_tensor, 0)?;
            }
        }

        Ok(x)
    }

    /// Checks whether state update is frozen at this step.
    pub fn is_frozen(&self, step: usize) -> bool {
        if let Some(f_step) = self.freeze_step {
            step >= f_step
        } else {
            false
        }
    }

    /// Modifies the gate activations according to intervention settings.
    pub fn apply_to_gate(&self, gate: &Tensor) -> Result<Tensor> {
        if self.disable_gates {
            Ok(gate.ones_like()?)
        } else if let Some(clamp_val) = self.gate_clamp {
            let clamped = clamp_val.clamp(0.0, 1.0);
            let ones = gate.ones_like()?;
            (ones * (clamped as f64)).map_err(|e| e.into())
        } else {
            Ok(gate.clone())
        }
    }

    /// Modifies update delta according to recurrence gain and recurrent disablement.
    pub fn apply_to_delta(&self, delta: &Tensor) -> Result<Tensor> {
        if self.disable_recurrent {
            Ok(delta.zeros_like()?)
        } else if (self.recurrence_gain - 1.0).abs() > 1e-6 {
            (delta * (self.recurrence_gain as f64)).map_err(|e| e.into())
        } else {
            Ok(delta.clone())
        }
    }

    /// Modifies perception tensor when spatial perception is bypassed or coordinate counterfactual is active.
    pub fn apply_to_perception(&self, perception: &Tensor, identity: &Tensor) -> Result<Tensor> {
        let (b, l, c) = identity.dims3()?;
        let (_, _, cin) = perception.dims3()?;

        let mut perc = if self.bypass_perception || self.identity_only_perception {
            // Replace gradient and laplacian parts with zeros
            let zero_spatial = Tensor::zeros((b, l, 2 * c), DType::F32, identity.device())?;
            if cin > 3 * c {
                let extra = perception.narrow(2, 3 * c, cin - 3 * c)?;
                Tensor::cat(&[identity, &zero_spatial, &extra], 2)?
            } else {
                Tensor::cat(&[identity, &zero_spatial], 2)?
            }
        } else {
            perception.clone()
        };

        if let Some(ref mode) = self.coord_mode {
            if cin % c == 1 {
                let new_coords = crate::nca::NeuralCellularAutomaton::generate_coordinates(b, l, Some(mode), perc.device())?;
                let prefix = perc.narrow(2, 0, cin - 1)?;
                perc = Tensor::cat(&[&prefix, &new_coords], 2)?;
            }
        }

        Ok(perc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intervention_gate_and_delta() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::ones((1, 4, 8), DType::F32, &dev)?;

        // Default: no change
        let cfg = InterventionConfig::default();
        assert!(!cfg.is_active());
        let g = cfg.apply_to_gate(&t)?;
        assert_eq!(g.to_vec3::<f32>()?, t.to_vec3::<f32>()?);

        // Gate clamp
        let cfg_clamp = InterventionConfig {
            gate_clamp: Some(0.42),
            ..Default::default()
        };
        assert!(cfg_clamp.is_active());
        let g_clamp = cfg_clamp.apply_to_gate(&t)?;
        let val = g_clamp.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((val - 0.42).abs() < 1e-5);

        // Disable recurrent delta
        let cfg_norec = InterventionConfig {
            disable_recurrent: true,
            ..Default::default()
        };
        let d = cfg_norec.apply_to_delta(&t)?;
        let d_val = d.flatten_all()?.to_vec1::<f32>()?[0];
        assert_eq!(d_val, 0.0);

        // Recurrence gain
        let cfg_gain = InterventionConfig {
            recurrence_gain: 2.5,
            ..Default::default()
        };
        let d_gain = cfg_gain.apply_to_delta(&t)?;
        let d_gain_val = d_gain.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((d_gain_val - 2.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_intervention_channel_ablation_and_reset() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::ones((1, 2, 4), DType::F32, &dev)?;

        // Channel ablation
        let cfg_abl = InterventionConfig {
            ablate_channel_indices: vec![1, 3],
            ..Default::default()
        };
        let abl_t = cfg_abl.apply_to_state(0, &t, &dev)?;
        let abl_vec = abl_t.to_vec3::<f32>()?;
        assert_eq!(abl_vec[0][0], vec![1.0, 0.0, 1.0, 0.0]);

        // Reset step
        let cfg_reset = InterventionConfig {
            reset_step: Some(5),
            ..Default::default()
        };
        let not_reset = cfg_reset.apply_to_state(4, &t, &dev)?;
        assert_eq!(not_reset.flatten_all()?.to_vec1::<f32>()?[0], 1.0);
        let was_reset = cfg_reset.apply_to_state(5, &t, &dev)?;
        assert_eq!(was_reset.flatten_all()?.to_vec1::<f32>()?[0], 0.0);

        Ok(())
    }

    #[test]
    fn test_intervention_batch_shuffle_and_inversion_gain() -> Result<()> {
        let dev = Device::Cpu;
        // 1. Test batch rotation: B=3, L=2, C=2
        // Batch 0 filled with 10.0, Batch 1 filled with 20.0, Batch 2 filled with 30.0
        let b0 = Tensor::full(10.0f32, (1, 2, 2), &dev)?;
        let b1 = Tensor::full(20.0f32, (1, 2, 2), &dev)?;
        let b2 = Tensor::full(30.0f32, (1, 2, 2), &dev)?;
        let batch_t = Tensor::cat(&[&b0, &b1, &b2], 0)?;

        let cfg_shuffle = InterventionConfig {
            shuffle_batch: true,
            ..Default::default()
        };
        assert!(cfg_shuffle.is_active());

        let shuffled = cfg_shuffle.apply_to_state(0, &batch_t, &dev)?;
        let s_vec = shuffled.to_vec3::<f32>()?;
        // Index 0 receives index 1 (20.0), Index 1 receives index 2 (30.0), Index 2 receives index 0 (10.0)
        assert_eq!(s_vec[0][0][0], 20.0);
        assert_eq!(s_vec[1][0][0], 30.0);
        assert_eq!(s_vec[2][0][0], 10.0);

        // 2. Test inversion swap: recurrence_gain = -1.0
        let cfg_inv = InterventionConfig {
            recurrence_gain: -1.0,
            ..Default::default()
        };
        assert!(cfg_inv.is_active());
        let delta = Tensor::full(3.5f32, (1, 2, 2), &dev)?;
        let inverted_delta = cfg_inv.apply_to_delta(&delta)?;
        let inv_vec = inverted_delta.to_vec3::<f32>()?;
        assert_eq!(inv_vec[0][0][0], -3.5);

        Ok(())
    }
}
