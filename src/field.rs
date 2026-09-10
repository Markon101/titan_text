use crate::config::FieldConfig;
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrequencyDecomposition {
    pub low_energy: f32,
    pub mid_energy: f32,
    pub high_energy: f32,
    pub total_energy: f32,
    pub pct_low: f32,
    pub pct_mid: f32,
    pub pct_high: f32,
}

#[derive(Clone, Debug)]
pub struct MorphogenicField {
    /// 1D spatial cell states: [batch, seq_len, channels]
    pub x: Tensor,
    pub config: FieldConfig,
}

impl MorphogenicField {
    /// Initializes an all-zero field: [batch, seq_len, channels]
    pub fn zeros(batch_size: usize, config: &FieldConfig, device: &Device) -> Result<Self> {
        let x = Tensor::zeros((batch_size, config.seq_len, config.channels), candle_core::DType::F32, device)?;
        Ok(Self {
            x,
            config: config.clone(),
        })
    }

    /// Creates a field from an existing tensor
    pub fn from_tensor(x: Tensor, config: &FieldConfig) -> Self {
        Self {
            x,
            config: config.clone(),
        }
    }

    /// Computes normalized Euclidean L2 distance: ||x - y||
    pub fn l2_distance(&self, other: &Self) -> Result<f32> {
        let diff = (&self.x - &other.x)?;
        let sq = diff.sqr()?;
        let mean = sq.mean_all()?.to_scalar::<f32>()?;
        Ok(mean.sqrt())
    }

    /// Computes physical state energy: E = 0.5 * mean(||x||^2)
    pub fn energy(&self) -> Result<f32> {
        let sq = self.x.sqr()?;
        let mean_sq = sq.mean_all()?.to_scalar::<f32>()?;
        Ok(0.5 * mean_sq)
    }

    /// Computes mean and variance of the field states
    pub fn mean_and_var(&self) -> Result<(f32, f32)> {
        let mean = self.x.mean_all()?.to_scalar::<f32>()?;
        let diff = (&self.x - (mean as f64))?;
        let var = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        Ok((mean, var))
    }

    /// 1D Discrete Fourier spatial frequency decomposition of state energy.
    /// Partitions energy into Low (k in [0, L/6]), Mid (k in (L/6, L/3]), High (k in (L/3, L/2]).
    pub fn spatial_frequency_decomposition(&self) -> Result<FrequencyDecomposition> {
        let (b, l, c) = self.x.dims3()?;
        let flat = self.x.to_vec3::<f32>()?;
        let half_l = l / 2;
        let low_cutoff = (l / 6).max(1);
        let mid_cutoff = (l / 3).max(low_cutoff + 1);

        let mut low_pow = 0.0f64;
        let mut mid_pow = 0.0f64;
        let mut high_pow = 0.0f64;

        let two_pi = 2.0 * std::f64::consts::PI;

        for batch_idx in 0..b {
            for ch in 0..c {
                for k in 0..=half_l {
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    for n in 0..l {
                        let val = flat[batch_idx][n][ch] as f64;
                        let angle = two_pi * (k as f64) * (n as f64) / (l as f64);
                        re += val * angle.cos();
                        im -= val * angle.sin();
                    }
                    let power = (re * re + im * im) / (l as f64);
                    if k <= low_cutoff {
                        low_pow += power;
                    } else if k <= mid_cutoff {
                        mid_pow += power;
                    } else {
                        high_pow += power;
                    }
                }
            }
        }

        let total = low_pow + mid_pow + high_pow;
        let denom = total.max(1e-12);

        Ok(FrequencyDecomposition {
            low_energy: low_pow as f32,
            mid_energy: mid_pow as f32,
            high_energy: high_pow as f32,
            total_energy: total as f32,
            pct_low: ((low_pow / denom) * 100.0) as f32,
            pct_mid: ((mid_pow / denom) * 100.0) as f32,
            pct_high: ((high_pow / denom) * 100.0) as f32,
        })
    }

    /// Injects a spatial perturbation p scaled by eps: x + eps * p
    pub fn inject_perturbation(&self, p: &Tensor, eps: f32) -> Result<Self> {
        let scaled_p = (p * (eps as f64))?;
        let perturbed_x = (&self.x + &scaled_p)?;
        Ok(Self {
            x: perturbed_x,
            config: self.config.clone(),
        })
    }

    /// Injects damage by zeroing out a contiguous window of cells [start .. start + len]
    #[allow(dead_code)]
    pub fn inject_damage(&self, start: usize, len: usize, device: &Device) -> Result<Self> {
        let (b, l, c) = self.x.dims3()?;
        let mut data = self.x.to_vec3::<f32>()?;
        
        for batch_idx in 0..b {
            for offset in 0..len {
                let cell_idx = (start + offset) % l;
                for ch in 0..c {
                    data[batch_idx][cell_idx][ch] = 0.0;
                }
            }
        }
        
        let new_x = Tensor::new(data, device)?;
        Ok(Self {
            x: new_x,
            config: self.config.clone(),
        })
    }
}
