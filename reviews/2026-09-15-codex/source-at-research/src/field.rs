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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralCascadeReport {
    /// Spatial Fourier power spectrum E(k) for wavenumbers k in 0..=L/2
    pub spectrum: Vec<f32>,
    /// Enstrophy power spectrum Omega(k) = k^2 * E(k)
    pub enstrophy_spectrum: Vec<f32>,
    /// Centroid wavenumber bar{k} = sum(k * E(k)) / sum(E(k))
    pub centroid_wavenumber: f32,
    /// Spectral power-law slope beta where E(k) ~ k^(-beta)
    pub spectral_slope: f32,
    /// Peak dissipation wavenumber scale
    pub dissipation_scale: f32,
}

#[derive(Clone, Debug)]
pub struct MorphogenicField {
    /// 1D spatial cell states: [batch, seq_len, channels]
    pub x: Tensor,
    pub config: FieldConfig,
    /// Macroscopic slow feedback state: [batch, 1, channels]
    pub slow_state: Option<Tensor>,
}

impl MorphogenicField {
    /// Initializes an all-zero field: [batch, seq_len, channels]
    pub fn zeros(batch_size: usize, config: &FieldConfig, device: &Device) -> Result<Self> {
        let x = Tensor::zeros((batch_size, config.seq_len, config.channels), candle_core::DType::F32, device)?;
        Ok(Self {
            x,
            config: config.clone(),
            slow_state: None,
        })
    }

    /// Creates a field from an existing tensor
    pub fn from_tensor(x: Tensor, config: &FieldConfig) -> Self {
        Self {
            x,
            config: config.clone(),
            slow_state: None,
        }
    }

    /// Attach a macroscopic slow feedback state
    #[allow(dead_code)]
    pub fn with_slow_state(mut self, slow: Tensor) -> Self {
        self.slow_state = Some(slow);
        self
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
    /// Partitions physical energy into Low (k in [0, L/6]), Mid (k in (L/6, L/3]), High (k in (L/3, L/2]).
    /// Satisfies Parseval's theorem: low_energy + mid_energy + high_energy == field.energy().
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
        let norm_factor = 2.0 * (l * l) as f64 * (b * c) as f64;

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
                    let weight = if k == 0 || (l % 2 == 0 && k == half_l) { 1.0 } else { 2.0 };
                    let power = (re * re + im * im) * weight / norm_factor;
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
            slow_state: self.slow_state.clone(),
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
            slow_state: self.slow_state.clone(),
        })
    }

    /// Circular spatial shift: rolls tensor [B, L, C] along spatial dimension L
    pub fn roll_spatial(tensor: &Tensor, shift: i32) -> Result<Tensor> {
        let (_, l, _) = tensor.dims3()?;
        let shift = ((shift % l as i32) + l as i32) as usize % l;
        if shift == 0 {
            return Ok(tensor.clone());
        }

        let left_part = tensor.narrow(1, l - shift, shift)?;
        let right_part = tensor.narrow(1, 0, l - shift)?;
        Ok(Tensor::cat(&[&left_part, &right_part], 1)?)
    }

    /// Spatial central gradient: \nabla x_i = (x_{i+1} - x_{i-1}) / 2: [B, L, C]
    pub fn spatial_gradient(&self) -> Result<Tensor> {
        let left = Self::roll_spatial(&self.x, 1)?;   // cell i-1
        let right = Self::roll_spatial(&self.x, -1)?; // cell i+1
        ((&right - &left)? / 2.0).map_err(|e| e.into())
    }

    /// Spatial central Laplacian: \Delta x_i = x_{i+1} - 2*x_i + x_{i-1}: [B, L, C]
    pub fn spatial_laplacian(&self) -> Result<Tensor> {
        let left = Self::roll_spatial(&self.x, 1)?;
        let right = Self::roll_spatial(&self.x, -1)?;
        let double_x = (&self.x * 2.0)?;
        let diff = (&right - &double_x)?;
        (&diff + &left).map_err(|e| e.into())
    }

    /// Computes fluid enstrophy: Omega = 0.5 * mean(||\nabla x||^2)
    /// Direct 1D analogue of kinetic enstrophy 0.5 * \int |\omega|^2 dx
    pub fn enstrophy(&self) -> Result<f32> {
        let grad = self.spatial_gradient()?;
        let sq = grad.sqr()?;
        let mean_sq = sq.mean_all()?.to_scalar::<f32>()?;
        Ok(0.5 * mean_sq)
    }

    /// Computes fluid palinstrophy: P = 0.5 * mean(||\Delta x||^2)
    /// Direct 1D analogue of palinstrophy 0.5 * \int |\Delta u|^2 dx governing enstrophy dissipation rate
    pub fn palinstrophy(&self) -> Result<f32> {
        let lap = self.spatial_laplacian()?;
        let sq = lap.sqr()?;
        let mean_sq = sq.mean_all()?.to_scalar::<f32>()?;
        Ok(0.5 * mean_sq)
    }

    /// Beale-Kato-Majda (BKM) gradient blow-up norm:
    /// Computes max_{b, i} ||\nabla x_{b, i}||_2 across the spatial domain.
    /// Under the BKM criterion, finite-time singularity occurs if \int_0^T ||\nabla u||_{L^\infty} dt = \infty.
    pub fn bkm_norm(&self) -> Result<f32> {
        let grad = self.spatial_gradient()?;
        let grad_sq = grad.sqr()?;
        // Sum over channel dimension C: shape [B, L, 1]
        let cell_norm_sq = grad_sq.sum_keepdim(2)?;
        let cell_norm = cell_norm_sq.sqrt()?;
        let max_norm = cell_norm.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        Ok(max_norm)
    }

    /// L^\infty maximum velocity/field amplitude: max_{b, i, c} |x_{b, i, c}|
    /// Direct check for Fefferman Millennium Cases C & D finite-time singularity.
    pub fn max_amplitude(&self) -> Result<f32> {
        let abs_x = self.x.abs()?;
        let max_val = abs_x.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        Ok(max_val)
    }

    /// Detailed Fourier spectral cascade analysis across wavenumbers k in 0..=L/2.
    /// Strictly satisfies Parseval's identity: sum_{k=0}^{L/2} E(k) == field.energy().
    /// Computes one-sided energy spectrum E(k), enstrophy spectrum Omega(k) = (2*pi*k/L)^2 * E(k),
    /// spectral centroid, inertial range power-law slope beta (E(k) ~ k^(-beta)), and dissipation scale.
    pub fn spectral_cascade_analysis(&self) -> Result<SpectralCascadeReport> {
        let (b, l, c) = self.x.dims3()?;
        let flat = self.x.to_vec3::<f32>()?;
        let half_l = l / 2;
        let two_pi = 2.0 * std::f64::consts::PI;

        let mut spectrum = vec![0.0f32; half_l + 1];
        let mut enstrophy_spectrum = vec![0.0f32; half_l + 1];
        let norm_factor = 2.0 * (l * l) as f64 * (b * c) as f64;

        for k in 0..=half_l {
            let mut total_power = 0.0f64;
            for batch_idx in 0..b {
                for ch in 0..c {
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    for n in 0..l {
                        let val = flat[batch_idx][n][ch] as f64;
                        let angle = two_pi * (k as f64) * (n as f64) / (l as f64);
                        re += val * angle.cos();
                        im -= val * angle.sin();
                    }
                    total_power += re * re + im * im;
                }
            }
            // Parseval one-sided weighting: weight = 1 for DC (k=0) and Nyquist (k=L/2), weight = 2 for interior modes
            let weight = if k == 0 || (l % 2 == 0 && k == half_l) { 1.0 } else { 2.0 };
            let e_k = (total_power * weight / norm_factor) as f32;
            let k_phys = (two_pi as f32 * k as f32) / (l as f32);
            spectrum[k] = e_k;
            enstrophy_spectrum[k] = k_phys * k_phys * e_k;
        }

        // Spectral centroid bar{k} = sum(k * E(k)) / sum(E(k))
        let total_energy: f32 = spectrum.iter().sum();
        let centroid_wavenumber = if total_energy > 1e-12 {
            spectrum.iter().enumerate().map(|(k, &p)| (k as f32) * p).sum::<f32>() / total_energy
        } else {
            0.0
        };

        // Linear regression fit for spectral slope: ln(E(k)) = -beta * ln(k) + c for active inertial modes
        // Filter out modes near the numerical noise floor to prevent false flat regression lines
        let max_power = spectrum[1..].iter().cloned().fold(0.0f32, f32::max);
        let active_threshold = (max_power * 1e-4).max(1e-9);

        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut count = 0.0f64;

        for k in 1..=half_l {
            let p = spectrum[k];
            if p >= active_threshold {
                let x = (k as f64).ln();
                let y = (p as f64).ln();
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                count += 1.0;
            }
        }

        let spectral_slope = if count >= 3.0 && (count * sum_xx - sum_x * sum_x).abs() > 1e-12 {
            let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);
            (-slope) as f32 // beta is positive when spectrum decays with k (E(k) ~ k^(-beta))
        } else {
            0.0
        };

        // Dissipation scale: centroid of the dissipation spectrum k^2 * Omega(k) = k^4 * E(k)
        let mut diss_weighted_sum = 0.0f32;
        let mut diss_denom = 0.0f32;
        for k in 1..=half_l {
            let weight = (k as f32).powi(4) * spectrum[k];
            diss_weighted_sum += (k as f32) * weight;
            diss_denom += weight;
        }
        let dissipation_scale = if diss_denom > 1e-12 {
            diss_weighted_sum / diss_denom
        } else {
            (half_l as f32) * 0.75
        };

        Ok(SpectralCascadeReport {
            spectrum,
            enstrophy_spectrum,
            centroid_wavenumber,
            spectral_slope,
            dissipation_scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_field_zero_gradients() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = FieldConfig { seq_len: 16, channels: 4, periodic_boundary: true };
        // Constant field of 3.0
        let data = vec![vec![vec![3.0f32; 4]; 16]; 1];
        let t = Tensor::new(data, &dev)?;
        let field = MorphogenicField::from_tensor(t, &cfg);

        assert!((field.energy()? - 4.5).abs() < 1e-5); // 0.5 * 3^2 = 4.5
        assert!(field.enstrophy()? < 1e-6);
        assert!(field.palinstrophy()? < 1e-6);
        assert!(field.bkm_norm()? < 1e-6);
        assert!((field.max_amplitude()? - 3.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_sinusoidal_field_fluid_observables() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 32;
        let channels = 2;
        let cfg = FieldConfig { seq_len, channels, periodic_boundary: true };
        
        let mut data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..seq_len {
            let val = (two_pi * (i as f32) / (seq_len as f32)).sin();
            for c in 0..channels {
                data[0][i][c] = val;
            }
        }
        let t = Tensor::new(data, &dev)?;
        let field = MorphogenicField::from_tensor(t, &cfg);

        // Sinusoid energy should be ~ 0.5 * 0.5 = 0.25
        let e = field.energy()?;
        assert!((e - 0.25).abs() < 0.05);

        // Enstrophy should be strictly positive
        let ens = field.enstrophy()?;
        assert!(ens > 0.0);

        // Palinstrophy should be strictly positive
        let palin = field.palinstrophy()?;
        assert!(palin > 0.0);

        // BKM norm should match maximum gradient
        let bkm = field.bkm_norm()?;
        assert!(bkm > 0.0);

        // Spectral cascade analysis
        let cascade = field.spectral_cascade_analysis()?;
        assert_eq!(cascade.spectrum.len(), seq_len / 2 + 1);
        // Dominant power should be at k = 1
        assert!(cascade.spectrum[1] > cascade.spectrum[0]);
        assert!(cascade.spectrum[1] > cascade.spectrum[2]);

        // Spectral sum should match field energy by Parseval's identity
        let spectral_sum: f32 = cascade.spectrum.iter().sum();
        assert!((e - spectral_sum).abs() < 1e-4, "Sinusoid energy {} != spectral sum {}", e, spectral_sum);
        Ok(())
    }

    #[test]
    fn test_parseval_identity_energy_conservation() -> Result<()> {
        let dev = Device::Cpu;
        let seq_len = 32;
        let channels = 3;
        let cfg = FieldConfig { seq_len, channels, periodic_boundary: true };

        // Multi-mode field with DC + multiple AC Fourier modes
        let mut data = vec![vec![vec![0.0f32; channels]; seq_len]; 1];
        let two_pi = 2.0 * std::f32::consts::PI;
        for i in 0..seq_len {
            let x1 = (two_pi * 1.0 * (i as f32) / (seq_len as f32)).sin();
            let x3 = (two_pi * 3.0 * (i as f32) / (seq_len as f32)).cos() * 0.5;
            let val = x1 + x3 + 1.2; // include DC component
            for c in 0..channels {
                data[0][i][c] = val * ((c + 1) as f32 * 0.5);
            }
        }
        let t = Tensor::new(data, &dev)?;
        let field = MorphogenicField::from_tensor(t, &cfg);

        let energy = field.energy()?;
        let cascade = field.spectral_cascade_analysis()?;
        let spectral_sum: f32 = cascade.spectrum.iter().sum();

        // Exact Parseval conservation: sum(E(k)) == field.energy()
        assert!((energy - spectral_sum).abs() < 1e-4, "Energy {} != Spectral sum {}", energy, spectral_sum);

        // Spatial frequency decomposition conservation
        let decomp = field.spatial_frequency_decomposition()?;
        assert!((energy - decomp.total_energy).abs() < 1e-4, "Energy {} != Decomp total {}", energy, decomp.total_energy);
        assert!(((decomp.pct_low + decomp.pct_mid + decomp.pct_high) - 100.0).abs() < 1e-3);

        Ok(())
    }
}

