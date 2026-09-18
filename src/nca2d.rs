//! 2D Neural Cellular Automata (NCA) Engine for Titan Text
//!
//! Provides:
//! - Exact 5-point von Neumann discrete stencil perception: [Identity, Gradient_X, Gradient_Y, Laplacian_2D, Macro_Feedback]
//! - Compact Manifold RMS Normalization (per-cell over channels) to prevent norm blowup
//! - 2D Navier-Stokes viscous dissipation with adaptive von Neumann substepping (D_max = 0.25)
//! - Damped residual state updates to suppress period-2 checkerboard oscillations
//! - Zero-dependency Hilbert space-filling curve coordinate mapping for 1D sequence embedding

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Maximum stable explicit diffusion number for 2D 5-point Laplacian stencil
pub const STABLE_DIFFUSION_LIMIT_2D: f32 = 0.25;

/// Configuration for 2D Neural Cellular Automata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Nca2DConfig {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub hidden_dim: usize,
    pub step_size: f32,
    pub update_rate: f32,
    pub activation: String,
    pub viscosity: f32,
    pub feedback_mode: String,
    pub feedback_weight: f32,
    pub state_norm: String,
    pub damping_alpha: f32,
    pub periodic_boundary: bool,
}

impl Default for Nca2DConfig {
    fn default() -> Self {
        Self {
            height: 16,
            width: 16,
            channels: 32,
            hidden_dim: 64,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
            viscosity: 0.0,
            feedback_mode: "none".to_string(),
            feedback_weight: 0.2,
            state_norm: "rms".to_string(),
            damping_alpha: 0.8,
            periodic_boundary: true,
        }
    }
}

impl Nca2DConfig {
    pub fn has_feedback(&self) -> bool {
        self.feedback_mode != "none"
    }

    pub fn perception_channels(&self) -> usize {
        if self.has_feedback() {
            self.channels * 5
        } else {
            self.channels * 4
        }
    }
}

/// Zero-dependency Hilbert space-filling curve mapping
pub struct HilbertCurve;

impl HilbertCurve {
    #[inline]
    fn rot(n: usize, x: &mut usize, y: &mut usize, rx: usize, ry: usize) {
        if ry == 0 {
            if rx == 1 {
                *x = n.wrapping_sub(1).wrapping_sub(*x);
                *y = n.wrapping_sub(1).wrapping_sub(*y);
            }
            core::mem::swap(x, y);
        }
    }

    /// Maps distance `d` along Hilbert curve in `[0, n*n)` to grid coordinate `(x, y)` in `[0, n) x [0, n)`
    pub fn d2xy(n: usize, d: usize) -> (usize, usize) {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(d < n * n, "distance out of range");
        let mut rx: usize;
        let mut ry: usize;
        let mut t = d;
        let mut x = 0usize;
        let mut y = 0usize;
        let mut s = 1usize;
        while s < n {
            rx = 1 & (t / 2);
            ry = 1 & (t ^ rx);
            Self::rot(s, &mut x, &mut y, rx, ry);
            x += s * rx;
            y += s * ry;
            t /= 4;
            s <<= 1;
        }
        (x, y)
    }

    /// Maps grid coordinate `(x, y)` in `[0, n) x [0, n)` to distance `d` along Hilbert curve in `[0, n*n)`
    pub fn xy2d(n: usize, x: usize, y: usize) -> usize {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(x < n && y < n, "point out of range");
        let mut rx: usize;
        let mut ry: usize;
        let mut d = 0usize;
        let mut xx = x;
        let mut yy = y;
        let mut s = n >> 1;
        while s > 0 {
            rx = ((xx & s) > 0) as usize;
            ry = ((yy & s) > 0) as usize;
            d += s * s * ((3 * rx) ^ ry);
            Self::rot(s, &mut xx, &mut yy, rx, ry);
            s >>= 1;
        }
        d
    }

    /// Returns precomputed Hilbert grid coordinates `[(x_0, y_0), ..., (x_{L-1}, y_{L-1})]`
    pub fn sequence_coords(n: usize, seq_len: usize) -> Vec<(usize, usize)> {
        assert!(seq_len <= n * n, "sequence length exceeds grid capacity");
        (0..seq_len).map(|d| Self::d2xy(n, d)).collect()
    }
}

/// 2D Neural Cellular Automaton operating on state tensor [B, H, W, C]
#[derive(Clone)]
pub struct NeuralCellularAutomaton2D {
    pub dense1: Linear,
    pub dense_delta: Linear,
    pub dense_gate: Linear,
    pub cfg: Nca2DConfig,
}

impl NeuralCellularAutomaton2D {
    pub fn new(vb: VarBuilder, cfg: &Nca2DConfig) -> Result<Self> {
        let in_channels = cfg.perception_channels();
        let dense1 = linear(in_channels, cfg.hidden_dim, vb.pp("dense1"))?;
        let dense_delta = linear(cfg.hidden_dim, cfg.channels, vb.pp("dense_delta"))?;
        let dense_gate = linear(cfg.hidden_dim, cfg.channels, vb.pp("dense_gate"))?;

        Ok(Self {
            dense1,
            dense_delta,
            dense_gate,
            cfg: cfg.clone(),
        })
    }

    /// Circular spatial shift along height dimension (dim 1)
    pub fn roll_h(tensor: &Tensor, shift: i32) -> Result<Tensor> {
        let (_, h, _, _) = tensor.dims4()?;
        let shift = ((shift % h as i32) + h as i32) as usize % h;
        if shift == 0 {
            return Ok(tensor.clone());
        }
        let top = tensor.narrow(1, h - shift, shift)?;
        let bottom = tensor.narrow(1, 0, h - shift)?;
        Ok(Tensor::cat(&[&top, &bottom], 1)?)
    }

    /// Circular spatial shift along width dimension (dim 2)
    pub fn roll_w(tensor: &Tensor, shift: i32) -> Result<Tensor> {
        let (_, _, w, _) = tensor.dims4()?;
        let shift = ((shift % w as i32) + w as i32) as usize % w;
        if shift == 0 {
            return Ok(tensor.clone());
        }
        let left = tensor.narrow(2, w - shift, shift)?;
        let right = tensor.narrow(2, 0, w - shift)?;
        Ok(Tensor::cat(&[&left, &right], 2)?)
    }

    /// Gathers 4 spatial orthogonal neighbors: (left, right, up, down)
    pub fn neighbors(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, h, w, c) = x.dims4()?;
        if self.cfg.periodic_boundary {
            let left = Self::roll_w(x, 1)?;
            let right = Self::roll_w(x, -1)?;
            let up = Self::roll_h(x, 1)?;
            let down = Self::roll_h(x, -1)?;
            return Ok((left, right, up, down));
        }

        // Zero boundary conditions: pad zeros at edges
        let zero_col = Tensor::zeros((b, h, 1, c), x.dtype(), x.device())?;
        let left = if w > 1 {
            Tensor::cat(&[&zero_col, &x.narrow(2, 0, w - 1)?], 2)?
        } else {
            zero_col.clone()
        };
        let right = if w > 1 {
            Tensor::cat(&[&x.narrow(2, 1, w - 1)?, &zero_col], 2)?
        } else {
            zero_col
        };

        let zero_row = Tensor::zeros((b, 1, w, c), x.dtype(), x.device())?;
        let up = if h > 1 {
            Tensor::cat(&[&zero_row, &x.narrow(1, 0, h - 1)?], 1)?
        } else {
            zero_row.clone()
        };
        let down = if h > 1 {
            Tensor::cat(&[&x.narrow(1, 1, h - 1)?, &zero_row], 1)?
        } else {
            zero_row
        };

        Ok((left, right, up, down))
    }

    /// Computes 5-point von Neumann perception tensor [B, H, W, P*C]
    pub fn perceive_with_feedback(&self, x: &Tensor, slow_state: Option<&Tensor>) -> Result<Tensor> {
        let (left, right, up, down) = self.neighbors(x)?;

        // Horizontal gradient: (right - left) / 2
        let grad_x = ((&right - &left)? / 2.0)?;

        // Vertical gradient: (down - up) / 2
        let grad_y = ((&down - &up)? / 2.0)?;

        // 2D Laplacian: left + right + up + down - 4 * x
        let sum_neighbors = ((&left + &right)? + (&up + &down)?)?;
        let center_x4 = (x * 4.0)?;
        let lap2d = (&sum_neighbors - &center_x4)?;

        if self.cfg.has_feedback() {
            let (_, h, w, _) = x.dims4()?;
            let s = match slow_state {
                Some(s_ten) => s_ten.clone(),
                None => x.mean_keepdim(1)?.mean_keepdim(2)?, // [B, 1, 1, C]
            };
            let s_broadcast = s.repeat((1, h, w, 1))?;
            Ok(Tensor::cat(&[x, &grad_x, &grad_y, &lap2d, &s_broadcast], candle_core::D::Minus1)?.contiguous()?)
        } else {
            Ok(Tensor::cat(&[x, &grad_x, &grad_y, &lap2d], candle_core::D::Minus1)?.contiguous()?)
        }
    }

    /// Computes local 2D perception without explicit slow state
    pub fn perceive(&self, x: &Tensor) -> Result<Tensor> {
        self.perceive_with_feedback(x, None)
    }

    /// Applies per-cell Compact Manifold RMS Normalization across channels:
    /// Each cell's state vector is scaled by 1 / sqrt(mean(x^2) + eps), bounding energy to 0.5
    pub fn normalize_state(&self, s: &Tensor) -> Result<Tensor> {
        match self.cfg.state_norm.as_str() {
            "rms" => {
                let mean_sq = s.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let rms = (mean_sq + 1e-5)?.sqrt()?;
                Ok(s.broadcast_div(&rms)?)
            }
            "layer_norm" => {
                let mean = s.mean_keepdim(candle_core::D::Minus1)?;
                let centered = s.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let std = (var + 1e-5)?.sqrt()?;
                Ok(centered.broadcast_div(&std)?)
            }
            _ => Ok(s.clone()),
        }
    }

    /// Applies 2D Navier-Stokes viscous dissipation with von Neumann stability substepping:
    /// Maximum stable substep in 2D is D_max = 0.25. Substepping partitions total diffusion
    /// D = nu * dt into N = ceil(D / 0.25) substeps of size D / N <= 0.25.
    pub fn dissipate(&self, tensor: &Tensor, effective_diff: f32) -> Result<Tensor> {
        if effective_diff <= 0.0 {
            return Ok(tensor.clone());
        }

        let n_sub = (effective_diff / STABLE_DIFFUSION_LIMIT_2D).ceil().max(1.0) as usize;
        let sub_dt = effective_diff as f64 / n_sub as f64;
        let mut x = tensor.clone();

        for _ in 0..n_sub {
            let (left, right, up, down) = self.neighbors(&x)?;
            let sum_neighbors = ((&left + &right)? + (&up + &down)?)?;
            let center_x4 = (&x * 4.0)?;
            let lap2d = (&sum_neighbors - &center_x4)?;
            x = (&x + (lap2d * sub_dt)?)?;
        }

        Ok(x)
    }

    /// Single synchronous 2D NCA step:
    /// 1. Perceive 5-point von Neumann stencil (+ macro feedback)
    /// 2. Apply MLP: dense1 -> activation -> (dense_delta, dense_gate)
    /// 3. Viscous dissipation on update field
    /// 4. Damped residual update: S_{t+1} = S_t + alpha * step_size * (delta * gate)
    /// 5. Compact Manifold RMS normalization
    pub fn step(&self, x: &Tensor, slow_state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let perception = self.perceive_with_feedback(x, slow_state)?;

        // Forward through MLP
        let h = self.dense1.forward(&perception)?;
        let activated = match self.cfg.activation.as_str() {
            "tanh" => h.tanh()?,
            _ => candle_nn::Activation::Gelu.forward(&h)?,
        };

        let delta = self.dense_delta.forward(&activated)?;
        let gate = candle_nn::ops::sigmoid(&self.dense_gate.forward(&activated)?)?;
        let raw_update = delta.mul(&gate)?;

        // Apply viscous dissipation
        let effective_viscosity = self.cfg.viscosity * self.cfg.step_size;
        let dissipated_update = self.dissipate(&raw_update, effective_viscosity)?;

        // Apply damped residual update to suppress checkerboard oscillations
        let alpha = self.cfg.damping_alpha as f64;
        let step_scale = alpha * self.cfg.step_size as f64;
        let scaled_update = (dissipated_update * step_scale)?;
        let mut new_x = (x + scaled_update)?;

        // Normalize state on compact manifold
        new_x = self.normalize_state(&new_x)?;

        // Update slow state
        let pool_x = new_x.mean_keepdim(1)?.mean_keepdim(2)?; // [B, 1, 1, C]
        let s_next = match self.cfg.feedback_mode.as_str() {
            "dual_timescale" => {
                let beta = self.cfg.feedback_weight as f64;
                let prev_s = match slow_state {
                    Some(s) => s.clone(),
                    None => pool_x.clone(),
                };
                let blended = ((prev_s * (1.0 - beta))? + (pool_x * beta)?)?;
                blended
            }
            "global_pool" => pool_x,
            _ => pool_x,
        };

        Ok((new_x, s_next))
    }

    /// Autonomous rollout for `steps` iterations
    pub fn forward_rollout(&self, x: &Tensor, steps: usize) -> Result<Tensor> {
        let mut current_x = x.clone();
        let mut current_s = None;

        for _ in 0..steps {
            let (next_x, next_s) = self.step(&current_x, current_s.as_ref())?;
            current_x = next_x;
            current_s = Some(next_s);
        }

        Ok(current_x)
    }

    /// Adaptive kinetic rollout: halts when local kinetic energy E_kin = ||S_t - S_{t-1}||^2 falls below threshold
    pub fn forward_rollout_adaptive(
        &self,
        x: &Tensor,
        max_steps: usize,
        min_steps: usize,
        energy_threshold: f32,
    ) -> Result<(Tensor, usize, f32)> {
        let mut current_x = x.clone();
        let mut current_s = None;
        let mut steps_taken = 0;
        let mut last_e_kin = 0.0f32;

        for step_idx in 0..max_steps {
            let (next_x, next_s) = self.step(&current_x, current_s.as_ref())?;
            let diff = (&next_x - &current_x)?;
            let e_kin = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
            last_e_kin = e_kin;
            current_x = next_x;
            current_s = Some(next_s);
            steps_taken = step_idx + 1;

            if steps_taken >= min_steps && e_kin < energy_threshold {
                break;
            }
        }

        Ok((current_x, steps_taken, last_e_kin))
    }

    /// Parameter count breakdown
    pub fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let p1 = self.dense1.weight().elem_count() + self.dense1.bias().map_or(0, |b| b.elem_count());
        let p_delta = self.dense_delta.weight().elem_count() + self.dense_delta.bias().map_or(0, |b| b.elem_count());
        let p_gate = self.dense_gate.weight().elem_count() + self.dense_gate.bias().map_or(0, |b| b.elem_count());

        vec![
            ("spatial_2d_perception_stencil".to_string(), 0),
            ("hidden_feature_mlp_dense1".to_string(), p1),
            ("directional_update_dense_delta".to_string(), p_delta),
            ("adaptive_gate_dense_gate".to_string(), p_gate),
            ("viscous_dissipation_damping_2d".to_string(), 0),
        ]
    }
}

/// Configuration for 2D Latent Morphogenic Blackboard
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlackboardConfig {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub embed_dim: usize,
    pub nca_ticks_per_token: usize,
    pub post_sequence_ticks: usize,
    pub neighborhood_read: bool,
    /// Opt-in: normalize only the addressed cell, preserving untouched cells.
    #[serde(default)]
    pub normalize_written_cell_only: bool,
    pub nca_cfg: Nca2DConfig,
}

impl Default for BlackboardConfig {
    fn default() -> Self {
        let nca_cfg = Nca2DConfig {
            height: 16,
            width: 16,
            channels: 32,
            hidden_dim: 64,
            feedback_mode: "global_pool".to_string(),
            state_norm: "rms".to_string(),
            damping_alpha: 0.8,
            ..Default::default()
        };
        Self {
            height: 16,
            width: 16,
            channels: 32,
            embed_dim: 64,
            nca_ticks_per_token: 2,
            post_sequence_ticks: 4,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg,
        }
    }
}

/// Spatial coordinate routing mode for tokens onto 2D lattice
/// Spatial coordinate routing mode for tokens onto 2D lattice
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialCoordMode {
    /// Optimal continuous Hilbert space-filling curve (preserves 2D spatial locality)
    Hilbert,
    /// Deterministic pseudo-random coordinate permutation fixed across training and evaluation
    FixedShuffled(u64),
    /// Fresh pseudo-random coordinate permutation for every sample in the batch
    PerSampleShuffled(u64),
    /// Standard Hilbert mapping during training, scrambled permutation during inference
    InferenceOnlyShuffled(u64),
}

/// Readout intervention mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadIntervention {
    /// Normal active reading from 2D memory field
    Active,
    /// Zeroed readout control: sets read vector to 0 to measure residual stream autonomy (H0_A)
    Zeroed,
}

/// Write intervention mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteIntervention {
    /// Normal writing to 2D blackboard
    Active,
    /// Disabled writes: blackboard receives no token inputs (field stays zero/initial)
    Disabled,
    /// Freeze state: writes allowed up to step k, frozen thereafter
    FreezeAfter(usize),
}

/// Temporal corruption intervention mode
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum TemporalIntervention {
    /// No temporal perturbation
    None,
    /// Zero out blackboard field at a fraction of sequence length (e.g. 0.25=Early, 0.50=Mid, 0.75=Late)
    ZeroAtFraction(f32),
    /// Zero out blackboard field at a specific step index
    ZeroAtStep(usize),
    /// Inject Gaussian noise N(0, sigma^2) at a specific step index
    NoiseAtStep(usize, f32),
}

/// Recurrent dynamics deliberation regime
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DynamicsMode {
    /// Fixed T_b steps of 2D NCA physics per token
    FixedTicks(usize),
    /// Static buffer control: T_b = 0 steps (no morphogenic diffusion)
    StaticBuffer,
    /// Adaptive kinetic halting: halts when local kinetic energy E_kin falls below threshold
    AdaptiveKinetic { max_steps: usize, min_steps: usize, threshold: f32 },
}

/// Generates a deterministic, bijective permutation of [0, n) using SplitMix64 and Fisher-Yates
pub fn generate_shuffled_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let rand_val = (z ^ (z >> 31)) as usize;
        let j = rand_val % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// Detailed outcome of a sequence rollout through the Latent Blackboard
#[derive(Debug)]
pub struct BlackboardRolloutOutput {
    pub fused: Tensor,            // [B, L, D]
    pub reads: Tensor,            // [B, L, D]
    pub final_field: Tensor,       // [B, H, W, C]
    pub bur: f32,                 // Blackboard Utilization Ratio
    pub total_steps_executed: usize,
    pub avg_steps_per_token: f32,
    pub mean_e_kin: f32,
    pub step_histogram: Vec<usize>,
    pub hit_max_fraction: f32,
}

/// Dual-System Latent Morphogenic Blackboard:
/// 1D sequence models read/write to a continuous 2D NCA working memory field [B, H, W, C].
/// Causal guarantee: Read at step t precedes Write at step t, preserving strict autoregressive causality.
#[derive(Clone)]
pub struct LatentBlackboard2D {
    pub nca: NeuralCellularAutomaton2D,
    pub write_proj: Linear,
    pub read_proj: Linear,
    pub cfg: BlackboardConfig,
}

impl LatentBlackboard2D {
    pub fn new(vb: VarBuilder, cfg: &BlackboardConfig) -> Result<Self> {
        let nca = NeuralCellularAutomaton2D::new(vb.pp("nca"), &cfg.nca_cfg)?;
        let write_proj = linear(cfg.embed_dim, cfg.channels, vb.pp("write_proj"))?;
        let read_proj = candle_nn::linear_b(cfg.channels, cfg.embed_dim, false, vb.pp("read_proj"))?;
        Ok(Self {
            nca,
            write_proj,
            read_proj,
            cfg: cfg.clone(),
        })
    }

    /// Initializes a blank blackboard field of shape [B, H, W, C]
    pub fn init_field(&self, batch_size: usize, device: &candle_core::Device) -> Result<Tensor> {
        Ok(Tensor::zeros((batch_size, self.cfg.height, self.cfg.width, self.cfg.channels), candle_core::DType::F32, device)?)
    }

    /// Resolves coordinate mapping for a specific token position and sample index
    pub fn resolve_coord(
        &self,
        pos: usize,
        sample_idx: usize,
        coord_mode: SpatialCoordMode,
        is_training: bool,
    ) -> (usize, usize) {
        let capacity = self.cfg.height * self.cfg.width;
        match coord_mode {
            SpatialCoordMode::Hilbert => {
                HilbertCurve::d2xy(self.cfg.height, pos % capacity)
            }
            SpatialCoordMode::FixedShuffled(seed) => {
                let perm = generate_shuffled_permutation(capacity, seed);
                let cell = perm[pos % capacity];
                (cell % self.cfg.width, cell / self.cfg.width)
            }
            SpatialCoordMode::PerSampleShuffled(seed) => {
                let s_b = seed.wrapping_add((sample_idx as u64).wrapping_mul(1013904223));
                let perm = generate_shuffled_permutation(capacity, s_b);
                let cell = perm[pos % capacity];
                (cell % self.cfg.width, cell / self.cfg.width)
            }
            SpatialCoordMode::InferenceOnlyShuffled(seed) => {
                if is_training {
                    HilbertCurve::d2xy(self.cfg.height, pos % capacity)
                } else {
                    let perm = generate_shuffled_permutation(capacity, seed);
                    let cell = perm[pos % capacity];
                    (cell % self.cfg.width, cell / self.cfg.width)
                }
            }
        }
    }

    /// Generates spatial masks for a batch of coordinates [B, H, W, 1] or [1, H, W, 1]
    fn spatial_mask_batch(
        &self,
        coords: &[(usize, usize)],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let h = self.cfg.height;
        let w = self.cfg.width;
        let b = coords.len();
        let mut mask = vec![0.0f32; b * h * w];

        for (idx, &(x, y)) in coords.iter().enumerate() {
            let base = idx * h * w;
            if self.cfg.neighborhood_read {
                mask[base + y * w + x] += 0.5;
                let left_x = (x + w - 1) % w;
                let right_x = (x + 1) % w;
                let up_y = (y + h - 1) % h;
                let down_y = (y + 1) % h;
                mask[base + y * w + left_x] += 0.125;
                mask[base + y * w + right_x] += 0.125;
                mask[base + up_y * w + x] += 0.125;
                mask[base + down_y * w + x] += 0.125;
            } else {
                mask[base + y * w + x] = 1.0;
            }
        }

        Ok(Tensor::from_vec(mask, (b, h, w, 1), device)?)
    }

    /// Generates sharp write masks for exact slot assignment [B, H, W, 1] or [1, H, W, 1]
    fn write_mask_batch(
        &self,
        coords: &[(usize, usize)],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let h = self.cfg.height;
        let w = self.cfg.width;
        let b = coords.len();
        let mut mask = vec![0.0f32; b * h * w];

        for (idx, &(x, y)) in coords.iter().enumerate() {
            let base = idx * h * w;
            mask[base + y * w + x] = 1.0;
        }

        Ok(Tensor::from_vec(mask, (b, h, w, 1), device)?)
    }

    /// Fully controlled rollout with all experimental interventions supported
    pub fn forward_controlled(
        &self,
        token_embeddings: &Tensor,
        coord_mode: SpatialCoordMode,
        read_mode: ReadIntervention,
        write_mode: WriteIntervention,
        temporal_mode: TemporalIntervention,
        dynamics_mode: DynamicsMode,
        is_training: bool,
    ) -> Result<BlackboardRolloutOutput> {
        let (b, l, _) = token_embeddings.dims3()?;
        let mut current_field = self.init_field(b, token_embeddings.device())?;
        let mut reads = Vec::with_capacity(l);
        let mut total_steps = 0;
        let mut sum_e_kin = 0.0f32;
        let mut step_hist = vec![0usize; 16];
        let mut hit_max_count = 0usize;

        for t in 0..l {
            // 0. Temporal intervention check
            let apply_temporal = match temporal_mode {
                TemporalIntervention::None => false,
                TemporalIntervention::ZeroAtFraction(frac) => {
                    let trigger = ((l as f32) * frac).round() as usize;
                    t == trigger
                }
                TemporalIntervention::ZeroAtStep(k) => t == k,
                TemporalIntervention::NoiseAtStep(k, _) => t == k,
            };
            if apply_temporal {
                match temporal_mode {
                    TemporalIntervention::NoiseAtStep(_, sigma) => {
                        let noise = Tensor::randn(0.0f32, sigma, current_field.shape(), current_field.device())?;
                        current_field = (&current_field + &noise)?;
                    }
                    _ => {
                        current_field = Tensor::zeros(current_field.shape(), current_field.dtype(), current_field.device())?;
                    }
                }
            }

            // 1. Resolve coordinates for this step
            let coords: Vec<(usize, usize)> = match coord_mode {
                SpatialCoordMode::PerSampleShuffled(_) => {
                    (0..b).map(|s| self.resolve_coord(t, s, coord_mode, is_training)).collect()
                }
                _ => {
                    vec![self.resolve_coord(t, 0, coord_mode, is_training)]
                }
            };

            // 2. Read before write ensures strict autoregressive causality
            let r_t = match read_mode {
                ReadIntervention::Active => {
                    let mask = self.spatial_mask_batch(&coords, current_field.device())?;
                    let masked = current_field.broadcast_mul(&mask)?;
                    let mem_c = masked.sum(1)?.sum(1)?; // [B, C]
                    self.read_proj.forward(&mem_c)? // [B, embed_dim]
                }
                ReadIntervention::Zeroed => {
                    Tensor::zeros((b, self.cfg.embed_dim), token_embeddings.dtype(), token_embeddings.device())?
                }
            };
            reads.push(r_t.unsqueeze(1)?);

            // 3. Write token t into blackboard
            let token_t = token_embeddings.narrow(1, t, 1)?.squeeze(1)?;
            let write_allowed = match write_mode {
                WriteIntervention::Active => true,
                WriteIntervention::Disabled => false,
                WriteIntervention::FreezeAfter(k) => t <= k,
            };
            let written_field = if write_allowed {
                let delta_c = self.write_proj.forward(&token_t)?;
                let delta_broadcast = delta_c.unsqueeze(1)?.unsqueeze(2)?;
                let mask = self.write_mask_batch(&coords, current_field.device())?;
                let field_delta = mask.broadcast_mul(&delta_broadcast)?;
                let updated = (&current_field + &field_delta)?;
                self.normalize_after_write(&current_field, &updated, &mask)?
            } else {
                current_field.clone()
            };

            // 4. Cellular Deliberation Dynamics
            let (next_field, steps_t, e_kin_t) = match dynamics_mode {
                DynamicsMode::FixedTicks(ticks) => {
                    let f = if ticks > 0 {
                        self.nca.forward_rollout(&written_field, ticks)?
                    } else {
                        written_field
                    };
                    (f, ticks, 0.0f32)
                }
                DynamicsMode::StaticBuffer => (written_field, 0, 0.0f32),
                DynamicsMode::AdaptiveKinetic { max_steps, min_steps, threshold } => {
                    let (f, steps, e_kin) = self.nca.forward_rollout_adaptive(&written_field, max_steps, min_steps, threshold)?;
                    if steps >= max_steps {
                        hit_max_count += 1;
                    }
                    (f, steps, e_kin)
                }
            };

            current_field = next_field;
            total_steps += steps_t;
            sum_e_kin += e_kin_t;
            if steps_t < step_hist.len() {
                step_hist[steps_t] += 1;
            }
        }

        // Optional post-sequence deliberation
        if self.cfg.post_sequence_ticks > 0 && !matches!(dynamics_mode, DynamicsMode::StaticBuffer) {
            current_field = self.nca.forward_rollout(&current_field, self.cfg.post_sequence_ticks)?;
        }

        let reads_refs: Vec<&Tensor> = reads.iter().collect();
        let read_seq = Tensor::cat(&reads_refs, 1)?;
        let bur = Self::compute_bur(&read_seq, token_embeddings)?;
        let fused = (token_embeddings + &read_seq)?;
        let avg_steps = if l > 0 { total_steps as f32 / l as f32 } else { 0.0 };
        let mean_e_kin = if l > 0 { sum_e_kin / l as f32 } else { 0.0 };
        let hit_max_fraction = if l > 0 { hit_max_count as f32 / l as f32 } else { 0.0 };

        Ok(BlackboardRolloutOutput {
            fused,
            reads: read_seq,
            final_field: current_field,
            bur,
            total_steps_executed: total_steps,
            avg_steps_per_token: avg_steps,
            mean_e_kin,
            step_histogram: step_hist,
            hit_max_fraction,
        })
    }

    /// Reads memory vector from blackboard at sequence position `pos` (legacy Hilbert wrapper)
    pub fn read_token(&self, field: &Tensor, pos: usize) -> Result<Tensor> {
        let (x, y) = self.resolve_coord(pos, 0, SpatialCoordMode::Hilbert, false);
        let mask = self.spatial_mask_batch(&[(x, y)], field.device())?;
        let masked = field.broadcast_mul(&mask)?;
        let mem_c = masked.sum(1)?.sum(1)?;
        Ok(self.read_proj.forward(&mem_c)?)
    }

    /// Writes token embedding into blackboard cell at Hilbert coordinate `(x, y)` (legacy wrapper)
    pub fn write_token(&self, field: &Tensor, token_embed: &Tensor, pos: usize) -> Result<Tensor> {
        let (x, y) = self.resolve_coord(pos, 0, SpatialCoordMode::Hilbert, false);
        let delta_c = self.write_proj.forward(token_embed)?;
        let delta_broadcast = delta_c.unsqueeze(1)?.unsqueeze(2)?;
        let mask = self.write_mask_batch(&[(x, y)], field.device())?;
        let field_delta = mask.broadcast_mul(&delta_broadcast)?;
        let updated = (field + field_delta)?;
        self.normalize_after_write(field, &updated, &mask)
    }

    fn normalize_after_write(&self, previous: &Tensor, updated: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let normalized = self.nca.normalize_state(updated)?;
        if self.cfg.normalize_written_cell_only {
            // Untouched cells use the identity derivative instead of repeatedly
            // multiplying the zero-state RMS derivative, 1/sqrt(epsilon).
            Ok((normalized.broadcast_mul(mask)? + previous.broadcast_mul(&(1.0 - mask)?)?)?)
        } else {
            Ok(normalized)
        }
    }

    /// Advances the causal dual-system loop for an entire sequence of token embeddings [B, L, D]
    pub fn process_sequence(&self, token_embeddings: &Tensor) -> Result<(Tensor, Tensor)> {
        let out = self.forward_controlled(
            token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(self.cfg.nca_ticks_per_token),
            false,
        )?;
        Ok((out.reads, out.final_field))
    }

    /// Computes the Blackboard Utilization Ratio (BUR):
    /// BUR = ||read|| / (||read|| + ||residual||)
    /// Used as a live anti-bypass telemetry metric to verify the 2D blackboard is actively computing.
    pub fn compute_bur(read_tensor: &Tensor, residual_tensor: &Tensor) -> Result<f32> {
        let read_norm = read_tensor.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let res_norm = residual_tensor.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let total = read_norm + res_norm;
        if total < 1e-7 {
            Ok(0.0)
        } else {
            Ok(read_norm / total)
        }
    }

    /// Advances the dual-system causal sequence loop with residual integration:
    /// Returns (fused_embeddings [B, L, D], final_field [B, H, W, C], mean_bur: f32)
    pub fn forward_with_residual(&self, token_embeddings: &Tensor) -> Result<(Tensor, Tensor, f32)> {
        let out = self.forward_controlled(
            token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(self.cfg.nca_ticks_per_token),
            false,
        )?;
        Ok((out.fused, out.final_field, out.bur))
    }

    /// Parameter breakdown across blackboard and NCA components
    pub fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let mut breakdown = self.nca.parameter_breakdown();
        let p_write = self.write_proj.weight().elem_count() + self.write_proj.bias().map_or(0, |b| b.elem_count());
        let p_read = self.read_proj.weight().elem_count() + self.read_proj.bias().map_or(0, |b| b.elem_count());
        breakdown.push(("blackboard_write_projection".to_string(), p_write));
        breakdown.push(("blackboard_read_projection".to_string(), p_read));
        breakdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_hilbert_curve_bijective_roundtrip() {
        // Test round-trip mapping for n = 2, 4, 8, 16, 32
        for &n in &[2usize, 4, 8, 16, 32] {
            let total_points = n * n;
            for d in 0..total_points {
                let (x, y) = HilbertCurve::d2xy(n, d);
                assert!(x < n && y < n, "coords out of bounds at n={}, d={}: ({}, {})", n, d, x, y);
                let d_recovered = HilbertCurve::xy2d(n, x, y);
                assert_eq!(d, d_recovered, "roundtrip mismatch at n={}, d={}", n, d);
            }
        }
    }

    #[test]
    fn test_hilbert_curve_locality_preservation() {
        // Verify that consecutive points on Hilbert curve are Manhattan distance 1 apart (adjacent)
        let n = 16;
        for d in 0..(n * n - 1) {
            let (x1, y1) = HilbertCurve::d2xy(n, d);
            let (x2, y2) = HilbertCurve::d2xy(n, d + 1);
            let dx = (x1 as isize - x2 as isize).abs();
            let dy = (y1 as isize - y2 as isize).abs();
            let manhattan = dx + dy;
            assert_eq!(manhattan, 1, "Hilbert curve is continuous: steps must be exactly 1 cell apart (got {} between {} and {})", manhattan, d, d + 1);
        }
    }

    #[test]
    fn test_2d_nca_perception_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = Nca2DConfig {
            height: 8,
            width: 8,
            channels: 8,
            hidden_dim: 16,
            feedback_mode: "none".to_string(),
            ..Default::default()
        };

        let nca2d = NeuralCellularAutomaton2D::new(vb, &cfg)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, 8, 8, 8), &dev)?;

        // Perception without feedback: 4 * C channels = 32
        let p = nca2d.perceive(&x)?;
        assert_eq!(p.dims4()?, (2, 8, 8, 32));

        // Step forward
        let (next_x, _) = nca2d.step(&x, None)?;
        assert_eq!(next_x.dims4()?, (2, 8, 8, 8));

        Ok(())
    }

    #[test]
    fn test_2d_nca_macro_feedback_expansion() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = Nca2DConfig {
            height: 8,
            width: 8,
            channels: 8,
            hidden_dim: 16,
            feedback_mode: "global_pool".to_string(),
            ..Default::default()
        };

        let nca2d = NeuralCellularAutomaton2D::new(vb, &cfg)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, 8, 8, 8), &dev)?;

        // Perception with feedback: 5 * C channels = 40
        let p = nca2d.perceive(&x)?;
        assert_eq!(p.dims4()?, (2, 8, 8, 40));

        let (next_x, slow_s) = nca2d.step(&x, None)?;
        assert_eq!(next_x.dims4()?, (2, 8, 8, 8));
        assert_eq!(slow_s.dims4()?, (2, 1, 1, 8));

        Ok(())
    }

    #[test]
    fn test_2d_nca_rms_compact_manifold_invariance() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = Nca2DConfig {
            height: 16,
            width: 16,
            channels: 16,
            hidden_dim: 32,
            state_norm: "rms".to_string(),
            ..Default::default()
        };

        let nca2d = NeuralCellularAutomaton2D::new(vb, &cfg)?;
        // Start from large initial random energy
        let x = Tensor::randn(0.0f32, 10.0f32, (1, 16, 16, 16), &dev)?;

        // Roll out 32 steps
        let rolled = nca2d.forward_rollout(&x, 32)?;

        // Under RMS normalization, each cell's mean square over channels must be strictly bounded to ~1.0
        let mean_sq = rolled.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let mean_sq_val = mean_sq.mean_all()?.to_scalar::<f32>()?;
        assert!(
            (mean_sq_val - 1.0).abs() < 0.05,
            "2D RMS normalization failed: expected mean_sq ~ 1.0, got {}",
            mean_sq_val
        );

        Ok(())
    }

    #[test]
    fn test_2d_viscous_dissipation_substepping_stability() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = Nca2DConfig {
            height: 8,
            width: 8,
            channels: 4,
            hidden_dim: 16,
            viscosity: 2.0, // High viscosity (D = 2.0 > D_max = 0.25)
            step_size: 1.0,
            ..Default::default()
        };

        let nca2d = NeuralCellularAutomaton2D::new(vb, &cfg)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 8, 8, 4), &dev)?;

        // Dissipate with large effective diffusion (D = 2.0 -> requires 8 substeps)
        let dissipated = nca2d.dissipate(&x, 2.0)?;

        // Verify no NaN or Inf
        let mean_sq = dissipated.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(mean_sq.is_finite(), "2D viscous dissipation exploded to non-finite values");

        Ok(())
    }

    #[test]
    fn test_latent_blackboard_causal_read_write_loop() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 2,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 6, 24), &dev)?;

        let (reads, final_field) = blackboard.process_sequence(&token_embeddings)?;

        // Output read shapes must match [B, L, D]
        assert_eq!(reads.dims3()?, (2, 6, 24));
        // Final blackboard field must match [B, H, W, C]
        assert_eq!(final_field.dims4()?, (2, 8, 8, 16));

        // Read at step 0 must be exactly zero because the blackboard was empty before any write
        let first_read = reads.narrow(1, 0, 1)?.squeeze(1)?;
        let first_read_norm = first_read.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(first_read_norm, 0.0, "Causality violation: step 0 read from empty blackboard must be exactly 0");

        // Subsequent reads must have non-zero energy as written token information propagates
        let later_read = reads.narrow(1, 1, 1)?.squeeze(1)?;
        let later_read_norm = later_read.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(later_read_norm > 0.0, "Subsequent reads must contain propagating memory");

        Ok(())
    }

    #[test]
    fn test_2d_nca_adaptive_kinetic_halting() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = Nca2DConfig {
            height: 8,
            width: 8,
            channels: 8,
            hidden_dim: 16,
            viscosity: 0.1,
            state_norm: "rms".to_string(),
            damping_alpha: 0.5, // Strong damping drives state to fixed point
            ..Default::default()
        };

        let nca2d = NeuralCellularAutomaton2D::new(vb, &cfg)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 8, 8, 8), &dev)?;

        // Run adaptive rollout with max 20 steps, min 2 steps, threshold 0.1
        let (final_state, steps_taken, final_e_kin) =
            nca2d.forward_rollout_adaptive(&x, 20, 2, 0.2)?;

        assert!(steps_taken >= 2, "Adaptive halting must respect min_steps");
        assert!(steps_taken <= 20, "Adaptive halting must respect max_steps");
        assert!(final_e_kin.is_finite(), "Kinetic energy must be finite");
        assert_eq!(final_state.dims4()?, (1, 8, 8, 8));

        Ok(())
    }

    #[test]
    fn test_latent_blackboard_residual_integration_and_bur() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 1,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 5, 24), &dev)?;

        let (fused, final_field, bur) = blackboard.forward_with_residual(&token_embeddings)?;

        // Fused shape must match [B, L, D]
        assert_eq!(fused.dims3()?, (2, 5, 24));
        assert_eq!(final_field.dims4()?, (2, 8, 8, 16));

        // BUR must be strictly between 0 and 1 (non-zero active computation)
        assert!(bur > 0.0, "BUR must be positive; got {}", bur);
        assert!(bur < 1.0, "BUR must be strictly less than 1.0; got {}", bur);

        Ok(())
    }

    #[test]
    fn test_controlled_interventions_zeroed_readout() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 0,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 5, 24), &dev)?;

        // Zeroed readout: read vector must be identically zero
        let out = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Zeroed,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        let reads_norm = out.reads.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert_eq!(reads_norm, 0.0, "Zeroed readout must produce exactly zero read vectors");
        assert_eq!(out.bur, 0.0, "BUR under zeroed readout must be exactly 0");

        // Fused output under zeroed readout must equal token_embeddings exactly
        let diff = (&out.fused - &token_embeddings)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-7, "Fused output must match raw embeddings when readout is zeroed");

        Ok(())
    }

    #[test]
    fn test_controlled_interventions_write_disabled_and_freeze() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 0,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 6, 24), &dev)?;

        // Write disabled: blackboard receives no token inputs, so final field is identical for any input
        let out_disabled = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Disabled,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;
        let tokens2 = Tensor::randn(10.0f32, 5.0f32, (2, 6, 24), &dev)?;
        let out_disabled2 = blackboard.forward_controlled(
            &tokens2,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Disabled,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        let diff = (&out_disabled.final_field - &out_disabled2.final_field)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-7, "Field must be identical regardless of input when writes are disabled; got diff {}", diff);

        // Freeze after step 2: writes accepted for t <= 2, frozen thereafter
        let out_freeze = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::FreezeAfter(2),
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        let freeze_norm = out_freeze.final_field.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(freeze_norm > 0.0, "Frozen blackboard should have non-zero state from first 3 writes");

        Ok(())
    }

    #[test]
    fn test_tripartite_coordinate_modes() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 0,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 6, 24), &dev)?;

        // Mode 1: Hilbert
        let out_hilbert = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        // Mode 2: Fixed Shuffled
        let out_shuffled = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::FixedShuffled(42),
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        // Mode 3: Per-Sample Shuffled
        let out_persample = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::PerSampleShuffled(42),
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        // Mode 4: Inference Only Shuffled (in eval mode)
        let out_inf_shuffled = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::InferenceOnlyShuffled(42),
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        // All modes produce valid finite outputs
        assert_eq!(out_hilbert.fused.dims3()?, (2, 6, 24));
        assert_eq!(out_shuffled.fused.dims3()?, (2, 6, 24));
        assert_eq!(out_persample.fused.dims3()?, (2, 6, 24));
        assert_eq!(out_inf_shuffled.fused.dims3()?, (2, 6, 24));

        // Shuffled routing produces different state trajectories from Hilbert
        let diff = (&out_hilbert.final_field - &out_shuffled.final_field)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!(diff > 1e-4, "Shuffled coordinates must produce different field dynamics from Hilbert");

        Ok(())
    }

    #[test]
    fn test_temporal_corruption_intervention() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let cfg = BlackboardConfig {
            height: 8,
            width: 8,
            channels: 16,
            embed_dim: 24,
            nca_ticks_per_token: 1,
            post_sequence_ticks: 0,
            neighborhood_read: true,
            normalize_written_cell_only: false,
            nca_cfg: Nca2DConfig {
                height: 8,
                width: 8,
                channels: 16,
                hidden_dim: 32,
                ..Default::default()
            },
        };

        let blackboard = LatentBlackboard2D::new(vb, &cfg)?;
        let token_embeddings = Tensor::randn(0.0f32, 1.0f32, (2, 8, 24), &dev)?;

        // Zero at fraction 0.5 (mid sequence)
        let out_mid_zero = blackboard.forward_controlled(
            &token_embeddings,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::ZeroAtFraction(0.5),
            DynamicsMode::FixedTicks(1),
            false,
        )?;

        assert_eq!(out_mid_zero.fused.dims3()?, (2, 8, 24));
        assert!(out_mid_zero.bur > 0.0);

        Ok(())
    }
}
