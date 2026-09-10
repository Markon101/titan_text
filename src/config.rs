use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldConfig {
    /// Number of spatial cells in 1D lattice (sequence dimension L)
    pub seq_len: usize,
    /// Continuous hidden feature channels per cell (C)
    pub channels: usize,
    /// Boundary condition: periodic (ring) vs zero-padding
    pub periodic_boundary: bool,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            seq_len: 32,
            channels: 64,
            periodic_boundary: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NcaConfig {
    /// Hidden dimension in local update MLP
    pub hidden_dim: usize,
    /// Integration step size alpha
    pub step_size: f32,
    /// Stochastic update rate (1.0 = fully synchronous)
    pub update_rate: f32,
    /// Activation function: "gelu" or "tanh"
    pub activation: String,
}

impl Default for NcaConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 96,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Training epochs
    pub epochs: usize,
    /// Developmental recurrence steps per observation (T)
    pub dev_steps: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate for AdamW
    pub lr: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Checkpoint save interval
    pub save_every: usize,
    /// Logging interval
    pub log_every: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 200,
            dev_steps: 8,
            batch_size: 8,
            lr: 0.003,
            weight_decay: 0.01,
            save_every: 50,
            log_every: 10,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Autonomous unroll horizon
    pub horizon: usize,
    /// Perturbation epsilon
    pub perturbation_eps: f32,
    /// Low-pass filter window size
    pub lp_window: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            horizon: 64,
            perturbation_eps: 0.01,
            lp_window: 8,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TitanConfig {
    pub field: FieldConfig,
    pub nca: NcaConfig,
    pub train: TrainConfig,
    pub probe: ProbeConfig,
}

impl Default for TitanConfig {
    fn default() -> Self {
        Self {
            field: FieldConfig::default(),
            nca: NcaConfig::default(),
            train: TrainConfig::default(),
            probe: ProbeConfig::default(),
        }
    }
}
