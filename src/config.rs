use crate::intervention::InterventionConfig;
use crate::latent::LatentConfig;
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
    /// Navier-Stokes physical viscous dissipation coefficient nu (nu * Laplacian)
    #[serde(default)]
    pub viscosity: f32,
    /// Macro recursive feedback loop mode: "none", "global_pool", "dual_timescale"
    #[serde(default = "default_feedback_mode")]
    pub feedback_mode: String,
    /// Feedback coupling rate beta in (0, 1] for dual timescale leaky integration
    #[serde(default = "default_feedback_weight")]
    pub feedback_weight: f32,
    /// State normalization mode on intermediate cellular states: "none", "rms", "bounded", "layer_norm"
    #[serde(default = "default_state_norm")]
    pub state_norm: String,
    /// Maximum RMS threshold for "bounded" state normalization mode
    #[serde(default = "default_bound_threshold")]
    pub bound_threshold: f32,
    /// Damping alpha factor on residual update (default: 1.0)
    #[serde(default = "default_damping_alpha")]
    pub damping_alpha: f32,
    /// Leaky integration contraction rate lambda on state x_{t+1} = (1 - lambda)*x_t + ... (default: 0.0)
    #[serde(default = "default_leaky_lambda")]
    pub leaky_lambda: f32,
    /// Whether to append a static 1D spatial coordinate channel p_i in [-1, 1] to the perception vector
    #[serde(default)]
    pub coord_channel: bool,
    /// Coordinate counterfactual applied DURING TRAINING/DEVELOPMENT whenever
    /// the coordinate channel is active: None = intact positions, or one of
    /// "zeroed" | "constant" | "shuffled" | "reversed". Used for the
    /// training-time sham arm (constant = same added dimensionality, no
    /// positional information). Defaults to None (prior behavior unchanged).
    #[serde(default)]
    pub coord_train_mode: Option<String>,
    /// Whether to use a strictly causal / directed spatial stencil N(i) = {i-1, i} (DAG fold)
    #[serde(default)]
    pub causal_stencil: bool,
    /// Explicit Carry Register (ECR): number of dedicated carry channels Cc in state x (default: 0)
    #[serde(default)]
    pub carry_channels: usize,
    /// Fast carry skip stride k (default: 1; if > 1, carry channels include k-cell skip transport)
    #[serde(default = "default_carry_skip_stride")]
    pub carry_skip_stride: usize,
    /// Whether carry channels are bidirectional (split between forward left-to-right and backward right-to-left)
    #[serde(default)]
    pub carry_bidirectional: bool,
    /// Discrete carry projection / drift mitigation mode: "none", "ste_round", "ste_sign", "bistable"
    #[serde(default = "default_carry_quantization")]
    pub carry_quantization: String,
    /// Persistent Input: whether to concatenate initial token seed embedding to perception vector
    #[serde(default)]
    pub persistent_input: bool,
    /// Macro grid stride s for hierarchy mode (default: 2; 1 for degenerate control)
    #[serde(default = "default_macro_stride")]
    pub macro_stride: usize,
    /// Macro update clock period k (default: 2; macro updates every k ticks)
    #[serde(default = "default_macro_period")]
    pub macro_period: usize,
    /// Macro channel dimension (default: 32)
    #[serde(default = "default_macro_channels")]
    pub macro_channels: usize,
    /// Macro downsampling mode: "mean" or "walsh" (default: "walsh")
    #[serde(default = "default_macro_downsampler")]
    pub macro_downsampler: String,
    /// Macro coupling mode: "perception" or "state_derivative" (default: "state_derivative")
    #[serde(default = "default_macro_coupling")]
    pub macro_coupling: String,
    /// Macro state-derivative coupling gain gamma (default: 0.2)
    #[serde(default = "default_macro_gamma")]
    pub macro_gamma: f32,
    /// Macro emergent bistable potential lambda (default: 0.1)
    #[serde(default = "default_macro_lambda")]
    pub macro_lambda: f32,
}

fn default_macro_stride() -> usize {
    2
}

fn default_macro_period() -> usize {
    2
}

fn default_macro_channels() -> usize {
    32
}

fn default_macro_downsampler() -> String {
    "walsh".to_string()
}

fn default_macro_coupling() -> String {
    "state_derivative".to_string()
}

fn default_macro_gamma() -> f32 {
    0.2
}

fn default_macro_lambda() -> f32 {
    0.1
}

fn default_feedback_mode() -> String {
    "none".to_string()
}

fn default_feedback_weight() -> f32 {
    0.2
}

fn default_state_norm() -> String {
    "none".to_string()
}

fn default_bound_threshold() -> f32 {
    1.5
}

fn default_damping_alpha() -> f32 {
    1.0
}

fn default_leaky_lambda() -> f32 {
    0.0
}

fn default_carry_skip_stride() -> usize {
    1
}

fn default_carry_quantization() -> String {
    "none".to_string()
}

impl NcaConfig {
    pub fn has_feedback(&self) -> bool {
        self.feedback_mode != "none"
    }

    pub fn is_hierarchy(&self) -> bool {
        self.feedback_mode == "hierarchy"
    }
}

impl Default for NcaConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 96,
            step_size: 0.5,
            update_rate: 1.0,
            activation: "gelu".to_string(),
            viscosity: 0.0,
            feedback_mode: default_feedback_mode(),
            feedback_weight: default_feedback_weight(),
            state_norm: default_state_norm(),
            bound_threshold: default_bound_threshold(),
            damping_alpha: default_damping_alpha(),
            leaky_lambda: default_leaky_lambda(),
            coord_channel: false,
            coord_train_mode: None,
            causal_stencil: false,
            carry_channels: 0,
            carry_skip_stride: 1,
            carry_bidirectional: false,
            carry_quantization: default_carry_quantization(),
            persistent_input: false,
            macro_stride: default_macro_stride(),
            macro_period: default_macro_period(),
            macro_channels: default_macro_channels(),
            macro_downsampler: default_macro_downsampler(),
            macro_coupling: default_macro_coupling(),
            macro_gamma: default_macro_gamma(),
            macro_lambda: default_macro_lambda(),
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
    /// Horizon training regime: "fixed", "randomized", "multi_tick", "jitter", "stability_tail", "early_exit"
    #[serde(default = "default_horizon_mode")]
    pub horizon_mode: String,
    /// Minimum horizon for randomized/jitter/multi-tick training
    #[serde(default = "default_horizon_min")]
    pub horizon_min: usize,
    /// Maximum horizon for randomized/jitter/multi-tick training
    #[serde(default = "default_horizon_max")]
    pub horizon_max: usize,
    /// Jitter magnitude for jitter training
    #[serde(default = "default_horizon_jitter")]
    pub horizon_jitter: usize,
    /// Stability tail length K (number of steps to supervise after first correct step)
    #[serde(default = "default_stability_tail")]
    pub stability_tail: usize,
    /// Multi-tick loss weighting decay
    #[serde(default = "default_multi_tick_decay")]
    pub multi_tick_decay: f32,
    /// Tail equilibrium loss weight lambda_eq (attractor settling constraint)
    #[serde(default = "default_tail_equilibrium_weight")]
    pub tail_equilibrium_weight: f32,
    /// Number of tail steps K to apply equilibrium loss on
    #[serde(default = "default_tail_equilibrium_ticks")]
    pub tail_equilibrium_ticks: usize,
    /// Optional single query slot to supervise (e.g. 3 for Slot 3 only)
    #[serde(default)]
    pub target_slot: Option<usize>,
    /// Auxiliary deep-supervision weight alpha: per-tick masked CE at interior
    /// slot positions, added as L + alpha * mean(aux). 0.0 disables (default).
    #[serde(default = "default_aux_supervision_weight")]
    pub aux_supervision_weight: f32,
    /// Sham auxiliary control: identical aux machinery but constant target
    /// token id 1 (bias-absorbable control, convention of C-COORD-015).
    #[serde(default)]
    pub aux_sham: bool,
    /// RD-018b clean-substrate marker. Fresh runs through the seeded vNext
    /// path record this in their manifests (deterministic initialization from
    /// `seed`, advancing training data stream, gradient clipping, dedicated
    /// auxiliary head, post-update auxiliary supervision). Defaults to false
    /// so legacy checkpoints and their continuations keep the historical
    /// training semantics unchanged.
    #[serde(default)]
    pub vnext_substrate: bool,
    /// RD-018b: maximum global L2 gradient norm for the vNext path; when the
    /// total gradient norm exceeds this bound, all gradients are rescaled
    /// before the optimizer step (0.0 disables clipping). Ignored by the
    /// legacy path.
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: f32,
}

fn default_horizon_mode() -> String {
    "fixed".to_string()
}
fn default_horizon_min() -> usize {
    2
}
fn default_horizon_max() -> usize {
    16
}
fn default_horizon_jitter() -> usize {
    2
}
fn default_stability_tail() -> usize {
    4
}
fn default_multi_tick_decay() -> f32 {
    1.0
}
fn default_tail_equilibrium_weight() -> f32 {
    0.0
}
fn default_tail_equilibrium_ticks() -> usize {
    2
}
fn default_aux_supervision_weight() -> f32 {
    0.0
}
fn default_grad_clip_norm() -> f32 {
    1.0
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
            horizon_mode: default_horizon_mode(),
            horizon_min: default_horizon_min(),
            horizon_max: default_horizon_max(),
            horizon_jitter: default_horizon_jitter(),
            stability_tail: default_stability_tail(),
            multi_tick_decay: default_multi_tick_decay(),
            tail_equilibrium_weight: default_tail_equilibrium_weight(),
            tail_equilibrium_ticks: default_tail_equilibrium_ticks(),
            target_slot: None,
            aux_supervision_weight: default_aux_supervision_weight(),
            aux_sham: false,
            vnext_substrate: false,
            grad_clip_norm: default_grad_clip_norm(),
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
    /// Smooth external forcing amplitude for Navier-Stokes singularity probe
    #[serde(default)]
    pub forcing_amplitude: f32,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            horizon: 64,
            perturbation_eps: 0.01,
            lp_window: 8,
            forcing_amplitude: 0.2,
        }
    }
}

fn default_model() -> String {
    "nca".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TitanConfig {
    /// Explicit opt-in laboratory substrate. Historical commands reject this
    /// field rather than accidentally interpreting transport weights as NCA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substrate: Option<crate::substrate::SubstrateConfig>,
    pub field: FieldConfig,
    pub nca: NcaConfig,
    pub train: TrainConfig,
    pub probe: ProbeConfig,
    #[serde(default)]
    pub latent: LatentConfig,
    #[serde(default)]
    pub intervention: InterventionConfig,
    #[serde(default = "default_model")]
    pub model: String,
}

impl Default for TitanConfig {
    fn default() -> Self {
        Self {
            substrate: None,
            field: FieldConfig::default(),
            nca: NcaConfig::default(),
            train: TrainConfig::default(),
            probe: ProbeConfig::default(),
            latent: LatentConfig::default(),
            intervention: InterventionConfig::default(),
            model: "nca".to_string(),
        }
    }
}

impl TitanConfig {
    /// Validate effective settings, including values inherited from a checkpoint.
    pub fn validate(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(self.substrate.is_none(), "substrate manifests require titan_substrate; legacy commands cannot load them");
        ensure!(
            self.field.seq_len > 0 && self.field.seq_len <= i32::MAX as usize,
            "sequence length must be positive and fit spatial indexing"
        );
        ensure!(
            self.field.channels > 0 && self.nca.hidden_dim > 0,
            "field channels and NCA hidden dimension must be positive"
        );
        ensure!(
            self.nca.step_size.is_finite() && self.nca.step_size > 0.0,
            "NCA step_size must be finite and positive"
        );
        ensure!(
            self.nca.update_rate.is_finite() && (0.0..=1.0).contains(&self.nca.update_rate),
            "NCA update_rate must be in [0, 1]"
        );
        ensure!(
            matches!(self.nca.activation.as_str(), "gelu" | "tanh"),
            "NCA activation must be gelu or tanh"
        );
        ensure!(
            matches!(self.nca.feedback_mode.as_str(), "none" | "global_pool" | "dual_timescale" | "hierarchy"),
            "nca feedback_mode must be one of: none, global_pool, dual_timescale, hierarchy"
        );
        ensure!(
            self.nca.feedback_weight.is_finite() && (0.0..=1.0).contains(&self.nca.feedback_weight),
            "nca feedback_weight must be in [0, 1]"
        );
        ensure!(
            self.nca.macro_stride >= 1,
            "nca macro_stride must be >= 1"
        );
        ensure!(
            self.nca.macro_period >= 1,
            "nca macro_period must be >= 1"
        );
        ensure!(
            self.nca.macro_channels >= 1,
            "nca macro_channels must be >= 1"
        );
        ensure!(
            matches!(self.nca.macro_downsampler.as_str(), "mean" | "walsh"),
            "nca macro_downsampler must be one of: mean, walsh"
        );
        ensure!(
            matches!(self.nca.macro_coupling.as_str(), "perception" | "state_derivative"),
            "nca macro_coupling must be one of: perception, state_derivative"
        );
        ensure!(
            self.nca.macro_gamma.is_finite() && (0.0..=1.0).contains(&self.nca.macro_gamma),
            "nca macro_gamma must be in [0, 1]"
        );
        ensure!(
            self.nca.macro_lambda.is_finite() && (0.0..=1.0).contains(&self.nca.macro_lambda),
            "nca macro_lambda must be in [0, 1]"
        );
        ensure!(
            matches!(self.nca.state_norm.as_str(), "none" | "rms" | "bounded" | "bounded_h_only" | "layer_norm"),
            "nca state_norm must be one of: none, rms, bounded, bounded_h_only, layer_norm"
        );
        ensure!(
            self.nca.bound_threshold.is_finite() && self.nca.bound_threshold > 0.0,
            "nca bound_threshold must be finite and positive"
        );
        ensure!(
            self.nca.damping_alpha.is_finite() && self.nca.damping_alpha > 0.0 && self.nca.damping_alpha <= 2.0,
            "nca damping_alpha must be in (0, 2]"
        );
        ensure!(
            self.nca.leaky_lambda.is_finite() && self.nca.leaky_lambda >= 0.0 && self.nca.leaky_lambda < 1.0,
            "nca leaky_lambda must be in [0, 1)"
        );
        ensure!(
            self.nca.viscosity.is_finite() && self.nca.viscosity >= 0.0,
            "viscosity must be finite and non-negative"
        );
        let substeps = (self.nca.viscosity * self.nca.step_size / 0.25).ceil();
        ensure!(
            substeps.is_finite() && substeps < usize::MAX as f32,
            "viscosity and step_size exceed supported diffusion substep range"
        );
        ensure!(
            self.train.tail_equilibrium_weight.is_finite() && self.train.tail_equilibrium_weight >= 0.0,
            "train tail_equilibrium_weight must be finite and non-negative"
        );
        ensure!(
            self.train.aux_supervision_weight.is_finite() && self.train.aux_supervision_weight >= 0.0,
            "train aux_supervision_weight must be finite and non-negative"
        );
        ensure!(
            self.train.grad_clip_norm.is_finite() && self.train.grad_clip_norm >= 0.0,
            "train grad_clip_norm must be finite and non-negative"
        );
        ensure!(
            self.train.tail_equilibrium_ticks > 0,
            "train tail_equilibrium_ticks must be positive"
        );
        ensure!(
            self.train.epochs > 0 && self.train.dev_steps > 0 && self.train.batch_size > 0,
            "training epochs, dev_steps, and batch_size must be positive"
        );
        ensure!(
            self.train.log_every > 0 && self.train.save_every > 0,
            "training log_every and save_every must be positive"
        );
        ensure!(
            self.train.lr.is_finite() && self.train.lr > 0.0,
            "learning rate must be finite and positive"
        );
        ensure!(
            self.train.weight_decay.is_finite() && self.train.weight_decay >= 0.0,
            "weight_decay must be finite and non-negative"
        );
        ensure!(
            matches!(self.train.horizon_mode.as_str(), "fixed" | "randomized" | "multi_tick" | "jitter" | "stability_tail" | "early_exit"),
            "train horizon_mode must be one of: fixed, randomized, multi_tick, jitter, stability_tail, early_exit"
        );
        ensure!(
            self.train.horizon_min > 0 && self.train.horizon_max >= self.train.horizon_min,
            "train horizon_min must be > 0 and horizon_max >= horizon_min"
        );
        ensure!(
            self.probe.horizon > 0 && self.probe.lp_window > 0,
            "probe horizon and lp_window must be positive"
        );
        let denominator = 2.0 * self.probe.perturbation_eps * self.probe.perturbation_eps;
        ensure!(self.probe.perturbation_eps.is_finite() && self.probe.perturbation_eps > 0.0
            && denominator.is_finite() && denominator > 0.0,
            "probe perturbation_eps must be positive with a finite, nonzero f32 squared denominator");
        ensure!(
            self.probe.forcing_amplitude.is_finite() && self.probe.forcing_amplitude >= 0.0,
            "probe forcing_amplitude must be finite and non-negative"
        );
        ensure!(
            self.latent.slow_timescale_cadence > 0,
            "slow_timescale_cadence must be a positive integer"
        );
        ensure!(
            (0.0..=1.0).contains(&self.latent.slow_channel_fraction),
            "slow_channel_fraction must be in [0, 1]"
        );
        ensure!(
            matches!(self.model.as_str(), "nca" | "transformer" | "gru" | "rnn" | "simple-recurrent"),
            "model must be nca, transformer, gru, rnn, or simple-recurrent"
        );
        Ok(())
    }
}
