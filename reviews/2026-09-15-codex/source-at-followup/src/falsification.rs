use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::nca2d::{
    BlackboardConfig, BlackboardRolloutOutput, DynamicsMode, LatentBlackboard2D, Nca2DConfig,
    ReadIntervention, SpatialCoordMode, TemporalIntervention, WriteIntervention,
};
use crate::tasks::{ArithmeticStratum, TaskEngine};
use crate::vocab::TokenInterface;

/// Mathematical utility for exact binomial McNemar test for paired binary observations.
/// Computes exact two-sided p-value under the null hypothesis p = 0.5.
pub fn exact_mcnemar_binomial(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut terms = Vec::with_capacity(k + 1);
    let mut log_binom = 0.0f64;
    terms.push(log_binom);
    for i in 1..=k {
        log_binom += ((n - i + 1) as f64).ln() - (i as f64).ln();
        terms.push(log_binom);
    }
    let log_half_n = (n as f64) * (2.0f64).ln();
    let mut p_val = 0.0f64;
    for term in terms {
        p_val += (term - log_half_n).exp();
    }
    p_val *= 2.0;
    p_val.min(1.0)
}

/// Combines independent p-values across random seeds using Fisher's method.
/// Uses the exact closed-form survival function of the Chi-Square distribution with df = 2m.
pub fn combine_fisher_pvalues(p_values: &[f64]) -> f64 {
    let m = p_values.len();
    if m == 0 {
        return 1.0;
    }
    let mut x_sq = 0.0f64;
    for &p in p_values {
        let clamped = p.max(1e-15).min(1.0);
        x_sq += -2.0 * clamped.ln();
    }
    let half_x = x_sq / 2.0;
    let mut sum = 0.0f64;
    let mut term = 1.0f64;
    for k in 0..m {
        sum += term;
        term *= half_x / ((k + 1) as f64);
    }
    (sum * (-half_x).exp()).min(1.0).max(0.0)
}

/// Computes empirical bootstrap confidence interval for paired accuracy difference (Acc1 - Acc2).
pub fn bootstrap_difference_ci(diffs: &[f32], n_boot: usize, alpha: f32) -> (f32, f32) {
    let n = diffs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(0xbf58476d1ce4e5b9);
    let mut boot_means = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let mut sum = 0.0f32;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += diffs[idx];
        }
        boot_means.push(sum / n as f32);
    }

    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = ((alpha / 2.0) * n_boot as f32).round() as usize;
    let hi_idx = (((1.0 - alpha / 2.0) * n_boot as f32).round() as usize).min(n_boot - 1);

    (boot_means[lo_idx], boot_means[hi_idx])
}

/// Parameter-matched 1D Recurrent Control (Useless-Compute Baseline).
/// Matches LatentBlackboard2D parameter count to within +/- 1% while executing
/// recurrent refinement iterations without 2D lattice or spatial NCA physics.
#[derive(Clone)]
pub struct UselessCompute1D {
    pub dense1: Linear,
    pub dense2: Linear,
    pub k_rec: usize,
    pub alpha: f32,
}

impl UselessCompute1D {
    pub fn new(vb: VarBuilder, embed_dim: usize, hidden_dim: usize, k_rec: usize) -> Result<Self> {
        let dense1 = linear(embed_dim, hidden_dim, vb.pp("dense1"))?;
        let dense2 = linear(hidden_dim, embed_dim, vb.pp("dense2"))?;
        Ok(Self {
            dense1,
            dense2,
            k_rec,
            alpha: 0.8,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for _ in 0..self.k_rec {
            let h = self.dense1.forward(&current)?;
            let act = candle_nn::Activation::Gelu.forward(&h)?;
            let delta = self.dense2.forward(&act)?;
            current = ((&current * (1.0 - self.alpha as f64))? + (&delta * (self.alpha as f64))?)?;
        }
        Ok(current)
    }

    pub fn parameter_count(&self) -> usize {
        let p1 = self.dense1.weight().elem_count() + self.dense1.bias().map_or(0, |b| b.elem_count());
        let p2 = self.dense2.weight().elem_count() + self.dense2.bias().map_or(0, |b| b.elem_count());
        p1 + p2
    }
}

/// Unified Dual-System Model wrapper supporting all 5 comparative arms
pub struct DualSystemModel {
    pub interface: TokenInterface,
    pub blackboard: Option<LatentBlackboard2D>,
    pub useless_control: Option<UselessCompute1D>,
    pub arm: ArmKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArmKind {
    Arm1BlackboardFixed,
    Arm2BlackboardAdaptive,
    Arm3UselessCompute,
    Arm4RetrainedShuffled,
    Arm5StaticBuffer,
}

impl ArmKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Arm1BlackboardFixed => "Arm 1 (2D Latent Blackboard, Fixed T=2)",
            Self::Arm2BlackboardAdaptive => "Arm 2 (2D Latent Blackboard, Adaptive Halting)",
            Self::Arm3UselessCompute => "Arm 3 (Tokenwise Recurrent Refinement Control)",
            Self::Arm4RetrainedShuffled => "Arm 4 (Retrained Fixed Shuffled Coordinates)",
            Self::Arm5StaticBuffer => "Arm 5 (Static Buffer Control, T=0)",
        }
    }
}

impl DualSystemModel {
    pub fn new(vb: VarBuilder, arm: ArmKind, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        let interface = TokenInterface::new(vb.pp("interface"), vocab_size, embed_dim)?;

        match arm {
            ArmKind::Arm3UselessCompute => {
                // 32 -> 256 -> 32 MLP yields 16,672 parameters (matching blackboard's 16,544)
                let useless = UselessCompute1D::new(vb.pp("useless"), embed_dim, 256, 4)?;
                Ok(Self {
                    interface,
                    blackboard: None,
                    useless_control: Some(useless),
                    arm,
                })
            }
            _ => {
                let bb_cfg = BlackboardConfig {
                    height: 8,
                    width: 8,
                    channels: 32,
                    embed_dim,
                    nca_ticks_per_token: match arm {
                        ArmKind::Arm5StaticBuffer => 0,
                        _ => 2,
                    },
                    post_sequence_ticks: 0,
                    neighborhood_read: true,
                    normalize_written_cell_only: false,
                    nca_cfg: Nca2DConfig {
                        height: 8,
                        width: 8,
                        channels: 32,
                        hidden_dim: 64,
                        viscosity: 0.1,
                        feedback_mode: "global_pool".to_string(),
                        state_norm: "rms".to_string(),
                        damping_alpha: 0.8,
                        ..Default::default()
                    },
                };
                let blackboard = LatentBlackboard2D::new(vb.pp("blackboard"), &bb_cfg)?;
                Ok(Self {
                    interface,
                    blackboard: Some(blackboard),
                    useless_control: None,
                    arm,
                })
            }
        }
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        coord_mode: SpatialCoordMode,
        read_mode: ReadIntervention,
        write_mode: WriteIntervention,
        temporal_mode: TemporalIntervention,
        dynamics_mode: DynamicsMode,
        is_training: bool,
    ) -> Result<(Tensor, Option<BlackboardRolloutOutput>)> {
        let embeds = self.interface.embed_tokens(tokens)?;

        if let Some(ref useless) = self.useless_control {
            let processed = useless.forward(&embeds)?;
            let logits = self.interface.logits(&processed)?;
            Ok((logits, None))
        } else if let Some(ref bb) = self.blackboard {
            let out = bb.forward_controlled(
                &embeds,
                coord_mode,
                read_mode,
                write_mode,
                temporal_mode,
                dynamics_mode,
                is_training,
            )?;
            let logits = self.interface.logits(&out.fused)?;
            Ok((logits, Some(out)))
        } else {
            anyhow::bail!("Uninitialized model arm");
        }
    }

    pub fn total_parameters(&self) -> usize {
        let p_embed = self.interface.embedding.embeddings().elem_count();
        let p_proj = self.interface.projection.weight().elem_count()
            + self.interface.projection.bias().map_or(0, |b| b.elem_count());
        let p_iface = p_embed + p_proj;

        let p_bb = self
            .blackboard
            .as_ref()
            .map_or(0, |b| b.parameter_breakdown().iter().map(|(_, c)| *c).sum());
        let p_useless = self.useless_control.as_ref().map_or(0, |u| u.parameter_count());

        p_iface + p_bb + p_useless
    }

    pub fn recurrent_updates_per_token(&self) -> usize {
        match self.arm {
            ArmKind::Arm1BlackboardFixed => 2,
            ArmKind::Arm2BlackboardAdaptive => 3, // nominal average
            ArmKind::Arm3UselessCompute => 4,
            ArmKind::Arm4RetrainedShuffled => 2,
            ArmKind::Arm5StaticBuffer => 0,
        }
    }

    pub fn approx_macs_per_token(&self) -> usize {
        match self.arm {
            ArmKind::Arm1BlackboardFixed | ArmKind::Arm4RetrainedShuffled => {
                // 64 cells * (160*64 + 64*32 + 64*32) * 2 ticks + 2*1024 = 1,837,056 MACs
                1_837_056
            }
            ArmKind::Arm2BlackboardAdaptive => 1_837_056,
            ArmKind::Arm3UselessCompute => {
                // 4 * (32*256 + 256*32) = 65,536 MACs
                65_536
            }
            ArmKind::Arm5StaticBuffer => 2_048,
        }
    }
}

/// Evaluates query token accuracy and returns (overall_acc, per_instance_correct)
pub fn evaluate_query_accuracy(
    logits: &Tensor,
    targets: &Tensor,
    loss_mask: &Tensor,
) -> Result<(f32, Vec<bool>)> {
    let (b, l, _) = logits.dims3()?;
    let preds = logits.argmax(candle_core::D::Minus1)?;
    let preds_vec = preds.to_vec2::<u32>()?;
    let targets_vec = targets.to_vec2::<u32>()?;
    let mask_vec = loss_mask.to_vec2::<f32>()?;

    let mut total_correct = 0;
    let mut total_queries = 0;
    let mut per_instance_correct = Vec::with_capacity(b);

    for i in 0..b {
        let mut inst_all_correct = true;
        let mut inst_has_queries = false;
        for j in 0..l {
            if mask_vec[i][j] == 1.0 {
                inst_has_queries = true;
                total_queries += 1;
                if preds_vec[i][j] == targets_vec[i][j] {
                    total_correct += 1;
                } else {
                    inst_all_correct = false;
                }
            }
        }
        per_instance_correct.push(inst_has_queries && inst_all_correct);
    }

    let acc = if total_queries > 0 {
        total_correct as f32 / total_queries as f32
    } else {
        0.0
    };

    Ok((acc, per_instance_correct))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmTrainResult {
    pub arm: ArmKind,
    pub arm_label: String,
    pub seed: u64,
    pub total_params: usize,
    pub recurrent_updates_per_token: usize,
    pub approx_macs_per_token: usize,
    pub train_loss: f32,
    pub train_acc: f32,
    pub val_acc: f32,
    pub learning_curve: Vec<(usize, f32, f32)>, // (epoch, loss, acc)
    pub per_sample_correct: Vec<bool>,
    pub stratified_accs: Vec<(String, f32)>,
    pub avg_bur: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterventionResult {
    pub label: String,
    pub accuracy: f32,
    pub acc_by_carry_depth: Vec<(usize, f32)>,
    pub bur_by_carry_depth: Vec<(usize, f32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveHaltingTelemetry {
    pub step_histogram: Vec<usize>,
    pub mean_steps: f32,
    pub median_steps: f32,
    pub p95_steps: f32,
    pub fraction_hitting_max: f32,
    pub failed_efficiency_criterion: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtrapolationEvaluation {
    pub in_distribution_acc: f32, // K <= 3
    pub extrap_plus1_acc: Option<f32>, // K = 4; None until measured separately
    pub extrap_plus2_acc: Option<f32>, // K = 5; None until measured separately
    pub long_partial_acc: f32, // Mixed depths, explicitly not an exact K panel
    pub extrap_plus3_acc: f32,    // K >= 6
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FalsificationReport {
    pub timestamp: String,
    pub git_commit: String,
    pub statistical_unit: String,
    pub validity_notes: Vec<String>,
    pub arms: Vec<ArmTrainResult>,
    pub interventions: Vec<InterventionResult>,
    pub adaptive_telemetry: AdaptiveHaltingTelemetry,
    pub extrapolation: ExtrapolationEvaluation,
    // Statistical Hypothesis Testing:
    pub h0_a_readout_delta: f32,
    pub h0_a_mcnemar_p: f64,
    pub h0_a_bootstrap_ci: (f32, f32),
    pub h0_a_rejected: bool,

    pub h0_b_static_delta: f32,
    pub h0_b_mcnemar_p: f64,
    pub h0_b_bootstrap_ci: (f32, f32),
    pub h0_b_rejected: bool,

    pub h0_c_topo_delta: f32,
    pub h0_c_mcnemar_p: f64,
    pub h0_c_bootstrap_ci: (f32, f32),
    pub h0_c_rejected: bool,

    pub h0_d_adaptive_advantage: bool,
    pub summary_verdict: String,
}

/// Evaluates model in memory-safe microbatches of `chunk_size` to eliminate any OOM risk
pub fn eval_model_batched(
    model: &DualSystemModel,
    engine: &TaskEngine,
    total_samples: usize,
    chunk_size: usize,
    seq_len: usize,
    stratum: Option<ArithmeticStratum>,
    max_carry_depth: Option<usize>,
    base_seed: usize,
    coord_mode: SpatialCoordMode,
    read_mode: ReadIntervention,
    write_mode: WriteIntervention,
    temporal_mode: TemporalIntervention,
    dynamics_mode: DynamicsMode,
    device: &Device,
) -> Result<(f32, Vec<bool>, f32)> {
    let mut total_correct = 0usize;
    let mut total_queries = 0usize;
    let mut per_sample_correct = Vec::with_capacity(total_samples);
    let mut bur_sum = 0.0f32;
    let mut n_chunks = 0usize;

    let mut samples_done = 0;
    while samples_done < total_samples {
        let chunk_len = (total_samples - samples_done).min(chunk_size);
        let chunk_seed = base_seed.wrapping_add(samples_done);
        let batch = engine.generate_stratified_arithmetic_batch(
            chunk_len,
            seq_len,
            stratum,
            max_carry_depth,
            chunk_seed,
            device,
        )?;

        let (logits, rollout_opt) = model.forward(
            &batch.inputs,
            coord_mode,
            read_mode,
            write_mode,
            temporal_mode,
            dynamics_mode,
            false,
        )?;

        let preds = logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;
        let tgts = batch.targets.to_vec2::<u32>()?;
        let mask = batch.loss_mask.to_vec2::<f32>()?;

        for i in 0..chunk_len {
            let mut sample_all_correct = true;
            let mut sample_has_queries = false;
            for j in 0..seq_len {
                if mask[i][j] == 1.0 {
                    sample_has_queries = true;
                    total_queries += 1;
                    if preds[i][j] == tgts[i][j] {
                        total_correct += 1;
                    } else {
                        sample_all_correct = false;
                    }
                }
            }
            per_sample_correct.push(sample_has_queries && sample_all_correct);
        }

        if let Some(r) = rollout_opt {
            bur_sum += r.bur;
            n_chunks += 1;
        }

        samples_done += chunk_len;
    }

    let overall_acc = if total_queries > 0 {
        total_correct as f32 / total_queries as f32
    } else {
        0.0
    };
    let avg_bur = if n_chunks > 0 {
        bur_sum / n_chunks as f32
    } else {
        0.0
    };

    Ok((overall_acc, per_sample_correct, avg_bur))
}

/// Evaluates an intervention configuration across `total_samples` in microbatches of `chunk_size`
/// and stratifies query token accuracy and BUR by carry depth.
pub fn eval_intervention_stratified(
    model: &DualSystemModel,
    engine: &TaskEngine,
    label: &str,
    total_samples: usize,
    chunk_size: usize,
    seq_len: usize,
    base_seed: usize,
    coord_mode: SpatialCoordMode,
    read_mode: ReadIntervention,
    write_mode: WriteIntervention,
    temporal_mode: TemporalIntervention,
    dynamics_mode: DynamicsMode,
    device: &Device,
) -> Result<(InterventionResult, Vec<bool>)> {
    let mut total_correct = 0usize;
    let mut total_queries = 0usize;
    let mut per_sample_correct = Vec::with_capacity(total_samples);
    let mut depth_correct = vec![0usize; 10];
    let mut depth_total = vec![0usize; 10];
    let mut depth_bur_sum = vec![0.0f32; 10];
    let mut depth_bur_count = vec![0usize; 10];

    let mut samples_done = 0;
    while samples_done < total_samples {
        let chunk_len = (total_samples - samples_done).min(chunk_size);
        let chunk_seed = base_seed.wrapping_add(samples_done);
        let batch = engine.generate_stratified_arithmetic_batch(
            chunk_len,
            seq_len,
            None,
            None,
            chunk_seed,
            device,
        )?;
        let chunk_depths = batch.carry_depths.as_ref().unwrap();

        let (logits, rollout_opt) = model.forward(
            &batch.inputs,
            coord_mode,
            read_mode,
            write_mode,
            temporal_mode,
            dynamics_mode,
            false,
        )?;

        let preds = logits.argmax(candle_core::D::Minus1)?.to_vec2::<u32>()?;
        let tgts = batch.targets.to_vec2::<u32>()?;
        let mask = batch.loss_mask.to_vec2::<f32>()?;
        let chunk_bur = rollout_opt.as_ref().map_or(0.0f32, |r| r.bur);

        for i in 0..chunk_len {
            let d = chunk_depths[i].min(9);
            depth_bur_sum[d] += chunk_bur;
            depth_bur_count[d] += 1;

            let mut sample_all_correct = true;
            let mut sample_has_queries = false;
            for j in 0..seq_len {
                if mask[i][j] == 1.0 {
                    sample_has_queries = true;
                    total_queries += 1;
                    depth_total[d] += 1;
                    if preds[i][j] == tgts[i][j] {
                        total_correct += 1;
                        depth_correct[d] += 1;
                    } else {
                        sample_all_correct = false;
                    }
                }
            }
            per_sample_correct.push(sample_has_queries && sample_all_correct);
        }

        samples_done += chunk_len;
    }

    let overall_acc = if total_queries > 0 {
        total_correct as f32 / total_queries as f32
    } else {
        0.0
    };

    let mut acc_by_depth = Vec::new();
    let mut bur_by_depth = Vec::new();
    for d in 0..10 {
        if depth_total[d] > 0 {
            acc_by_depth.push((d, depth_correct[d] as f32 / depth_total[d] as f32));
        }
        if depth_bur_count[d] > 0 {
            bur_by_depth.push((d, depth_bur_sum[d] / depth_bur_count[d] as f32));
        }
    }

    let res = InterventionResult {
        label: label.to_string(),
        accuracy: overall_acc,
        acc_by_carry_depth: acc_by_depth,
        bur_by_carry_depth: bur_by_depth,
    };

    Ok((res, per_sample_correct))
}

/// Reject invalid gradients before AdamW can corrupt fresh experiment weights.
pub fn ensure_finite_gradients(varmap: &VarMap, gradients: &candle_core::backprop::GradStore) -> Result<()> {
    for (name, var) in varmap.data().lock().unwrap().iter() {
        if let Some(gradient) = gradients.get(var) {
            anyhow::ensure!(gradient.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()),
                "non-finite gradient in {}; experiment stopped before optimizer update", name);
        }
    }
    Ok(())
}

/// Quantile of the observed integer step counts, using nearest rank.
pub fn step_count_quantile(histogram: &[usize], quantile: f64) -> f32 {
    let total: usize = histogram.iter().sum();
    if total == 0 { return 0.; }
    let rank = (quantile * total as f64).ceil().max(1.) as usize;
    let mut cumulative = 0;
    for (steps, count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= rank { return steps as f32; }
    }
    (histogram.len() - 1) as f32
}

/// Complete-answer effect, matching the binary outcomes supplied to McNemar.
pub fn paired_exact_delta(left: &[bool], right: &[bool]) -> f32 {
    assert_eq!(left.len(), right.len());
    if left.is_empty() { return 0.; }
    left.iter().zip(right).map(|(&a,&b)| i32::from(a)-i32::from(b)).sum::<i32>() as f32 / left.len() as f32
}

/// Trains a single arm model for the specified seed and epochs
pub fn train_arm(
    arm: ArmKind,
    seed: u64,
    epochs: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    seq_len: usize,
    lr: f64,
    device: &Device,
) -> Result<(DualSystemModel, ArmTrainResult)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let engine = TaskEngine::new();
    let model = DualSystemModel::new(vb, arm, engine.vocab.size(), 32)?;
    crate::baselines::initialize_seeded(&varmap, seed)?;

    let opt_params = ParamsAdamW {
        lr,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), opt_params)?;

    let coord_mode = match arm {
        ArmKind::Arm4RetrainedShuffled => SpatialCoordMode::FixedShuffled(seed),
        _ => SpatialCoordMode::Hilbert,
    };
    let dynamics_mode = match arm {
        ArmKind::Arm2BlackboardAdaptive => DynamicsMode::AdaptiveKinetic {
            max_steps: 6,
            min_steps: 1,
            threshold: 0.08,
        },
        ArmKind::Arm5StaticBuffer => DynamicsMode::StaticBuffer,
        _ => DynamicsMode::FixedTicks(2),
    };

    let mut learning_curve = Vec::with_capacity(epochs);
    let mut last_train_loss = 0.0f32;
    let mut last_train_acc = 0.0f32;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;

        for step in 0..steps_per_epoch {
            let step_seed = (seed as usize).wrapping_mul(1000).wrapping_add(epoch * 100 + step);
            // In training, restrict carry depth to <= 3 for extrapolation disentanglement
            let batch = engine.generate_stratified_arithmetic_batch(
                batch_size,
                seq_len,
                None,
                Some(3),
                step_seed,
                device,
            )?;

            let (logits, _) = model.forward(
                &batch.inputs,
                coord_mode,
                ReadIntervention::Active,
                WriteIntervention::Active,
                TemporalIntervention::None,
                dynamics_mode,
                true,
            )?;

            let loss = model
                .interface
                .masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
            let loss_val = loss.to_scalar::<f32>()?;

            anyhow::ensure!(loss_val.is_finite(), "non-finite loss in {:?}, seed {}, epoch {}, step {}", arm, seed, epoch, step);
            let gradients = loss.backward()?;
            ensure_finite_gradients(&varmap, &gradients)?;
            optimizer.step(&gradients)?;

            let (acc, _) = evaluate_query_accuracy(&logits, &batch.targets, &batch.loss_mask)?;
            epoch_loss += loss_val;
            epoch_acc += acc;
        }

        let mean_loss = epoch_loss / steps_per_epoch as f32;
        let mean_acc = epoch_acc / steps_per_epoch as f32;

        learning_curve.push((epoch + 1, mean_loss, mean_acc));
        last_train_loss = mean_loss;
        last_train_acc = mean_acc;

        println!(
            "    [Arm {} | Seed {:3}] Epoch {:2}/{:2} | Loss: {:.4} | Train Acc: {:5.2}%",
            arm.label(),
            seed,
            epoch + 1,
            epochs,
            mean_loss,
            mean_acc * 100.0
        );
    }

    // Evaluate on held-out N=256 stratified test set using microbatches of 32
    let (val_acc, per_sample_correct, avg_bur) = eval_model_batched(
        &model,
        &engine,
        256,
        32,
        seq_len,
        None,
        None,
        (seed as usize).wrapping_add(99999),
        coord_mode,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        dynamics_mode,
        device,
    )?;

    // Stratified accuracy checks across S1 (Ripple) and S4 (NoCarry) in microbatches
    let (s1_acc, _, _) = eval_model_batched(
        &model,
        &engine,
        64,
        32,
        seq_len,
        Some(ArithmeticStratum::MaxRipple),
        None,
        (seed as usize).wrapping_add(1001),
        coord_mode,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        dynamics_mode,
        device,
    )?;

    let (s4_acc, _, _) = eval_model_batched(
        &model,
        &engine,
        64,
        32,
        seq_len,
        Some(ArithmeticStratum::NoCarry),
        None,
        (seed as usize).wrapping_add(1004),
        coord_mode,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        dynamics_mode,
        device,
    )?;

    println!(
        "    --> Val Acc: {:5.2}% | S1 (Ripple): {:5.2}% | S4 (NoCarry): {:5.2}% | BUR: {:.3}",
        val_acc * 100.0,
        s1_acc * 100.0,
        s4_acc * 100.0,
        avg_bur
    );

    let stratified_accs = vec![
        ("S1_MaxRipple".to_string(), s1_acc),
        ("S4_NoCarry".to_string(), s4_acc),
    ];

    let result = ArmTrainResult {
        arm,
        arm_label: arm.label().to_string(),
        seed,
        total_params: model.total_parameters(),
        recurrent_updates_per_token: model.recurrent_updates_per_token(),
        approx_macs_per_token: model.approx_macs_per_token(),
        train_loss: last_train_loss,
        train_acc: last_train_acc,
        val_acc,
        learning_curve,
        per_sample_correct,
        stratified_accs,
        avg_bur,
    };

    Ok((model, result))
}

/// Runs the complete, immutable falsification experiment battery across arms, seeds, and interventions
pub fn run_falsification_battery(
    seeds: &[u64],
    epochs: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    seq_len: usize,
    lr: f64,
    device: &Device,
) -> Result<FalsificationReport> {
    anyhow::ensure!(!seeds.is_empty(), "at least one seed is required");
    anyhow::ensure!(epochs > 0 && steps_per_epoch > 0 && batch_size > 0 && seq_len >= 16,
        "epochs, steps, and batch must be positive; arithmetic seq_len must be >= 16");
    anyhow::ensure!(lr.is_finite() && lr > 0., "learning rate must be finite and positive");
    let engine = TaskEngine::new();
    let mut arm_results = Vec::new();
    let mut primary_arm1: Option<DualSystemModel> = None;
    let mut primary_arm2: Option<DualSystemModel> = None;

    // 1. Multi-Seed Training Battery across all 5 arms
    for &arm in &[
        ArmKind::Arm1BlackboardFixed,
        ArmKind::Arm2BlackboardAdaptive,
        ArmKind::Arm3UselessCompute,
        ArmKind::Arm4RetrainedShuffled,
        ArmKind::Arm5StaticBuffer,
    ] {
        println!("\n=== Training Comparative Arm: {} ===", arm.label());
        for &seed in seeds {
            let (model, res) = train_arm(
                arm,
                seed,
                epochs,
                steps_per_epoch,
                batch_size,
                seq_len,
                lr,
                device,
            )?;
            if arm == ArmKind::Arm1BlackboardFixed && primary_arm1.is_none() {
                primary_arm1 = Some(model);
            } else if arm == ArmKind::Arm2BlackboardAdaptive && primary_arm2.is_none() {
                primary_arm2 = Some(model);
            }
            arm_results.push(res);
        }
    }

    let primary_arm1 = primary_arm1.expect("Primary Arm 1 model required for intervention suite");
    let primary_arm2 = primary_arm2.expect("Primary Arm 2 model required for adaptive telemetry");

    println!("\n=== Executing Controlled Intervention Suite on Primary Arm 1 ===");
    let mut interventions = Vec::new();

    // Baseline Intact
    let (intact_res, intact_sample_corr) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Intact (Active Read/Write, Hilbert, T=2)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Intact: Acc = {:5.2}%", intact_res.accuracy * 100.0);
    interventions.push(intact_res);

    // H0_A: Zeroed Readout
    let (zeroed_res, zeroed_sample_corr) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Ablation H0_A: Zeroed Readout",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Zeroed,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] H0_A Zeroed Readout: Acc = {:5.2}%", zeroed_res.accuracy * 100.0);
    interventions.push(zeroed_res);

    // H0_B: Static T=0 Buffer
    let (static_res, static_sample_corr) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Ablation H0_B: Static Buffer T=0",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::StaticBuffer,
        device,
    )?;
    println!("  [Intervention] H0_B Static Buffer T=0: Acc = {:5.2}%", static_res.accuracy * 100.0);
    interventions.push(static_res);

    // Coordinate Perturbations: Inference-Only Shuffled & Per-Sample Shuffled
    let (inf_shuf_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Topology: Inference-Only Shuffled",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::InferenceOnlyShuffled(42),
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Inference-Only Shuffled: Acc = {:5.2}%", inf_shuf_res.accuracy * 100.0);
    interventions.push(inf_shuf_res);

    let (sample_shuf_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Topology: Per-Sample Shuffled",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::PerSampleShuffled(42),
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Per-Sample Shuffled: Acc = {:5.2}%", sample_shuf_res.accuracy * 100.0);
    interventions.push(sample_shuf_res);

    // Temporal Interventions (Early=L/4, Mid=L/2, Late=3L/4)
    let (temp_early_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Temporal: Zero Early (t=L/4=8)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::ZeroAtFraction(0.25),
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Zero Early (t=8): Acc = {:5.2}%", temp_early_res.accuracy * 100.0);
    interventions.push(temp_early_res);

    let (temp_mid_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Temporal: Zero Mid (t=L/2=16)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::ZeroAtFraction(0.50),
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Zero Mid (t=16): Acc = {:5.2}%", temp_mid_res.accuracy * 100.0);
    interventions.push(temp_mid_res);

    let (temp_late_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Temporal: Zero Late (t=3L/4=24)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::ZeroAtFraction(0.75),
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Zero Late (t=24): Acc = {:5.2}%", temp_late_res.accuracy * 100.0);
    interventions.push(temp_late_res);

    // Read vs Write Isolation
    let (iso_write_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Isolation: Write-Disabled (Reads Preserved)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Disabled,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Write-Disabled: Acc = {:5.2}%", iso_write_res.accuracy * 100.0);
    interventions.push(iso_write_res);

    let (iso_read_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Isolation: Read-Disabled (Writes Preserved)",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Zeroed,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Read-Disabled: Acc = {:5.2}%", iso_read_res.accuracy * 100.0);
    interventions.push(iso_read_res);

    let (iso_freeze_res, _) = eval_intervention_stratified(
        &primary_arm1,
        &engine,
        "Isolation: Freeze Writes After Step 16",
        256,
        32,
        seq_len,
        12345,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::FreezeAfter(16),
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;
    println!("  [Intervention] Freeze Writes @16: Acc = {:5.2}%", iso_freeze_res.accuracy * 100.0);
    interventions.push(iso_freeze_res);

    // 3. Adaptive Halting Telemetry on Arm 2 (first seed)
    println!("\n=== Evaluating Adaptive Halting Telemetry on Primary Arm 2 ===");
    let mut step_hist = vec![0usize; 16];
    let mut total_steps = 0usize;
    let mut hit_max_count = 0usize;
    let mut total_tokens = 0usize;

    let mut samples_done = 0;
    while samples_done < 256 {
        let chunk_len = (256 - samples_done).min(32);
        let chunk_seed = 12345usize.wrapping_add(samples_done);
        let batch = engine.generate_stratified_arithmetic_batch(
            chunk_len,
            seq_len,
            None,
            None,
            chunk_seed,
            device,
        )?;

        let (_, rollout_opt) = primary_arm2.forward(
            &batch.inputs,
            SpatialCoordMode::Hilbert,
            ReadIntervention::Active,
            WriteIntervention::Active,
            TemporalIntervention::None,
            DynamicsMode::AdaptiveKinetic {
                max_steps: 6,
                min_steps: 1,
                threshold: 0.08,
            },
            false,
        )?;

        if let Some(r) = rollout_opt {
            for (step_idx, &count) in r.step_histogram.iter().enumerate() {
                if step_idx < step_hist.len() {
                    step_hist[step_idx] += count * chunk_len;
                }
            }
            total_steps += r.total_steps_executed * chunk_len;
            total_tokens += chunk_len * seq_len;
            let chunk_hit_max = (r.hit_max_fraction * (chunk_len * seq_len) as f32).round() as usize;
            hit_max_count += chunk_hit_max;
        }

        samples_done += chunk_len;
    }

    let hit_max_frac = if total_tokens > 0 {
        hit_max_count as f32 / total_tokens as f32
    } else {
        0.0
    };
    let avg_steps = if total_tokens > 0 {
        total_steps as f32 / total_tokens as f32
    } else {
        0.0
    };
    let failed_halting = hit_max_frac > 0.80;

    println!(
        "  [Adaptive Telemetry] Avg Steps/Token: {:.2} | Hit Max Frac: {:5.2}% | Criterion Failed: {}",
        avg_steps,
        hit_max_frac * 100.0,
        failed_halting
    );

    let adaptive_telemetry = AdaptiveHaltingTelemetry {
        median_steps: step_count_quantile(&step_hist, 0.5),
        p95_steps: step_count_quantile(&step_hist, 0.95),
        mean_steps: avg_steps,
        step_histogram: step_hist,
        fraction_hitting_max: hit_max_frac,
        failed_efficiency_criterion: failed_halting,
    };

    // 4. Algorithmic Extrapolation (Carry Depth vs Extrapolation Distance)
    println!("\n=== Evaluating Algorithmic Extrapolation (Carry Depth Hold-Out) ===");
    let (in_acc, _, _) = eval_model_batched(
        &primary_arm1,
        &engine,
        128,
        32,
        seq_len,
        None,
        Some(3), // In-distribution: K <= 3
        54321,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;

    let (s1_acc, _, _) = eval_model_batched(
        &primary_arm1,
        &engine,
        128,
        32,
        seq_len,
        Some(ArithmeticStratum::MaxRipple), // Extrapolation +3: Hostile Ripple (K >= 6)
        None,
        54322,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;

    let (s2_acc, _, _) = eval_model_batched(
        &primary_arm1,
        &engine,
        128,
        32,
        seq_len,
        Some(ArithmeticStratum::LongPartial), // Extrapolation +1 (K=4) & +2 (K=5)
        None,
        54323,
        SpatialCoordMode::Hilbert,
        ReadIntervention::Active,
        WriteIntervention::Active,
        TemporalIntervention::None,
        DynamicsMode::FixedTicks(2),
        device,
    )?;

    println!("  [Depth panels] K<=3: {:.2}% | Mixed long-partial: {:.2}% | Max ripple: {:.2}%",
        in_acc * 100.0, s2_acc * 100.0, s1_acc * 100.0);

    let extrapolation = ExtrapolationEvaluation {
        in_distribution_acc: in_acc,
        extrap_plus1_acc: None,
        extrap_plus2_acc: None,
        long_partial_acc: s2_acc,
        extrap_plus3_acc: s1_acc,
    };

    // 5. Statistical Hypothesis Testing
    println!("\n=== Computing Statistical Hypothesis Tests (Paired Exact McNemar & Bootstrap) ===");
    // H0_A: Intact vs Zeroed Readout
    let h0_a_delta = paired_exact_delta(&intact_sample_corr, &zeroed_sample_corr);
    let mut b_count = 0;
    let mut c_count = 0;
    let n_eval = intact_sample_corr.len();
    let mut diffs_a = Vec::with_capacity(n_eval);
    for i in 0..n_eval {
        let y1 = if intact_sample_corr[i] { 1.0f32 } else { 0.0f32 };
        let y2 = if zeroed_sample_corr[i] { 1.0f32 } else { 0.0f32 };
        diffs_a.push(y1 - y2);
        if intact_sample_corr[i] && !zeroed_sample_corr[i] {
            b_count += 1;
        } else if !intact_sample_corr[i] && zeroed_sample_corr[i] {
            c_count += 1;
        }
    }
    let h0_a_p = exact_mcnemar_binomial(b_count, c_count);
    let h0_a_ci = bootstrap_difference_ci(&diffs_a, 2000, 0.01 / 3.0);
    let h0_a_rejected = h0_a_delta > 0.05 && h0_a_p < 0.01 / 3.0 && h0_a_ci.0 > 0.05;

    // H0_B: Intact vs Static T=0
    let h0_b_delta = paired_exact_delta(&intact_sample_corr, &static_sample_corr);
    let mut b_b = 0;
    let mut c_b = 0;
    let mut diffs_b = Vec::with_capacity(n_eval);
    for i in 0..n_eval {
        let y1 = if intact_sample_corr[i] { 1.0f32 } else { 0.0f32 };
        let y2 = if static_sample_corr[i] { 1.0f32 } else { 0.0f32 };
        diffs_b.push(y1 - y2);
        if intact_sample_corr[i] && !static_sample_corr[i] {
            b_b += 1;
        } else if !intact_sample_corr[i] && static_sample_corr[i] {
            c_b += 1;
        }
    }
    let h0_b_p = exact_mcnemar_binomial(b_b, c_b);
    let h0_b_ci = bootstrap_difference_ci(&diffs_b, 2000, 0.01 / 3.0);
    let h0_b_rejected = h0_b_delta > 0.05 && h0_b_p < 0.01 / 3.0 && h0_b_ci.0 > 0.05;

    // H0_C: Arm 1 (Hilbert) vs Arm 4 (Retrained Shuffled)
    let arm1_res = arm_results.iter().find(|r| r.arm == ArmKind::Arm1BlackboardFixed).unwrap();
    let arm4_res = arm_results.iter().find(|r| r.arm == ArmKind::Arm4RetrainedShuffled).unwrap();
    let h0_c_delta = paired_exact_delta(&arm1_res.per_sample_correct, &arm4_res.per_sample_correct);
    let mut b_c = 0;
    let mut c_c = 0;
    let n_c = arm1_res.per_sample_correct.len().min(arm4_res.per_sample_correct.len());
    let mut diffs_c = Vec::with_capacity(n_c);
    for i in 0..n_c {
        let y1 = if arm1_res.per_sample_correct[i] { 1.0f32 } else { 0.0f32 };
        let y2 = if arm4_res.per_sample_correct[i] { 1.0f32 } else { 0.0f32 };
        diffs_c.push(y1 - y2);
        if arm1_res.per_sample_correct[i] && !arm4_res.per_sample_correct[i] {
            b_c += 1;
        } else if !arm1_res.per_sample_correct[i] && arm4_res.per_sample_correct[i] {
            c_c += 1;
        }
    }
    let h0_c_p = exact_mcnemar_binomial(b_c, c_c);
    let h0_c_ci = bootstrap_difference_ci(&diffs_c, 2000, 0.01 / 3.0);
    let h0_c_rejected = h0_c_delta > 0.05 && h0_c_p < 0.01 / 3.0 && h0_c_ci.0 > 0.05;

    // Halting alone cannot establish an accuracy/compute advantage.
    let h0_d_adv = false;

    let mut verdicts = Vec::new();
    if h0_a_rejected {
        verdicts.push("H0_A: Complete-answer advantage over zeroed readout detected on the first-seed panel.");
    } else {
        verdicts.push("H0_A INCONCLUSIVE: Complete-answer advantage over zeroed readout was not established.");
    }

    if h0_b_rejected {
        verdicts.push("H0_B: Complete-answer advantage over inference-time T=0 detected on the first-seed panel.");
    } else {
        verdicts.push("H0_B INCONCLUSIVE: Complete-answer advantage over inference-time T=0 was not established.");
    }

    if h0_c_rejected {
        verdicts.push("H0_C: Complete-answer advantage over the retrained shuffled arm detected for the first seed.");
    } else {
        verdicts.push("H0_C INCONCLUSIVE: Complete-answer advantage over the retrained shuffled arm was not established.");
    }

    if failed_halting {
        verdicts.push("H0_D NOT ESTABLISHED: Adaptive halting hit its maximum on >80% of batch-token decisions.");
    } else {
        verdicts.push("H0_D NOT TESTED: Halting telemetry is not a matched accuracy/compute comparison.");
    }

    let summary_verdict = verdicts.join(" | ");

    Ok(FalsificationReport {
        timestamp: format!("unix:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()),
        git_commit: std::process::Command::new("git").args(["rev-parse", "HEAD"]).output()
            .ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into()),
        statistical_unit: "complete answer; paired first-seed comparisons; Bonferroni family alpha=0.01".into(),
        validity_notes: vec![
            "Git commit alone does not identify dirty source; archive source hashes with each run.".into(),
            "Digit accuracy is descriptive; all hypothesis deltas and intervals use complete-answer correctness.".into(),
            "Digit priors and observation duplicates must be audited before interpreting task performance.".into(),
            "Arm 3 is tokenwise recurrent refinement, not a sequence-memory baseline; MACs are not matched.".into(),
            "Carry-depth BUR values use batch aggregates and cannot establish per-instance utilization.".into(),
            "Adaptive decisions are shared within each microbatch; no efficiency advantage is tested.".into(),
            "Exact K=4 and K=5 extrapolation panels were not measured and are null.".into(),
        ],
        arms: arm_results,
        interventions,
        adaptive_telemetry,
        extrapolation,
        h0_a_readout_delta: h0_a_delta,
        h0_a_mcnemar_p: h0_a_p,
        h0_a_bootstrap_ci: h0_a_ci,
        h0_a_rejected,
        h0_b_static_delta: h0_b_delta,
        h0_b_mcnemar_p: h0_b_p,
        h0_b_bootstrap_ci: h0_b_ci,
        h0_b_rejected,
        h0_c_topo_delta: h0_c_delta,
        h0_c_mcnemar_p: h0_c_p,
        h0_c_bootstrap_ci: h0_c_ci,
        h0_c_rejected,
        h0_d_adaptive_advantage: h0_d_adv,
        summary_verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_answer_effect_and_step_quantiles_use_observed_units() {
        // Different numbers of correct digits do not change all-false exact outcomes.
        assert_eq!(paired_exact_delta(&[false, false], &[false, false]), 0.);
        assert_eq!(paired_exact_delta(&[true, true, false, false], &[false, true, false, false]), 0.25);
        let hist = [0, 248, 0, 0, 8];
        assert_eq!(step_count_quantile(&hist, 0.5), 1.);
        assert_eq!(step_count_quantile(&hist, 0.95), 1.);
        assert_eq!(step_count_quantile(&hist, 1.), 4.);
    }

    #[test]
    fn empty_falsification_seed_list_fails_before_training() {
        assert!(run_falsification_battery(&[], 1, 1, 1, 16, 0.001, &Device::Cpu).is_err());
    }

    #[test]
    fn opt_in_local_write_normalization_keeps_static_gradients_finite() -> Result<()> {
        let engine = TaskEngine::new();
        let batch = engine.generate_stratified_arithmetic_batch(2, 32, None, Some(3), 123, &Device::Cpu)?;
        for seed in [42, 101, 202] {
            let vm = VarMap::new();
            let mut model = DualSystemModel::new(VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu),
                ArmKind::Arm5StaticBuffer, engine.vocab.size(), 32)?;
            crate::baselines::initialize_seeded(&vm, seed)?;
            for local in [false, true] {
                model.blackboard.as_mut().unwrap().cfg.normalize_written_cell_only = local;
                let (logits, _) = model.forward(&batch.inputs, SpatialCoordMode::Hilbert,
                    ReadIntervention::Active, WriteIntervention::Active, TemporalIntervention::None,
                    DynamicsMode::StaticBuffer, true)?;
                let loss = model.interface.masked_cross_entropy_loss(&logits, &batch.targets, &batch.loss_mask)?;
                assert!(loss.to_scalar::<f32>()?.is_finite());
                let gradients = loss.backward()?;
                assert_eq!(ensure_finite_gradients(&vm, &gradients).is_ok(), local,
                    "seed {}, local {}", seed, local);
                if local {
                    let vars = vm.data().lock().unwrap();
                    let g = gradients.get(&vars["blackboard.write_proj.weight"]).unwrap();
                    assert!(g.abs()?.sum_all()?.to_scalar::<f32>()? > 0.);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_exact_mcnemar_binomial() {
        // Equal discordance -> p = 1.0
        assert_eq!(exact_mcnemar_binomial(10, 10), 1.0);
        assert_eq!(exact_mcnemar_binomial(0, 0), 1.0);

        // Extreme discordance -> p < 0.001
        let p_extreme = exact_mcnemar_binomial(25, 0);
        assert!(p_extreme < 1e-5, "Expected tiny p, got {}", p_extreme);

        // Moderate discordance: 15 vs 1 -> p < 0.01
        let p_mod = exact_mcnemar_binomial(15, 1);
        assert!(p_mod < 0.01, "Expected p < 0.01, got {}", p_mod);
    }

    #[test]
    fn test_combine_fisher_pvalues() {
        let pvals = vec![0.001, 0.005, 0.002];
        let combined = combine_fisher_pvalues(&pvals);
        assert!(combined < 1e-4, "Expected very small combined p, got {}", combined);

        let p_null = vec![0.5, 0.6, 0.7];
        let comb_null = combine_fisher_pvalues(&p_null);
        assert!(comb_null > 0.5, "Expected non-significant combined p, got {}", comb_null);
    }

    #[test]
    fn test_bootstrap_difference_ci() {
        let diffs = vec![0.1f32; 100];
        let (lo, hi) = bootstrap_difference_ci(&diffs, 500, 0.05);
        assert!((lo - 0.1).abs() < 1e-4);
        assert!((hi - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_useless_compute_1d_parameter_match() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let useless = UselessCompute1D::new(vb, 32, 256, 4)?;
        let p_count = useless.parameter_count();
        // 32*256+256 + 256*32+32 = 8448 + 8224 = 16672
        assert_eq!(p_count, 16_672);

        let x = Tensor::randn(0.0f32, 1.0f32, (2, 8, 32), &dev)?;
        let out = useless.forward(&x)?;
        assert_eq!(out.dims3()?, (2, 8, 32));

        Ok(())
    }
}
