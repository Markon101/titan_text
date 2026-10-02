use crate::config::TitanConfig;
use crate::dataset::SequenceDataset;
use crate::field::{FrequencyDecomposition, MorphogenicField};
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::TokenInterface;
use anyhow::Result;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Fixed stream seed for per-step validation batches (RD-018b). Kept at the
/// historical value so heldout evaluation is reproducible across calls.
pub const VAL_EVAL_SEED: usize = 0;

/// Fixed seed for the deduplicated heldout evaluation panel (vNext harness
/// convention, matching `titan_substrate`).
pub const HELDOUT_PANEL_SEED: usize = 9_000_001;

/// RD-018b heldout panel cap: the iterated-parity validation split has 789
/// unique rows at L=16. Use [`Trainer::evaluate_val_panel`] for other sizes.
pub const MAX_HELDOUT_ROWS: usize = 789;

/// Overdraw factor for the heldout panel: draw `rows * overdraw` validation
/// rows, then keep the first `rows` unique observations.
const HELDOUT_OVERDRAW: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepDiagnostics {
    pub train_loss: f32,
    pub train_acc: f32,
    pub val_loss: f32,
    pub val_acc: f32,
    /// Total gradient L2 norm before clipping (historical field value).
    pub grad_norm: f32,
    /// RD-018b pre-clip total gradient L2 norm (equals `grad_norm`).
    #[serde(default)]
    pub grad_norm_pre_clip: f32,
    /// RD-018b post-clip total gradient L2 norm (<= `config.train.grad_clip_norm`
    /// whenever clipping engaged). Non-finite values are a divergence signal;
    /// writers must finite-filter like the existing manifest fields.
    #[serde(default)]
    pub grad_norm_post_clip: f32,
    /// RD-018b stream seed used to draw this step's training batch
    /// (fixed 0 on the legacy path).
    #[serde(default)]
    pub train_stream_seed: usize,
    pub update_magnitude: f32,
    pub hidden_mean: f32,
    pub hidden_var: f32,
    pub state_energy: f32,
    pub enstrophy: f32,
    pub palinstrophy: f32,
    pub bkm_norm: f32,
    pub freq_decomp: FrequencyDecomposition,
    /// Mean auxiliary interior-slot CE this step (0.0 when aux disabled)
    pub aux_loss: f32,
    /// RD-018b per-tick auxiliary CE at post-update interior ticks, parallel
    /// to `aux_tick_indices` (empty on the legacy path or when aux is off).
    #[serde(default)]
    pub aux_tick_losses: Vec<f32>,
    /// RD-018b ticks at which `aux_tick_losses` were evaluated.
    #[serde(default)]
    pub aux_tick_indices: Vec<usize>,
}

/// RD-018b exhaustive heldout evaluation report (see
/// [`Trainer::evaluate_val_exhaustive`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValExhaustiveReport {
    /// Number of unique heldout rows actually evaluated.
    pub rows: usize,
    /// Masked (or full) cross-entropy aggregated over all evaluated cells.
    pub loss: f32,
    /// Overall token accuracy over evaluated cells.
    pub accuracy: f32,
    /// Accuracy at query-slot positions (s+1)*4-1 for each slot s.
    pub per_slot_accuracy: Vec<f32>,
    /// Number of scored cells contributing to each slot.
    pub per_slot_count: Vec<usize>,
}

/// RD-018b auxiliary supervision schedule for the fixed-horizon regime:
/// evenly spaced interior ticks at multiples of `max(t_target / 4, 1)`
/// strictly below the terminal tick, plus the terminal tick itself (covered
/// by the main terminal loss, so it is never duplicated). States are read
/// post-update at every tick.
pub fn aux_supervision_ticks(t_target: usize) -> Vec<usize> {
    let every = (t_target / 4).max(1);
    let mut ticks = Vec::new();
    let mut tick = every;
    while tick < t_target {
        ticks.push(tick);
        tick += every;
    }
    if t_target > 0 {
        ticks.push(t_target);
    }
    ticks
}

/// Interior auxiliary ticks only (the terminal tick is excluded because the
/// terminal readout already supervises the final state).
pub fn aux_interior_ticks(t_target: usize) -> Vec<usize> {
    let mut ticks = aux_supervision_ticks(t_target);
    ticks.pop();
    ticks
}

/// RD-018b deterministic weight initialization from an explicit seed,
/// independent of Candle's process-local RNG and HashMap iteration order.
/// Scheme by variable name: `*bias` -> zeros, embedding weights ->
/// normal(0, 0.02), other weights -> Kaiming-uniform with bound
/// `sqrt(6 / fan_in)`.
pub fn initialize_weights_seeded(varmap: &VarMap, seed: u64) -> Result<()> {
    let variables = varmap.data().lock().unwrap();
    let mut names: Vec<String> = variables.keys().cloned().collect();
    names.sort();
    let mut rng = StdRng::seed_from_u64(seed);
    for name in names {
        let var = &variables[&name];
        let dims = var.dims();
        let fan_in = dims.get(1).copied().unwrap_or(1).max(1);
        let values: Vec<f32> = if name.ends_with("bias") {
            vec![0.0; var.elem_count()]
        } else if name.contains("embed") {
            (0..var.elem_count())
                .map(|_| {
                    let v: f32 = StandardNormal.sample(&mut rng);
                    v * 0.02
                })
                .collect()
        } else {
            let bound = (6.0 / fan_in as f32).sqrt();
            (0..var.elem_count())
                .map(|_| rng.gen_range(-bound..bound))
                .collect()
        };
        var.set(&Tensor::from_vec(values, var.shape(), var.device())?.to_dtype(var.dtype())?)?;
    }
    Ok(())
}

pub struct Trainer {
    pub nca: NeuralCellularAutomaton,
    pub interface: TokenInterface,
    pub optimizer: AdamW,
    pub varmap: VarMap,
    pub config: TitanConfig,
    pub dataset: SequenceDataset,
    pub device: Device,
    pub step_count: usize,
    /// RD-018b clean-substrate mode: advancing training stream keyed by
    /// `step_count`, gradient clipping, dedicated aux head, post-update aux
    /// supervision, and per-tick aux telemetry. False = historical training
    /// semantics (fixed stream seed 0, no clipping, shared aux projection).
    pub vnext_substrate: bool,
    /// Max global L2 gradient norm used when `vnext_substrate` is true
    /// (0.0 disables clipping).
    pub grad_clip_norm: f32,
}

impl Trainer {
    /// Historical constructor: unseeded Candle initialization and legacy
    /// training semantics. Kept intact for old checkpoints and reproducibility
    /// of the historical protocol; use [`Trainer::new_seeded`] for the RD-018b
    /// clean substrate.
    pub fn new(config: TitanConfig, task: &str, device: &Device) -> Result<Self> {
        config.validate()?;
        let dataset = SequenceDataset::new(task);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let nca = NeuralCellularAutomaton::new(vb.pp("nca"), &config.nca, &config.field)?;
        let interface = TokenInterface::new(vb.pp("interface"), dataset.vocab.size(), config.field.channels)?;

        let opt_params = ParamsAdamW {
            lr: config.train.lr,
            weight_decay: config.train.weight_decay,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), opt_params)?;

        Ok(Self {
            nca,
            interface,
            optimizer,
            varmap,
            config,
            dataset,
            device: device.clone(),
            step_count: 0,
            vnext_substrate: false,
            grad_clip_norm: 0.0,
        })
    }

    /// RD-018b clean-substrate constructor. Deterministically initializes
    /// every parameter (including a dedicated auxiliary head) from `seed`,
    /// then enables the advancing data stream, gradient clipping, and
    /// post-update auxiliary supervision. Per-step validation keeps using the
    /// separate fixed [`VAL_EVAL_SEED`].
    pub fn new_seeded(
        config: TitanConfig,
        task: &str,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        let mut trainer = Self::new(config, task, device)?;
        let vb = VarBuilder::from_varmap(&trainer.varmap, DType::F32, device);
        trainer.interface.attach_aux_head(vb.pp("interface"))?;
        initialize_weights_seeded(&trainer.varmap, seed)?;
        trainer.vnext_substrate = true;
        trainer.grad_clip_norm = trainer.config.train.grad_clip_norm;
        Ok(trainer)
    }

    /// RD-018b stream seed for the training batch consumed by the next
    /// `train_step`. Fresh batches advance with `step_count`; the legacy path
    /// keeps the historical fixed seed 0.
    fn train_stream_seed(&self) -> usize {
        if self.vnext_substrate {
            self.step_count
        } else {
            0
        }
    }

    /// Draws the training batch for the next `train_step` (single source of
    /// truth shared with stream-advance tests).
    fn draw_train_batch(&self, batch_size: usize) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        self.dataset.sample_batch_seeded_with_mask(
            batch_size,
            self.config.field.seq_len,
            false,
            self.train_stream_seed(),
            &self.device,
        )
    }

    /// Canonically ordered variables (name-sorted) for order-independent
    /// floating-point reductions on the RD-018b path, so identical seeds give
    /// bit-identical runs across processes. The legacy path keeps the
    /// historical `all_vars()` order untouched.
    fn reduction_vars(&self) -> Vec<Var> {
        if !self.vnext_substrate {
            return self.varmap.all_vars();
        }
        let data = self.varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        names.iter().map(|name| data[*name].clone()).collect()
    }

    fn apply_target_slot_filter(&self, maybe_mask: Option<&Tensor>) -> Result<Option<Tensor>> {
        match (maybe_mask, self.config.train.target_slot) {
            (Some(mask), Some(slot)) => {
                let (b, l) = mask.dims2()?;
                let mut mask_vals = mask.to_vec2::<f32>()?;
                let chunk_size = 4;
                let target_pos = (slot + 1) * chunk_size - 1;
                for row in mask_vals.iter_mut() {
                    for i in 0..l {
                        if i != target_pos {
                            row[i] = 0.0;
                        }
                    }
                }
                Ok(Some(Tensor::from_vec(
                    mask_vals.into_iter().flatten().collect(),
                    (b, l),
                    &self.device,
                )?))
            }
            (Some(mask), None) => Ok(Some(mask.clone())),
            (None, _) => Ok(None),
        }
    }

    /// Builds a [B, L] mask with 1.0 at interior slot positions (slots 1..=num_slots-2,
    /// i.e. excluding the boundary slot 0 and the final readout slot) for auxiliary
    /// deep supervision. Slot chunk size is fixed at 4 by the dataset convention.
    fn aux_interior_mask(&self, batch_size: usize) -> Result<Tensor> {
        let l = self.config.field.seq_len;
        let num_slots = l / 4;
        anyhow::ensure!(
            num_slots >= 3,
            "--aux-supervision requires seq_len covering at least 3 slots (L >= 12)"
        );
        let mut mask = vec![0.0f32; batch_size * l];
        for s in 1..=(num_slots - 2) {
            let pos = (s + 1) * 4 - 1;
            for b in 0..batch_size {
                mask[b * l + pos] = 1.0;
            }
        }
        Ok(Tensor::from_vec(mask, (batch_size, l), &self.device)?)
    }

    /// Auxiliary interior-slot cross-entropy on given logits. Under --aux-sham,
    /// targets are replaced by a constant token id 1 through identical machinery.
    fn aux_interior_loss(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        aux_mask: &Tensor,
    ) -> Result<Tensor> {
        let aux_targets: Tensor = if self.config.train.aux_sham {
            let (b, l) = targets.dims2()?;
            Tensor::from_vec(vec![1u32; b * l], (b, l), &self.device)?
        } else {
            targets.clone()
        };
        self.interface.masked_cross_entropy_loss(logits, &aux_targets, aux_mask)
    }

    /// Evaluates model performance on a fixed held-out validation batch.
    /// Uses the fixed [`VAL_EVAL_SEED`] so repeated calls are reproducible.
    pub fn evaluate_val(&self, batch_size: usize) -> Result<(f32, f32)> {
        let (val_inputs, val_targets, raw_mask) = self.dataset.sample_batch_seeded_with_mask(
            batch_size,
            self.config.field.seq_len,
            true,
            VAL_EVAL_SEED,
            &self.device,
        )?;
        let maybe_mask = self.apply_target_slot_filter(raw_mask.as_ref())?;
        let seed = self.interface.embed_tokens(&val_inputs)?;
        let mut field = MorphogenicField::from_tensor(seed, &self.config.field);

        for _ in 0..self.config.train.dev_steps {
            field = self.nca.step_field(&field, &self.device)?;
        }

        let logits = self.interface.logits(&field.x)?;
        let (loss, acc) = if let Some(ref mask) = maybe_mask {
            let l = self.interface.masked_cross_entropy_loss(&logits, &val_targets, mask)?.to_scalar::<f32>()?;
            let a = self.interface.masked_accuracy(&logits, &val_targets, mask)?;
            (l, a)
        } else {
            let l = self.interface.cross_entropy_loss(&logits, &val_targets)?.to_scalar::<f32>()?;
            let a = self.interface.accuracy(&logits, &val_targets)?;
            (l, a)
        };

        Ok((loss, acc))
    }

    /// RD-018b exhaustive heldout evaluation: evaluates the deduplicated
    /// heldout panel (up to [`MAX_HELDOUT_ROWS`] unique rows, fixed
    /// [`HELDOUT_PANEL_SEED`]) in chunks of `batch_size`, reporting overall
    /// loss/accuracy plus per-query-slot accuracy.
    pub fn evaluate_val_exhaustive(&self, batch_size: usize) -> Result<ValExhaustiveReport> {
        self.evaluate_val_panel(MAX_HELDOUT_ROWS, batch_size)
    }

    /// Like [`Trainer::evaluate_val_exhaustive`] but with an explicit panel
    /// size (`max_rows`); stops at the number of unique heldout rows actually
    /// available.
    pub fn evaluate_val_panel(
        &self,
        max_rows: usize,
        batch_size: usize,
    ) -> Result<ValExhaustiveReport> {
        anyhow::ensure!(batch_size > 0, "eval batch size must be positive");
        anyhow::ensure!(max_rows > 0, "heldout panel size must be positive");
        let (inputs, targets, maybe_mask) = self.heldout_panel(max_rows)?;
        let rows = inputs.dim(0)?;
        let seq_len = self.config.field.seq_len;
        let num_slots = seq_len / 4;

        let mut slot_hits = vec![0usize; num_slots];
        let mut slot_count = vec![0usize; num_slots];
        let mut hits = 0usize;
        let mut count = 0usize;
        let mut loss_numer = 0.0f64;
        let mut loss_denom = 0.0f64;

        let mut start = 0usize;
        while start < rows {
            let n = batch_size.min(rows - start);
            let chunk_inputs = inputs.narrow(0, start, n)?;
            let chunk_targets = targets.narrow(0, start, n)?;
            let chunk_mask = match &maybe_mask {
                Some(mask) => self
                    .apply_target_slot_filter(Some(&mask.narrow(0, start, n)?))?,
                None => None,
            };

            let seed_embeddings = self.interface.embed_tokens(&chunk_inputs)?;
            let mut field = MorphogenicField::from_tensor(seed_embeddings, &self.config.field);
            for _ in 0..self.config.train.dev_steps {
                field = self.nca.step_field(&field, &self.device)?;
            }
            let logits = self.interface.logits(&field.x)?;

            let (chunk_loss, active) = if let Some(ref mask) = chunk_mask {
                let l = self.interface.masked_cross_entropy_loss(&logits, &chunk_targets, mask)?.to_scalar::<f32>()?;
                let cells = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
                (l, cells)
            } else {
                let l = self.interface.cross_entropy_loss(&logits, &chunk_targets)?.to_scalar::<f32>()?;
                (l, (n * seq_len) as f32)
            };
            loss_numer += chunk_loss as f64 * active as f64;
            loss_denom += active as f64;

            let preds = logits
                .argmax(candle_core::D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec2::<u32>()?;
            let tgts = chunk_targets.to_dtype(DType::U32)?.to_vec2::<u32>()?;
            let mask_vals = chunk_mask
                .as_ref()
                .map(|mask| mask.to_vec2::<f32>())
                .transpose()?;
            for (r, (pred_row, tgt_row)) in preds.iter().zip(tgts.iter()).enumerate() {
                for pos in 0..seq_len {
                    let scored = match &mask_vals {
                        Some(mask) => mask[r][pos] > 0.5,
                        None => true,
                    };
                    if !scored {
                        continue;
                    }
                    let correct = pred_row[pos] == tgt_row[pos];
                    hits += usize::from(correct);
                    count += 1;
                    if pos % 4 == 3 {
                        let slot = pos / 4;
                        if slot < num_slots {
                            slot_count[slot] += 1;
                            slot_hits[slot] += usize::from(correct);
                        }
                    }
                }
            }
            start += n;
        }

        Ok(ValExhaustiveReport {
            rows,
            loss: if loss_denom > 0.0 { (loss_numer / loss_denom) as f32 } else { 0.0 },
            accuracy: if count > 0 { hits as f32 / count as f32 } else { 0.0 },
            per_slot_accuracy: slot_hits
                .iter()
                .zip(slot_count.iter())
                .map(|(hits, total)| if *total > 0 { *hits as f32 / *total as f32 } else { 0.0 })
                .collect(),
            per_slot_count: slot_count,
        })
    }

    /// Deduplicated heldout panel drawn with the fixed panel seed from the
    /// reserved validation split (vNext harness convention: overdraw, then
    /// keep the first unique observations).
    fn heldout_panel(&self, max_rows: usize) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let seq_len = self.config.field.seq_len;
        let (raw_inputs, raw_targets, raw_mask) = self.dataset.sample_batch_seeded_with_mask(
            max_rows * HELDOUT_OVERDRAW,
            seq_len,
            true,
            HELDOUT_PANEL_SEED,
            &self.device,
        )?;
        let mut seen: HashSet<Vec<u32>> = HashSet::new();
        let mut indices: Vec<u32> = Vec::new();
        for (i, row) in raw_inputs.to_vec2::<u32>()?.into_iter().enumerate() {
            if seen.insert(row) {
                indices.push(i as u32);
                if indices.len() >= max_rows {
                    break;
                }
            }
        }
        anyhow::ensure!(!indices.is_empty(), "heldout panel produced no unique rows");
        let idx = Tensor::new(indices.as_slice(), &self.device)?;
        let inputs = raw_inputs.index_select(&idx, 0)?;
        let targets = raw_targets.index_select(&idx, 0)?;
        let mask = raw_mask.map(|mask| mask.index_select(&idx, 0)).transpose()?;
        Ok((inputs, targets, mask))
    }

    /// Performs one training step with complete dynamical diagnostics and horizon-robust training regimes
    pub fn train_step(&mut self, batch_size: usize) -> Result<StepDiagnostics> {
        // RD-018b: the vNext path advances the training stream with
        // `step_count`; the legacy path keeps the historical fixed seed 0.
        let stream_seed = self.train_stream_seed();
        let (inputs, targets, raw_mask) = self.draw_train_batch(batch_size)?;
        let maybe_mask = self.apply_target_slot_filter(raw_mask.as_ref())?;

        // 1. Embed initial tokens into continuous seed field: [B, L, C]
        let seed_embeddings = self.interface.embed_tokens(&inputs)?;
        let mut field = MorphogenicField::from_tensor(seed_embeddings, &self.config.field);

        // 2. Select horizon based on horizon_mode
        let base_t = self.config.train.dev_steps;
        let mode = self.config.train.horizon_mode.as_str();
        let (t_target, is_multi, is_tail) = match mode {
            "randomized" => {
                let span = self.config.train.horizon_max.saturating_sub(self.config.train.horizon_min).max(1);
                let hash = ((self.step_count as u64).wrapping_mul(6364136223846793005).wrapping_add(self.config.train.seed ^ 0x9E3779B97F4A7C15)) as usize;
                let t = self.config.train.horizon_min + (hash % (span + 1));
                (t, false, false)
            }
            "jitter" => {
                let j = self.config.train.horizon_jitter;
                let delta = if j > 0 { (self.step_count % (2 * j + 1)) as isize - j as isize } else { 0 };
                let t = (base_t as isize + delta).max(self.config.train.horizon_min as isize).min(self.config.train.horizon_max as isize) as usize;
                (t, false, false)
            }
            "multi_tick" => (self.config.train.horizon_max, true, false),
            "stability_tail" => (base_t, false, true),
            _ => (base_t, false, false),
        };

        let mut total_update_mag = 0.0f32;
        let mut steps_executed = 0usize;
        let mut ce_loss_acc = None;
        let mut final_logits = None;

        // Auxiliary deep supervision: per-tick masked CE at interior slot positions
        let aux_weight = self.config.train.aux_supervision_weight;
        let aux_mask = if aux_weight > 0.0 {
            Some(self.aux_interior_mask(batch_size)?)
        } else {
            None
        };
        let mut aux_acc: Option<Tensor> = None;
        let mut aux_count = 0f32;
        // RD-018b per-tick auxiliary telemetry: CE values at the supervised
        // interior ticks (parallel index/value vectors, empty on the legacy
        // path where only the folding mean is recorded).
        let mut aux_tick_losses: Vec<f32> = Vec::new();
        let mut aux_tick_indices: Vec<usize> = Vec::new();

        if is_multi {
            // Multi-tick supervision: accumulate loss from horizon_min to horizon_max
            let h_min = self.config.train.horizon_min;
            let h_max = self.config.train.horizon_max;
            let mut loss_count = 0f32;

            for t in 1..=h_max {
                let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
                total_update_mag += update_mag;
                steps_executed += 1;
                field = next_field;

                if t >= h_min {
                    let logits_t = self.interface.logits(&field.x)?;
                    let loss_t = if let Some(ref mask) = maybe_mask {
                        self.interface.masked_cross_entropy_loss(&logits_t, &targets, mask)?
                    } else {
                        self.interface.cross_entropy_loss(&logits_t, &targets)?
                    };
                    ce_loss_acc = match ce_loss_acc {
                        None => Some(loss_t),
                        Some(prev) => Some((&prev + &loss_t)?),
                    };
                    loss_count += 1.0;
                    if let Some(ref am) = aux_mask {
                        // Dedicated aux head on the RD-018b path (shared
                        // terminal projection on the legacy path).
                        let aux_logits_t = self.interface.aux_logits(&field.x)?;
                        let al = self.aux_interior_loss(&aux_logits_t, &targets, am)?;
                        if self.vnext_substrate {
                            aux_tick_losses.push(al.to_scalar::<f32>()?);
                            aux_tick_indices.push(t);
                        }
                        aux_acc = match aux_acc {
                            None => Some(al),
                            Some(prev) => Some((&prev + &al)?),
                        };
                        aux_count += 1.0;
                    }
                    final_logits = Some(logits_t);
                }
            }
            if let Some(total_loss) = ce_loss_acc {
                ce_loss_acc = Some((total_loss / (loss_count.max(1.0) as f64))?);
            }
        } else if is_tail {
            // Stability-tail supervision: unroll to base_t, then supervise K tail ticks
            let tail_k = self.config.train.stability_tail;
            for _ in 0..base_t {
                let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
                total_update_mag += update_mag;
                steps_executed += 1;
                field = next_field;
            }
            let logits_t = self.interface.logits(&field.x)?;
            let loss_t = if let Some(ref mask) = maybe_mask {
                self.interface.masked_cross_entropy_loss(&logits_t, &targets, mask)?
            } else {
                self.interface.cross_entropy_loss(&logits_t, &targets)?
            };
            let mut tail_loss = loss_t;
            if let Some(ref am) = aux_mask {
                let aux_logits_t = self.interface.aux_logits(&field.x)?;
                let al = self.aux_interior_loss(&aux_logits_t, &targets, am)?;
                if self.vnext_substrate {
                    aux_tick_losses.push(al.to_scalar::<f32>()?);
                    aux_tick_indices.push(base_t);
                }
                aux_acc = Some(al);
                aux_count += 1.0;
            }
            final_logits = Some(logits_t);

            for k in 0..tail_k {
                let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
                total_update_mag += update_mag;
                steps_executed += 1;
                field = next_field;
                let logits_k = self.interface.logits(&field.x)?;
                let loss_k = if let Some(ref mask) = maybe_mask {
                    self.interface.masked_cross_entropy_loss(&logits_k, &targets, mask)?
                } else {
                    self.interface.cross_entropy_loss(&logits_k, &targets)?
                };
                let scaled_k = (loss_k * 0.5)?;
                tail_loss = (&tail_loss + &scaled_k)?;
                if let Some(ref am) = aux_mask {
                    let aux_logits_k = self.interface.aux_logits(&field.x)?;
                    let al = self.aux_interior_loss(&aux_logits_k, &targets, am)?;
                    if self.vnext_substrate {
                        aux_tick_losses.push(al.to_scalar::<f32>()?);
                        aux_tick_indices.push(base_t + 1 + k);
                    }
                    aux_acc = match aux_acc {
                        None => Some(al),
                        Some(prev) => Some((&prev + &al)?),
                    };
                    aux_count += 1.0;
                }
            }
            let weight_sum = 1.0 + (tail_k as f64) * 0.5;
            ce_loss_acc = Some((tail_loss / weight_sum)?);
        } else {
            // Fixed / Randomized / Jitter single target horizon with optional contractive tail equilibrium loss
            let tail_eq_weight = self.config.train.tail_equilibrium_weight;
            let tail_eq_k = self.config.train.tail_equilibrium_ticks.max(1);
            let tail_start = t_target.saturating_sub(tail_eq_k) + 1;
            let mut tail_eq_loss_acc = None;
            let mut tail_count = 0f32;

            // RD-018b schedule: evenly spaced interior ticks strictly below the
            // terminal tick. The terminal tick is excluded because the final
            // readout already supervises that state (no duplicate).
            let aux_ticks = if self.vnext_substrate {
                aux_interior_ticks(t_target)
            } else {
                Vec::new()
            };

            for step_idx in 1..=t_target {
                let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
                total_update_mag += update_mag;
                steps_executed += 1;

                if self.vnext_substrate {
                    // RD-018b: supervise POST-update states at scheduled ticks
                    // through the dedicated aux head, with per-tick telemetry.
                    if aux_weight > 0.0 && aux_ticks.contains(&step_idx) {
                        let logits_t = self.interface.aux_logits(&next_field.x)?;
                        if let Some(ref am) = aux_mask {
                            let al = self.aux_interior_loss(&logits_t, &targets, am)?;
                            aux_tick_losses.push(al.to_scalar::<f32>()?);
                            aux_tick_indices.push(step_idx);
                            aux_acc = match aux_acc {
                                None => Some(al),
                                Some(prev) => Some((&prev + &al)?),
                            };
                            aux_count += 1.0;
                        }
                    }
                } else if aux_weight > 0.0 {
                    // Historical legacy aux: pre-update read at multiples of
                    // t_target/4 (kept byte-for-byte for old checkpoints).
                    let aux_every = (t_target / 4).max(1);
                    if step_idx % aux_every == 0 {
                        let logits_t = self.interface.logits(&field.x)?;
                        if let Some(ref am) = aux_mask {
                            let al = self.aux_interior_loss(&logits_t, &targets, am)?;
                            aux_acc = match aux_acc {
                                None => Some(al),
                                Some(prev) => Some((&prev + &al)?),
                            };
                            aux_count += 1.0;
                        }
                    }
                }

                if tail_eq_weight > 0.0 && step_idx >= tail_start {
                    let diff = (&next_field.x - &field.x)?;
                    let num = diff.sqr()?.mean_all()?;
                    let den = (field.x.sqr()?.mean_all()? + 1e-4)?;
                    let rel_step_loss = (num / den)?;
                    tail_eq_loss_acc = match tail_eq_loss_acc {
                        None => Some(rel_step_loss),
                        Some(acc) => Some((&acc + &rel_step_loss)?),
                    };
                    tail_count += 1.0;
                }

                field = next_field;
            }
            let logits = self.interface.logits(&field.x)?;
            let loss = if let Some(ref mask) = maybe_mask {
                self.interface.masked_cross_entropy_loss(&logits, &targets, mask)?
            } else {
                self.interface.cross_entropy_loss(&logits, &targets)?
            };

            let combined_loss = if let Some(eq_loss) = tail_eq_loss_acc {
                let mean_eq = (eq_loss / (tail_count.max(1.0) as f64))?;
                let scaled_eq = (mean_eq * (tail_eq_weight as f64))?;
                (&loss + &scaled_eq)?
            } else {
                loss
            };

            ce_loss_acc = Some(combined_loss);
            final_logits = Some(logits.clone());
            if !self.vnext_substrate {
                if let Some(ref am) = aux_mask {
                    let al = self.aux_interior_loss(&logits, &targets, am)?;
                    aux_acc = match aux_acc {
                        None => Some(al),
                        Some(prev) => Some((&prev + &al)?),
                    };
                    aux_count += 1.0;
                }
            }
        }

        // Combine auxiliary deep supervision into the total loss: L + alpha * mean(aux)
        let mut aux_diag_value = 0.0f32;
        let ce_loss = match (ce_loss_acc, aux_acc, aux_weight > 0.0) {
            (Some(ce), Some(aux_acc_t), true) => {
                let mean_aux = (aux_acc_t / (aux_count.max(1.0) as f64))?;
                aux_diag_value = mean_aux.to_scalar::<f32>()?;
                (&ce + &(mean_aux * (aux_weight as f64))?)?
            }
            (Some(ce), _, _) => ce,
            (None, _, _) => anyhow::bail!("training step produced no cross-entropy loss"),
        };

        let logits = final_logits.ok_or_else(|| anyhow::anyhow!("training step produced no logits"))?;
        let avg_update_mag = total_update_mag / steps_executed.max(1) as f32;


        let train_loss = ce_loss.to_scalar::<f32>()?;
        let train_acc = if let Some(ref mask) = maybe_mask {
            self.interface.masked_accuracy(&logits, &targets, mask)?
        } else {
            self.interface.accuracy(&logits, &targets)?
        };

        self.step_count += 1;

        // 5. Backpropagation, exact gradient norm, and RD-018b clipping
        let grads = ce_loss.backward()?;
        let mut sum_sq_grad = 0.0f32;
        for var in self.reduction_vars() {
            if let Some(g) = grads.get(&var) {
                let sq: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                sum_sq_grad += sq;
            }
        }
        let grad_norm_pre_clip = sum_sq_grad.sqrt();
        let mut grads = grads;
        let grad_norm_post_clip = self.clip_gradients_in_place(&mut grads, grad_norm_pre_clip)?;

        // 6. Optimizer update step
        self.optimizer.step(&grads)?;

        // 7. Field state metrics: mean, variance, energy, enstrophy, palinstrophy, BKM norm, and frequency decomposition
        let (hidden_mean, hidden_var) = field.mean_and_var()?;
        let state_energy = field.energy()?;
        let enstrophy = field.enstrophy()?;
        let palinstrophy = field.palinstrophy()?;
        let bkm_norm = field.bkm_norm()?;
        let freq_decomp = field.spatial_frequency_decomposition()?;

        // 8. Held-out validation evaluation
        let (val_loss, val_acc) = self.evaluate_val(batch_size)?;

        Ok(StepDiagnostics {
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            grad_norm: grad_norm_pre_clip,
            grad_norm_pre_clip,
            grad_norm_post_clip,
            train_stream_seed: stream_seed,
            update_magnitude: avg_update_mag,
            hidden_mean,
            hidden_var,
            state_energy,
            enstrophy,
            palinstrophy,
            bkm_norm,
            freq_decomp,
            aux_loss: aux_diag_value,
            aux_tick_losses,
            aux_tick_indices,
        })
    }

    /// RD-018b global-norm gradient clipping. Clipping is engaged only when
    /// `grad_clip_norm > 0` (vNext path) and the pre-clip norm is finite and
    /// above the bound; every gradient in `grads` is rescaled in place.
    /// Returns the post-clip total L2 norm (equal to the pre-clip norm when
    /// clipping did not engage).
    fn clip_gradients_in_place(
        &self,
        grads: &mut GradStore,
        grad_norm_pre_clip: f32,
    ) -> Result<f32> {
        if self.grad_clip_norm <= 0.0
            || !grad_norm_pre_clip.is_finite()
            || grad_norm_pre_clip <= self.grad_clip_norm
        {
            return Ok(grad_norm_pre_clip);
        }
        let scale = self.grad_clip_norm / (grad_norm_pre_clip + 1e-6);
        for var in self.reduction_vars() {
            if let Some(g) = grads.get(&var) {
                let scaled = (g * scale as f64)?;
                grads.insert(&var, scaled);
            }
        }
        // Re-measure the clipped norm and guarantee the recorded bound.
        let mut sum_sq = 0.0f32;
        for var in self.reduction_vars() {
            if let Some(g) = grads.get(&var) {
                let sq: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                sum_sq += sq;
            }
        }
        Ok(sum_sq.sqrt().min(self.grad_clip_norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointManager, ModelManifest, SCHEMA_VERSION};

    #[test]
    fn test_resume_training_from_checkpoint() -> Result<()> {
        let dev = Device::Cpu;
        let mut config = TitanConfig::default();
        config.field.seq_len = 16;
        config.field.channels = 16;
        config.nca.hidden_dim = 32;
        config.train.dev_steps = 2;
        config.train.batch_size = 2;

        let temp_path = std::env::temp_dir().join(format!("titan_test_resume_{}_{}", std::process::id(), rand::random::<u32>()));
        let temp_dir = temp_path.to_str().unwrap();

        // 1. Initial training session (3 steps)
        let mut trainer1 = Trainer::new(config.clone(), "text", &dev)?;
        let mut last_diag = None;
        for _ in 0..3 {
            last_diag = Some(trainer1.train_step(config.train.batch_size)?);
        }
        let diag1 = last_diag.unwrap();
        let val_eval1 = trainer1.evaluate_val(config.train.batch_size)?;

        // 2. Save checkpoint
        let mut manifest = ModelManifest {
            schema_version: SCHEMA_VERSION,
            git_commit: "test_git".to_string(),
            random_seed: 42,
            cumulative_step: 3,
            dev_steps: config.train.dev_steps,
            train_loss: Some(diag1.train_loss).filter(|v| v.is_finite()),
            train_accuracy: Some(diag1.train_acc).filter(|v| v.is_finite()),
            val_loss: Some(diag1.val_loss).filter(|v| v.is_finite()),
            val_accuracy: Some(diag1.val_acc).filter(|v| v.is_finite()),
            grad_norm: Some(diag1.grad_norm).filter(|v| v.is_finite()),
            state_energy: Some(diag1.state_energy).filter(|v| v.is_finite()),
            param_count: 100,
            checkpoint_hash: "".to_string(),
            timestamp_unix: 123456,
            config: config.clone(),
            task: "text".to_string(),
        };
        CheckpointManager::save(temp_dir, &mut manifest, &trainer1.varmap)?;

        // 3. Resume training session in new Trainer instance
        let loaded_manifest = CheckpointManager::load_manifest(temp_dir)?;
        assert_eq!(loaded_manifest.cumulative_step, 3);
        assert_eq!(loaded_manifest.task, "text");

        let mut trainer2 = Trainer::new(loaded_manifest.config.clone(), &loaded_manifest.task, &dev)?;
        CheckpointManager::load_weights(temp_dir, &mut trainer2.varmap, &dev)?;

        // Verify loaded weights produce identical validation loss & accuracy as trainer1
        let val_eval2 = trainer2.evaluate_val(config.train.batch_size)?;
        assert_eq!(val_eval1.0, val_eval2.0, "Validation loss must match exactly on resumed trainer");
        assert_eq!(val_eval1.1, val_eval2.1, "Validation accuracy must match exactly on resumed trainer");

        // Train 3 more steps
        let diag2 = trainer2.train_step(config.train.batch_size)?;
        assert!(!diag2.train_loss.is_nan());
        assert!(!diag2.grad_norm.is_nan());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
        Ok(())
    }

    #[test]
    fn test_tail_equilibrium_loss_backprop() -> Result<()> {
        let dev = Device::Cpu;
        let mut config = TitanConfig::default();
        config.field.seq_len = 16;
        config.field.channels = 16;
        config.nca.hidden_dim = 32;
        config.train.dev_steps = 4;
        config.train.batch_size = 2;
        config.train.tail_equilibrium_weight = 0.5;
        config.train.tail_equilibrium_ticks = 2;

        let mut trainer = Trainer::new(config, "text", &dev)?;
        let diag = trainer.train_step(2)?;

        assert!(diag.train_loss > 0.0, "Train loss must be positive");
        assert!(diag.grad_norm > 0.0, "Gradients through equilibrium loss must be non-zero");
        assert!(diag.train_loss.is_finite(), "Train loss must be finite");
        assert!(diag.grad_norm.is_finite(), "Grad norm must be finite");

        Ok(())
    }

    /// Small algorithmic config shared by the RD-018b acceptance tests.
    fn vnext_test_config() -> TitanConfig {
        let mut config = TitanConfig::default();
        config.field.seq_len = 16;
        config.field.channels = 16;
        config.nca.hidden_dim = 32;
        config.train.dev_steps = 4;
        config.train.batch_size = 2;
        config.train.epochs = 1;
        config.train.lr = 0.003;
        config
    }

    fn var_values(trainer: &Trainer, name: &str) -> Result<Vec<f32>> {
        let data = trainer.varmap.data().lock().unwrap();
        let var = data
            .get(name)
            .unwrap_or_else(|| panic!("missing variable {name}"));
        Ok(var.as_tensor().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// RD-018b acceptance 1: deterministic seeded initialization. Two seeded
    /// trainers on the same seed are bit-identical (weights and first
    /// train_step loss); a different seed gives different weights.
    #[test]
    fn test_seeded_init_is_deterministic() -> Result<()> {
        let dev = Device::Cpu;
        let config = vnext_test_config();

        let mut a = Trainer::new_seeded(config.clone(), "iterated-parity", &dev, 1234)?;
        let mut b = Trainer::new_seeded(config.clone(), "iterated-parity", &dev, 1234)?;
        let c = Trainer::new_seeded(config, "iterated-parity", &dev, 4321)?;

        for name in [
            "nca.dense1.weight",
            "nca.dense_gate.bias",
            "interface.embed.weight",
            "interface.proj.weight",
            "interface.aux_proj.weight",
        ] {
            let va = var_values(&a, name)?;
            let vb = var_values(&b, name)?;
            assert_eq!(va, vb, "{name} must be bit-identical for the same seed");
        }
        let diff_seed = var_values(&c, "nca.dense1.weight")?;
        assert_ne!(
            var_values(&a, "nca.dense1.weight")?,
            diff_seed,
            "a different seed must produce different weights"
        );

        let diag_a = a.train_step(2)?;
        let diag_b = b.train_step(2)?;
        assert_eq!(
            diag_a.train_loss, diag_b.train_loss,
            "same seed must reproduce the first train_step loss exactly"
        );
        assert_eq!(diag_a.train_acc, diag_b.train_acc);
        Ok(())
    }

    /// RD-018b acceptance 2: the training batch advances with `step_count`
    /// while validation batches stay fixed-seed reproducible.
    #[test]
    fn test_advancing_train_stream_with_fixed_val() -> Result<()> {
        let dev = Device::Cpu;
        let config = vnext_test_config();
        let mut trainer = Trainer::new_seeded(config, "iterated-parity", &dev, 7)?;

        let (in0, _, _) = trainer.draw_train_batch(2)?;
        trainer.step_count = 1;
        let (in1, _, _) = trainer.draw_train_batch(2)?;
        assert_ne!(
            in0.to_vec2::<u32>()?,
            in1.to_vec2::<u32>()?,
            "consecutive steps must draw different training batches"
        );
        trainer.step_count = 0;

        let (val_a, _, _) = trainer
            .dataset
            .sample_batch_seeded_with_mask(2, 16, true, VAL_EVAL_SEED, &dev)?;
        let (val_b, _, _) = trainer
            .dataset
            .sample_batch_seeded_with_mask(2, 16, true, VAL_EVAL_SEED, &dev)?;
        assert_eq!(
            val_a.to_vec2::<u32>()?,
            val_b.to_vec2::<u32>()?,
            "validation batches must be reproducible at the fixed eval seed"
        );

        let first = trainer.train_step(2)?;
        assert_eq!(first.train_stream_seed, 0);
        let second = trainer.train_step(2)?;
        assert_eq!(second.train_stream_seed, 1);
        assert_eq!(trainer.step_count, 2);

        let legacy = Trainer::new(vnext_test_config(), "iterated-parity", &dev)?;
        assert_eq!(legacy.train_stream_seed(), 0);
        Ok(())
    }

    /// RD-018b acceptance 3: exhaustive heldout evaluation on the full
    /// iterated-parity L=16 validation split (789 unique rows) with per-slot
    /// accuracy; the panel is fixed-seed reproducible.
    #[test]
    fn test_evaluate_val_exhaustive_heldout_panel() -> Result<()> {
        let dev = Device::Cpu;
        let mut config = vnext_test_config();
        config.train.dev_steps = 2;
        let trainer = Trainer::new_seeded(config, "iterated-parity", &dev, 11)?;

        let report = trainer.evaluate_val_exhaustive(64)?;
        assert_eq!(
            report.rows, MAX_HELDOUT_ROWS,
            "the iterated-parity L=16 val split must yield all {MAX_HELDOUT_ROWS} unique rows"
        );
        assert_eq!(report.per_slot_accuracy.len(), 4);
        assert_eq!(report.per_slot_count, vec![MAX_HELDOUT_ROWS; 4]);
        assert!(report.accuracy >= 0.0 && report.accuracy <= 1.0);
        assert!(report.loss.is_finite());
        for acc in &report.per_slot_accuracy {
            assert!(*acc >= 0.0 && *acc <= 1.0);
        }
        let again = trainer.evaluate_val_exhaustive(64)?;
        assert_eq!(report.rows, again.rows);
        assert_eq!(report.loss, again.loss);
        assert_eq!(report.accuracy, again.accuracy);

        // A smaller explicit panel is a prefix subset and stays consistent.
        let small = trainer.evaluate_val_panel(17, 8)?;
        assert_eq!(small.rows, 17);
        Ok(())
    }

    /// RD-018b acceptance 4: a deliberately huge gradient is clipped to the
    /// configured max L2 norm, pre/post norms are recorded, and legacy runs
    /// are unclipped.
    #[test]
    fn test_gradient_clipping_bounds_total_norm() -> Result<()> {
        let dev = Device::Cpu;
        let config = vnext_test_config();
        let trainer = Trainer::new_seeded(config.clone(), "iterated-parity", &dev, 21)?;

        // Deliberately huge synthetic gradients: backpropagate a scaled
        // sum-of-squares over every parameter (grad = 2e6 * param).
        let mut loss: Option<Tensor> = None;
        for var in trainer.varmap.all_vars() {
            let term = var.as_tensor().sqr()?.sum_all()?;
            loss = Some(match loss {
                None => term,
                Some(prev) => (&prev + &term)?,
            });
        }
        let loss = (loss.expect("varmap is non-empty") * 1e6)?;
        let mut grads = loss.backward()?;
        let pre: f32 = {
            let mut sum_sq = 0.0f32;
            for var in trainer.varmap.all_vars() {
                if let Some(g) = grads.get(&var) {
                    let sq: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                    sum_sq += sq;
                }
            }
            sum_sq.sqrt()
        };
        assert!(
            pre > trainer.grad_clip_norm,
            "synthetic gradient norm {pre} must exceed the clip bound {}",
            trainer.grad_clip_norm
        );

        let post = trainer.clip_gradients_in_place(&mut grads, pre)?;
        assert!(
            post <= trainer.grad_clip_norm,
            "post-clip norm {post} must not exceed the clip bound {}",
            trainer.grad_clip_norm
        );
        assert!(post > 0.0);

        // The rescaled gradients stored in the GradStore must honor the bound.
        let mut sum_sq_after = 0.0f32;
        for var in trainer.varmap.all_vars() {
            if let Some(g) = grads.get(&var) {
                let sq: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                sum_sq_after += sq;
            }
        }
        let measured = sum_sq_after.sqrt();
        assert!(
            measured <= trainer.grad_clip_norm + 1e-6,
            "stored clipped gradients must honor the bound (measured {measured})"
        );

        // Integration: a tiny clip bound guarantees real train_step gradients
        // get clipped and both norms are recorded.
        let mut tight = config;
        tight.train.grad_clip_norm = 1e-4;
        let mut tight_trainer = Trainer::new_seeded(tight, "iterated-parity", &dev, 22)?;
        let diag = tight_trainer.train_step(2)?;
        assert!(diag.grad_norm_pre_clip > 0.0);
        assert!(
            diag.grad_norm_post_clip <= 1e-4,
            "post-clip norm {} must not exceed 1e-4",
            diag.grad_norm_post_clip
        );
        assert_eq!(diag.grad_norm, diag.grad_norm_pre_clip);

        // Legacy path never clips.
        let mut legacy = Trainer::new(vnext_test_config(), "iterated-parity", &dev)?;
        let legacy_diag = legacy.train_step(2)?;
        assert_eq!(legacy_diag.grad_norm_post_clip, legacy_diag.grad_norm_pre_clip);
        Ok(())
    }

    /// RD-018b acceptance 5: the auxiliary tick schedule is exactly the
    /// intended interior set plus the terminal tick (no duplicates), and aux
    /// CE is read from post-update states at the scheduled ticks.
    #[test]
    fn test_aux_schedule_and_post_update_timing() -> Result<()> {
        assert_eq!(aux_supervision_ticks(16), vec![4, 8, 12, 16]);
        assert_eq!(aux_interior_ticks(16), vec![4, 8, 12]);
        assert_eq!(aux_supervision_ticks(8), vec![2, 4, 6, 8]);
        assert_eq!(aux_supervision_ticks(4), vec![1, 2, 3, 4]);
        for t in [1usize, 3, 5, 12, 16] {
            let ticks = aux_supervision_ticks(t);
            assert_eq!(ticks.last().copied(), Some(t));
            let mut dedup = ticks.clone();
            dedup.dedup();
            assert_eq!(ticks, dedup, "tick schedule must not contain duplicates");
            for &tick in &ticks[..ticks.len().saturating_sub(1)] {
                assert!(tick < t, "interior ticks must exclude the terminal tick");
            }
        }

        // Timing check: replay the deterministic first step and compare the
        // diagnostics against CE at post-update (and pre-update) states.
        let dev = Device::Cpu;
        let mut config = vnext_test_config();
        config.train.dev_steps = 4;
        config.train.aux_supervision_weight = 1.0;
        let mut trainer = Trainer::new_seeded(config, "iterated-parity", &dev, 31)?;

        let batch_size = 2usize;
        let (inputs, targets, _) = trainer.draw_train_batch(batch_size)?;
        let aux_mask = trainer.aux_interior_mask(batch_size)?;

        let initial = MorphogenicField::from_tensor(
            trainer.interface.embed_tokens(&inputs)?,
            &trainer.config.field,
        );
        let initial_ce = {
            let logits0 = trainer.interface.aux_logits(&initial.x)?;
            trainer
                .aux_interior_loss(&logits0, &targets, &aux_mask)?
                .to_scalar::<f32>()?
        };
        let mut field = initial;
        let mut expected_post: Vec<(usize, f32)> = Vec::new();
        for tick in 1..=4usize {
            field = trainer.nca.step(&field, &dev)?.0;
            if tick < 4 {
                // Post-update state at the scheduled interior ticks.
                let logits_t = trainer.interface.aux_logits(&field.x)?;
                let al = trainer
                    .aux_interior_loss(&logits_t, &targets, &aux_mask)?
                    .to_scalar::<f32>()?;
                expected_post.push((tick, al));
            }
        }
        assert_eq!(expected_post.len(), 3);

        let diag = trainer.train_step(batch_size)?;
        assert_eq!(
            diag.aux_tick_indices,
            vec![1, 2, 3],
            "vNext fixed-regime aux must supervise the interior tick set only"
        );
        assert_eq!(diag.aux_tick_losses.len(), 3);
        for (i, (tick, expected)) in expected_post.iter().enumerate() {
            assert_eq!(diag.aux_tick_indices[i], *tick);
            assert!(
                (diag.aux_tick_losses[i] - expected).abs() < 1e-6,
                "aux CE at tick {tick} must be read from the post-update state (got {}, expected {})",
                diag.aux_tick_losses[i],
                expected
            );
        }
        assert!(
            (diag.aux_tick_losses[0] - initial_ce).abs() > 1e-6,
            "tick-1 aux CE must not equal the pre-update (initial) state CE"
        );
        assert!(!diag.aux_tick_indices.contains(&4), "terminal tick must not be duplicated");

        // The legacy path keeps only the historical folding mean.
        let mut legacy_config = vnext_test_config();
        legacy_config.train.dev_steps = 4;
        legacy_config.train.aux_supervision_weight = 1.0;
        let mut legacy = Trainer::new(legacy_config, "iterated-parity", &dev)?;
        let legacy_diag = legacy.train_step(2)?;
        assert!(legacy_diag.aux_tick_indices.is_empty());
        assert!(legacy_diag.aux_tick_losses.is_empty());
        Ok(())
    }
}
