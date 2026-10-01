use crate::config::TitanConfig;
use crate::dataset::SequenceDataset;
use crate::field::{FrequencyDecomposition, MorphogenicField};
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::TokenInterface;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepDiagnostics {
    pub train_loss: f32,
    pub train_acc: f32,
    pub val_loss: f32,
    pub val_acc: f32,
    pub grad_norm: f32,
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
}

impl Trainer {
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
        })
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

    /// Evaluates model performance on the held-out validation sequences
    pub fn evaluate_val(&self, batch_size: usize) -> Result<(f32, f32)> {
        let (val_inputs, val_targets, raw_mask) = self.dataset.sample_val_batch_with_mask(batch_size, self.config.field.seq_len, &self.device)?;
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

    /// Performs one training step with complete dynamical diagnostics and horizon-robust training regimes
    pub fn train_step(&mut self, batch_size: usize) -> Result<StepDiagnostics> {
        let (inputs, targets, raw_mask) = self.dataset.sample_train_batch_with_mask(batch_size, self.config.field.seq_len, &self.device)?;
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
                        let al = self.aux_interior_loss(&logits_t, &targets, am)?;
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
                let al = self.aux_interior_loss(&logits_t, &targets, am)?;
                aux_acc = Some(al);
                aux_count += 1.0;
            }
            final_logits = Some(logits_t);

            for _ in 0..tail_k {
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
                    let al = self.aux_interior_loss(&logits_k, &targets, am)?;
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

            for step_idx in 1..=t_target {
                let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
                total_update_mag += update_mag;
                steps_executed += 1;

                // Aux deep supervision at ~4 evenly spaced intermediate ticks
                if aux_weight > 0.0 {
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
            if let Some(ref am) = aux_mask {
                let al = self.aux_interior_loss(&logits, &targets, am)?;
                aux_acc = match aux_acc {
                    None => Some(al),
                    Some(prev) => Some((&prev + &al)?),
                };
                aux_count += 1.0;
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

        // 5. Backpropagation and exact gradient norm calculation
        let grads = ce_loss.backward()?;
        let mut sum_sq_grad = 0.0f32;
        for var in self.varmap.all_vars() {
            if let Some(g) = grads.get(&var) {
                let sq: f32 = g.sqr()?.sum_all()?.to_scalar()?;
                sum_sq_grad += sq;
            }
        }
        let grad_norm = sum_sq_grad.sqrt();

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
            grad_norm,
            update_magnitude: avg_update_mag,
            hidden_mean,
            hidden_var,
            state_energy,
            enstrophy,
            palinstrophy,
            bkm_norm,
            freq_decomp,
            aux_loss: aux_diag_value,
        })
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
}
