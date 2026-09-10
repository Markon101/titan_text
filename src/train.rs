use crate::config::TitanConfig;
use crate::dataset::SequenceDataset;
use crate::field::{FrequencyDecomposition, MorphogenicField};
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::TokenInterface;
use anyhow::Result;
use candle_core::{DType, Device};
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
}

pub struct Trainer {
    pub nca: NeuralCellularAutomaton,
    pub interface: TokenInterface,
    pub optimizer: AdamW,
    pub varmap: VarMap,
    pub config: TitanConfig,
    pub dataset: SequenceDataset,
    pub device: Device,
}

impl Trainer {
    pub fn new(config: TitanConfig, task: &str, device: &Device) -> Result<Self> {
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
        })
    }

    /// Evaluates model performance on the held-out validation sequences
    pub fn evaluate_val(&self, batch_size: usize) -> Result<(f32, f32)> {
        let (val_inputs, val_targets) = self.dataset.sample_val_batch(batch_size, self.config.field.seq_len, &self.device)?;
        let seed = self.interface.embed_tokens(&val_inputs)?;
        let mut field = MorphogenicField::from_tensor(seed, &self.config.field);

        for _ in 0..self.config.train.dev_steps {
            field = self.nca.step_field(&field, &self.device)?;
        }

        let logits = self.interface.logits(&field.x)?;
        let loss = self.interface.cross_entropy_loss(&logits, &val_targets)?.to_scalar::<f32>()?;
        let acc = self.interface.accuracy(&logits, &val_targets)?;

        Ok((loss, acc))
    }

    /// Performs one training step with complete dynamical diagnostics
    pub fn train_step(&mut self, batch_size: usize) -> Result<StepDiagnostics> {
        let (inputs, targets) = self.dataset.sample_train_batch(batch_size, self.config.field.seq_len, &self.device)?;

        // 1. Embed initial tokens into continuous seed field: [B, L, C]
        let seed_embeddings = self.interface.embed_tokens(&inputs)?;
        let mut field = MorphogenicField::from_tensor(seed_embeddings, &self.config.field);

        // 2. Unroll developmental steps (BPTT) and record update magnitudes
        let mut total_update_mag = 0.0f32;
        for _ in 0..self.config.train.dev_steps {
            let (next_field, update_mag) = self.nca.step(&field, &self.device)?;
            total_update_mag += update_mag;
            field = next_field;
        }
        let avg_update_mag = total_update_mag / self.config.train.dev_steps as f32;

        // 3. Project final developmental state to token logits
        let logits = self.interface.logits(&field.x)?;

        // 4. Compute sequence cross-entropy loss & accuracy
        let ce_loss = self.interface.cross_entropy_loss(&logits, &targets)?;
        let train_loss = ce_loss.to_scalar::<f32>()?;
        let train_acc = self.interface.accuracy(&logits, &targets)?;

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

        let temp_dir = "checkpoints/test_resume_training_tmp";

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
            train_loss: diag1.train_loss,
            train_accuracy: diag1.train_acc,
            val_loss: diag1.val_loss,
            val_accuracy: diag1.val_acc,
            grad_norm: diag1.grad_norm,
            state_energy: diag1.state_energy,
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
}
