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

        // 7. Field state metrics: mean, variance, energy E = 0.5 * mean(||x||^2), and frequency decomposition
        let (hidden_mean, hidden_var) = field.mean_and_var()?;
        let state_energy = field.energy()?;
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
            freq_decomp,
        })
    }
}
