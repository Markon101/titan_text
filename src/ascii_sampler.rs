//! Autoregressive Free-Running ASCII Sampler for Titan Text.
//!
//! Provides token-by-token free-running generation with:
//! - Context window sliding and causal NCA latent recurrence
//! - Temperature and top-k probabilistic sampling
//! - Causal latent tick budget sweeps (tau in 0, 1, 2, 4, 8, 16)
//! - Recurrent state lesion ablation (--lesion-state)
//! - Full structural metrics evaluation on every generated sample

use crate::ascii_corpus::{evaluate_ascii_metrics, AsciiCorpus, AsciiMetrics};
use crate::config::TitanConfig;
use crate::field::MorphogenicField;
use crate::nca::NeuralCellularAutomaton;
use crate::vocab::{TokenInterface, Vocab};
use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub prompt: String,
    pub max_len: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub tau: usize,
    pub seed: u64,
    pub lesion_state: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSample {
    pub id: String,
    pub prompt: String,
    pub raw_output: String,
    pub full_text: String,
    pub token_ids: Vec<usize>,
    pub stop_reason: String,
    pub steps_generated: usize,
    pub config: GenerationConfig,
    pub metrics: AsciiMetrics,
}

pub struct AsciiSampler<'a> {
    pub nca: &'a NeuralCellularAutomaton,
    pub interface: &'a TokenInterface,
    pub vocab: &'a Vocab,
    pub config: &'a TitanConfig,
    pub device: &'a Device,
}

impl<'a> AsciiSampler<'a> {
    pub fn new(
        nca: &'a NeuralCellularAutomaton,
        interface: &'a TokenInterface,
        vocab: &'a Vocab,
        config: &'a TitanConfig,
        device: &'a Device,
    ) -> Self {
        Self {
            nca,
            interface,
            vocab,
            config,
            device,
        }
    }

    pub fn generate(&self, gen_cfg: &GenerationConfig, corpus: Option<&AsciiCorpus>) -> Result<GenerationSample> {
        let mut rng = StdRng::seed_from_u64(gen_cfg.seed);
        let mut current_ids = self.vocab.encode(&gen_cfg.prompt);
        if current_ids.is_empty() {
            current_ids.push(self.vocab.bos_id);
        }

        let context_len = self.config.field.seq_len;
        let mut generated_tokens = Vec::new();
        let mut stop_reason = "max_length".to_string();

        for _ in 0..gen_cfg.max_len {
            // Window of at most context_len
            let window_len = current_ids.len().min(context_len);
            let start = current_ids.len() - window_len;
            let window = &current_ids[start..];

            // Pad window to context_len
            let mut padded = window.to_vec();
            padded.resize(context_len, self.vocab.pad_id);
            let u32_ids: Vec<u32> = padded.iter().map(|&id| id as u32).collect();
            let input_tensor = Tensor::from_slice(&u32_ids, (1, context_len), self.device)?;

            let seed_embed = self.interface.embed_tokens(&input_tensor)?;
            let mut field = MorphogenicField::from_tensor(seed_embed, &self.config.field);

            // Latent recurrence ticks (tau)
            if !gen_cfg.lesion_state {
                for _ in 0..gen_cfg.tau {
                    field = self.nca.step_field(&field, self.device)?;
                }
            }

            let logits = self.interface.logits(&field.x)?; // [1, context_len, vocab_size]
            let active_pos = window_len - 1;
            let next_logits_tensor = logits.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?;
            let mut logits_vec = next_logits_tensor.to_vec1::<f32>()?;

            // Suppress special non-content tokens
            if self.vocab.pad_id < logits_vec.len() {
                logits_vec[self.vocab.pad_id] = f32::NEG_INFINITY;
            }
            if self.vocab.bos_id < logits_vec.len() {
                logits_vec[self.vocab.bos_id] = f32::NEG_INFINITY;
            }
            if self.vocab.unk_id < logits_vec.len() {
                logits_vec[self.vocab.unk_id] = f32::NEG_INFINITY;
            }

            // Sampling
            let next_token = if gen_cfg.temperature <= 1e-4 {
                // Greedy argmax
                let mut best_id = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (id, &val) in logits_vec.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_id = id;
                    }
                }
                best_id
            } else {
                // Temperature scaling
                let temp = gen_cfg.temperature.max(0.01);
                for val in logits_vec.iter_mut() {
                    *val /= temp;
                }

                // Top-k filtering if requested
                if gen_cfg.top_k > 0 && gen_cfg.top_k < logits_vec.len() {
                    let mut sorted_indices: Vec<usize> = (0..logits_vec.len()).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        logits_vec[b].partial_cmp(&logits_vec[a]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let cutoff_val = logits_vec[sorted_indices[gen_cfg.top_k.min(sorted_indices.len()) - 1]];
                    for val in logits_vec.iter_mut() {
                        if *val < cutoff_val {
                            *val = f32::NEG_INFINITY;
                        }
                    }
                }

                // Numerically stable softmax
                let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                let mut probs = vec![0.0f32; logits_vec.len()];
                for (i, &l) in logits_vec.iter().enumerate() {
                    if l > f32::NEG_INFINITY {
                        let e = (l - max_logit).exp();
                        probs[i] = e;
                        sum_exp += e;
                    }
                }

                if sum_exp <= 0.0 {
                    0
                } else {
                    for p in probs.iter_mut() {
                        *p /= sum_exp;
                    }
                    // Multinomial sample
                    let roll: f32 = rng.gen();
                    let mut accum = 0.0f32;
                    let mut sampled_id = probs.len().saturating_sub(1);
                    for (id, &p) in probs.iter().enumerate() {
                        accum += p;
                        if roll <= accum {
                            sampled_id = id;
                            break;
                        }
                    }
                    sampled_id
                }
            };

            if next_token == self.vocab.eos_id {
                stop_reason = "eos".to_string();
                break;
            }

            current_ids.push(next_token);
            generated_tokens.push(next_token);
        }

        let raw_output = self.vocab.decode_raw(&generated_tokens);
        let full_text = format!("{}{}", gen_cfg.prompt, raw_output);

        let default_corpus = AsciiCorpus::new_balanced(10, 42);
        let train_recs = match corpus {
            Some(c) => &c.train_records,
            None => &default_corpus.train_records,
        };
        let metrics = evaluate_ascii_metrics(&raw_output, train_recs);

        let id = format!("sample_seed{}_tau{}", gen_cfg.seed, gen_cfg.tau);

        Ok(GenerationSample {
            id,
            prompt: gen_cfg.prompt.clone(),
            raw_output,
            full_text,
            token_ids: generated_tokens.clone(),
            stop_reason,
            steps_generated: generated_tokens.len(),
            config: gen_cfg.clone(),
            metrics,
        })
    }
}
