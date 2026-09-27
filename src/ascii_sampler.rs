//! Autoregressive Free-Running ASCII Sampler with Adaptive Recurrent Compute for Titan Text.
//!
//! Provides token-by-token free-running generation with:
//! - Context window sliding and causal NCA latent recurrence
//! - Fixed-tau baseline control (tau in 0, 1, 2, 4, 8, 16)
//! - Adaptive recurrent compute: deterministic state/output stability halting
//! - Sham/random compute baseline: variable tau sampled from uniform or empirical distribution
//! - Causal state lesion ablation (--lesion-state)
//! - Per-token tick tracking: position, character class, entropy, probability, ticks used, halting reason
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
use std::collections::HashMap;

/// Halting computation mode for latent recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HaltingMode {
    /// Fixed recurrent compute: exactly tau ticks per token.
    Fixed,
    /// Adaptive compute: halt when state/output stability threshold is satisfied.
    Adaptive,
    /// Random compute: sham control sampling ticks uniformly from [tau_min, tau_max].
    Random,
    /// Explicit per-token tick schedule: used for Arm C (shuffled / permuted adaptive sequence control).
    Schedule,
}

impl Default for HaltingMode {
    fn default() -> Self {
        Self::Fixed
    }
}

impl HaltingMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fixed" | "constant" => Some(Self::Fixed),
            "adaptive" | "adapt" => Some(Self::Adaptive),
            "random" | "sham" | "uniform" => Some(Self::Random),
            "schedule" | "permuted" | "shuffled" => Some(Self::Schedule),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
            Self::Random => "random",
            Self::Schedule => "schedule",
        }
    }
}

/// Metric used to measure stability across recurrent ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HaltingMetric {
    /// L2 norm of state update (x_t - x_{t-1}) at the active token position.
    StateDelta,
    /// Relative state change: ||x_t - x_{t-1}|| / (||x_t|| + eps).
    RelativeDelta,
    /// Maximum absolute logit change across the vocabulary at the active token position.
    LogitDelta,
    /// Absolute change in predictive entropy |H_t - H_{t-1}|.
    EntropyDelta,
    /// Cosine distance: 1.0 - cos(x_t, x_{t-1}) at the active token position.
    Cosine,
}

impl Default for HaltingMetric {
    fn default() -> Self {
        Self::StateDelta
    }
}

impl HaltingMetric {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "state_delta" | "state" | "l2" => Some(Self::StateDelta),
            "relative_delta" | "relative" | "rel" => Some(Self::RelativeDelta),
            "logit_delta" | "logits" | "logit" => Some(Self::LogitDelta),
            "entropy_delta" | "entropy" => Some(Self::EntropyDelta),
            "cosine" | "cos" => Some(Self::Cosine),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StateDelta => "state_delta",
            Self::RelativeDelta => "relative_delta",
            Self::LogitDelta => "logit_delta",
            Self::EntropyDelta => "entropy_delta",
            Self::Cosine => "cosine",
        }
    }
}

fn default_halting_mode() -> HaltingMode {
    HaltingMode::Fixed
}

fn default_tau_min() -> usize {
    1
}

fn default_tau_max() -> usize {
    16
}

fn default_halting_metric() -> HaltingMetric {
    HaltingMetric::StateDelta
}

fn default_halting_threshold() -> f32 {
    0.08
}

fn default_halting_patience() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub prompt: String,
    pub max_len: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub tau: usize,
    pub seed: u64,
    pub lesion_state: bool,

    #[serde(default = "default_halting_mode")]
    pub halting_mode: HaltingMode,

    #[serde(default = "default_tau_min")]
    pub tau_min: usize,

    #[serde(default = "default_tau_max")]
    pub tau_max: usize,

    #[serde(default = "default_halting_metric")]
    pub halting_metric: HaltingMetric,

    #[serde(default = "default_halting_threshold")]
    pub halting_threshold: f32,

    #[serde(default = "default_halting_patience")]
    pub halting_patience: usize,

    #[serde(default)]
    pub tau_schedule: Option<Vec<usize>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            prompt: "<BOX>\n".to_string(),
            max_len: 64,
            temperature: 0.7,
            top_k: 10,
            tau: 4,
            seed: 42,
            lesion_state: false,
            halting_mode: HaltingMode::Fixed,
            tau_min: 1,
            tau_max: 16,
            halting_metric: HaltingMetric::StateDelta,
            halting_threshold: 0.08,
            halting_patience: 1,
            tau_schedule: None,
        }
    }
}

/// Trace record for a single emitted token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTickTrace {
    pub position: usize,
    pub token_id: usize,
    pub char_emitted: String,
    pub char_class: String,
    pub ticks_used: usize,
    pub halt_reason: String,
    pub final_metric_val: Option<f32>,
    pub entropy: f32,
    pub token_prob: f32,
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

    pub total_ticks: usize,
    pub mean_tau: f32,
    pub median_tau: f32,
    pub min_tau: usize,
    pub max_tau: usize,
    pub tau_by_char_class: HashMap<String, f32>,
    pub token_traces: Vec<TokenTickTrace>,
}

pub struct AsciiSampler<'a> {
    pub nca: &'a NeuralCellularAutomaton,
    pub interface: &'a TokenInterface,
    pub vocab: &'a Vocab,
    pub config: &'a TitanConfig,
    pub device: &'a Device,
}

pub fn classify_char(c: char) -> &'static str {
    match c {
        '\n' => "newline",
        ' ' | '\t' => "whitespace",
        '+' | '-' | '=' | '|' | ':' | '#' | '*' | '/' | '\\' | '_' => "boundary",
        'a'..='z' | 'A'..='Z' | '0'..='9' => "alphanumeric",
        _ => "symbol",
    }
}

fn compute_entropy(logits: &[f32]) -> f32 {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            sum_exp += (l - max_l).exp();
        }
    }
    if sum_exp <= 0.0 {
        return 0.0;
    }
    let mut ent = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            let p = (l - max_l).exp() / sum_exp;
            if p > 1e-12 {
                ent -= p * p.ln();
            }
        }
    }
    ent
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
        let mut token_traces = Vec::new();
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
            let active_pos = window_len - 1;

            // Compute ticks using selected HaltingMode
            let (ticks_used, halt_reason, final_metric) = if gen_cfg.lesion_state {
                // Recurrence disabled
                (0, "lesion".to_string(), None)
            } else {
                match gen_cfg.halting_mode {
                    HaltingMode::Fixed => {
                        for _ in 0..gen_cfg.tau {
                            field = self.nca.step_field(&field, self.device)?;
                        }
                        (gen_cfg.tau, "fixed".to_string(), None)
                    }
                    HaltingMode::Random => {
                        let min_t = gen_cfg.tau_min.max(1);
                        let max_t = gen_cfg.tau_max.max(min_t);
                        let target_ticks = rng.gen_range(min_t..=max_t);
                        for _ in 0..target_ticks {
                            field = self.nca.step_field(&field, self.device)?;
                        }
                        (target_ticks, "random".to_string(), None)
                    }
                    HaltingMode::Schedule => {
                        let step = generated_tokens.len();
                        let target_ticks = gen_cfg.tau_schedule.as_ref()
                            .and_then(|sched| sched.get(step).copied())
                            .unwrap_or(gen_cfg.tau);
                        for _ in 0..target_ticks {
                            field = self.nca.step_field(&field, self.device)?;
                        }
                        (target_ticks, "schedule".to_string(), None)
                    }
                    HaltingMode::Adaptive => {
                        let min_t = gen_cfg.tau_min.max(1);
                        let max_t = gen_cfg.tau_max.max(min_t);
                        let patience = gen_cfg.halting_patience.max(1);
                        let mut used = min_t;
                        let mut reason = "tau_max".to_string();
                        let mut metric_val = 0.0f32;
                        let mut consecutive_stable = 0;

                        // Cache previous logits/entropy across iterations to eliminate redundant forward passes
                        let (mut prev_logits, mut prev_entropy) = match gen_cfg.halting_metric {
                            HaltingMetric::LogitDelta | HaltingMetric::EntropyDelta => {
                                let logits = self.interface.logits(&field.x)?;
                                let l_vec = logits.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                let ent = compute_entropy(&l_vec);
                                (Some(l_vec), Some(ent))
                            }
                            _ => (None, None),
                        };

                        for tick in 1..=max_t {
                            let prev_x = match gen_cfg.halting_metric {
                                HaltingMetric::StateDelta | HaltingMetric::RelativeDelta | HaltingMetric::Cosine => {
                                    Some(field.x.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?)
                                }
                                _ => None,
                            };

                            let next_field = self.nca.step_field(&field, self.device)?;

                            metric_val = match gen_cfg.halting_metric {
                                HaltingMetric::StateDelta => {
                                    let curr_x = next_field.x.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                    let p_x = prev_x.as_ref().unwrap();
                                    p_x.iter().zip(curr_x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
                                }
                                HaltingMetric::RelativeDelta => {
                                    let curr_x = next_field.x.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                    let p_x = prev_x.as_ref().unwrap();
                                    let diff_l2: f32 = p_x.iter().zip(curr_x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                                    let norm_curr: f32 = curr_x.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
                                    diff_l2 / (norm_curr + 1e-6)
                                }
                                HaltingMetric::Cosine => {
                                    let curr_x = next_field.x.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                    let p_x = prev_x.as_ref().unwrap();
                                    let dot: f32 = p_x.iter().zip(curr_x.iter()).map(|(a, b)| a * b).sum();
                                    let norm_p: f32 = p_x.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
                                    let norm_c: f32 = curr_x.iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
                                    let cos = dot / (norm_p * norm_c + 1e-6);
                                    (1.0 - cos).max(0.0)
                                }
                                HaltingMetric::LogitDelta => {
                                    let next_logits = self.interface.logits(&next_field.x)?;
                                    let curr_l = next_logits.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                    let p_l = prev_logits.as_ref().unwrap();
                                    let delta = p_l.iter().zip(curr_l.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                                    prev_logits = Some(curr_l);
                                    delta
                                }
                                HaltingMetric::EntropyDelta => {
                                    let next_logits = self.interface.logits(&next_field.x)?;
                                    let curr_l = next_logits.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
                                    let curr_ent = compute_entropy(&curr_l);
                                    let delta = (curr_ent - prev_entropy.unwrap()).abs();
                                    prev_entropy = Some(curr_ent);
                                    delta
                                }
                            };

                            field = next_field;
                            used = tick;

                            if tick >= min_t {
                                if metric_val <= gen_cfg.halting_threshold {
                                    consecutive_stable += 1;
                                    if consecutive_stable >= patience {
                                        reason = "threshold".to_string();
                                        break;
                                    }
                                } else {
                                    consecutive_stable = 0;
                                }
                            }
                        }

                        (used, reason, Some(metric_val))
                    }
                }
            };

            let logits = self.interface.logits(&field.x)?; // [1, context_len, vocab_size]
            let next_logits_tensor = logits.narrow(1, active_pos, 1)?.squeeze(1)?.squeeze(0)?;
            let mut logits_vec = next_logits_tensor.to_vec1::<f32>()?;

            // Compute predictive entropy and raw distribution before masking
            let entropy = compute_entropy(&logits_vec);

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
            let (next_token, token_prob) = if gen_cfg.temperature <= 1e-4 {
                // Greedy argmax
                let mut best_id = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (id, &val) in logits_vec.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_id = id;
                    }
                }
                let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits_vec.iter().filter(|&&v| v > f32::NEG_INFINITY).map(|&v| (v - max_logit).exp()).sum();
                let prob = if sum_exp > 0.0 && best_val > f32::NEG_INFINITY {
                    (best_val - max_logit).exp() / sum_exp
                } else {
                    1.0f32
                };
                (best_id, prob)
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
                    (0, 0.0f32)
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
                    let p = probs[sampled_id];
                    (sampled_id, p)
                }
            };

            let newline_id = self.vocab.newline_id();
            let (char_emitted, char_class) = if Some(next_token) == newline_id {
                ("\\n".to_string(), "newline".to_string())
            } else if next_token == self.vocab.eos_id {
                ("<eos>".to_string(), "special".to_string())
            } else if next_token == self.vocab.bos_id {
                ("<bos>".to_string(), "special".to_string())
            } else if next_token == self.vocab.pad_id {
                ("<pad>".to_string(), "special".to_string())
            } else if let Some(ch) = self.vocab.id_to_char(next_token) {
                let s = if ch == '\n' {
                    "\\n".to_string()
                } else if ch == '\t' {
                    "\\t".to_string()
                } else {
                    ch.to_string()
                };
                (s, classify_char(ch).to_string())
            } else {
                (format!("<tok{}>", next_token), "unknown".to_string())
            };

            token_traces.push(TokenTickTrace {
                position: generated_tokens.len(),
                token_id: next_token,
                char_emitted,
                char_class,
                ticks_used,
                halt_reason,
                final_metric_val: final_metric,
                entropy,
                token_prob,
            });

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

        let total_ticks: usize = token_traces.iter().map(|t| t.ticks_used).sum();
        let mean_tau = if !token_traces.is_empty() {
            total_ticks as f32 / token_traces.len() as f32
        } else {
            0.0
        };

        let mut sorted_ticks: Vec<usize> = token_traces.iter().map(|t| t.ticks_used).collect();
        sorted_ticks.sort_unstable();
        let median_tau = if sorted_ticks.is_empty() {
            0.0
        } else if sorted_ticks.len() % 2 == 1 {
            sorted_ticks[sorted_ticks.len() / 2] as f32
        } else {
            let mid = sorted_ticks.len() / 2;
            (sorted_ticks[mid - 1] as f32 + sorted_ticks[mid] as f32) / 2.0
        };
        let min_tau = sorted_ticks.first().copied().unwrap_or(0);
        let max_tau = sorted_ticks.last().copied().unwrap_or(0);

        let mut class_sums: HashMap<String, (usize, usize)> = HashMap::new();
        for t in &token_traces {
            let entry = class_sums.entry(t.char_class.clone()).or_insert((0, 0));
            entry.0 += t.ticks_used;
            entry.1 += 1;
        }
        let mut tau_by_char_class: HashMap<String, f32> = HashMap::new();
        for (class, (sum, count)) in class_sums {
            tau_by_char_class.insert(class, sum as f32 / count as f32);
        }

        let id = format!(
            "sample_seed{}_{}_{}",
            gen_cfg.seed,
            gen_cfg.halting_mode.as_str(),
            if gen_cfg.halting_mode == HaltingMode::Fixed {
                gen_cfg.tau.to_string()
            } else {
                format!("{}-{}", gen_cfg.tau_min, gen_cfg.tau_max)
            }
        );

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
            total_ticks,
            mean_tau,
            median_tau,
            min_tau,
            max_tau,
            tau_by_char_class,
            token_traces,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_classification() {
        assert_eq!(classify_char('\n'), "newline");
        assert_eq!(classify_char(' '), "whitespace");
        assert_eq!(classify_char('+'), "boundary");
        assert_eq!(classify_char('-'), "boundary");
        assert_eq!(classify_char('='), "boundary");
        assert_eq!(classify_char('|'), "boundary");
        assert_eq!(classify_char(':'), "boundary");
        assert_eq!(classify_char('A'), "alphanumeric");
        assert_eq!(classify_char('5'), "alphanumeric");
        assert_eq!(classify_char('?'), "symbol");
    }

    #[test]
    fn test_compute_entropy() {
        let uniform = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32];
        let ent = compute_entropy(&uniform);
        let expected = (4.0f32).ln();
        assert!((ent - expected).abs() < 1e-4);

        let peaked = vec![100.0f32, 0.0f32, 0.0f32];
        let ent_peaked = compute_entropy(&peaked);
        assert!(ent_peaked < 1e-4);
    }

    #[test]
    fn test_halting_mode_parsing() {
        assert_eq!(HaltingMode::parse("fixed"), Some(HaltingMode::Fixed));
        assert_eq!(HaltingMode::parse("adaptive"), Some(HaltingMode::Adaptive));
        assert_eq!(HaltingMode::parse("random"), Some(HaltingMode::Random));
        assert_eq!(HaltingMode::parse("sham"), Some(HaltingMode::Random));
        assert_eq!(HaltingMode::parse("schedule"), Some(HaltingMode::Schedule));
        assert_eq!(HaltingMode::parse("permuted"), Some(HaltingMode::Schedule));
        assert_eq!(HaltingMode::parse("shuffled"), Some(HaltingMode::Schedule));
        assert_eq!(HaltingMode::parse("invalid"), None);
    }

    #[test]
    fn test_halting_metric_parsing() {
        assert_eq!(HaltingMetric::parse("state_delta"), Some(HaltingMetric::StateDelta));
        assert_eq!(HaltingMetric::parse("relative_delta"), Some(HaltingMetric::RelativeDelta));
        assert_eq!(HaltingMetric::parse("logit_delta"), Some(HaltingMetric::LogitDelta));
        assert_eq!(HaltingMetric::parse("entropy_delta"), Some(HaltingMetric::EntropyDelta));
        assert_eq!(HaltingMetric::parse("cosine"), Some(HaltingMetric::Cosine));
        assert_eq!(HaltingMetric::parse("invalid"), None);
    }
}
