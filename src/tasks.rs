use crate::vocab::Vocab;
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Algorithmic task identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    Text,
    Dyck,
    DelayedRecall,
    BracketDepth,
    Parity,
    Reverse,
    HiddenRule,
    Associative,
    AmbiguousBasin,
    ColumnArithmetic,
    IteratedParity,
    IteratedParityDense,
    IteratedParityCarrier,
    IteratedSumDense,
}

impl TaskKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" => Some(Self::Text),
            "dyck" => Some(Self::Dyck),
            "delayed-recall" | "delayed_recall" => Some(Self::DelayedRecall),
            "bracket-depth" | "bracket_depth" | "bracket" => Some(Self::BracketDepth),
            "parity" => Some(Self::Parity),
            "reverse" | "string-reverse" => Some(Self::Reverse),
            "hidden-rule" | "hidden_rule" => Some(Self::HiddenRule),
            "associative" | "associative-recall" => Some(Self::Associative),
            "ambiguous-basin" | "ambiguous_basin" | "ambiguous" => Some(Self::AmbiguousBasin),
            "column-arithmetic" | "column_arithmetic" | "arithmetic" => Some(Self::ColumnArithmetic),
            "iterated-parity" | "chunked-parity" | "state-parity" | "ippr" => Some(Self::IteratedParity),
            "iterated-parity-dense" | "ippr-dense" | "dense-parity" => Some(Self::IteratedParityDense),
            "iterated-parity-carrier" | "ippr-carrier" | "carrier-parity" => Some(Self::IteratedParityCarrier),
            "iterated-sum-dense" | "chunked-sum" | "sum-dense" | "iterated-sum" => Some(Self::IteratedSumDense),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Dyck => "dyck",
            Self::DelayedRecall => "delayed-recall",
            Self::BracketDepth => "bracket-depth",
            Self::Parity => "parity",
            Self::Reverse => "reverse",
            Self::HiddenRule => "hidden-rule",
            Self::Associative => "associative",
            Self::AmbiguousBasin => "ambiguous-basin",
            Self::ColumnArithmetic => "column-arithmetic",
            Self::IteratedParity => "iterated-parity",
            Self::IteratedParityDense => "iterated-parity-dense",
            Self::IteratedParityCarrier => "iterated-parity-carrier",
            Self::IteratedSumDense => "iterated-sum-dense",
        }
    }

    pub fn is_iterated_parity(&self) -> bool {
        matches!(self, Self::IteratedParity | Self::IteratedParityDense | Self::IteratedParityCarrier)
    }

    pub fn is_chunked_sequential(&self) -> bool {
        matches!(self, Self::IteratedParity | Self::IteratedParityDense | Self::IteratedParityCarrier | Self::IteratedSumDense)
    }
}

/// A structured batch from an algorithmic task.
#[allow(dead_code)]
pub struct TaskBatch {
    pub inputs: Tensor,      // [batch, seq_len]
    pub targets: Tensor,     // [batch, seq_len]
    pub loss_mask: Tensor,   // [batch, seq_len], 1.0 where evaluation counts, 0.0 where ignored
    pub prompt_text: String, // Decoded representation of first sample
    pub target_text: String, // Decoded target representation of first sample
    pub carry_depths: Option<Vec<usize>>, // Per-sample ground truth carry propagation chain lengths
}

/// Stratum specification for ColumnArithmetic evaluation
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithmeticStratum {
    MaxRipple,   // S1: 99...9 + 1 (worst-case ripple carry)
    LongPartial, // S2: variable carry depth >= 2
    CarryKill,   // S3: carry terminates early
    NoCarry,     // S4: control with 0 carries
}

/// Computes the maximum carry chain length and total carries for addition a + b in base 10
#[allow(dead_code)]
pub fn calculate_carry_chain(a: u64, b: u64, d: usize) -> (usize, usize) {
    let mut max_chain = 0;
    let mut current_chain = 0;
    let mut total_carries = 0;
    let mut carry = 0;
    for i in 0..d {
        let da = (a / 10u64.pow(i as u32)) % 10;
        let db = (b / 10u64.pow(i as u32)) % 10;
        let col_sum = da + db + carry;
        if col_sum >= 10 {
            carry = 1;
            current_chain += 1;
            total_carries += 1;
            if current_chain > max_chain {
                max_chain = current_chain;
            }
        } else {
            carry = 0;
            current_chain = 0;
        }
    }
    (max_chain, total_carries)
}

pub struct TaskEngine {
    pub vocab: Vocab,
}

/// Changes whenever algorithmic observations, labels, or split assignment change.
pub const TASK_SCHEMA_VERSION: &str = "algorithmic-v4-column-arithmetic-20260915";

/// Stable, non-secret hash of the complete observation. Identical inputs always
/// belong to the same split, regardless of generation seed or batch size.
fn observation_hash(input: &[char]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &c in input {
        hash = (hash ^ c as u64).wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^ (hash >> 31)
}

impl TaskEngine {
    pub fn new() -> Self {
        Self {
            vocab: Vocab::new_ascii(),
        }
    }

    pub fn generate_batch(
        &self,
        kind: TaskKind,
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        device: &Device,
    ) -> Result<TaskBatch> {
        self.generate_batch_seeded(kind, batch_size, seq_len, is_val, 0, device)
    }

    pub fn generate_batch_seeded(
        &self,
        kind: TaskKind,
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        seed: usize,
        device: &Device,
    ) -> Result<TaskBatch> {
        ensure!(batch_size > 0, "task batch_size must be positive");
        ensure!(seq_len > 0, "task seq_len must be positive");
        ensure!(
            batch_size.checked_mul(seq_len).is_some(),
            "task dimensions overflow"
        );
        if matches!(kind, TaskKind::Text | TaskKind::Dyck) {
            // Historical next-token objectives remain explicitly legacy controls.
            let ds = crate::dataset::SequenceDataset::new(kind.name());
            let (inputs, targets) = if is_val {
                ds.sample_val_batch(batch_size, seq_len, device)?
            } else {
                ds.sample_train_batch(batch_size, seq_len, device)?
            };
            // TaskEngine has one ASCII vocabulary, including when the legacy
            // dataset internally uses its smaller Dyck vocabulary.
            let convert = |tensor: Tensor| -> Result<Tensor> {
                let rows = tensor.to_vec2::<u32>()?;
                let ids: Vec<u32> = rows
                    .iter()
                    .flat_map(|row| {
                        let text = ds
                            .vocab
                            .decode(&row.iter().map(|&v| v as usize).collect::<Vec<_>>());
                        self.vocab
                            .encode(&text)
                            .into_iter()
                            .map(|v| v as u32)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok(Tensor::from_vec(ids, (batch_size, seq_len), device)?)
            };
            return Ok(TaskBatch {
                inputs: convert(inputs)?,
                targets: convert(targets)?,
                loss_mask: Tensor::ones((batch_size, seq_len), DType::F32, device)?,
                prompt_text: format!("legacy {} batch", kind.name()),
                target_text: String::new(),
                carry_depths: None,
            });
        }
        ensure!(
            seq_len >= 8,
            "algorithmic tasks require seq_len >= 8; shorter layouts are ambiguous or degenerate"
        );
        let mut rng = StdRng::seed_from_u64(
            seed as u64
                ^ if is_val {
                    0xd6e8feb86659fd93
                } else {
                    0xa0761d6478bd642f
                },
        );
        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        let mut masks = Vec::with_capacity(batch_size * seq_len);
        let mut prompt_text = String::new();
        let mut target_text = String::new();
        for b in 0..batch_size {
            let mut selected = None;
            for _ in 0..10_000 {
                let row = Self::generate_row(kind, seq_len, is_val, &mut rng);
                if (observation_hash(&row.0) % 5 == 0) == is_val {
                    selected = Some(row);
                    break;
                }
            }
            let (input, target, mask) = selected.ok_or_else(|| {
                anyhow::anyhow!(
                    "no examples available in requested split for {} length {}",
                    kind.name(),
                    seq_len
                )
            })?;
            if b == 0 {
                prompt_text = input.iter().collect();
                target_text = target.iter().collect();
            }
            inputs.extend(
                self.vocab
                    .encode(&input.iter().collect::<String>())
                    .into_iter()
                    .map(|v| v as u32),
            );
            targets.extend(
                self.vocab
                    .encode(&target.iter().collect::<String>())
                    .into_iter()
                    .map(|v| v as u32),
            );
            masks.extend(mask);
        }
        Ok(TaskBatch {
            inputs: Tensor::from_vec(inputs, (batch_size, seq_len), device)?,
            targets: Tensor::from_vec(targets, (batch_size, seq_len), device)?,
            loss_mask: Tensor::from_vec(masks, (batch_size, seq_len), device)?,
            prompt_text,
            target_text,
            carry_depths: None,
        })
    }

    /// Generates a structured, stratified batch of ColumnArithmetic examples
    /// with known carry propagation chains and optional out-of-distribution depth bounds.
    #[allow(dead_code)]
    pub fn generate_stratified_arithmetic_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        stratum: Option<ArithmeticStratum>,
        max_carry_depth: Option<usize>,
        seed: usize,
        device: &Device,
    ) -> Result<TaskBatch> {
        ensure!(batch_size > 0, "batch_size must be positive");
        ensure!(seq_len >= 16, "ColumnArithmetic requires seq_len >= 16");
        ensure!(batch_size.checked_mul(seq_len).is_some(), "task dimensions overflow");

        let d = ((seq_len - 4) / 3).min(18).max(1);
        ensure!(stratum != Some(ArithmeticStratum::MaxRipple)
            || max_carry_depth.map_or(true, |k| k >= d),
            "MaxRipple requires carry depth {}; requested upper bound is incompatible", d);
        let max_d_val = 10u64.pow(d as u32);
        let mut rng = StdRng::seed_from_u64(seed as u64 ^ 0x9e3779b97f4a7c15);

        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);
        let mut masks = Vec::with_capacity(batch_size * seq_len);
        let mut carry_depths = Vec::with_capacity(batch_size);
        let mut prompt_text = String::new();
        let mut target_text = String::new();

        for b in 0..batch_size {
            let mut chosen_a = 0u64;
            let mut chosen_b = 0u64;
            let mut chosen_chain = 0usize;
            let mut accepted = false;

            for _ in 0..10_000 {
                let (a, b) = match stratum {
                    Some(ArithmeticStratum::MaxRipple) => {
                        let val_a = max_d_val - 1; // 99...9
                        let val_b = rng.gen_range(1..10u64.min(max_d_val));
                        (val_a, val_b)
                    }
                    Some(ArithmeticStratum::NoCarry) => {
                        let mut val_a = 0u64;
                        let mut val_b = 0u64;
                        for i in 0..d {
                            let da = rng.gen_range(0..10u64);
                            let db = rng.gen_range(0..(10 - da));
                            val_a += da * 10u64.pow(i as u32);
                            val_b += db * 10u64.pow(i as u32);
                        }
                        (val_a, val_b)
                    }
                    Some(ArithmeticStratum::LongPartial) => {
                        let k = if d > 2 { rng.gen_range(2..d) } else { 1 };
                        let val_a = 10u64.pow(k as u32) - 1;
                        let val_b = rng.gen_range(1..10u64);
                        let shift = rng.gen_range(0..=(d - k));
                        (val_a * 10u64.pow(shift as u32), val_b * 10u64.pow(shift as u32))
                    }
                    Some(ArithmeticStratum::CarryKill) => {
                        let k = if d > 2 { rng.gen_range(1..d - 1) } else { 1 };
                        let val_a = (10u64.pow(k as u32) - 1) + 10u64.pow((k + 1) as u32);
                        let val_b = rng.gen_range(1..10u64);
                        (val_a % max_d_val, val_b % max_d_val)
                    }
                    None => {
                        if rng.gen_bool(0.5) {
                            let val_a = max_d_val - 1;
                            let val_b = rng.gen_range(1..10u64.min(max_d_val));
                            (val_a, val_b)
                        } else {
                            let val_a = rng.gen_range(0..max_d_val);
                            let val_b = rng.gen_range(0..max_d_val);
                            (val_a, val_b)
                        }
                    }
                };

                let (chain_len, _) = calculate_carry_chain(a, b, d);
                if let Some(max_k) = max_carry_depth {
                    if chain_len > max_k {
                        continue;
                    }
                }
                chosen_a = a;
                chosen_b = b;
                chosen_chain = chain_len;
                accepted = true;
                break;
            }

            ensure!(accepted, "unable to sample requested arithmetic stratum within 10000 attempts");
            let sum = chosen_a + chosen_b;
            let mut in_chars = vec!['.'; seq_len];
            let mut tgt_chars = vec!['.'; seq_len];
            let mut m_vec = vec![0.0f32; seq_len];

            let str_a = format!("{:0width$}", chosen_a, width = d);
            let str_b = format!("{:0width$}", chosen_b, width = d);
            let str_sum = format!("{:0width$}", sum, width = d + 1);

            for (i, c) in str_a.chars().enumerate() {
                in_chars[i] = c;
            }
            in_chars[d] = '+';
            for (i, c) in str_b.chars().enumerate() {
                in_chars[d + 1 + i] = c;
            }
            in_chars[2 * d + 1] = '=';
            for (i, c) in str_sum.chars().enumerate() {
                let pos = 2 * d + 2 + i;
                in_chars[pos] = '?';
                tgt_chars[pos] = c;
                m_vec[pos] = 1.0;
            }

            if b == 0 {
                prompt_text = in_chars.iter().collect();
                target_text = tgt_chars.iter().collect();
            }

            inputs.extend(
                self.vocab
                    .encode(&in_chars.iter().collect::<String>())
                    .into_iter()
                    .map(|v| v as u32),
            );
            targets.extend(
                self.vocab
                    .encode(&tgt_chars.iter().collect::<String>())
                    .into_iter()
                    .map(|v| v as u32),
            );
            masks.extend(m_vec);
            carry_depths.push(chosen_chain);
        }

        Ok(TaskBatch {
            inputs: Tensor::from_vec(inputs, (batch_size, seq_len), device)?,
            targets: Tensor::from_vec(targets, (batch_size, seq_len), device)?,
            loss_mask: Tensor::from_vec(masks, (batch_size, seq_len), device)?,
            prompt_text,
            target_text,
            carry_depths: Some(carry_depths),
        })
    }

    fn generate_row(
        kind: TaskKind,
        l: usize,
        is_val: bool,
        rng: &mut StdRng,
    ) -> (Vec<char>, Vec<char>, Vec<f32>) {
        let mut input = vec![' '; l];
        let mut target = vec![' '; l];
        let mut mask = vec![0.; l];
        match kind {
            TaskKind::DelayedRecall => {
                let n = if l >= 16 { 4 } else { 2 };
                input.fill('.');
                for i in 0..n {
                    input[i] = (b'A' + rng.gen_range(0..8)) as char;
                }
                for i in 0..n {
                    let q = l - n + i;
                    input[q] = '?';
                    target[q] = input[i];
                    mask[q] = 1.;
                }
            }
            TaskKind::BracketDepth => {
                // Depth tracking, not next-token prediction. At fixed length,
                // depth parity is necessarily fixed; report its prior as control.
                let mut depth = 0u8;
                for c in &mut input[..l - 2] {
                    let up = depth == 0 || (depth < 5 && rng.gen_bool(0.5));
                    if up {
                        depth += 1;
                        *c = '(';
                    } else {
                        depth -= 1;
                        *c = ')';
                    }
                }
                input[l - 2] = '=';
                target[l - 2] = (b'0' + depth) as char;
                mask[l - 2] = 1.;
            }
            TaskKind::Parity => {
                let mut parity = 0u8;
                for c in &mut input[..l - 2] {
                    let bit = rng.gen_range(0..2);
                    *c = (b'0' + bit) as char;
                    parity ^= bit;
                }
                input[l - 2] = '=';
                target[l - 2] = (b'0' + parity) as char;
                mask[l - 2] = 1.;
            }
            TaskKind::Reverse => {
                let n = (l - 1) / 2;
                for c in &mut input[..n] {
                    *c = (b'a' + rng.gen_range(0..8)) as char;
                }
                input[n] = ':';
                for i in 0..n {
                    input[n + 1 + i] = '?';
                    target[n + 1 + i] = input[n - 1 - i];
                    mask[n + 1 + i] = 1.;
                }
            }
            TaskKind::HiddenRule => {
                // The instruction is explicit: this is conditional digit
                // transformation, not induction of an unobserved rule.
                let add = rng.gen_bool(0.5);
                input[0] = if add { '+' } else { '-' };
                for i in 1..l {
                    let digit: u8 = rng.gen_range(0..10);
                    input[i] = (b'0' + digit) as char;
                    target[i] = (b'0' + (digit + if add { 1 } else { 9 }) % 10) as char;
                    mask[i] = 1.;
                }
            }
            TaskKind::Associative => {
                let mut keys = ['k', 'm', 'p', 'r'];
                keys.shuffle(rng);
                let n = ((l - 2) / 4).min(keys.len());
                let mut values = Vec::new();
                for i in 0..n {
                    let value = (b'0' + rng.gen_range(0..10)) as char;
                    values.push(value);
                    input[4 * i..4 * i + 4].copy_from_slice(&[keys[i], '=', value, ';']);
                }
                let selected = rng.gen_range(0..n);
                // Fix query at the end so length controls separation from the dictionary.
                input[l - 2] = keys[selected];
                input[l - 1] = '?';
                target[l - 1] = values[selected];
                mask[l - 1] = 1.;
            }
            TaskKind::AmbiguousBasin => {
                // Context-conditioned lookup with distractors, no basin claim.
                // Independent cue selects one of two possible digit outputs.
                for c in &mut input {
                    *c = (b'a' + rng.gen_range(0..8)) as char;
                }
                let alpha = rng.gen_bool(0.5);
                input[0] = if alpha { 'A' } else { 'B' };
                let cue: u8 = rng.gen_range(0..8);
                input[l - 2] = (b'0' + cue) as char;
                input[l - 1] = '?';
                target[l - 1] = (b'0' + (cue + if alpha { 1 } else { 2 }) % 10) as char;
                mask[l - 1] = 1.;
            }
            TaskKind::ColumnArithmetic => {
                // Multi-digit addition with carry propagation.
                // Layout for length l >= 16:
                // d digits for operand A, '+', d digits for operand B, '=', d+1 query slots '?'
                // remaining cells padded with '.'
                let d = ((l - 4) / 3).min(18).max(1);
                input.fill('.');
                target.fill('.');
                // 50% chance of hostile ripple carry (e.g. 99...9 + 1)
                let hostile_carry = rng.gen_bool(0.5);
                let max_d_val = 10u64.pow(d as u32);
                let (val_a, val_b) = if hostile_carry {
                    let a = max_d_val - 1; // 99...9
                    let b = rng.gen_range(1..10u64.min(max_d_val));
                    (a, b)
                } else {
                    let a = rng.gen_range(0..max_d_val);
                    let b = rng.gen_range(0..max_d_val);
                    (a, b)
                };
                let sum = val_a + val_b;
                let str_a = format!("{:0width$}", val_a, width = d);
                let str_b = format!("{:0width$}", val_b, width = d);
                let str_sum = format!("{:0width$}", sum, width = d + 1);

                for (i, c) in str_a.chars().enumerate() {
                    input[i] = c;
                }
                input[d] = '+';
                for (i, c) in str_b.chars().enumerate() {
                    input[d + 1 + i] = c;
                }
                input[2 * d + 1] = '=';
                for (i, c) in str_sum.chars().enumerate() {
                    let pos = 2 * d + 2 + i;
                    input[pos] = '?';
                    target[pos] = c;
                    mask[pos] = 1.;
                }
            }
            TaskKind::IteratedParity | TaskKind::IteratedParityDense | TaskKind::IteratedParityCarrier => {
                // Chunked / Iterated Parity with Positional Readout (IPPR)
                // Divides sequence into chunks of 4 tokens: 3 data bits + 1 query slot '?'.
                // Target at query slot is the cumulative parity of all data bits up to that chunk.
                let chunk_size = 4;
                let num_chunks = l / chunk_size;
                let mut cum_parity = 0u8;
                for chunk in 0..num_chunks {
                    let base = chunk * chunk_size;
                    for j in 0..chunk_size - 1 {
                        let bit = rng.gen_range(0..2);
                        input[base + j] = (b'0' + bit) as char;
                        cum_parity ^= bit;

                        match kind {
                            TaskKind::IteratedParityDense => {
                                target[base + j] = (b'0' + cum_parity) as char;
                                mask[base + j] = if !is_val { 0.5 } else { 0.0 };
                            }
                            TaskKind::IteratedParityCarrier => {
                                if j == chunk_size - 2 {
                                    // Carrier bit: last data bit before query slot
                                    target[base + j] = (b'0' + cum_parity) as char;
                                    mask[base + j] = if !is_val { 0.5 } else { 0.0 };
                                }
                            }
                            _ => {}
                        }
                    }
                    let q_pos = base + chunk_size - 1;
                    input[q_pos] = '?';
                    target[q_pos] = (b'0' + cum_parity) as char;
                    mask[q_pos] = 1.0;
                }
            }
            TaskKind::IteratedSumDense => {
                // Chunked / Iterated Bounded Running Sum (Lipschitz Continuous Invariant)
                // Divides sequence into chunks of 4 tokens: 3 step tokens + 1 query slot '?'.
                // Step tokens are '+', '-', '0' (increments +1, -1, 0).
                // Target at query slot is cumulative bounded sum S in [0, 9], encoded as ASCII '0'..'9'.
                // Initial sum starts centered at 5.
                let chunk_size = 4;
                let num_chunks = l / chunk_size;
                let mut sum: i32 = 5;
                for chunk in 0..num_chunks {
                    let base = chunk * chunk_size;
                    for j in 0..chunk_size - 1 {
                        let delta: i32 = rng.gen_range(-1..=1);
                        let c = match delta {
                            -1 => '-',
                            0 => '0',
                            1 => '+',
                            _ => unreachable!(),
                        };
                        input[base + j] = c;
                        sum = (sum + delta).clamp(0, 9);
                        target[base + j] = (b'0' + sum as u8) as char;
                        mask[base + j] = if !is_val { 0.5 } else { 0.0 };
                    }
                    let q_pos = base + chunk_size - 1;
                    input[q_pos] = '?';
                    target[q_pos] = (b'0' + sum as u8) as char;
                    mask[q_pos] = 1.0;
                }
            }
            TaskKind::Text | TaskKind::Dyck => unreachable!(),
        }
        (input, target, mask)
    }

    #[allow(dead_code)]
    pub fn generate_suite(
        &self,
        tasks: &[TaskKind],
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        device: &Device,
    ) -> Result<Vec<(TaskKind, TaskBatch)>> {
        use rayon::prelude::*;
        tasks
            .par_iter()
            .map(|&kind| {
                Ok((
                    kind,
                    self.generate_batch(kind, batch_size, seq_len, is_val, device)?,
                ))
            })
            .collect()
    }

    /// Structural checks only. Chance equality with a neighboring digit is NOT
    /// evidence of leakage. Causal visibility and shortcut tests are separate.
    pub fn verify_task_suite_integrity(&self, device: &Device) -> Result<()> {
        for &kind in &ALGORITHMIC_TASKS {
            let batch = self.generate_batch_seeded(kind, 64, 32, false, 7, device)?;
            let inputs = batch.inputs.to_vec2::<u32>()?;
            let targets = batch.targets.to_vec2::<u32>()?;
            let masks = batch.loss_mask.to_vec2::<f32>()?;
            for ((input, target), mask) in inputs.iter().zip(&targets).zip(&masks) {
                ensure!(mask.iter().any(|&v| v == 1.), "empty task mask");
                ensure!(
                    mask.iter().all(|&v| v == 0. || v == 1.),
                    "nonbinary task mask"
                );
                ensure!(
                    input
                        .iter()
                        .chain(target)
                        .all(|&id| id != self.vocab.unk_id as u32
                            && (id as usize) < self.vocab.size()),
                    "invalid task token"
                );
            }
        }
        Ok(())
    }
}

pub const ALGORITHMIC_TASKS: [TaskKind; 9] = [
    TaskKind::DelayedRecall,
    TaskKind::BracketDepth,
    TaskKind::Parity,
    TaskKind::Reverse,
    TaskKind::HiddenRule,
    TaskKind::Associative,
    TaskKind::AmbiguousBasin,
    TaskKind::ColumnArithmetic,
    TaskKind::IteratedParity,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_task_suite_generation() -> Result<()> {
        TaskEngine::new().verify_task_suite_integrity(&Device::Cpu)
    }

    #[test]
    fn invalid_task_shapes_fail_without_panicking() {
        let e = TaskEngine::new();
        for kind in ALGORITHMIC_TASKS {
            for l in 0..8 {
                assert!(e.generate_batch(kind, 1, l, false, &Device::Cpu).is_err());
            }
            assert!(e.generate_batch(kind, 0, 32, false, &Device::Cpu).is_err());
        }
    }

    #[test]
    fn task_seed_determinism_diversity_and_content_disjoint_splits() -> Result<()> {
        let e = TaskEngine::new();
        for kind in ALGORITHMIC_TASKS {
            let a = e
                .generate_batch_seeded(kind, 1024, 16, false, 42, &Device::Cpu)?
                .inputs
                .to_vec2::<u32>()?;
            let same = e
                .generate_batch_seeded(kind, 1024, 16, false, 42, &Device::Cpu)?
                .inputs
                .to_vec2::<u32>()?;
            let other = e
                .generate_batch_seeded(kind, 1024, 16, false, 43, &Device::Cpu)?
                .inputs
                .to_vec2::<u32>()?;
            let val = e
                .generate_batch_seeded(kind, 1024, 16, true, 42, &Device::Cpu)?
                .inputs
                .to_vec2::<u32>()?;
            assert_eq!(a, same, "{} must be reproducible", kind.name());
            assert_ne!(a, other, "{} must use its seed", kind.name());
            let train_set: HashSet<_> = a.iter().collect();
            assert!(train_set.len() > 32, "{} collapsed universe", kind.name());
            assert!(
                val.iter().all(|row| !train_set.contains(row)),
                "{} split overlap",
                kind.name()
            );
        }
        Ok(())
    }

    #[test]
    fn labels_match_independent_oracles_and_masks() -> Result<()> {
        let e = TaskEngine::new();
        for kind in ALGORITHMIC_TASKS {
            for val in [false, true] {
                let batch = e.generate_batch_seeded(kind, 128, 16, val, 813, &Device::Cpu)?;
                let ins = batch.inputs.to_vec2::<u32>()?;
                let tgts = batch.targets.to_vec2::<u32>()?;
                let masks = batch.loss_mask.to_vec2::<f32>()?;
                for ((ids, t), m) in ins.iter().zip(&tgts).zip(&masks) {
                    let s: Vec<char> = e
                        .vocab
                        .decode(&ids.iter().map(|&v| v as usize).collect::<Vec<_>>())
                        .chars()
                        .collect();
                    let expected: Vec<(usize, char)> = match kind {
                        TaskKind::DelayedRecall => (0..4).map(|i| (12 + i, s[i])).collect(),
                        TaskKind::BracketDepth => vec![(
                            14,
                            char::from_digit(
                                s[..14]
                                    .iter()
                                    .map(|&c| if c == '(' { 1i32 } else { -1 })
                                    .sum::<i32>() as u32,
                                10,
                            )
                            .unwrap(),
                        )],
                        TaskKind::Parity => vec![(
                            14,
                            if s[..14].iter().filter(|&&c| c == '1').count() % 2 == 0 {
                                '0'
                            } else {
                                '1'
                            },
                        )],
                        TaskKind::Reverse => (0..7).map(|i| (8 + i, s[6 - i])).collect(),
                        TaskKind::HiddenRule => (1..16)
                            .map(|i| {
                                (
                                    i,
                                    char::from_digit(
                                        (s[i].to_digit(10).unwrap()
                                            + if s[0] == '+' { 1 } else { 9 })
                                            % 10,
                                        10,
                                    )
                                    .unwrap(),
                                )
                            })
                            .collect(),
                        TaskKind::Associative => {
                            let map: HashMap<_, _> =
                                s[..12].chunks(4).map(|p| (p[0], p[2])).collect();
                            vec![(15, map[&s[14]])]
                        }
                        TaskKind::AmbiguousBasin => vec![(
                            15,
                            char::from_digit(
                                (s[14].to_digit(10).unwrap() + if s[0] == 'A' { 1 } else { 2 })
                                    % 10,
                                10,
                            )
                            .unwrap(),
                        )],
                        TaskKind::ColumnArithmetic => {
                            let str_a: String = s[0..4].iter().collect();
                            let str_b: String = s[5..9].iter().collect();
                            let a: u64 = str_a.parse().unwrap();
                            let b: u64 = str_b.parse().unwrap();
                            let sum = a + b;
                            let str_sum = format!("{:05}", sum);
                            str_sum
                                .chars()
                                .enumerate()
                                .map(|(i, c)| (10 + i, c))
                                .collect()
                        }
                        TaskKind::IteratedParity => {
                            let mut expected = Vec::new();
                            let mut cum = 0u8;
                            for chunk in 0..4 {
                                let base = chunk * 4;
                                for j in 0..3 {
                                    cum ^= s[base + j].to_digit(10).unwrap() as u8;
                                }
                                expected.push((base + 3, (b'0' + cum) as char));
                            }
                            expected
                        }
                        _ => unreachable!(),
                    };
                    assert_eq!(m.iter().filter(|&&v| v == 1.).count(), expected.len());
                    for (pos, c) in expected {
                        assert_eq!(m[pos], 1.);
                        assert_eq!(t[pos], e.vocab.encode(&c.to_string())[0] as u32);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn marginal_controls_cannot_solve_randomized_tasks() -> Result<()> {
        let e = TaskEngine::new();
        for kind in [
            TaskKind::Parity,
            TaskKind::HiddenRule,
            TaskKind::Associative,
            TaskKind::AmbiguousBasin,
        ] {
            let batch = e.generate_batch_seeded(kind, 4096, 16, true, 91, &Device::Cpu)?;
            let input = batch.inputs.to_vec2::<u32>()?;
            let target = batch.targets.to_vec2::<u32>()?;
            let mut labels = HashMap::<u32, usize>::new();
            let mut local = HashMap::<u32, HashMap<u32, usize>>::new();
            for (x, y) in input.iter().zip(&target) {
                let pos = if kind == TaskKind::Parity { 14 } else { 15 };
                *labels.entry(y[pos]).or_default() += 1;
                // Local digit for hidden-rule; query key/cue for other tasks.
                let cue = x[if kind == TaskKind::HiddenRule { 15 } else { 14 }];
                *local.entry(cue).or_default().entry(y[pos]).or_default() += 1;
            }
            let prior = *labels.values().max().unwrap() as f32 / 4096.;
            let local_acc = local
                .values()
                .map(|counts| counts.values().max().unwrap())
                .sum::<usize>() as f32
                / 4096.;
            assert!(prior < 0.60, "{} prior={prior}", kind.name());
            assert!(local_acc < 0.65, "{} local={local_acc}", kind.name());
            println!(
                "task={} majority={prior:.4} cue_lookup={local_acc:.4}",
                kind.name()
            );
        }
        Ok(())
    }

    #[test]
    fn task_engine_dyck_uses_declared_vocabulary() -> Result<()> {
        let e = TaskEngine::new();
        let batch = e.generate_batch(TaskKind::Dyck, 1, 16, false, &Device::Cpu)?;
        let ids = batch.inputs.flatten_all()?.to_vec1::<u32>()?;
        assert!(ids
            .iter()
            .all(|&v| e.vocab.encode("()").contains(&(v as usize))));
        Ok(())
    }

    #[test]
    fn impossible_arithmetic_constraints_do_not_silently_emit_zero_plus_zero() {
        let engine = TaskEngine::new();
        assert!(engine.generate_stratified_arithmetic_batch(1, 32,
            Some(ArithmeticStratum::MaxRipple), Some(3), 42, &Device::Cpu).is_err());
        // The rejection sampler must also fail for incompatible other strata.
        assert!(engine.generate_stratified_arithmetic_batch(1, 16,
            Some(ArithmeticStratum::LongPartial), Some(0), 42, &Device::Cpu).is_err());
    }

    #[test]
    fn test_column_arithmetic_carry_chain_oracle() -> Result<()> {
        // 999 + 1 = 1000: carry ripples through all 3 digits
        assert_eq!(calculate_carry_chain(999, 1, 3), (3, 3));

        // 123 + 456 = 579: zero carries
        assert_eq!(calculate_carry_chain(123, 456, 3), (0, 0));

        // 199 + 1 = 200: carry propagates 2 digits (units and tens), then stops
        assert_eq!(calculate_carry_chain(199, 1, 3), (2, 2));

        // 9909 + 1 = 9910: units carry 1 digit, stops at 0
        assert_eq!(calculate_carry_chain(9909, 1, 4), (1, 1));

        let e = TaskEngine::new();
        // Stratum S1: MaxRipple
        let b_ripple = e.generate_stratified_arithmetic_batch(16, 32, Some(ArithmeticStratum::MaxRipple), None, 42, &Device::Cpu)?;
        let depths = b_ripple.carry_depths.unwrap();
        // For d = (32-4)/3 = 9, max ripple carry depth is 9
        assert!(depths.iter().all(|&d| d >= 8), "MaxRipple must have deep carry chains");

        // Stratum S4: NoCarry
        let b_nocarry = e.generate_stratified_arithmetic_batch(16, 32, Some(ArithmeticStratum::NoCarry), None, 42, &Device::Cpu)?;
        let depths_no = b_nocarry.carry_depths.unwrap();
        assert!(depths_no.iter().all(|&d| d == 0), "NoCarry must have strictly 0 carry depth");

        // Bounded carry depth for training
        let b_bounded = e.generate_stratified_arithmetic_batch(32, 32, None, Some(3), 42, &Device::Cpu)?;
        let depths_b = b_bounded.carry_depths.unwrap();
        assert!(depths_b.iter().all(|&d| d <= 3), "Bounded batch must not exceed max_carry_depth");

        Ok(())
    }

    #[test]
    fn test_audit_max_ripple_constant_zero_bias_demonstration() -> Result<()> {
        let e = TaskEngine::new();
        // Generate MaxRipple batch: 99..9 + 1 = 100..0
        let b_ripple = e.generate_stratified_arithmetic_batch(
            64, 32, Some(ArithmeticStratum::MaxRipple), None, 42, &Device::Cpu)?;
        let tgts = b_ripple.targets.to_vec2::<u32>()?;
        let masks = b_ripple.loss_mask.to_vec2::<f32>()?;
        let zero_id = e.vocab.encode("0")[0] as u32;

        let mut ripple_zero_matches = 0;
        let mut ripple_total_queries = 0;
        let mut ripple_complete_zeros = 0;

        for (row_t, row_m) in tgts.iter().zip(&masks) {
            let mut all_zero = true;
            for (&t, &m) in row_t.iter().zip(row_m) {
                if m == 1.0 {
                    ripple_total_queries += 1;
                    if t == zero_id {
                        ripple_zero_matches += 1;
                    } else {
                        all_zero = false;
                    }
                }
            }
            if all_zero {
                ripple_complete_zeros += 1;
            }
        }
        let ripple_digit_acc = ripple_zero_matches as f32 / ripple_total_queries as f32;
        let ripple_complete_acc = ripple_complete_zeros as f32 / tgts.len() as f32;

        // On MaxRipple, constant-0 achieves > 75% digit accuracy because of leading 1 followed by zeros
        assert!(
            ripple_digit_acc > 0.75,
            "MaxRipple digit accuracy for constant-0 must exceed 75% due to zeros in 100..0: got {:.2}%",
            ripple_digit_acc * 100.0
        );
        // BUT complete-answer accuracy is strictly 0.0% because the leading digit is '1'
        assert_eq!(
            ripple_complete_acc, 0.0,
            "Complete-answer accuracy for constant-0 on MaxRipple must be strictly 0%"
        );

        // Compare with generic/random addition
        let b_random = e.generate_batch_seeded(TaskKind::ColumnArithmetic, 64, 32, false, 42, &Device::Cpu)?;
        let rand_tgts = b_random.targets.to_vec2::<u32>()?;
        let rand_masks = b_random.loss_mask.to_vec2::<f32>()?;
        let mut rand_zero_matches = 0;
        let mut rand_total_queries = 0;
        for (row_t, row_m) in rand_tgts.iter().zip(&rand_masks) {
            for (&t, &m) in row_t.iter().zip(row_m) {
                if m == 1.0 {
                    rand_total_queries += 1;
                    if t == zero_id {
                        rand_zero_matches += 1;
                    }
                }
            }
        }
        let rand_digit_acc = rand_zero_matches as f32 / rand_total_queries as f32;

        // On legacy random arithmetic, the hard-coded `hostile_carry = rng.gen_bool(0.5)`
        // causes 50% of examples to be 99..9 + b, artificially driving constant-0 digit accuracy
        // to ~43-45% (matching Codex Review Finding 2: seed 42 = 44.26%, seed 101 = 45.39%).
        assert!(
            rand_digit_acc > 0.38 && rand_digit_acc < 0.50,
            "Legacy ColumnArithmetic has 50% hostile carry, yielding ~43-45% zeros: got {:.2}%",
            rand_digit_acc * 100.0
        );

        println!(
            "Audit verification passed: MaxRipple constant-0 digit_acc={:.2}% vs legacy random digit_acc={:.2}%, complete_acc=0%",
            ripple_digit_acc * 100.0,
            rand_digit_acc * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_audit_train_val_split_disjointness_and_carry_depth_labels() -> Result<()> {
        let e = TaskEngine::new();
        // Check train vs validation split disjointness
        let train_b = e.generate_batch_seeded(TaskKind::ColumnArithmetic, 256, 32, false, 100, &Device::Cpu)?;
        let val_b = e.generate_batch_seeded(TaskKind::ColumnArithmetic, 256, 32, true, 100, &Device::Cpu)?;

        let train_ins = train_b.inputs.to_vec2::<u32>()?;
        let val_ins = val_b.inputs.to_vec2::<u32>()?;

        let train_set: HashSet<Vec<u32>> = train_ins.into_iter().collect();
        let val_set: HashSet<Vec<u32>> = val_ins.into_iter().collect();

        assert!(
            train_set.is_disjoint(&val_set),
            "Train and validation observation sets must be strictly disjoint"
        );

        // Check carry depth bounds for in-distribution vs extrapolation
        let id_b = e.generate_stratified_arithmetic_batch(64, 32, None, Some(2), 123, &Device::Cpu)?;
        let ood_b = e.generate_stratified_arithmetic_batch(64, 32, Some(ArithmeticStratum::LongPartial), None, 123, &Device::Cpu)?;

        let id_depths = id_b.carry_depths.unwrap();
        let ood_depths = ood_b.carry_depths.unwrap();

        assert!(
            id_depths.iter().all(|&d| d <= 2),
            "In-distribution carry depths must be <= 2"
        );
        assert!(
            ood_depths.iter().all(|&d| d >= 2),
            "Extrapolation carry depths must be >= 2"
        );

        Ok(())
    }

    #[test]
    fn test_iterated_parity_oracle_and_recurrent_depth_requirement() -> Result<()> {
        let e = TaskEngine::new();
        let batch = e.generate_batch_seeded(TaskKind::IteratedParity, 128, 32, false, 777, &Device::Cpu)?;
        let ins = batch.inputs.to_vec2::<u32>()?;
        let tgts = batch.targets.to_vec2::<u32>()?;
        let masks = batch.loss_mask.to_vec2::<f32>()?;

        let zero_id = e.vocab.encode("0")[0] as u32;
        let one_id = e.vocab.encode("1")[0] as u32;

        let mut zero_count = 0;
        let mut one_count = 0;
        let mut query_count = 0;

        for ((row_in, row_t), row_m) in ins.iter().zip(&tgts).zip(&masks) {
            let s: Vec<char> = e.vocab.decode(&row_in.iter().map(|&v| v as usize).collect::<Vec<_>>()).chars().collect();
            let mut cum = 0u8;
            for chunk in 0..8 {
                let base = chunk * 4;
                for j in 0..3 {
                    cum ^= s[base + j].to_digit(10).unwrap() as u8;
                }
                let q_pos = base + 3;
                assert_eq!(row_m[q_pos], 1.0, "Mask must be 1 at query position {}", q_pos);
                let expected_c = (b'0' + cum) as char;
                let actual_id = row_t[q_pos];
                assert_eq!(actual_id, e.vocab.encode(&expected_c.to_string())[0] as u32);

                if actual_id == zero_id {
                    zero_count += 1;
                } else if actual_id == one_id {
                    one_count += 1;
                }
                query_count += 1;
            }
        }

        // Verify balanced label distribution
        let zero_frac = zero_count as f32 / query_count as f32;
        let one_frac = one_count as f32 / query_count as f32;
        assert!(
            (zero_frac - 0.5).abs() < 0.08,
            "IteratedParity targets must be balanced: zero_frac={:.3}, one_frac={:.3}",
            zero_frac, one_frac
        );

        Ok(())
    }

    #[test]
    fn test_iterated_parity_dense_and_carrier_auxiliary_masks() -> Result<()> {
        let e = TaskEngine::new();
        let dev = Device::Cpu;

        // 1. Check Dense Train (is_val = false)
        let tr_dense = e.generate_batch_seeded(TaskKind::IteratedParityDense, 16, 16, false, 42, &dev)?;
        let tr_dense_m = tr_dense.loss_mask.to_vec2::<f32>()?;
        for row in tr_dense_m {
            for chunk in 0..4 {
                let base = chunk * 4;
                // data bits should have mask 0.5
                assert_eq!(row[base], 0.5);
                assert_eq!(row[base + 1], 0.5);
                assert_eq!(row[base + 2], 0.5);
                // query slot has mask 1.0
                assert_eq!(row[base + 3], 1.0);
            }
        }

        // 2. Check Dense Val (is_val = true) -> data bits must be 0.0, query slots 1.0
        let va_dense = e.generate_batch_seeded(TaskKind::IteratedParityDense, 16, 16, true, 42, &dev)?;
        let va_dense_m = va_dense.loss_mask.to_vec2::<f32>()?;
        for row in va_dense_m {
            for chunk in 0..4 {
                let base = chunk * 4;
                assert_eq!(row[base], 0.0);
                assert_eq!(row[base + 1], 0.0);
                assert_eq!(row[base + 2], 0.0);
                assert_eq!(row[base + 3], 1.0);
            }
        }

        // 3. Check Carrier Train (is_val = false) -> only carrier bit base + 2 has 0.5
        let tr_carrier = e.generate_batch_seeded(TaskKind::IteratedParityCarrier, 16, 16, false, 42, &dev)?;
        let tr_carrier_m = tr_carrier.loss_mask.to_vec2::<f32>()?;
        for row in tr_carrier_m {
            for chunk in 0..4 {
                let base = chunk * 4;
                assert_eq!(row[base], 0.0);
                assert_eq!(row[base + 1], 0.0);
                assert_eq!(row[base + 2], 0.5);
                assert_eq!(row[base + 3], 1.0);
            }
        }

        // 4. Check Carrier Val (is_val = true) -> all data bits 0.0, query slots 1.0
        let va_carrier = e.generate_batch_seeded(TaskKind::IteratedParityCarrier, 16, 16, true, 42, &dev)?;
        let va_carrier_m = va_carrier.loss_mask.to_vec2::<f32>()?;
        for row in va_carrier_m {
            for chunk in 0..4 {
                let base = chunk * 4;
                assert_eq!(row[base], 0.0);
                assert_eq!(row[base + 1], 0.0);
                assert_eq!(row[base + 2], 0.0);
                assert_eq!(row[base + 3], 1.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_iterated_sum_dense_generation() -> Result<()> {
        let e = TaskEngine::new();
        let dev = Device::Cpu;

        let batch_train = e.generate_batch_seeded(TaskKind::IteratedSumDense, 8, 16, false, 123, &dev)?;
        let m_train = batch_train.loss_mask.to_vec2::<f32>()?;
        assert_eq!(m_train.len(), 8);
        assert_eq!(m_train[0].len(), 16);

        // Verify training masks: step bits 0.5, query bits 1.0
        for row in &m_train {
            for chunk in 0..4 {
                let base = chunk * 4;
                assert_eq!(row[base], 0.5);
                assert_eq!(row[base + 1], 0.5);
                assert_eq!(row[base + 2], 0.5);
                assert_eq!(row[base + 3], 1.0);
            }
        }

        // Verify validation masks: step bits 0.0, query bits 1.0
        let batch_val = e.generate_batch_seeded(TaskKind::IteratedSumDense, 8, 16, true, 123, &dev)?;
        let m_val = batch_val.loss_mask.to_vec2::<f32>()?;
        for row in &m_val {
            for chunk in 0..4 {
                let base = chunk * 4;
                assert_eq!(row[base], 0.0);
                assert_eq!(row[base + 1], 0.0);
                assert_eq!(row[base + 2], 0.0);
                assert_eq!(row[base + 3], 1.0);
            }
        }

        // Verify decoded characters
        let inputs_vec = batch_val.inputs.to_vec2::<u32>()?;
        let targets_vec = batch_val.targets.to_vec2::<u32>()?;
        for b in 0..8 {
            let in_chars: Vec<char> = inputs_vec[b].iter().map(|&id| e.vocab.decode(&[id as usize]).chars().next().unwrap()).collect();
            let tgt_chars: Vec<char> = targets_vec[b].iter().map(|&id| e.vocab.decode(&[id as usize]).chars().next().unwrap()).collect();
            for chunk in 0..4 {
                let base = chunk * 4;
                assert!(in_chars[base] == '+' || in_chars[base] == '-' || in_chars[base] == '0');
                assert!(in_chars[base + 1] == '+' || in_chars[base + 1] == '-' || in_chars[base + 1] == '0');
                assert!(in_chars[base + 2] == '+' || in_chars[base + 2] == '-' || in_chars[base + 2] == '0');
                assert_eq!(in_chars[base + 3], '?');
                assert!(tgt_chars[base + 3].is_ascii_digit());
            }
        }

        Ok(())
    }
}
