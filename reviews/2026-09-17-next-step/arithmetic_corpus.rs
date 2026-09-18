//! Opt-in arithmetic corpus with exact carry-depth conditioning.
//!
//! This module is intentionally self-contained and does not touch legacy task
//! generators, training, checkpoints, or Cargo dependencies. It produces pure
//! serializable string/data records.
//!
//! Limitations (explicit):
//! - Carry bit patterns with exact max-run k are sampled by bounded rejection.
//!   This is not a uniform distribution over all such patterns; it is a
//!   rejection-sampled distribution. We do not claim uniform conditioned
//!   sampling of (a_digit, b_digit) pairs either.
//! - Digit pairs at each column are chosen uniformly from the valid set
//!   conditioned on (incoming carry, outgoing carry). This is a simple
//!   constructive scheme, not a uniform distribution over all operand pairs
//!   with the given carry depth.
//! - No 99...9+1 templates are used.
//! - MaxRipple is preserved elsewhere as a historical stress fixture and is
//!   not part of this corpus.

use anyhow::{anyhow, bail, Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for this corpus format.
pub const ARITHMETIC_CORPUS_SCHEMA: &str = "titan_text.arithmetic_corpus.v1";

/// Hard resource bounds.
pub const MAX_WIDTH: usize = 9;
pub const MAX_DEPTH: usize = MAX_WIDTH;
pub const MAX_COUNT_PER_DEPTH: usize = 4096;
pub const MAX_TOTAL_ROWS: usize = 200_000;

/// Bounded retry budget for unique-ID and carry-pattern sampling.
const MAX_ID_RETRIES: usize = 4096;
const MAX_CARRY_RETRIES: usize = 100_000;

/// Split assignment buckets (80/10/10).
const SPLIT_TRAIN_MAX: u64 = 80;
const SPLIT_VAL_MAX: u64 = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusConfig {
    pub seed: u64,
    pub width: usize,
    pub depths: Vec<usize>,
    pub count_per_depth: usize,
}

impl CorpusConfig {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.width > MAX_WIDTH {
            bail!("width must be in 1..={}, got {}", MAX_WIDTH, self.width);
        }
        if self.depths.is_empty() {
            bail!("depths must be non-empty");
        }
        let mut seen = BTreeSet::new();
        for &d in &self.depths {
            if d > self.width {
                bail!("depth {} exceeds width {}", d, self.width);
            }
            if !seen.insert(d) {
                bail!("duplicate depth {}", d);
            }
        }
        if self.count_per_depth == 0 {
            bail!("count_per_depth must be > 0");
        }
        if self.count_per_depth > MAX_COUNT_PER_DEPTH {
            bail!(
                "count_per_depth {} exceeds hard bound {}",
                self.count_per_depth, MAX_COUNT_PER_DEPTH
            );
        }
        let total = self.depths.len().saturating_mul(self.count_per_depth);
        if total > MAX_TOTAL_ROWS {
            bail!("total rows {} exceeds hard bound {}", total, MAX_TOTAL_ROWS);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusRecord {
    pub id: String,
    pub a: u64,
    pub b: u64,
    pub width: usize,
    pub depth: usize,
    pub split: Split,
    pub observation: String,
    pub targets: String,
    pub mask: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    pub schema: String,
    pub seed: u64,
    pub config: CorpusConfig,
    pub records: Vec<CorpusRecord>,
}

/// FNV1a over defined bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Canonicalize unordered operands: (min, max).
pub fn canonical_operands(a: u64, b: u64) -> (u64, u64) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Collision-free ID string of width and canonical operands.
pub fn record_id(width: usize, a: u64, b: u64) -> String {
    let (lo, hi) = canonical_operands(a, b);
    format!("w{}_a{}_b{}", width, lo, hi)
}

/// Seed-independent split assignment from canonical ID.
pub fn split_for(width: usize, a: u64, b: u64) -> Split {
    let (lo, hi) = canonical_operands(a, b);
    let id = record_id(width, lo, hi);
    let bucket = fnv1a(id.as_bytes()) % 100;
    if bucket < SPLIT_TRAIN_MAX {
        Split::Train
    } else if bucket < SPLIT_VAL_MAX {
        Split::Validation
    } else {
        Split::Test
    }
}

/// Compute exact maximum consecutive carry-chain length including final overflow carry.
///
/// Definition: process columns from least significant to most significant. A column
/// has outgoing carry 1 if a_digit + b_digit + incoming_carry >= 10. The carry depth
/// is the length of the longest run of consecutive columns whose outgoing carry is 1,
/// including the final overflow carry (the carry out of the most significant column).
pub fn carry_depth(a: u64, b: u64, width: usize) -> usize {
    let mut carry_in: u64 = 0;
    let mut run: usize = 0;
    let mut best: usize = 0;
    let mut aa = a;
    let mut bb = b;
    for _ in 0..width {
        let ad = aa % 10;
        let bd = bb % 10;
        aa /= 10;
        bb /= 10;
        let s = ad + bd + carry_in;
        let carry_out = if s >= 10 { 1 } else { 0 };
        if carry_out == 1 {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
        carry_in = carry_out;
    }
    // Final overflow carry out of the most significant column.
    if carry_in == 1 {
        run += 1;
        if run > best {
            best = run;
        }
    }
    best
}

/// Integer-addition oracle: format a+b zero-padded to width+1 digits.
pub fn oracle_sum_string(a: u64, b: u64, width: usize) -> String {
    let s = a + b;
    let total = width + 1;
    let mut out = String::with_capacity(total);
    let digits = format!("{}", s);
    if digits.len() > total {
        // Should not happen for valid width-bounded operands, but be explicit.
        return digits;
    }
    for _ in 0..(total - digits.len()) {
        out.push('0');
    }
    out.push_str(&digits);
    out
}

/// Sample a carry bit pattern of length `width+1` (columns 0..width-1 plus final
/// overflow carry) with exact maximum consecutive run of 1s equal to `k`.
///
/// Uses bounded rejection. Not uniform over all such patterns.
fn sample_carry_pattern(rng: &mut StdRng, width: usize, k: usize) -> Result<Vec<u8>> {
    let n = width + 1;
    if k > n {
        bail!("k {} exceeds pattern length {}", k, n);
    }
    for _ in 0..MAX_CARRY_RETRIES {
        let mut bits = vec![0u8; n];
        for b in bits.iter_mut() {
            *b = if rng.gen::<bool>() { 1 } else { 0 };
        }
        if max_run(&bits) == k {
            return Ok(bits);
        }
    }
    bail!(
        "carry pattern sampling exhausted retries for width={} k={}",
        width, k
    )
}

fn max_run(bits: &[u8]) -> usize {
    let mut run = 0usize;
    let mut best = 0usize;
    for &b in bits {
        if b == 1 {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

/// Choose a valid (a_digit, b_digit) pair at a column given incoming and outgoing carry.
///
/// Valid means: a_digit + b_digit + carry_in >= 10 iff carry_out == 1.
fn sample_digit_pair(rng: &mut StdRng, carry_in: u64, carry_out: u64) -> (u64, u64) {
    // Enumerate valid pairs (small space: 100 candidates).
    let mut candidates: Vec<(u64, u64)> = Vec::with_capacity(100);
    for ad in 0..10u64 {
        for bd in 0..10u64 {
            let s = ad + bd + carry_in;
            let co = if s >= 10 { 1 } else { 0 };
            if co == carry_out {
                candidates.push((ad, bd));
            }
        }
    }
    debug_assert!(!candidates.is_empty());
    let idx = rng.gen_range(0..candidates.len());
    candidates[idx]
}

/// Build a single record for the given width and target depth.
fn build_record(
    rng: &mut StdRng,
    width: usize,
    target_depth: usize,
    used_ids: &mut BTreeSet<String>,
) -> Result<CorpusRecord> {
    for _ in 0..MAX_ID_RETRIES {
        let bits = sample_carry_pattern(rng, width, target_depth)?;
        // bits[0] is outgoing carry of column 0 (least significant).
        // bits[width] is final overflow carry.
        let mut a: u64 = 0;
        let mut b: u64 = 0;
        let mut place: u64 = 1;
        let mut carry_in: u64 = 0;
        for col in 0..width {
            let carry_out = bits[col] as u64;
            let (ad, bd) = sample_digit_pair(rng, carry_in, carry_out);
            a += ad * place;
            b += bd * place;
            place *= 10;
            carry_in = carry_out;
        }
        // Final overflow carry must match bits[width].
        if carry_in != bits[width] as u64 {
            continue;
        }
        // Verify exact depth via independent oracle.
        let actual = carry_depth(a, b, width);
        if actual != target_depth {
            continue;
        }
        let id = record_id(width, a, b);
        if used_ids.contains(&id) {
            continue;
        }
        used_ids.insert(id.clone());
        let split = split_for(width, a, b);
        let (obs, targets, mask) = make_observation(a, b, width);
        return Ok(CorpusRecord {
            id,
            a,
            b,
            width,
            depth: actual,
            split,
            observation: obs,
            targets,
            mask,
        });
    }
    bail!(
        "unique-ID sampling exhausted retries for width={} depth={}",
        width, target_depth
    )
}

/// Build observation, targets, mask.
///
/// Observation: zero-padded A+B= followed by width+1 '?' characters, optionally
/// padded with '.' to a fixed total length. We pad to a fixed length of
/// (2*width + 2) + (width + 1) + 1 to keep panels uniform; the exact padding is
/// a design choice documented here.
///
/// Targets: digits at '?' positions, '.' elsewhere.
/// Mask: '1' at '?' positions, '0' elsewhere.
fn make_observation(a: u64, b: u64, width: usize) -> (String, String, String) {
    let a_str = format!("{:0width$}", a, width = width);
    let b_str = format!("{:0width$}", b, width = width);
    let sum_str = oracle_sum_string(a, b, width);
    let query_len = width + 1;
    let core = format!("{}+{}={}", a_str, b_str, "?".repeat(query_len));
    // Fixed total length for uniform panels.
    let total_len = core.len() + 1;
    let mut observation = core;
    while observation.len() < total_len {
        observation.push('.');
    }
    // Targets and mask over the full observation length.
    let mut targets = String::with_capacity(observation.len());
    let mut mask = String::with_capacity(observation.len());
    let query_start = observation.len() - query_len;
    for (i, _ch) in observation.chars().enumerate() {
        if i >= query_start {
            let d = sum_str.as_bytes()[i - query_start] as char;
            targets.push(d);
            mask.push('1');
        } else {
            targets.push('.');
            mask.push('0');
        }
    }
    (observation, targets, mask)
}

/// Build the full corpus.
pub fn build_corpus(config: &CorpusConfig) -> Result<Corpus> {
    config.validate().context("invalid corpus config")?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut used_ids: BTreeSet<String> = BTreeSet::new();
    let mut records: Vec<CorpusRecord> = Vec::new();
    for &depth in &config.depths {
        for _ in 0..config.count_per_depth {
            let rec = build_record(&mut rng, config.width, depth, &mut used_ids)?;
            records.push(rec);
        }
    }
    Ok(Corpus {
        schema: ARITHMETIC_CORPUS_SCHEMA.to_string(),
        seed: config.seed,
        config: config.clone(),
        records,
    })
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthPositionCount {
    pub depth: usize,
    pub position: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitOverlap {
    pub train_validation: usize,
    pub train_test: usize,
    pub validation_test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub digit_micro: f64,
    pub equal_depth_position_macro: f64,
    pub complete_answer: f64,
    pub leading_overflow_digit: f64,
    pub remaining_digit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audit {
    pub schema: String,
    pub seed: u64,
    pub config: CorpusConfig,
    pub total_records: usize,
    pub depth_counts: BTreeMap<usize, usize>,
    pub position_digit_counts: Vec<DepthPositionCount>,
    pub unique_ids: bool,
    pub split_overlap: SplitOverlap,
    pub oracle_mask_ok: bool,
    pub constant_zero: BaselineMetrics,
    pub global_digit_prior: BaselineMetrics,
    pub position_prior: BaselineMetrics,
    pub examples: Vec<CorpusRecord>,
}

/// Fit a digit prior over train targets only.
fn fit_global_digit_prior(corpus: &Corpus) -> [f64; 10] {
    let mut counts = [0f64; 10];
    for r in &corpus.records {
        if r.split != Split::Train {
            continue;
        }
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] == b'1' {
                if let Some(d) = ch.to_digit(10) {
                    counts[d as usize] += 1.0;
                }
            }
        }
    }
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return [0.1; 10];
    }
    for c in counts.iter_mut() {
        *c /= total;
    }
    counts
}

/// Fit a position prior over train targets only: for each (depth, position) cell,
/// the most frequent digit.
fn fit_position_prior(corpus: &Corpus) -> BTreeMap<(usize, usize), u8> {
    let mut counts: BTreeMap<(usize, usize), [usize; 10]> = BTreeMap::new();
    for r in &corpus.records {
        if r.split != Split::Train {
            continue;
        }
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] == b'1' {
                if let Some(d) = ch.to_digit(10) {
                    let key = (r.depth, i);
                    let entry = counts.entry(key).or_insert([0usize; 10]);
                    entry[d as usize] += 1;
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    for (k, arr) in counts {
        let mut best = 0usize;
        let mut best_d = 0u8;
        for (d, &c) in arr.iter().enumerate() {
            if c > best {
                best = c;
                best_d = d as u8;
            }
        }
        out.insert(k, best_d);
    }
    out
}

fn evaluate<F>(corpus: &Corpus, predict: F) -> Result<BaselineMetrics>
where
    F: Fn(&CorpusRecord, usize) -> u8,
{
    if corpus.records.is_empty() {
        bail!("cannot evaluate empty corpus");
    }
    let mut digit_correct = 0usize;
    let mut digit_total = 0usize;
    let mut cell_correct: BTreeMap<(usize, usize), (usize, usize)> = BTreeMap::new();
    let mut complete_correct = 0usize;
    let mut complete_total = 0usize;
    let mut leading_correct = 0usize;
    let mut leading_total = 0usize;
    let mut remaining_correct = 0usize;
    let mut remaining_total = 0usize;
    for r in &corpus.records {
        let mut all_ok = true;
        let mut any = false;
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] != b'1' {
                continue;
            }
            any = true;
            let truth = ch.to_digit(10).unwrap_or(0) as u8;
            let pred = predict(r, i);
            let ok = pred == truth;
            if ok {
                digit_correct += 1;
            }
            digit_total += 1;
            let cell = cell_correct.entry((r.depth, i)).or_insert((0, 0));
            cell.1 += 1;
            if ok {
                cell.0 += 1;
            }
            if !ok {
                all_ok = false;
            }
            // Leading overflow digit is the most significant query position.
            let query_positions: Vec<usize> = r
                .mask
                .char_indices()
                .filter(|(_, c)| *c == '1')
                .map(|(i, _)| i)
                .collect();
            if let Some(&last) = query_positions.last() {
                if i == last {
                    leading_total += 1;
                    if ok {
                        leading_correct += 1;
                    }
                } else {
                    remaining_total += 1;
                    if ok {
                        remaining_correct += 1;
                    }
                }
            }
        }
        if any {
            complete_total += 1;
            if all_ok {
                complete_correct += 1;
            }
        }
    }
    let digit_micro = if digit_total == 0 {
        bail!("no query positions in corpus");
    } else {
        digit_correct as f64 / digit_total as f64
    };
    let mut macro_sum = 0.0f64;
    let mut macro_n = 0usize;
    for (_, (c, t)) in cell_correct.iter() {
        if *t > 0 {
            macro_sum += *c as f64 / *t as f64;
            macro_n += 1;
        }
    }
    let equal_depth_position_macro = if macro_n == 0 {
        bail!("no (depth,position) cells");
    } else {
        macro_sum / macro_n as f64
    };
    let complete_answer = if complete_total == 0 {
        bail!("no complete answers");
    } else {
        complete_correct as f64 / complete_total as f64
    };
    let leading_overflow_digit = if leading_total == 0 {
        bail!("no leading overflow positions");
    } else {
        leading_correct as f64 / leading_total as f64
    };
    let remaining_digit = if remaining_total == 0 {
        bail!("no remaining positions");
    } else {
        remaining_correct as f64 / remaining_total as f64
    };
    Ok(BaselineMetrics {
        digit_micro,
        equal_depth_position_macro,
        complete_answer,
        leading_overflow_digit,
        remaining_digit,
    })
}

/// Build a JSON-serializable audit.
pub fn audit_corpus(corpus: &Corpus) -> Result<Audit> {
    if corpus.records.is_empty() {
        bail!("cannot audit empty corpus");
    }
    let mut depth_counts: BTreeMap<usize, usize> = BTreeMap::new();
    let mut position_digit_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut ids: BTreeSet<String> = BTreeSet::new();
    let mut unique_ids = true;
    let mut train_ids: BTreeSet<String> = BTreeSet::new();
    let mut val_ids: BTreeSet<String> = BTreeSet::new();
    let mut test_ids: BTreeSet<String> = BTreeSet::new();
    let mut oracle_mask_ok = true;
    for r in &corpus.records {
        *depth_counts.entry(r.depth).or_insert(0) += 1;
        if !ids.insert(r.id.clone()) {
            unique_ids = false;
        }
        match r.split {
            Split::Train => {
                train_ids.insert(r.id.clone());
            }
            Split::Validation => {
                val_ids.insert(r.id.clone());
            }
            Split::Test => {
                test_ids.insert(r.id.clone());
            }
        }
        // Oracle/mask verification.
        let expected = oracle_sum_string(r.a, r.b, r.width);
        let query_positions: Vec<usize> = r
            .mask
            .char_indices()
            .filter(|(_, c)| *c == '1')
            .map(|(i, _)| i)
            .collect();
        if query_positions.len() != r.width + 1 {
            oracle_mask_ok = false;
        }
        for (qi, &pos) in query_positions.iter().enumerate() {
            let t = r.targets.as_bytes()[pos] as char;
            if t != expected.as_bytes()[qi] as char {
                oracle_mask_ok = false;
            }
            *position_digit_counts.entry((r.depth, pos)).or_insert(0) += 1;
        }
        // Non-query positions must be '.' in targets.
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] == b'0' && ch != '.' {
                oracle_mask_ok = false;
            }
        }
    }
    let split_overlap = SplitOverlap {
        train_validation: train_ids.intersection(&val_ids).count(),
        train_test: train_ids.intersection(&test_ids).count(),
        validation_test: val_ids.intersection(&test_ids).count(),
    };
    let position_digit_counts_vec: Vec<DepthPositionCount> = position_digit_counts
        .into_iter()
        .map(|((depth, position), count)| DepthPositionCount {
            depth,
            position,
            count,
        })
        .collect();

    let global_prior = fit_global_digit_prior(corpus);
    let position_prior = fit_position_prior(corpus);

    let constant_zero = evaluate(corpus, |_r, _i| 0)?;
    let global_digit_prior_metrics = evaluate(corpus, |_r, _i| {
        let mut best = 0usize;
        let mut best_p = 0.0f64;
        for (d, &p) in global_prior.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = d;
            }
        }
        best as u8
    })?;
    let position_prior_metrics = evaluate(corpus, |r, i| {
        *position_prior.get(&(r.depth, i)).unwrap_or(&0)
    })?;

    let examples: Vec<CorpusRecord> = corpus.records.iter().take(5).cloned().collect();

    Ok(Audit {
        schema: corpus.schema.clone(),
        seed: corpus.seed,
        config: corpus.config.clone(),
        total_records: corpus.records.len(),
        depth_counts,
        position_digit_counts: position_digit_counts_vec,
        unique_ids,
        split_overlap,
        oracle_mask_ok,
        constant_zero,
        global_digit_prior: global_digit_prior_metrics,
        position_prior: position_prior_metrics,
        examples,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_small_width_oracle() {
        for width in 1..=3usize {
            let max = 10u64.pow(width as u32);
            for a in 0..max {
                for b in 0..max {
                    let d = carry_depth(a, b, width);
                    // Independent recomputation via string addition.
                    let sa = format!("{:0width$}", a, width = width);
                    let sb = format!("{:0width$}", b, width = width);
                    let mut carry = 0u64;
                    let mut run = 0usize;
                    let mut best = 0usize;
                    let ad: Vec<u64> = sa.chars().rev().map(|c| c.to_digit(10).unwrap() as u64).collect();
                    let bd: Vec<u64> = sb.chars().rev().map(|c| c.to_digit(10).unwrap() as u64).collect();
                    for i in 0..width {
                        let s = ad[i] + bd[i] + carry;
                        let co = if s >= 10 { 1 } else { 0 };
                        if co == 1 {
                            run += 1;
                            if run > best {
                                best = run;
                            }
                        } else {
                            run = 0;
                        }
                        carry = co;
                    }
                    if carry == 1 {
                        run += 1;
                        if run > best {
                            best = run;
                        }
                    }
                    assert_eq!(d, best, "a={} b={} width={}", a, b, width);
                }
            }
        }
    }

    #[test]
    fn exact_depth_across_k_and_seeds() {
        for seed in [1u64, 2, 3] {
            for width in 1..=4usize {
                for k in 0..=width {
                    let cfg = CorpusConfig {
                        seed,
                        width,
                        depths: vec![k],
                        count_per_depth: 4,
                    };
                    let corpus = build_corpus(&cfg).expect("build");
                    for r in &corpus.records {
                        assert_eq!(r.depth, k, "seed={} width={} k={}", seed, width, k);
                        assert_eq!(carry_depth(r.a, r.b, width), k);
                    }
                }
            }
        }
    }

    #[test]
    fn reproducibility_same_seed() {
        let cfg = CorpusConfig {
            seed: 42,
            width: 4,
            depths: vec![0, 1, 2],
            count_per_depth: 8,
        };
        let c1 = build_corpus(&cfg).unwrap();
        let c2 = build_corpus(&cfg).unwrap();
        assert_eq!(c1.records.len(), c2.records.len());
        for (a, b) in c1.records.iter().zip(c2.records.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.a, b.a);
            assert_eq!(a.b, b.b);
            assert_eq!(a.split, b.split);
        }
    }

    #[test]
    fn swap_invariant_split_ownership() {
        let cfg = CorpusConfig {
            seed: 7,
            width: 4,
            depths: vec![0, 1, 2, 3, 4],
            count_per_depth: 16,
        };
        let corpus = build_corpus(&cfg).unwrap();
        for r in &corpus.records {
            let s1 = split_for(r.width, r.a, r.b);
            let s2 = split_for(r.width, r.b, r.a);
            assert_eq!(s1, s2);
            assert_eq!(r.split, s1);
        }
    }

    #[test]
    fn unique_and_disjoint_panels() {
        let cfg = CorpusConfig {
            seed: 11,
            width: 4,
            depths: vec![0, 1, 2, 3, 4],
            count_per_depth: 32,
        };
        let corpus = build_corpus(&cfg).unwrap();
        let audit = audit_corpus(&corpus).unwrap();
        assert!(audit.unique_ids);
        assert_eq!(audit.split_overlap.train_validation, 0);
        assert_eq!(audit.split_overlap.train_test, 0);
        assert_eq!(audit.split_overlap.validation_test, 0);
    }

    #[test]
    fn masks_and_query_only_labels() {
        let cfg = CorpusConfig {
            seed: 13,
            width: 3,
            depths: vec![0, 1, 2, 3],
            count_per_depth: 8,
        };
        let corpus = build_corpus(&cfg).unwrap();
        for r in &corpus.records {
            assert_eq!(r.observation.len(), r.targets.len());
            assert_eq!(r.observation.len(), r.mask.len());
            let q = r.mask.chars().filter(|c| *c == '1').count();
            assert_eq!(q, r.width + 1);
            for (i, ch) in r.targets.chars().enumerate() {
                if r.mask.as_bytes()[i] == b'0' {
                    assert_eq!(ch, '.');
                } else {
                    assert!(ch.is_ascii_digit());
                }
            }
        }
    }

    #[test]
    fn invalid_requests_rejected() {
        let bad = CorpusConfig {
            seed: 1,
            width: 0,
            depths: vec![0],
            count_per_depth: 1,
        };
        assert!(build_corpus(&bad).is_err());
        let bad2 = CorpusConfig {
            seed: 1,
            width: 4,
            depths: vec![5],
            count_per_depth: 1,
        };
        assert!(build_corpus(&bad2).is_err());
        let bad3 = CorpusConfig {
            seed: 1,
            width: 4,
            depths: vec![0],
            count_per_depth: 0,
        };
        assert!(build_corpus(&bad3).is_err());
    }

    #[test]
    fn empty_corpus_metrics_rejected() {
        let empty = Corpus {
            schema: ARITHMETIC_CORPUS_SCHEMA.to_string(),
            seed: 0,
            config: CorpusConfig {
                seed: 0,
                width: 1,
                depths: vec![0],
                count_per_depth: 1,
            },
            records: vec![],
        };
        assert!(audit_corpus(&empty).is_err());
    }

    #[test]
    fn prior_fitting_and_balanced_metrics() {
        let cfg = CorpusConfig {
            seed: 21,
            width: 4,
            depths: vec![0, 1, 2, 3, 4],
            count_per_depth: 16,
        };
        let corpus = build_corpus(&cfg).unwrap();
        let audit = audit_corpus(&corpus).unwrap();
        for m in [
            &audit.constant_zero,
            &audit.global_digit_prior,
            &audit.position_prior,
        ] {
            assert!(m.digit_micro.is_finite());
            assert!(m.equal_depth_position_macro.is_finite());
            assert!(m.complete_answer.is_finite());
            assert!(m.leading_overflow_digit.is_finite());
            assert!(m.remaining_digit.is_finite());
        }
    }

    #[test]
    fn default_config_shape() {
        let cfg = CorpusConfig {
            seed: 1234,
            width: 4,
            depths: vec![0, 1, 2, 3, 4],
            count_per_depth: 64,
        };
        let corpus = build_corpus(&cfg).unwrap();
        assert_eq!(corpus.records.len(), 960);
    }
}
