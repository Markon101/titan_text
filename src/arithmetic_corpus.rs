//! Opt-in arithmetic corpus with exact carry-depth conditioning.
//!
//! Self-contained: no legacy task generators, training, checkpoints, or new
//! Cargo dependencies. Produces pure serializable string/data records.
//!
//! Limitations (explicit):
//! - Carry bit patterns of length `width` with exact maximum run `k` are
//!   enumerated exhaustively (2^width <= 512) and one is chosen uniformly.
//!   This is uniform over patterns, not over operand pairs.
//! - Digit pairs at each column are chosen uniformly from the valid set
//!   conditioned on (incoming carry, outgoing carry). This is NOT a uniform
//!   distribution over all operand pairs with the given carry depth.
//! - No 99...9+1 templates are used.
//! - MaxRipple is preserved elsewhere as a historical stress fixture and is
//!   not part of this corpus.

use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for this corpus format.
pub const ARITHMETIC_CORPUS_SCHEMA: &str = "titan_text.arithmetic_corpus.v1";

/// Hard resource bounds.
pub const MAX_WIDTH: usize = 9;
pub const MAX_COUNT_PER_DEPTH: usize = 4096;

/// Bounded retry budget shared per row for unique-ID sampling.
const MAX_ROW_RETRIES: usize = 4096;

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
                self.count_per_depth,
                MAX_COUNT_PER_DEPTH
            );
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
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
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

/// Exact maximum consecutive carry-chain length over `width` columns.
///
/// Process columns least-significant first. A column has outgoing carry 1 iff
/// a_digit + b_digit + incoming_carry >= 10. The carry depth is the length of
/// the longest run of consecutive columns whose outgoing carry is 1. There are
/// exactly `width` outgoing carry bits (one per column); the final overflow
/// carry is the outgoing carry of the most significant column and is already
/// counted in that run.
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
    best
}

/// Integer-addition oracle: format a+b zero-padded to width+1 digits.
pub fn oracle_sum_string(a: u64, b: u64, width: usize) -> String {
    let s = a + b;
    let total = width + 1;
    let digits = format!("{}", s);
    if digits.len() > total {
        return digits;
    }
    let mut out = String::with_capacity(total);
    for _ in 0..(total - digits.len()) {
        out.push('0');
    }
    out.push_str(&digits);
    out
}

/// Enumerate all 2^width carry bit patterns with exact maximum run `k`.
///
/// Returns a non-empty vector for 0 <= k <= width. Uniform choice over the
/// returned patterns is uniform over all patterns with exact max-run k.
fn patterns_with_max_run(width: usize, k: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let n: u32 = 1u32 << width;
    for mask in 0..n {
        let mut bits = Vec::with_capacity(width);
        for col in 0..width {
            bits.push(((mask >> col) & 1) as u8);
        }
        if max_run(&bits) == k {
            out.push(bits);
        }
    }
    out
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

/// Choose a valid (a_digit, b_digit) pair at a column given incoming and
/// outgoing carry. Valid means a_digit + b_digit + carry_in >= 10 iff
/// carry_out == 1.
fn sample_digit_pair(rng: &mut StdRng, carry_in: u64, carry_out: u64) -> (u64, u64) {
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
    requested_split: Split,
    used_ids: &mut BTreeSet<String>,
) -> Result<CorpusRecord> {
    let patterns = patterns_with_max_run(width, target_depth);
    if patterns.is_empty() {
        bail!(
            "no carry patterns with exact max-run {} for width {}",
            target_depth,
            width
        );
    }
    for _ in 0..MAX_ROW_RETRIES {
        let bits = &patterns[rng.gen_range(0..patterns.len())];
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
        let actual = carry_depth(a, b, width);
        if actual != target_depth {
            continue;
        }
        let id = record_id(width, a, b);
        if split_for(width, a, b) != requested_split || used_ids.contains(&id) {
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
        width,
        target_depth
    )
}

/// Build observation, targets, mask.
///
/// Observation: zero-padded A+B= followed by width+1 '?' characters, then one
/// '.' of padding. Labels are placed only at '?' positions; other target
/// positions are '.'.
///
/// The query region starts at 2*width + 2 (after `A`, `+`, `B`, `=`) and spans
/// width+1 characters. The FIRST query position is the leading overflow digit.
fn make_observation(a: u64, b: u64, width: usize) -> (String, String, String) {
    let a_str = format!("{:0width$}", a, width = width);
    let b_str = format!("{:0width$}", b, width = width);
    let sum_str = oracle_sum_string(a, b, width);
    let query_len = width + 1;
    let core = format!("{}+{}={}", a_str, b_str, "?".repeat(query_len));
    let mut observation = core;
    observation.push('.');
    let query_start = 2 * width + 2;
    let mut targets = String::with_capacity(observation.len());
    let mut mask = String::with_capacity(observation.len());
    for i in 0..observation.len() {
        if i >= query_start && i < query_start + query_len {
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

/// Build the full corpus. Generates `count_per_depth` records for EACH of
/// Train, Validation, Test at each requested depth, using rejection by the
/// immutable `split_for` assignment and by duplicate ID.
pub fn build_corpus(config: &CorpusConfig) -> Result<Corpus> {
    config.validate().context("invalid corpus config")?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut used_ids: BTreeSet<String> = BTreeSet::new();
    let mut records: Vec<CorpusRecord> = Vec::new();
    let splits = [Split::Train, Split::Validation, Split::Test];
    for &depth in &config.depths {
        for &split in &splits {
            for _ in 0..config.count_per_depth {
                records.push(build_record(
                    &mut rng,
                    config.width,
                    depth,
                    split,
                    &mut used_ids,
                )?);
            }
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
    pub digit_histograms: BTreeMap<String, [usize; 10]>,
    pub unique_ids: bool,
    pub split_overlap: SplitOverlap,
    pub oracle_mask_ok: bool,
    pub constant_zero: BaselineMetrics,
    pub global_digit_prior: BaselineMetrics,
    pub position_prior: BaselineMetrics,
    pub by_split: BTreeMap<String, SplitMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub rows: usize,
    pub constant_zero: BaselineMetrics,
    pub global_digit_prior: BaselineMetrics,
    pub position_prior: BaselineMetrics,
}

/// Fit a global digit prior over train targets only.
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

/// Fit the most frequent digit at each position using training rows only.
fn fit_position_prior(corpus: &Corpus) -> BTreeMap<usize, u8> {
    let mut counts: BTreeMap<usize, [usize; 10]> = BTreeMap::new();
    for r in &corpus.records {
        if r.split != Split::Train {
            continue;
        }
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] == b'1' {
                if let Some(d) = ch.to_digit(10) {
                    let key = i;
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
        let query_positions: Vec<usize> = r
            .mask
            .char_indices()
            .filter(|(_, c)| *c == '1')
            .map(|(i, _)| i)
            .collect();
        if query_positions.is_empty() {
            continue;
        }
        let first_query = query_positions[0];
        let mut all_ok = true;
        for &pos in &query_positions {
            let truth = r.targets.as_bytes()[pos] as char;
            let truth = truth.to_digit(10).unwrap_or(0) as u8;
            let pred = predict(r, pos);
            let ok = pred == truth;
            if ok {
                digit_correct += 1;
            }
            digit_total += 1;
            let cell = cell_correct.entry((r.depth, pos)).or_insert((0, 0));
            cell.1 += 1;
            if ok {
                cell.0 += 1;
            }
            if !ok {
                all_ok = false;
            }
            if pos == first_query {
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
        complete_total += 1;
        if all_ok {
            complete_correct += 1;
        }
    }
    if digit_total == 0 {
        bail!("no query positions in corpus");
    }
    let digit_micro = digit_correct as f64 / digit_total as f64;
    let mut macro_sum = 0.0f64;
    let mut macro_n = 0usize;
    for (_, (c, t)) in cell_correct.iter() {
        if *t > 0 {
            macro_sum += *c as f64 / *t as f64;
            macro_n += 1;
        }
    }
    if macro_n == 0 {
        bail!("no (depth,position) cells");
    }
    let equal_depth_position_macro = macro_sum / macro_n as f64;
    if complete_total == 0 {
        bail!("no complete answers");
    }
    let complete_answer = complete_correct as f64 / complete_total as f64;
    if leading_total == 0 {
        bail!("no leading overflow positions");
    }
    let leading_overflow_digit = leading_correct as f64 / leading_total as f64;
    if remaining_total == 0 {
        bail!("no remaining positions");
    }
    let remaining_digit = remaining_correct as f64 / remaining_total as f64;
    Ok(BaselineMetrics {
        digit_micro,
        equal_depth_position_macro,
        complete_answer,
        leading_overflow_digit,
        remaining_digit,
    })
}

/// Validate a deserialized corpus before scoring. Returns an error (never
/// panics) on malformed input.
pub fn validate_corpus(corpus: &Corpus) -> Result<()> {
    if corpus.schema != ARITHMETIC_CORPUS_SCHEMA {
        bail!("unexpected schema: {}", corpus.schema);
    }
    if corpus.seed != corpus.config.seed {
        bail!(
            "seed mismatch: corpus {} vs config {}",
            corpus.seed,
            corpus.config.seed
        );
    }
    corpus
        .config
        .validate()
        .context("invalid config in corpus")?;
    if corpus.records.len() != 3 * corpus.config.depths.len() * corpus.config.count_per_depth {
        bail!("record count does not match configuration");
    }
    let width = corpus.config.width;
    let max_operand = 10u64.pow(width as u32);
    let mut ids: BTreeSet<String> = BTreeSet::new();
    let mut per_split_depth: BTreeMap<(String, usize), usize> = BTreeMap::new();
    for r in &corpus.records {
        if r.width != width {
            bail!("record width {} != config width {}", r.width, width);
        }
        if r.a >= max_operand || r.b >= max_operand {
            bail!("operand out of bounds: a={} b={} width={}", r.a, r.b, width);
        }
        if !corpus.config.depths.contains(&r.depth) {
            bail!("record depth {} exceeds width {}", r.depth, width);
        }
        if r.id != record_id(r.width, r.a, r.b) {
            bail!("record id {} does not match operands", r.id);
        }
        if r.split != split_for(r.width, r.a, r.b) {
            bail!("record split {:?} does not match split_for", r.split);
        }
        if !ids.insert(r.id.clone()) {
            bail!("duplicate record id {}", r.id);
        }
        if r.observation.len() != r.targets.len() || r.observation.len() != r.mask.len() {
            bail!("length mismatch in record {}", r.id);
        }
        if !r.observation.is_ascii() || !r.targets.is_ascii() || !r.mask.is_ascii() {
            bail!("non-ASCII content in record {}", r.id);
        }
        let (obs, targets, mask) = make_observation(r.a, r.b, r.width);
        if r.observation != obs || r.targets != targets || r.mask != mask {
            bail!("record {} has invalid observation, targets, or mask", r.id);
        }
        let expected = oracle_sum_string(r.a, r.b, r.width);
        let query_positions: Vec<usize> = r
            .mask
            .char_indices()
            .filter(|(_, c)| *c == '1')
            .map(|(i, _)| i)
            .collect();
        if query_positions.len() != r.width + 1 {
            bail!(
                "record {} has {} query positions, expected {}",
                r.id,
                query_positions.len(),
                r.width + 1
            );
        }
        for (qi, &pos) in query_positions.iter().enumerate() {
            let t = r.targets.as_bytes()[pos] as char;
            if t != expected.as_bytes()[qi] as char {
                bail!("record {} target mismatch at query {}", r.id, qi);
            }
        }
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] == b'0' && ch != '.' {
                bail!("record {} non-query target is not '.'", r.id);
            }
        }
        if carry_depth(r.a, r.b, r.width) != r.depth {
            bail!(
                "record {} depth {} does not match carry_depth",
                r.id,
                r.depth
            );
        }
        *per_split_depth
            .entry((r.split.as_str().to_string(), r.depth))
            .or_insert(0) += 1;
    }
    for &depth in &corpus.config.depths {
        for split in [Split::Train, Split::Validation, Split::Test] {
            let key = (split.as_str().to_string(), depth);
            let got = per_split_depth.get(&key).copied().unwrap_or(0);
            if got != corpus.config.count_per_depth {
                bail!(
                    "split={} depth={} has {} records, expected {}",
                    split.as_str(),
                    depth,
                    got,
                    corpus.config.count_per_depth
                );
            }
        }
    }
    Ok(())
}

/// Build a JSON-serializable audit.
pub fn audit_corpus(corpus: &Corpus) -> Result<Audit> {
    validate_corpus(corpus)?;
    let mut depth_counts: BTreeMap<usize, usize> = BTreeMap::new();
    let mut position_digit_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut digit_histograms: BTreeMap<String, [usize; 10]> = BTreeMap::new();
    let mut train_ids: BTreeSet<String> = BTreeSet::new();
    let mut val_ids: BTreeSet<String> = BTreeSet::new();
    let mut test_ids: BTreeSet<String> = BTreeSet::new();
    for r in &corpus.records {
        *depth_counts.entry(r.depth).or_insert(0) += 1;
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
        for (i, ch) in r.targets.chars().enumerate() {
            if r.mask.as_bytes()[i] != b'1' {
                continue;
            }
            *position_digit_counts.entry((r.depth, i)).or_insert(0) += 1;
            if let Some(d) = ch.to_digit(10) {
                let key = format!("{}:{}:{}", r.split.as_str(), r.depth, i);
                let entry = digit_histograms.entry(key).or_insert([0usize; 10]);
                entry[d as usize] += 1;
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
        let mut best_p = -1.0f64;
        for (d, &p) in global_prior.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = d;
            }
        }
        best as u8
    })?;
    let position_prior_metrics = evaluate(corpus, |_r, i| *position_prior.get(&i).unwrap_or(&0))?;

    let global_digit = (0..10)
        .max_by(|&a, &b| {
            global_prior[a]
                .partial_cmp(&global_prior[b])
                .unwrap()
                .then_with(|| b.cmp(&a))
        })
        .unwrap() as u8;
    let mut by_split = BTreeMap::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let mut panel = corpus.clone();
        panel.records.retain(|r| r.split == split);
        by_split.insert(
            split.as_str().to_string(),
            SplitMetrics {
                rows: panel.records.len(),
                constant_zero: evaluate(&panel, |_, _| 0)?,
                global_digit_prior: evaluate(&panel, |_, _| global_digit)?,
                position_prior: evaluate(&panel, |_, i| position_prior[&i])?,
            },
        );
    }

    Ok(Audit {
        schema: corpus.schema.clone(),
        seed: corpus.seed,
        config: corpus.config.clone(),
        total_records: corpus.records.len(),
        depth_counts,
        position_digit_counts: position_digit_counts_vec,
        digit_histograms,
        unique_ids: true,
        split_overlap,
        oracle_mask_ok: true,
        constant_zero,
        global_digit_prior: global_digit_prior_metrics,
        position_prior: position_prior_metrics,
        by_split,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent carry oracle via prefix-sum indicator:
    /// column j has outgoing carry iff a%10^j + b%10^j >= 10^j.
    fn oracle_carry_depth(a: u64, b: u64, width: usize) -> usize {
        let mut run = 0usize;
        let mut best = 0usize;
        for j in 1..=width {
            let p = 10u64.pow(j as u32);
            let carry = (a % p) + (b % p) >= p;
            if carry {
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

    #[test]
    fn exhaustive_width1_and_2_carry_oracle() {
        for width in 1..=2usize {
            let max = 10u64.pow(width as u32);
            for a in 0..max {
                for b in 0..max {
                    assert_eq!(
                        carry_depth(a, b, width),
                        oracle_carry_depth(a, b, width),
                        "a={} b={} width={}",
                        a,
                        b,
                        width
                    );
                }
            }
        }
    }

    #[test]
    fn known_carry_depth_examples() {
        assert_eq!(carry_depth(9, 1, 1), 1);
        assert_eq!(carry_depth(9999, 1, 4), 4);
        assert_eq!(carry_depth(0, 0, 4), 0);
        assert_eq!(carry_depth(1, 2, 4), 0);
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
                        count_per_depth: 2,
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
            count_per_depth: 4,
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
            count_per_depth: 8,
        };
        let corpus = build_corpus(&cfg).unwrap();
        for r in &corpus.records {
            assert_eq!(split_for(r.width, r.a, r.b), split_for(r.width, r.b, r.a));
            assert_eq!(r.split, split_for(r.width, r.a, r.b));
        }
    }

    #[test]
    fn stable_fnv_fixture() {
        // Known FNV1a-64 values for fixed byte strings.
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn unique_and_disjoint_panels() {
        let cfg = CorpusConfig {
            seed: 11,
            width: 4,
            depths: vec![0, 1, 2, 3, 4],
            count_per_depth: 16,
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
            count_per_depth: 4,
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
    fn overflow_is_first_query() {
        let cfg = CorpusConfig {
            seed: 17,
            width: 2,
            depths: vec![2],
            count_per_depth: 4,
        };
        let corpus = build_corpus(&cfg).unwrap();
        for r in &corpus.records {
            let first = r.mask.find('1').unwrap();
            let (obs, targets, mask) = make_observation(r.a, r.b, r.width);
            if r.observation != obs || r.targets != targets || r.mask != mask {
                panic!("record {} has invalid observation, targets, or mask", r.id);
            }
            let expected = oracle_sum_string(r.a, r.b, r.width);
            assert_eq!(
                r.targets.as_bytes()[first] as char,
                expected.chars().next().unwrap()
            );
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
    fn exhaustion_width1_count100() {
        // Width 1 has only 100 distinct canonical operand pairs; requesting
        // 100 per split per depth cannot be satisfied for all splits.
        let cfg = CorpusConfig {
            seed: 5,
            width: 1,
            depths: vec![0],
            count_per_depth: 100,
        };
        assert!(build_corpus(&cfg).is_err());
    }

    #[test]
    fn corrupt_corpora_rejected() {
        let cfg = CorpusConfig {
            seed: 3,
            width: 2,
            depths: vec![0, 1],
            count_per_depth: 2,
        };
        let good = build_corpus(&cfg).unwrap();
        validate_corpus(&good).unwrap();

        // Corrupt mask.
        let mut c = good.clone();
        c.records[0].mask = "0".repeat(c.records[0].mask.len());
        assert!(validate_corpus(&c).is_err());

        // Corrupt depth.
        let mut c = good.clone();
        c.records[0].depth = 99;
        assert!(validate_corpus(&c).is_err());

        // Corrupt id.
        let mut c = good.clone();
        c.records[0].id = "bogus".to_string();
        assert!(validate_corpus(&c).is_err());

        // Corrupt split.
        let mut c = good.clone();
        c.records[0].split = match c.records[0].split {
            Split::Train => Split::Test,
            _ => Split::Train,
        };
        assert!(validate_corpus(&c).is_err());

        // Corrupt target.
        let mut c = good.clone();
        let pos = c.records[0].mask.find('1').unwrap();
        let mut bytes = c.records[0].targets.clone().into_bytes();
        bytes[pos] = b'9';
        c.records[0].targets = String::from_utf8(bytes).unwrap();
        assert!(validate_corpus(&c).is_err());

        // Empty corpus.
        let mut c = good.clone();
        c.records.clear();
        assert!(validate_corpus(&c).is_err());
    }

    #[test]
    fn tiny_hand_computed_baseline_metrics() {
        // Build a tiny corpus and check constant-zero metrics by hand.
        let cfg = CorpusConfig {
            seed: 99,
            width: 1,
            depths: vec![0],
            count_per_depth: 1,
        };
        let corpus = build_corpus(&cfg).unwrap();
        let audit = audit_corpus(&corpus).unwrap();
        // Every record has width+1 = 2 query positions.
        let total_queries: usize = corpus.records.iter().map(|r| r.width + 1).sum();
        let zeros: usize = corpus
            .records
            .iter()
            .flat_map(|r| r.targets.chars().zip(r.mask.chars()))
            .filter(|(c, m)| *m == '1' && *c == '0')
            .count();
        let expected = zeros as f64 / total_queries as f64;
        assert!((audit.constant_zero.digit_micro - expected).abs() < 1e-12);
    }

    #[test]
    fn position_prior_uses_only_position() {
        let cfg = CorpusConfig {
            seed: 23,
            width: 3,
            depths: vec![0, 1, 2, 3],
            count_per_depth: 8,
        };
        let mut corpus = build_corpus(&cfg).unwrap();
        let prior = fit_position_prior(&corpus);
        let global = fit_global_digit_prior(&corpus);
        assert_eq!(prior.len(), cfg.width + 1);
        for r in &mut corpus.records {
            r.depth = 99;
            if r.split != Split::Train {
                r.targets = r
                    .targets
                    .chars()
                    .map(|c| if c.is_ascii_digit() { '9' } else { c })
                    .collect();
            }
        }
        assert_eq!(prior, fit_position_prior(&corpus));
        assert_eq!(global, fit_global_digit_prior(&corpus));
    }

    #[test]
    fn metrics_balance_depth_and_identify_overflow() {
        let config = CorpusConfig {
            seed: 0,
            width: 1,
            depths: vec![0, 1],
            count_per_depth: 1,
        };
        let mut records = Vec::new();
        for (a, b) in [(0, 0), (0, 0), (9, 2)] {
            let (observation, targets, mask) = make_observation(a, b, 1);
            records.push(CorpusRecord {
                id: record_id(1, a, b),
                a,
                b,
                width: 1,
                depth: carry_depth(a, b, 1),
                split: Split::Train,
                observation,
                targets,
                mask,
            });
        }
        // Deliberately unbalanced scoring fixture, independent of corpus sampling.
        let mut corpus = Corpus {
            schema: ARITHMETIC_CORPUS_SCHEMA.into(),
            seed: 0,
            config,
            records,
        };
        let m = evaluate(&corpus, |_, _| 0).unwrap();
        assert!((m.digit_micro - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(m.equal_depth_position_macro, 0.5);
        assert!((m.complete_answer - 2.0 / 3.0).abs() < 1e-12);
        corpus.records.truncate(1);
        let (observation, targets, mask) = make_observation(4, 5, 1);
        corpus.records[0].observation = observation;
        corpus.records[0].targets = targets;
        corpus.records[0].mask = mask;
        let m = evaluate(&corpus, |_, _| 0).unwrap();
        assert_eq!(m.leading_overflow_digit, 1.0);
        assert_eq!(m.remaining_digit, 0.0);
        assert_eq!(m.complete_answer, 0.0);
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
        validate_corpus(&corpus).unwrap();
    }
}
