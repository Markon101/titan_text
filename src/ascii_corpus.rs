//! Procedural ASCII Art Corpus and Structural Evaluation Metrics for Titan Text.
//!
//! Generates diverse, controllable, multiline ASCII compositions across
//! multiple structural families:
//! - Box (simple, double, nested, cornered)
//! - Checker (alternating motifs, blocks)
//! - Diamond (hollow and patterned diamonds)
//! - Maze (orthogonal grid corridors and junctions)
//! - Banner (framed motifs and decorative titles)
//! - Face (symmetric emoticons and stylized animal faces)
//! - Mountain (geometric peaks, ridges, horizons)
//! - Abstract (slanted diagonals, cellular glyph textures)
//!
//! Includes deterministic split generation (Train / Val / Test) and
//! empirical metric calculation (symmetry, collapse rates, n-gram entropy,
//! and nearest-neighbor edit distance against training items).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

/// Schema identifier for the Titan Text ASCII Corpus.
pub const ASCII_CORPUS_SCHEMA: &str = "titan_text.ascii_corpus.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AsciiFamily {
    Box,
    Checker,
    Diamond,
    Maze,
    Banner,
    Face,
    Mountain,
    Abstract,
}

impl AsciiFamily {
    pub fn all() -> &'static [Self] {
        &[
            Self::Box,
            Self::Checker,
            Self::Diamond,
            Self::Maze,
            Self::Banner,
            Self::Face,
            Self::Mountain,
            Self::Abstract,
        ]
    }

    pub fn prompt_tag(self) -> &'static str {
        match self {
            Self::Box => "<BOX>",
            Self::Checker => "<CHECKER>",
            Self::Diamond => "<DIAMOND>",
            Self::Maze => "<MAZE>",
            Self::Banner => "<BANNER>",
            Self::Face => "<FACE>",
            Self::Mountain => "<MOUNTAIN>",
            Self::Abstract => "<ABSTRACT>",
        }
    }

    pub fn parse_tag(s: &str) -> Option<Self> {
        match s.trim() {
            "<BOX>" | "box" => Some(Self::Box),
            "<CHECKER>" | "checker" => Some(Self::Checker),
            "<DIAMOND>" | "diamond" => Some(Self::Diamond),
            "<MAZE>" | "maze" => Some(Self::Maze),
            "<BANNER>" | "banner" => Some(Self::Banner),
            "<FACE>" | "face" => Some(Self::Face),
            "<MOUNTAIN>" | "mountain" => Some(Self::Mountain),
            "<ABSTRACT>" | "abstract" => Some(Self::Abstract),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsciiStage {
    StageA, // Rectangular structure, consistent line widths, spacing
    StageB, // Symmetry and repeated motifs
    StageC, // Category-conditioned recognizable objects
    StageD, // Abstract and complex compositions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsciiRecord {
    pub id: String,
    pub family: AsciiFamily,
    pub stage: AsciiStage,
    pub prompt: String,
    pub content: String,
    pub full_text: String,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsciiMetrics {
    pub total_chars: usize,
    pub valid_char_ratio: f32,
    pub line_count: usize,
    pub mean_line_width: f32,
    pub max_line_width: usize,
    pub non_whitespace_density: f32,
    pub row_diversity: f32,
    pub col_diversity: f32,
    pub repeated_char_collapse_rate: f32,
    pub repeated_line_collapse_rate: f32,
    pub newline_frequency: f32,
    pub horizontal_symmetry: f32,
    pub vertical_symmetry: f32,
    pub bigram_entropy: f32,
    pub nearest_edit_similarity: f32,
    pub exact_training_match: bool,
}

pub struct AsciiCorpus {
    pub train_records: Vec<AsciiRecord>,
    pub val_records: Vec<AsciiRecord>,
    pub test_records: Vec<AsciiRecord>,
}

impl AsciiCorpus {
    /// Generates a balanced, deterministic procedural ASCII corpus with disjoint splits
    pub fn new_balanced(count_per_family: usize, base_seed: u64) -> Self {
        let mut train_records = Vec::new();
        let mut val_records = Vec::new();
        let mut test_records = Vec::new();

        let mut rng = StdRng::seed_from_u64(base_seed);

        for &fam in AsciiFamily::all() {
            let mut family_records = Vec::new();
            let mut seen_hashes = HashSet::new();

            let mut attempts = 0;
            while family_records.len() < count_per_family && attempts < count_per_family * 50 {
                attempts += 1;
                let rec = generate_family_instance(fam, &mut rng);
                let key = rec.content.clone();
                if !seen_hashes.contains(&key) {
                    seen_hashes.insert(key);
                    family_records.push(rec);
                }
            }

            // Disjoint 80% train, 10% val, 10% test split per family
            let n = family_records.len();
            let n_train = (n * 8) / 10;
            let n_val = (n - n_train) / 2;

            for (i, mut rec) in family_records.into_iter().enumerate() {
                rec.id = format!("{}_{:03}", fam.prompt_tag().replace(['<', '>'], ""), i);
                if i < n_train {
                    train_records.push(rec);
                } else if i < n_train + n_val {
                    val_records.push(rec);
                } else {
                    test_records.push(rec);
                }
            }
        }

        Self {
            train_records,
            val_records,
            test_records,
        }
    }

    pub fn get_split(&self, split: Split) -> &[AsciiRecord] {
        match split {
            Split::Train => &self.train_records,
            Split::Validation => &self.val_records,
            Split::Test => &self.test_records,
        }
    }
}

/// Generates a single procedural instance for a given family
pub fn generate_family_instance(fam: AsciiFamily, rng: &mut StdRng) -> AsciiRecord {
    match fam {
        AsciiFamily::Box => generate_box(rng),
        AsciiFamily::Checker => generate_checker(rng),
        AsciiFamily::Diamond => generate_diamond(rng),
        AsciiFamily::Maze => generate_maze(rng),
        AsciiFamily::Banner => generate_banner(rng),
        AsciiFamily::Face => generate_face(rng),
        AsciiFamily::Mountain => generate_mountain(rng),
        AsciiFamily::Abstract => generate_abstract(rng),
    }
}

fn generate_box(rng: &mut StdRng) -> AsciiRecord {
    let style = rng.gen_range(0..4);
    let w = rng.gen_range(5..=10);
    let h = rng.gen_range(3..=6);
    let prompt = "<BOX>\n".to_string();

    let mut lines = Vec::new();
    match style {
        0 => {
            // Standard cornered box
            lines.push(format!("+{}+", "-".repeat(w - 2)));
            for _ in 0..(h - 2) {
                lines.push(format!("|{}|", " ".repeat(w - 2)));
            }
            lines.push(format!("+{}+", "-".repeat(w - 2)));
        }
        1 => {
            // Hash solid border
            lines.push("#".repeat(w));
            for _ in 0..(h - 2) {
                lines.push(format!("#{}#", " ".repeat(w - 2)));
            }
            lines.push("#".repeat(w));
        }
        2 => {
            // Nested or dotted border
            lines.push(format!("+{}+", "=".repeat(w - 2)));
            for _ in 0..(h - 2) {
                lines.push(format!(":{} :", ".".repeat(w - 3)));
            }
            lines.push(format!("+{}+", "=".repeat(w - 2)));
        }
        _ => {
            // Star border
            lines.push("*".repeat(w));
            for _ in 0..(h - 2) {
                lines.push(format!("*{}*", " ".repeat(w - 2)));
            }
            lines.push("*".repeat(w));
        }
    }

    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Box,
        stage: AsciiStage::StageA,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_checker(rng: &mut StdRng) -> AsciiRecord {
    let w = rng.gen_range(6..=10);
    let h = rng.gen_range(3..=5);
    let pair_choice = rng.gen_range(0..3);
    let (c1, c2) = match pair_choice {
        0 => ('#', ' '),
        1 => ('+', '.'),
        _ => ('*', ' '),
    };

    let prompt = "<CHECKER>\n".to_string();
    let mut lines = Vec::new();
    for row in 0..h {
        let mut line = String::with_capacity(w);
        for col in 0..w {
            let ch = if (row + col) % 2 == 0 { c1 } else { c2 };
            line.push(ch);
        }
        lines.push(line);
    }

    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Checker,
        stage: AsciiStage::StageB,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_diamond(rng: &mut StdRng) -> AsciiRecord {
    let radius = rng.gen_range(2..=4); // Height = 2*radius, width = 2*radius
    let prompt = "<DIAMOND>\n".to_string();
    let mut lines = Vec::new();

    // Top half
    for i in 0..radius {
        let outer = radius - 1 - i;
        let inner = 2 * i;
        lines.push(format!("{}/{}\\", " ".repeat(outer), " ".repeat(inner)));
    }
    // Bottom half
    for i in (0..radius).rev() {
        let outer = radius - 1 - i;
        let inner = 2 * i;
        lines.push(format!("{}\\{}/", " ".repeat(outer), " ".repeat(inner)));
    }

    let h = lines.len();
    let w = 2 * radius;
    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Diamond,
        stage: AsciiStage::StageB,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_maze(rng: &mut StdRng) -> AsciiRecord {
    let cols = rng.gen_range(2..=4);
    let prompt = "<MAZE>\n".to_string();
    let mut lines = Vec::new();

    // Top wall
    lines.push(format!("+{}+", "-+".repeat(cols - 1)));
    // Interior row with randomized openings
    for _ in 0..2 {
        let mut interior = String::from("|");
        for _ in 0..(cols - 1) {
            interior.push(if rng.gen_bool(0.6) { ' ' } else { '|' });
            interior.push(if rng.gen_bool(0.4) { '|' } else { ' ' });
        }
        interior.push('|');
        lines.push(interior);

        let mut junction = String::from("+");
        for _ in 0..(cols - 1) {
            junction.push(if rng.gen_bool(0.5) { '-' } else { ' ' });
            junction.push('+');
        }
        lines.push(junction);
    }
    // Bottom wall
    lines.push(format!("+{}+", "-+".repeat(cols - 1)));

    let h = lines.len();
    let w = lines[0].len();
    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Maze,
        stage: AsciiStage::StageC,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_banner(rng: &mut StdRng) -> AsciiRecord {
    let titles = ["TITAN", "NEURAL", "LATTICE", "MORPH", "SYNTAX"];
    let title = titles[rng.gen_range(0..titles.len())];
    let padding = rng.gen_range(1..=3);
    let border_char = if rng.gen_bool(0.5) { '=' } else { '*' };
    let prompt = "<BANNER>\n".to_string();

    let inner_text = format!("{}{}{}", " ".repeat(padding), title, " ".repeat(padding));
    let w = inner_text.len() + 2;
    let border = border_char.to_string().repeat(w);

    let mut lines = Vec::new();
    lines.push(border.clone());
    lines.push(format!("|{}|", inner_text));
    lines.push(border);

    let h = lines.len();
    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Banner,
        stage: AsciiStage::StageC,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_face(rng: &mut StdRng) -> AsciiRecord {
    let eyes = [("o", "o"), ("^", "^"), ("-", "-"), ("*", "*")];
    let mouths = [".", "_", "w", "v"];
    let (e1, e2) = eyes[rng.gen_range(0..eyes.len())];
    let m = mouths[rng.gen_range(0..mouths.len())];
    let prompt = "<FACE>\n".to_string();

    let ear_style = rng.gen_range(0..3);
    let mut lines = Vec::new();
    match ear_style {
        0 => {
            lines.push(r" /\_/\ ".to_string());
            lines.push(format!(r"( {e1}{m}{e2} )"));
            lines.push(r" > ^ < ".to_string());
        }
        1 => {
            lines.push(r" (..) ".to_string());
            lines.push(format!(r"( {e1}{m}{e2} )"));
            lines.push(r#" ("") "#.to_string());
        }
        _ => {
            lines.push(r" |\_/| ".to_string());
            lines.push(format!(r"[ {e1}_{e2} ]"));
            lines.push(r"  ===  ".to_string());
        }
    }

    let h = lines.len();
    let w = lines[0].len();
    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Face,
        stage: AsciiStage::StageC,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_mountain(rng: &mut StdRng) -> AsciiRecord {
    let prompt = "<MOUNTAIN>\n".to_string();
    let variant = rng.gen_range(0..3);
    let mut lines = Vec::new();

    match variant {
        0 => {
            lines.push("   /\\     ".to_string());
            lines.push("  /  \\ /\\ ".to_string());
            lines.push(" / /\\ \\/ \\".to_string());
            lines.push("/__________\\".to_string());
        }
        1 => {
            lines.push("    /\\    ".to_string());
            lines.push("   /  \\   ".to_string());
            lines.push("  / /\\ \\  ".to_string());
            lines.push(" /______\\ ".to_string());
        }
        _ => {
            lines.push("  /\\   /\\ ".to_string());
            lines.push(" /  \\ /  \\".to_string());
            lines.push("/____V____\\".to_string());
        }
    }

    let h = lines.len();
    let w = lines[0].len();
    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Mountain,
        stage: AsciiStage::StageC,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

fn generate_abstract(rng: &mut StdRng) -> AsciiRecord {
    let w = rng.gen_range(6..=10);
    let h = rng.gen_range(3..=5);
    let motif = rng.gen_range(0..3);
    let prompt = "<ABSTRACT>\n".to_string();

    let mut lines = Vec::new();
    match motif {
        0 => {
            // Slashes
            for r in 0..h {
                let s: String = (0..w).map(|c| if (r + c) % 2 == 0 { '/' } else { '\\' }).collect();
                lines.push(s);
            }
        }
        1 => {
            // Cellular blocks
            for r in 0..h {
                let s: String = (0..w).map(|c| if (r * 3 + c) % 4 == 0 { '#' } else { '.' }).collect();
                lines.push(s);
            }
        }
        _ => {
            // Zig-zag waves
            for r in 0..h {
                let s: String = (0..w).map(|c| if (r + c) % 3 == 0 { '~' } else { '-' }).collect();
                lines.push(s);
            }
        }
    }

    let content = lines.join("\n") + "\n";
    let full_text = format!("{}{}", prompt, content);
    AsciiRecord {
        id: String::new(),
        family: AsciiFamily::Abstract,
        stage: AsciiStage::StageD,
        prompt,
        content,
        full_text,
        width: w,
        height: h,
    }
}

// ============================================================================
// Objective Structural Evaluation Metrics Calculator
// ============================================================================

/// Evaluates empirical properties of generated ASCII string relative to training corpus
pub fn evaluate_ascii_metrics(text: &str, train_records: &[AsciiRecord]) -> AsciiMetrics {
    let total_chars = text.len();
    if total_chars == 0 {
        return AsciiMetrics {
            total_chars: 0,
            valid_char_ratio: 0.0,
            line_count: 0,
            mean_line_width: 0.0,
            max_line_width: 0,
            non_whitespace_density: 0.0,
            row_diversity: 0.0,
            col_diversity: 0.0,
            repeated_char_collapse_rate: 0.0,
            repeated_line_collapse_rate: 0.0,
            newline_frequency: 0.0,
            horizontal_symmetry: 0.0,
            vertical_symmetry: 0.0,
            bigram_entropy: 0.0,
            nearest_edit_similarity: 0.0,
            exact_training_match: false,
        };
    }

    // 1. Valid character ratio (printable ASCII + newline)
    let valid_count = text.chars().filter(|&c| c == '\n' || (c >= ' ' && c <= '~')).count();
    let valid_char_ratio = valid_count as f32 / total_chars as f32;

    // 2. Lines & dimensions
    let raw_lines: Vec<&str> = text.lines().collect();
    let line_count = raw_lines.len();
    let max_line_width = raw_lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let total_line_len: usize = raw_lines.iter().map(|l| l.len()).sum();
    let mean_line_width = if line_count > 0 {
        total_line_len as f32 / line_count as f32
    } else {
        0.0
    };

    // 3. Non-whitespace density
    let non_ws_count = text.chars().filter(|c| !c.is_whitespace()).count();
    let non_whitespace_density = non_ws_count as f32 / total_chars as f32;

    // 4. Repeated character collapse rate
    let mut repeated_char_count = 0usize;
    let chars: Vec<char> = text.chars().collect();
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] && !chars[i].is_whitespace() {
            repeated_char_count += 1;
        }
    }
    let repeated_char_collapse_rate = if chars.len() > 1 {
        repeated_char_count as f32 / (chars.len() - 1) as f32
    } else {
        0.0
    };

    // 5. Repeated line collapse rate
    let mut repeated_lines = 0usize;
    for i in 1..raw_lines.len() {
        if raw_lines[i] == raw_lines[i - 1] && !raw_lines[i].is_empty() {
            repeated_lines += 1;
        }
    }
    let repeated_line_collapse_rate = if raw_lines.len() > 1 {
        repeated_lines as f32 / (raw_lines.len() - 1) as f32
    } else {
        0.0
    };

    // 6. Row & column diversity
    let unique_lines: HashSet<&str> = raw_lines.iter().copied().collect();
    let row_diversity = if line_count > 0 {
        unique_lines.len() as f32 / line_count as f32
    } else {
        0.0
    };

    let col_diversity = if max_line_width > 0 && line_count > 0 {
        let mut col_unique_sum = 0usize;
        for c in 0..max_line_width {
            let mut col_chars = HashSet::new();
            for line in &raw_lines {
                if let Some(ch) = line.chars().nth(c) {
                    col_chars.insert(ch);
                }
            }
            col_unique_sum += col_chars.len();
        }
        (col_unique_sum as f32 / max_line_width as f32) / (line_count as f32).max(1.0)
    } else {
        0.0
    };

    // 7. Newline frequency
    let newline_count = text.chars().filter(|&c| c == '\n').count();
    let newline_frequency = newline_count as f32 / total_chars as f32;

    // 8. Horizontal symmetry (reflection about vertical axis)
    let mut h_sym_scores = Vec::new();
    for line in &raw_lines {
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        if len >= 2 {
            let mut matches = 0usize;
            let half = len / 2;
            for i in 0..half {
                let left = chars[i];
                let right = chars[len - 1 - i];
                // Check exact or mirrored pair
                if left == right
                    || (left == '(' && right == ')')
                    || (left == ')' && right == '(')
                    || (left == '[' && right == ']')
                    || (left == '/' && right == '\\')
                    || (left == '\\' && right == '/')
                    || (left == '<' && right == '>')
                {
                    matches += 1;
                }
            }
            h_sym_scores.push(matches as f32 / half as f32);
        }
    }
    let horizontal_symmetry = if !h_sym_scores.is_empty() {
        h_sym_scores.iter().sum::<f32>() / h_sym_scores.len() as f32
    } else {
        0.0
    };

    // 9. Vertical symmetry (reflection about horizontal axis)
    let mut v_sym = 0.0f32;
    if line_count >= 2 {
        let mut row_matches = 0usize;
        let half = line_count / 2;
        for i in 0..half {
            let top = raw_lines[i];
            let bot = raw_lines[line_count - 1 - i];
            if top == bot || top.len() == bot.len() {
                row_matches += 1;
            }
        }
        v_sym = row_matches as f32 / half as f32;
    }
    let vertical_symmetry = v_sym;

    // 10. Bigram entropy
    let mut bigram_counts: BTreeMap<(char, char), usize> = BTreeMap::new();
    let mut bigram_total = 0usize;
    for w in chars.windows(2) {
        *bigram_counts.entry((w[0], w[1])).or_insert(0) += 1;
        bigram_total += 1;
    }
    let mut bigram_entropy = 0.0f32;
    if bigram_total > 0 {
        for &count in bigram_counts.values() {
            let p = count as f32 / bigram_total as f32;
            if p > 0.0 {
                bigram_entropy -= p * p.ln();
            }
        }
    }

    // 11. Nearest training example edit similarity & exact match
    let mut min_norm_edit = 1.0f32;
    let mut exact_match = false;
    for train_rec in train_records {
        let train_text = &train_rec.content;
        if text.trim() == train_text.trim() {
            exact_match = true;
            min_norm_edit = 0.0;
            break;
        }
        let dist = levenshtein_distance(text.trim(), train_text.trim());
        let max_len = text.trim().len().max(train_text.trim().len()).max(1);
        let norm_edit = dist as f32 / max_len as f32;
        if norm_edit < min_norm_edit {
            min_norm_edit = norm_edit;
        }
    }
    let nearest_edit_similarity = (1.0 - min_norm_edit).clamp(0.0, 1.0);

    AsciiMetrics {
        total_chars,
        valid_char_ratio,
        line_count,
        mean_line_width,
        max_line_width,
        non_whitespace_density,
        row_diversity,
        col_diversity,
        repeated_char_collapse_rate,
        repeated_line_collapse_rate,
        newline_frequency,
        horizontal_symmetry,
        vertical_symmetry,
        bigram_entropy,
        nearest_edit_similarity,
        exact_training_match: exact_match,
    }
}

/// Simple Levenshtein distance on character slices
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_corpus_generation_and_splits() {
        let corpus = AsciiCorpus::new_balanced(20, 42);
        assert!(!corpus.train_records.is_empty());
        assert!(!corpus.val_records.is_empty());
        assert!(!corpus.test_records.is_empty());

        // Verify disjoint splits
        let train_contents: HashSet<&str> = corpus.train_records.iter().map(|r| r.content.as_str()).collect();
        for val_rec in &corpus.val_records {
            assert!(!train_contents.contains(val_rec.content.as_str()), "Val record leaked into train!");
        }
    }

    #[test]
    fn test_ascii_metrics_calculation() {
        let corpus = AsciiCorpus::new_balanced(10, 101);
        let sample = "+----+\n|    |\n|    |\n+----+\n";
        let metrics = evaluate_ascii_metrics(sample, &corpus.train_records);

        assert_eq!(metrics.valid_char_ratio, 1.0);
        assert_eq!(metrics.line_count, 4);
        assert_eq!(metrics.max_line_width, 6);
        assert!(metrics.horizontal_symmetry > 0.8);
        assert!(metrics.vertical_symmetry > 0.4);
    }
}
