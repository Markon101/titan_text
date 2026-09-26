use crate::tasks::{TaskEngine, TaskKind};
use crate::vocab::Vocab;
use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct SequenceDataset {
    pub vocab: Vocab,
    pub task_name: String,
    train_phrases: Vec<String>,
    val_phrases: Vec<String>,
    task_engine: TaskEngine,
    algorithmic_kind: Option<TaskKind>,
    pub ascii_corpus: Option<crate::ascii_corpus::AsciiCorpus>,
}

impl SequenceDataset {
    pub fn new(task: &str) -> Self {
        if let Some(kind) = TaskKind::parse(task) {
            match kind {
                TaskKind::Text => {
                    let vocab = Vocab::new_ascii();
                    let train_phrases = vec![
                        "the morphogenic sequence field evolves in time. ".to_string(),
                        "local cellular interactions produce global syntax. ".to_string(),
                        "attractor dynamics stabilize linguistic states. ".to_string(),
                        "recurrent feedback drives sequence emergence. ".to_string(),
                    ];
                    let val_phrases = vec![
                        "hierarchical coupling guides sequence attractors. ".to_string(),
                        "nonlinear cellular networks learn local syntax. ".to_string(),
                    ];
                    Self {
                        vocab,
                        task_name: "text".to_string(),
                        train_phrases,
                        val_phrases,
                        task_engine: TaskEngine::new(),
                        algorithmic_kind: None,
                        ascii_corpus: None,
                    }
                }
                TaskKind::Dyck => {
                    let vocab = Vocab::new_synthetic();
                    Self {
                        vocab,
                        task_name: "dyck".to_string(),
                        train_phrases: vec![],
                        val_phrases: vec![],
                        task_engine: TaskEngine::new(),
                        algorithmic_kind: None,
                        ascii_corpus: None,
                    }
                }
                TaskKind::Ascii => {
                    let vocab = Vocab::new_ascii_art();
                    let corpus = crate::ascii_corpus::AsciiCorpus::new_balanced(50, 42);
                    Self {
                        vocab,
                        task_name: "ascii".to_string(),
                        train_phrases: vec![],
                        val_phrases: vec![],
                        task_engine: TaskEngine::new(),
                        algorithmic_kind: None,
                        ascii_corpus: Some(corpus),
                    }
                }
                other => {
                    let vocab = Vocab::new_ascii();
                    Self {
                        vocab,
                        task_name: other.name().to_string(),
                        train_phrases: vec![],
                        val_phrases: vec![],
                        task_engine: TaskEngine::new(),
                        algorithmic_kind: Some(other),
                        ascii_corpus: None,
                    }
                }
            }
        } else {
            // Default to text
            let vocab = Vocab::new_ascii();
            let train_phrases = vec![
                "the morphogenic sequence field evolves in time. ".to_string(),
                "local cellular interactions produce global syntax. ".to_string(),
                "attractor dynamics stabilize linguistic states. ".to_string(),
                "recurrent feedback drives sequence emergence. ".to_string(),
            ];
            let val_phrases = vec![
                "hierarchical coupling guides sequence attractors. ".to_string(),
                "nonlinear cellular networks learn local syntax. ".to_string(),
            ];
            Self {
                vocab,
                task_name: "text".to_string(),
                train_phrases,
                val_phrases,
                task_engine: TaskEngine::new(),
                algorithmic_kind: None,
                ascii_corpus: None,
            }
        }
    }

    /// Samples a batch from the training set
    pub fn sample_train_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.sample_batch_internal(batch_size, seq_len, false, device)
    }

    /// Samples a batch from the reserved held-out validation set
    pub fn sample_val_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        self.sample_batch_internal(batch_size, seq_len, true, device)
    }

    /// Samples a batch from the training set along with an optional loss mask
    #[allow(dead_code)]
    pub fn sample_train_batch_with_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        self.sample_batch_internal_with_mask(batch_size, seq_len, false, device)
    }

    /// Samples a batch from the validation set along with an optional loss mask
    #[allow(dead_code)]
    pub fn sample_val_batch_with_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        self.sample_batch_internal_with_mask(batch_size, seq_len, true, device)
    }

    fn sample_batch_internal(
        &self,
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let (inputs, targets, _) =
            self.sample_batch_internal_with_mask(batch_size, seq_len, is_val, device)?;
        Ok((inputs, targets))
    }

    fn sample_batch_internal_with_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        self.sample_batch_seeded_with_mask(batch_size, seq_len, is_val, 0, device)
    }

    /// Explicit stream seed for fresh reproducible algorithmic training batches.
    /// Legacy text/Dyck still repeat their historical finite corpus.
    pub fn sample_batch_seeded_with_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        is_val: bool,
        seed: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        anyhow::ensure!(
            batch_size > 0 && seq_len > 0,
            "dataset dimensions must be positive"
        );
        if let Some(kind) = self.algorithmic_kind {
            let batch = self
                .task_engine
                .generate_batch_seeded(kind, batch_size, seq_len, is_val, seed, device)?;
            return Ok((batch.inputs, batch.targets, Some(batch.loss_mask)));
        }

        if self.task_name == "ascii" {
            let corpus = self.ascii_corpus.as_ref().expect("ascii corpus must exist");
            let records = if is_val {
                corpus.get_split(crate::ascii_corpus::Split::Validation)
            } else {
                corpus.get_split(crate::ascii_corpus::Split::Train)
            };
            let mut inputs = Vec::with_capacity(batch_size * seq_len);
            let mut targets = Vec::with_capacity(batch_size * seq_len);
            let mut mask = Vec::with_capacity(batch_size * seq_len);

            for b in 0..batch_size {
                let rec = &records[(b + seed) % records.len()];
                let encoded = self.vocab.encode(&rec.full_text);
                for i in 0..seq_len {
                    if i < encoded.len() {
                        inputs.push(encoded[i] as u32);
                        if i + 1 < encoded.len() {
                            targets.push(encoded[i + 1] as u32);
                            mask.push(1.0f32);
                        } else if i + 1 == encoded.len() {
                            targets.push(self.vocab.eos_id as u32);
                            mask.push(1.0f32);
                        } else {
                            targets.push(self.vocab.pad_id as u32);
                            mask.push(0.0f32);
                        }
                    } else {
                        inputs.push(self.vocab.pad_id as u32);
                        targets.push(self.vocab.pad_id as u32);
                        mask.push(0.0f32);
                    }
                }
            }
            let input_tensor = Tensor::from_slice(&inputs, (batch_size, seq_len), device)?;
            let target_tensor = Tensor::from_slice(&targets, (batch_size, seq_len), device)?;
            let mask_tensor = Tensor::from_slice(&mask, (batch_size, seq_len), device)?;
            return Ok((input_tensor, target_tensor, Some(mask_tensor)));
        }

        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);

        if self.task_name == "dyck" {
            // Task B: Algorithmic Dyck-1 nested sequence
            // Train uses depths 1..=3; Held-out validation uses depths 4..=6
            for b in 0..batch_size {
                let mut seq = Vec::with_capacity(seq_len);
                let mut depth: i32 = 0;
                let max_depth: i32 = if is_val { 6 } else { 3 };
                let min_depth: i32 = if is_val { 2 } else { 0 };

                for i in 0..seq_len {
                    let c = if depth <= min_depth {
                        depth += 1;
                        '('
                    } else if depth >= max_depth || i == seq_len - 1 {
                        depth -= 1;
                        ')'
                    } else if (b + i) % 2 == 0 {
                        depth += 1;
                        '('
                    } else {
                        depth -= 1;
                        ')'
                    };
                    seq.push(c);
                }

                let ids = self.vocab.encode(&seq.into_iter().collect::<String>());
                for i in 0..seq_len {
                    inputs.push(ids[i] as u32);
                    targets.push(ids[(i + 1) % seq_len] as u32);
                }
            }
        } else {
            // Task A: Natural text next-token prediction with disjoint train / val phrases
            let phrases = if is_val {
                &self.val_phrases
            } else {
                &self.train_phrases
            };
            for b in 0..batch_size {
                let phrase = &phrases[b % phrases.len()];
                let phrase_ids = self.vocab.encode(phrase);
                let p_len = phrase_ids.len();
                let offset = (b * 7) % p_len;

                for i in 0..seq_len {
                    let in_id = phrase_ids[(i + offset) % p_len];
                    let out_id = phrase_ids[(i + offset + 1) % p_len];
                    inputs.push(in_id as u32);
                    targets.push(out_id as u32);
                }
            }
        }

        let input_tensor = Tensor::from_slice(&inputs, (batch_size, seq_len), device)?;
        let target_tensor = Tensor::from_slice(&targets, (batch_size, seq_len), device)?;
        Ok((input_tensor, target_tensor, None))
    }
}
