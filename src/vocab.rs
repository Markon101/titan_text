use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct Vocab {
    chars: Vec<char>,
    char_to_id: HashMap<char, usize>,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: usize,
}

impl Vocab {
    /// Creates a standard printable ASCII + special tokens vocabulary
    pub fn new_ascii() -> Self {
        let mut chars = Vec::new();
        let mut char_to_id = HashMap::new();
        
        let pad_id = 0;
        let bos_id = 1;
        let eos_id = 2;
        let unk_id = 3;
        
        // Special placeholder tokens as unique dummy chars
        chars.push('\0'); // <pad>
        chars.push('\u{1}'); // <bos>
        chars.push('\u{2}'); // <eos>
        chars.push('\u{3}'); // <unk>
        
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
        }

        // Add standard printable ASCII (32 to 126: space, digits, letters, punctuation)
        for b in 32u8..=126u8 {
            let c = b as char;
            let id = chars.len();
            chars.push(c);
            char_to_id.insert(c, id);
        }

        Self {
            chars,
            char_to_id,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        }
    }

    /// Creates a dedicated ASCII art vocabulary containing:
    /// - Special control tokens (<pad>, <bos>, <eos>, <unk>)
    /// - Newline '\n' (id 4)
    /// - Printable ASCII 32..=126 (ids 5..=99)
    pub fn new_ascii_art() -> Self {
        let mut chars = Vec::new();
        let mut char_to_id = HashMap::new();

        let pad_id = 0;
        let bos_id = 1;
        let eos_id = 2;
        let unk_id = 3;

        chars.push('\0'); // <pad>
        chars.push('\u{1}'); // <bos>
        chars.push('\u{2}'); // <eos>
        chars.push('\u{3}'); // <unk>
        chars.push('\n'); // <newline> id = 4

        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
        }

        // Add standard printable ASCII (32 to 126: space, digits, letters, punctuation)
        for b in 32u8..=126u8 {
            let c = b as char;
            let id = chars.len();
            chars.push(c);
            char_to_id.insert(c, id);
        }

        Self {
            chars,
            char_to_id,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        }
    }

    /// Creates a synthetic Dyck-1 / algorithmic language vocab (parentheses, tokens, blanks)
    pub fn new_synthetic() -> Self {
        let chars = vec!['\0', '\u{1}', '\u{2}', '\u{3}', '(', ')', '[', ']', 'a', 'b', 'c', '0', '1', ' ', '#'];
        let mut char_to_id = HashMap::new();
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
        }
        Self {
            chars,
            char_to_id,
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            unk_id: 3,
        }
    }

    pub fn size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&self.unk_id))
            .collect()
    }

    #[allow(dead_code)]
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                if id == self.pad_id {
                    ' '
                } else if id == self.bos_id || id == self.eos_id || id == self.unk_id {
                    '·'
                } else if id < self.chars.len() {
                    self.chars[id]
                } else {
                    '?'
                }
            })
            .collect()
    }

    /// Decodes token ids directly into original character sequence, stopping at eos_id
    pub fn decode_raw(&self, ids: &[usize]) -> String {
        ids.iter()
            .take_while(|&&id| id != self.eos_id)
            .filter_map(|&id| {
                if id == self.pad_id || id == self.bos_id || id == self.unk_id {
                    None
                } else if id < self.chars.len() {
                    Some(self.chars[id])
                } else {
                    None
                }
            })
            .collect()
    }
}

#[allow(dead_code)]
pub struct TokenInterface {
    pub embedding: Embedding,
    pub projection: Linear,
    pub vocab_size: usize,
    pub channels: usize,
}

impl TokenInterface {
    pub fn new(vb: VarBuilder, vocab_size: usize, channels: usize) -> Result<Self> {
        let embedding = embedding(vocab_size, channels, vb.pp("embed"))?;
        let projection = linear(channels, vocab_size, vb.pp("proj"))?;
        Ok(Self {
            embedding,
            projection,
            vocab_size,
            channels,
        })
    }

    /// Embeds a batch of token ids into continuous field vectors [batch, seq_len, channels]
    pub fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.embedding.forward(tokens)?)
    }

    /// Projects cell states [batch, seq_len, channels] to token logits [batch, seq_len, vocab_size]
    pub fn logits(&self, states: &Tensor) -> Result<Tensor> {
        Ok(self.projection.forward(states)?)
    }

    /// Computes cross-entropy loss against target token ids [batch, seq_len]
    pub fn cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (b, l, v) = logits.dims3()?;
        let flat_logits = logits.reshape((b * l, v))?;
        let flat_targets = targets.reshape((b * l,))?;
        
        let log_sm = candle_nn::ops::log_softmax(&flat_logits, 1)?;
        let loss = candle_nn::loss::nll(&log_sm, &flat_targets)?;
        Ok(loss)
    }

    /// Computes token prediction accuracy
    pub fn accuracy(&self, logits: &Tensor, targets: &Tensor) -> Result<f32> {
        let preds = logits.argmax(candle_core::D::Minus1)?;
        let preds_vec: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let targets_vec: Vec<u32> = targets.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        
        if preds_vec.is_empty() {
            return Ok(0.0);
        }
        let matches = preds_vec.iter().zip(targets_vec.iter()).filter(|(p, t)| p == t).count();
        Ok(matches as f32 / preds_vec.len() as f32)
    }

    /// Computes masked cross-entropy loss against target token ids [batch, seq_len] with mask [batch, seq_len]
    #[allow(dead_code)]
    pub fn masked_cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, l, v) = logits.dims3()?;
        let flat_logits = logits.reshape((b * l, v))?;
        let flat_targets = targets.reshape((b * l,))?;
        let flat_mask = mask.reshape((b * l,))?;
        
        let log_sm = candle_nn::ops::log_softmax(&flat_logits, 1)?;
        let target_log_probs = log_sm.gather(&flat_targets.unsqueeze(1)?, 1)?.squeeze(1)?;
        let masked_nll = (target_log_probs.neg()? * flat_mask.clone())?;
        let mask_sum = flat_mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
        let loss = (masked_nll.sum_all()? / (mask_sum as f64))?;
        Ok(loss)
    }

    /// Computes masked token prediction accuracy
    #[allow(dead_code)]
    pub fn masked_accuracy(&self, logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<f32> {
        let preds = logits.argmax(candle_core::D::Minus1)?;
        let preds_vec: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let targets_vec: Vec<u32> = targets.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let mask_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        
        anyhow::ensure!(
            preds_vec.len() == mask_vec.len() && targets_vec.len() == mask_vec.len(),
            "Tensor length mismatch in masked accuracy calculation"
        );
        let mut matches = 0usize;
        let mut total_active = 0usize;
        for i in 0..preds_vec.len() {
            if mask_vec[i] > 0.5 {
                total_active += 1;
                if preds_vec[i] == targets_vec[i] {
                    matches += 1;
                }
            }
        }
        if total_active == 0 {
            return Ok(0.0);
        }
        Ok(matches as f32 / total_active as f32)
    }
}
