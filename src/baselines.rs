use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, linear, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Fresh-model initialization, independent of Candle's process-local RNG and
/// HashMap iteration order. Never call after loading an existing checkpoint.
/// Policy v1: normal(0,1) embeddings, zero biases, fan-in uniform weights.
pub fn initialize_seeded(varmap: &VarMap, seed: u64) -> Result<()> {
    let variables = varmap.data().lock().unwrap();
    let mut names: Vec<_> = variables.keys().collect();
    names.sort();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for name in names {
        let var = &variables[name];
        let is_embedding = name.contains("embed");
        let bound = (3.0 / var.dims().last().copied().unwrap_or(1) as f32).sqrt();
        let values: Vec<f32> = (0..var.elem_count())
            .map(|_| {
                if name.ends_with("bias") {
                    0.0
                } else if is_embedding {
                    StandardNormal.sample(&mut rng)
                } else {
                    rng.gen_range(-bound..bound)
                }
            })
            .collect();
        var.set(&Tensor::from_vec(values, var.shape(), var.device())?.to_dtype(var.dtype())?)?;
    }
    Ok(())
}

/// Factory shared by the legacy trainer and the explicit-batch research harness.
pub fn build_baseline(
    model_kind: &str,
    vb: VarBuilder,
    vocab_size: usize,
    channels: usize,
    depth: usize,
) -> Result<Box<dyn SequenceModel>> {
    anyhow::ensure!(
        vocab_size > 0 && channels > 0 && depth > 0,
        "baseline dimensions and depth must be positive"
    );
    Ok(match model_kind {
        "transformer" => Box::new(TransformerBaseline::new(
            vb,
            vocab_size,
            channels,
            channels * 2,
        )?),
        "gru" => Box::new(GruBaseline::new(vb, vocab_size, channels)?),
        "rnn" | "simple-recurrent" => {
            Box::new(SimpleRecurrentBaseline::new(vb, vocab_size, channels)?)
        }
        "untied" | "pointwise" => Box::new(UntiedFeedforwardBaseline::new(
            vb,
            vocab_size,
            channels,
            depth,
            channels * 2,
        )?),
        "tied-local" => Box::new(LocalDepthBaseline::new(
            vb, vocab_size, channels, depth, true,
        )?),
        "untied-local" => Box::new(LocalDepthBaseline::new(
            vb, vocab_size, channels, depth, false,
        )?),
        _ => anyhow::bail!("Unknown baseline model kind: {}", model_kind),
    })
}

/// Common trait for all sequence models (NCA, Transformer, GRU, Simple Recurrent).
pub trait SequenceModel {
    /// Forward pass taking token IDs [batch, seq_len], producing logits [batch, seq_len, vocab_size]
    fn forward(&self, tokens: &Tensor) -> Result<Tensor>;

    /// Forward pass taking a single token ID [batch, 1] and previous hidden state, returning (logits, next_hidden)
    #[allow(dead_code)]
    fn step(&self, token: &Tensor, hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)>;

    /// Parameter breakdown by subsystem: (subsystem_name, parameter_count)
    fn parameter_breakdown(&self) -> Vec<(String, usize)>;

    /// Total trainable parameters
    fn total_parameters(&self) -> usize {
        self.parameter_breakdown()
            .iter()
            .map(|(_, count)| *count)
            .sum()
    }
}

/// Brutally simple causal Transformer baseline.
/// Architecture:
/// - Embedding [V, C]
/// - Causal Self-Attention with projection [C -> C]
/// - Layer MLP with GELU [C -> H -> C]
/// - Readout projection [C -> V]
pub struct TransformerBaseline {
    pub embed: Embedding,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub mlp1: Linear,
    pub mlp2: Linear,
    pub readout: Linear,
    pub channels: usize,
    pub vocab_size: usize,
}

impl TransformerBaseline {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        channels: usize,
        mlp_hidden: usize,
    ) -> Result<Self> {
        let embed = embedding(vocab_size, channels, vb.pp("embed"))?;
        let q_proj = linear(channels, channels, vb.pp("q_proj"))?;
        let k_proj = linear(channels, channels, vb.pp("k_proj"))?;
        let v_proj = linear(channels, channels, vb.pp("v_proj"))?;
        let out_proj = linear(channels, channels, vb.pp("out_proj"))?;
        let mlp1 = linear(channels, mlp_hidden, vb.pp("mlp1"))?;
        let mlp2 = linear(mlp_hidden, channels, vb.pp("mlp2"))?;
        let readout = linear(channels, vocab_size, vb.pp("readout"))?;

        Ok(Self {
            embed,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            mlp1,
            mlp2,
            readout,
            channels,
            vocab_size,
        })
    }

    /// Forward pass through causal self-attention + MLP
    pub fn forward_internal(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, l) = tokens.dims2()?;
        let embedding = self.embed.forward(tokens)?; // [B, L, C]
                                                     // Fixed positions provide order and extend to unseen lengths.
        let positions: Vec<f32> = (0..l)
            .flat_map(|position| {
                (0..self.channels).map(move |channel| {
                    let angle = position as f64
                        / 10000_f64.powf((2 * (channel / 2)) as f64 / self.channels as f64);
                    if channel % 2 == 0 {
                        angle.sin() as f32
                    } else {
                        angle.cos() as f32
                    }
                })
            })
            .collect();
        let position = Tensor::from_vec(positions, (1, l, self.channels), tokens.device())?;
        let x = embedding.broadcast_add(&position)?;

        // 1. Causal self-attention
        let q = self.q_proj.forward(&x)?; // [B, L, C]
        let k = self.k_proj.forward(&x)?; // [B, L, C]
        let v = self.v_proj.forward(&x)?; // [B, L, C]

        let scale = 1.0 / (self.channels as f64).sqrt();
        let q_scaled = (q * scale)?;
        let scores = q_scaled.matmul(&k.transpose(1, 2)?)?; // [B, L, L]

        // Causal lower-triangular mask
        let mut mask_vec = vec![0.0f32; l * l];
        for i in 0..l {
            for j in 0..l {
                if j > i {
                    mask_vec[i * l + j] = -1e9f32;
                }
            }
        }
        let mask =
            Tensor::from_slice(&mask_vec, (1, l, l), tokens.device())?.broadcast_as((b, l, l))?;
        let masked_scores = (&scores + &mask)?;
        let attn_weights = candle_nn::ops::softmax(&masked_scores, 2)?;
        let attn_out = attn_weights.matmul(&v)?; // [B, L, C]
        let attn_proj = self.out_proj.forward(&attn_out)?;

        // Residual 1
        let x1 = (&x + &attn_proj)?;

        // 2. Feedforward MLP
        let h = self.mlp1.forward(&x1)?;
        let h_act = candle_nn::Activation::Gelu.forward(&h)?;
        let mlp_out = self.mlp2.forward(&h_act)?;

        // Residual 2
        let x2 = (&x1 + &mlp_out)?;

        // Readout
        let logits = self.readout.forward(&x2)?;
        Ok(logits)
    }
}

impl SequenceModel for TransformerBaseline {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.forward_internal(tokens)
    }

    fn step(&self, token: &Tensor, hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        // State is the token prefix; recomputation preserves full-forward
        // semantics. This correctness path is not a KV cache.
        let prefix = match hidden {
            Some(previous) => Tensor::cat(&[previous, token], 1)?,
            None => token.clone(),
        };
        let logits = self.forward_internal(&prefix)?;
        let last = logits.narrow(1, prefix.dim(1)? - 1, 1)?;
        Ok((last, prefix))
    }

    fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let embed_params = self.vocab_size * self.channels;
        let attn_params = 4 * (self.channels * self.channels + self.channels);
        let mlp1_params =
            self.mlp1.weight().elem_count() + self.mlp1.bias().map_or(0, |b| b.elem_count());
        let mlp2_params =
            self.mlp2.weight().elem_count() + self.mlp2.bias().map_or(0, |b| b.elem_count());
        let readout_params = self.channels * self.vocab_size + self.vocab_size;

        vec![
            ("token_embedding".to_string(), embed_params),
            ("causal_attention".to_string(), attn_params),
            ("mlp_feedforward".to_string(), mlp1_params + mlp2_params),
            ("readout_projection".to_string(), readout_params),
        ]
    }
}

/// Brutally simple GRU sequence baseline.
/// Standard GRU equations:
/// r_t = sigmoid(W_xr x_t + W_hr h_{t-1} + b_r)
/// z_t = sigmoid(W_xz x_t + W_hz h_{t-1} + b_z)
/// n_t = tanh(W_xn x_t + b_xn + r_t * (W_hn h_{t-1} + b_hn))
/// h_t = (1 - z_t) * h_{t-1} + z_t * n_t
pub struct GruBaseline {
    pub embed: Embedding,
    // Combined gates linear layers for efficiency:
    // W_x maps x [C] -> 3 * C (r, z, n)
    pub w_x: Linear,
    // W_h maps h [C] -> 3 * C (r, z, n)
    pub w_h: Linear,
    pub readout: Linear,
    pub channels: usize,
    pub vocab_size: usize,
}

impl GruBaseline {
    pub fn new(vb: VarBuilder, vocab_size: usize, channels: usize) -> Result<Self> {
        let embed = embedding(vocab_size, channels, vb.pp("embed"))?;
        let w_x = linear(channels, 3 * channels, vb.pp("w_x"))?;
        let w_h = linear(channels, 3 * channels, vb.pp("w_h"))?;
        let readout = linear(channels, vocab_size, vb.pp("readout"))?;

        Ok(Self {
            embed,
            w_x,
            w_h,
            readout,
            channels,
            vocab_size,
        })
    }

    /// Single GRU step taking token vector x_t [B, C] and hidden state h_{t-1} [B, C]
    pub fn step_cell(&self, x_t: &Tensor, h_prev: &Tensor) -> Result<Tensor> {
        let c = self.channels;
        let gates_x = self.w_x.forward(x_t)?; // [B, 3*C]
        let gates_h = self.w_h.forward(h_prev)?; // [B, 3*C]

        let r_x = gates_x.narrow(1, 0, c)?;
        let z_x = gates_x.narrow(1, c, c)?;
        let n_x = gates_x.narrow(1, 2 * c, c)?;

        let r_h = gates_h.narrow(1, 0, c)?;
        let z_h = gates_h.narrow(1, c, c)?;
        let n_h = gates_h.narrow(1, 2 * c, c)?;

        // Reset gate: r = sigmoid(r_x + r_h)
        let r = candle_nn::ops::sigmoid(&(&r_x + &r_h)?)?;

        // Update gate: z = sigmoid(z_x + z_h)
        let z = candle_nn::ops::sigmoid(&(&z_x + &z_h)?)?;

        // Candidate state: n = tanh(n_x + r * n_h)
        let r_nh = r.mul(&n_h)?;
        let n = (&n_x + &r_nh)?.tanh()?;

        // Next hidden state: h_t = (1 - z) * h_prev + z * n
        let one_minus_z = (1.0 - &z)?;
        let keep_part = one_minus_z.mul(h_prev)?;
        let new_part = z.mul(&n)?;
        let h_next = (&keep_part + &new_part)?;

        Ok(h_next)
    }
}

impl SequenceModel for GruBaseline {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, l) = tokens.dims2()?;
        let x = self.embed.forward(tokens)?; // [B, L, C]
        let mut h = Tensor::zeros((b, self.channels), DType::F32, tokens.device())?;
        let mut step_outputs = Vec::with_capacity(l);

        for t in 0..l {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
            h = self.step_cell(&x_t, &h)?;
            let logits_t = self.readout.forward(&h)?; // [B, V]
            step_outputs.push(logits_t.unsqueeze(1)?); // [B, 1, V]
        }

        Ok(Tensor::cat(&step_outputs.iter().collect::<Vec<_>>(), 1)?)
    }

    fn step(&self, token: &Tensor, hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, _) = token.dims2()?;
        let x_t = self.embed.forward(token)?.squeeze(1)?;
        let zero_h;
        let h_prev = match hidden {
            Some(h) => h,
            None => {
                zero_h = Tensor::zeros((b, self.channels), DType::F32, token.device())?;
                &zero_h
            }
        };
        let h_next = self.step_cell(&x_t, h_prev)?;
        let logits = self.readout.forward(&h_next)?.unsqueeze(1)?;
        Ok((logits, h_next))
    }

    fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let embed_params = self.vocab_size * self.channels;
        let wx_params =
            self.w_x.weight().elem_count() + self.w_x.bias().map_or(0, |b| b.elem_count());
        let wh_params =
            self.w_h.weight().elem_count() + self.w_h.bias().map_or(0, |b| b.elem_count());
        let readout_params = self.channels * self.vocab_size + self.vocab_size;

        vec![
            ("token_embedding".to_string(), embed_params),
            ("gru_recurrent_core".to_string(), wx_params + wh_params),
            ("readout_projection".to_string(), readout_params),
        ]
    }
}

/// Simple Recurrent Model baseline (Elman RNN):
/// h_t = tanh(W_x x_t + W_h h_{t-1} + b)
pub struct SimpleRecurrentBaseline {
    pub embed: Embedding,
    pub w_x: Linear,
    pub w_h: Linear,
    pub readout: Linear,
    pub channels: usize,
    pub vocab_size: usize,
}

impl SimpleRecurrentBaseline {
    pub fn new(vb: VarBuilder, vocab_size: usize, channels: usize) -> Result<Self> {
        let embed = embedding(vocab_size, channels, vb.pp("embed"))?;
        let w_x = linear(channels, channels, vb.pp("w_x"))?;
        let w_h = linear(channels, channels, vb.pp("w_h"))?;
        let readout = linear(channels, vocab_size, vb.pp("readout"))?;

        Ok(Self {
            embed,
            w_x,
            w_h,
            readout,
            channels,
            vocab_size,
        })
    }

    pub fn step_cell(&self, x_t: &Tensor, h_prev: &Tensor) -> Result<Tensor> {
        let in_proj = self.w_x.forward(x_t)?;
        let rec_proj = self.w_h.forward(h_prev)?;
        let h_next = (&in_proj + &rec_proj)?.tanh()?;
        Ok(h_next)
    }
}

impl SequenceModel for SimpleRecurrentBaseline {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, l) = tokens.dims2()?;
        let x = self.embed.forward(tokens)?;
        let mut h = Tensor::zeros((b, self.channels), DType::F32, tokens.device())?;
        let mut step_outputs = Vec::with_capacity(l);

        for t in 0..l {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
            h = self.step_cell(&x_t, &h)?;
            let logits_t = self.readout.forward(&h)?;
            step_outputs.push(logits_t.unsqueeze(1)?);
        }

        Ok(Tensor::cat(&step_outputs.iter().collect::<Vec<_>>(), 1)?)
    }

    fn step(&self, token: &Tensor, hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, _) = token.dims2()?;
        let x_t = self.embed.forward(token)?.squeeze(1)?;
        let zero_h;
        let h_prev = match hidden {
            Some(h) => h,
            None => {
                zero_h = Tensor::zeros((b, self.channels), DType::F32, token.device())?;
                &zero_h
            }
        };
        let h_next = self.step_cell(&x_t, h_prev)?;
        let logits = self.readout.forward(&h_next)?.unsqueeze(1)?;
        Ok((logits, h_next))
    }

    fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let embed_params = self.vocab_size * self.channels;
        let wx_params =
            self.w_x.weight().elem_count() + self.w_x.bias().map_or(0, |b| b.elem_count());
        let wh_params =
            self.w_h.weight().elem_count() + self.w_h.bias().map_or(0, |b| b.elem_count());
        let readout_params = self.channels * self.vocab_size + self.vocab_size;

        vec![
            ("token_embedding".to_string(), embed_params),
            ("elman_recurrent_core".to_string(), wx_params + wh_params),
            ("readout_projection".to_string(), readout_params),
        ]
    }
}

/// Untied feedforward baseline (Requirement 6):
/// Applies D independent (non-weight-tied) MLP transformation layers to test
/// whether recurrence (weight tying across depth) or merely depth / extra parameters drive performance.
pub struct UntiedFeedforwardBaseline {
    pub embed: Embedding,
    pub layers: Vec<(Linear, Linear)>,
    pub readout: Linear,
    pub channels: usize,
    #[allow(dead_code)]
    pub depth: usize,
    pub vocab_size: usize,
}

impl UntiedFeedforwardBaseline {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        channels: usize,
        depth: usize,
        mlp_hidden: usize,
    ) -> Result<Self> {
        let embed = embedding(vocab_size, channels, vb.pp("embed"))?;
        let mut layers = Vec::with_capacity(depth);
        for d in 0..depth {
            let p_d = vb.pp(format!("layer_{}", d));
            let l1 = linear(channels, mlp_hidden, p_d.pp("l1"))?;
            let l2 = linear(mlp_hidden, channels, p_d.pp("l2"))?;
            layers.push((l1, l2));
        }
        let readout = linear(channels, vocab_size, vb.pp("readout"))?;
        Ok(Self {
            embed,
            layers,
            readout,
            channels,
            depth,
            vocab_size,
        })
    }
}

impl SequenceModel for UntiedFeedforwardBaseline {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut x = self.embed.forward(tokens)?;
        for (l1, l2) in &self.layers {
            let h = candle_nn::Activation::Gelu.forward(&l1.forward(&x)?)?;
            let delta = l2.forward(&h)?;
            x = (&x + &delta)?;
        }
        self.readout.forward(&x).map_err(Into::into)
    }

    fn step(&self, token: &Tensor, _hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        Ok((self.forward(token)?, self.embed.forward(token)?))
    }

    fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let embed_params = self.vocab_size * self.channels;
        let mut layer_params = 0usize;
        for (l1, l2) in &self.layers {
            let (in1, out1) = (self.channels, l1.weight().dims()[0]);
            let (in2, out2) = (l2.weight().dims()[1], l2.weight().dims()[0]);
            layer_params += in1 * out1 + out1 + in2 * out2 + out2;
        }
        let readout_params = self.channels * self.vocab_size + self.vocab_size;
        vec![
            ("token_embedding".to_string(), embed_params),
            ("untied_layers".to_string(), layer_params),
            ("readout_projection".to_string(), readout_params),
        ]
    }
}

/// Minimal spatial competitor: x <- x + 0.1*tanh(W2*tanh(W1*[left,x,right])).
/// Zero boundary, no adaptive gates, diffusion, extra pathways or trainable positions.
/// Tied and untied variants use identical equations and receptive fields.
pub struct LocalDepthBaseline {
    pub embed: Embedding,
    pub layers: Vec<(Linear, Linear)>,
    pub readout: Linear,
    pub depth: usize,
    pub tied: bool,
}

impl LocalDepthBaseline {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        channels: usize,
        depth: usize,
        tied: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            depth > 0 && channels > 0,
            "positive local depth and channels required"
        );
        let embed = embedding(vocab_size, channels, vb.pp("embed"))?;
        let mut layers = Vec::new();
        for d in 0..if tied { 1 } else { depth } {
            let p = vb.pp(format!("layer_{}", d));
            layers.push((
                linear(channels * 3, channels * 2, p.pp("l1"))?,
                linear(channels * 2, channels, p.pp("l2"))?,
            ));
        }
        let readout = linear(channels, vocab_size, vb.pp("readout"))?;
        Ok(Self {
            embed,
            layers,
            readout,
            depth,
            tied,
        })
    }

    pub fn state_at(&self, tokens: &Tensor, steps: usize) -> Result<Tensor> {
        anyhow::ensure!(
            self.tied || steps <= self.depth,
            "untied depth cannot exceed trained layer count"
        );
        let mut x = self.embed.forward(tokens)?;
        let (b, l, c) = x.dims3()?;
        anyhow::ensure!(l > 0, "sequence cannot be empty");
        let zero = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        for d in 0..steps {
            let (left, right) = if l == 1 {
                (zero.clone(), zero.clone())
            } else {
                (
                    Tensor::cat(&[&zero, &x.narrow(1, 0, l - 1)?], 1)?,
                    Tensor::cat(&[&x.narrow(1, 1, l - 1)?, &zero], 1)?,
                )
            };
            let perception = Tensor::cat(&[&left, &x, &right], 2)?;
            let (l1, l2) = &self.layers[if self.tied { 0 } else { d }];
            let delta = (l2.forward(&l1.forward(&perception)?.tanh()?)?.tanh()? * 0.1)?;
            x = (&x + delta)?;
        }
        Ok(x)
    }
}

impl SequenceModel for LocalDepthBaseline {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.readout.forward(&self.state_at(tokens, self.depth)?)?)
    }
    fn step(&self, _token: &Tensor, _hidden: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        anyhow::bail!(
            "local spatial baseline has no causal token-stream step; use state_at for latent ticks"
        )
    }
    fn parameter_breakdown(&self) -> Vec<(String, usize)> {
        let core = self
            .layers
            .iter()
            .map(|(a, b)| {
                a.weight().elem_count()
                    + a.bias().map_or(0, |v| v.elem_count())
                    + b.weight().elem_count()
                    + b.bias().map_or(0, |v| v.elem_count())
            })
            .sum();
        vec![
            (
                "token_embedding".into(),
                self.embed.embeddings().elem_count(),
            ),
            (
                if self.tied {
                    "tied_local_layers"
                } else {
                    "untied_local_layers"
                }
                .into(),
                core,
            ),
            (
                "readout_projection".into(),
                self.readout.weight().elem_count()
                    + self.readout.bias().map_or(0, |v| v.elem_count()),
            ),
        ]
    }
}

/// Result of training a baseline model (Requirement 5)
#[allow(dead_code)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BaselineTrainResult {
    pub model_kind: String,
    pub total_parameters: usize,
    pub final_train_loss: f32,
    pub final_train_acc: f32,
    pub final_val_loss: f32,
    pub final_val_acc: f32,
    pub epochs: usize,
}

/// Trains a baseline sequence model using AdamW and masked loss (Requirement 5)
#[allow(dead_code)]
pub fn train_baseline_model(
    model_kind: &str,
    task: &str,
    epochs: usize,
    lr: f64,
    weight_decay: f64,
    batch_size: usize,
    seq_len: usize,
    channels: usize,
    depth: usize,
    seed: u64,
    device: &Device,
) -> Result<(Box<dyn SequenceModel>, BaselineTrainResult)> {
    let vocab_size = crate::tasks::TaskEngine::new().vocab.size();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let model = build_baseline(model_kind, vb, vocab_size, channels, depth)?;
    initialize_seeded(&varmap, seed)?;
    anyhow::ensure!(
        epochs > 0 && batch_size > 0 && seq_len > 0,
        "training dimensions and steps must be positive"
    );
    anyhow::ensure!(
        lr.is_finite() && lr > 0.0 && weight_decay.is_finite() && weight_decay >= 0.0,
        "invalid optimizer configuration"
    );
    let kind = crate::tasks::TaskKind::parse(task)
        .ok_or_else(|| anyhow::anyhow!("unknown task: {}", task))?;
    let engine = crate::tasks::TaskEngine::new();

    let total_params = model.total_parameters();
    let opt_params = ParamsAdamW {
        lr,
        weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), opt_params)?;

    let mut last_train_loss = 0.0f32;
    let mut last_train_acc = 0.0f32;

    for step in 0..epochs {
        let batch = engine.generate_batch_seeded(
            kind,
            batch_size,
            seq_len,
            false,
            seed.wrapping_add(step as u64) as usize,
            device,
        )?;
        let (inputs, targets, maybe_mask) = (batch.inputs, batch.targets, Some(batch.loss_mask));
        let logits = model.forward(&inputs)?;
        let (b, l, v) = logits.dims3()?;

        let ce_loss = if let Some(ref mask) = maybe_mask {
            let flat_l = logits.reshape((b * l, v))?;
            let flat_t = targets.reshape((b * l,))?;
            let flat_m = mask.reshape((b * l,))?;
            let log_sm = candle_nn::ops::log_softmax(&flat_l, 1)?;
            let t_log_probs = log_sm.gather(&flat_t.unsqueeze(1)?, 1)?.squeeze(1)?;
            let m_nll = (t_log_probs.neg()? * flat_m.clone())?;
            let m_sum = flat_m.sum_all()?.to_scalar::<f32>()?.max(1.0);
            (m_nll.sum_all()? / (m_sum as f64))?
        } else {
            let flat_l = logits.reshape((b * l, v))?;
            let flat_t = targets.reshape((b * l,))?;
            let log_sm = candle_nn::ops::log_softmax(&flat_l, 1)?;
            candle_nn::loss::nll(&log_sm, &flat_t)?
        };

        last_train_loss = ce_loss.to_scalar::<f32>()?;

        // Train accuracy
        let preds = logits.argmax(candle_core::D::Minus1)?;
        let preds_vec: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let targets_vec: Vec<u32> = targets.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        if let Some(ref mask) = maybe_mask {
            let m_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;
            let mut active = 0usize;
            let mut matches = 0usize;
            for i in 0..preds_vec.len() {
                if m_vec.get(i).copied().unwrap_or(0.0) > 0.5 {
                    active += 1;
                    if preds_vec[i] == targets_vec[i] {
                        matches += 1;
                    }
                }
            }
            last_train_acc = if active > 0 {
                matches as f32 / active as f32
            } else {
                0.0
            };
        } else {
            let matches = preds_vec
                .iter()
                .zip(&targets_vec)
                .filter(|(p, t)| p == t)
                .count();
            last_train_acc = matches as f32 / preds_vec.len().max(1) as f32;
        }

        let grads = ce_loss.backward()?;
        optimizer.step(&grads)?;
    }

    // Validation evaluation
    let batch = engine.generate_batch_seeded(
        kind,
        batch_size,
        seq_len,
        true,
        seed.wrapping_add(0x4f1bbcdc) as usize,
        device,
    )?;
    let (val_inputs, val_targets, maybe_val_mask) =
        (batch.inputs, batch.targets, Some(batch.loss_mask));
    let val_logits = model.forward(&val_inputs)?;
    let (b, l, v) = val_logits.dims3()?;

    let (val_loss, val_acc) = if let Some(ref mask) = maybe_val_mask {
        let flat_l = val_logits.reshape((b * l, v))?;
        let flat_t = val_targets.reshape((b * l,))?;
        let flat_m = mask.reshape((b * l,))?;
        let log_sm = candle_nn::ops::log_softmax(&flat_l, 1)?;
        let t_log_probs = log_sm.gather(&flat_t.unsqueeze(1)?, 1)?.squeeze(1)?;
        let m_nll = (t_log_probs.neg()? * flat_m.clone())?;
        let m_sum = flat_m.sum_all()?.to_scalar::<f32>()?.max(1.0);
        let l_val = (m_nll.sum_all()? / (m_sum as f64))?.to_scalar::<f32>()?;

        let preds = val_logits.argmax(candle_core::D::Minus1)?;
        let preds_vec: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let targets_vec: Vec<u32> = val_targets.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let m_vec: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        let mut active = 0usize;
        let mut matches = 0usize;
        for i in 0..preds_vec.len() {
            if m_vec.get(i).copied().unwrap_or(0.0) > 0.5 {
                active += 1;
                if preds_vec[i] == targets_vec[i] {
                    matches += 1;
                }
            }
        }
        let a_val = if active > 0 {
            matches as f32 / active as f32
        } else {
            0.0
        };
        (l_val, a_val)
    } else {
        let flat_l = val_logits.reshape((b * l, v))?;
        let flat_t = val_targets.reshape((b * l,))?;
        let log_sm = candle_nn::ops::log_softmax(&flat_l, 1)?;
        let l_val = candle_nn::loss::nll(&log_sm, &flat_t)?.to_scalar::<f32>()?;

        let preds = val_logits.argmax(candle_core::D::Minus1)?;
        let preds_vec: Vec<u32> = preds.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let targets_vec: Vec<u32> = val_targets.flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        let matches = preds_vec
            .iter()
            .zip(&targets_vec)
            .filter(|(p, t)| p == t)
            .count();
        let a_val = matches as f32 / preds_vec.len().max(1) as f32;
        (l_val, a_val)
    };

    Ok((
        model,
        BaselineTrainResult {
            model_kind: model_kind.to_string(),
            total_parameters: total_params,
            final_train_loss: last_train_loss,
            final_train_acc: last_train_acc,
            final_val_loss: val_loss,
            final_val_acc: val_acc,
            epochs,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn seeded_models_have_repeatable_outputs_and_connected_weights() -> Result<()> {
        let device = Device::Cpu;
        let tokens = Tensor::new(&[[1u32, 2, 3, 4], [3, 1, 4, 2]], &device)?;
        let targets = Tensor::new(&[2u32, 1], &device)?;
        for kind in [
            "transformer",
            "gru",
            "rnn",
            "pointwise",
            "tied-local",
            "untied-local",
        ] {
            let first = VarMap::new();
            let model = build_baseline(
                kind,
                VarBuilder::from_varmap(&first, DType::F32, &device),
                8,
                4,
                3,
            )?;
            initialize_seeded(&first, 771)?;
            let second = VarMap::new();
            let repeat = build_baseline(
                kind,
                VarBuilder::from_varmap(&second, DType::F32, &device),
                8,
                4,
                3,
            )?;
            initialize_seeded(&second, 771)?;
            let output = model.forward(&tokens)?;
            assert_eq!(
                output.flatten_all()?.to_vec1::<f32>()?,
                repeat.forward(&tokens)?.flatten_all()?.to_vec1::<f32>()?,
                "seed repeat: {}",
                kind
            );
            initialize_seeded(&second, 772)?;
            assert_ne!(
                output.flatten_all()?.to_vec1::<f32>()?,
                repeat.forward(&tokens)?.flatten_all()?.to_vec1::<f32>()?,
                "seed variation: {}",
                kind
            );
            let scored = output.narrow(1, 3, 1)?.squeeze(1)?;
            let loss = candle_nn::loss::nll(&candle_nn::ops::log_softmax(&scored, 1)?, &targets)?;
            let grads = loss.backward()?;
            for (name, variable) in first.data().lock().unwrap().iter() {
                if !name.ends_with("weight") {
                    continue;
                }
                let grad = grads
                    .get(variable)
                    .unwrap_or_else(|| panic!("missing gradient {}:{}", kind, name));
                let norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                assert!(
                    norm.is_finite() && norm > 0.0,
                    "bad gradient {}:{} = {}",
                    kind,
                    name,
                    norm
                );
            }
            assert_eq!(
                model.total_parameters(),
                first
                    .all_vars()
                    .iter()
                    .map(|v| v.elem_count())
                    .sum::<usize>()
            );
        }
        Ok(())
    }

    #[test]
    fn causal_models_prefix_and_streaming_outputs_agree() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
        let b = Tensor::new(&[[1u32, 2, 7, 6]], &device)?;
        for kind in ["transformer", "gru", "rnn", "pointwise"] {
            let vm = VarMap::new();
            let model = build_baseline(
                kind,
                VarBuilder::from_varmap(&vm, DType::F32, &device),
                8,
                4,
                2,
            )?;
            initialize_seeded(&vm, 811)?;
            let full = model.forward(&a)?;
            let changed = model.forward(&b)?;
            let prefix_diff = (&full.narrow(1, 0, 2)? - &changed.narrow(1, 0, 2)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(prefix_diff < 1e-6, "future-token leakage {}", kind);
            let mut hidden = None;
            for tick in 0..4 {
                let (output, state) = model.step(&a.narrow(1, tick, 1)?, hidden.as_ref())?;
                let error = (&output - &full.narrow(1, tick, 1)?)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(
                    error < 1e-5,
                    "stream mismatch {} tick {}: {}",
                    kind,
                    tick,
                    error
                );
                hidden = Some(state);
            }
        }
        Ok(())
    }

    #[test]
    fn local_depth_has_finite_propagation_and_no_boundary_wrap() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let model = LocalDepthBaseline::new(
            VarBuilder::from_varmap(&vm, DType::F32, &device),
            8,
            4,
            3,
            true,
        )?;
        initialize_seeded(&vm, 712)?;
        let a = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;
        let b = Tensor::new(&[[1u32, 2, 3, 4, 6]], &device)?;
        let x = model.state_at(&a, 1)?;
        let y = model.state_at(&b, 1)?;
        assert_eq!(
            x.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?,
            y.narrow(1, 0, 3)?.flatten_all()?.to_vec1::<f32>()?
        );
        assert_ne!(
            x.narrow(1, 3, 1)?.flatten_all()?.to_vec1::<f32>()?,
            y.narrow(1, 3, 1)?.flatten_all()?.to_vec1::<f32>()?
        );
        let untied = LocalDepthBaseline::new(
            VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device),
            8,
            4,
            2,
            false,
        )?;
        assert!(untied.state_at(&a, 3).is_err());
        Ok(())
    }

    #[test]
    fn test_baselines_forward_and_params() -> Result<()> {
        let dev = Device::Cpu;
        let vocab_size = 20;
        let channels = 16;
        let seq_len = 8;
        let batch_size = 2;

        let tokens = Tensor::zeros((batch_size, seq_len), DType::U32, &dev)?;

        // 1. Transformer
        let vm_tf = VarMap::new();
        let vb_tf = VarBuilder::from_varmap(&vm_tf, DType::F32, &dev);
        let transformer = TransformerBaseline::new(vb_tf, vocab_size, channels, 32)?;
        let tf_logits = transformer.forward(&tokens)?;
        assert_eq!(tf_logits.dims3()?, (batch_size, seq_len, vocab_size));
        let tf_params = transformer.total_parameters();
        assert!(tf_params > 0);

        // 2. GRU
        let vm_gru = VarMap::new();
        let vb_gru = VarBuilder::from_varmap(&vm_gru, DType::F32, &dev);
        let gru = GruBaseline::new(vb_gru, vocab_size, channels)?;
        let gru_logits = gru.forward(&tokens)?;
        assert_eq!(gru_logits.dims3()?, (batch_size, seq_len, vocab_size));
        let gru_params = gru.total_parameters();
        assert!(gru_params > 0);

        // 3. Simple Recurrent
        let vm_sr = VarMap::new();
        let vb_sr = VarBuilder::from_varmap(&vm_sr, DType::F32, &dev);
        let sr = SimpleRecurrentBaseline::new(vb_sr, vocab_size, channels)?;
        let sr_logits = sr.forward(&tokens)?;
        assert_eq!(sr_logits.dims3()?, (batch_size, seq_len, vocab_size));
        let sr_params = sr.total_parameters();
        assert!(sr_params > 0);

        // Parameter report check
        for model_name in ["Transformer", "GRU", "SimpleRNN"] {
            let breakdown = match model_name {
                "Transformer" => transformer.parameter_breakdown(),
                "GRU" => gru.parameter_breakdown(),
                _ => sr.parameter_breakdown(),
            };
            for (subsystem, count) in breakdown {
                assert!(
                    count > 0,
                    "Subsystem {} should have non-zero params",
                    subsystem
                );
            }
        }

        Ok(())
    }
}
