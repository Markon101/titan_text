# Generative ASCII Architecture & Causal Autoregression

*Document Version*: 1.0  
*Date*: 2026-09-26  
*Repository*: `titan_text`  
*Module References*: [`src/ascii_sampler.rs`](../src/ascii_sampler.rs), [`src/ascii_corpus.rs`](../src/ascii_corpus.rs), [`src/vocab.rs`](../src/vocab.rs), [`src/cli.rs`](../src/cli.rs)

---

## 1. Overview & Architectural Motivation

Titan Text originated as a 1D neural cellular sequence laboratory investigating recurrent latent dynamics, coordinate channels, and pushdown automata emulation (e.g. Dyck-4, IPPR). Prior experiments operated predominantly in discriminative or parallel-readout modes, where an input sequence of length $L$ was observed and queries were resolved after $T$ developmental updates across all spatial cells.

The **Generative ASCII Campaign** establishes the smallest viable extension of the Titan NCA into a genuine free-running autoregressive generative model. Rather than expanding parameter count or adopting standard transformer/RNN components, it retains the pure Neural Cellular Automaton (~43.8k parameters) and operates token-by-token using causal 1D stencil perception.

---

## 2. Core Architecture Specifications

### 2.1 Vocabulary & Token Layout
Implemented in [`src/vocab.rs`](../src/vocab.rs) via `Vocab::new_ascii_art()`:
- **Index 0**: `<pad>` (padding token)
- **Index 1**: `<bos>` (beginning of sequence)
- **Index 2**: `<eos>` (end of sequence / halt trigger)
- **Index 3**: `<unk>` (unknown token)
- **Index 4**: `\n` (newline, ASCII byte 10)
- **Indices 5..=99**: Printable ASCII characters (byte values 32 through 126: space, symbols, digits, upper/lower alpha)
- **Total Vocabulary Size**: 100 tokens.

Legacy 99-token vocabulary (`Vocab::new_ascii()`) without newline is strictly preserved for backwards compatibility with historical discriminative checkpoints.

### 2.2 Autoregressive Causal Perception & Sampling
Implemented in [`src/ascii_sampler.rs`](../src/ascii_sampler.rs) (`AsciiSampler`):
- **Context Window**: Sliding window of length $L = 48$.
- **Causal Stencil**: The perception operator uses causal left-stencil:
  $$N(i) = \{i-1, i\}$$
  ensuring that each position $i$ depends strictly on $i$ and historical predecessor tokens $j < i$, preventing rightward information leakage.
- **Latent Recurrence $\tau$**: At each generation step $t$, the current context is passed into the NCA field, and recurrent updates are iterated for $\tau$ discrete ticks:
  $$\mathbf{x}_{\tau+1} = \mathbf{x}_\tau + \alpha \cdot \Delta \mathbf{x}(\mathbf{x}_\tau)$$
  where $\Delta \mathbf{x}$ is gated residual cellular velocity.
- **Logit Readout**: Logits for the next token are projected from the terminal active position $w-1$.
- **Decoding Dynamics**:
  - Softmax with temperature scaling: $P(v) \propto \exp(z_v / T)$.
  - Top-$k$ filtering to eliminate low-probability tail entropy.
  - Greedy argmax mode for $T \le 10^{-4}$.
  - Termination occurs when `<eos>` is emitted or token budget (`--max-len`) is exhausted.

### 2.3 Parameter Count & Complexity
- **Embedding**: $100 \times 64 = 6,400$ weights.
- **Cellular Perception & Dense Layers**:
  - Perception projection: $3 \times 64 \to 96 = 18,432$ weights.
  - Directional delta: $96 \to 64 = 6,144$ weights.
  - Channel gate: $96 \to 64 = 6,144$ weights.
- **Logit Readout**: $64 \times 100 = 6,400$ weights.
- **Total Parameters**: Exactly **43,844 parameters** (pure Rust + Candle, ARM64 Termux execution).

---

## 3. Procedural ASCII Curriculum (`src/ascii_corpus.rs`)

Rather than unconstrained internet web scraping, training utilizes deterministic, procedural structural families:

1. **`<BOX>`**: Rectangles with boundaries (`+`, `-`, `|`, `#`, `*`) of varying dimensions.
2. **`<CHECKER>`**: Alternating binary textures (`# `, `.*`, `* `).
3. **`<DIAMOND>`**: Horizontally and vertically symmetric diamond contours (`/`, `\`).
4. **`<MAZE>`**: Orthogonal corridors and wall junctions.
5. **`<BANNER>`**: Centered headers framed by double or star borders (`========`).
6. **`<FACE>`**: Symmetric emoticons and stylized animal faces (`( o.o )`, ` > ^ < `).
7. **`<MOUNTAIN>`**: Sloping ridges, peaks, and horizon lines (`/\`, `_`).
8. **`<ABSTRACT>`**: Cellular glyph blocks, zig-zag waves, and diagonals.

### 4-Stage Difficulty Progression:
- **Stage A**: Basic delimiters, newlines, and rectangular bounding.
- **Stage B**: Motif repetition and reflection symmetry.
- **Stage C**: Closed geometric category generation conditioned on prompt control tokens.
- **Stage D**: Non-deterministic abstract compositions.

---

## 4. Empirical Causal Findings

Evaluating across fixed seed batteries (`42, 101, 202, 303, 404`) and tick sweeps ($\tau \in \{0, 1, 2, 4, 8, 16\}$):

1. **Causal Necessity of Recurrence**:
   - $\tau=0$ (feedforward projection only) and `--lesion-state` collapse immediately to `<eos>` (0 chars emitted).
   - $\tau=2$ produces degenerate single-character loops (`######`).
   - $\tau=4$ (training distribution) and $\tau=8$ reliably construct multiline closed boxes with borders (`+====+`), walls (`:`), and clean internal spacing ($63.5\%$ horizontal symmetry).
   - $\tau=16$ exhibits boundary over-deliberation (`+=================`), demonstrating that latent compute horizon causally dictates spatial token extension.
2. **Zero Rote Memorization**:
   - Exact training matches: **0 / 96 (0.0%)**.
   - Mean nearest Levenshtein edit similarity: $0.391 \pm 0.21$ (vs $0.081$ for untrained noise).
   - The model learns the underlying geometric grammar rather than playing back stored examples.

---

## 5. Known Limitations & Research Debt

1. **Boundary Saturation at Deep Horizons ($\tau=16$)**:
   - Constant $\tau$ across all tokens over-deliberates on simple linear runs (e.g. repeated `-` or `=`).
   - **Research Debt `RD-011`**: Implement adaptive step halting (`AdaptiveHalting` / kinetic energy threshold) per token, allowing early stopping on linear runs while preserving deep deliberation at corners and newlines.
2. **Vertical Column Coherence on Large Grids ($L \ge 64$)**:
   - While short boxes ($H \le 6$) close accurately, wider vertical spans experience column shear.
   - Integrating discrete carry channels (`--carry-channels 16 --carry-quantization ste_sign`) is the primary candidate to maintain multi-line column memory.
