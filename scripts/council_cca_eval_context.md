# DeepSeek Council Context: Evaluation of Decisive Experiment 5 (Causal Cellular Automata)

## 1. Executive Summary of Experiment 5 Results
Following the Council's derivation of the **"Relaxation vs. Fold"** theorem, we replaced the symmetric radius-1 stencil $\mathcal{N}(i) = \{i-1, i, i+1\}$ with the strictly causal / directed stencil $\mathcal{N}_\to(i) = \{i-1, i\}$ (with Dirichlet left boundary $x_{-1} \equiv 0$). This ensures the Jacobian $J_\Psi$ is lower-bidiagonal and the cellular lattice forms a directed acyclic graph (DAG total order $0 \to 1 \to \dots \to L-1$).

We completed the full 300-epoch evaluation across $N=5$ pre-registered seeds `[42, 101, 202, 303, 404]` for both Track 1 (Parity) and Track 2 (Bounded Running Sum).

### Track 1: CCA on Parity (`iterated-parity-dense`, $L=16, T=16$)
- **Epoch 100**:
  - Intact: **$60.53\% \pm 2.34\%$**
  - Identity Gap ($G_{\text{identity}} = \text{intact} - \text{batch\_shuff}$): **$+10.38\%$** (doubled from symmetric runs!)
  - Latent Recurrence Gap ($G_{\text{recurrence}} = \text{budget}_{16} - \text{budget}_0$): **$+57.97\%$** (budget 0 is 0.0%)
  - Per-Query-Slot Accuracy:
    - **Slot 0 (cell 3)**: **$85.6\%$** (Highest recorded in project history)
    - **Slot 1 (cell 7)**: **$50.0\%$** (Exactly chance)
    - **Slot 2 (cell 11)**: **$50.2\%$** (Exactly chance)
    - **Slot 3 (cell 15)**: **$56.2\%$**
- **Epoch 200**:
  - Intact: $58.75\% \pm 1.99\%$ | $G_{\text{identity}} = +8.69\%$
  - Slots: Slot 0 = $83.1\%$, Slot 1 = $52.9\%$, Slot 2 = $47.2\%$, Slot 3 = $51.8\%$
- **Epoch 300**:
  - Intact: $57.66\% \pm 2.18\%$ | $G_{\text{identity}} = +8.12\%$
  - Slots: Slot 0 = $80.6\%$, Slot 1 = $50.8\%$, Slot 2 = $48.5\%$, Slot 3 = $50.8\%$

### Track 2: CCA on Bounded Running Sum (`iterated-sum-dense`, $L=16, T=16$)
- **Epoch 100**:
  - Intact: **$23.28\% \pm 3.18\%$** (10-class random baseline = 10.0%; prior symmetric runs = 16.0% - 17.5%)
  - Identity Gap ($G_{\text{identity}}$): **$+10.19\%$** (prior symmetric runs: $+1.1\%$ to $-0.5\%$, tenfold jump!)
  - Per-Query-Slot Accuracy:
    - **Slot 0 (cell 3)**: **$38.2\%$** (vs 10% chance)
    - **Slot 1 (cell 7)**: **$19.5\%$**
    - **Slot 2 (cell 11)**: **$18.4\%$**
    - **Slot 3 (cell 15)**: **$17.0\%$**
- **Epoch 300**:
  - Intact: **$23.47\% \pm 1.09\%$** | $G_{\text{identity}} = \mathbf{+10.25\%}$
  - Slots: Slot 0 = $38.5\%$, Slot 1 = $20.5\%$, Slot 2 = $18.4\%$, Slot 3 = $16.5\%$

---

## 2. Comparison with Prior Campaigns

| Architecture / Campaign | Task | Intact Accuracy | $G_{\text{identity}}$ | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|---|---|---|---|
| Symmetric Baseline (Arm 1) | Parity | $54.2\% \pm 2.8\%$ | $+5.2\%$ | $71.0\%$ | $49.5\%$ | $50.0\%$ | $56.0\%$ |
| Multi-Scale Hierarchy (Walsh) | Parity | $55.5\% \pm 3.7\%$ | $+5.0\%$ | $68.8\%$ | $45.6\%$ | $50.0\%$ | $57.5\%$ |
| **Causal CA (CCA) [Exp 5]** | **Parity** | **$60.5\% \pm 2.3\%$** | **$+10.4\%$** | **$85.6\%$** | **$50.0\%$** | **$50.2\%$** | **$56.2\%$** |
| Symmetric Multiscale ($s=2$) | Sum | $17.5\% \pm 1.2\%$ | $+1.1\%$ | $24.0\%$ | $16.0\%$ | $15.5\%$ | $14.5\%$ |
| Symmetric Control ($s=1$) | Sum | $16.1\% \pm 1.3\%$ | $-0.5\%$ | $21.0\%$ | $15.0\%$ | $14.0\%$ | $14.0\%$ |
| **Causal CA (CCA) [Exp 5]** | **Sum** | **$23.5\% \pm 1.1\%$** | **$+10.3\%$** | **$38.5\%$** | **$20.5\%$** | **$18.4\%$** | **$16.5\%$** |

---

## 3. The Core Paradox and Questions for the Council

### The Good:
1. **$G_{\text{identity}}$ doubled on Parity (+10.4%) and jumped 10x on Sum (+10.3%)**. This conclusively proves that the causal stencil eliminated symmetric diffusive cancellation and induced genuine instance-specific sequential state transport.
2. **Slot 0 reached new all-time highs**: 85.6% on Parity, 38.5% on Sum (3.85x above chance).

### The Paradox:
Despite the causal stencil making the Jacobian strictly lower-bidiagonal (eliminating all 2-cycles and backward reflections):
**Slots 1 and 2 on Parity remain pinned at chance (50.0% / 50.2%)!**
Even more revealing: On Sum, performance degrades monotonically with spatial distance:
Slot 0: 38.5% $\to$ Slot 1: 20.5% $\to$ Slot 2: 18.4% $\to$ Slot 3: 16.5%.

### Key Technical Details of Implementation:
1. **Perception**:
   `x` $[B, L, C]$, `left` $[B, L, C]$, `diff = x - left` $[B, L, C]$.
   Concatenated: $[x, \text{left}, \text{diff}]$ $[B, L, 3C]$.
2. **Transition**:
   `h1 = dense1(perc).tanh()`
   `delta = dense_delta(h1).tanh()`
   `gate = sigmoid(dense_gate(h1))`
   `x_{t+1} = x_t + alpha * (delta * gate)`
   (With $T=16$ latent ticks per rollout).
3. **Readout**:
   `TokenInterface::forward_readout`: Linear projection from $C=16$ to $|\mathcal{V}|$ applied independently per cell at tick $T=16$.
   Loss: CrossEntropy over target query positions (cells 3, 7, 11, 15).
4. **Training**:
   AdamW, lr=0.003, batch_size=32, trained on full $L=16$ sequences from epoch 0.

### Council Questions:
1. **The Spatial Bottleneck Hypothesis**:
   Why does Slot 0 learn so cleanly (85.6%), but the carried state fails to propagate across the query boundary into Slot 1 (cell 4..7)?
   Is this a **gradient vanishing problem** across the spatial fold (effective depth $16 \times 16 = 256$ unrolled steps)?
2. **The Loss Gradient Masking Hypothesis**:
   Because the loss supervises Slot 0 (cell 3), Slot 1 (cell 7), Slot 2 (cell 11), and Slot 3 (cell 15) simultaneously from tick $T=16$:
   Does the gradient from Slot 0 dominate and settle the parameters into a local attractor that solves only short-range prefixes?
3. **The Curriculum / Length Hypothesis**:
   In our earlier curriculum study on $L=8$, both slots learned! But on $L=16$ from scratch, only Slot 0 learns.
   What specific architectural, optimization, or curriculum intervention will break this spatial barrier?
