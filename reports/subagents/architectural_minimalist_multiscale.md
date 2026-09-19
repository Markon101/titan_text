=== [ARCHITECTURAL-MINIMALIST] (Model: deepseek/deepseek-v4.1-flash, Time: 50.09s) ===
# Option A: Minimal 1D Micro-Macro Cellular Hierarchy — Implementation Spec

## 0. Design Rationale (Occam's Razor Applied)

Before specifying, I strip the proposal to its load-bearing parts:

**Kept:**
- Two fields (micro $h$, macro $M$) — required for the $s=1$ vs $s=2$ causal control.
- Local pooling $D$ and local upsampling $U$ — the only mechanism that can expand receptive field per tick.
- Local zero-mean stencil $w_j = M_j - \tfrac12(M_{j-1}+M_{j+1})$ — the adversarial reviewer's fix. This is a **discrete Laplacian-like high-pass**, not a global mean. It is strictly local, translation-equivariant, and has zero DC response by construction.
- Hard-bounded gain $\gamma \le 0.10$.

**Cut (dead weight from the packet):**
- "Contraction/stability proofs" for a coupled map lattice with $\tanh$ — $\tanh$ is already 1-Lipschitz; residual drift $\alpha<1$ gives a trivial contraction bound. No new math needed.
- "Phase transition dynamics for discrete parity solitons" — this is continuum-hydrodynamics language on a 16-cell discrete grid. There are no solitons at $L=16$. Delete.
- "Optimal macro timescale ratio $k$" — sweep $k \in \{1,2,4\}$ empirically; do not derive.
- Any global operator (mean, attention, GRU). The reviewer is right: global mean subtraction couples all $L$ cells and reintroduces the DC-cheat vector via a different door.

**The single non-obvious claim to defend:** the local stencil $w_j$ is the *only* modulation channel from macro to micro. If $s=2$ beats $s=1$ with this channel, spatial coarsening is causal. If it doesn't, the hierarchy is decorative.

---

## 1. Tensor Shapes and Forward Equations

### Shapes (per batch element)

| Symbol | Shape | Meaning |
|---|---|---|
| $h^{(t)}$ | $[L, C]$ = $[16, 64]$ | micro field |
| $M^{(t)}$ | $[L_M, C_M]$ = $[8, 32]$ | macro field ($s=2$) |
| $w^{(t)}$ | $[L, C_M]$ | upsampled local-difference modulation |
| $e$ | $[L, C_{\text{in}}]$ | input embedding (bits + query mask) |
| $y$ | $[L, 1]$ | per-cell logit |

### Micro step (every tick $t \in [1,\tau]$)

Perception (radius-1 stencil, concatenated):
$$
p_i^{(t)} = \big[\, h_{i-1}^{(t-1)},\ h_i^{(t-1)},\ h_{i+1}^{(t-1)},\ e_i,\ \gamma\, w_i^{(t-1)} \,\big] \in \mathbb{R}^{3C + C_{\text{in}} + C_M}
$$

Update:
$$
\Delta h_i^{(t)} = \alpha \cdot \tanh\!\big(W_p\, p_i^{(t)} + b_p\big), \qquad h_i^{(t)} = h_i^{(t-1)} + \Delta h_i^{(t)}
$$

with $\alpha = 0.5$ fixed (not learned — one fewer parameter, and it caps per-tick drift).

Readout (shared linear head, applied every tick for dense aux loss):
$$
y_i^{(t)} = W_y\, h_i^{(t)} + b_y \in \mathbb{R}
$$

### Macro step (only when $t \equiv 0 \pmod k$)

Pooling $D$ (non-overlapping, $s=2$):
$$
M_j^{(t)} \leftarrow \tfrac12\big(h_{2j}^{(t)} + h_{2j+1}^{(t)}\big) \quad\text{projected:}\quad M_j \leftarrow W_D\, \bar{h}_j + b_D
$$
where $\bar h_j \in \mathbb{R}^C$ is the mean, and $W_D \in \mathbb{R}^{C_M \times C}$.

Macro update (radius-1 stencil on macro grid):
$$
q_j = \big[\, M_{j-1},\ M_j,\ M_{j+1} \,\big] \in \mathbb{R}^{3 C_M}
$$
$$
M_j \leftarrow M_j + \alpha_M \tanh\!\big(W_M\, q_j + b_M\big), \qquad \alpha_M = 0.5
$$

### Local zero-mean modulation $w$ (the reviewer's fix)

Compute the **local high-pass** on the macro grid:
$$
\tilde w_j = M_j - \tfrac12\big(M_{j-1} + M_{j+1}\big) \in \mathbb{R}^{C_M}
$$

This is a discrete 1D Laplacian (up to sign). It has **exact zero response to any constant field** and is strictly local (3 macro cells). No global reduction.

Upsample to micro resolution ($s=2$):
$$
w_i = \tilde w_{\lfloor i/2 \rfloor}
$$

Gain:
$$
\gamma = \gamma_{\max}\,\sigma(\tilde\gamma), \qquad \gamma_{\max}=0.10,\ \tilde\gamma \in \mathbb{R}\ \text{(scalar, learned)}
$$

**Boundary handling:** use **periodic** wrap on both micro and macro grids. This matches the packet's observed Slot-3 wrap-around behavior and avoids padding parameters.

---

## 2. Parameter Count

| Component | Shape | Params |
|---|---|---|
| $W_p$ | $(3C + C_{\text{in}} + C_M) \times C$ = $(192 + 3 + 32) \times 64$ | 14,528 |
| $b_p$ | $64$ | 64 |
| $W_y$ | $64 \times 1$ | 64 |
| $b_y$ | $1$ | 1 |
| $W_D$ | $32 \times 64$ | 2,048 |
| $b_D$ | $32$ | 32 |
| $W_M$ | $3 C_M \times C_M$ = $96 \times 32$ | 3,072 |
| $b_M$ | $32$ | 32 |
| $\tilde\gamma$ | scalar | 1 |
| **Total** | | **19,842** |

Under 20k parameters. Fits in L2 on Snapdragon 8 Elite. No embedding table (input is 3 bits + query mask, one-hot into $C_{\text{in}}=3$).

**Comparison to the rejected GRU path:** a single 128-unit GRU is ~50k params and would trivially solve the task. This spec is 2.5× smaller and cannot cheat.

---

## 3. Rust / Candle Structs and Loop

```rust
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

#[derive(Clone, Debug)]
pub struct NcaHierarchyConfig {
    pub l_micro: usize,       // 16
    pub c_micro: usize,       // 64
    pub c_in: usize,          // 3  (x0, x1, x2 one-hot; query slot uses all-zero)
    pub stride: usize,        // 2 for hierarchy, 1 for degenerate control
    pub c_macro: usize,       // 32
    pub tau: usize,           // 16
    pub macro_period: usize,  // k: macro updates every k micro ticks
    pub alpha: f64,           // 0.5
    pub alpha_macro: f64,     // 0.5
    pub gamma_max: f64,       // 0.10
}

impl NcaHierarchyConfig {
    pub fn l_macro(&self) -> usize { self.l_micro / self.stride }
}

pub struct NcaHierarchy {
    cfg: NcaHierarchyConfig,
    w_p: Linear,       // (3C + C_in + C_M) -> C
    w_y: Linear,       // C -> 1
    w_d: Linear,       // C -> C_M
    w_m: Linear,       // 3 C_M -> C_M
    gamma_raw: Tensor, // scalar
}

impl NcaHierarchy {
    pub fn new(cfg: NcaHierarchyConfig, vb: VarBuilder) -> Result<Self> {
        let c = cfg.c_micro;
        let cm = cfg.c_macro;
        let in_dim = 3 * c + cfg.c_in + cm;
        Ok(Self {
            w_p: linear(in_dim, c, vb.pp("w_p"))?,
            w_y: linear(c, 1, vb.pp("w_y"))?,
            w_d: linear(c, cm, vb.pp("w_d"))?,
            w_m: linear(3 * cm, cm, vb.pp("w_m"))?,
            gamma_raw: vb.get(&[], "gamma_raw")?,
            cfg,
        })
    }

    /// Periodic shift by +1 along the length axis (dim 1 of [B, L, C]).
    fn roll_plus1(x: &Tensor) -> Result<Tensor> {
        let l = x.dim(1)?;
        let head = x.narrow(1, l - 1, 1)?;
        let tail = x.narrow(1, 0, l - 1)?;
        Tensor::cat(&[&head, &tail], 1)
    }
    fn roll_minus1(x: &Tensor) -> Result<Tensor> {
        let l = x.dim(1)?;
        let head = x.narrow(1, 0, 1)?;
        let tail = x.narrow(1, 1, l - 1)?;
        Tensor::cat(&[&tail, &head], 1)
    }

    /// Local high-pass on macro grid: w_j = M_j - 0.5*(M_{j-1} + M_{j+1}).
    /// Strictly local, zero DC response, no global reduction.
    fn local_highpass(m: &Tensor) -> Result<Tensor> {
        let left = Self::roll_minus1(m)?;
        let right = Self::roll_plus1(m)?;
        let avg = ((left + right)? * 0.5)?;
        m - avg
    }

    /// Non-overlapping mean-pool [B, L, C] -> [B, L/s, C].
    fn pool_mean(x: &Tensor, s: usize) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let lm = l / s;
        x.reshape((b, lm, s, c))?.mean(D::Minus2)
    }

    /// Nearest-neighbor upsample [B, L_M, C] -> [B, L_M*s, C].
    fn upsample_nearest(x: &Tensor, s: usize) -> Result<Tensor> {
        let (b, lm, c) = x.dims3()?;
        x.reshape((b, lm, 1, c))?
            .broadcast_as((b, lm, s, c))?
            .reshape((b, lm * s, c))
    }

    pub fn forward(&self, emb: &Tensor) -> Result<Tensor> {
        // emb: [B, L, C_in]
        let (b, l, _) = emb.dims3()?;
        let dev = emb.device();
        let c = self.cfg.c_micro;
        let cm = self.cfg.c_macro;

        let mut h = Tensor::zeros((b, l, c), DType::F32, dev)?;
        let mut m = Tensor::zeros((b, self.cfg.l_macro(), cm), DType::F32, dev)?;

        let gamma = (self.gamma_raw.sigmoid()? * self.cfg.gamma_max)?;

        // Collect per-tick logits for dense aux supervision.
        let mut logits_per_tick: Vec<Tensor> = Vec::with_capacity(self.cfg.tau);

        for t in 1..=self.cfg.tau {
            // --- modulation from macro to micro ---
            let w_macro = Self::local_highpass(&m)?;              // [B, L_M, C_M]
            let w = Self::upsample_nearest(&w_macro, self.cfg.stride)?; // [B, L, C_M]
            let w = (w * gamma)?;

            // --- micro perception ---
            let h_l = Self::roll_minus1(&h)?;
            let h_r = Self::roll_plus1(&h)?;
            let p = Tensor::cat(&[&h_l, &h, &h_r, emb, &w], D::Minus1)?; // [B, L, 3C+C_in+C_M]
            let delta = self.w_p.forward(&p)?.tanh()?;
            h = (h + (delta * self.cfg.alpha)?)?;

            // --- readout every tick ---
            logits_per_tick.push(self.w_y.forward(&h)?);          // [B, L, 1]

            // --- macro update on slow clock ---
            if t % self.cfg.macro_period == 0 {
                let pooled = Self::pool_mean(&h, self.cfg.stride)?; // [B, L_M, C]
                let m_in = self.w_d.forward(&pooled)?;              // [B, L_M, C_M]
                let m_l = Self::roll_minus1(&m_in)?;
                let m_r = Self::roll_plus1(&m_in)?;
                let q = Tensor::cat(&[&m_l, &m_in, &m_r], D::Minus1)?;
                let dm = self.w_m.forward(&q)?.tanh()?;
                m = (m_in + (dm * self.cfg.alpha_macro)?)?;
            }
        }

        // Stack: [B, tau, L, 1]
        Tensor::stack(&logits_per_tick, 1)
    }
}
```

### Step execution loop (training)

```rust
// Sparse loss at query slots {3, 7, 11, 15}; dense aux loss at all cells.
let logits = model.forward(&emb)?;                 // [B, tau, L, 1]
let logits_final = logits.narrow(1, cfg.tau - 1, 1)?.squeeze(1)?; // [B, L, 1]

let sparse = masked_ce(&logits_final, &targets, &query_mask)?;
let dense  = masked_ce(&logits, &prefix_targets, &all_mask)?;
let loss = sparse + 0.5 * dense;
```

---

## 4. Degenerate Control ($s=1$)

Same struct, same parameter count **except** $W_D$ becomes $C \times C_M$ and $W_M$ unchanged. Set:

```rust
NcaHierarchyConfig {
    stride: 1,          // L_M = L = 16, no spatial coarsening
    c_macro: 32,
    macro_period: 2,
    ..default
}
```

At $s=1$ the macro field has the same resolution as micro. The local high-pass $w_j = M_j - \tfrac12(M_{j-1}+M_{j+1})$ still applies, so the *only* difference between control and treatment is whether pooling co
