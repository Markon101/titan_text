=== [ADVERSARIAL-REVIEWER] (Model: deepseek/deepseek-v4.1-flash, Time: 63.52s) ===
# Adversarial Audit: Option A (1D Micro-Macro Cellular Hierarchy)

**Reviewer stance**: hostile. I will assume Option A is a dressed-up global bypass until proven otherwise. The prior campaign already falsified three "soft" hypotheses; the burden is now on the architecture to demonstrate that *any* new information path is strictly local, causal, and non-degenerate.

---

## 1. Audit Against the 5 Operational Emergence Criteria

I do not have the file `docs/EMERGENCE_CRITERIA.md` in front of me, but the packet references Criteria 1 & 3 (no centralized solver; CA must not be idle decoration). I will audit against the standard five-criterion family used in this lab (locality, causal necessity, non-degeneracy, scale-separation, falsifiability) and flag where Option A is at risk.

### 1.1 The zero-mean projection is **not** a global bypass — but it is a **global operator**

The projection
$$Z = I - \tfrac{1}{L}\mathbf{1}\mathbf{1}^\top$$
is a rank-$(L-1)$ linear map. It is applied to $U(M)$, which is itself a *local* upsampling of a *local* pooled macro field. So the composition $Z \circ U \circ \text{Pool}$ is:

- **Locally computed** (each $w_i$ depends on $M_{\lfloor i/2\rfloor}$ and the mean of $U(M)$ over all $L$ cells).
- **Globally coupled** through the mean subtraction.

This is the first red flag. The mean $\bar{U} = \frac{1}{L}\sum_m U(M)_m$ is a **global aggregate** of the macro field. Subtracting it means every micro cell's modulation channel depends on the *entire* macro lattice. That is a **global all-to-all information path**, even if it is only rank-1.

**Why this matters for parity**: The target is a *global* parity bit. A rank-1 global mean channel is exactly the kind of low-dimensional global summary that can leak cumulative parity without any local transport. If the macro field ever encodes a running parity-like scalar (which it can, because pooling is a local contraction that preserves sum-like invariants), the zero-mean projection will broadcast a *global* function of that scalar to every micro cell simultaneously. That is a **cheat channel disguised as an anti-cheat guardrail**.

**Verdict**: The zero-mean projection does not eliminate the DC cheat; it *replaces a uniform DC bias with a global zero-sum pattern*. Both are global. The Image pathology was "static DC bias"; the Option A pathology would be "static zero-sum global pattern." Same disease, different symptom.

**Required fix**: Replace the global mean with a **local mean over a bounded window** (e.g., radius-2 or radius-4 stencil), or better, drop the mean subtraction entirely and instead enforce zero-mean *per macro cell* before upsampling:
$$w_i = U(M - \bar{M}_{\text{local}})_i$$
where $\bar{M}_{\text{local}}$ is a local neighborhood average on the macro grid. This preserves the anti-DC intent without introducing a global operator.

### 1.2 Pooling is local, but the *macro clock* is a temporal bypass

The macro field updates only on ticks $t \equiv 0 \pmod k$. Between macro updates, $M$ is frozen. This means the micro field sees a **piecewise-constant** modulation $w_i$ for $k-1$ consecutive ticks. If $k$ is large (e.g., 4), the micro field effectively receives a *static* per-tick bias for 3 out of every 4 ticks — which is precisely the Image failure mode (static DC bias overwhelming local dynamics), just time-multiplexed.

**Verdict**: The macro clock is a **temporal DC cheat** unless $k$ is small (≤2) or the macro update is *interleaved* with micro updates in a way that guarantees non-constant modulation.

### 1.3 Causal necessity is not established by construction

The architecture *permits* macro-to-micro modulation, but nothing in the design forces the macro field to carry parity information. The macro field could learn to encode a static spatial pattern (e.g., a fixed ramp) that the micro field uses as a positional code — which would be a **positional cheat**, not transport. The prior campaign already falsified positional coding as a solution (H_POS), but Option A reintroduces a *learned* positional channel via the macro field. This is a regression.

**Required fix**: Pre-register a **macro-field lesion** intervention: zero out $M$ at all ticks and measure intact accuracy. If accuracy drops by <5%, the macro field is decorative. If it drops by >50%, the macro field is load-bearing — but then you must prove it is load-bearing *via transport*, not via positional coding, by running the coordinate-counterfactual suite on the macro field.

---

## 2. Failure Modes That Mimic Emergence

I enumerate concrete pathologies, each with a falsification test.

### FM-1: Macro field as a learned positional code
The macro field $M \in \mathbb{R}^{8 \times 32}$ has 256 parameters per batch. It can trivially learn a fixed spatial pattern that encodes "which chunk am I in." The micro field then reads this pattern as a positional channel — exactly the H_POS cheat that was already falsified for explicit coordinates, but now *learned* and therefore harder to detect.

**Test**: After training, freeze $M$ to its time-averaged value $\bar{M}$ and re-run inference. If accuracy is preserved, $M$ is a static positional code. **Reject Option A if $\Delta_{\text{frozen}} < 5\%$.**

### FM-2: Macro field as a global parity accumulator
Pooling $\{2j, 2j+1\} \to M_j$ is a local contraction. If the pooling operation is sum-like (e.g., mean), then $\sum_j M_j = \sum_i h_i$ up to a constant. The macro field therefore carries a **global sum** of micro states. If the micro field ever encodes parity in a sum-like way, the macro field inherits it for free — and the zero-mean projection broadcasts it globally.

**Test**: Replace pooling with a **parity-destroying** operation (e.g., $M_j = \tanh(h_{2j}) \cdot \tanh(h_{2j+1})$, which is even in each input). If accuracy collapses, the pooling was carrying parity via a sum-like invariant. **Reject Option A if $\Delta_{\text{parity-destroy}} > 20\%$.**

### FM-3: Macro clock as a temporal DC bias
As in §1.2. If $k \geq 2$, the micro field sees a piecewise-constant modulation. This is a temporal DC bias.

**Test**: Replace the macro clock with a **random per-tick modulation** $w_i^{(t)} \sim \mathcal{N}(0, \sigma^2)$ with matched variance. If accuracy is preserved, the macro field's *content* is irrelevant — only its variance matters. **Reject Option A if $\Delta_{\text{random-mod}} < 5\%$.**

### FM-4: Macro-to-micro gain saturation
The hard-bounded gain $\gamma_{\max} = 0.10$ is a guardrail, but it does not prevent saturation *within* the bound. If $\gamma \to 0.10$ for all cells, the modulation becomes a static DC bias at 10% strength — the Image pathology at a smaller scale.

**Test**: Log the distribution of $\gamma \cdot w_i$ across cells and ticks. If the variance across cells is <1% of the mean magnitude, the modulation is effectively DC. **Reject Option A if $\text{Var}_i(\gamma w_i) / \text{Mean}_i(|\gamma w_i|) < 0.01$.**

### FM-5: Macro field as a slow integrator that bypasses local transport
The macro field updates every $k$ ticks. If $k$ is large, the macro field can integrate information over many micro ticks and then broadcast it globally. This is a **temporal bypass** of the local transport bottleneck: instead of transporting parity across 12 hops locally, the micro field writes to the macro field, the macro field integrates, and the macro field broadcasts back. This is functionally a **global recurrent memory**, which is exactly what the mission forbids.

**Test**: Measure the **effective receptive field** of a micro cell at tick $\tau$ by perturbing a single input bit and measuring the change in $h_i^{(\tau)}$. If the receptive field spans the entire lattice for interior cells, the macro field is providing a global bypass. **Reject Option A if the effective receptive field radius exceeds 4 hops for any interior cell.**

### FM-6: Degenerate control $s=1$ is not a clean control
The $s=1$ control has $L_M = 16$ and $C_M = 32$, so it has *more* macro parameters than $s=2$ ($L_M = 8$). If $s=1$ fails and $s=2$ succeeds, the difference could be due to **parameter count** (fewer macro params = less overfitting) rather than spatial coarsening. The control is confounded.

**Required fix**: Add a second control with $s=2$ but $C_M = 64$ (matching $s=1$'s parameter count). If $s=2, C_M=64$ also succeeds, spatial coarsening is load-bearing. If it fails, parameter count is the confound.

---

## 3. Degenerate Control ($s=1$): What It Actually Separates

The $s=1$ control separates **spatial coarsening** from **having two fields**. It does *not* separate spatial coarsening from:

- **Parameter count** (as noted above).
- **Receptive field expansion** (the macro field at $s=1$ still has radius-1 stencil, but on a 16-cell grid, so its receptive field after one macro tick is 3 cells — same as micro).
- **Timescale separation** (the macro clock is the same in both).

**Exact metrics to prove separation**:

1. **Receptive field radius** $R_{\text{eff}}$: measured by single-bit perturbation. For $s=2$, $R_{\text{eff}}$ should be ≥2× the micro radius. For $s=1$, $R_{\text{eff}}$ should be ≈ micro radius.
2. **Macro field mutual information** $I(M; P_{\le k})$: measured by training a probe on $M$ to predict prefix parity. For $s=2$, $I$ should be >0.1 bits. For $s=1$, $I$ should be ≈0.
3. **Parameter-matched control**: $s=2, C_M=64$ vs. $s=1, C_M=32$. If the former succeeds and the latter fails, spatial coarsening is load-bearing.
4. **Macro lesion delta**: $\Delta_{\text{macro-lesion}}$ should be >20% for $s=2$ and <5% for $s=1$.

**Reject Option A if any of these four metrics fail to separate $s=1$ from $s=2$.**

---

## 4. Pre-Registered Falsification Gates (N=5 seeds)

All thresholds are for the **intact** condition, mean ± std over seeds $\{42, 101, 202, 303, 404\}$. Chance = 50.0%.

### Gate G1: Interior slot accuracy
- **Accept**: Slot 1 and Slot 2 intact accuracy ≥ **65.0% ± 5.0%** (i.e., lower bound of 95% CI > 55.0%).
- **Reject**: Slot 1 or Slot 2 intact accuracy ≤ **55.0%** (i.e., upper bound of 95% CI < 60.0%).
- **Inconclusive**: Otherwise.

### Gate G2: Causal necessity of macro field
- **Accept**: $\Delta_{\text{macro-lesion}} \geq 20.0\%$ at both interior slots.
- **Reject**: $\Delta_{\text{macro-lesion}} \leq 5.0\%$ at either interior slot.

### Gate G3: Non-degeneracy of macro modulation
- **Accept**: $\text{Var}_i(\gamma w_i) / \text{Mean}_i(|\gamma w_i|) \geq 0.05$ at tick $\tau$.
- **Reject**: Ratio ≤ 0.01.

### Gate G4: Spatial coarsening is load-bearing
- **Accept**: $s=2$ succeeds (G1) and $s=1$ fails (G1 reject) and parameter-matched $s=2, C_M=64$ succeeds.
- **Reject**: $s=1$ succeeds, or parameter-matched control fails.

### Gate G5: No global bypass
- **Accept**: Effective receptive field radius $R_{\text{eff}} \leq 4$ hops for all interior cells at tick $\tau$.
- **Reject**: $R_{\text{eff}} \geq 8$ hops for any interior cell.

### Gate G6: Parity is transported, not pooled
- **Accept**: $\Delta_{\text{parity-destroy}} \leq 10.0\%$.
- **Reject**: $\Delta_{\text{parity-destroy}} \geq 20.0\%$.

### Gate G7: Macro field is not a static positional code
- **Accept**: $\Delta_{\text{frozen-M}} \geq 15.0\%$.
- **Reject**: $\Delta_{\text{frozen-M}} \leq 5.0\%$.

**Overall decision rule**: Option A is **accepted** iff G1–G7 all pass. It is **rejected** if any of G2, G4, G5, G6, G7 fail. G1 and G3 are necessary but not sufficient.

---

## 5. The 10 Mandatory Evaluative Primitives

Per the Recursive Self-Questioning Protocol, every claim about Option A must be evaluated against these ten primitives. I list each with the specific Option A instantiation and the failure mode it guards against.

| # | Primitive | Option A Instantiation | Guards Against |
|---|-----------|------------------------|----------------|
| 1 | **Locality** | Every micro update depends only on $[h_{i-1}, h_i, h_{i+1}]$ and $w_i$; every macro update depends only on $[M_{j-1}, M_j, M_{j+1}]$. | Global all-to-all attention. |
| 2 | **Causal necessity** | Macro lesion must degrade interior accuracy by ≥20%. | Decorative macro field. |
| 3 | **Non-degeneracy** | $\gamma w_i$ must have spatial variance ≥5% of mean magnitude. | DC bias saturation. |
| 4 | **Scale separation** | Macro clock $k$ must be >1 and macro lattice $L_M < L$. | Trivial identity hierarchy. |
| 5 | **Falsifiability** | Every gate G1–G7 has a pre-registered numerical threshold. | Post-hoc rationalization. |
| 6 | **Reversibility** | Macro lesion, parity-destroy pooling, frozen-M, random-mod must all be runnable without retraining. | Irreproducible interventions. |
| 7 | **Minimality** | No centralized GRU, no global attention, no learned positional embeddings. | Weak mimicry. |
| 8 | **Transport** | Effective receptive field radius ≤4 hops. | Global bypass via macro broadcast. |
| 9 | **Parity-specificity** | Parity-destroy pooling must not degrade accuracy by >10%. | Sum-like invariant leakage. |
| 10 | **Seed
