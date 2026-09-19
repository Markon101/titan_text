=== [ADVERSARIAL-REVIEWER] (Model: deepseek/deepseek-v4.1-flash, Time: 156.29s) ===
# Adversarial Audit: Arm H vs Arm C — 300-Epoch Hierarchy Campaign

## 0. Framing the Audit Correctly

Before touching the four candidate mechanisms, I want to name the failure mode of the *narrative* itself. The report is written as if the interesting question is "why did coarsening fail to transport the prefix signal?" That framing already smuggles in an assumption: that there *was* a prefix signal to transport, and that Arm H's macro field was doing something dynamical that got blocked. The data do not support that framing. The honest reading is more deflationary, and I'll argue it below.

---

## 1. Adversarial Audit: What Does $d = +0.07$ Actually Mean?

### 1.1 The equality is not "near-miss," it is "null"

Paired Cohen's $d = +0.07$ across $N=5$ seeds with per-seed SDs of $4.85\%$–$7.15\%$ is not a weak effect. It is **indistinguishable from zero at any reasonable power**. With $N=5$, the standard error of the paired difference is roughly $\sigma_{\text{diff}}/\sqrt{5}$. Even if the paired differences were perfectly correlated (best case), you'd need $|\bar{d}| \gtrsim 0.9$ to reach $p<0.05$ two-sided. The observed $+0.47\%$ absolute gap is ~1/15th of that threshold.

**Implication:** Arm H and Arm C are the *same model* for practical purposes. Any claim that the macro field "partially worked" or "was on the right track" is unfalsifiable narrative. The correct prior is: **the macro pathway contributed nothing measurable to task accuracy.**

### 1.2 The slot-level data actively contradict the "passive passenger" story

Look at Slot 2 at Epoch 300:

| | Arm H | Arm C |
|---|---|---|
| Slot 2 ($q_2=11$) | **45.62%** | **54.38%** |

Arm H is *worse* than Arm C at Slot 2 by ~9 points, and Arm C's Slot 2 is *above* chance while Arm H's is *below* chance. Meanwhile Slot 0 shows the opposite: H=75.6%, C=66.3%. This is not a "passive passenger" pattern (which would predict H ≈ C everywhere). It is a **redistribution pattern**: Arm H has shifted capacity toward Slot 0 and away from Slot 2.

This is the signature of a **capacity/optimization tradeoff**, not a dynamical transport failure. The macro field is not inert — it is *actively consuming gradient budget and representational capacity* that would otherwise go to interior slots, and it is spending that budget on Slot 0 (the easiest slot, where local parity is trivially readable).

**Falsification test:** Train Arm H with the macro pathway present but with its output *detached* from the micro perception tensor (gradient stop). If Slot 0 advantage persists, the macro field is acting as a structural prior/regularizer, not a signal carrier. If Slot 0 advantage vanishes, the macro field is genuinely doing something at Slot 0 — but then you must explain why it can't do it at Slots 1–3.

### 1.3 The "identity gap" is a red herring

$G_{\text{identity}} = +7.19\%$ for *both* arms at Epoch 300 is presented as evidence of recurrence. It is not. An identity gap of this size is exactly what you'd expect from a model that has learned a **position-dependent bias** (Slot 0 is easier than Slot 2 regardless of input). The batch-state shuffle control confirms this: Slot 0 drops from 75.6% → 53.1% under shuffle (real recurrence), but interior slots show **0.0% identity gap** (no recurrence at all). So the "recurrence" is localized entirely to Slot 0, which is also the slot where local parity is trivially decodable.

**Conclusion of §1:** The macro field is not a passive passenger. It is an **active misallocation** of capacity toward the easiest subproblem. The $d=0.07$ aggregate hides a real, structured, *harmful* redistribution.

---

## 2. Dynamical & Algorithmic Analysis of the Four Candidate Mechanisms

I'll evaluate each mechanism on its merits, then rank them by explanatory power.

### Mechanism 1: Mean Pooling Destroys Parity (Linear Downsampling vs. XOR)

**Verdict: Correct in spirit, but the report states it too weakly.**

The claim "$z \oplus z' \neq (z+z')/2$" is true but trivial. The real problem is sharper:

- Parity is a **$\mathbb{Z}_2$-valued function**. Its natural representation is a sign or a phase, not a magnitude.
- Mean pooling is a **$\mathbb{R}$-linear, translation-invariant, low-pass operator**. It commutes with the DC component and annihilates high-frequency content.
- If the micro NCA encodes local parity as an antipodal pair $(+a, -a)$ (which is the natural encoding for a bounded $\tanh$ state), then $P_j = (a + (-a))/2 = 0$. **The parity signal is annihilated exactly, not attenuated.**
- If the encoding is $(+a, +a)$ for even and $(-a, -a)$ for odd (magnitude encoding), then $P_j = \pm a$, and parity survives — but only if the micro cells agree in sign, which requires them to have *already solved* the cross-cell parity problem, which is the thing you're trying to compute.

So mean pooling is either **destructive** (antipodal encoding) or **tautological** (magnitude encoding requires the answer already). There is no regime in which it helps.

**Falsification test:** Replace mean pooling with (a) a learned linear projection, (b) a signed sum $\sum_m (-1)^m z_{sj+m}$, (c) a max-pool, (d) a small MLP. If any of these unlocks interior slots, Mechanism 1 is confirmed. If none do, the problem is downstream.

### Mechanism 2: Stiff Micro Attractors + Weak Coupling ($\gamma \le 0.10$)

**Verdict: Plausible but under-specified. The report conflates two distinct failure modes.**

The claim is that micro cells have "converged to a strong local limit cycle" and a 10% perturbation can't knock them out. Two problems:

1. **The micro cells have NOT converged to a limit cycle.** The local parity probe is 98–100%, but the *cumulative* prefix probe is at chance. A limit cycle that computes local parity but not prefix parity is not a "stiff attractor" — it's a **feedforward feature detector** that has learned to read 3 bits and output a sign. That's a shallow function. It doesn't need to be "knocked out"; it needs to be *composed* with a downstream accumulator, and the architecture provides no mechanism for that composition.

2. **The coupling is not just weak — it's structurally wrong.** The macro modulation $w_i = \tilde{w}_{\lfloor i/s \rfloor} \cdot \tanh(\text{gate}) \cdot 0.10$ is a **multiplicative gate on perception**. It modulates *what the micro cell sees*, not *what the micro cell computes*. For an accumulator, you need to modulate the *transition function* (e.g., "add this bit to your state") or the *state itself* (e.g., "inject a carry"). Gating perception is the wrong intervention point.

**Falsification test:** Ablate $\gamma$ from 0.10 → 1.0 (remove the tanh·0.10 cap). If interior slots improve, Mechanism 2 is partially confirmed. If they don't, the coupling *location* is the problem, not the magnitude. Separately, try injecting the macro signal into the *state update* rather than the perception tensor.

### Mechanism 3: Macro Clock Too Short ($\tau=16, k=2 \implies 8$ steps)

**Verdict: This is a real constraint, but it is being used as an excuse.**

The report says "a disturbance at site 0 can just barely reach site 7 at step 7." This is true for a *single* propagation. But the macro field is not a passive wire — it's a recurrent dynamical system. If it were doing useful computation, it would need *multiple* passes over the same information, which 8 steps cannot provide.

However, this mechanism is **not the primary failure**. Here's why: if the macro field were doing *anything* useful, you'd see *some* signal at Slot 1 (adjacent to Slot 0) even with 8 steps. You see 45.62% — below chance. The macro field is not "running out of time"; it is **not running at all**.

**Falsification test:** Increase $\tau$ to 64 (32 macro steps) and re-run. If interior slots remain at chance, Mechanism 3 is falsified. If they improve, it's confirmed — but then you must explain why the *micro* field (which has 64 steps) also fails, since the micro field has plenty of time.

### Mechanism 4: Parity is Unnatural for Continuous Diffusion

**Verdict: This is the strongest mechanism, and the report buries it as a throwaway.**

Parity is the **worst-case function** for any continuous, local, low-pass dynamical system. Specifically:

- Parity has **maximum Fourier frequency** — every bit flip inverts the output.
- Continuous diffusion/differential operators are **low-pass filters**. They attenuate exactly the frequencies parity lives in.
- The task `iterated-parity-dense` with $\tau=16$ requires **16 sequential XOR operations**, each of which is a global sign flip. This is a **deep, non-local, high-frequency** computation.
- The micro NCA solves *local* parity (3 bits) because 3 bits is shallow enough to be learned as a feedforward feature. It fails at *cumulative* parity because cumulative parity requires **sequential composition**, which is exactly what continuous attractor dynamics are bad at.

This is not a bug in the architecture. It is a **fundamental mismatch between the task's algebraic structure ($\mathbb{Z}_2$ sequential composition) and the model's algebraic structure ($\mathbb{R}$ continuous diffusion)**.

**Falsification test:** Run the same architecture on a task with the same depth but **low-frequency** structure — e.g., iterated *majority* or iterated *sum-mod-3* with a smooth encoding. If the hierarchy succeeds there, Mechanism 4 is confirmed and the parity failure is task-specific, not architectural.

---

## 3. Gate-by-Gate Audit (G1–G7)

The report only lists G1–G4. I'll audit those and infer the missing G5–G7 from context.

| Gate | Status | Audit |
|---|---|---|
| **G1** (Interior Recurrence ≥ 55%) | FAILED | Correctly failed. Slots 1–2 at 45.6% is *below* chance, which is worse than "no signal" — it suggests the model has learned an *anti-correlated* bias at those slots. |
| **G2** (Macro Latching ≥ 70%) | FAILED | Correctly failed. Prefix parity at 45–50% is pure chance. The macro field has no latch. |
| **G3** (Anti-Saturation) | PASSED | This gate is **vacuous**. "Zero DC response" and "no explosive divergence" are necessary conditions, not evidence of function. A dead model passes G3. Do not count this as a win. |
| **G4** (Coarsening Superiority $d \ge 0.5$) | FAILED | Correctly failed. $d=0.07$ is null. |
| **G5** (presumably: Slot-0 advantage) | ? | If G5 was "Slot 0 ≥ 70%," Arm H passes (75.6%) but Arm C also passes (66.3% is close). This gate is not discriminating. |
| **G6** (presumably: State lesion collapse) | PASSED | Both arms collapse to 0% under $z \to 0$. This proves state is *used*, not that it's used *recurrently*. A feedforward model with a state buffer would also collapse. |
| **G7** (presumably: Batch-shuffle sensitivity) | PARTIAL | Slot 0 shows real recurrence (75.6% → 53.1%). Interior slots show 0.0% identity gap — **no recurrence at all**. This gate should be marked FAILED for interior slots. |

**Critical observation:** G3 and G6 are **necessary-condition gates** that any functioning model passes. They are being counted as evidence of success. This is a form of **gate inflation** — padding the pass count with gates that don't discriminate. The honest scorecard is: **0 substantive gates passed, 4 substantive gates failed.**

---

## 4. Synthesis: What Is Actually Happening?

Pulling the threads together, the most parsimonious explanation is:

1. **The micro NCA learns a shallow feedforward feature detector** for local 3-bit parity. This is easy: 3 bits is a small Boolean function, learnable by a 2-layer MLP embedded in the NCA's perception-update rule.

2. **The macro field is architecturally incapable of composing these local features into a cumulative accumulator**, for three independent reasons:
   - Mean pooling destroys the antipodal encoding of parity (Mechanism 1).
   - The coupling modulates perception, not state transitions (Mechanism 2).
   - The macro clock is too short for multi-pass composition (Mechanism 3).

3. **The macro field's only measurable effect is to redistribute capacity** — boosting Slot 0 (where local features suffice) and degrading Slot 2 (where they don't). This is a **capacity competition** effect, not a dynamical one.

4. **The task itself is adversarial to the architecture** (Mechanism 4). Parity is the worst-case function for continuous diffusion. The fact that the micro NCA solves *local* parity is a testament to the expressiveness of the perception MLP, not to the dynamical system.

**The $d=0.07$ is not a near-miss. It is the signature of a macro field that is doing nothing useful and something mildly harmful.**

---

## 5. Principled Path Forward for Titan Text

The user's mandate is **genuine emergent synthesis, no weak mimicry, no central GRU bypass**. Given the audit, here is the correct path:

### 5.1 Do not "fix" the hierarchy on parity

Parity is a **diagnostic task**, not a target task. Its value was to stress-test sequential composition. It has done its job: it revealed that the current hierarchy cannot compose. Do not spend more compute trying to make it work on parity. That would be **overfitting the architecture to a pathological task**.

### 5.2 Replace mean pooling with a parity-compatible downsampler

If you want to keep testing on parity-like tasks, the downsampler must be **$\mathbb{Z}_2$-aware**. Options:
- **Signed sum:** $P_j = \sum_m (-1)^m z_{sj+m}$ (preserves antipodal encoding).
- **Learned linear projection** with no non-negativity constraint.
- **XOR-like nonlinearity:** $P_j = \tanh(\alpha \cdot \prod_m \text{sign}(z_{sj+m}))$ — but this is fragile.

Test all three. If none unlock interior slots, the problem is not the downsampler.

### 5.3 Move the coupling from perception to state transition

The macro signal should modulate **how the micro cell updates its state**, not **what it perceives**. Concretely:
$$z_i^{t+1} = z_i^t + \Delta t \cdot f_\theta(z_i^t, \mathcal{N}(z_i^t), w_{\lfloor i/s \rfloor})$$
where $w$ enters $f_\theta$ as a **gain on the state derivative**, not as an additive bias on perception. This is the difference between "look here" and "accumulate this."

### 5.4 Test on a low-frequency sequential task first

Before returning to parity, validate the hierarchy on a task with **smooth sequential structure** — e.g., iterated *running sum* with bounded output, or iterated *majority*. If the hierarchy works there, you've isolated the failure to parity's high-frequency structure (Mechanism 4). If it fails there too, the hierarchy is broken independent of task.

### 5.5 Accept that some tasks require discrete state

If parity-like tasks are a target, the honest conclusion may be: **continuous diffusion is the wrong substrate**. A hybrid architecture with a small discrete accumulator (e.g., a 1-bit flip-flop per macro cell) would be more appropriate. This is not "weak mimicry" — it is **matching the substrate to the task's algebraic structure**. The user's mandate against "central GRU bypass" is about avoiding a monolithic recurrent core, not about forbidding discrete state.

### 5.6 Pre-register the next campaign with discriminating gates

The current gate set is inflated with necessary-condition gates (G3, G6). The next campaign should pre-register:
- **G-A:** Interior slot accuracy ≥ 60% (above chance by ≥ 10 points).
- **G-B:** Prefix parity probe ≥ 70% at *all* slots, not just Slot 0.
- **G-C:** Batch-shuffle identity gap ≥ 5% at *interior* slots (not just Slot 0).
- **G-D:** Ablation of macro pathway degrades interior slots by ≥ 5 points (proves macro is load-bearing).
- **G-E:** Downsampler ablation (mean vs. signed-sum vs. learned) shows ≥ 5-point difference (proves downsampler matters).

If G-A through G-E all fail, the hierarchy is falsified as a mechanism for sequential composition, and the correct move is to **abandon the hierarchy for parity-like tasks** rather than iterate on it.

---

## 6. Bottom Line

- **$d=0.07$ is a null result, not a near-miss.** The macro field is not a passive passenger; it is an active misallocation of capacity toward Slot 0.
- **Mechanism 4 (parity vs. continuous diffusion) is the deepest explanation**, but Mechanisms 1 and 2 are also real and independently sufficient to block composition.
- **Mechanism 3
