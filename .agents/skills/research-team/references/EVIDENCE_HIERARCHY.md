# Evidence-First Hierarchy & Falsification Standards

In this research team, **empirical evidence strictly outranks model confidence, consensus, or eloquence**.

```
Deterministic Measurements > Seed Sweeps & Lesions > Deductive Inferences > Untested Hypotheses > Model Consensus
```

---

## 1. The Strict Evidence Taxonomy

Every finding, claim, and assertion must be categorized into one of five explicit tiers:

| Tier | Definition | Standard of Proof |
| :--- | :--- | :--- |
| **`[OBSERVATION]`** | Raw, unmanipulated outputs from the system. | Verifiable file paths, run IDs, raw loss values, exact stack traces, or console outputs. |
| **`[MEASUREMENT]`** | Quantified metrics derived via a documented method with baseline comparison. | Seed-averaged metrics, confidence intervals, Cohen's d effect sizes, and paired controls. |
| **`[INFERENCE]`** | Deductive or inductive conclusion derived strictly from verified measurements. | Mathematically or logically sound implications of measurements. Must cite supporting `[MEASUREMENT]`. |
| **`[HYPOTHESIS]`** | Testable, falsifiable causal proposition that has NOT yet been decisively proven. | Must state both the proposed mechanism AND the decisive experiment that could falsify it. |
| **`[SPECULATION]`** | Plausible conjecture or intuitive hypothesis without empirical support. | Must be explicitly labeled as speculative; never presented as confirmed fact. |

---

## 2. Hard Anti-Patterns

1. **Model Consensus as Truth**:
   - Multiple agents (Gemini, DeepSeek, Codex) agreeing on an interpretation is **not evidence**.
   - Consensus without deterministic measurement is merely collective speculation.

2. **Jev Probabilities as Ground Truth**:
   - Jev provides System-1 classification and consistency evaluation based on supplied text.
   - Jev probabilities are **not** statistical significance ($p$-values) or empirical confirmation.
   - A Jev verdict cannot label its own calibration outcome or reset an empirical-progress counter. Only independent measurements/review artifacts can do that.
   - Failure defaults, malformed responses, and failed ensemble members provide no scientific evidence. Synthetic controller benchmarks do not establish real performance gains.

3. **Untested Emerging Narratives**:
   - Phrases like *"the model has developed an internal world model"* or *"emergent hydrodynamics"* are forbidden without lesion tests, linear probes, and zero-radius perception controls.

4. **Missing Negative Controls**:
   - Any claim that architectural feature $X$ is responsible for performance must include an ablation where $X$ is disabled or zeroed out under identical seeds.

---

## 3. Titan-Specific Falsification Standards

For research inside the Titan ecosystem:
- **Baseline Floor**: A dummy predictor (constant token, immediate neighbor) must be evaluated. If a model fails to outperform the dummy floor, it has learned nothing.
- **Local Perception Trap**: A perception-radius-0 model must fail on non-local sequential tasks. If it succeeds, the task generator has local leakage.
- **Proper Loss Masking**: Loss and validation metrics must only be evaluated on target prediction tokens (`mask == 1.0`), never on prompt/padding tokens.
- **Reproducibility**: Claims must hold across at least 3 random seeds; single-seed "miracles" are treated as initialization noise.
