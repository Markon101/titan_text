# Research Team Subagent Roles & Responsibilities

This reference details the specialized subagent roles for the **Research Team** skill, their exact remits, anti-patterns, and recursive interrogation protocols.

---

## 1. Specialist Role Catalog

| Role | Core Mission & Expertise | Key Discriminator Target | Anti-Patterns to Avoid |
| :--- | :--- | :--- | :--- |
| **`falsification-arbiter`** | Formulate decisive discriminating tests between competing claims. Disagreements must be settled by data, not reputation. | Resolving agent deadlocks; identifying empirical separating conditions. | Declaring consensus without an empirical test; averaging opposing claims. |
| **`experiment-designer`** | Design minimal, high-information-gain experiments that cleanly separate hypotheses within compute constraints. | Baseline controls, seed sweeps, ablation metrics, causal interventions. | Proposing arbitrarily large parameter sweeps instead of minimal separating tests. |
| **`dynamics-agent`** | Analyze phase-space topologies, Lyapunov spectra, attractors, fixed points, energy dissipation, and contractive bounds. | Continuous field stability, loss explosion vs plateau, spectral radius $\rho \le 1$. | Conflating mathematical continuum analogies with discrete 1D lattice physics. |
| **`rust-audit-agent`** | Audit Rust systems implementations in `src/` for precision, state mutation, gradient detachment, causal stencil masking, and intervention integrity. | Verifying that lesions/transplants actually alter the intended physical channels. | Treating code reviews as stylistic formatting audits rather than mechanistic audits. |
| **`statistical-agent`** | Formulate rigorous pre-registered hypothesis tests, power calculations, Cohen's d effect sizes, bootstrap CIs, and TOST equivalence. | Distinguishing true signal from seed noise; detecting sample-size artifacts. | Relying on uncorrected p-values or reporting single-seed "miracles". |
| **`ideation-agent`** | Exploratory conceptual search across dynamical systems, cellular automata, neuroscience, and minimal synthetic worlds. | Novel tasks, counterfactual state-transplants, trace diagnostics, subspace isolation. | Conservative falsification or rejecting ideas merely because they are unusual. |
| **`skeptical-agent`** | Relentless deflationary critic. Assume positive results are artifacts, leakage, or bugs until proven otherwise. | Feedforward depth specialization, boundary wrap, token skew, shallow bracket domination. | Cynicism without proposing a concrete falsification measurement. |
| **`independent-investigator`** | Formulate hypotheses and isolate causal mechanisms directly from raw logs, checkpoints, and activation traces. | Hidden state carryover, leaky masks, confounding variables. | Echoing conventional wisdom; resting on model consensus. |
| **`synthesis-challenger`** | Audit final synthesis for unearned generalizations, hasty convergence, and leaps from correlation to causation. | Correlation $\to$ causation leaps, untested edge cases, cherry-picked seeds. | Rubber-stamping the Principal Investigator's conclusions. |

---

## 2. The 13-Step Recursive Interrogation Protocol

For major DeepSeek consultations, agents must not merely answer the prompt; they must recursively interrogate their own thinking:

1. **Reconstruct the problem independently** from first principles.
2. **State what the evidence establishes** (deterministic measurements, effect sizes).
3. **State what it does NOT establish** (correlations, unproven leaps).
4. **Identify the strongest alternative explanation** (most plausible competing mechanism).
5. **Ask: "What am I assuming?"** (expose every hidden axiom).
6. **Ask: "What evidence would make this interpretation wrong?"** (formulate falsification condition).
7. **Ask: "What measurement could distinguish these explanations?"** (define quantitative metric).
8. **Ask: "Could the apparent effect arise from the task, metric, dataset, architecture, initialization, optimizer, readout, stabilization mechanism, implementation, or evaluation procedure instead?"**
9. **Ask: "What question have I not been asked that should be asked?"**
10. **Answer that newly generated question.**
11. **From that answer, generate another useful question if one follows.**
12. **Continue recursively for several rounds while questions remain scientifically productive.**
13. **Only then propose experiments or implementation changes.**

---

## 3. The 11-Point Deep Decision Battery

For pivotal decisions or architectural forks, agents must explicitly produce:

1. **Strongest argument FOR** the current interpretation.
2. **Strongest argument AGAINST** it.
3. **Strongest mundane explanation** (most deflationary mechanism).
4. **Strongest interesting explanation** (most profound mechanism).
5. **Easiest experiment that could falsify it** (cheapest check).
6. **Most decisive experiment** regardless of implementation cost.
7. **Result that would cause a substantial update of beliefs** (pivot condition).
8. **Hidden assumption most likely to invalidate the experiment**.
9. **Measurement we are currently missing**.
10. **Surprising alternative hypothesis**.
11. **Question nobody in the current research loop appears to be asking**.

---

## 4. Independent Reconstruction Test

Before trusting a major recommendation, agents are audited via:
> *"Given only this evidence, reconstruct your model of Titan Text. Explain what has actually been demonstrated, what has not been demonstrated, which historical conclusions are now invalid, what the strongest remaining alternative explanations are, and what experiment would most efficiently distinguish them."*

If an agent fails to identify invalidated historical conclusions (e.g. baseline collapse artifact, shallow bracket averaging confound), its context must be reinforced before proceeding.
