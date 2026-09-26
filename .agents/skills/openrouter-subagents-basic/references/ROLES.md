# OpenRouter DeepSeek 4.1 Flash Subagent Roles

This reference documents the specialized roles available in the `openrouter-subagents` skill.

## Role Catalog

| Role | Primary Purpose | Key Verification Target |
| :--- | :--- | :--- |
| `task-auditor` | Audit task generators for leakage, shortcuts, split mismatch | Answer injection, parity locks, negative drift, static dictionaries |
| `adversarial-reviewer` | Challenge headline claims and impressive results | Mundane feedforward mechanisms vs true recurrence, BPTT artifacts |
| `architectural-minimalist` | Apply Occam's razor to identify bloat | Dead code, pseudo-physics baggage (Navier-Stokes on 1D grid) |
| `code-reviewer` | Inspect code for implementation bugs | State mutations, loss masks, gradients, shape mismatches, unseeded RNGs |
| `experiment-designer` | Design discriminating controls and empirical sweeps | Minimal separating experiments, baseline controls, seed sweeps |
| `falsification-arbiter` | Compare competing claims across models/reviewers | Formulate CLAIM A, CLAIM B, WHY THEY DIFFER, DISCRIMINATING EXPERIMENT |
| `coder` | Provide minimal, robust patches and tests | Bug fixes, mask wiring, clean task implementations |
| `researcher` | Synthesize theoretical formulations and audit data | Mathematical rigor, continuum vs discrete physics, spectral claims |
| `ideation-agent` | Unconstrained scientific ideation and hypothesis generation | Counterfactual state-transplants, subspace interventions, minimal synthetic worlds |
| `dynamics-agent` | Mathematical analysis of phase space and attractors | Lyapunov exponents, energy dissipation, continuum limits, trajectory drift |
| `statistical-agent` | Quantitative measurement, hypothesis testing, effect sizes | Pre-registered CI gates, paired Cohen's d, bootstrap intervals, seed noise bands |
| `skeptical-agent` | Deflationary analysis and mundane non-recurrent explanations | Output head calibration biases, position 3 priors, boundary wrap-around |

## Adversarial Protocol & Recursive Self-Questioning

```
Builder creates. Codex reasons. DeepSeek challenges. Grumpy Reviewer complains. Minimalist deletes. Experiments decide. Then recurse.
```

1. **Independent Evaluation**: Models must not rubber-stamp claims. DeepSeek 4.1 Flash acts as an independent adversarial auditor.
2. **Concrete Line Citations**: Every finding must reference exact code lines or mathematical equations.
3. **Cheapest Discriminating Experiment**: Reject unfalsifiable debates; immediately specify the smallest test that separates competing hypotheses.

### Mandatory Recursive Self-Questioning Protocol
For major consultations, agents must not simply answer the prompt; they must recursively interrogate their own reasoning:
1. Reconstruct the problem independently from the Canonical Research-State Packet.
2. Explicitly distinguish: what the evidence establishes vs. what it does NOT establish.
3. Identify the strongest mundane / deflationary explanation.
4. Interrogate assumptions: "What am I assuming? What evidence would make this interpretation wrong?"
5. Generate unasked questions: "What question have I not been asked that should be asked?"
6. Produce 10 explicit evaluative primitives:
   - Strongest argument FOR the current interpretation.
   - Strongest argument AGAINST it.
   - Strongest mundane explanation.
   - Strongest interesting explanation.
   - Easiest experiment that could falsify it.
   - Most decisive experiment regardless of implementation cost.
   - Result that would cause the agent to substantially update its beliefs.
   - Hidden assumption most likely to invalidate the experiment.
   - Measurement we are currently missing.
   - Surprising alternative hypothesis.
7. Recursively interrogate those answers before proposing changes.
