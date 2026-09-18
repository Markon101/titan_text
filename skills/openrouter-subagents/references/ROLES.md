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

## Adversarial Protocol

```
Builder creates. Codex reasons. DeepSeek challenges. Grumpy Reviewer complains. Minimalist deletes. Experiments decide. Then recurse.
```

1. **Independent Evaluation**: Models must not rubber-stamp claims. DeepSeek 4.1 Flash acts as an independent adversarial auditor.
2. **Concrete Line Citations**: Every finding must reference exact code lines or mathematical equations.
3. **Cheapest Discriminating Experiment**: Reject unfalsifiable debates; immediately specify the smallest test that separates competing hypotheses.
