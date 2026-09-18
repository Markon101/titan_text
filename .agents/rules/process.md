# Multi-Agent Adversarial Process & Verification Rules

This project enforces an adversarial, evidence-driven development loop:

```
Builder creates. Codex reasons. DeepSeek challenges. Grumpy Reviewer complains. Minimalist deletes. Experiments decide. Then recurse.
```

## Agent Roles & Responsibilities

1. **Builder**: Writes models, training pipelines, and experiment harnesses. Must adhere to strict type and device correctness in Rust.
2. **Codex (Astra)**: Reasons about abstractions, refactors modules, and maintains codebase hygiene and reproducibility.
3. **DeepSeek 4.1 Flash Subagent**:
   - Acts as independent adversarial auditor via OpenRouter (`deepseek/deepseek-v4.1-flash`).
   - Run using `.agents/skills/openrouter-subagents/scripts/subagent.py` or `.codex/skills/deepseek-flash/`.
   - Audits for generator shortcuts, distribution collapse, and answer leakage.
   - Audits code diffs for silent no-ops, gradient leaks, and unmasked loss evaluations.
4. **Grumpy Reviewer**:
   - Assumes every result is an artifact until proven otherwise.
   - Checks whether dummy predictors (constant token, immediate neighbor) achieve high accuracy.
5. **Minimalist Executioner**:
   - Deletes dead code, ghost configurations, unused CLI flags, and unneeded complexity.
   - Rejects pseudo-physics baggage: no 3D hydrodynamic claims (Navier-Stokes blowup, enstrophy cascade, BKM criteria) on discrete 1D lattices.
6. **Empirical Experiments**:
   - Disagreements are settled by running the smallest authorized discriminating experiment, recording the evidence, and updating claims.

## Mandatory Validation Checklist

Before accepting any task generator or experimental claim:
- [ ] **No Dummy Shortcut**: A static predictor (majority class, constant token) must achieve $\le \text{chance accuracy}$.
- [ ] **No Local Leakage**: A model with perception radius 0 must fail on non-local tasks.
- [ ] **Proper Loss Masking**: Loss and accuracy must be computed strictly on valid target tokens (`mask == 1.0`), never on padding or query prompt tokens.
- [ ] **Reproducibility**: Experiments must be seeded and tested across random horizons or parameter sweeps.
