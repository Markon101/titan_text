# TITAN TEXT: MULTI-AGENT ADVERSARIAL PROCESS HANDBOOK

**Version**: 2.0 (Post-Audit Takeover)  
**Date**: September 2026  
**Core Motto**:  
> *Builder creates. Codex reasons. DeepSeek challenges. Grumpy Reviewer complains. Minimalist deletes. Experiments decide. Then recurse.*

---

## 1. System Philosophy & The Falsification Imperative

Titan Text explores recurrent 1D cellular sequence models. Because neural cellular automata and dynamical systems easily produce complex-looking behaviors, researchers and LLM agents are prone to narrative inflation:
- Mistaking feedforward depth specialization for "paced latent computation".
- Mistaking unmasked dummy token predictions for "high sequence accuracy".
- Mistaking deterministic cyclic schedules for "stochastic horizon robustness".
- Mistaking 1D discrete smoothing for "Millennium Prize Navier-Stokes singularity arrest".

To prevent self-deception, all development is governed by **Adversarial Process Verification**.

---

## 2. The Collaborative Agent Roles

1. **Builder (Gemini / Antigravity)**:
   - Primary code author for Rust pipelines, Candle tensors, neural cellular automata, and training loops.
   - Enforces strict memory and compute efficiency on mobile/Termux ARM64 hardware.
2. **Codex (Astra / GPT-6)**:
   - System architect and high-level reasoning agent.
   - Manages module refactoring, type safety, documentation hygiene, and git provenance.
3. **DeepSeek 4.1 Flash Subagent (OpenRouter)**:
   - Independent adversarial auditor powered by `deepseek/deepseek-v4.1-flash`.
   - Invoked via `.agents/skills/openrouter-subagents/scripts/subagent.py`.
   - Audits mathematical proofs, identifies task shortcuts, examines code diffs, and proposes falsification controls.
4. **Grumpy Reviewer**:
   - Harsh skeptic. Assumes every positive result is caused by answer leakage, distribution collapse, or trivial heuristics until proven otherwise.
5. **Minimalist Executioner**:
   - Occam's razor auditor. Deletes dead code, ghost configuration fields, and pseudo-physics decorations.
6. **Empirical Experiments**:
   - The supreme arbiter. Claims are settled only by verifiable code and data, never by model reputation or narrative assertions.

---

## 3. Subagent Skill: `openrouter-subagents`

### Location
- Workspace: `.agents/skills/openrouter-subagents/` (and mirrored in `skills/openrouter-subagents/`)
- Global: `~/.gemini/config/skills/openrouter-subagents/` and `~/.codex/skills/openrouter-subagents/`

### Configuration & API Key
The OpenRouter API key is permanently embedded and stored in:
- `~/.bashrc` & `~/.bash_profile`
- `~/.config/openrouter/api_key`
- Built-in fallback within `scripts/subagent.py`

### Common Subagent Commands
```bash
# 1. Smoke test
python .agents/skills/openrouter-subagents/scripts/subagent.py smoke

# 2. Review git diff before commit
python .agents/skills/openrouter-subagents/scripts/subagent.py review-diff --role code-reviewer

# 3. Audit a specific task generator
python .agents/skills/openrouter-subagents/scripts/subagent.py run \
  --role task-auditor \
  --file src/tasks.rs:220-318 \
  --task "Audit for shortcuts, negative drift, and parity locking."

# 4. Batch execution
python .agents/skills/openrouter-subagents/scripts/subagent.py batch \
  --file reviews/2026-09-14/deepseek-jobs.json \
  --output-dir reports/subagents
```

---

## 4. Verification Checklists Before Results Acceptance

1. **Dummy Baseline Control**:
   - Run a static dummy predictor (majority token, previous token, constant output).
   - The trained model is only considered functional if it strictly outperforms the dummy baseline on masked tokens.
2. **Ablation Against Feedforward Horizon**:
   - Horizon robustness must be evaluated on unseen steps $\tau \notin [h_{\min}, h_{\max}]$.
   - Multi-step loss and randomized unroll must be compared against fixed-horizon models.
3. **Loss Mask Rigor**:
   - Loss and accuracy must be computed **strictly** where `mask == 1.0`.
   - Never report unmasked accuracy over padding tokens.
4. **Seed Sweeps**:
   - All benchmarks must be run across multiple seeds with explicit seed arguments.
