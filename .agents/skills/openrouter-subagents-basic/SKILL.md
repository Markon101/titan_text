---
name: openrouter-subagents-basic
description: >-
  Lightweight baseline OpenRouter DeepSeek 4.1 Flash subagent skill.
  Preserves minimal orchestration: Gemini as primary with direct DeepSeek 4.1 Flash subagents.
  Does not contain the adaptive governor, Codex CLI, or Jev routing machinery.
---

# OpenRouter DeepSeek 4.1 Flash Subagents (Basic / Lightweight)

This skill provides the clean, minimal baseline subagent delegation running on **DeepSeek V4.1 Flash** via OpenRouter.
It is retained as a deliberately simple, predictable version without complex multi-agent routing.

## Capabilities & Subagent Roles

The subagent engine supports specialized roles tailored for research and auditing:

- **`task-auditor`**: Detects generator leakage, trivial shortcuts, constant-output heuristics, and distribution collapse.
- **`adversarial-reviewer`**: Harshly audits headline claims and identifies mundane mechanisms masquerading as emergent phenomena.
- **`architectural-minimalist`**: Identifies dead code, pseudo-physics baggage, and unneeded abstractions.
- **`code-reviewer`**: Audits Rust and Python code for state mutation bugs, loss masks, gradients, and unseeded RNGs.
- **`experiment-designer`**: Formulates minimal high-information-gain discriminating experiments.
- **`falsification-arbiter`**: Resolves disagreements across agents and defines decisive empirical tests.
- **`coder`**: Proposes minimal, verified patches.

See [ROLES.md](./references/ROLES.md) for full role definitions and guidelines.

## Quick Invocations

The subagent engine is located at `scripts/subagent.py`.

### 1. Connectivity & Smoke Test
```bash
python .agents/skills/openrouter-subagents-basic/scripts/subagent.py smoke
```

### 2. Audit a File or Function Span
Delegate a code span to a specialized subagent:
```bash
python .agents/skills/openrouter-subagents-basic/scripts/subagent.py run \
  --role task-auditor \
  --file src/tasks.rs:250-360 \
  --task "Audit for shortcuts, negative drift, and parity locking."
```

### 3. Review Current Git Diff
Run an automated adversarial code review on active uncommitted changes:
```bash
python .agents/skills/openrouter-subagents-basic/scripts/subagent.py review-diff \
  --role adversarial-reviewer
```

### 4. Run Batch Subagent Jobs
Process a batch job definition file:
```bash
python .agents/skills/openrouter-subagents-basic/scripts/subagent.py batch \
  --file reviews/2026-09-14/deepseek-jobs.json \
  --output-dir reports/subagents
```

### 5. Python API Integration
```python
from subagent import run_subagent

result = run_subagent(
    task="Verify whether cumsum parity alternates for (sample_idx * 13 + i * 17) % 2.",
    role="task-auditor",
    enable_reasoning=True,
)
print(result["answer"])
```

## Security & Persistence
- The OpenRouter API key is permanently embedded and stored at `~/.config/openrouter/api_key` and in the environment.
- Subagent requests never log or echo raw secret keys.
