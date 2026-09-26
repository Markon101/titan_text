---
name: openrouter-subagents
description: >-
  Adaptive multi-agent coding and research skill with Gemini primary, DeepSeek parallel subagents,
  Codex CLI specialist, Jev System-1 structured decision governor, and lightweight telemetry.
---

# Upgraded Multi-Agent System (OpenRouter Subagents + Cognitive Governor)

This skill provides an adaptive multi-agent orchestration architecture for coding and research.
Gemini remains the primary agent and decision maker, augmented by cheap structured judgment,
independent adversarial auditing, and local code specialists.

## Agent Roles & Division of Labor

- **Gemini**: Primary planner, integrator, coder, researcher, and synthesizer. Owns the workflow.
- **DeepSeek 4.1 Flash**: Cheap parallel researcher, critic, ideation agent, alternative reasoning path, and adversarial reviewer via OpenRouter.
- **Codex CLI (`gpt-6-astra`)**: Independent coding implementation/review/debugging specialist when code-local uncertainty is high or deep defect review is needed.
- **Jev (`typesafe/jev-1.13`)**: Ultra-fast, cheap System-1 structured decision layer used for routing, uncertainty decomposition, stopping decisions, and validation gates.
- **Astra Escalation**: Available via Codex CLI (`gpt-6-astra`) for high-value ambiguity, profound architectural dilemmas, or subtle cross-domain faults.

---

## 1. Jev Structured Decisions (`scripts/jev_decide.py`)

Jev returns advisory classifications and model probability distributions. Its scores do not establish empirical support, statistical significance, calibration, or permission to act. Gemini owns routing and checks source artifacts before reporting conclusions. Actual latency and cost must come from measured calls.

### Core Decision Schemas
- `THINKING_BUDGET`: `minimal`, `normal`, `deep`, `escalate`
- `NEXT_ACTION`: `act`, `test`, `continue_reasoning`, `targeted_read`, `broad_read`, `spawn_critic`, `spawn_coder`, `escalate`, `stop`
- `UNCERTAINTY_TYPE`: `implementation`, `missing_evidence`, `conceptual`, `requirements`, `conflicting_evidence`, `low_uncertainty`
- `CRITIC_REQUIRED`: `none`, `DeepSeek`, `Codex`, `expensive_reviewer`
- `EVIDENCE_STATUS`: `supported`, `partially_supported`, `insufficient`, `contradicted`

### Invocations
```bash
# Query thinking budget
python .agents/skills/openrouter-subagents/scripts/jev_decide.py THINKING_BUDGET "Refactor parser error handling"

# Query uncertainty type
python .agents/skills/openrouter-subagents/scripts/jev_decide.py UNCERTAINTY_TYPE "Disagreement on loss masking behavior"

# Evaluate evidence status
python .agents/skills/openrouter-subagents/scripts/jev_decide.py EVIDENCE_STATUS "Claim: 99% accuracy. Evidence: logs show train pass, test unseeded."
```

In Python:
```python
from jev_decide import jev_decide

res = jev_decide("THINKING_BUDGET", "Implement cache invalidation")
print(res["decision"], res["probabilities"])
```
If Jev is unavailable or its response is malformed, the envelope marks a fallback with an empty probability distribution. Custom scientific decisions return `inconclusive`; routing defaults are heuristics only. Never interpret a fallback as model confidence. Ensembles exclude failed calls, report `partial`/`unavailable`, and cannot claim unanimity with missing trials. CLI failure/partial results exit nonzero. Repeated or differently framed calls measure model agreement, not independent scientific replication.

---

## 2. Cognitive Governor & Stopping Behavior (`scripts/cognitive_governor.py`)

The governor enforces **meaningful decision boundaries** and prevents unproductive recursive reflection.
The goal is to maximize useful/correct work per unit of compute—correctness always outranks token savings.

### Decision Boundaries
1. **Initial Task Sizing**: Check `THINKING_BUDGET` & `UNCERTAINTY_TYPE`.
2. **Post-Inspection Uncertainty**: Check if targeted read or broad read is actually necessary.
3. **Subagent Spawning**: Spawn DeepSeek or Codex only when uncertainty warrants it; no agent is spawned merely because it is available.
4. **Stopping Rule**: If consecutive reasoning passes $\ge 2$ without new empirical evidence (code edits, test runs, file reads), force `act`, `test`, or `stop`.
5. **Advisory Sanity Check**: Use `EVIDENCE_STATUS` to identify claims needing inspection. Only independently checked artifacts can support completion; a model verdict is never new evidence.

### Quick Usage
```bash
# Assess initial task
python .agents/skills/openrouter-subagents/scripts/cognitive_governor.py assess "Fix typo in tests/test_runner.rs"

# Check next step and enforce stopping heuristics
python .agents/skills/openrouter-subagents/scripts/cognitive_governor.py next "Reflected on approach twice already" --passes 2
```

---

## 3. Codex CLI Integration (`scripts/codex_cli.py`)

Calls the local non-interactive Codex CLI (`gpt-6-astra`) with focused context in read-only sandbox.

```bash
# Connectivity smoke test
python .agents/skills/openrouter-subagents/scripts/codex_cli.py smoke

# Independent code review on uncommitted changes
python .agents/skills/openrouter-subagents/scripts/codex_cli.py review-diff

# Focused implementation proposal
python .agents/skills/openrouter-subagents/scripts/codex_cli.py implement \
  "Write minimal zero-copy tensor slice conversion" \
  --file src/tensor.rs:100-160

# Debug a test failure
python .agents/skills/openrouter-subagents/scripts/codex_cli.py debug \
  "assertion failed: left == right: 42 vs 40" \
  --file src/solver.rs
```

---

## 4. DeepSeek 4.1 Flash Subagents (`scripts/subagent.py`)

Preserves fast subagent delegation for research, adversarial auditing, kernel review, and exploratory ideation.
DeepSeek 4.1 Flash is used as a **high-context research collaborator**; token economy is secondary to scientific correctness and independent reconstruction.

### Supported Roles
- `researcher`: High-context scientific synthesis and empirical analysis.
- `ideation-agent`: Unconventional conceptual search across dynamical systems, cellular automata, and synthetic worlds.
- `falsification-arbiter`: Formulate discriminating experiments between competing claims.
- `rust-audit-agent`: Audit Rust kernels in `src/` for precision, state mutation, masking, and intervention integrity.
- `dynamics-agent`: Analyze phase space, Lyapunov drift, attractors, and contractive bounds.
- `statistical-agent`: Formulate pre-registered tests, Cohen's d, bootstrap CIs, and TOST equivalence.
- `skeptical-agent` / `adversarial-reviewer`: Relentless deflationary critique; find mundane non-recurrent mechanisms.
- `experiment-designer`: Design minimal, high-information-gain discriminating tests.

```bash
# Smoke test
python .agents/skills/openrouter-subagents/scripts/subagent.py smoke

# Run adversarial review on file span
python .agents/skills/openrouter-subagents/scripts/subagent.py run \
  --role adversarial-reviewer \
  --file src/tasks.rs:250-360 \
  --task "Audit for shortcuts, negative drift, and parity locking."

# Run Rust systems kernel audit
python .agents/skills/openrouter-subagents/scripts/subagent.py run \
  --role rust-audit-agent \
  --file src/intervention.rs:1-120 \
  --task "Audit carry lesion and state transplant implementation for side effects."
```

---

## 5. Lightweight Telemetry & Rolling Optimization

Logs task complexity, budgets, tool expenditures, reviewer yields, and rework rates out-of-context to `.agents/telemetry/events.jsonl`.
Policy thresholds (`config/policy.json`) are inspectable and never silently auto-modified on small sample sizes.

```bash
# View summary statistics
python .agents/skills/openrouter-subagents/scripts/telemetry.py summary

# View recent events
python .agents/skills/openrouter-subagents/scripts/telemetry.py tail -n 5

# Analyze historical performance and propose policy adjustments
python .agents/skills/openrouter-subagents/scripts/policy_analyzer.py
```

---

## 6. Trajectory & Diminishing Returns Sensor (`scripts/trajectory_sensor.py`)

Non-causal trajectory sensor forecasting diminishing cognitive returns, rework risk, and branch explosion:
```bash
# Analyze trend of historical rework or costs
python .agents/skills/openrouter-subagents/scripts/trajectory_sensor.py trend rework_needed

# Predict rework probability for task category and budget
python .agents/skills/openrouter-subagents/scripts/trajectory_sensor.py rework-prob coding minimal
```

---

## 7. Probability Calibration Tracker (`scripts/calibration_tracker.py`)

Tracks prediction-outcome pairs in `.agents/telemetry/calibration_records.jsonl`. Record outcomes separately after an independent measurement or adjudicated review, with a required `--outcome-source` artifact reference. Never use Jev's prediction as its own outcome. Legacy records without outcome provenance remain on disk but are excluded from metrics, as are invalid distributions. Brier score and ECE are descriptive; reaching 25 samples does not prove calibration. Interpret them on held-out outcomes for the relevant decision type and task distribution.

```bash
# Generate calibration report
python .agents/skills/openrouter-subagents/scripts/calibration_tracker.py report
```

---

## 8. Shadow Policy Replay & Ablation Harness (`scripts/policy_replay.py`)

Simulates policy changes (`no_jev`, `no_critic`, `no_branching`, `no_diversity`, `no_confidence_gating`) using assumed cost multipliers and rework risks. This is sensitivity analysis, not measured counterfactual performance:

```bash
# Run shadow replay under current policy
python .agents/skills/openrouter-subagents/scripts/policy_replay.py

# Run component ablation (e.g. no_critic lesion)
python .agents/skills/openrouter-subagents/scripts/policy_replay.py --ablation no_critic
```

---

## 9. Synthetic Routing Simulation (`scripts/benchmark_suite.py`)

This script does not execute coding or research tasks. Tokens, task costs, rework, and portions of latency are modeled assumptions. Output is labeled `measurement_kind: simulation`; it cannot demonstrate controller savings or correctness improvements. Previous tables claiming measured savings or zero rework from this script were unsupported and must not be reused as empirical evidence.

```bash
# Offline simulation (default); no model calls
python .agents/skills/openrouter-subagents/scripts/benchmark_suite.py --quick

# Optional paid routing calls; task outcomes still remain simulated
python .agents/skills/openrouter-subagents/scripts/benchmark_suite.py --quick --live-routing
```

Each condition uses a fresh coordinator with temporary research state and telemetry; simulation does not alter the real research packet or frontier. The `no_jev` condition never queries Jev. Unimplemented branching/diversity ablations are not advertised by this runner. To establish a performance benefit, execute matched real tasks with independent outcome checks, measured total cost/latency, and separately reported failures.

---

## 10. Router Training Dataset Export & Uncertainty Math

Successful router decisions (excluding fallbacks) are enriched with distribution metrics (`top_1_prob`, `runner_up_prob`, `margin`, `entropy`) and exported to `.agents/telemetry/router_training_data.jsonl`:

```python
from telemetry import TelemetryTracker

tracker = TelemetryTracker("research", "complex")
# Export recorded decisions as offline router training samples
tracker.export_router_training_samples("Task prompt", reward_signal=1.0)
```

---

## Routing Guidelines Summary

| Situation | Routing Path |
| :--- | :--- |
| **Simple / high confidence task** | Gemini handles directly |
| **Code-local / implementation uncertainty** | Gemini $\to$ Codex CLI or deterministic unit tests |
| **Cheap critique, shortcut audit, or ideation** | Gemini $\to$ DeepSeek subagents |
| **Broad research or theoretical synthesis** | Gemini + DeepSeek parallel branches $\to$ Gemini synthesis |
| **Subtle cross-domain fault or high-value dilemma** | Escalation target (Codex `gpt-6-astra`) |
