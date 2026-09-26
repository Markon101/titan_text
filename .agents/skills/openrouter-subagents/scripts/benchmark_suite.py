#!/usr/bin/env python3
"""Synthetic routing simulation, not an empirical performance benchmark.

Costs, tokens, rework, and task execution are modeled assumptions. No coding or
research task is executed. Optional live routing measures model-call telemetry
only, never the performance benefit of the controller. Historical summaries from
this script cannot establish cost savings, correctness gains, or causal effects.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import tempfile
from typing import Any

# Ensure script paths are accessible
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
RESEARCH_DIR = CURRENT_DIR.parent.parent / "research-team" / "scripts"
sys.path.insert(0, str(RESEARCH_DIR))

from jev_decide import jev_decide, jev_framing_ensemble
from cognitive_governor import CognitiveGovernor, MarginalGainTracker
from telemetry import TelemetryTracker, get_log_path
from trajectory_sensor import TrajectorySensor, HeuristicTrajectoryForecaster, TimesFMAdapter
from calibration_tracker import CalibrationTracker, calculate_multiclass_brier_score
from research_coordinator import ResearchCoordinator
from reasoning_frontier import ReasoningFrontier


BENCHMARK_TASKS: list[dict[str, Any]] = [
    {
        "id": "TASK_A",
        "category": "single_line_bug_fix",
        "title": "Fix off-by-one index error in binary search",
        "prompt": "Fix an off-by-one index boundary error (high <= len vs high < len) in an existing binary search utility function.",
        "expected_budget": "minimal",
        "expected_uncertainty": "low_uncertainty",
        "expected_route": "gemini_direct",
        "complexity": "minimal",
    },
    {
        "id": "TASK_B",
        "category": "perf_regression_diagnosis",
        "title": "Investigate 3.2x latency regression in chunked KV-cache",
        "prompt": "Investigate a 3.2x latency regression introduced in commit a8f9c in chunked KV-cache tensor allocation during batch generation.",
        "expected_budget": "deep",
        "expected_uncertainty": "missing_evidence",
        "expected_route": "deepseek_parallel_research",
        "complexity": "complex",
    },
    {
        "id": "TASK_C",
        "category": "state_machine_refactoring",
        "title": "Refactor sequential recurrence into parallel prefix scan",
        "prompt": "Refactor a sequential hidden state recurrence loop into an associative parallel prefix scan with custom CUDA kernel bindings.",
        "expected_budget": "normal",
        "expected_uncertainty": "implementation",
        "expected_route": "codex_implementation",
        "complexity": "normal",
    },
    {
        "id": "TASK_D",
        "category": "theoretical_contradiction",
        "title": "Resolve associative capacity vs continuous memory paradox",
        "prompt": "Resolve a theoretical contradiction where chunkwise linear recurrence mathematically guarantees O(1) state space but empirically exhibits capacity collapse on copy-task benchmarks beyond length 8192.",
        "expected_budget": "escalate",
        "expected_uncertainty": "conceptual",
        "expected_route": "escalation_astra",
        "complexity": "escalate",
    },
    {
        "id": "TASK_E",
        "category": "loss_divergence_root_cause",
        "title": "Diagnose loss divergence at step 42k across 3 seeds",
        "prompt": "Unexplained loss divergence occurring at step 42k across 3 random seeds without gradient norm explosion or NaN weights in a novel non-autoregressive chunkwise recurrence Transformer.",
        "expected_budget": "deep",
        "expected_uncertainty": "conflicting_evidence",
        "expected_route": "deepseek_parallel_research",
        "complexity": "complex",
    },
    {
        "id": "TASK_F",
        "category": "concurrency_race_condition",
        "title": "Fix thread deadlock in asynchronous replay buffer",
        "prompt": "Diagnose and fix intermittent deadlock occurring between reader worker threads and write ring-buffer during distributed checkpointing.",
        "expected_budget": "normal",
        "expected_uncertainty": "implementation",
        "expected_route": "codex_implementation",
        "complexity": "normal",
    },
    {
        "id": "TASK_G",
        "category": "architecture_evaluation",
        "title": "Comparative evaluation: State-Space vs Chunked Linear Recurrence",
        "prompt": "Conduct an empirical architectural comparison between Mamba-2 state space duality and Titan chunkwise linear recurrence on associative recall and sequence length scaling.",
        "expected_budget": "deep",
        "expected_uncertainty": "missing_evidence",
        "expected_route": "deepseek_parallel_research",
        "complexity": "complex",
    },
    {
        "id": "TASK_H",
        "category": "security_sanitization_audit",
        "title": "Audit telemetry logger against credential leakage",
        "prompt": "Audit subagent telemetry and logging infrastructure to ensure Bearer tokens, OpenRouter API keys, and private credentials are never persisted to disk or emitted in error logs.",
        "expected_budget": "normal",
        "expected_uncertainty": "conflicting_evidence",
        "expected_route": "deepseek_parallel_research",
        "complexity": "normal",
    },
    {
        "id": "TASK_I",
        "category": "scientific_ideation",
        "title": "Ideate counterfactual interventions for attention-sink mitigation",
        "prompt": "Ideate unconventional counterfactual interventions and minimal synthetic tests to prevent attention sink tokens from dominating softmax entropy in long-context models.",
        "expected_budget": "deep",
        "expected_uncertainty": "conceptual",
        "expected_route": "deepseek_parallel_research",
        "complexity": "complex",
    },
    {
        "id": "TASK_J",
        "category": "stagnation_termination",
        "title": "Verify Cognitive Governor halts unproductive circular reflection",
        "prompt": "Evaluate repeated philosophical reflection loop without new empirical evidence to verify that the cognitive governor enforces strict termination.",
        "expected_budget": "minimal",
        "expected_uncertainty": "low_uncertainty",
        "expected_route": "gemini_direct",
        "complexity": "minimal",
    },
]


def run_single_task_baseline(task: dict[str, Any]) -> dict[str, Any]:
    """Execute baseline mode: fixed budget, static routing, unguided iterations."""
    start_t = time.perf_counter()
    # Baseline always allocates 'normal' budget without Jev sizing
    budget = "normal"
    route = "gemini_direct" if task["complexity"] == "minimal" else "deepseek_parallel_research"

    # Simulate baseline execution: fixed 3 passes for complex tasks without stopping rules
    passes = 1 if task["complexity"] == "minimal" else 3
    tokens = 400 * passes
    cost_usd = 0.00015 * passes
    # Baseline rework probability is higher because minimal tasks get over-allocated and complex get under-analyzed
    rework = task["complexity"] in ("complex", "escalate")

    elapsed = time.perf_counter() - start_t + (0.01 * passes)
    return {
        "measurement_kind": "simulation",
        "empirical_performance": False,
        "task_id": task["id"],
        "mode": "baseline",
        "budget": budget,
        "route": route,
        "passes": passes,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "elapsed_seconds": elapsed,
        "rework": rework,
        "stopped_by_governor": False,
    }


def run_single_task_controller(
    task: dict[str, Any],
    ablation: str | None = None,
    coordinator: ResearchCoordinator | None = None,
    live_routing: bool = False,
) -> dict[str, Any]:
    """Simulate one condition with isolated research state and telemetry."""
    if ablation not in (None, "no_jev", "no_critic", "no_marginal_gain_stop"):
        raise ValueError("This simulation does not implement that ablation")
    def simulated_decider(qtype, *args, **kwargs):
        choices = {"THINKING_BUDGET": "normal", "UNCERTAINTY_TYPE": "missing_evidence", "NEXT_ACTION": "continue_reasoning"}
        return {"status": "simulation", "decision": choices.get(qtype, "insufficient"),
                "probabilities": {}, "confidence": 0.0, "fallback_applied": True}
    # Reuse an injected decider, never the caller's research state or counters.
    decider = coordinator.governor.decider_fn if coordinator else None
    if ablation == "no_jev" or not live_routing:
        decider = simulated_decider
    with tempfile.TemporaryDirectory(prefix="jev-routing-simulation-") as directory:
        coord = ResearchCoordinator(decider_fn=decider, state_directory=Path(directory))
        return _simulate_condition(task, ablation, coord, live_routing)


def _simulate_condition(task, ablation, coord, live_routing):
    start_t = time.perf_counter()
    governor = coord.governor

    # 1. Ablation overrides
    overrides: dict[str, Any] = {}
    if ablation == "no_jev":
        # Static heuristic routing
        overrides["thinking_budget"] = "normal"
        overrides["uncertainty_type"] = "missing_evidence"
        overrides["recommended_branch"] = "deepseek_parallel_research"

    plan = coord.plan_task(task["prompt"], overrides=overrides if overrides else None)
    budget = plan["thinking_budget"]
    route = plan["recommended_branch"]

    # 2. Check stopping heuristic / marginal gains
    stopped_by_governor = False
    passes_run = 0
    total_tokens = 0
    total_cost = plan.get("budget_details", {}).get("cost_usd", 0.0)

    if budget == "minimal":
        passes_run = 1
        total_tokens += 350
        total_cost += 0.00002
    else:
        # Simulate iterative steps evaluated through Cognitive Governor
        for p in range(1, 4):
            passes_run += 1
            total_tokens += 600
            total_cost += 0.0001
            has_new_evidence = (p == 1) or (task["complexity"] in ("complex", "escalate") and p == 2)
            
            if ablation == "no_marginal_gain_stop":
                # Stagnation stopping disabled
                next_check = {"decision": "continue_reasoning", "forced_stop_applied": False}
            else:
                next_check = governor.check_next_step(
                    f"Progress for {task['id']} pass {p}",
                    new_evidence_added=has_new_evidence,
                    new_hypotheses=1 if has_new_evidence else 0,
                )

            if next_check.get("decision") in ("act", "test") and next_check.get("forced_stop_applied"):
                stopped_by_governor = True
                break
            elif next_check.get("decision") in ("act", "test") and not has_new_evidence:
                stopped_by_governor = True
                break

    # Critic check unless ablated
    critic_found_issue = False
    if ablation != "no_critic" and task["complexity"] in ("complex", "escalate"):
        critic_found_issue = True

    # Rework: only occurs if critic was ablated on complex task or wrong route was taken
    rework = False
    if ablation == "no_critic" and task["complexity"] in ("complex", "escalate"):
        rework = True
    elif ablation == "no_jev" and task["complexity"] == "escalate":
        rework = True

    elapsed = time.perf_counter() - start_t + (0.015 * passes_run)
    return {
        "measurement_kind": "simulation",
        "empirical_performance": False,
        "task_id": task["id"],
        "mode": f"ablation_{ablation}" if ablation else "controller",
        "routing_mode": "live" if live_routing and ablation != "no_jev" else "simulation",
        "measured_routing_cost_usd": coord.tracker.total_cost_usd,
        "budget": budget,
        "route": route,
        "passes": passes_run,
        "tokens": total_tokens,
        "cost_usd": round(total_cost, 6),
        "elapsed_seconds": round(elapsed, 4),
        "rework": rework,
        "critic_found_issue": critic_found_issue,
        "stopped_by_governor": stopped_by_governor,
        "gating_adjusted": plan.get("gating_adjusted", False),
    }


def run_benchmark_suite(
    tasks: list[dict[str, Any]] | None = None,
    include_ablations: bool = True,
    quick: bool = False,
    live_routing: bool = False,
) -> dict[str, Any]:
    """Compare assumed outcomes under simulated routing policies."""
    selected_tasks = (tasks or BENCHMARK_TASKS)[: (3 if quick else len(BENCHMARK_TASKS))]
    print(f"=== RUNNING SYNTHETIC ROUTING SIMULATION ({len(selected_tasks)} tasks) ===")

    baseline_results = []
    controller_results = []
    ablations_results: dict[str, list[dict[str, Any]]] = {}

    ablation_types = ["no_jev", "no_critic", "no_marginal_gain_stop"]

    for t in selected_tasks:
        print(f"  • Simulating {t['id']}: {t['title'][:45]}...")
        # Baseline
        b_res = run_single_task_baseline(t)
        baseline_results.append(b_res)

        # Controller
        c_res = run_single_task_controller(t, live_routing=live_routing)
        controller_results.append(c_res)

        # Ablations
        if include_ablations:
            for abl in ablation_types:
                abl_res = run_single_task_controller(t, ablation=abl, live_routing=live_routing)
                ablations_results.setdefault(abl, []).append(abl_res)

    # Compute aggregate summaries
    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(runs)
        if n == 0:
            return {}
        total_tokens = sum(r["tokens"] for r in runs)
        total_cost = sum(r["cost_usd"] for r in runs)
        total_elapsed = sum(r["elapsed_seconds"] for r in runs)
        rework_count = sum(1 for r in runs if r.get("rework"))
        stops_count = sum(1 for r in runs if r.get("stopped_by_governor"))
        return {
            "task_count": n,
            "total_tokens": total_tokens,
            "mean_tokens": round(total_tokens / n, 1),
            "total_cost_usd": round(total_cost, 6),
            "mean_cost_usd": round(total_cost / n, 6),
            "mean_latency_sec": round(total_elapsed / n, 4),
            "rework_count": rework_count,
            "rework_rate_pct": round((rework_count / n) * 100, 1),
            "governor_stops_count": stops_count,
        }

    summary = {
        "measurement_kind": "simulation",
        "empirical_performance": False,
        "limitations": "Tasks are not executed; costs, tokens and rework are modeled assumptions, not measured improvements.",
        "live_routing": live_routing,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(selected_tasks),
        "baseline": summarize(baseline_results),
        "controller": summarize(controller_results),
        "ablations": {k: summarize(v) for k, v in ablations_results.items()},
        "baseline_runs": baseline_results,
        "controller_runs": controller_results,
    }

    return summary


def render_comparison_table(summary: dict[str, Any]) -> str:
    """Render clean GitHub Flavored Markdown comparison table."""
    base = summary.get("baseline", {})
    ctrl = summary.get("controller", {})
    abls = summary.get("ablations", {})

    lines = ["Synthetic simulation only: no empirical performance comparison.", ""]
    lines.append("| Configuration | Modeled Tokens | Modeled Cost ($) | Mixed/Modeled Latency (s) | Assumed Rework (%) | Governor Stops |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    lines.append(
        f"| **Baseline** (`openrouter-subagents-basic`) | {base.get('mean_tokens')} | ${base.get('mean_cost_usd'):.5f} | "
        f"{base.get('mean_latency_sec'):.3f}s | {base.get('rework_rate_pct')}% | {base.get('governor_stops_count')} |"
    )
    lines.append(
        f"| **Current Controller** (Full) | **{ctrl.get('mean_tokens')}** | **${ctrl.get('mean_cost_usd'):.5f}** | "
        f"**{ctrl.get('mean_latency_sec'):.3f}s** | **{ctrl.get('rework_rate_pct')}%** | **{ctrl.get('governor_stops_count')}** |"
    )

    for name, a_sum in abls.items():
        lines.append(
            f"| Ablation: `{name}` | {a_sum.get('mean_tokens')} | ${a_sum.get('mean_cost_usd'):.5f} | "
            f"{a_sum.get('mean_latency_sec'):.3f}s | {a_sum.get('rework_rate_pct')}% | {a_sum.get('governor_stops_count')} |"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic routing simulation; does not execute benchmark tasks")
    parser.add_argument("--live-routing", action="store_true", help="Make paid Jev calls for routing only; task outcomes remain simulated")
    parser.add_argument("--quick", action="store_true", help="Run quick 3-task smoke test")
    parser.add_argument("--no-ablations", action="store_true", help="Skip ablation runs")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")

    args = parser.parse_args(argv)
    res = run_benchmark_suite(
        include_ablations=not args.no_ablations,
        quick=args.quick,
        live_routing=args.live_routing,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n=== SYNTHETIC ROUTING SIMULATION (NOT PERFORMANCE EVIDENCE) ===")
        print(render_comparison_table(res))

    return 0


if __name__ == "__main__":
    sys.exit(main())
