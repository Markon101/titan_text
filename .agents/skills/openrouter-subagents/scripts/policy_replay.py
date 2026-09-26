#!/usr/bin/env python3
"""Policy Replay, Shadow Evaluation, and Architectural Ablation Harness.

Implements Parts 20 & 21 of the Cognitive Control Architecture:
- Shadow Replay: Replays historical task states against alternative policy configurations
  to measure routing divergence, estimated compute delta, and rework risk.
- Controlled Ablation Harness: Models systemic impacts of lesioning components:
  - no_jev (pure generative routing)
  - no_critic (disable adversarial reviews)
  - no_branching (force single linear path)
  - no_diversity (pure exploitation pruning)
  - no_confidence_gating (static point choice)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from telemetry import read_events, get_log_path


POLICY_FILE = Path(".agents/skills/openrouter-subagents/config/policy.json")


class PolicyReplayHarness:
    def __init__(self, policy: dict[str, Any] | None = None) -> None:
        self.policy = policy or {}

    def evaluate(self, events: list[dict[str, Any]], ablation: str | None = None) -> dict[str, Any]:
        return replay_evaluation(self.policy, events, ablation=ablation)

    def run_all_ablations(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        ablations = ["no_jev", "no_critic", "no_branching", "no_diversity", "no_confidence_gating"]
        return {
            abl: replay_evaluation(self.policy, events, ablation=abl)
            for abl in ablations
        }


def simulate_task_under_policy(
    event: dict[str, Any],
    policy: dict[str, Any],
    *,
    ablation: str | None = None,
) -> dict[str, Any]:
    """Simulate routing decision and compute cost for an event under a candidate policy."""
    orig_budget = event.get("thinking_budget", "normal")
    orig_cost = float(event.get("total_cost_usd", 0.0))
    orig_rework = bool(event.get("rework_needed"))
    category = event.get("category", "general")

    governor_cfg = policy.get("governor", {})
    min_thresh = governor_cfg.get("minimal_budget_confidence_threshold", 0.70)
    esc_thresh = governor_cfg.get("escalation_probability_threshold", 0.60)

    sim_budget = orig_budget
    sim_routing = "gemini_direct"
    sim_cost = orig_cost
    sim_rework_risk = 0.0

    # 1. Ablation overrides
    if ablation == "no_jev":
        # Without Jev, defaults to conservative normal budget
        sim_budget = "normal"
        sim_routing = "gemini_unsupervised"
        sim_cost = orig_cost * 1.35  # Token overhead from extra prompt reflection
    elif ablation == "no_critic":
        # Skip DeepSeek critic passes
        sim_routing = "gemini_direct_uncritiqued"
        sim_cost = max(0.0001, orig_cost * 0.40)
        # Rework risk rises if critic is removed
        sim_rework_risk = 0.35
    elif ablation == "no_branching":
        sim_routing = "single_linear_path"
        sim_cost = orig_cost * 0.70
        sim_rework_risk = 0.20 if sim_budget in ("deep", "escalate") else 0.05
    elif ablation == "no_diversity":
        sim_routing = "greedy_beam_pruning"
        sim_cost = orig_cost * 0.85
    elif ablation == "no_confidence_gating":
        # Static point choice without runner-up protection
        if orig_budget == "minimal" and orig_rework:
            sim_rework_risk = 0.50
    else:
        # Standard candidate policy simulation
        if orig_budget == "minimal":
            sim_routing = "gemini_direct"
        elif orig_budget == "escalate":
            sim_routing = "escalate_astra"
            sim_cost += 0.015
        elif category in ("research", "experiment"):
            sim_routing = "deepseek_parallel_research"
            sim_cost += 0.0004
        else:
            sim_routing = "gemini_direct"

    return {
        "task_id": event.get("task_id"),
        "original_budget": orig_budget,
        "simulated_budget": sim_budget,
        "original_cost": orig_cost,
        "simulated_cost": round(sim_cost, 6),
        "simulated_routing": sim_routing,
        "simulated_rework_risk": round(sim_rework_risk, 3),
        "routing_diverged": sim_routing != event.get("routing_decisions", [""])[0],
    }


def replay_evaluation(
    candidate_policy: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    ablation: str | None = None,
) -> dict[str, Any]:
    """Execute shadow replay across historical event log."""
    if not events:
        return {"status": "no_data", "evaluated_tasks": 0}

    results = []
    total_orig_cost = 0.0
    total_sim_cost = 0.0
    divergence_count = 0
    total_projected_rework_risk = 0.0

    for e in events:
        sim = simulate_task_under_policy(e, candidate_policy, ablation=ablation)
        results.append(sim)
        total_orig_cost += sim["original_cost"]
        total_sim_cost += sim["simulated_cost"]
        if sim["routing_diverged"]:
            divergence_count += 1
        total_projected_rework_risk += sim["simulated_rework_risk"]

    n = len(events)
    cost_delta = total_sim_cost - total_orig_cost
    pct_cost_change = (cost_delta / total_orig_cost * 100) if total_orig_cost > 0 else 0.0

    return {
        "status": "ok",
        "measurement_kind": "simulation",
        "empirical_performance": False,
        "limitations": "Cost multipliers and rework risks are assumed, not measured counterfactual outcomes.",
        "evaluated_tasks": n,
        "ablation": ablation or "none",
        "routing_divergence_rate": round(divergence_count / n, 3) if n else 0.0,
        "total_original_cost_usd": round(total_orig_cost, 6),
        "total_simulated_cost_usd": round(total_sim_cost, 6),
        "cost_delta_usd": round(cost_delta, 6),
        "percent_cost_change": round(pct_cost_change, 2),
        "mean_projected_rework_risk": round(total_projected_rework_risk / n, 3) if n else 0.0,
        "simulations": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy Replay & Ablation Evaluator CLI")
    parser.add_argument("--policy-file", default=str(POLICY_FILE))
    parser.add_argument("--ablation", choices=["no_jev", "no_critic", "no_branching", "no_diversity", "no_confidence_gating"])
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    events = read_events()
    policy_path = Path(args.policy_file)
    if not policy_path.is_file():
        print(f"Policy file not found: {policy_path}", file=sys.stderr)
        return 1

    policy = json.loads(policy_path.read_text())
    res = replay_evaluation(policy, events, ablation=args.ablation)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=== SYNTHETIC POLICY REPLAY (ASSUMPTIONS, NOT MEASURED EFFECTS) ===")
        print(f"Evaluated Tasks:          {res['evaluated_tasks']}")
        print(f"Ablation Condition:       {res['ablation']}")
        print(f"Routing Divergence Rate:  {res['routing_divergence_rate']:.1%}")
        print(f"Original Compute Cost:    ${res['total_original_cost_usd']:.6f} USD")
        print(f"Simulated Compute Cost:   ${res['total_simulated_cost_usd']:.6f} USD (Δ {res['percent_cost_change']:+.1f}%)")
        print(f"Mean Projected Rework:    {res['mean_projected_rework_risk']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
