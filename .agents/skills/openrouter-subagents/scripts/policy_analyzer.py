#!/usr/bin/env python3
"""Policy Analyzer for Rolling Compute Optimization.

Inspects historical telemetry data from .agents/telemetry/events.jsonl, evaluates
task outcomes, reviewer yield, rework rates, and routing efficiency, and generates
principled policy revision proposals.

Does NOT silently overwrite the live policy. Emits inspectable diffs/recommendations
for human or PI review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from telemetry import read_events


POLICY_FILE = Path(".agents/skills/openrouter-subagents/config/policy.json")
MIN_TASKS_FOR_POLICY_REVISION = 10
MIN_ROLE_INVOCATIONS_FOR_PRUNING = 8


def analyze_policy_performance(events: list[dict[str, Any]], current_policy: dict[str, Any]) -> dict[str, Any]:
    if not events:
        return {"status": "no_data", "message": "No events found to analyze.", "total_tasks": 0, "recommendations": []}

    total_tasks = len(events)
    rework_count = sum(1 for e in events if e.get("rework_needed"))
    rework_rate = rework_count / total_tasks if total_tasks else 0.0

    reviewer_catches = sum(1 for e in events if e.get("reviewer_found_problem"))
    reviewer_yield = reviewer_catches / total_tasks if total_tasks else 0.0

    escalation_events = [e for e in events if e.get("thinking_budget") == "escalate" or "escalate" in str(e.get("routing_decisions"))]
    escalation_material = sum(1 for e in escalation_events if e.get("escalation_materially_changed"))
    escalation_impact_rate = (escalation_material / len(escalation_events)) if escalation_events else 0.0

    # Categorize by complexity and budget
    budget_rework: dict[str, list[bool]] = {}
    for e in events:
        b = e.get("thinking_budget", "normal")
        budget_rework.setdefault(b, []).append(bool(e.get("rework_needed")))

    recommendations = []
    thresholds = current_policy.get("governor", {})
    subagents = current_policy.get("subagents", {})

    # Minimum sample size guard: prevent thrashing policy on small n
    has_sufficient_samples = total_tasks >= MIN_TASKS_FOR_POLICY_REVISION

    # Heuristic 1: If minimal budget tasks have high rework (> 25%), suggest raising confidence threshold
    minimal_reworks = budget_rework.get("minimal", [])
    if has_sufficient_samples and len(minimal_reworks) >= 5:
        min_rework_rate = sum(minimal_reworks) / len(minimal_reworks)
        if min_rework_rate > 0.25:
            current_conf = thresholds.get("minimal_budget_confidence_threshold", 0.70)
            recommendations.append({
                "target": "governor.minimal_budget_confidence_threshold",
                "current": current_conf,
                "proposed": min(current_conf + 0.10, 0.95),
                "rationale": f"Minimal-budget tasks show high rework rate ({min_rework_rate:.1%}). Tighten confidence threshold.",
            })

    # Heuristic 2: If deep escalations rarely produce material change (< 20%), suggest raising escalation threshold
    if has_sufficient_samples and len(escalation_events) >= 5 and escalation_impact_rate < 0.20:
        current_esc = thresholds.get("escalation_probability_threshold", 0.60)
        recommendations.append({
            "target": "governor.escalation_probability_threshold",
            "current": current_esc,
            "proposed": min(current_esc + 0.10, 0.85),
            "rationale": f"Escalations rarely yielded material impact ({escalation_impact_rate:.1%}). Conserve compute.",
        })

    # Role-specific yield tracking
    role_stats: dict[str, dict[str, int]] = {}
    category_role_stats: dict[str, dict[str, dict[str, int]]] = {}
    for e in events:
        cat = e.get("category", "general")
        found = bool(e.get("reviewer_found_problem"))
        for r in e.get("deepseek_roles", []):
            role_stats.setdefault(r, {"invocations": 0, "problems_found": 0})
            role_stats[r]["invocations"] += 1
            if found:
                role_stats[r]["problems_found"] += 1

            category_role_stats.setdefault(cat, {}).setdefault(r, {"invocations": 0, "problems_found": 0})
            category_role_stats[cat][r]["invocations"] += 1
            if found:
                category_role_stats[cat][r]["problems_found"] += 1

    # Heuristic 3: Check reviewer utility
    if reviewer_yield > 0.40:
        recommendations.append({
            "target": "routing_rules.cheap_critique",
            "current": "keep active",
            "proposed": "keep active",
            "rationale": f"Adversarial reviewers have high yield ({reviewer_yield:.1%}). Retain regular critic passes.",
        })

    # Heuristic 4: Prune low-yield roles for specific categories (requires MIN_ROLE_INVOCATIONS_FOR_PRUNING)
    for cat, roles in category_role_stats.items():
        for r, data in roles.items():
            if data["invocations"] >= MIN_ROLE_INVOCATIONS_FOR_PRUNING and data["problems_found"] == 0:
                recommendations.append({
                    "target": "role_routing.prune_low_yield_role",
                    "current": f"Role '{r}' active for category '{cat}'",
                    "proposed": f"Place '{r}' on exploratory probation (sample 1 in 5 tasks) for '{cat}'",
                    "rationale": f"Role '{r}' had 0 problems caught across {data['invocations']} invocations in '{cat}' tasks. Move to probation to avoid permanent blindspot.",
                })

    return {
        "total_tasks": total_tasks,
        "rework_rate": round(rework_rate, 3),
        "reviewer_yield": round(reviewer_yield, 3),
        "role_stats": role_stats,
        "escalation_events_count": len(escalation_events),
        "escalation_impact_rate": round(escalation_impact_rate, 3),
        "recommendations": recommendations,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Telemetry & Propose Policy Revisions")
    parser.add_argument("--policy-file", default=str(POLICY_FILE))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    events = read_events()
    policy_path = Path(args.policy_file)
    if not policy_path.is_file():
        print(f"Policy file not found: {policy_path}", file=sys.stderr)
        return 1

    policy = json.loads(policy_path.read_text())
    report = analyze_policy_performance(events, policy)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=== POLICY PERFORMANCE ANALYSIS ===")
    print(f"Total Analyzed Tasks:    {report['total_tasks']}")
    print(f"Overall Rework Rate:     {report.get('rework_rate', 0.0):.1%}")
    print(f"Reviewer Catch Yield:    {report.get('reviewer_yield', 0.0):.1%}")
    print(f"Escalation Impact Rate:  {report.get('escalation_impact_rate', 0.0):.1%} ({report.get('escalation_events_count', 0)} escalations)")

    recs = report.get("recommendations", [])
    if not recs:
        print("\nPolicy Status: Current thresholds are operating within expected tolerance. No revision proposed.")
    else:
        print("\nProposed Policy Revisions (Requires human / PI confirmation):")
        for i, r in enumerate(recs, 1):
            print(f" [{i}] Target:   {r['target']}")
            print(f"     Current:  {r['current']}")
            print(f"     Proposed: {r['proposed']}")
            print(f"     Reason:   {r['rationale']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
