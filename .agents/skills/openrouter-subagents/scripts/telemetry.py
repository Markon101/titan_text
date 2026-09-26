#!/usr/bin/env python3
"""Lightweight Telemetry Logger for Rolling Optimization.

Logs append-friendly JSONL records of task execution, cognitive budget decisions,
routing actions, reviewer discoveries, and resource expenditures.
Telemetry is stored out-of-context at .agents/telemetry/events.jsonl.
Strictly logs metadata and metrics; never prints or stores secrets, raw tokens, or large files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid
from typing import Any


DEFAULT_LOG_PATH = Path(".agents/telemetry/events.jsonl")
DEFAULT_ROUTER_DATA_PATH = Path(".agents/telemetry/router_training_data.jsonl")


def get_log_path(override_path: str | Path | None = None) -> Path:
    if override_path:
        p = Path(override_path)
    else:
        # Check policy file if present
        policy_path = Path(".agents/skills/openrouter-subagents/config/policy.json")
        p = DEFAULT_LOG_PATH
        if policy_path.is_file():
            try:
                data = json.loads(policy_path.read_text())
                rel = data.get("telemetry", {}).get("log_path")
                if rel:
                    p = Path(rel)
            except Exception:
                pass
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


import re

SENSITIVE_KEY_SUBSTRINGS = (
    "key", "token", "auth", "secret", "password", "passwd", "pwd",
    "credential", "session", "cookie", "private"
)

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[a-zA-Z0-9_\-]{16,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{16,}", re.IGNORECASE),
]


def sanitize_value(val: Any) -> Any:
    """Recursively scrub sensitive keys and regex secrets from dicts, lists, and strings."""
    if isinstance(val, dict):
        clean_dict = {}
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(term in k_lower for term in SENSITIVE_KEY_SUBSTRINGS):
                continue
            clean_dict[k] = sanitize_value(v)
        return clean_dict
    elif isinstance(val, list):
        return [sanitize_value(item) for item in val]
    elif isinstance(val, str):
        cleaned = val
        for pat in SECRET_PATTERNS:
            cleaned = pat.sub("[REDACTED_SECRET]", cleaned)
        return cleaned
    else:
        return val


def append_event(event: dict[str, Any], log_path: Path | None = None) -> None:
    """Append a single event dictionary as a JSON line."""
    target = log_path or get_log_path()
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "event_id" not in event:
        event["event_id"] = str(uuid.uuid4())[:8]

    # Sanitize: prevent accidental secret leakage with recursive and regex scrubbing
    sanitized = sanitize_value(event)

    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitized) + "\n")


def calculate_distribution_entropy(probs: dict[str, float]) -> float:
    """Calculate Shannon entropy in nats from probability distribution."""
    if not probs:
        return 0.0
    total = sum(probs.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p in probs.values():
        p_norm = p / total
        if p_norm > 1e-12:
            entropy -= p_norm * math.log(p_norm)
    return round(entropy, 4)


def append_router_sample(sample: dict[str, Any], path: Path | None = None) -> None:
    """Append a structured router training sample to JSONL dataset."""
    target = path or DEFAULT_ROUTER_DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp_utc" not in sample:
        sample["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    sanitized = sanitize_value(sample)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitized) + "\n")


class TelemetryTracker:
    """In-memory session tracker that flushes an event record upon task completion."""

    def __init__(
        self,
        task_category: str = "general",
        approx_complexity: str = "normal",
        task_id: str | None = None,
    ) -> None:
        self.task_id = task_id or f"task_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        self.category = task_category
        self.complexity = approx_complexity
        self.start_time = datetime.now(timezone.utc)
        self.selected_budget = "normal"
        self.routing_decisions: list[str] = []
        self.jev_calls: list[dict[str, Any]] = []
        self.deepseek_calls: list[dict[str, Any]] = []
        self.codex_calls: list[dict[str, Any]] = []
        self.files_gathered: list[dict[str, Any]] = []
        self.tests_run: list[dict[str, Any]] = []
        self.rework_needed: bool = False
        self.reviewer_found_problem: bool = False
        self.escalation_materially_changed: bool = False
        self.total_cost_usd: float = 0.0

    def record_budget(self, budget: str) -> None:
        self.selected_budget = budget

    def record_routing(self, decision: str) -> None:
        self.routing_decisions.append(decision)

    def record_jev(
        self,
        decision_type: str,
        choice: str,
        *,
        probabilities: dict[str, float] | None = None,
        confidence: float = 0.0,
        cost_usd: float = 0.0,
        elapsed_seconds: float = 0.0,
        fallback: bool = False,
    ) -> None:
        top_1 = confidence
        runner_up = 0.0
        margin = 0.0
        entropy = 0.0

        if probabilities:
            sorted_probs = sorted(probabilities.values(), reverse=True)
            if sorted_probs:
                top_1 = round(sorted_probs[0], 4)
                runner_up = round(sorted_probs[1], 4) if len(sorted_probs) > 1 else 0.0
                margin = round(top_1 - runner_up, 4)
            entropy = calculate_distribution_entropy(probabilities)

        entry = {
            "type": decision_type,
            "choice": choice,
            "probabilities": probabilities,
            "confidence": confidence,
            "top_1_prob": top_1,
            "runner_up_prob": runner_up,
            "margin": margin,
            "entropy": entropy,
            "cost_usd": cost_usd,
            "elapsed_seconds": elapsed_seconds,
            "fallback": fallback,
        }
        self.jev_calls.append(entry)
        self.total_cost_usd += cost_usd

    def export_router_training_samples(
        self,
        task_prompt: str,
        reward_signal: float = 1.0,
        router_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Export each Jev routing decision from this session as a router training datapoint."""
        samples = []
        for call in self.jev_calls:
            if call.get("fallback") or not call.get("probabilities"):
                continue
            s = {
                "task_id": self.task_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task_category": self.category,
                "task_complexity": self.complexity,
                "task_prompt": task_prompt[:500],
                "decision_type": call.get("type"),
                "chosen_action": call.get("choice"),
                "probabilities": call.get("probabilities"),
                "top_1_prob": call.get("top_1_prob"),
                "runner_up_prob": call.get("runner_up_prob"),
                "margin": call.get("margin"),
                "entropy": call.get("entropy"),
                "reward_signal": reward_signal,
                "cost_usd": call.get("cost_usd", 0.0),
                "elapsed_seconds": call.get("elapsed_seconds", 0.0),
                "fallback": call.get("fallback", False),
            }
            append_router_sample(s, path=router_path)
            samples.append(s)
        return samples

    def record_deepseek(
        self,
        role: str,
        *,
        tokens: int | None = None,
        elapsed_seconds: float = 0.0,
        purpose: str = "",
    ) -> None:
        self.deepseek_calls.append({
            "role": role,
            "tokens": tokens,
            "elapsed_seconds": elapsed_seconds,
            "purpose": purpose[:80] if purpose else "",
        })

    def record_codex(
        self,
        action: str,
        *,
        tokens: int | None = None,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.codex_calls.append({
            "action": action,
            "tokens": tokens,
            "elapsed_seconds": elapsed_seconds,
        })

    def record_context(self, file_path: str, lines: int = 0, bytes_count: int = 0) -> None:
        self.files_gathered.append({
            "file": file_path,
            "lines": lines,
            "bytes": bytes_count,
        })

    def record_test(self, test_name: str, passed: bool) -> None:
        self.tests_run.append({"name": test_name, "passed": passed})

    def finalize(
        self,
        status: str = "success",
        *,
        rework_needed: bool = False,
        reviewer_found_problem: bool = False,
        escalation_materially_changed: bool = False,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        """Compile session into an event and flush to JSONL log."""
        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - self.start_time).total_seconds()
        self.rework_needed = rework_needed
        self.reviewer_found_problem = reviewer_found_problem
        self.escalation_materially_changed = escalation_materially_changed

        event = {
            "task_id": self.task_id,
            "timestamp": end_time.isoformat(),
            "category": self.category,
            "approx_complexity": self.complexity,
            "thinking_budget": self.selected_budget,
            "routing_decisions": self.routing_decisions,
            "jev_calls_count": len(self.jev_calls),
            "jev_decisions": [f"{j['type']}:{j['choice']}" for j in self.jev_calls],
            "deepseek_calls_count": len(self.deepseek_calls),
            "deepseek_roles": [d["role"] for d in self.deepseek_calls],
            "codex_calls_count": len(self.codex_calls),
            "files_gathered_count": len(self.files_gathered),
            "tests_run_count": len(self.tests_run),
            "tests_passed": sum(1 for t in self.tests_run if t["passed"]),
            "rework_needed": self.rework_needed,
            "reviewer_found_problem": self.reviewer_found_problem,
            "escalation_materially_changed": self.escalation_materially_changed,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_elapsed_seconds": round(elapsed, 2),
            "completion_status": status,
        }

        append_event(event, log_path=log_path)
        return event


def read_events(log_path: Path | None = None) -> list[dict[str, Any]]:
    target = log_path or get_log_path()
    if not target.is_file():
        return []
    events = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Telemetry CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # summary
    subparsers.add_parser("summary", help="Show summary statistics of logged task events")

    # tail
    tail_p = subparsers.add_parser("tail", help="Show the most recent logged task events")
    tail_p.add_argument("-n", "--limit", type=int, default=5, help="Number of recent events to show")

    args = parser.parse_args(argv)

    events = read_events()
    if args.subcommand == "summary":
        if not events:
            print("No telemetry events recorded yet.")
            return 0
        total = len(events)
        categories: dict[str, int] = {}
        budgets: dict[str, int] = {}
        total_jev = sum(e.get("jev_calls_count", 0) for e in events)
        total_deepseek = sum(e.get("deepseek_calls_count", 0) for e in events)
        total_codex = sum(e.get("codex_calls_count", 0) for e in events)
        rework_count = sum(1 for e in events if e.get("rework_needed"))
        reviewer_problems = sum(1 for e in events if e.get("reviewer_found_problem"))
        total_cost = sum(e.get("total_cost_usd", 0.0) for e in events)

        for e in events:
            cat = e.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            b = e.get("thinking_budget", "unknown")
            budgets[b] = budgets.get(b, 0) + 1

        print("=== TELEMETRY SUMMARY ===")
        print(f"Total Tasks:          {total}")
        print(f"Total Jev Calls:      {total_jev}")
        print(f"Total DeepSeek Calls: {total_deepseek}")
        print(f"Total Codex Calls:    {total_codex}")
        print(f"Total Cost Logged:    ${total_cost:.4f} USD")
        print(f"Rework Rate:          {rework_count}/{total} ({rework_count/total*100:.1f}%)")
        print(f"Reviewer Catch Rate:  {reviewer_problems}/{total} ({reviewer_problems/total*100:.1f}%)")
        print("\nCategories:")
        for cat, cnt in sorted(categories.items()):
            print(f"  - {cat}: {cnt}")
        print("\nThinking Budgets:")
        for b, cnt in sorted(budgets.items()):
            print(f"  - {b}: {cnt}")

    elif args.subcommand == "tail":
        if not events:
            print("No telemetry events recorded yet.")
            return 0
        recent = events[-args.limit :]
        print(f"=== LAST {len(recent)} EVENTS ===")
        for e in recent:
            print(
                f"[{e.get('timestamp', '')[:19]}] {e.get('task_id')} | Cat: {e.get('category')} | "
                f"Budget: {e.get('thinking_budget')} | Jev: {e.get('jev_calls_count')} | "
                f"DS: {e.get('deepseek_calls_count')} | Codex: {e.get('codex_calls_count')} | Status: {e.get('completion_status')}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
