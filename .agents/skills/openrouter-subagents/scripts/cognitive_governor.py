#!/usr/bin/env python3
"""Lightweight Cognitive Governor for Antigravity & Multi-Agent Orchestration.

Manages token efficiency, adaptive compute, meaningful decision boundaries,
and stopping heuristics without compromising correctness or research rigor.

Decision Boundaries:
1. Initial task sizing (THINKING_BUDGET)
2. Uncertainty classification (UNCERTAINTY_TYPE)
3. Critic requirement check (CRITIC_REQUIRED)
4. Deciding whether another reasoning pass has expected value (NEXT_ACTION + Stopping Heuristic)
5. Evidence verification gate (EVIDENCE_STATUS)

Enforces strong stopping behavior:
- Blocks unproductive recursive reflection.
- If consecutive reasoning iterations >= max_consecutive without new evidence, forces ACT, TEST, or STOP.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from jev_decide import jev_decide
from telemetry import TelemetryTracker
from trajectory_sensor import TrajectorySensor


POLICY_PATH = Path(".agents/skills/openrouter-subagents/config/policy.json")


class MarginalGainTracker:
    """Tracks incremental cognitive and empirical gains across reasoning passes.

    Guarantees stopping when consecutive passes produce no new evidence,
    hypotheses, tests, or branch eliminations.
    """

    def __init__(self, patience: int = 2, min_gain: float = 0.05) -> None:
        self.patience = patience
        self.min_gain = min_gain
        self.gain_history: list[float] = []
        self.step_records: list[dict[str, Any]] = []

    def record_step(
        self,
        new_evidence: int = 0,
        new_hypotheses: int = 0,
        new_tests: int = 0,
        branches_eliminated: int = 0,
    ) -> float:
        """Record gains from a reasoning pass and return composite gain score."""
        step_gain = (
            new_evidence * 1.0
            + new_hypotheses * 0.5
            + new_tests * 0.5
            + branches_eliminated * 0.8
        )
        self.gain_history.append(step_gain)
        self.step_records.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "new_evidence": new_evidence,
            "new_hypotheses": new_hypotheses,
            "new_tests": new_tests,
            "branches_eliminated": branches_eliminated,
            "step_gain": step_gain,
        })
        return step_gain

    def should_halt(self) -> tuple[bool, str]:
        """Check if marginal gains have stagnated over the patience window."""
        if len(self.gain_history) < self.patience:
            return False, ""
        recent = self.gain_history[-self.patience:]
        total_recent_gain = sum(recent)
        if total_recent_gain < self.min_gain:
            return (
                True,
                (
                    f"Marginal gain stagnated: sum of last {self.patience} passes is "
                    f"{total_recent_gain:.3f} < minimum required {self.min_gain:.3f}. "
                    "Halting reflection."
                ),
            )
        return False, ""

    def get_gain_history(self) -> list[float]:
        return list(self.gain_history)


def load_policy() -> dict[str, Any]:
    if POLICY_PATH.is_file():
        try:
            return json.loads(POLICY_PATH.read_text())
        except Exception:
            pass
    return {
        "models": {"jev": "typesafe/jev-1.13"},
        "governor": {
            "max_consecutive_reasoning_passes": 2,
            "escalation_probability_threshold": 0.60,
            "minimal_budget_confidence_threshold": 0.70,
        },
    }


class CognitiveGovernor:
    def __init__(self, tracker: TelemetryTracker | None = None, decider_fn: Any = None, *, telemetry_path: Path | None = None) -> None:
        self.policy = load_policy()
        self.tracker = tracker
        self.decider_fn = decider_fn or jev_decide
        self.jev_model = self.policy.get("models", {}).get("jev", "typesafe/jev-1.13")
        self.max_passes = self.policy.get("governor", {}).get("max_consecutive_reasoning_passes", 2)
        self.consecutive_reasoning_passes = 0
        self.evidence_gathered_count = 0
        patience = self.policy.get("marginal_gain", {}).get("patience", 2)
        min_gain = self.policy.get("marginal_gain", {}).get("min_required_gain", 0.05)
        self.marginal_tracker = MarginalGainTracker(patience=patience, min_gain=min_gain)
        self.trajectory_sensor = TrajectorySensor(log_path=telemetry_path)

    def assess_initial_task(self, task: str) -> dict[str, Any]:
        """Boundary 1 & 2: Sizing & Uncertainty."""
        res_budget = self.decider_fn("THINKING_BUDGET", task, model=self.jev_model)
        res_uncert = self.decider_fn("UNCERTAINTY_TYPE", task, model=self.jev_model)

        budget = res_budget.get("decision", "normal")
        uncert = res_uncert.get("decision", "low_uncertainty")

        # Dynamic Confidence Gating (Runner-Up Aware):
        min_conf_thresh = self.policy.get("governor", {}).get("minimal_budget_confidence_threshold", 0.70)
        probs = res_budget.get("probabilities") or {}
        conf = res_budget.get("confidence", 0.0)

        gating_adjusted = False
        gating_reason = ""
        # If Jev picked minimal, but confidence is shaky (< threshold) or runner-up has high mass:
        if budget == "minimal" and (conf < min_conf_thresh or probs.get("normal", 0.0) >= 0.30 or probs.get("deep", 0.0) >= 0.25 or probs.get("escalate", 0.0) >= 0.25):
            if probs.get("escalate", 0.0) >= 0.25:
                budget = "escalate"
                gating_adjusted = True
                gating_reason = f"Upgraded minimal -> escalate (shaky confidence {conf:.2f} with escalate prob {probs.get('escalate', 0.0):.2f})"
            elif probs.get("deep", 0.0) >= 0.25:
                budget = "deep"
                gating_adjusted = True
                gating_reason = f"Upgraded minimal -> deep (shaky confidence {conf:.2f} with deep prob {probs.get('deep', 0.0):.2f})"
            else:
                budget = "normal"
                gating_adjusted = True
                gating_reason = f"Upgraded minimal -> normal (confidence {conf:.2f} < {min_conf_thresh:.2f} or normal prob >= 0.30)"

        if self.tracker:
            self.tracker.record_budget(budget)
            self.tracker.record_jev(
                "THINKING_BUDGET",
                budget,
                probabilities=res_budget.get("probabilities"),
                confidence=res_budget.get("confidence", 0.0),
                cost_usd=res_budget.get("cost_usd", 0.0),
                elapsed_seconds=res_budget.get("elapsed_seconds", 0.0),
                fallback=res_budget.get("fallback_applied", False),
            )
            self.tracker.record_jev(
                "UNCERTAINTY_TYPE",
                uncert,
                probabilities=res_uncert.get("probabilities"),
                confidence=res_uncert.get("confidence", 0.0),
                cost_usd=res_uncert.get("cost_usd", 0.0),
                elapsed_seconds=res_uncert.get("elapsed_seconds", 0.0),
                fallback=res_uncert.get("fallback_applied", False),
            )

        # Recommended immediate routing
        suggested_route = "gemini_direct"
        if budget == "escalate" or res_budget.get("probabilities", {}).get("escalate", 0.0) >= self.policy.get("governor", {}).get("escalation_probability_threshold", 0.60):
            suggested_route = "escalate_astra"
        elif uncert == "implementation":
            suggested_route = "codex_or_tests"
        elif uncert in ("conceptual", "conflicting_evidence") and budget in ("normal", "deep"):
            suggested_route = "deepseek_critic"

        return {
            "thinking_budget": budget,
            "budget_details": res_budget,
            "uncertainty_type": uncert,
            "uncertainty_details": res_uncert,
            "suggested_route": suggested_route,
            "gating_adjusted": gating_adjusted,
            "gating_reason": gating_reason,
        }

    def check_next_step(
        self,
        current_state: str,
        new_evidence_added: bool = False,
        *,
        new_hypotheses: int = 0,
        new_tests: int = 0,
        branches_eliminated: int = 0,
    ) -> dict[str, Any]:
        """Boundary 4, 5, 6: Stopping behavior & next action."""
        if new_evidence_added:
            self.consecutive_reasoning_passes = 0
            self.evidence_gathered_count += 1
            self.marginal_tracker.record_step(
                new_evidence=1,
                new_hypotheses=new_hypotheses,
                new_tests=new_tests,
                branches_eliminated=branches_eliminated,
            )
        else:
            self.consecutive_reasoning_passes += 1
            self.marginal_tracker.record_step(
                new_evidence=0,
                new_hypotheses=new_hypotheses,
                new_tests=new_tests,
                branches_eliminated=branches_eliminated,
            )

        # 1. Strong stopping heuristic: if repeated reflection without evidence, force action/test/stop
        if self.consecutive_reasoning_passes > self.max_passes:
            return {
                "decision": "act",
                "reason": (
                    f"Cognitive Governor stopping rule triggered: {self.consecutive_reasoning_passes} "
                    f"consecutive reasoning passes occurred without new empirical evidence. "
                    "Halting unproductive reflection. Proceeding directly to action or testing."
                ),
                "forced_stop_applied": True,
                "confidence": 1.0,
                "probabilities": {"act": 1.0},
            }

        # 2. Trajectory diminishing returns sensor
        diminish_pred = self.trajectory_sensor.predict_diminishing_returns(
            self.consecutive_reasoning_passes,
            self.marginal_tracker.get_gain_history(),
        )

        # 3. Query Jev for NEXT_ACTION
        res = self.decider_fn("NEXT_ACTION", current_state, model=self.jev_model)
        action = res.get("decision", "act")

        # 4. If Jev suggests continuing reasoning but we are at the limit, intervene
        if action == "continue_reasoning" and self.consecutive_reasoning_passes >= self.max_passes:
            action = "test"
            res["decision"] = "test"
            res["forced_override"] = True
        elif action == "continue_reasoning" and diminish_pred.get("prob_diminishing_returns", 0.0) >= 0.85:
            action = "test"
            res["decision"] = "test"
            res["forced_override"] = True

        if self.tracker:
            self.tracker.record_jev(
                "NEXT_ACTION",
                action,
                probabilities=res.get("probabilities"),
                confidence=res.get("confidence", 0.0),
                cost_usd=res.get("cost_usd", 0.0),
                elapsed_seconds=res.get("elapsed_seconds", 0.0),
                fallback=res.get("fallback_applied", False),
            )
            self.tracker.record_routing(f"action_{action}")

        return {
            "decision": action,
            "details": res,
            "consecutive_passes": self.consecutive_reasoning_passes,
            "forced_stop_applied": res.get("forced_override", False),
            "diminishing_returns_prediction": diminish_pred,
            "marginal_gain_history": self.marginal_tracker.get_gain_history(),
        }

    def calculate_utility(
        self,
        expected_info_gain: float,
        *,
        tokens: int = 0,
        latency_sec: float = 0.0,
        cost_usd: float = 0.0,
        prob_rework: float = 0.0,
        discovery_bonus: float = 0.0,
    ) -> float:
        """Compute expected utility U(action) according to configured policy parameters."""
        u_cfg = self.policy.get("utility", {})
        lambda_tok = u_cfg.get("lambda_tokens", 0.00005)
        lambda_lat = u_cfg.get("lambda_latency", 0.05)
        lambda_cost = u_cfg.get("lambda_cost", 1.0)
        lambda_rework = u_cfg.get("lambda_rework", 0.50)
        lambda_disc = u_cfg.get("lambda_disc", 0.40)
        v_info = u_cfg.get("v_info", 1.0)

        expected_value = expected_info_gain * v_info
        penalties = (
            (lambda_tok * tokens)
            + (lambda_lat * latency_sec)
            + (lambda_cost * cost_usd)
            + (lambda_rework * prob_rework)
        )
        bonuses = lambda_disc * discovery_bonus
        return round(expected_value - penalties + bonuses, 5)

    def evaluate_branching(
        self,
        task: str,
        uncertainty_type: str,
        active_branches_count: int = 0,
    ) -> dict[str, Any]:
        """Determine whether to branch hypothesis exploration into multiple paths."""
        explosion_check = self.trajectory_sensor.predict_branch_explosion(
            active_branches_count,
            expansion_rate=2.0,
        )
        if explosion_check["needs_pruning_gate"]:
            return {
                "should_branch": False,
                "reason": (
                    f"Branch limit reached ({active_branches_count} active). "
                    f"Must prune or test before expanding further."
                ),
                "explosion_check": explosion_check,
            }

        branch_types_warranted = {"conceptual", "missing_evidence", "conflicting_evidence"}
        should_branch = uncertainty_type in branch_types_warranted

        return {
            "should_branch": should_branch,
            "reason": (
                f"Uncertainty type '{uncertainty_type}' warrants parallel branching"
                if should_branch
                else f"Uncertainty type '{uncertainty_type}' does not require branching"
            ),
            "explosion_check": explosion_check,
        }

    def verify_evidence_status(self, claims_and_evidence: str) -> dict[str, Any]:
        """Boundary 7: Final evidence sanity check."""
        res = self.decider_fn("EVIDENCE_STATUS", claims_and_evidence, model=self.jev_model)
        status = res.get("decision", "insufficient") if res.get("status") == "ok" and not res.get("fallback_applied") else "insufficient"

        if self.tracker:
            self.tracker.record_jev(
                "EVIDENCE_STATUS",
                status,
                probabilities=res.get("probabilities"),
                confidence=res.get("confidence", 0.0),
                cost_usd=res.get("cost_usd", 0.0),
                elapsed_seconds=res.get("elapsed_seconds", 0.0),
                fallback=res.get("fallback_applied", False),
            )

        return {
            "evidence_status": status,
            "details": res,
            "is_supported": False,
            "model_supports_claim": status in ("supported", "partially_supported"),
            "advisory_only": True,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cognitive Governor CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # assess
    assess_p = subparsers.add_parser("assess", help="Assess task thinking budget & uncertainty")
    assess_p.add_argument("task", help="Task prompt or description")
    assess_p.add_argument("--json", action="store_true")

    # next
    next_p = subparsers.add_parser("next", help="Evaluate next action and stopping heuristic")
    next_p.add_argument("state", help="Current progress, findings, and remaining questions")
    next_p.add_argument("--passes", type=int, default=0, help="Number of consecutive reasoning passes so far")
    next_p.add_argument("--json", action="store_true")

    # verify
    verify_p = subparsers.add_parser("verify", help="Check evidence support status")
    verify_p.add_argument("text", help="Text containing claims and empirical evidence")
    verify_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    governor = CognitiveGovernor()

    if args.subcommand == "assess":
        res = governor.assess_initial_task(args.task)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== COGNITIVE GOVERNOR: TASK ASSESSMENT ===")
            print(f"Thinking Budget:   {res['thinking_budget']}")
            print(f"Uncertainty Type:  {res['uncertainty_type']}")
            print(f"Suggested Route:   {res['suggested_route']}")

    elif args.subcommand == "next":
        governor.consecutive_reasoning_passes = args.passes
        res = governor.check_next_step(args.state)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== COGNITIVE GOVERNOR: NEXT ACTION ===")
            print(f"Next Action:       {res['decision']}")
            if res.get("reason"):
                print(f"Governor Note:     {res['reason']}")

    elif args.subcommand == "verify":
        res = governor.verify_evidence_status(args.text)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== COGNITIVE GOVERNOR: EVIDENCE STATUS ===")
            print(f"Status:            {res['evidence_status']}")
            print("Empirical support: requires independent artifact verification")

    return 0


if __name__ == "__main__":
    sys.exit(main())
