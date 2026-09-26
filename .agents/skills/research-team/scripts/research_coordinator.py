#!/usr/bin/env python3
"""Master Research Team Coordinator (Principal Investigator Workflow).

Orchestrates multi-agent scientific investigations, Titan experiment designs,
falsification campaigns, and architectural evaluations.

Workflow:
1. TASK -> Decomposition -> Jev Budget & Routing Assessment
2. Routing Branches:
   - simple: Gemini direct
   - implementation: Codex CLI or deterministic tests
   - research: Parallel DeepSeek specialists
   - escalation: Codex CLI (gpt-6-astra)
3. Synthesis -> Advisory Jev Support & Uncertainty Review
   - supported: PI inspects supporting artifacts before drawing conclusions
   - implementation uncertainty: Codex / tests
   - missing evidence: Targeted research
   - disagreement: Adversarial DeepSeek
   - conceptual anomaly: Escalation
4. Final Synthesis + Explicit Remaining Uncertainty
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

# Import sibling modules from openrouter-subagents
OPENROUTER_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "openrouter-subagents" / "scripts"
sys.path.insert(0, str(OPENROUTER_SCRIPTS))

from subagent import run_subagent
from jev_decide import jev_decide
from codex_cli import run_codex_exec, is_codex_available
from telemetry import TelemetryTracker
from cognitive_governor import CognitiveGovernor

# Local imports
from research_state import ResearchState
from recursive_interrogator import run_recursive_interrogation
from reasoning_frontier import ReasoningFrontier
from ideation_engine import run_ideation_cycle
from research_meeting import run_full_research_meeting, DEFAULT_MEETING_ROLES
from independent_reconstruction import run_reconstruction_test
from context_builder import ContextBuilder


RESEARCH_ROLES: dict[str, str] = {
    "independent-investigator": (
        "You are an independent empirical investigator. "
        "Formulate testable hypotheses, examine raw logs/checkpoints, and isolate confounding variables."
    ),
    "skeptical-reviewer": (
        "You are a grumpy skeptical reviewer. Assume every positive result is an artifact or leakage "
        "until proven otherwise. Find the most mundane, deflationary non-recurrent mechanism."
    ),
    "experiment-critic": (
        "You are an empirical experiment-design critic. "
        "Audit proposed experiments for missing baseline controls, unseeded RNGs, split contamination, "
        "and compute inefficiencies."
    ),
    "counter-hypothesis-generator": (
        "You are a counter-hypothesis specialist. Given claim A, formulate distinct, competing "
        "explanations B and C that could produce the identical empirical observation."
    ),
    "failure-mode-hunter": (
        "You are a failure-mode hunter. Probe boundary conditions, sequence length generalization failures, "
        "vanishing gradient regimes, and phase transition instabilities."
    ),
    "literature-scout": (
        "You are a repository and literature scout. Identify existing precedents, canonical architectures, "
        "and established benchmarks relevant to the problem."
    ),
    "synthesis-challenger": (
        "You are a synthesis challenger. Scrutinize the final synthesis for unearned generalizations, "
        "unsupported leaps from correlation to causation, and premature convergence."
    ),
    "ideation-agent": (
        "You are an exploratory AI research ideation scientist. Propose counterfactual state-transplants, "
        "subspace interventions, and minimal synthetic worlds."
    ),
}


class ResearchCoordinator:
    def __init__(self, topic: str = "Research Campaign", *, decider_fn: Any = None, state_directory: Path | None = None) -> None:
        self.state = ResearchState(
            path=state_directory / "research_packet.json" if state_directory else None,
            archive_path=state_directory / "research_packet_archive.json" if state_directory else None,
        )
        self.frontier = ReasoningFrontier(storage_path=state_directory / "reasoning_frontier.json" if state_directory else None)
        self.tracker = TelemetryTracker(task_category="research", approx_complexity="complex")
        self.governor = CognitiveGovernor(tracker=self.tracker, decider_fn=decider_fn or jev_decide,
                                          telemetry_path=state_directory / "events.jsonl" if state_directory else None)
        self._seen_evidence_ids: set[str] = set()

    def check_and_restore_archived_context(self, text_prompt: str) -> list[str]:
        """Phase 9: Check for references to archived items and restore them into context."""
        restored = []
        item_ids = re.findall(r"\b([HQ]\d+|EXP_[a-zA-Z0-9_\-]+)\b", text_prompt)
        for item_id in item_ids:
            found = self.state.retrieve_from_archive(item_id)
            if found:
                restored.append(f"Restored archived context for {item_id}: {json.dumps(found)}")
        return restored

    def plan_task(self, task: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Phase 1: Initial decomposition, Jev budget/routing assessment, and governor branching."""
        # Auto-archive stale items if soft budget is exceeded
        budget_status = self.state.get_budget_status()
        if budget_status.get("exceeds_soft"):
            self.state.archive_stale()

        gov_res = self.governor.assess_initial_task(task)
        budget = gov_res.get("thinking_budget", "normal")
        uncert = gov_res.get("uncertainty_type", "missing_evidence")
        branch_suggestion = gov_res.get("suggested_route", "deepseek_parallel_research")

        # Evaluate branching
        active_b_count = len(self.frontier.branches)
        branching_eval = self.governor.evaluate_branching(task, uncert, active_branches_count=active_b_count)

        # Human / PI Override Model (Phase 14)
        overrides_applied: dict[str, Any] = {}
        if overrides:
            if "thinking_budget" in overrides and overrides["thinking_budget"] in ("minimal", "normal", "deep", "escalate"):
                budget = overrides["thinking_budget"]
                overrides_applied["thinking_budget"] = budget
                self.tracker.record_routing(f"override_budget_{budget}")
            if "uncertainty_type" in overrides:
                uncert = overrides["uncertainty_type"]
                overrides_applied["uncertainty_type"] = uncert
                self.tracker.record_routing(f"override_uncert_{uncert}")
            if "recommended_branch" in overrides:
                branch_suggestion = overrides["recommended_branch"]
                overrides_applied["recommended_branch"] = branch_suggestion
                self.tracker.record_routing(f"override_branch_{branch_suggestion}")
            if "should_branch" in overrides:
                branching_eval["should_branch"] = bool(overrides["should_branch"])
                branching_eval["reason"] = "Explicit Human/PI override"
                overrides_applied["should_branch"] = branching_eval["should_branch"]
                self.tracker.record_routing(f"override_branching_{branching_eval['should_branch']}")

        # Map governor suggested route to research coordinator branch convention
        if branch_suggestion == "escalate_astra":
            branch = "escalation_astra"
        elif branch_suggestion == "codex_or_tests":
            branch = "codex_implementation"
        elif branch_suggestion == "deepseek_critic":
            branch = "deepseek_parallel_research"
        elif branch_suggestion == "gemini_direct":
            branch = "gemini_direct"
        else:
            branch = branch_suggestion

        self.tracker.record_budget(budget)
        self.tracker.record_routing(f"branch_{branch}")
        self.state.add_question(f"Initial inquiry: {task}", priority="high")

        return {
            "task": task,
            "thinking_budget": budget,
            "budget_details": gov_res.get("budget_details", {}),
            "uncertainty_type": uncert,
            "uncertainty_details": gov_res.get("uncertainty_details", {}),
            "recommended_branch": branch,
            "recommended_roles": self._recommend_roles(uncert),
            "branching_evaluation": branching_eval,
            "overrides_applied": overrides_applied,
            "gating_adjusted": gov_res.get("gating_adjusted", False),
            "gating_reason": gov_res.get("gating_reason", ""),
        }

    def _recommend_roles(self, uncertainty_type: str) -> list[str]:
        if uncertainty_type == "conceptual":
            return ["counter-hypothesis-generator", "skeptical-reviewer"]
        elif uncertainty_type == "missing_evidence":
            return ["independent-investigator", "experiment-critic"]
        elif uncertainty_type == "conflicting_evidence":
            return ["skeptical-reviewer", "synthesis-challenger"]
        else:
            return ["independent-investigator", "skeptical-reviewer"]

    def run_parallel_investigators(
        self,
        question: str,
        roles: list[str],
        *,
        context: str = "",
        max_workers: int = 3,
    ) -> list[dict[str, Any]]:
        """Phase 2: Spawn focused DeepSeek specialists concurrently with auto-restored archive context."""
        restored_context = self.check_and_restore_archived_context(question + " " + context)
        combined_context = context
        if restored_context:
            combined_context = "\n[AUTO-RESTORED ARCHIVE CONTEXT]\n" + "\n".join(restored_context) + "\n\n" + context

        results = []

        def worker(r: str) -> dict[str, Any]:
            prompt = f"{question}\n\nMaintain compact, evidence-first reporting."
            res = run_subagent(
                prompt,
                role="researcher",
                context=f"Role instruction: {RESEARCH_ROLES.get(r, '')}\n\n{combined_context}",
                max_tokens=2048,
            )
            res["research_role"] = r
            return res

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, r): r for r in roles}
            for fut in as_completed(futures):
                r_name = futures[fut]
                try:
                    out = fut.result()
                    results.append(out)
                    self.tracker.record_deepseek(r_name, tokens=out.get("usage", {}).get("total_tokens"))
                except Exception as e:
                    results.append({"research_role": r_name, "status": "error", "error": str(e)})

        return results

    def verify_synthesis_gate(self, claims: str, evidence: str, *, evidence_ids: list[str] | None = None) -> dict[str, Any]:
        """Advisory review only. The PI validates artifacts and decides conclusions.

        evidence_ids are caller-verified immutable run/artifact references (including
        hashes when files can change). Repeating an ID is not new evidence. No
        calibration outcome is generated from this model's own prediction.
        """
        if evidence_ids is not None and (not isinstance(evidence_ids, list)
                or any(not isinstance(eid, str) or not eid.strip() for eid in evidence_ids)):
            raise ValueError("evidence_ids must be nonempty artifact reference strings")
        new_ids = set(evidence_ids or []) - self._seen_evidence_ids
        self._seen_evidence_ids.update(new_ids)
        state_repr = f"CLAIMS:\n{claims}\n\nEVIDENCE:\n{evidence}"
        gov_verify = self.governor.verify_evidence_status(state_repr)
        status = gov_verify.get("evidence_status", "insufficient")
        support_res = gov_verify.get("details", {})

        model_supports_claim = status in ("supported", "partially_supported")
        next_res = self.governor.check_next_step(state_repr, new_evidence_added=bool(new_ids))
        next_action = next_res.get("decision", "act")

        # Map to next research team branch
        multi_branch_test = None
        if status == "supported":
            action_branch = "inspect_supporting_artifacts"
        elif status == "contradicted":
            action_branch = "investigate_contradiction"
        elif next_action == "spawn_critic":
            action_branch = "adversarial_deepseek_audit"
        elif next_action == "escalate":
            action_branch = "escalate_to_astra"
        elif next_action == "test":
            action_branch = "execute_discriminating_experiment"
            multi_branch_test = self.get_discriminating_experiment()
        else:
            action_branch = "targeted_evidence_gathering"

        return {
            "evidence_status": status,
            "support_details": support_res,
            "next_action": next_action,
            "action_branch": action_branch,
            "is_supported": False,
            "model_supports_claim": model_supports_claim,
            "requires_empirical_verification": True,
            "advisory_only": True,
            "new_evidence_ids": sorted(new_ids),
            "discriminating_test_proposal": multi_branch_test,
            "governor_next_details": next_res,
        }

    def run_ideation(self, problem_statement: str, context: str = "") -> dict[str, Any]:
        """Phase 2b: High-temperature exploratory ideation with low-variance Jev selection."""
        res = run_ideation_cycle(problem_statement, frontier=self.frontier, context=context)
        self.tracker.record_routing("ideation_cycle_completed")
        return res

    def get_discriminating_experiment(self, top_k: int = 3) -> dict[str, Any]:
        """Phase 3b: Select top branches and formulate a multi-branch discriminating test."""
        active_branches = self.frontier.select_active_frontier(top_k=top_k, preserve_diversity=True)
        disc_test = self.frontier.find_multi_branch_discriminating_test(active_branches)
        if disc_test.get("status") == "ready":
            self.tracker.record_routing("multi_branch_test_formulated")
        return {
            "active_branches": [b.to_dict() for b in active_branches],
            "discriminating_test": disc_test,
        }

    def add_hypothesis_branch(
        self,
        hypothesis: str,
        branch_type: str = "causal_hypothesis",
        plausibility: float = 0.5,
        expected_info_gain: float = 0.5,
        novelty: float = 0.5,
        action: str | None = None,
    ) -> tuple[str, bool]:
        """Add a hypothesis to the reasoning frontier with semantic deduplication."""
        return self.frontier.add_branch(
            hypothesis,
            branch_type=branch_type,
            plausibility=plausibility,
            expected_information_gain=expected_info_gain,
            novelty_score=novelty,
            next_discriminating_action=action,
        )

    def escalate_to_astra(self, dilemma: str, context: str = "") -> dict[str, Any]:
        """Phase 4: Escalate high-value dilemma to Codex CLI (gpt-6-astra)."""
        prompt = (
            "ESCALATION: You are acting as the Astra high-level reasoning specialist. "
            "Resolve this profound architectural dilemma or conceptual anomaly with rigorous proofs and analysis:\n"
            f"{dilemma}"
        )
        res = run_codex_exec(prompt, context=context, timeout=240)
        self.tracker.record_codex("escalate_astra", tokens=res.get("tokens_used"))
        self.tracker.record_routing("astra_escalation_completed")
        return res

    def run_meeting(
        self,
        objective: str,
        roles: list[str] | None = None,
        context: str = "",
    ) -> dict[str, Any]:
        """Phase 5: Conduct multi-agent research meeting with independent & synthesis rounds."""
        meeting_roles = roles or DEFAULT_MEETING_ROLES
        res = run_full_research_meeting(objective, roles=meeting_roles, context=context)
        self.tracker.record_routing("multi_agent_meeting_completed")
        return res

    def run_independent_reconstruction(
        self,
        objective: str = "Independent Model Reconstruction of Titan Text",
        context: str = "",
    ) -> dict[str, Any]:
        """Phase 6: Audit DeepSeek model reconstruction against ground truth evidence packet."""
        res = run_reconstruction_test(objective=objective, context=context)
        self.tracker.record_routing(f"reconstruction_test_{res['audit']['grade'].lower()}")
        return res

    def run_recursive_interrogation(
        self,
        target: str,
        deep_decision: bool = False,
        depth: int = 2,
        context: str = "",
    ) -> dict[str, Any]:
        """Phase 7: Run 13-step recursive self-questioning or 11-point decision battery."""
        res = run_recursive_interrogation(
            target,
            context=context,
            max_depth=depth,
            enable_reasoning=True,
            deep_decision=deep_decision,
        )
        self.tracker.record_routing("recursive_interrogation_completed")
        return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research Team Coordinator CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # plan
    plan_p = subparsers.add_parser("plan", help="Decompose research task and query Jev routing")
    plan_p.add_argument("task", help="Research objective or question")
    plan_p.add_argument("--json", action="store_true")

    # investigate
    inv_p = subparsers.add_parser("investigate", help="Run parallel DeepSeek specialized investigators")
    inv_p.add_argument("question", help="Specific investigation question")
    inv_p.add_argument("--roles", default="independent-investigator,skeptical-reviewer", help="Comma-separated roles")
    inv_p.add_argument("--context", default="")
    inv_p.add_argument("--json", action="store_true")

    # verify
    ver_p = subparsers.add_parser("verify", help="Check evidence support gate and determine next branch")
    ver_p.add_argument("claims", help="Claims or proposed synthesis")
    ver_p.add_argument("evidence", help="Measurements, test outputs, or logs")
    ver_p.add_argument("--json", action="store_true")
    ver_p.add_argument("--evidence-id", action="append", default=[], help="PI-verified immutable run/artifact reference")

    # escalate
    esc_p = subparsers.add_parser("escalate", help="Escalate deep dilemma to Codex (gpt-6-astra)")
    esc_p.add_argument("dilemma", help="Conceptual dilemma or cross-domain contradiction")
    esc_p.add_argument("--context", default="")
    esc_p.add_argument("--json", action="store_true")

    # ideate
    ide_p = subparsers.add_parser("ideate", help="Run high-temperature ideation and low-variance selection")
    ide_p.add_argument("problem", help="Problem statement or anomaly to ideate on")
    ide_p.add_argument("--context", default="")
    ide_p.add_argument("--json", action="store_true")

    # frontier
    frt_p = subparsers.add_parser("frontier", help="View active reasoning frontier and discriminating tests")
    frt_p.add_argument("-k", "--top-k", type=int, default=3)
    frt_p.add_argument("--json", action="store_true")

    # discriminate
    dis_p = subparsers.add_parser("discriminate", help="Formulate multi-branch discriminating experiment")
    dis_p.add_argument("-k", "--top-k", type=int, default=3)
    dis_p.add_argument("--json", action="store_true")

    # meeting
    meet_p = subparsers.add_parser("meeting", help="Run multi-agent research meeting council")
    meet_p.add_argument("objective", help="Milestone objective or dilemma")
    meet_p.add_argument("--roles", default=",".join(DEFAULT_MEETING_ROLES[:4]), help="Comma-separated roles")
    meet_p.add_argument("--all-roles", action="store_true", help="Include all 7 council roles")
    meet_p.add_argument("--context", default="")
    meet_p.add_argument("--json", action="store_true")

    # reconstruct
    rec_p = subparsers.add_parser("reconstruct", help="Run independent model reconstruction audit")
    rec_p.add_argument("--objective", default="Independent Model Reconstruction of Titan Text")
    rec_p.add_argument("--context", default="")
    rec_p.add_argument("--json", action="store_true")

    # interrogate
    int_p = subparsers.add_parser("interrogate", help="Run 13-step or 11-point recursive interrogation")
    int_p.add_argument("target", help="Claim, hypothesis, or dilemma to interrogate")
    int_p.add_argument("--deep-decision", action="store_true", help="Execute 11-point deep decision battery")
    int_p.add_argument("--depth", type=int, default=2)
    int_p.add_argument("--context", default="")
    int_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    coordinator = ResearchCoordinator()

    if args.subcommand == "plan":
        plan = coordinator.plan_task(args.task)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print("=== RESEARCH TEAM: TASK PLAN ===")
            print(f"Task:              {plan['task']}")
            print(f"Thinking Budget:   {plan['thinking_budget']}")
            print(f"Uncertainty:       {plan['uncertainty_type']}")
            print(f"Recommended Branch:{plan['recommended_branch']}")
            print(f"Recommended Roles: {', '.join(plan['recommended_roles'])}")

    elif args.subcommand == "investigate":
        role_list = [r.strip() for r in args.roles.split(",") if r.strip()]
        res = coordinator.run_parallel_investigators(args.question, role_list, context=args.context)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"=== PARALLEL INVESTIGATION ({len(res)} agents completed) ===")
            for item in res:
                print(f"\n--- [{item.get('research_role', 'agent').upper()}] ---")
                print(item.get("answer", item.get("error", "No output")))

    elif args.subcommand == "verify":
        gate = coordinator.verify_synthesis_gate(args.claims, args.evidence, evidence_ids=args.evidence_id)
        if args.json:
            print(json.dumps(gate, indent=2))
        else:
            print("=== RESEARCH TEAM: SYNTHESIS GATE ===")
            print(f"Jev assessment:    {gate['evidence_status']} (advisory)")
            print(f"Action Branch:     {gate['action_branch']}")
            print("Empirical support: requires independent artifact verification")
            if gate.get("discriminating_test_proposal"):
                print(f"Discriminating Test: {json.dumps(gate['discriminating_test_proposal']['discriminating_test'], indent=2)}")

    elif args.subcommand == "escalate":
        esc = coordinator.escalate_to_astra(args.dilemma, context=args.context)
        if args.json:
            print(json.dumps(esc, indent=2))
        else:
            print(f"=== ASTRA ESCALATION [{esc.get('model', 'codex')}] ===")
            print(esc.get("answer", esc.get("error", "No output")))

    elif args.subcommand == "ideate":
        res = coordinator.run_ideation(args.problem, context=args.context)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== RESEARCH TEAM: IDEATION CYCLE ===")
            print(f"Note: {res.get('intensity_note')}")
            print(f"New branches added to frontier: {len(res.get('new_branches_added', []))}")
            for acc in res.get("accepted_details", []):
                print(f"  • [{acc.get('name')}] {acc.get('hypothesis_text')}")

    elif args.subcommand == "frontier":
        if args.json:
            print(json.dumps([b.to_dict() for b in coordinator.frontier.branches.values()], indent=2))
        else:
            print(coordinator.frontier.render_frontier_summary())

    elif args.subcommand == "discriminate":
        res = coordinator.get_discriminating_experiment(top_k=args.top_k)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== RESEARCH TEAM: DISCRIMINATING EXPERIMENT ===")
            print(json.dumps(res["discriminating_test"], indent=2))

    elif args.subcommand == "meeting":
        roles = DEFAULT_MEETING_ROLES if args.all_roles else [r.strip() for r in args.roles.split(",") if r.strip()]
        res = coordinator.run_meeting(args.objective, roles=roles, context=args.context)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== RESEARCH TEAM: MULTI-AGENT COUNCIL MEETING ===")
            print(f"Objective: {args.objective}")
            print(f"Participating: {', '.join(res['pi_synthesis']['participating_roles'])}")
            print("\nWarnings against premature conclusions:")
            for w in res["pi_synthesis"]["warnings_against_premature_conclusions"]:
                print(f"  • {w}")

    elif args.subcommand == "reconstruct":
        res = coordinator.run_independent_reconstruction(objective=args.objective, context=args.context)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            audit = res["audit"]
            print(f"=== INDEPENDENT RECONSTRUCTION AUDIT [{audit['grade']}] ===")
            print(f"Score: {audit['total_score']} | Recommendation: {audit['recommendation']}")

    elif args.subcommand == "interrogate":
        res = coordinator.run_recursive_interrogation(
            args.target,
            deep_decision=args.deep_decision,
            depth=args.depth,
            context=args.context,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== RECURSIVE INTERROGATION ===")
            print(res.get("final_output", res.get("output", "")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
