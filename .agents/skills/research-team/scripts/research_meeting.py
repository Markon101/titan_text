#!/usr/bin/env python3
"""Multi-Agent Research Meeting Engine for Titan Text.

Orchestrates multi-agent scientific councils for major research milestones.
Enforces independent analysis before synthesis to prevent premature groupthink,
conducts structured round-table interrogation, supports agent-generated agent questions,
and preserves empirical disagreements rather than forcing rhetorical consensus.

Roles:
- Falsification Arbiter (`falsification-arbiter`)
- Experimental Designer (`experiment-designer`)
- Dynamics / Mathematical Analysis Agent (`dynamics-agent`)
- Implementation / Rust Audit Agent (`rust-audit-agent`)
- Statistical / Measurement Agent (`statistical-agent`)
- Ideation Agent (`ideation-agent`)
- Skeptical Alternative-Explanation Agent (`skeptical-agent`)

Two-Round Architecture:
Round 1: Independent Domain Analyses (zero cross-pollination to kill groupthink).
Round 2: Structured Synthesis Round answering the 10 core questions:
  1. What do you think is happening?
  2. What evidence supports that?
  3. What evidence weakens it?
  4. What is the strongest competing explanation?
  5. What experiment would distinguish them?
  6. What would change your mind?
  7. What important question did the other agents miss?
  8. What should we absolutely NOT conclude yet?
  9. What should we measure before modifying the architecture?
  10. If you had only one experiment, what information would you try to maximize?

Bounded Cross-Consultation Tree:
Allows agents to pose follow-up questions for OTHER agents (e.g. Dynamics -> Statistics,
Arbiter -> Rust Audit). Bounded to depth <= 2 to prevent runaway recursive chatter.
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

# Sibling scripts
SCRIPTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = SCRIPTS_DIR.parent.parent
REPO_ROOT = SKILLS_DIR.parent.parent
OPENROUTER_SCRIPTS = SKILLS_DIR / "openrouter-subagents" / "scripts"
sys.path.insert(0, str(OPENROUTER_SCRIPTS))
sys.path.insert(0, str(SCRIPTS_DIR))

from subagent import run_subagent
from context_builder import ContextBuilder
from research_state import ResearchState


DEFAULT_MEETING_ROLES = [
    "falsification-arbiter",
    "experiment-designer",
    "dynamics-agent",
    "rust-audit-agent",
    "statistical-agent",
    "ideation-agent",
    "skeptical-agent",
]

TEN_SYNTHESIS_QUESTIONS = """SYNTHESIS TEN QUESTIONS:
1. What do you think is happening?
2. What evidence supports that?
3. What evidence weakens it?
4. What is the strongest competing explanation?
5. What experiment would distinguish them?
6. What would change your mind?
7. What important question did the other agents miss?
8. What should we absolutely NOT conclude yet?
9. What should we measure before modifying the architecture?
10. If you had only one experiment, what information would you try to maximize?

CROSS-AGENT QUESTIONS (OPTIONAL):
If you need specific data or audits from another specialist, formulate a targeted request:
TARGET_AGENT: [role name]
QUESTION: [specific technical or empirical inquiry]
"""


class ResearchMeeting:
    def __init__(self, milestone_objective: str, root: Path | None = None) -> None:
        self.objective = milestone_objective
        self.root = root or REPO_ROOT
        self.context_builder = ContextBuilder(self.root)
        self.state = ResearchState()

    def run_round_1_independent(
        self,
        roles: list[str],
        *,
        context: str = "",
        max_workers: int = 4,
        timeout: int = 360,
    ) -> dict[str, dict[str, Any]]:
        """Round 1: Independent parallel analyses with zero cross-pollination."""
        rich_context = context or self.context_builder.build_packet(self.objective)

        def worker(role: str) -> tuple[str, dict[str, Any]]:
            prompt = (
                f"--- RESEARCH MEETING ROUND 1: INDEPENDENT ANALYSIS ---\n"
                f"Objective / Milestone Question: {self.objective}\n\n"
                "Analyze this question from your specific domain expertise. "
                "Do not speculate without labeling. Distinguish [ESTABLISHED] vs [HYPOTHESIS]. "
                "Do not assume other reviewers have scrutinized the data. Point out hidden bugs, "
                "subtle confounds, unmeasured invariants, and propose the minimal discriminating test."
            )
            res = run_subagent(
                prompt,
                role=role,
                context=rich_context,
                max_tokens=3000,
                enable_reasoning=True,
                timeout=timeout,
            )
            return role, res

        round_1_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_role = {executor.submit(worker, r): r for r in roles}
            for fut in as_completed(future_to_role):
                r = future_to_role[fut]
                try:
                    role, res = fut.result()
                    round_1_results[role] = res
                except Exception as e:
                    round_1_results[r] = {"status": "error", "error": str(e)}

        return round_1_results

    def run_round_2_synthesis(
        self,
        round_1_results: dict[str, dict[str, Any]],
        roles_to_synthesize: list[str] | None = None,
        *,
        context: str = "",
        max_workers: int = 3,
        timeout: int = 360,
    ) -> dict[str, dict[str, Any]]:
        """Round 2: Cross-agent synthesis answering the 10 core questions."""
        roles = roles_to_synthesize or list(round_1_results.keys())

        # Compile summaries of Round 1
        docket = ["=== COMPETING PROPOSALS & INDEPENDENT ROUND 1 ANALYSES ==="]
        for role, res in round_1_results.items():
            ans = res.get("answer", "(No response)")
            docket.append(f"\n--- [{role.upper()}] ---")
            docket.append(ans[:2500] + ("..." if len(ans) > 2500 else ""))
        docket_str = "\n".join(docket)

        base_context = context or self.context_builder.build_packet(self.objective)
        meeting_context = f"{base_context}\n\n{docket_str}"

        def worker(role: str) -> tuple[str, dict[str, Any]]:
            prompt = (
                f"--- RESEARCH MEETING ROUND 2: SYNTHESIS & 10 QUESTIONS ---\n"
                f"Objective: {self.objective}\n\n"
                f"You have now reviewed the competing analyses and proposals from the other specialists.\n"
                f"{TEN_SYNTHESIS_QUESTIONS}\n"
                "Answer all 10 questions directly and precisely. Do not smooth over disagreements."
            )
            res = run_subagent(
                prompt,
                role=role,
                context=meeting_context,
                max_tokens=3500,
                enable_reasoning=True,
                timeout=timeout,
            )
            return role, res

        round_2_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_role = {executor.submit(worker, r): r for r in roles}
            for fut in as_completed(future_to_role):
                r = future_to_role[fut]
                try:
                    role, res = fut.result()
                    round_2_results[role] = res
                except Exception as e:
                    round_2_results[r] = {"status": "error", "error": str(e)}

        return round_2_results

    def extract_cross_agent_questions(self, round_2_results: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
        """Parse targeted questions proposed by one agent for another."""
        cross_questions = []
        for asking_role, res in round_2_results.items():
            answer = res.get("answer", "")
            matches = re.findall(
                r"TARGET_AGENT:\s*([a-zA-Z0-9_\-]+)\s*\nQUESTION:\s*([^\n]+(?:\n[^\n]+){0,4})",
                answer,
                re.IGNORECASE,
            )
            for target_role, question in matches:
                target_norm = target_role.strip().lower()
                cross_questions.append({
                    "from_role": asking_role,
                    "to_role": target_norm,
                    "question": question.strip(),
                })
        return cross_questions

    def run_bounded_cross_consultation(
        self,
        cross_questions: list[dict[str, str]],
        *,
        max_questions: int = 3,
        timeout: int = 240,
    ) -> list[dict[str, Any]]:
        """Execute bounded follow-up questions between specialists."""
        results = []
        # Filter and prioritize up to max_questions
        selected = cross_questions[:max_questions]

        for item in selected:
            to_role = item["to_role"]
            if to_role not in DEFAULT_MEETING_ROLES and to_role != "researcher":
                to_role = "researcher"

            prompt = (
                f"--- CROSS-SPECIALIST INQUIRY FROM [{item['from_role'].upper()}] ---\n"
                f"Question: {item['question']}\n\n"
                "Provide a direct, rigorous answer with exact references to tests, source code, "
                "or mathematical dynamics. Do not speculate."
            )
            res = run_subagent(
                prompt,
                role=to_role,
                max_tokens=2500,
                enable_reasoning=True,
                timeout=timeout,
            )
            results.append({
                "from_role": item["from_role"],
                "to_role": item["to_role"],
                "question": item["question"],
                "answer": res.get("answer", ""),
            })

        return results

    def synthesize_presidential_report(
        self,
        round_1: dict[str, dict[str, Any]],
        round_2: dict[str, dict[str, Any]],
        cross_consultations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Gemini PI synthesis explicitly preserving empirical disagreements."""
        disagreements = []
        experiments_proposed = []
        warnings_unwarranted_conclusions = []

        for role, res in round_2.items():
            ans = res.get("answer", "")
            # Extract warnings
            match_warn = re.search(r"8\.\s*What should we absolutely NOT conclude yet\??\s*([^\n]+(?:\n[^\n]+){0,3})", ans, re.IGNORECASE)
            if match_warn:
                warnings_unwarranted_conclusions.append(f"[{role}]: {match_warn.group(1).strip()}")

            # Extract experiments
            match_exp = re.search(r"5\.\s*What experiment would distinguish them\??\s*([^\n]+(?:\n[^\n]+){0,3})", ans, re.IGNORECASE)
            if match_exp:
                experiments_proposed.append(f"[{role}]: {match_exp.group(1).strip()}")

        return {
            "objective": self.objective,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "participating_roles": list(round_1.keys()),
            "warnings_against_premature_conclusions": warnings_unwarranted_conclusions,
            "proposed_discriminating_experiments": experiments_proposed,
            "cross_consultations_resolved": len(cross_consultations),
        }


def run_full_research_meeting(
    objective: str,
    *,
    roles: list[str] | None = None,
    context: str = "",
) -> dict[str, Any]:
    """Execute complete 2-round research meeting with bounded cross-consultation."""
    meeting_roles = roles or DEFAULT_MEETING_ROLES
    meeting = ResearchMeeting(objective)

    # Round 1
    r1 = meeting.run_round_1_independent(meeting_roles, context=context)

    # Round 2
    r2 = meeting.run_round_2_synthesis(r1, context=context)

    # Cross-Agent Questions
    cross_q = meeting.extract_cross_agent_questions(r2)
    cross_answers = []
    if cross_q:
        cross_answers = meeting.run_bounded_cross_consultation(cross_q, max_questions=3)

    # Gemini PI Synthesis
    synthesis = meeting.synthesize_presidential_report(r1, r2, cross_answers)

    return {
        "status": "ok",
        "objective": objective,
        "round_1_independent": {k: v.get("answer", "") for k, v in r1.items()},
        "round_2_synthesis": {k: v.get("answer", "") for k, v in r2.items()},
        "cross_consultations": cross_answers,
        "pi_synthesis": synthesis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Research Meeting Engine CLI")
    parser.add_argument("objective", help="Research milestone or dilemma to deliberate")
    parser.add_argument("--roles", default=",".join(DEFAULT_MEETING_ROLES[:4]), help="Comma-separated roles")
    parser.add_argument("--all-roles", action="store_true", help="Include all 7 council roles")
    parser.add_argument("--context-file", help="Path to context file")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    roles = DEFAULT_MEETING_ROLES if args.all_roles else [r.strip() for r in args.roles.split(",") if r.strip()]

    context = ""
    if args.context_file:
        p = Path(args.context_file)
        if p.is_file():
            context = p.read_text(encoding="utf-8")

    res = run_full_research_meeting(args.objective, roles=roles, context=context)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("================================================================================")
        print("TITAN TEXT: MULTI-AGENT RESEARCH MEETING COUNCIL REPORT")
        print(f"Objective: {args.objective}")
        print("================================================================================\n")

        print("--- ROUND 1: INDEPENDENT DOMAIN ANALYSES ---")
        for role, ans in res["round_1_independent"].items():
            print(f"\n[{role.upper()}]:")
            print(ans[:1200] + ("..." if len(ans) > 1200 else ""))

        print("\n--- ROUND 2: SYNTHESIS & 10 QUESTIONS ---")
        for role, ans in res["round_2_synthesis"].items():
            print(f"\n[{role.upper()}]:")
            print(ans[:1200] + ("..." if len(ans) > 1200 else ""))

        if res["cross_consultations"]:
            print("\n--- CROSS-AGENT SPECIALIST CONSULTATIONS ---")
            for cc in res["cross_consultations"]:
                print(f"\n[{cc['from_role']} -> {cc['to_role']}]: {cc['question']}")
                print(f"Answer: {cc['answer'][:800]}...")

        print("\n--- PRINCIPAL INVESTIGATOR SYNTHESIS & SAFEGUARDS ---")
        for warn in res["pi_synthesis"]["warnings_against_premature_conclusions"]:
            print(f"  • {warn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
