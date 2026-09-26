#!/usr/bin/env python3
"""Expanded High-Context Ideation Engine for Titan Text.

Dedicated exploratory AI research ideation scientist using DeepSeek 4.1 Flash.
Searches conceptual space broadly across dynamical systems, cellular automata,
neuroscience, information theory, control theory, reservoir computing,
statistical mechanics, and algorithmic learning.

Explicitly poses the 11 exploratory questions:
1. "If Titan Text is doing something scientifically interesting that our current
   experiments are incapable of detecting, what might it be?"
2. "What experiment would reveal behavior we are currently averaging away?"
3. "What variable have we treated as a nuisance that might actually contain the phenomenon?"
4. "What result would be surprising under every current hypothesis?"
5. "What minimal synthetic world would force Titan to reveal whether it possesses
   persistent, compositional, iterative latent computation?"
6. "What experiment sounds strange at first but would produce highly discriminating evidence?"
7. "What can we measure directly from the saved per-step activation traces that we have not yet considered?"
8. "What interventions could distinguish information storage, information transport,
   iterative transformation, attractor-like stabilization, generic nonlinear conditioning,
   and genuine task-specific recurrent computation?"
9. "Can we construct two inputs that are locally indistinguishable but require different
   outputs solely because of information that must have propagated through recurrent state?"
10. "Can we construct counterfactual state-transplant, state-rescue, state-swap,
    partial-channel transplant, delayed intervention, or trajectory-splicing experiments?"
11. "Can we discover the smallest latent subspace carrying task-relevant information
    and then intervene specifically on that subspace?"

Outputs:
- Conservative / High-Value Experiments (ranked by information gain per unit compute)
- Unconventional High-Risk / High-Information Experiments (preserved unconditionally)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
from jev_decide import jev_decide
from context_builder import ContextBuilder
from reasoning_frontier import ReasoningFrontier, compute_text_similarity


IDEATION_SYSTEM_PROMPT = """You are a dedicated, high-context exploratory scientific ideation collaborator.
Your purpose is NOT conservative review, code auditing, or rubber-stamping existing frameworks.
Your purpose is to search the conceptual space broadly across:
- dynamical systems & chaos theory
- cellular automata & lattice fluids
- neuroscience & neural manifolds
- information theory & transfer entropy
- control theory & contraction metrics
- recurrent computation & reservoir computing
- algorithm learning & pushdown automata
- statistical mechanics & phase transitions.

DO NOT ASSUME OUR CURRENT FRAMING IS COMPLETE OR CORRECT.
You must actively consider:
- What behavior are our current experiments averaging away?
- What variable treated as a nuisance (e.g. logit temperature, spatial roll phase, step drift)
  might actually contain the phenomenon?
- How to construct counterfactual state-transplants, state-rescues, state-swaps, partial-channel
  transplants, delayed interventions, and trajectory-splicing experiments.
- How to isolate the minimal latent subspace carrying task-relevant information.
- What minimal synthetic world would force Titan to reveal whether it possesses persistent,
  compositional, iterative latent computation.

STRICT TAXONOMY FOR PROPOSALS:
Every proposal must distinguish:
- Category: "conservative_high_value" OR "unconventional_high_risk_high_info"
- Mechanism: The proposed causal phenomenon.
- Diagnostic/Intervention: The exact operational measurement.
- Expected Information Gain: What hypothesis space is partitioned.
- Estimated Compute/Effort: Low, Medium, or High.

TAG ALL SPECULATIVE CLAIMS WITH [SPECULATION].
"""

IDEATION_QUESTIONS_BLOCK = """EXPLORATORY CORE INQUIRIES:
1. If Titan Text is doing something scientifically interesting that our current experiments are incapable of detecting, what might it be?
2. What experiment would reveal behavior we are currently averaging away?
3. What variable have we treated as a nuisance that might actually contain the phenomenon?
4. What result would be surprising under every current hypothesis?
5. What minimal synthetic world would force Titan to reveal whether it possesses persistent, compositional, iterative latent computation?
6. What experiment sounds strange at first but would produce highly discriminating evidence?
7. What can we measure directly from the saved per-step activation traces that we have not yet considered?
8. What interventions could distinguish information storage, information transport, iterative transformation, attractor-like stabilization, generic nonlinear conditioning, and genuine task-specific recurrent computation?
9. Can we construct two inputs that are locally indistinguishable but require different outputs solely because of information that must have propagated through recurrent state?
10. Can we construct counterfactual state-transplant, state-rescue, state-swap, partial-channel transplant, delayed intervention, or trajectory-splicing experiments?
11. Can we discover the smallest latent subspace carrying task-relevant information and then intervene specifically on that subspace?
"""


def generate_candidate_ideas(
    problem_statement: str,
    *,
    temperature: float = 0.85,
    candidate_count: int = 4,
    context: str = "",
    timeout: int = 360,
) -> dict[str, Any]:
    """Execute high-temperature exploratory ideation generation."""
    builder = ContextBuilder()
    rich_context = context or builder.build_packet(problem_statement)

    task_prompt = (
        f"--- RESEARCH OBJECTIVE / EXPLORATION TARGET ---\n"
        f"{problem_statement}\n\n"
        f"{IDEATION_QUESTIONS_BLOCK}\n\n"
        "Generate a structured portfolio containing:\n"
        "1. At least 2 CONSERVATIVE / HIGH-VALUE experiments (ranked by information gain per unit compute).\n"
        "2. At least 2 UNCONVENTIONAL HIGH-RISK / HIGH-INFORMATION experiments that should not be discarded merely because they are unusual.\n"
        "3. Concrete mathematical diagnostics from saved per-step activation traces.\n"
        "4. Exact counterfactual causal intervention protocols.\n\n"
        "Respond with a valid JSON object matching this schema:\n"
        "{\n"
        '  "unconventional_ideas": [\n'
        '    {"name": "...", "category": "unconventional_high_risk_high_info", "mechanism": "...", "intervention_protocol": "...", "information_gain": "...", "estimated_compute": "low|medium|high", "target_inquiry_number": 1}\n'
        '  ],\n'
        '  "conservative_ideas": [\n'
        '    {"name": "...", "category": "conservative_high_value", "mechanism": "...", "discriminating_test": "...", "information_gain": "...", "estimated_compute": "low|medium|high"}\n'
        '  ],\n'
        '  "trace_diagnostics": ["...", "..."],\n'
        '  "counterfactual_world_proposal": "..."\n'
        "}"
    )

    res = run_subagent(
        task_prompt,
        role="ideation-agent",
        context=f"{IDEATION_SYSTEM_PROMPT}\n\n{rich_context}",
        temperature=temperature,
        max_tokens=4096,
        json_answer=True,
        timeout=timeout,
    )

    parsed = res.get("parsed_json")
    if not isinstance(parsed, dict):
        parsed = {
            "unconventional_ideas": [],
            "conservative_ideas": [],
            "trace_diagnostics": [],
            "counterfactual_world_proposal": res.get("answer", ""),
        }

    return parsed


def filter_and_score_candidates(
    ideas_portfolio: dict[str, Any] | list[dict[str, Any]],
    frontier: ReasoningFrontier,
) -> list[dict[str, Any]]:
    """Phase 2: Low-variance structured selection using Jev sensors with diversity bypass."""
    accepted = []

    if isinstance(ideas_portfolio, list):
        all_candidates = list(ideas_portfolio)
    elif isinstance(ideas_portfolio, dict):
        all_candidates = []
        for item in ideas_portfolio.get("unconventional_ideas", []):
            item["category"] = "unconventional_high_risk_high_info"
            all_candidates.append(item)
        for item in ideas_portfolio.get("conservative_ideas", []):
            item["category"] = "conservative_high_value"
            all_candidates.append(item)
    else:
        all_candidates = []

    for cand in all_candidates:
        name = cand.get("name", "IdeationProposal")
        mech = cand.get("mechanism", "")
        test = cand.get("intervention_protocol", cand.get("discriminating_test", ""))
        full_text = f"{name}: {mech}. Protocol: {test}"
        category = cand.get("category", "conservative_high_value")

        # Duplicate check against frontier
        is_duplicate = False
        for b in frontier.branches.values():
            if b.status == "active":
                if compute_text_similarity(full_text, b.hypothesis_or_action) >= 0.70:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue

        # Jev sensors
        res_plaus = jev_decide("PLAUSIBILITY", full_text)
        res_nov = jev_decide("NOVELTY", full_text)
        res_test = jev_decide("TESTABILITY", full_text)

        judgments_available = all(r.get("status") == "ok" and not r.get("fallback_applied") for r in (res_plaus, res_nov, res_test))
        plaus_choice = res_plaus.get("decision", "inconclusive")
        nov_choice = res_nov.get("decision", "novel")
        test_choice = res_test.get("decision", "requires_harness")

        plaus_scores = {"highly_plausible": 1.0, "plausible": 0.75, "unlikely": 0.35, "implausible": 0.05}
        nov_scores = {"radical": 1.0, "novel": 0.75, "incremental": 0.40, "derivative": 0.10}

        plaus_num = plaus_scores.get(plaus_choice, 0.5)
        nov_num = nov_scores.get(nov_choice, 0.5)

        # UNCONVENTIONAL DIVERSITY BYPASS:
        # High-risk / high-information proposals are preserved unconditionally
        # as long as they are not derivative noise.
        is_diversity_preserved = False
        if category == "unconventional_high_risk_high_info":
            is_diversity_preserved = True
            # Boost novelty for ranking preservation
            nov_num = max(nov_num, 0.85)
        elif plaus_choice == "unlikely" and nov_choice in ("novel", "radical") and test_choice in ("immediately_testable", "requires_harness"):
            is_diversity_preserved = True
        elif judgments_available and plaus_choice == "implausible" and nov_choice in ("derivative", "incremental"):
            continue

        accepted.append({
            "name": name,
            "category": category,
            "hypothesis_text": full_text,
            "plausibility": plaus_num,
            "novelty_score": nov_num,
            "testability": test_choice,
            "discriminating_action": test,
            "is_diversity_preserved": is_diversity_preserved,
            "is_diversity_pass": is_diversity_preserved,
            "information_gain": cand.get("information_gain", "high"),
            "estimated_compute": cand.get("estimated_compute", "medium"),
            "requires_manual_review": not judgments_available,
            "jev_statuses": [r.get("status", "unavailable") for r in (res_plaus, res_nov, res_test)],
            "jev_judgments": {
                "plausibility": res_plaus.get("probabilities"),
                "novelty": res_nov.get("probabilities"),
                "testability": res_test.get("probabilities"),
            },
        })

    return accepted


def run_ideation_cycle(
    problem_statement: str,
    frontier: ReasoningFrontier | None = None,
    *,
    context: str = "",
) -> dict[str, Any]:
    """Execute complete high-context ideation and selection cycle."""
    rf = frontier or ReasoningFrontier()

    active = [b for b in rf.branches.values() if b.status == "active"]
    avg_novelty = sum(b.novelty_score for b in active) / len(active) if active else 0.5

    temperature = 0.90 if avg_novelty < 0.40 else 0.85

    portfolio = generate_candidate_ideas(
        problem_statement,
        temperature=temperature,
        context=context,
    )

    accepted = filter_and_score_candidates(portfolio, rf)

    added_branch_ids = []
    for acc in accepted:
        bid, is_new = rf.add_branch(
            acc["hypothesis_text"],
            branch_type="unconventional_exploration" if acc["is_diversity_preserved"] else "causal_hypothesis",
            plausibility=acc["plausibility"],
            novelty_score=acc["novelty_score"],
            next_discriminating_action=acc["discriminating_action"],
            jev_probabilities=acc["jev_judgments"]["plausibility"],
        )
        if is_new:
            added_branch_ids.append(bid)

    return {
        "status": "ok",
        "temperature": temperature,
        "portfolio": portfolio,
        "accepted_candidates_count": len(accepted),
        "new_branches_added": added_branch_ids,
        "accepted_details": accepted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expanded High-Context Ideation Engine CLI")
    parser.add_argument("objective", help="Exploration target, anomaly, or question")
    parser.add_argument("--context", default="")
    parser.add_argument("--context-file")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    context = args.context
    if args.context_file:
        p = Path(args.context_file)
        if p.is_file():
            context = p.read_text(encoding="utf-8")

    res = run_ideation_cycle(args.objective, context=context)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("================================================================================")
        print("TITAN TEXT: DEDICATED IDEATION AGENT PORTFOLIO")
        print(f"Target: {args.objective}")
        print(f"Accepted {res['accepted_candidates_count']} proposals into Reasoning Frontier:")
        print("================================================================================\n")

        for acc in res["accepted_details"]:
            tag = "[UNCONVENTIONAL / HIGH-INFO]" if acc["is_diversity_preserved"] else "[CONSERVATIVE / HIGH-VALUE]"
            print(f"• {tag} {acc['name']}")
            print(f"  Plausibility: {acc['plausibility']:.2f} | Novelty: {acc['novelty_score']:.2f} | Compute: {acc['estimated_compute']}")
            print(f"  Hypothesis:   {acc['hypothesis_text']}")
            print(f"  Info Gain:    {acc['information_gain']}\n")

        trace_diags = res.get("portfolio", {}).get("trace_diagnostics", [])
        if trace_diags:
            print("Trace Diagnostics Proposed:")
            for td in trace_diags:
                print(f"  - {td}")

        world = res.get("portfolio", {}).get("counterfactual_world_proposal")
        if world:
            print(f"\nCounterfactual Synthetic World Proposal:\n{world}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
