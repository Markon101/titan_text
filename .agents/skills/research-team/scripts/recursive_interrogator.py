#!/usr/bin/env python3
"""Expanded Disciplined Recursive Self-Questioning Engine for DeepSeek Collaborators.

Implements the complete scientific falsification and recursive interrogation workflow
for DeepSeek 4.1 Flash research collaborators.

Protocols:
1. 13-Step Recursive Interrogation Workflow:
   - Step 1: Reconstruct the problem independently.
   - Step 2: State what evidence establishes.
   - Step 3: State what evidence does NOT establish.
   - Step 4: Identify strongest alternative explanation.
   - Step 5: Ask "What am I assuming?"
   - Step 6: Ask "What evidence would make this interpretation wrong?"
   - Step 7: Ask "What measurement could distinguish these explanations?"
   - Step 8: Ask "Could apparent effect arise from task, metric, dataset, architecture,
             initialization, optimizer, readout, stabilization, implementation, or evaluation?"
   - Step 9: Ask "What question have I not been asked that should be asked?"
   - Step 10: Answer newly generated question.
   - Step 11: From answer, generate another productive question if one follows.
   - Step 12: Continue recursively for several rounds while scientifically productive.
   - Step 13: Only then propose discriminating experiments or implementation changes.

2. 11-Point Deep Decision Interrogation (for pivotal scientific choices):
   - Strongest argument FOR
   - Strongest argument AGAINST
   - Strongest mundane explanation
   - Strongest interesting explanation
   - Easiest experiment that could falsify it
   - Most decisive experiment regardless of cost
   - Result that would cause substantial belief update
   - Hidden assumption most likely to invalidate experiment
   - Measurement currently missing
   - Surprising alternative hypothesis
   - Question nobody in the current research loop is asking
   Followed by recursive interrogation of the most critical answers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

# Import sibling subagent engine
SCRIPTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = SCRIPTS_DIR.parent.parent
REPO_ROOT = SKILLS_DIR.parent.parent
OPENROUTER_SCRIPTS = SKILLS_DIR / "openrouter-subagents" / "scripts"
sys.path.insert(0, str(OPENROUTER_SCRIPTS))
sys.path.insert(0, str(SCRIPTS_DIR))

from subagent import run_subagent
from context_builder import ContextBuilder


SYSTEM_PROMPT_13_STEP = """You are a high-context scientific research collaborator and falsification investigator.
You are investigating the Titan Text recurrent neural cellular automata architecture.

Do NOT simply accept the question as framed or rubber-stamp existing interpretations.
You must execute the rigorous 13-step recursive self-questioning protocol:

PHASE I: INDEPENDENT RECONSTRUCTION & BOUNDARY SPECIFICATION
1. RECONSTRUCTION: Reconstruct the problem and architecture independently from first principles.
2. WHAT_ESTABLISHED: State what you believe the empirical evidence genuinely ESTABLISHES.
3. WHAT_NOT_ESTABLISHED: State what the evidence does NOT establish (unearned leaps, correlations).
4. STRONGEST_ALTERNATIVE: Identify the strongest competing or deflationary alternative explanation.

PHASE II: ADVERSARIAL SKEPTICISM & CONFOUND SCRUTINY
5. ASSUMPTIONS: Ask and answer: "What am I assuming?" (List every hidden axiom).
6. FALSIFYING_EVIDENCE: Ask and answer: "What specific observation would prove this interpretation WRONG?"
7. DISCRIMINATING_MEASUREMENT: Ask and answer: "What quantitative measurement distinguishes these explanations?"
8. SYSTEM_CONFUND_AUDIT: Could the apparent effect arise from:
   - Task definition or dataset generator leakage?
   - Metric choice or unweighted token averaging?
   - Architecture or feedforward stencil lookahead?
   - Initialization bias or unseeded RNG?
   - Optimizer dynamics or learning rate schedules?
   - Readout projection or logit saturation?
   - Stabilization mechanism (e.g. STE clipping vs continuous norm)?
   - Evaluation harness or causal masking bugs?

PHASE III: RECURSIVE QUESTION DISCOVERY
9. UNASKED_QUESTION_1: "What important question has nobody asked that SHOULD be asked?"
10. ANSWER_1: Answer that newly generated question rigorously.
11. UNASKED_QUESTION_2: From that answer, what deeper second-order question emerges?
12. ANSWER_2: Answer that second follow-up question.

PHASE IV: DISCRIMINATING EXPERIMENTS
13. EXPERIMENTAL_PROPOSAL: Only now propose:
   - Easiest cheap experiment to falsify.
   - Most decisive discriminating experiment regardless of compute cost.
   - Concrete implementation guidance and baseline controls.

STRICT EVIDENCE TAXONOMY:
Tag every factual claim with:
[ESTABLISHED], [SUPPORTED], [HYPOTHESIS], [FALSIFIED], [ARTIFACT], or [SPECULATION].
Do not use vague buzzwords ("reasoning", "attractor manifold", "algorithm", "memory") without operational metrics.
"""

SYSTEM_PROMPT_DEEP_DECISION = """You are a senior scientific arbiter conducting a Deep Decision Interrogation
for a pivotal fork in the Titan Text research campaign.

You must rigorously produce the 11-point decision battery:
1. STRONGEST_ARG_FOR: Strongest empirical and theoretical argument FOR the current interpretation.
2. STRONGEST_ARG_AGAINST: Strongest empirical and theoretical argument AGAINST it.
3. STRONGEST_MUNDANE_EXPLANATION: Most deflationary, boring explanation that fits the data.
4. STRONGEST_INTERESTING_EXPLANATION: Most profound, novel mechanistic explanation.
5. EASIEST_FALSIFYING_TEST: The cheapest, lowest-overhead experiment that could decisively falsify it.
6. MOST_DECISIVE_EXPERIMENT: The gold-standard experiment regardless of implementation or compute cost.
7. PIVOT_RESULT: The exact quantitative metric or result that would force you to abandon your belief.
8. FRAGILE_ASSUMPTION: The hidden premise most likely to invalidate the current experiment.
9. MISSING_MEASUREMENT: The critical observable or diagnostic we are currently failing to measure.
10. SURPRISING_ALTERNATIVE_HYPOTHESIS: A radical, unconventional hypothesis consistent with evidence.
11. UNASKED_RESEARCH_QUESTION: The fundamental question nobody in the current loop appears to be asking.

FOLLOW-UP RECURSION:
After presenting all 11 points, identify the single most dangerous fragile assumption (#8) and
the missing measurement (#9), and recursively interrogate them to propose a concrete protocol.
"""


def is_repetitive(text_a: str, text_b: str, threshold: float = 0.75) -> bool:
    """Check Jaccard similarity of word sets to detect redundant repetition."""
    words_a = set(re.findall(r"\b[a-z]{4,}\b", text_a.lower()))
    words_b = set(re.findall(r"\b[a-z]{4,}\b", text_b.lower()))
    if not words_a or not words_b:
        return False
    intersection = words_a.intersection(words_b)
    union = words_a.union(words_b)
    sim = len(intersection) / len(union)
    return sim >= threshold


def run_13_step_interrogation(
    target_claim_or_question: str,
    *,
    context: str = "",
    max_rounds: int = 2,
    enable_reasoning: bool = True,
    timeout: int = 360,
) -> dict[str, Any]:
    """Execute the full 13-step recursive self-questioning workflow."""
    builder = ContextBuilder()
    rich_context = context or builder.build_packet(target_claim_or_question)

    history: list[dict[str, Any]] = []
    current_input = (
        f"--- TARGET INQUIRY ---\n{target_claim_or_question}\n\n"
        "Execute the 13-step recursive self-questioning protocol. "
        "Maintain high density, mathematical rigor, and operational definitions."
    )

    stopped_early = False
    stop_reason = ""

    for rnd in range(1, max_rounds + 1):
        if rnd > 1:
            current_input = (
                f"--- RECURSIVE PASS {rnd} OF {max_rounds} ---\n"
                f"Prior pass output concluded with these key unasked questions and proposed tests:\n"
                f"{history[-1]['output']}\n\n"
                "Interrogate the deepest unasked question (#11) and the experimental proposal (#13). "
                "Challenge the core premise. What hidden confound did this proposal introduce? "
                "If diminishing returns have been reached, respond with 'NO_NEW_INFORMATION' and finalize."
            )

        res = run_subagent(
            current_input,
            role="researcher",
            context=f"{SYSTEM_PROMPT_13_STEP}\n\n{rich_context}",
            max_tokens=4096,
            enable_reasoning=enable_reasoning,
            timeout=timeout,
        )

        answer = res.get("answer", "").strip()

        if "NO_NEW_INFORMATION" in answer.upper() or len(answer) < 80:
            stopped_early = True
            stop_reason = "Subagent reported NO_NEW_INFORMATION (asymptotic epistemic convergence)."
            history.append({"round": rnd, "output": answer, "novel": False})
            break

        if rnd > 1 and is_repetitive(history[-1]["output"], answer):
            stopped_early = True
            stop_reason = f"Repetition detected between round {rnd-1} and round {rnd} (Jaccard >= 0.75)."
            break

        history.append({"round": rnd, "output": answer, "novel": True})

    return {
        "status": "ok",
        "protocol": "13-step-recursive-questioning",
        "rounds_completed": len(history),
        "passes_completed": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "history": history,
        "final_output": history[-1]["output"] if history else "",
    }


def run_deep_decision_interrogation(
    decision_dilemma: str,
    *,
    context: str = "",
    enable_reasoning: bool = True,
    timeout: int = 360,
) -> dict[str, Any]:
    """Execute the 11-point Deep Decision Interrogation for pivotal choices."""
    builder = ContextBuilder()
    rich_context = context or builder.build_packet(decision_dilemma)

    prompt = (
        f"--- PIVOTAL RESEARCH DECISION / DILEMMA ---\n"
        f"{decision_dilemma}\n\n"
        "Execute the complete 11-point decision battery, followed by recursive interrogation "
        "of the most fragile assumption and missing measurement."
    )

    res = run_subagent(
        prompt,
        role="researcher",
        context=f"{SYSTEM_PROMPT_DEEP_DECISION}\n\n{rich_context}",
        max_tokens=4096,
        enable_reasoning=enable_reasoning,
        timeout=timeout,
    )

    answer = res.get("answer", "").strip()
    return {
        "status": "ok",
        "protocol": "deep-decision-11-point",
        "decision_dilemma": decision_dilemma,
        "output": answer,
        "usage": res.get("usage", {}),
    }


def run_recursive_interrogation(
    claim_or_finding: str,
    *,
    context: str = "",
    max_depth: int = 2,
    enable_reasoning: bool = False,
    timeout: int = 240,
    full_13_step: bool = True,
    deep_decision: bool = False,
) -> dict[str, Any]:
    """Unified entry point for recursive interrogation."""
    if deep_decision:
        return run_deep_decision_interrogation(
            claim_or_finding,
            context=context,
            enable_reasoning=enable_reasoning,
            timeout=timeout,
        )
    elif full_13_step:
        return run_13_step_interrogation(
            claim_or_finding,
            context=context,
            max_rounds=max_depth,
            enable_reasoning=enable_reasoning,
            timeout=timeout,
        )
    else:
        # Fast 2-pass interrogation
        return run_13_step_interrogation(
            claim_or_finding,
            context=context,
            max_rounds=1,
            enable_reasoning=enable_reasoning,
            timeout=timeout,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expanded Recursive Self-Questioning Engine")
    parser.add_argument("target", help="Claim, hypothesis, or research question to interrogate")
    parser.add_argument("--context", default="", help="Inline context string")
    parser.add_argument("--context-file", help="Path to file containing additional context")
    parser.add_argument("--depth", type=int, default=2, help="Maximum recursive passes (default: 2)")
    parser.add_argument("--deep-decision", action="store_true", help="Execute 11-point deep decision battery")
    parser.add_argument("--fast", action="store_true", help="Run fast single-pass interrogation")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning tokens")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    context = args.context
    if args.context_file:
        p = Path(args.context_file)
        if p.is_file():
            context = p.read_text(encoding="utf-8") + "\n\n" + context

    if args.deep_decision:
        res = run_deep_decision_interrogation(
            args.target,
            context=context,
            enable_reasoning=not args.no_reasoning,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("=== DEEP DECISION INTERROGATION (11-POINT BATTERY) ===")
            print(res.get("output", ""))
    else:
        res = run_13_step_interrogation(
            args.target,
            context=context,
            max_rounds=1 if args.fast else args.depth,
            enable_reasoning=not args.no_reasoning,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"=== 13-STEP RECURSIVE INTERROGATION ({res['rounds_completed']} pass(es)) ===")
            if res.get("stopped_early"):
                print(f"[STOPPED EARLY: {res.get('stop_reason')}]")
            print("\n" + res.get("final_output", ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
