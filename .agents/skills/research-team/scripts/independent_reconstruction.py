#!/usr/bin/env python3
"""Independent Reconstruction Test for DeepSeek Research Collaborators.

Before trusting a major consultation, tests whether a DeepSeek 4.1 Flash agent
can independently reconstruct the ground-truth state of Titan Text from the
evidence packet alone, without hallucinating, collapsing nuances, or reasoning
from superseded historical artifacts.

Canonical Interrogation Prompt:
"Given only this evidence, reconstruct your model of Titan Text. Explain what has
actually been demonstrated, what has not been demonstrated, which historical
conclusions are now invalid, what the strongest remaining alternative explanations
are, and what experiment would most efficiently distinguish them."

Audits:
1. Demonstrated vs Non-Demonstrated Boundary
2. Historical Invalidation / Artifact Awareness
3. Alternative Explanations Quality
4. Discriminating Experiment Feasibility & Information Gain
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

# Import sibling modules
SCRIPTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = SCRIPTS_DIR.parent.parent
REPO_ROOT = SKILLS_DIR.parent.parent
OPENROUTER_SCRIPTS = SKILLS_DIR / "openrouter-subagents" / "scripts"
sys.path.insert(0, str(OPENROUTER_SCRIPTS))
sys.path.insert(0, str(SCRIPTS_DIR))

from subagent import run_subagent
from context_builder import ContextBuilder
from research_state import ResearchState


SYSTEM_PROMPT = """You are a rigorous, independent scientific reviewer and theoretical evaluator.
You have been provided with primary evidence, metrics, code excerpts, and historical logs
from the Titan Text neural cellular automata research campaign.

Your task is to independently reconstruct the exact state of knowledge about Titan Text.
You must NOT simply repeat claims. Evaluate the evidence strictly and answer these 5 core questions:

1. WHAT HAS ACTUALLY BEEN DEMONSTRATED:
   What has been proven by deterministic measurements, lesions, and paired controls?
   (Be specific about metrics, effect sizes, task conditions, and seeds).

2. WHAT HAS NOT BEEN DEMONSTRATED:
   What attractive claims, generalizations, or buzzwords remain unproven or refuted?
   Where have researchers leaped from correlation to causation?

3. WHICH HISTORICAL CONCLUSIONS ARE NOW INVALID:
   Identify previous claims or results that were later discovered to be artifacts,
   flawed baselines, or superseded by subsequent experiments.
   Explain WHY they were invalid.

4. STRONGEST REMAINING ALTERNATIVE EXPLANATIONS:
   For the positive results that remain established (e.g. Dyck-4 bracket matching, ECR carry transfer),
   what is the strongest mundane or deflationary alternative explanation?

5. MOST EFFICIENT DISCRIMINATING EXPERIMENT:
   What single minimal experiment would most decisively distinguish the leading hypothesis
   from its strongest competitor?

Maintain high density, precision, and skepticism. Tag all items with [ESTABLISHED], [SUPPORTED],
[HYPOTHESIS], [FALSIFIED], [ARTIFACT], or [SPECULATION].
"""

RECONSTRUCTION_PROMPT = """Given only this evidence, reconstruct your model of Titan Text.
Explain what has actually been demonstrated, what has not been demonstrated, which historical
conclusions are now invalid, what the strongest remaining alternative explanations are,
and what experiment would most efficiently distinguish them."""


def audit_reconstruction(output: str) -> dict[str, Any]:
    """Audit the agent's independent reconstruction against known ground truth."""
    text_lower = output.lower()

    # 1. Check artifact awareness
    identified_baseline_artifact = (
        any(kw in text_lower for kw in ["untrained", "randomly initialized", "random weight", "uninitialized", "cmd_benchmark"])
        and any(kw in text_lower for kw in ["baseline", "transformer", "gru", "rnn", "0.0%"])
    )
    identified_fc4_artifact = any(
        kw in text_lower for kw in ["shallow", "d=1", "stratified", "fc-4", "nesting depth"]
    )
    identified_h1_falsification = any(
        kw in text_lower for kw in ["settling", "tau=48", "horizon", "correlation length"]
    )
    identified_dpda_fsa_bound = any(
        kw in text_lower for kw in ["finite state", "fsa", "type-3", "bounded dpda", "chomsky"]
    )

    artifact_score = sum([
        identified_baseline_artifact,
        identified_fc4_artifact,
        identified_h1_falsification,
        identified_dpda_fsa_bound,
    ]) / 4.0

    # 2. Check demonstrated boundaries
    recognized_recurrence_load = any(
        kw in text_lower for kw in ["lesion", "zero-tick", "causally required", "load-bearing", "d=15.56"]
    )
    recognized_ecr_carry = any(
        kw in text_lower for kw in ["carry", "ecr", "discrete", "ste", "sign"]
    )
    recognized_dyck_pushdown = any(
        kw in text_lower for kw in ["dyck", "pushdown", "bracket", "43.9%"]
    )

    demonstrated_score = sum([
        recognized_recurrence_load,
        recognized_ecr_carry,
        recognized_dyck_pushdown,
    ]) / 3.0

    # 3. Check non-demonstrated bounds
    recognized_loss_drift = any(
        kw in text_lower for kw in ["loss explosion", "loss drift", "lyapunov", "ocpd", "damping"]
    )
    recognized_ngram_confound = any(
        kw in text_lower for kw in ["bigram", "local", "shallow", "n-gram"]
    )
    non_demonstrated_score = sum([
        recognized_loss_drift,
        recognized_ngram_confound,
    ]) / 2.0

    # Overall grade
    total_score = (artifact_score * 0.4) + (demonstrated_score * 0.3) + (non_demonstrated_score * 0.3)

    if total_score >= 0.80 and artifact_score >= 0.75:
        grade = "PASS"
        recommendation = "Agent reconstruction is high-fidelity. Safe to proceed with experimental consultations."
    elif total_score >= 0.50:
        grade = "WARN"
        recommendation = (
            "Agent missed key historical artifacts or boundaries. "
            "Reinforce the research-state packet before trusting its recommendations."
        )
    else:
        grade = "FAIL"
        recommendation = (
            "Agent reconstruction is materially flawed or reasoned from superseded results. "
            "Improve context packet and re-run reconstruction test."
        )

    return {
        "grade": grade,
        "total_score": round(total_score, 2),
        "scores": {
            "artifact_invalidation_score": round(artifact_score, 2),
            "demonstrated_boundary_score": round(demonstrated_score, 2),
            "non_demonstrated_boundary_score": round(non_demonstrated_score, 2),
        },
        "flags": {
            "identified_baseline_artifact": identified_baseline_artifact,
            "identified_fc4_shallow_artifact": identified_fc4_artifact,
            "identified_h1_falsification": identified_h1_falsification,
            "identified_dpda_fsa_bound": identified_dpda_fsa_bound,
            "recognized_recurrence_necessity": recognized_recurrence_load,
            "recognized_ecr_carry": recognized_ecr_carry,
            "recognized_dyck_pushdown": recognized_dyck_pushdown,
            "recognized_continuous_loss_drift": recognized_loss_drift,
        },
        "recommendation": recommendation,
    }


def run_reconstruction_test(
    objective: str = "Independent Model Reconstruction of Titan Text",
    *,
    context: str = "",
    enable_reasoning: bool = True,
    timeout: int = 360,
) -> dict[str, Any]:
    """Execute independent reconstruction test and audit."""
    builder = ContextBuilder()
    rich_context = context or builder.build_packet(objective)

    res = run_subagent(
        RECONSTRUCTION_PROMPT,
        role="researcher",
        context=f"{SYSTEM_PROMPT}\n\n{rich_context}",
        max_tokens=4096,
        enable_reasoning=enable_reasoning,
        timeout=timeout,
    )

    answer = res.get("answer", "").strip()
    audit = audit_reconstruction(answer)

    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
        "reconstruction": answer,
        "usage": res.get("usage", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Titan Text Independent Reconstruction Test CLI")
    parser.add_argument("--objective", default="Reconstruct physical and empirical model of Titan Text")
    parser.add_argument("--context-file", help="Path to custom context packet file")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning tokens")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    context = ""
    if args.context_file:
        p = Path(args.context_file)
        if p.is_file():
            context = p.read_text(encoding="utf-8")

    res = run_reconstruction_test(
        objective=args.objective,
        context=context,
        enable_reasoning=not args.no_reasoning,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        audit = res["audit"]
        print("================================================================================")
        print(f"TITAN TEXT: INDEPENDENT RECONSTRUCTION AUDIT [{audit['grade']}] (Score: {audit['total_score']})")
        print("================================================================================")
        print(f"Recommendation: {audit['recommendation']}\n")
        print("Sub-Scores:")
        for k, v in audit["scores"].items():
            print(f"  • {k}: {v}")
        print("\nGround-Truth Diagnostic Flags:")
        for k, v in audit["flags"].items():
            status_sym = "✓" if v else "✗"
            print(f"  [{status_sym}] {k}")
        print("\n--- AGENT RECONSTRUCTION OUTPUT ---")
        print(res["reconstruction"])

    return 0 if res["audit"]["grade"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
