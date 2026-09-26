#!/usr/bin/env python3
"""Jev Scientific Governor for Titan Text.

Leverages TypeSafe's Jev model on OpenRouter as a fast, advisory, structured
System-1 cognitive governor for scientific hypothesis arbitration, causal dissociation
auditing, information gain assessment, and research routing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

# Add openrouter-subagents script path
REPO_ROOT = Path(__file__).resolve().parent.parent
SUBAGENTS_SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "openrouter-subagents" / "scripts"
if str(SUBAGENTS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBAGENTS_SCRIPT_DIR))

try:
    from jev_decide import jev_decide
except ImportError:
    # Graceful fallback mock if import fails
    def jev_decide(decision_type: str, state: Any, **kwargs) -> dict[str, Any]:
        return {
            "status": "fallback",
            "decision": "inconclusive",
            "confidence": 0.0,
            "fallback_applied": True,
            "probabilities": {},
            "error": "jev_decide import failed",
        }


def arbitrate_mechanism(evidence_summary: dict[str, Any]) -> dict[str, Any]:
    """Arbitrates among competing physical/algorithmic mechanisms for ECR."""
    schema = {
        "question_id": "ecr_mechanism_verdict",
        "type": "choice",
        "instructions": (
            "Given empirical evidence from the 10-condition mechanism matrix, parameter accounting, "
            "and causal ablation benchmarks, determine which mechanism primarily explains the ECR advantage."
        ),
        "criteria": {
            "inconclusive": "Evidence is missing, invalid, or insufficient to distinguish the alternatives.",
            "hyperbolic_advection": "Unitary, non-dissipative spatial carry transport overcoming continuous NCA exponential attenuation.",
            "receptive_field_dilation": "Purely geometric widening of the spatial receptive field per tick, equivalent to dilated convolution.",
            "capacity_decoupling": "Separation of persistent memory registers from transient feature computation preventing overwrite.",
            "optimization_gradient_highway": "Linear or identity residual pathways in carry channels allowing deeper gradient flow."
        },
        "default": "inconclusive"
    }
    state_str = json.dumps(evidence_summary, indent=2)
    return jev_decide("ecr_mechanism_verdict", state_str, custom_schema=schema)


def audit_causal_transplant(transplant_data: dict[str, Any]) -> dict[str, Any]:
    """Audits double dissociation in causal carry transplant experiments."""
    schema = {
        "question_id": "causal_transplant_verdict",
        "type": "choice",
        "instructions": (
            "Evaluate whether the causal carry transplant experiment demonstrates a rigorous double dissociation. "
            "Check whether swapping carry registers flips predictions to donor targets while hidden swap leaves predictions intact."
        ),
        "criteria": {
            "inconclusive": "Evidence is missing, invalid, or insufficient to distinguish the alternatives.",
            "decisive_double_dissociation": "Carry swap flips predictions (CME > 0.80) while hidden swap preserves original predictions (CME < 0.20).",
            "partial_carry_mediation": "Carry swap partially flips predictions (0.40 < CME < 0.80), indicating shared representation with hidden features.",
            "hidden_dominated": "Hidden swap flips predictions more than carry swap, falsifying the carry register hypothesis.",
            "symmetric_entanglement": "Both carry and hidden swaps degrade or flip predictions similarly, indicating tight non-separable coupling.",
            "inconclusive_artifact": "Predictions collapse to chance or constant output across all swapped conditions."
        },
        "default": "inconclusive"
    }
    state_str = json.dumps(transplant_data, indent=2)
    return jev_decide("causal_transplant_verdict", state_str, custom_schema=schema)


def score_falsification_risk(claim: str, empirical_data: dict[str, Any]) -> dict[str, Any]:
    """Scores the risk that a theoretical claim contradicts empirical measurements."""
    schema = {
        "question_id": "falsification_risk",
        "type": "choice",
        "instructions": "Evaluate whether the specified theoretical claim is contradicted or jeopardized by empirical observations.",
        "criteria": {
            "inconclusive": "Evidence is missing, invalid, or insufficient to distinguish the alternatives.",
            "falsified": "Claim is directly contradicted by empirical measurements or parameter invariants.",
            "high_tension": "Claim is in strong tension with observed scaling bounds, variance, or ablation degradation.",
            "moderate_tension": "Claim holds in narrow regimes but fails to explain key edge cases or transfer tasks.",
            "fully_consistent": "Claim is mathematically sound and corroborated by all empirical benchmark conditions."
        },
        "default": "inconclusive"
    }
    payload = {"claim": claim, "data": empirical_data}
    return jev_decide("falsification_risk", json.dumps(payload, indent=2), custom_schema=schema)


def recommend_next_phase(current_progress: dict[str, Any]) -> dict[str, Any]:
    """Routes the research team to the single highest information-gain next action."""
    schema = {
        "question_id": "research_next_step",
        "type": "choice",
        "instructions": "Given completed phases and current findings, what is the single highest-value research action to pursue next?",
        "criteria": {
            "inconclusive": "Evidence is missing, invalid, or insufficient to distinguish the alternatives.",
            "run_causal_transplant": "Execute Phase 5 causal carry transplant double-dissociation probe.",
            "run_carry_probing": "Execute Phase 4 carry decodability vs distance curves across slots.",
            "run_ood_length_sweep": "Execute Phase 8 zero-shot OOD length generalization (L=16 -> 128).",
            "run_velocity_diagram": "Execute Phase 6 (T, k, L) propagation velocity and recurrence depth diagram.",
            "write_final_synthesis": "Sufficient empirical and theoretical evidence collected; compile paper-grade report."
        },
        "default": "inconclusive"
    }
    state_str = json.dumps(current_progress, indent=2)
    return jev_decide("research_next_step", state_str, custom_schema=schema)


if __name__ == "__main__":
    print("Testing Jev Scientific Governor...")
    sample_progress = {
        "phase_1_parity_discrepancy": "RESOLVED (causal stencil provides +6.4% to +9.7% lift; historical 76.6% was single-batch upper tail)",
        "phase_2_parameter_audit": "RESOLVED (identical 43,715 parameters across Cc in {0, 16, 32})",
        "phase_3_mechanism_matrix": "RUNNING (10 conditions across 5 seeds)",
    }
    decision = recommend_next_phase(sample_progress)
    print("Recommendation:", decision.get("decision"))
    print("Confidence:", decision.get("confidence"))
    print("Distribution:", decision.get("probabilities"))
