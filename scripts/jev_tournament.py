#!/usr/bin/env python3
"""Jev Scientific Hypothesis Tournament for Titan Text.

Runs an automated, rigorous Bayesian tournament arbitrated by TypeSafe's Jev model
on OpenRouter across the primary competing scientific hypotheses for Titan NCA:

H1: Hyperbolic Advection (Unitary spatial transport v = k overcomes continuous diffusion bound xi = 4.53)
H2: Receptive Field Dilation (Stride k merely acts as a dilated convolution shortcut)
H3: Representation Decoupling (Carry scratchpad protects memory from destructive hidden updates)
H4: Optimization Gradient Highway (Carry channels act as linear identity residuals easing BPTT)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBAGENTS_SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "openrouter-subagents" / "scripts"
if str(SUBAGENTS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBAGENTS_SCRIPT_DIR))

from jev_decide import jev_decide

# The 4 Competing Scientific Hypotheses
HYPOTHESES = {
    "H1_hyperbolic_advection": {
        "title": "Hyperbolic Advection Transport",
        "statement": (
            "The ECR provides unitary, non-dissipative spatial transport c_i^{t+1} = c_{i-k}^t with velocity v = k. "
            "It overcomes the continuous NCA correlation length bound (xi ≈ 4.53 cells) by propagating discrete symbols "
            "ballistically rather than diffusively."
        )
    },
    "H2_receptive_field_dilation": {
        "title": "Receptive Field Dilation Shortcut",
        "statement": (
            "Stride k > 1 skip transport simply widens the effective receptive field per tick (like dilated convolution), "
            "bypassing intermediate cells without genuine continuous cellular dynamics."
        )
    },
    "H3_representation_decoupling": {
        "title": "Capacity / Representation Decoupling",
        "statement": (
            "The ECR advantage is primarily due to isolating persistent memory registers (C_carry) from transient non-linear "
            "token processing (C_hidden), preventing accumulated carries from being overwritten by local cell updates."
        )
    },
    "H4_gradient_highway": {
        "title": "Optimization Gradient Highway",
        "statement": (
            "Carry skip connections act as linear identity residual pathways that facilitate backpropagation through time "
            "over deep development horizons (T >= 16), eliminating vanishing/exploding gradients during training."
        )
    }
}

# Empirical Evidence Corpus
CUMULATIVE_EVIDENCE = [
    {
        "id": "E1_parameter_invariance",
        "name": "Parameter Invariance Audit",
        "data": (
            "Trainable parameters are strictly invariant at 43,715 across Cc in {0, 16, 32}. "
            "Carry channels do not add weights; they repartition total channels C=64. "
            "At Cc=32, local hidden capacity is halved (32 channels), yet accuracy improves from 90.2% to 100.0%. "
            "Advection shift is non-parametric (0 FLOPs, pure tensor slicing)."
        )
    },
    {
        "id": "E2_carry_probing_separation",
        "name": "Carry vs Hidden Decodability Separation",
        "data": (
            "Under zero-leakage balanced ripple chains (a_i + b_i = 9, untraind baseline = 50.0% chance): "
            "At d=0: Full State = 100%, Carry = 100%, Hidden = 100%. "
            "At d=4: Full State = 88.9%, Carry Register = 94.4%, Hidden State = 61.1% (+33.3% gap!). "
            "At d=8: Full State = 72.2%, Carry Register = 61.1%, Hidden State = 44.4% (hidden drops below chance). "
            "Spatial decodability decay of hidden state matches analytical continuous contraction length xi = 4.53."
        )
    },
    {
        "id": "E3_ood_length_lyapunov_divergence",
        "name": "OOD Length Generalization & Lyapunov Stability",
        "data": (
            "Trained at L=16, evaluated zero-shot up to L=64. "
            "Baseline Causal NCA dynamically scaled to T=68 suffers exponential Lyapunov divergence (Loss = 30.08, Acc = 15.7%). "
            "ECR-16 Bi dynamically scaled to T=68 remains strictly bounded (Loss = 8.75, Acc = 34.7%, 2.2x accuracy of baseline). "
            "Fixed T=16 exhibits exact lightcone reach cutoff: k=1 reaches d=16, k=2 reaches d=32 (retaining 61.1% at L=24)."
        )
    }
]


def run_tournament() -> dict[str, Any]:
    print("=" * 80)
    print("JEV SCIENTIFIC HYPOTHESIS TOURNAMENT: TITAN TEXT MECHANISMS")
    print("=" * 80)

    # 1. Evaluate Evidence-Hypothesis Compatibility Matrix
    print("\n[Stage 1] Assessing Evidence Compatibility for Each Hypothesis...")
    compatibility_matrix = {}

    for h_key, h_val in HYPOTHESES.items():
        compatibility_matrix[h_key] = {}
        print(f"\nEvaluating: {h_val['title']}")
        for ev in CUMULATIVE_EVIDENCE:
            schema = {
                "question_id": f"compat_{h_key}_{ev['id']}",
                "type": "choice",
                "instructions": (
                    f"Evaluate how well the empirical finding '{ev['name']}' supports, contradicts, "
                    f"or is explained by the hypothesis '{h_val['title']}'."
                ),
                "criteria": {
                    "strongly_supports": "The evidence directly corroborates this mechanism as the primary cause.",
                    "partially_consistent": "The evidence is consistent with this hypothesis, but could also support alternatives.",
                    "neutral": "The evidence neither supports nor challenges this hypothesis.",
                    "in_tension": "The evidence is difficult to reconcile with this hypothesis.",
                    "contradicts": "The evidence directly refutes or falsifies this hypothesis."
                },
                "default": "partially_consistent"
            }
            payload = {
                "hypothesis": h_val["statement"],
                "evidence": ev["data"]
            }
            res = jev_decide(f"compat_{h_key}_{ev['id']}", json.dumps(payload, indent=2), custom_schema=schema)
            decision = res.get("decision", "unknown")
            probs = res.get("probabilities", {})
            compatibility_matrix[h_key][ev["id"]] = {
                "decision": decision,
                "confidence": res.get("confidence", 0.0),
                "probabilities": probs
            }
            print(f"  • {ev['name']}: {decision} (conf: {res.get('confidence', 0.0):.2f})")

    # 2. Final Mechanism Arbitration Tournament
    print("\n[Stage 2] Running Multi-Hypothesis Arbitration...")
    summary_payload = {
        "hypotheses": {k: v["statement"] for k, v in HYPOTHESES.items()},
        "evidence": {e["id"]: e["data"] for e in CUMULATIVE_EVIDENCE},
        "compatibility": compatibility_matrix
    }

    schema_arbitration = {
        "question_id": "titan_nca_primary_mechanism",
        "type": "choice",
        "instructions": (
            "Given all empirical findings (parameter invariance, carry probing decodability curves, "
            "and OOD length Lyapunov stability), which hypothesis is the PRIMARY causal driver of the ECR advantage?"
        ),
        "criteria": {
            "H1_hyperbolic_advection": "Unitary hyperbolic advection overcoming continuous NCA spatial dissipation.",
            "H2_receptive_field_dilation": "Static receptive field dilation shortcut equivalent to dilated convolution.",
            "H3_representation_decoupling": "Memory scratchpad decoupling preventing destructive hidden overwrites.",
            "H4_gradient_highway": "Linear identity residual gradient highway accelerating BPTT optimization."
        },
        "default": "H1_hyperbolic_advection"
    }

    arb_res = jev_decide("titan_nca_primary_mechanism", json.dumps(summary_payload, indent=2), custom_schema=schema_arbitration)

    print("\n" + "=" * 80)
    print("TOURNAMENT VERDICT")
    print("=" * 80)
    print(f"Winning Hypothesis: {arb_res.get('decision')}")
    print(f"Confidence:         {arb_res.get('confidence', 0.0):.3f}")
    print("Posterior Distribution:")
    for k, p in sorted(arb_res.get("probabilities", {}).items(), key=lambda x: -x[1]):
        print(f"  {k:<30}: {p * 100:5.1f}%")

    out_file = Path("reports/jev_hypothesis_tournament.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "hypotheses": HYPOTHESES,
            "evidence": CUMULATIVE_EVIDENCE,
            "compatibility_matrix": compatibility_matrix,
            "arbitration": arb_res
        }, f, indent=2)
    print(f"\nSaved tournament results to {out_file}")
    return arb_res


if __name__ == "__main__":
    run_tournament()
