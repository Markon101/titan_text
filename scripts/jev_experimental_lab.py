#!/usr/bin/env python3
"""Jev Experimental Science Laboratory for Titan Text.

Advanced experimental suite using TypeSafe's Jev model on OpenRouter to:
1. Conduct Dynamic Bayesian Likelihood Updates across competing mechanisms as empirical data lands.
2. Arbitrate a 5-Way Next-Generation Architecture Tournament to determine the successor to ECR.
3. Perform Automated Adversarial Discrepancy Detection between theoretical bounds and empirical measurements.
4. Adjudicate the Subagent Cross-Examination Debate: 2-Velocity Split (k in {1,4}) vs 4-Octave Pyramid (k in {1,2,4,8}).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBAGENTS_SCRIPT_DIR = REPO_ROOT / ".agents" / "skills" / "openrouter-subagents" / "scripts"
if str(SUBAGENTS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBAGENTS_SCRIPT_DIR))

from jev_decide import jev_decide

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_bayesian_likelihood_update() -> dict[str, Any]:
    """Experiment 1: Full-Spectrum Bayesian Likelihood Updating across 4 Mechanism Hypotheses."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: JEV BAYESIAN MECHANISM LIKELIHOOD ARBITRATION")
    print("=" * 80)

    state = """EMPIRICAL EVIDENCE CORPUS (Direct Measurements on Titan Text):
1. Parameter & FLOP Invariance: Trainable parameters are IDENTICALLY 43,715 across C_c in {0, 16, 32}. At C_c=32, local hidden capacity is halved (32 channels), yet accuracy reaches 100%. Advection shift is non-parametric (0 FLOPs, pure tensor slicing). Perception filter is strictly 2-point causal {i-1, i} (no dilated weights exist in the codebase).
2. Carry Probing Separation (Zero-Leakage Balanced Chains, Chance=50%):
   - At d=0: Full State = 100%, Carry = 100%, Hidden = 100%.
   - At d=4: Full State = 88.9%, Carry = 94.4%, Hidden = 61.1% (+33.3% carry advantage).
   - At d=8: Full State = 72.2%, Carry = 61.1%, Hidden = 44.4% (Hidden collapses below chance 50%).
3. Causal Double Dissociation (do-calculus intervention at test time):
   - Intact: Acc = 63.1%, Loss = 2.1362.
   - Carry Lesion do(C=0): Acc collapses to 14.4% (-48.7%, chance=10%).
   - Hidden Lesion do(H=0): Acc collapses to 2.5%.
4. Zero-Shot OOD Length Generalization (L in {16, 24, 32, 48, 64}):
   - Baseline Causal NCA: Blackout at L > 16. Dynamic T(L)=68 causes catastrophic Lyapunov divergence: Loss = 30.08, Acc = 15.7%.
   - ECR-16 Bi (k=1): Dynamic T(L)=68 stays strictly bounded (Loss = 8.75, Acc = 34.7%, 2.2x accuracy of baseline).
   - ECR-32 Bi Skip-2 (k=2): Acc = 61.1% at L=24 (Loss 1.8066). Drops at L=32 under T=16 because roundtrip 2*ceil(32/2)=32 exceeds T=16.
   - ECR-32 Bi Skip-4 Deep: Solves L=64 under dynamic T=20 with 53.0% accuracy (Loss 2.2814).
   - STE-Sign Parity: Clamps cross-entropy loss flat between 0.84 and 0.89 across all lengths under Fixed T=16.
"""

    criteria = {
        "H1_advection_dominant": "Advection alone explains >80% of variance; decoupling is secondary.",
        "H3_decoupling_dominant": "Decoupling alone explains >80% of variance; advection is merely routing.",
        "H1_H3_unified_dual": "Unified Dual-Mechanism: Decoupling protects the symbol from nonlinear decay, while Advection transports it ballistically. Both are strictly necessary.",
        "H2_dilation_shortcut": "Empirical evidence supports dilated convolution shortcut over cellular transport.",
        "H4_gradient_highway": "Optimization ease (gradient flow) is the primary driver of all observed gains."
    }

    schema = {
        "question_id": "bayesian_mechanism_adjudication",
        "type": "choice",
        "instructions": (
            "Evaluate the posterior probability distribution over the mechanisms explaining ECR performance in Titan Text. "
            "Select the option that best reflects the empirical data."
        ),
        "criteria": criteria,
        "default": "H1_H3_unified_dual"
    }

    res = jev_decide("bayesian_mechanism_adjudication", state, custom_schema=schema)
    print(f"Jev Decision: {res.get('decision')} (Confidence: {res.get('confidence', 0.0):.2f})")
    print(f"Probabilities: {json.dumps(res.get('probabilities', {}), indent=2)}")
    return res


def run_next_gen_architecture_tournament() -> dict[str, Any]:
    """Experiment 2: Tournament Arbitrating 5 Proposed Next-Gen Titan Text Architectures."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: NEXT-GENERATION ARCHITECTURE TOURNAMENT")
    print("=" * 80)

    state = """We evaluate 5 architectural paradigms for the future of Titan Text:

CANDIDATE 1: M-ECR (Hierarchical Octave Multi-Velocity Advection)
- Channels partitioned into 4 octave bands: k in {1, 2, 4, 8}, each with 8 channels (32 carry channels total).
- Forward and backward advection across all 4 velocities simultaneously.
- Parameters: 43,715 (0 new weights). FLOPs: 0 extra (pure memory indexing).
- Roundtrip latency: O(log L) with base-2 pyramid.

CANDIDATE 2: CD-NCA (Continuous-Discrete Split NCA with Symplectic Projection)
- Formal bi-manifold state: H in R^{C_h} (dissipative semantic space) (+) C in {-1, 0, 1}^{C_c} (discrete symplectic manifold).
- Discrete update uses Straight-Through Estimator (STE) with strict discrete group operations.
- Guarantees bounded asymptotic energy lim_{T->inf} ||X(T)|| <= M, zero Lyapunov explosion.
- Parameters: 43,715.

CANDIDATE 3: SRC-NCA (Self-Routing Continuous-Velocity Carry Waves)
- Local cell computes dynamic velocity v_i(t) in [-k, +k] conditioned on local context.
- Implemented via continuous grid sampling / bilinear interpolation.
- Parameters: +1,024 weights (velocity prediction head). FLOPs: +15% per cell.

CANDIDATE 4: H-NCA (Hamiltonian / Symplectic Reversible NCA)
- State split into generalized coordinates (q, p). Update preserves phase space volume dq ^ dp = constant.
- Mathematically eliminates all dissipation and all explosion; perfectly energy-conserving.
- Challenging to parameterize non-linear language modeling functions within symplectic constraints.

CANDIDATE 5: HYBRID-NCA-BUS (Dual-Scale Local NCA + Global Latent Bus)
- Local 1D NCA cells augmented with a 1-token global summary vector broadcast to all cells every tick.
- Breaks strict cellular locality (introduces global communication channel).
- Highly effective for global sequence pooling, but violates pure cellular automaton physics.

CRITERIA FOR SELECTION:
1. Strict Locality & Cellular Purity (adheres to 1D NCA definition, no global attention/bus).
2. Parameter & FLOP Invariance (zero parameter bloat, 0 extra FLOPs on mobile/ARM).
3. Algorithmic Scaling (solves arbitrary-length ripple carry and long-range context).
4. Asymptotic Stability (bounded Lyapunov exponent under T -> inf).
5. Mobile / Termux ARM NEON hardware compatibility.
"""

    criteria = {
        "candidate_1_M_ECR": "Hierarchical Octave Multi-Velocity Advection (k in {1,2,4,8}) is the optimal next-gen design.",
        "candidate_2_CD_NCA": "Continuous-Discrete Split NCA with Symplectic STE projection is the optimal next-gen design.",
        "composite_M_CD_NCA": "COMPOSITE SYNTHESIS: Merge Candidate 1 and Candidate 2 into 'Continuous-Discrete Octave NCA' (CDO-NCA).",
        "candidate_3_SRC_NCA": "Self-Routing Continuous Velocity is superior despite parameter increase.",
        "candidate_4_H_NCA": "Hamiltonian Reversible NCA is the only mathematically sound solution.",
        "candidate_5_HYBRID_BUS": "Break pure locality and adopt Global Latent Bus for practical NLP performance."
    }

    schema = {
        "question_id": "next_gen_architecture_tournament",
        "type": "choice",
        "instructions": "Select the winning Next-Generation Architecture for Titan Text based on empirical and theoretical criteria.",
        "criteria": criteria,
        "default": "composite_M_CD_NCA"
    }

    res = jev_decide("next_gen_architecture_tournament", state, custom_schema=schema)
    print(f"Jev Decision: {res.get('decision')} (Confidence: {res.get('confidence', 0.0):.2f})")
    print(f"Probabilities: {json.dumps(res.get('probabilities', {}), indent=2)}")
    return res


def run_discrepancy_detection() -> dict[str, Any]:
    """Experiment 3: Automated Discrepancy & Anomaly Detection between Theory and Data."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: AUTOMATED ADVERSARIAL DISCREPANCY DETECTION")
    print("=" * 80)

    state = """AUDIT TARGETS:
OBSERVED DATA POINT 1:
Theoretical prediction: Bidirectional skip transport with stride k=2 should reach sequence length L=32 in T = 2 * ceil(32/2) = 32 ticks.
Empirical result under Fixed T=16: Accuracy at L=32 drops to 23.6% (Loss = 3.65).
Discrepancy: Is 23.6% an anomaly or does it exactly confirm the theoretical roundtrip threshold?

OBSERVED DATA POINT 2:
Theoretical prediction: STE discrete projection clamps spectrum to {-1, 0, 1}, completely halting Lyapunov explosion.
Empirical result: Cross-entropy loss under Fixed T=16 remains clamped between 0.84 and 0.89 across all lengths L in {16, 24, 32, 48, 64}.
However, accuracy hovers at 51-56% (near binary chance 50%).
Discrepancy: Why is loss perfectly clamped while accuracy is near chance when length exceeds reach? Is it numerical stability without information reach, or an artifact?

OBSERVED DATA POINT 3:
Under Dynamic T(L)=20 at L=64, ECR-32 Bi Skip-4 achieves 53.0% accuracy on 10-class digit addition (chance = 10.0%) with Loss = 2.2814, while Baseline Causal NCA dynamically scaled to T=68 collapses to Loss = 30.08 and 15.7% accuracy.
"""

    criteria = {
        "consistent_fully_explained": "Theoretical models (Bidirectional Roundtrip Law, STE Spectrum Clamping, Advection Reach) fully explain all observed data points with 0 contradictions.",
        "minor_reach_confound": "Accuracy hovering near chance under STE despite clamped loss is a reach-bounded information blackout, fully predicted by lightcone physics.",
        "unmodeled_boundary_aliasing": "Boundary reflection or phase-cancellation at sequence edges is causing unmodeled performance degradation.",
        "temporal_overfitting": "The model is overfitted to T=16 recurrence depth during training, limiting dynamic T generalization."
    }

    schema = {
        "question_id": "scientific_discrepancy_audit",
        "type": "choice",
        "instructions": "Determine whether theoretical predictions and empirical measurements are fully consistent or if an unmodeled anomaly exists.",
        "criteria": criteria,
        "default": "consistent_fully_explained"
    }

    res = jev_decide("scientific_discrepancy_audit", state, custom_schema=schema)
    print(f"Jev Decision: {res.get('decision')} (Confidence: {res.get('confidence', 0.0):.2f})")
    print(f"Probabilities: {json.dumps(res.get('probabilities', {}), indent=2)}")
    return res


def run_subagent_debate_arbitration() -> dict[str, Any]:
    """Experiment 4: Adjudicate the Subagent Cross-Examination Debate on Octave Strides."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SUBAGENT CROSS-EXAMINATION ARBITRATION")
    print("=" * 80)

    state = """DEBATE: HIERARCHICAL VELOCITY STRATEGY FOR TITAN TEXT NEXT-GEN

POSITION A (Velocity Architect & Theory Analyst):
- Implement full 4-Octave Pyramid: k in {1, 2, 4, 8} with 8 channels each (32 carry channels total).
- Justification: Bridges lengths up to L=128 in T=32 steps; mathematically equivalent to an embedded Neural Kogge-Stone parallel prefix adder. O(log L) communication complexity.

POSITION B (Adversarial Critic):
- Challenge: 32 carry channels divided by 2 directions (forward/backward) and 4 octaves leaves ONLY 4 CHANNELS per (k, direction).
- Hazard: 4 channels are insufficient to encode multi-digit ripple status and routing flags.
- Dispersion hazard: Different velocities create a desynchronized pulse train at downstream cells, causing phase decoherence and race conditions.
- Spatial aliasing: k=8 skips 7 cells; carries generated at intermediate cells cannot enter the fast bus.
- Counter-proposal: Implement a conservative 2-Velocity Split: k in {1, 4} with 8 channels per direction (16 forward: 8 at k=1, 8 at k=4; 16 backward: 8 at k=1, 8 at k=4).

HARDWARE AUDITOR INPUT:
- Naive Tensor::cat allocates 128 KB per step, evicting the 32-64 KB L1 cache on ARM Cortex-A mobile cores.
- A 2-velocity split has fewer memory slicing operations, easier SIMD/NEON register packing, and lower cache thrashing than a 4-octave split.
"""

    criteria = {
        "adopt_position_A_full_octave": "Adopt 4-Octave Pyramid (k in {1,2,4,8}): The O(log L) Kogge-Stone speedup outweighs channel capacity reduction.",
        "adopt_position_B_dual_velocity": "Adopt Position B (2-Velocity Split k in {1,4}): 8 channels per stream prevents channel starvation and eliminates dispersion race conditions while preserving fast ballistic transport.",
        "adaptive_learned_routing": "Neither: Make stride k dynamic or condition routing gates on carry confidence.",
        "hybrid_3_octave": "Compromise: 3-Octave split k in {1,2,4} with ~5-6 channels per stream."
    }

    schema = {
        "question_id": "velocity_strategy_adjudication",
        "type": "choice",
        "instructions": (
            "Arbitrate between Position A (4-Octave Kogge-Stone) and Position B (2-Velocity Split k in {1,4}). "
            "Consider representational channel capacity, dispersion risks, and ARM mobile cache constraints."
        ),
        "criteria": criteria,
        "default": "adopt_position_B_dual_velocity"
    }

    res = jev_decide("velocity_strategy_adjudication", state, custom_schema=schema)
    print(f"Jev Decision: {res.get('decision')} (Confidence: {res.get('confidence', 0.0):.2f})")
    print(f"Probabilities: {json.dumps(res.get('probabilities', {}), indent=2)}")
    return res


def main() -> None:
    print("Starting Jev Experimental Science Suite for Titan Text...")
    start_time = time.time()

    results = {}
    results["bayesian_mechanism"] = run_bayesian_likelihood_update()
    results["next_gen_architecture"] = run_next_gen_architecture_tournament()
    results["discrepancy_audit"] = run_discrepancy_detection()
    results["subagent_debate"] = run_subagent_debate_arbitration()

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    output_path = REPORTS_DIR / "jev_experimental_lab_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Jev Experimental Science Suite Complete in {elapsed:.2f}s!")
    print(f"Results written to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
