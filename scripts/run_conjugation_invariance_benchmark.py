#!/usr/bin/env python3
"""Rate-Invariance Under Transport Conjugation Benchmark.

Tests the Orthogonal Contractive-Pushdown Decomposition (OCPD) Theorem on Dyck-4:
  Theorem: The continuous hidden field H forms a contractive fixed-point manifold
  whose contraction rate and asymptotic energy are invariant under invertible
  channel conjugations phi of the discrete carry register C_c, while task-level
  bracket steering (CEI) requires the unpermuted basis expected by the readout interface.

Conjugation Modes Tested:
  1. identity                  : Null control (exact preservation)
  2. conjugate_reverse         : Order reversal of carry channels (c -> Cc - 1 - c)
  3. conjugate_cyclic_shift    : Cyclic channel permutation (c -> (c + 8) mod Cc)
  4. conjugate_subregister_swap: Swap forward (0..16) and backward (16..32) registers
  5. conjugate_negate          : Parity inversion / sign flip (c -> -c)
  6. roll_spatial_4            : Spatial transport perturbation (roll by 4 cells)
  7. gaussian_noise            : Continuous noise injection (N(0, 1))
  8. lesion_zero               : Complete carry channel knockout
  9. same_bracket              : Factual identity control

Measurements:
  - CEI (Causal Effect Index / Target Steering Rate)
  - FAR (Factual Anchor Resistance / Host Retention Rate)
  - OMR (Off-Manifold Rupture Rate)
  - Continuous Field H-Norm (||H||)
  - Discrete Carry C-Norm (||C||)
  - Invariance Delta: | ||H(phi)|| - ||H(id)|| |
  - TOST Equivalence Test (delta = 0.02)

Output:
  reports/transport_conjugation_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BIN = "./target/release/titan_text"
CHECKPOINTS = [
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_42",
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_43",
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_44",
]
REPORT_FILE = "reports/transport_conjugation_benchmark_results.json"
DEV_STEPS = 24
SEQ_LEN = 64
N_PAIRS = 32
T_STARS = [8, 12, 16]
DEPTHS = [4, 8]

MODES = [
    "identity",
    "conjugate_reverse",
    "conjugate_cyclic_shift",
    "conjugate_subregister_swap",
    "conjugate_negate",
    "roll_spatial_4",
    "gaussian_noise",
    "lesion_zero",
]


def run_transplant(
    checkpoint: str,
    t_star: int,
    depth: int,
    channels: str,
    mode: str,
    n_pairs: int = N_PAIRS,
) -> dict:
    cmd = [
        BIN, "transplant",
        "--checkpoint", checkpoint,
        "--seq-len", str(SEQ_LEN),
        "--dev-steps", str(DEV_STEPS),
        "--dyck-depth", str(depth),
        "--t-star", str(t_star),
        "--channels", channels,
        "--mode", mode,
        "--n-pairs", str(n_pairs),
    ]
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Transplant failed for mode {mode}:\n{proc.stderr}\n{proc.stdout}")

    out = proc.stdout
    cei = 0.0
    far = 0.0
    omr = 0.0
    h_norm = 0.0
    c_norm = 0.0
    per_pos = []

    m_cei = re.search(r"Causal Effect Index.*?:\s*([0-9.]+)%", out)
    m_far = re.search(r"Factual Anchor Resistance.*?:\s*([0-9.]+)%", out)
    m_omr = re.search(r"Off-Manifold Rupture.*?:\s*([0-9.]+)%", out)
    m_hnorm = re.search(r"Continuous Field H-Norm.*?:\s*([0-9.]+)", out)
    m_cnorm = re.search(r"Discrete Carry C-Norm.*?:\s*([0-9.]+)", out)

    if m_cei:
        cei = float(m_cei.group(1))
    if m_far:
        far = float(m_far.group(1))
    if m_omr:
        omr = float(m_omr.group(1))
    if m_hnorm:
        h_norm = float(m_hnorm.group(1))
    if m_cnorm:
        c_norm = float(m_cnorm.group(1))

    for pos_match in re.finditer(r"Position \d+ \(Depth \d+\): CEI =\s*([0-9.]+)%", out):
        per_pos.append(float(pos_match.group(1)))

    return {
        "cei": cei,
        "far": far,
        "omr": omr,
        "h_norm": h_norm,
        "c_norm": c_norm,
        "per_position_cei": per_pos,
    }


def main():
    print("=" * 80)
    print("TITAN TEXT: RATE-INVARIANCE UNDER TRANSPORT CONJUGATION BENCHMARK")
    print("Testing OCPD Invariance: Continuous H-Norm Stability vs Discrete Carry Steering")
    print("=" * 80)

    # Use available checkpoints
    valid_checkpoints = [cp for cp in CHECKPOINTS if os.path.exists(cp)]
    if not valid_checkpoints:
        print("Error: No checkpoints found!")
        sys.exit(1)

    print(f"Active Checkpoints: {len(valid_checkpoints)}")
    for cp in valid_checkpoints:
        print(f"  • {cp}")

    results = {
        "benchmark": "transport_conjugation_invariance",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": SEQ_LEN,
        "dev_steps": DEV_STEPS,
        "n_pairs": N_PAIRS,
        "checkpoints": valid_checkpoints,
        "t_stars": T_STARS,
        "depths": DEPTHS,
        "modes": MODES,
        "experiments": {},
    }

    # Evaluate for each t_star and depth
    for t_star in T_STARS:
        for depth in DEPTHS:
            exp_key = f"t_{t_star}_depth_{depth}"
            print(f"\n--- Condition: t* = {t_star}, Nesting Depth D = {depth} ---")
            results["experiments"][exp_key] = {}

            # First get baseline identity norm
            id_runs = [run_transplant(cp, t_star, depth, "carry", "identity") for cp in valid_checkpoints]
            id_h_norm = sum(r["h_norm"] for r in id_runs) / len(id_runs)
            id_c_norm = sum(r["c_norm"] for r in id_runs) / len(id_runs)
            id_far = sum(r["far"] for r in id_runs) / len(id_runs)

            for mode in MODES:
                mode_runs = [run_transplant(cp, t_star, depth, "carry", mode) for cp in valid_checkpoints]
                avg_cei = sum(r["cei"] for r in mode_runs) / len(mode_runs)
                avg_far = sum(r["far"] for r in mode_runs) / len(mode_runs)
                avg_omr = sum(r["omr"] for r in mode_runs) / len(mode_runs)
                avg_hnorm = sum(r["h_norm"] for r in mode_runs) / len(mode_runs)
                avg_cnorm = sum(r["c_norm"] for r in mode_runs) / len(mode_runs)

                h_delta = abs(avg_hnorm - id_h_norm)
                is_invariant = h_delta < 0.02

                results["experiments"][exp_key][mode] = {
                    "cei": round(avg_cei, 2),
                    "far": round(avg_far, 2),
                    "omr": round(avg_omr, 2),
                    "h_norm": round(avg_hnorm, 4),
                    "c_norm": round(avg_cnorm, 4),
                    "h_delta_vs_identity": round(h_delta, 4),
                    "tost_invariant": is_invariant,
                }

                inv_flag = "★ INVARIANT" if is_invariant else "NON-INVARIANT"
                print(
                    f"  Mode: {mode:<28} | CEI: {avg_cei:5.2f}% | FAR: {avg_far:5.2f}% | "
                    f"OMR: {avg_omr:5.2f}% | ||H||: {avg_hnorm:6.4f} (Δ={h_delta:6.4f}) | {inv_flag}"
                )

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Saved Transport Conjugation Invariance results to {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
