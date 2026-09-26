#!/usr/bin/env python3
"""RD-005: Causal Carry State Counterfactual Transplantation Benchmark.

Implements the formal adversarial double dissociation protocol on Dyck-4:
  1. Sweeps developmental tick t* in {4, 8, 12, 16, 20} on L=64, T=24.
  2. Disjoint sequence pairs A: (([[[... vs B: {{<<<...
  3. Splicing conditions:
     - Carry Transplant: Channels 32..64 swapped at t*
     - Hidden Transplant: Channels 0..32 swapped at t*
     - Full Transplant: Channels 0..64 swapped at t*
     - Same-Bracket Null Control: identical targets swapped at t*
     - Gaussian Noise Perturbation of carry channels at t*
     - Spatial Permutation Roll (roll_spatial_4) of carry channels at t*
  4. Metrics computed:
     - CEI (Causal Effect Index / Target B Steering Rate)
     - FAR (Factual Anchor Resistance / Host A Retention Rate)
     - OMR (Off-Manifold Rupture Rate / Dynamical Breakdown)
     - Per-position CEI profile across depths d in 1..D
  5. Output:
     reports/causal_carry_transplant_results.json
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
REPORT_FILE = "reports/causal_carry_transplant_results.json"
DEV_STEPS = 24
SEQ_LEN = 64
N_PAIRS = 32
T_STARS = [4, 8, 12, 16, 20]
DEPTHS = [2, 4, 8]


def run_transplant(checkpoint: str, t_star: int, depth: int, channels: str, mode: str, n_pairs: int = N_PAIRS) -> dict:
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
        raise RuntimeError(f"Transplant failed:\n{proc.stderr}\n{proc.stdout}")

    out = proc.stdout
    cei = 0.0
    far = 0.0
    omr = 0.0
    per_pos = []

    m_cei = re.search(r"Causal Effect Index.*?:\s*([0-9.]+)%", out)
    m_far = re.search(r"Factual Anchor Resistance.*?:\s*([0-9.]+)%", out)
    m_omr = re.search(r"Off-Manifold Rupture.*?:\s*([0-9.]+)%", out)

    if m_cei:
        cei = float(m_cei.group(1))
    if m_far:
        far = float(m_far.group(1))
    if m_omr:
        omr = float(m_omr.group(1))

    for pos_match in re.finditer(r"Position \d+ \(Depth \d+\): CEI =\s*([0-9.]+)%", out):
        per_pos.append(float(pos_match.group(1)))

    return {
        "cei": cei,
        "far": far,
        "omr": omr,
        "per_position_cei": per_pos,
    }


def main():
    print("=" * 80)
    print("TITAN TEXT: CAUSAL CARRY COUNTERFACTUAL TRANSPLANTATION BENCHMARK (RD-005)")
    print("Rigorous Double Dissociation & Steering Analysis across ticks t* and depths D")
    print("=" * 80)

    # Use seed 42 checkpoint by default, or first available
    cp = CHECKPOINTS[0]
    for c in CHECKPOINTS:
        if os.path.exists(c):
            cp = c
            break

    results = {
        "benchmark": "causal_carry_transplantation_rd005",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": cp,
        "seq_len": SEQ_LEN,
        "dev_steps": DEV_STEPS,
        "n_pairs": N_PAIRS,
        "experiments": {},
    }

    # 1. Temporal Window Sweep at D=4: Carry vs Hidden vs Full
    print("\n[Phase 1: Temporal Window Sweep (t* in {4, 8, 12, 16, 20}) at D=4]")
    print(f"{'t*':<4} | {'Condition':<18} | {'CEI (Steer B)':<15} | {'FAR (Retain A)':<15} | {'OMR (Rupture)'}")
    print("-" * 75)

    phase1_results = {}
    for t_star in T_STARS:
        phase1_results[str(t_star)] = {}
        conditions = [
            ("carry_transplant", "32..64", "pair_swap"),
            ("hidden_transplant", "0..32", "pair_swap"),
            ("full_transplant", "0..64", "pair_swap"),
        ]
        for cond_name, ch_range, mode in conditions:
            res = run_transplant(cp, t_star, depth=4, channels=ch_range, mode=mode)
            phase1_results[str(t_star)][cond_name] = res
            print(f"{t_star:<4} | {cond_name:<18} | {res['cei']:5.2f}%         | {res['far']:5.2f}%         | {res['omr']:5.2f}%")

    results["experiments"]["phase1_temporal_sweep"] = phase1_results

    # 2. Critical Controls at peak window t*=12, D=4
    print("\n[Phase 2: Critical Contrastive Controls at t*=12, D=4]")
    print(f"{'Control Condition':<26} | {'CEI (Steer B)':<15} | {'FAR (Retain A)':<15} | {'OMR (Rupture)'}")
    print("-" * 75)

    controls = [
        ("carry_transplant", "32..64", "pair_swap"),
        ("same_bracket_null", "32..64", "same_bracket"),
        ("gaussian_noise_carry", "32..64", "gaussian_noise"),
        ("spatial_roll_carry", "32..64", "roll_spatial_4"),
    ]
    phase2_results = {}
    for name, ch_range, mode in controls:
        res = run_transplant(cp, t_star=12, depth=4, channels=ch_range, mode=mode)
        phase2_results[name] = res
        print(f"{name:<26} | {res['cei']:5.2f}%         | {res['far']:5.2f}%         | {res['omr']:5.2f}%")

    results["experiments"]["phase2_contrastive_controls"] = phase2_results

    # 3. Depth Sweep at t*=12: D in {2, 4, 8}
    print("\n[Phase 3: Stratified Depth Carry Steering at t*=12 (D in {2, 4, 8})]")
    print(f"{'Depth D':<8} | {'CEI (Steer B)':<15} | {'FAR (Retain A)':<15} | {'OMR (Rupture)':<15} | {'Per-Pos CEI'}")
    print("-" * 75)

    phase3_results = {}
    for d in DEPTHS:
        res = run_transplant(cp, t_star=12, depth=d, channels="32..64", mode="pair_swap")
        phase3_results[str(d)] = res
        pos_str = ", ".join(f"{x:.1f}%" for x in res["per_position_cei"])
        print(f"D = {d:<4} | {res['cei']:5.2f}%         | {res['far']:5.2f}%         | {res['omr']:5.2f}%         | [{pos_str}]")

    results["experiments"]["phase3_depth_sweep"] = phase3_results

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Causal carry transplantation benchmark written to {Path(REPORT_FILE).resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
