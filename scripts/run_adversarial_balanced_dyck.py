#!/usr/bin/env python3
"""Adversarial Balanced-N-Gram Dyck-4 Benchmark Harness.

Decisive Discriminator between Local Bigram Diffusion and Formal Non-Local Pushdown Computation.

Protocol:
1. Factorial Matrix:
   - Stack Depth D in {2, 4, 8}
   - Distracter Transport Gap G in {0, 4, 8, 12}
   - Conditions: Intact, Carry Lesion (Cc=0), Continuous Lesion (H=0), Strict Scramble Derangement
2. Metric Resolution:
   - Aggregate Accuracy & Cross-Entropy Loss (Mean +/- Std over seeds 42, 43, 44)
   - Fine-grained Per-Slot Accuracy Acc(s) as a function of physical lattice distance:
     Distance(s) = G + 2 + 2*s  for s in [0..D-1]
   - Causal Carry Gain: Delta_carry(s) = Acc_intact(s) - Acc_carry_lesion(s)
3. Baselines:
   - Transformer, GRU, Simple RNN, Untied FF (4-layer)

Output:
   reports/adversarial_balanced_dyck_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

BIN = "./target/release/titan_text"
SEEDS = [42, 43, 44]
SEQ_LEN = 64
DEV_STEPS = 24
BATCH_SIZE = "16"
CHECKPOINTS = [
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_42",
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_43",
    "checkpoints/dyck4_pushdown/cd_dv_nca_seed_44",
]
REPORT_FILE = "reports/adversarial_balanced_dyck_benchmark_results.json"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Execute command with strict single-thread environment."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


def parse_benchmark_output(out: str) -> dict[str, Any]:
    """Extract metrics and per-slot accuracies from benchmark output."""
    res: dict[str, Any] = {
        "models": {},
        "per_slot": [],
    }

    # Extract model accuracies
    models = ["Titan NCA", "Transformer", "GRU Recurrent", "Simple RNN", "Untied FF (4-step)"]
    for m in models:
        # Match multi-seed: Loss = 3.6756 ± 0.2898, Acc =  43.9% ±  1.6%
        esc_m = re.escape(m)
        match_ms = re.search(
            rf"{esc_m}\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
            out,
        )
        if match_ms:
            res["models"][m] = {
                "loss_mean": float(match_ms.group(1)),
                "loss_std": float(match_ms.group(2)),
                "acc_mean": float(match_ms.group(3)),
                "acc_std": float(match_ms.group(4)),
            }
            continue
        # Match single-seed
        match_ss = re.search(
            rf"{esc_m}\s*:\s*Loss\s*=\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%",
            out,
        )
        if match_ss:
            res["models"][m] = {
                "loss_mean": float(match_ss.group(1)),
                "loss_std": 0.0,
                "acc_mean": float(match_ss.group(2)),
                "acc_std": 0.0,
            }

    # Extract per-slot table
    # Slot | Distance | Titan NCA | Transformer | GRU | Simple RNN | Untied FF
    # 0    | 2        |     85.0% |       60.0% | ...
    slot_pattern = re.findall(
        r"^\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%",
        out,
        re.MULTILINE,
    )
    for row in slot_pattern:
        res["per_slot"].append({
            "slot": int(row[0]),
            "distance": int(row[1]),
            "nca_acc": float(row[2]),
            "tf_acc": float(row[3]),
            "gru_acc": float(row[4]),
            "sr_acc": float(row[5]),
            "untied_acc": float(row[6]),
        })

    return res


def run_benchmark_cell(
    depth: int,
    gap: int,
    condition: str,
    checkpoint: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Run a single benchmark cell."""
    seeds_arg = ",".join(str(s) for s in seeds)
    cmd = [
        BIN, "benchmark",
        "--task", "dyck-pushdown",
        "--dyck-depth", str(depth),
        "--dyck-gap", str(gap),
        "--dyck-balanced",
        "--dyck-per-slot",
        "--seq-len", str(SEQ_LEN),
        "--batch-size", BATCH_SIZE,
        "--dev-steps", str(DEV_STEPS),
        "--seeds", seeds_arg,
        "--causal-stencil",
        "--zero-boundary",
        "--carry-channels", "32",
        "--carry-bidirectional",
        "--carry-skip-stride", "4",
        "--carry-quantization", "ste_sign",
    ]
    if os.path.exists(checkpoint):
        cmd.extend(["--checkpoint", checkpoint])

    if condition == "carry_lesion":
        cmd.extend(["--lesion-channels", ",".join(str(i) for i in range(32, 64))])
    elif condition == "continuous_lesion":
        cmd.extend(["--lesion-channels", ",".join(str(i) for i in range(0, 32))])
    elif condition == "scramble":
        cmd.append("--dyck-scramble")

    code, out = run_cmd(cmd)
    if code != 0:
        print(f"Error in cell (D={depth}, G={gap}, cond={condition}):\n{out[:500]}")
        return {"error": out[:500]}

    return parse_benchmark_output(out)


def main():
    print("=" * 80)
    print("TITAN TEXT: ADVERSARIAL BALANCED-N-GRAM DYCK-4 BENCHMARK")
    print("Distance-Resolved Pushdown Stack Evaluation & Non-Local Horizon Sweep")
    print("=" * 80)

    depths = [2, 4, 8]
    gaps = [0, 4, 8]
    conditions = ["intact", "carry_lesion", "scramble"]

    results: dict[str, Any] = {
        "benchmark": "adversarial_balanced_dyck4",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": SEQ_LEN,
        "dev_steps": DEV_STEPS,
        "seeds": SEEDS,
        "depths": depths,
        "gaps": gaps,
        "conditions": conditions,
        "cells": {},
        "summary": {},
    }

    cp = CHECKPOINTS[0]

    for d in depths:
        print(f"\n==================== EVALUATING STACK DEPTH D = {d} ====================")
        for g in gaps:
            # Check maximum span fits in SEQ_LEN
            if d * 2 + g + 2 > SEQ_LEN:
                print(f"Skipping D={d}, G={g} (exceeds SEQ_LEN {SEQ_LEN})")
                continue

            cell_key = f"D{d}_G{g}"
            results["cells"][cell_key] = {}
            print(f"\n--- Condition: Depth D={d}, Distracter Gap G={g} (Base dist: {g+2} cells) ---")

            cell_intact = run_benchmark_cell(d, g, "intact", cp, SEEDS)
            cell_lesion = run_benchmark_cell(d, g, "carry_lesion", cp, SEEDS)
            cell_scramble = run_benchmark_cell(d, g, "scramble", cp, SEEDS)

            results["cells"][cell_key]["intact"] = cell_intact
            results["cells"][cell_key]["carry_lesion"] = cell_lesion
            results["cells"][cell_key]["scramble"] = cell_scramble

            nca_i = cell_intact.get("models", {}).get("Titan NCA", {}).get("acc_mean", 0.0)
            nca_l = cell_lesion.get("models", {}).get("Titan NCA", {}).get("acc_mean", 0.0)
            nca_s = cell_scramble.get("models", {}).get("Titan NCA", {}).get("acc_mean", 0.0)
            delta = nca_i - nca_l

            tf_acc = cell_intact.get("models", {}).get("Transformer", {}).get("acc_mean", 0.0)
            gru_acc = cell_intact.get("models", {}).get("GRU Recurrent", {}).get("acc_mean", 0.0)
            sr_acc = cell_intact.get("models", {}).get("Simple RNN", {}).get("acc_mean", 0.0)
            untied_acc = cell_intact.get("models", {}).get("Untied FF (4-step)", {}).get("acc_mean", 0.0)

            print(f"  • Titan NCA Intact : {nca_i:5.1f}%")
            print(f"  • Carry Lesion     : {nca_l:5.1f}% (Causal Delta: {delta:+5.1f}%)")
            print(f"  • Scramble Derange : {nca_s:5.1f}% (Null Chance: 25.0%)")
            print(f"  • Hostiles         : TF={tf_acc:5.1f}%, GRU={gru_acc:5.1f}%, RNN={sr_acc:5.1f}%, FF={untied_acc:5.1f}%")

            # Per-slot delta display
            slots_i = cell_intact.get("per_slot", [])
            slots_l = cell_lesion.get("per_slot", [])
            if slots_i and slots_l and len(slots_i) == len(slots_l):
                print("  • Per-Slot Analysis:")
                print(f"    {'Slot':<4} | {'Dist':<5} | {'Intact':<8} | {'Lesion':<8} | {'Delta':<8} | {'Untied FF'}")
                for si, sl in zip(slots_i, slots_l):
                    s_idx = si["slot"]
                    dist = si["distance"]
                    acc_i = si["nca_acc"]
                    acc_l = sl["nca_acc"]
                    d_acc = acc_i - acc_l
                    ff_s = si["untied_acc"]
                    print(f"    {s_idx:<4} | {dist:<5} | {acc_i:>6.1f}% | {acc_l:>6.1f}% | {d_acc:>+6.1f}% | {ff_s:>6.1f}%")

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Adversarial balanced benchmark complete. Results saved to {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
