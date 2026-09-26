#!/usr/bin/env python3
"""Double Dissociation & Interaction Matrix on Dyck-4.

Rigorous evaluation of Theorem 15 (Direct-Sum Grammar Decomposition):
  Tests whether z = W_H h + W_C c is purely additive or exhibits strong nonlinear cross-coupling.

Conditions Evaluated on Dyck-4 (L=64, T=24, seeds 42, 43, 44):
  1. Intact: Full model (Cc=32, Ch=32)
  2. Carry Lesion: Channels 32..63 ablated (Cc = 0, Ch intact)
  3. Hidden Lesion: Channels 0..31 ablated (Ch = 0, Cc intact)
  4. Joint Lesion: All channels 0..63 ablated (Ch = 0, Cc = 0)
  5. Shuffled Recurrence: Cross-batch shear null control
  6. Subspace Interaction Term: Delta_interaction = Acc_intact - (Acc_carry_lesion + Acc_hidden_lesion - Chance)

Output:
  reports/dyck_double_dissociation_results.json
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
REPORT_FILE = "reports/dyck_double_dissociation_results.json"


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


def parse_benchmark_output(out: str) -> dict[str, float]:
    """Extract Titan NCA loss and accuracy metrics."""
    res = {}
    m = re.search(
        r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        out,
    )
    if m:
        res["loss_mean"] = float(m.group(1))
        res["loss_std"] = float(m.group(2))
        res["acc_mean"] = float(m.group(3))
        res["acc_std"] = float(m.group(4))
        return res

    m_simple = re.search(r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%", out)
    if m_simple:
        res["loss_mean"] = float(m_simple.group(1))
        res["loss_std"] = 0.0
        res["acc_mean"] = float(m_simple.group(2))
        res["acc_std"] = 0.0
        return res
    return res


def evaluate_intervention(condition_name: str, extra_flags: list[str]) -> dict[str, float]:
    accs = []
    losses = []

    for cp in CHECKPOINTS:
        cmd = [
            BIN, "benchmark",
            "--task", "dyck-pushdown",
            "--checkpoint", cp,
            "--seq-len", str(SEQ_LEN),
            "--batch-size", BATCH_SIZE,
            "--dev-steps", str(DEV_STEPS),
            "--causal-stencil",
            "--zero-boundary",
            "--carry-channels", "32",
            "--carry-bidirectional",
            "--carry-skip-stride", "4",
            "--carry-quantization", "ste_sign",
            "--seeds", "100,101,102",
        ] + extra_flags

        code, out = run_cmd(cmd)
        if code == 0:
            parsed = parse_benchmark_output(out)
            if "acc_mean" in parsed:
                accs.append(parsed["acc_mean"])
                losses.append(parsed["loss_mean"])

    mean_acc = sum(accs) / len(accs) if accs else 0.0
    std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / len(accs)) if len(accs) > 1 else 0.0
    mean_loss = sum(losses) / len(losses) if losses else 99.0
    std_loss = math.sqrt(sum((x - mean_loss) ** 2 for x in losses) / len(losses)) if len(losses) > 1 else 0.0

    return {
        "acc_mean": round(mean_acc, 2),
        "acc_std": round(std_acc, 2),
        "loss_mean": round(mean_loss, 4),
        "loss_std": round(std_loss, 4),
    }


def main():
    print("=" * 80)
    print("TITAN TEXT: DYCK-4 DOUBLE DISSOCIATION & INTERACTION MATRIX (L=64)")
    print("Testing Theorem 15 Linear Grammar Direct-Sum vs Nonlinear Coupling")
    print("=" * 80)

    conditions = {
        "intact": [],
        "carry_lesion": ["--lesion-channels", ",".join(str(i) for i in range(32, 64))],
        "hidden_lesion": ["--lesion-channels", ",".join(str(i) for i in range(0, 32))],
        "joint_lesion": ["--lesion-channels", ",".join(str(i) for i in range(0, 64))],
        "batch_shuffle": ["--lesion-shuffle"],
    }

    results = {
        "benchmark": "dyck4_double_dissociation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": SEQ_LEN,
        "dev_steps": DEV_STEPS,
        "conditions": {},
    }

    for name, flags in conditions.items():
        print(f"  Evaluating Condition: {name:<16}...", end="", flush=True)
        res = evaluate_intervention(name, flags)
        results["conditions"][name] = res
        print(f" Acc = {res['acc_mean']:5.2f}% ± {res['acc_std']:4.2f}% | Loss = {res['loss_mean']:.4f}")

    # Theoretical Additivity vs Interaction Analysis
    acc_intact = results["conditions"]["intact"]["acc_mean"]
    acc_carry = results["conditions"]["carry_lesion"]["acc_mean"]
    acc_hidden = results["conditions"]["hidden_lesion"]["acc_mean"]
    chance = 25.0

    # Under linear independence: (Acc_intact - Chance) = (Acc_carry - Chance) + (Acc_hidden - Chance)
    signal_intact = acc_intact - chance
    signal_carry_preserved = acc_carry - chance  # Hidden state contribution
    signal_hidden_preserved = acc_hidden - chance  # Carry state contribution
    interaction_gap = signal_intact - (signal_carry_preserved + signal_hidden_preserved)

    results["interaction_analysis"] = {
        "chance_floor": chance,
        "total_signal_intact": round(signal_intact, 2),
        "signal_hidden_alone": round(signal_carry_preserved, 2),
        "signal_carry_alone": round(signal_hidden_preserved, 2),
        "linear_additive_sum": round(signal_carry_preserved + signal_hidden_preserved, 2),
        "nonlinear_synergy_gap": round(interaction_gap, 2),
        "is_approximately_additive": abs(interaction_gap) < 5.0,
    }

    print("\n" + "=" * 80)
    print("THEOREM 15 INTERACTION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Total Above-Chance Signal (Intact) : {signal_intact:+5.2f}%")
    print(f"  Continuous Hidden Subspace Signal   : {signal_carry_preserved:+5.2f}%")
    print(f"  Discrete Carry Subspace Signal      : {signal_hidden_preserved:+5.2f}%")
    print(f"  Linear Direct-Sum Prediction        : {signal_carry_preserved + signal_hidden_preserved:+5.2f}%")
    print(f"  Nonlinear Cross-Coupling Synergy    : {interaction_gap:+5.2f}%")
    print(f"  Direct-Sum Status                   : {'APPROXIMATELY ADDITIVE' if abs(interaction_gap) < 5.0 else 'STRONG NONLINEAR COUPLING'}")
    print("=" * 80)

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Double dissociation results written to {Path(REPORT_FILE).resolve()}")


if __name__ == "__main__":
    main()
