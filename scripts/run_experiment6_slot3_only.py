#!/usr/bin/env python3
"""Experiment 6: Slot-3-Only (Cell 15) Falsification Test.

Decisive test formulated by DeepSeek Adversarial Reviewer and Dynamics Agent:
  - Does the Causal Cellular Automaton fail on interior slots because of
    multi-slot loss gradient masking by the local 4-cell window at Slot 0?
  - If loss is supervised ONLY at Slot 3 (cell 15), the local shortcut on Slot 0
    is completely removed. The shared kernel is forced to either learn the
    16-cell sequential carry chain or fail entirely.

Executes on iterated-parity-dense with --causal-stencil and --target-slot 3
across N=5 seeds [42, 101, 202, 303, 404] for 100 epochs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BIN = "./target/release/titan_text"

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout

def main():
    seeds = [42, 101, 202, 303, 404]
    base_dir = "checkpoints/campaign_exp6_slot3_only"
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING EXPERIMENT 6: SLOT-3-ONLY (CELL 15) FALSIFICATION TEST")
    print("Task: iterated-parity-dense | Causal Stencil: True | Target Slot: 3 only")
    print("Seeds: [42, 101, 202, 303, 404] | Epochs: 100 | Recurrence T: 16")
    print("=" * 80)

    t0 = time.time()
    for s in seeds:
        s_dir = os.path.join(base_dir, f"seed_{s}")
        os.makedirs(s_dir, exist_ok=True)
        cmd = [
            BIN, "train",
            "--task", "iterated-parity-dense",
            "--save-dir", s_dir,
            "--epochs", "100",
            "--dev-steps", "16",
            "--seq-len", "16",
            "--seed", str(s),
            "--causal-stencil",
            "--target-slot", "3",
            "--zero-boundary",
        ]
        code, out = run_cmd(cmd)
        if code != 0:
            print(f"[Error] Training failed on seed {s}:\n{out}")
            sys.exit(1)
        print(f"  Seed {s} trained 100 epochs successfully.")

    t_train = time.time() - t0
    print(f"\nTraining completed in {t_train:.1f}s. Evaluating all seeds across all slots...")

    # Evaluate all seeds on all budgets and all slots
    tmp_dir = tempfile.gettempdir()
    seed_evals = {}
    seeds_str = ",".join(str(s) for s in seeds)

    for s in seeds:
        s_dir = os.path.join(base_dir, f"seed_{s}")
        report_file = os.path.join(tmp_dir, f"eval_s_{s}_{time.time_ns()}.json")
        cmd = [
            BIN, "sweep",
            "--load-dir", s_dir,
            "--budgets", "0,1,2,4,8,12,16",
            "--seq-len", "16",
            "--batch-size", "32",
            "--seeds", seeds_str,
            "--output", report_file,
            "--causal-stencil",
            "--zero-boundary",
        ]
        code, out = run_cmd(cmd)
        if code != 0:
            print(f"[Error] Evaluation failed on seed {s}:\n{out}")
            continue

        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                data = json.load(f)
            os.remove(report_file)

            # Extract budget 16 stats
            reports = data.get("reports", [data])
            b16_accs = []
            slot_accs = None
            for r in reports:
                for pt in r.get("budgets", []):
                    if pt["latent_ticks"] == 16:
                        b16_accs.append(pt["accuracy"] * 100.0)
                        slots = pt.get("per_slot_accuracy", [])
                        if slot_accs is None:
                            slot_accs = [0.0] * len(slots)
                        for si, sv in enumerate(slots):
                            slot_accs[si] += sv * 100.0
            n = len(b16_accs)
            mean_acc = sum(b16_accs) / max(1, n)
            mean_slots = [s / max(1, n) for s in slot_accs] if slot_accs else []
            seed_evals[str(s)] = {
                "b16_acc": mean_acc,
                "slots": mean_slots,
            }
            print(f"  Seed {s} B=16 Intact: {mean_acc:.2f}% | Slots: {[round(x, 1) for x in mean_slots]}")

    # Aggregate across seeds
    all_b16 = [seed_evals[str(s)]["b16_acc"] for s in seeds if str(s) in seed_evals]
    mean_all = sum(all_b16) / len(all_b16) if all_b16 else 0.0
    all_slots = [seed_evals[str(s)]["slots"] for s in seeds if str(s) in seed_evals]
    avg_slots = []
    if all_slots:
        for i in range(len(all_slots[0])):
            avg_slots.append(sum(s[i] for s in all_slots) / len(all_slots))

    print("\n" + "=" * 80)
    print(f"EXPERIMENT 6 TERMINAL RESULT (Supervised ONLY on Slot 3):")
    print(f"Overall Intact Accuracy across 4 slots: {mean_all:.2f}%")
    print(f"Per-Query-Slot Mean Accuracies: {[round(x, 1) for x in avg_slots]}")
    print(f"  Slot 0 (cell 3,  unsupervised during training): {avg_slots[0]:.1f}%")
    print(f"  Slot 1 (cell 7,  unsupervised during training): {avg_slots[1]:.1f}%")
    print(f"  Slot 2 (cell 11, unsupervised during training): {avg_slots[2]:.1f}%")
    print(f"  Slot 3 (cell 15, SUPERVISED during training):   {avg_slots[3]:.1f}%")
    print("=" * 80)

    out_file = Path("reports/campaign_exp6_slot3_only.json")
    with open(out_file, "w") as f:
        json.dump({
            "mean_b16": mean_all,
            "avg_slots": avg_slots,
            "seed_evals": seed_evals,
            "t_train": t_train,
        }, f, indent=2)
    print(f"Saved results to: {out_file}")

if __name__ == "__main__":
    main()
