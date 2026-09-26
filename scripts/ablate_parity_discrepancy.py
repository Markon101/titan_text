#!/usr/bin/env python3
"""Ablation study to resolve the 76.6% vs 56-58% iterated-parity discrepancy.

Tests:
1. Condition A: Historical task-667 (--causal-stencil, --state-norm bounded, --epochs 100)
2. Condition B: Ablate causal stencil (bilateral periodic, --state-norm bounded, --epochs 100)
3. Condition C: Ablate state norm (--causal-stencil, --state-norm none, --epochs 100)
4. Condition D: Ablate epochs (--causal-stencil, --state-norm bounded, --epochs 50)
5. Condition E: Drift campaign baseline (bilateral periodic, --state-norm none, --epochs 50)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BIN = "./target/release/titan_text"

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, proc.stdout

def parse_benchmark_output(out: str) -> dict[str, float]:
    res = {}
    m_multi = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%", out)
    if m_multi:
        res["loss_mean"] = float(m_multi.group(1))
        res["loss_std"] = float(m_multi.group(2))
        res["acc_mean"] = float(m_multi.group(3))
        res["acc_std"] = float(m_multi.group(4))
        return res
    m_single = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%", out)
    if m_single:
        res["loss_mean"] = float(m_single.group(1))
        res["loss_std"] = 0.0
        res["acc_mean"] = float(m_single.group(2))
        res["acc_std"] = 0.0
        return res
    return res

def parse_train_val_acc(out: str) -> dict[str, float]:
    res = {}
    m_train = re.search(r"Train Accuracy\s*:\s*([0-9.]+)%\s*\(Loss:\s*([0-9.]+)\)", out)
    if m_train:
        res["train_acc"] = float(m_train.group(1))
        res["train_loss"] = float(m_train.group(2))
    m_val = re.search(r"Val Accuracy\s*:\s*([0-9.]+)%\s*\(Loss:\s*([0-9.]+)\)", out)
    if m_val:
        res["single_batch_val_acc"] = float(m_val.group(1))
        res["single_batch_val_loss"] = float(m_val.group(2))
    return res

def main():
    conditions = [
        {
            "name": "Cond A: Hist task-667 (Causal + Bounded + 100ep)",
            "causal": True,
            "state_norm": "bounded",
            "epochs": 100,
        },
        {
            "name": "Cond B: Ablate Causal (Periodic + Bounded + 100ep)",
            "causal": False,
            "state_norm": "bounded",
            "epochs": 100,
        },
        {
            "name": "Cond C: Ablate State-Norm (Causal + None + 100ep)",
            "causal": True,
            "state_norm": "none",
            "epochs": 100,
        },
        {
            "name": "Cond D: Ablate Epochs (Causal + Bounded + 50ep)",
            "causal": True,
            "state_norm": "bounded",
            "epochs": 50,
        },
        {
            "name": "Cond E: Drift Campaign Baseline (Periodic + None + 50ep)",
            "causal": False,
            "state_norm": "none",
            "epochs": 50,
        },
    ]

    base_dir = "checkpoints/ablation_parity_discrepancy"
    os.makedirs(base_dir, exist_ok=True)
    report_file = Path("reports/ablation_parity_discrepancy.json")
    os.makedirs(report_file.parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: PARITY DISCREPANCY ABLATION STUDY (Phase 1)")
    print("Testing 5 conditions on iterated-parity (L=16, T=16, Cc=16, Seed 42)")
    print("=" * 80)

    results = []
    t0 = time.time()

    for idx, cond in enumerate(conditions):
        name = cond["name"]
        print(f"\n[{idx+1}/5] Running {name}...")
        s_dir = os.path.join(base_dir, f"cond_{idx+1}")
        os.makedirs(s_dir, exist_ok=True)

        cmd_train = [
            BIN, "train",
            "--task", "iterated-parity",
            "--save-dir", s_dir,
            "--epochs", str(cond["epochs"]),
            "--dev-steps", "16",
            "--seq-len", "16",
            "--batch-size", "32",
            "--lr", "0.003",
            "--seed", "42",
            "--carry-channels", "16",
            "--state-norm", cond["state_norm"],
        ]
        if cond["causal"]:
            cmd_train.append("--causal-stencil")

        st = time.time()
        code, out = run_cmd(cmd_train)
        dur = time.time() - st
        if code != 0:
            print(f"  [ERROR] Training failed:\n{out}")
            continue

        train_metrics = parse_train_val_acc(out)

        # 1. Single batch val eval (seed 42)
        cmd_bench_single = [
            BIN, "benchmark",
            "--task", "iterated-parity",
            "--checkpoint", s_dir,
            "--seq-len", "16",
            "--dev-steps", "16",
            "--batch-size", "32",
            "--seeds", "42",
        ]
        code_s, out_s = run_cmd(cmd_bench_single)
        single_metrics = parse_benchmark_output(out_s) if code_s == 0 else {}

        # 2. Multi-seed 5-batch val eval (seeds 42, 43, 44, 45, 46)
        cmd_bench_multi = [
            BIN, "benchmark",
            "--task", "iterated-parity",
            "--checkpoint", s_dir,
            "--seq-len", "16",
            "--dev-steps", "16",
            "--batch-size", "32",
            "--seeds", "42,43,44,45,46",
        ]
        code_m, out_m = run_cmd(cmd_bench_multi)
        multi_metrics = parse_benchmark_output(out_m) if code_m == 0 else {}

        res = {
            "condition": cond,
            "train_time": dur,
            "train_metrics": train_metrics,
            "single_batch_eval": single_metrics,
            "multi_batch_eval": multi_metrics,
        }
        results.append(res)

        single_acc = single_metrics.get("acc_mean", 0.0)
        multi_acc = multi_metrics.get("acc_mean", 0.0)
        multi_std = multi_metrics.get("acc_std", 0.0)
        print(f"  ✓ Finished in {dur:.1f}s | Single-Batch (Seed 42): {single_acc:5.1f}% | 5-Batch Mean: {multi_acc:5.1f}% ± {multi_std:4.1f}%")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ABLATION STUDY COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # Summary Table
    print("\n─── PARITY DISCREPANCY FACTOR ABLATION RESULTS ──────────────────────────────")
    print(f"{'Condition':<44} | {'Single Batch (Seed 42)':<22} | {'5-Batch Mean ± Std'}")
    print("─────────────────────────────────────────────┼────────────────────────┼─────────────────────")
    for r in results:
        c_name = r["condition"]["name"]
        sb_acc = r["single_batch_eval"].get("acc_mean", 0.0)
        mb_acc = r["multi_batch_eval"].get("acc_mean", 0.0)
        mb_std = r["multi_batch_eval"].get("acc_std", 0.0)
        print(f"{c_name:<44} | {sb_acc:5.1f}%                  | {mb_acc:5.1f}% ± {mb_std:4.1f}%")
    print("─────────────────────────────────────────────┴────────────────────────┴─────────────────────")

    with open(report_file, "w") as f:
        json.dump({"results": results, "total_time": total_time}, f, indent=2)
    print(f"Results saved to {report_file}")

if __name__ == "__main__":
    main()
