#!/usr/bin/env python3
"""Column Arithmetic Scaling Campaign.

Evaluates carry capacity (Cc in {16, 32}), recurrence horizon (T in {16, 24}),
bidirectional carry, and skip stride (k in {1, 2, 4}) on column-arithmetic.
Tests whether scaling carry velocity and channels surpasses the Transformer baseline (64.4%).
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

def parse_train_output(out: str) -> dict[str, float]:
    """Parse final train and val metrics from training output."""
    res = {}
    m_train = re.search(r"Train Accuracy\s*:\s*([0-9.]+)%\s*\(Loss:\s*([0-9.]+)\)", out)
    if m_train:
        res["train_acc"] = float(m_train.group(1))
        res["train_loss"] = float(m_train.group(2))
    m_val = re.search(r"Val Accuracy\s*:\s*([0-9.]+)%\s*\(Loss:\s*([0-9.]+)\)", out)
    if m_val:
        res["val_acc"] = float(m_val.group(1))
        res["val_loss"] = float(m_val.group(2))
    return res

def parse_benchmark_output(out: str) -> dict[str, float]:
    """Parse multi-seed Titan NCA performance from benchmark output."""
    res = {}
    m = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%", out)
    if m:
        res["val_loss_mean"] = float(m.group(1))
        res["val_loss_std"] = float(m.group(2))
        res["val_acc_mean"] = float(m.group(3))
        res["val_acc_std"] = float(m.group(4))
    return res

def main():
    configs = [
        {
            "name": "ECR-16 Bi (k=1, T=16)",
            "carry_channels": 16,
            "skip_stride": 1,
            "bidirectional": True,
            "dev_steps": 16,
            "epochs": 100,
        },
        {
            "name": "ECR-16 Bi Skip-2 (k=2, T=16)",
            "carry_channels": 16,
            "skip_stride": 2,
            "bidirectional": True,
            "dev_steps": 16,
            "epochs": 100,
        },
        {
            "name": "ECR-32 Bi Skip-2 (k=2, T=16)",
            "carry_channels": 32,
            "skip_stride": 2,
            "bidirectional": True,
            "dev_steps": 16,
            "epochs": 100,
        },
        {
            "name": "ECR-32 Bi Skip-2 Deep (k=2, T=24)",
            "carry_channels": 32,
            "skip_stride": 2,
            "bidirectional": True,
            "dev_steps": 24,
            "epochs": 100,
        },
        {
            "name": "ECR-32 Bi Skip-4 Deep (k=4, T=24)",
            "carry_channels": 32,
            "skip_stride": 4,
            "bidirectional": True,
            "dev_steps": 24,
            "epochs": 100,
        },
    ]

    seeds = [42, 101, 202]
    eval_seeds_str = "42,43,44,45,46"
    base_dir = "checkpoints/campaign_column_arithmetic"
    os.makedirs(base_dir, exist_ok=True)
    report_file = Path("reports/column_arithmetic_campaign.json")
    os.makedirs(report_file.parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: COLUMN ARITHMETIC ARCHITECTURAL SCALING CAMPAIGN")
    print(f"Configs: {len(configs)} | Seeds: {seeds} | Eval Seeds: {eval_seeds_str}")
    print("Transformer Reference: 64.4% | Standard NCA: 49.4% | Dummy Chance: 45.0%")
    print("=" * 80)

    campaign_results = []
    t0 = time.time()

    for cfg in configs:
        name = cfg["name"]
        print(f"\nEvaluating Config: {name}")
        seed_results = []

        for s in seeds:
            s_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
            s_dir = os.path.join(base_dir, f"{s_name}_seed_{s}")
            os.makedirs(s_dir, exist_ok=True)

            cmd_train = [
                BIN, "train",
                "--task", "column-arithmetic",
                "--save-dir", s_dir,
                "--epochs", str(cfg["epochs"]),
                "--dev-steps", str(cfg["dev_steps"]),
                "--seq-len", "16",
                "--batch-size", "32",
                "--lr", "0.003",
                "--seed", str(s),
                "--carry-channels", str(cfg["carry_channels"]),
                "--carry-skip-stride", str(cfg["skip_stride"]),
            ]
            if cfg["bidirectional"]:
                cmd_train.append("--carry-bidirectional")

            st = time.time()
            code, out = run_cmd(cmd_train)
            dur = time.time() - st
            if code != 0:
                print(f"  [ERROR] Training failed for {name} seed {s}:\n{out}")
                continue

            train_metrics = parse_train_output(out)

            # Evaluate across 5 test seeds using benchmark
            cmd_bench = [
                BIN, "benchmark",
                "--task", "column-arithmetic",
                "--checkpoint", s_dir,
                "--seq-len", "16",
                "--dev-steps", str(cfg["dev_steps"]),
                "--batch-size", "32",
                "--seeds", eval_seeds_str,
            ]
            code_b, out_b = run_cmd(cmd_bench)
            eval_metrics = parse_benchmark_output(out_b) if code_b == 0 else {}

            val_acc = eval_metrics.get("val_acc_mean", train_metrics.get("val_acc", 0.0))
            train_acc = train_metrics.get("train_acc", 0.0)
            print(f"  ✓ Seed {s} (Time: {dur:.1f}s): Train Acc = {train_acc:5.1f}%, 5-Seed Val Acc = {val_acc:5.1f}%")
            seed_results.append({
                "seed": s,
                "train_time": dur,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "val_acc": val_acc,
            })

        accs = [sr["val_acc"] for sr in seed_results]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        var_acc = sum((x - mean_acc) ** 2 for x in accs) / max(1, len(accs) - 1) if accs else 0.0
        std_acc = var_acc ** 0.5

        campaign_results.append({
            "config": cfg,
            "seed_results": seed_results,
            "mean_val_acc": mean_acc,
            "std_val_acc": std_acc,
        })
        print(f"  --> {name} Final: {mean_acc:5.1f}% ± {std_acc:4.1f}%")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"COLUMN ARITHMETIC CAMPAIGN COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # Print Comparison Table
    print("\n─── COLUMN ARITHMETIC COMPARATIVE BENCHMARK ──────────────────────────────")
    print(f"{'Architecture / Config':<34} | {'Val Acc (Mean ± Std)':<22} | {'Delta vs Transformer'}")
    print("───────────────────────────────────┼────────────────────────┼─────────────────────")
    print(f"{'Dummy Baseline':<34} | {'45.0%':<22} | {'-19.4%':<21}")
    print(f"{'Standard NCA (T=8, Cc=0)':<34} | {'49.4%':<22} | {'-15.0%':<21}")
    print(f"{'Simple RNN':<34} | {'58.6%':<22} | {' -5.8%':<21}")
    print(f"{'Transformer (Ceiling)':<34} | {'64.4%':<22} | {'  0.0%':<21}")
    print(f"{'GRU Recurrent':<34} | {'70.3%':<22} | {' +5.9%':<21}")
    print("───────────────────────────────────┼────────────────────────┼─────────────────────")
    for cr in campaign_results:
        cfg_name = cr["config"]["name"]
        m = cr["mean_val_acc"]
        s = cr["std_val_acc"]
        delta = m - 64.4
        sign = "+" if delta >= 0 else ""
        print(f"{cfg_name:<34} | {m:5.1f}% ± {s:4.1f}%          | {sign}{delta:5.1f}%")
    print("──────────────────────────────────────────────────────────────────────────")

    with open(report_file, "w") as f:
        json.dump({"results": campaign_results, "total_time": total_time}, f, indent=2)
    print(f"Results saved to {report_file}")

if __name__ == "__main__":
    main()
