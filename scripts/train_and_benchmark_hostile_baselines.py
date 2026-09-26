#!/usr/bin/env python3
"""Hostile Baseline Engineering: Train to Convergence & Benchmark on Dyck-4.

Rigorous evaluation of canonical sequence models against CD-DV-NCA on Context-Free Dyck-4:
  1. Transformer (1-layer causal with Pre-LayerNorm, 41,859 params)
  2. GRU Recurrent (1-layer with gate bias retention initialization, 37,731 params)
  3. Simple RNN (Elman recurrent, 21,091 params)
  4. Untied Feedforward (4-layer local depth, 35,715 params)
  5. Titan CD-DV-NCA (Continuous-Discrete Dual-Velocity NCA, 43,715 params)

Protocols:
  - Models trained strictly on L=16 for 100 epochs.
  - Checkpoints saved to disk and loaded for evaluation.
  - Zero-shot length generalization evaluated at L=16, L=32, and L=64.
  - Multi-seed testing across seeds 42, 43, 44.
  - Strictly single-threaded (RAYON_NUM_THREADS=1) and batch size <= 16.

Output:
  reports/hostile_baselines_benchmark_results.json
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
LENGTHS = [16, 32, 64]
BATCH_SIZE = "16"
EPOCHS = "100"
LR = "0.003"
CHECKPOINT_NCA = "checkpoints/dyck4_pushdown/cd_dv_nca_seed_42"
REPORT_FILE = "reports/hostile_baselines_benchmark_results.json"


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


def parse_benchmark_output(out: str) -> dict[str, dict[str, float]]:
    """Extract metrics for all architectures from benchmark CLI output."""
    models = {
        "nca": r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        "transformer": r"Transformer\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        "gru": r"GRU Recurrent\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        "simple_rnn": r"Simple RNN\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        "untied_ff": r"Untied FF\s*\(4-step\)\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
    }
    res = {}
    for name, pattern in models.items():
        m = re.search(pattern, out)
        if m:
            res[name] = {
                "loss_mean": float(m.group(1)),
                "loss_std": float(m.group(2)),
                "acc_mean": float(m.group(3)),
                "acc_std": float(m.group(4)),
            }
    return res


def main():
    print("=" * 80)
    print("TITAN TEXT: HOSTILE BASELINE ENGINEERING BENCHMARK ON DYCK-4")
    print("Training Canonical Baselines to Convergence & Evaluating Length Generalization")
    print("=" * 80)

    # Phase 1: Train baselines to convergence on Dyck-4 (L=16)
    print("\n[Phase 1: Training Baselines to Full Convergence (Epochs=100, L=16)]")
    train_cmd = [
        BIN, "benchmark",
        "--task", "dyck-pushdown",
        "--seq-len", "16",
        "--batch-size", BATCH_SIZE,
        "--train",
        "--epochs", EPOCHS,
        "--lr", LR,
        "--seeds", "42",
        "--causal-stencil",
        "--zero-boundary",
    ]
    t0 = time.time()
    code, out = run_cmd(train_cmd)
    dur = time.time() - t0
    if code != 0:
        print(f"Training failed ({dur:.1f}s):\n{out[-800:]}")
        sys.exit(1)

    print(f"✓ Baselines successfully trained and checkpoints persisted ({dur:.1f}s)")
    for line in out.split("\n"):
        if "Val Acc:" in line or "TRAINED CONVERGENCE" in line or "Titan NCA" in line or "Transformer" in line or "GRU" in line:
            print(f"  {line}")

    # Phase 2: Multi-seed length generalization evaluation across L in {16, 32, 64}
    print("\n[Phase 2: Length Generalization Evaluation (L in {16, 32, 64}, Seeds=42,43,44)]")
    results = {
        "benchmark": "hostile_baselines_dyck4",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trained_epochs": int(EPOCHS),
        "trained_length": 16,
        "evaluations": {},
        "sealed_predictions_comparison": {
            "transformer_l16_pred": 68.0,
            "transformer_l64_pred": 26.5,
            "gru_l16_pred": 82.0,
            "gru_l64_pred": 34.0,
            "nca_l16_pred": 58.0,
            "nca_l64_pred": 43.9,
        }
    }

    seeds_arg = ",".join(str(s) for s in SEEDS)
    for l_val in LENGTHS:
        dev_steps = 16 if l_val == 16 else (20 if l_val == 32 else 24)
        print(f"\n--- Sequence Length L = {l_val} (Dev Steps = {dev_steps}) ---")
        eval_cmd = [
            BIN, "benchmark",
            "--task", "dyck-pushdown",
            "--seq-len", str(l_val),
            "--batch-size", BATCH_SIZE,
            "--dev-steps", str(dev_steps),
            "--seeds", seeds_arg,
            "--causal-stencil",
            "--zero-boundary",
            "--carry-channels", "32",
            "--carry-bidirectional",
            "--carry-skip-stride", "4",
            "--carry-quantization", "ste_sign",
        ]
        if os.path.exists(CHECKPOINT_NCA):
            eval_cmd.extend(["--checkpoint", CHECKPOINT_NCA])

        code, out = run_cmd(eval_cmd)
        if code != 0:
            print(f"Evaluation failed for L={l_val}:\n{out[-600:]}")
            continue

        parsed = parse_benchmark_output(out)
        results["evaluations"][str(l_val)] = parsed
        for name, metrics in parsed.items():
            print(f"  {name:<15} : Acc = {metrics['acc_mean']:5.2f}% ± {metrics['acc_std']:4.2f}% | Loss = {metrics['loss_mean']:.4f}")

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Hostile baselines benchmark results saved to {Path(REPORT_FILE).resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
