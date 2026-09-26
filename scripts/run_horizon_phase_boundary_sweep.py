#!/usr/bin/env python3
"""Empirical Horizon Phase Boundary & Causal Reach Sweep.

Evaluates the Ballistic Lightcone Causal Reach Theorem on Dyck-4:
  Predicted Law: T* >= 2 ceil(L / k)
  Tests whether crossing T* = 32 (at L=64, k=4) triggers a sharp phase boundary jump
  from ~43.9% (75% reach) to ~60% (100% causal closure), refuting Team B's compute scaling hypothesis.

Conditions Tested:
  - Arm A: CD-DV-NCA (k=4 skip stride, Cc=32, STE-sign) across T in {16, 24, 32, 40, 48}
  - Arm B: Baseline Causal NCA (k=1 nearest neighbor, Cc=0) across T in {16, 24, 32, 48}
  - Sequence Length: L=64
  - Multi-seed testing across seeds 42, 43, 44.

Output:
  reports/horizon_phase_boundary_sweep_results.json
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
TICKS = [16, 24, 32, 40, 48]
BATCH_SIZE = "16"

CHECKPOINT_CDDV = "checkpoints/dyck4_pushdown/cd_dv_nca_seed_42"
CHECKPOINT_BASE = "checkpoints/dyck4_pushdown/baseline_causal_seed_42"
REPORT_FILE = "reports/horizon_phase_boundary_sweep_results.json"


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
    """Extract Titan NCA loss and accuracy metrics from benchmark output."""
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


def main():
    print("=" * 80)
    print("TITAN TEXT: EMPIRICAL HORIZON PHASE BOUNDARY & CAUSAL REACH SWEEP (L=64)")
    print("Testing T* >= 2 ceil(L/k) Ballistic Causal Boundary vs Compute Hypothesis")
    print("=" * 80)

    seeds_arg = ",".join(str(s) for s in SEEDS)
    results = {
        "benchmark": "horizon_phase_boundary_dyck4",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": SEQ_LEN,
        "seeds": SEEDS,
        "cddv_sweep": {},
        "baseline_sweep": {},
        "sealed_predictions": {
            "cddv_k4": {
                "16": {"reach_ratio": 0.50, "acc_pred": 40.25},
                "24": {"reach_ratio": 0.75, "acc_pred": 50.88},
                "32": {"reach_ratio": 1.00, "acc_pred": 61.50},  # Causal closure
                "40": {"reach_ratio": 1.00, "acc_pred": 60.30},
                "48": {"reach_ratio": 1.00, "acc_pred": 59.10},
            },
            "baseline_k1": {
                "16": {"reach_ratio": 0.125, "acc_pred": 24.31},
                "24": {"reach_ratio": 0.188, "acc_pred": 26.97},
                "32": {"reach_ratio": 0.250, "acc_pred": 29.62},
                "48": {"reach_ratio": 0.375, "acc_pred": 34.94},
            }
        }
    }

    # Arm A: CD-DV-NCA (k=4) across T
    print("\n[Phase 1: CD-DV-NCA (k=4, Cc=32) Horizon Sweep across T in {16, 24, 32, 40, 48}]")
    for t_val in TICKS:
        r_val = min(1.0, (4 * t_val) / (2.0 * SEQ_LEN))
        cmd = [
            BIN, "benchmark",
            "--task", "dyck-pushdown",
            "--seq-len", str(SEQ_LEN),
            "--batch-size", BATCH_SIZE,
            "--dev-steps", str(t_val),
            "--seeds", seeds_arg,
            "--causal-stencil",
            "--zero-boundary",
            "--carry-channels", "32",
            "--carry-bidirectional",
            "--carry-skip-stride", "4",
            "--carry-quantization", "ste_sign",
        ]
        if os.path.exists(CHECKPOINT_CDDV):
            cmd.extend(["--checkpoint", CHECKPOINT_CDDV])

        code, out = run_cmd(cmd)
        parsed = parse_benchmark_output(out)
        acc = parsed.get("acc_mean", 0.0)
        acc_std = parsed.get("acc_std", 0.0)
        loss = parsed.get("loss_mean", 99.0)
        results["cddv_sweep"][str(t_val)] = {
            "reach_ratio": round(r_val, 3),
            "acc_mean": acc,
            "acc_std": acc_std,
            "loss_mean": loss,
        }
        closure_flag = "★ CAUSAL CLOSURE" if t_val == 32 else ("POST-CLOSURE" if t_val > 32 else "TRUNCATED")
        print(f"  T = {t_val:2d} ticks | Reach R = {r_val:5.3f} | Acc = {acc:5.2f}% ± {acc_std:4.2f}% | Loss = {loss:6.4f} | {closure_flag}")

    # Arm B: Baseline Causal NCA (k=1, Cc=0) across T
    print("\n[Phase 2: Baseline Causal NCA (k=1, Cc=0) Horizon Sweep across T in {16, 24, 32, 48}]")
    for t_val in [16, 24, 32, 48]:
        r_val = min(1.0, (1 * t_val) / (2.0 * SEQ_LEN))
        cmd = [
            BIN, "benchmark",
            "--task", "dyck-pushdown",
            "--seq-len", str(SEQ_LEN),
            "--batch-size", BATCH_SIZE,
            "--dev-steps", str(t_val),
            "--seeds", seeds_arg,
            "--causal-stencil",
            "--zero-boundary",
        ]
        if os.path.exists(CHECKPOINT_BASE):
            cmd.extend(["--checkpoint", CHECKPOINT_BASE])

        code, out = run_cmd(cmd)
        parsed = parse_benchmark_output(out)
        acc = parsed.get("acc_mean", 0.0)
        acc_std = parsed.get("acc_std", 0.0)
        loss = parsed.get("loss_mean", 99.0)
        results["baseline_sweep"][str(t_val)] = {
            "reach_ratio": round(r_val, 3),
            "acc_mean": acc,
            "acc_std": acc_std,
            "loss_mean": loss,
        }
        print(f"  T = {t_val:2d} ticks | Reach R = {r_val:5.3f} | Acc = {acc:5.2f}% ± {acc_std:4.2f}% | Loss = {loss:6.4f} | TRUNCATED")

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Master horizon phase boundary report written to {Path(REPORT_FILE).resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
