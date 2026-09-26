#!/usr/bin/env python3
"""Empirical Extreme OOD (L=128) Horizon Phase Boundary & Causal Reach Sweep.

Tests the Ballistic Lightcone Causal Reach Law on Dyck-4 at L=128 (8x train length L=16):
  Predicted Reach: R(T, L, k) = min(1.0, (k * T) / (2 * L))
  For L=128 and k=4:
    T* = 2 * ceil(128 / 4) = 64 ticks (Causal Closure Point)

Conditions Tested:
  - Arm A: CD-DV-NCA (k=4 skip stride, Cc=32, STE-sign) across T in {16, 24, 32, 48, 64, 80}
  - Arm B: Baseline Causal NCA (k=1 nearest neighbor, Cc=0) across T in {16, 32, 64}
  - Sequence Length: L=128
  - Multi-seed testing across seeds 42, 43, 44

Output:
  reports/ood_128_causal_reach_sweep_results.json
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
SEEDS = [42, 43, 44]
SEQ_LEN = 128
TICKS = [16, 24, 32, 48, 64, 80]
BATCH_SIZE = "16"

CHECKPOINT_CDDV = "checkpoints/dyck4_pushdown/cd_dv_nca_seed_42"
CHECKPOINT_BASE = "checkpoints/dyck4_pushdown/baseline_causal_seed_42"
REPORT_FILE = "reports/ood_128_causal_reach_sweep_results.json"


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
    print("TITAN TEXT: EXTREME OOD (L=128) CAUSAL REACH & HORIZON PHASE BOUNDARY SWEEP")
    print("Testing T* >= 2 ceil(L/k) = 64 ticks Ballistic Causal Closure on L=128")
    print("=" * 80)

    seeds_arg = ",".join(str(s) for s in SEEDS)
    results = {
        "benchmark": "extreme_ood_128_causal_reach",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_len": SEQ_LEN,
        "seeds": SEEDS,
        "cddv_sweep": {},
        "baseline_sweep": {},
        "theoretical_predictions": {
            "cddv_k4": {
                "16": {"reach_ratio": 0.250, "acc_regime": "chance_floor (25-30%)"},
                "24": {"reach_ratio": 0.375, "acc_regime": "sub_reach (30-35%)"},
                "32": {"reach_ratio": 0.500, "acc_regime": "half_reach (35-40%)"},
                "48": {"reach_ratio": 0.750, "acc_regime": "partial_closure (40-48%)"},
                "64": {"reach_ratio": 1.000, "acc_regime": "causal_closure (50-60%)"},
                "80": {"reach_ratio": 1.000, "acc_regime": "post_closure_saturation"},
            },
            "baseline_k1": {
                "16": {"reach_ratio": 0.062, "acc_regime": "chance_floor (25%)"},
                "32": {"reach_ratio": 0.125, "acc_regime": "chance_floor (25%)"},
                "64": {"reach_ratio": 0.250, "acc_regime": "chance_floor (25%)"},
            }
        }
    }

    # Arm A: CD-DV-NCA (k=4) across T
    print("\n[Phase 1: CD-DV-NCA (k=4, Cc=32, STE-sign) Horizon Sweep across T on L=128]")
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
        closure_flag = "★ CAUSAL CLOSURE (R=1.0)" if t_val == 64 else ("POST-CLOSURE" if t_val > 64 else f"TRUNCATED (R={r_val:.2f})")
        print(f"  T = {t_val:2d} ticks | Reach R = {r_val:5.3f} | Acc = {acc:5.2f}% ± {acc_std:4.2f}% | Loss = {loss:6.4f} | {closure_flag}")

    # Arm B: Baseline Causal NCA (k=1, Cc=0) across T
    print("\n[Phase 2: Baseline Causal NCA (k=1, Cc=0) Horizon Sweep across T on L=128]")
    for t_val in [16, 32, 64]:
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
    print(f"✓ Saved L=128 Horizon Sweep results to {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
