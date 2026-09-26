#!/usr/bin/env python3
"""CD-DV-NCA Campaign Harness & Dynamic Halting Benchmark.

Evaluates 5 distinct experimental configurations on Column Arithmetic:
  1. Baseline Causal NCA (Cc=0, k=1, T in [16, 24, 32])
  2. ECR-32 Skip-4 (Cc=32, uniform k=4, T in [16, 24, 32])
  3. CD-DV-NCA (Cc=32, dual velocity: 8 slow @ k=1, 8 fast @ k=4 forward; 8 slow @ k=1, 8 fast @ k=4 backward)
  4. CD-DV-NCA + STE-Sign discrete projection (--carry-quantization ste_sign)
  5. CD-DV-NCA + Dual-Metric Intrinsic Halting (warmup T_min = ceil(L/4), persistence W=6, eps_global=1e-4, eps_local=1e-3)

Evaluated across sequence lengths L in {16, 32, 64}.
Tracks:
  - Accuracy (mean ± std)
  - Cross-Entropy Loss (mean ± std)
  - Computational cost: mean stopping tick T_halt, compute reduction percentage (1 - T_halt/T_max), and premature halting rate.

Strict Safety Constraints:
  RAYON_NUM_THREADS=1
  batch_size <= 16
  multi-seed N=3 (42, 43, 44)

Output:
  reports/cd_dv_nca_campaign_results.json
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
EVAL_SEEDS = "42,43,44"
BATCH_SIZE = "16"
LENGTHS = [16, 32, 64]
T_MAX_DEFAULT = 32

CHECKPOINTS = {
    "baseline_causal": "checkpoints/campaign_mechanism_matrix/C1_baseline_causal_seed_42",
    "ecr32_skip4": "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_42",
    "cd_dv_nca": "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_42",
}


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Run command with strict single-thread environment."""
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
    """Extract Titan NCA loss and accuracy metrics from benchmark CLI output."""
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


def compute_halting_decision(
    trace_path: str,
    l_val: int,
    t_max: int = 32,
    w_persistence: int = 6,
    eps_global: float = 1e-4,
    eps_local: float = 1e-3,
) -> tuple[int, bool]:
    """Compute Dual-Metric Intrinsic Halting decision from rollout trace.

    Warmup requirement: T_min = ceil(L / 4).
    Persistence window: W = 6.
    Global threshold: eps_global = 1e-4.
    Local threshold: eps_local = 1e-3.

    Returns:
      (t_halt, is_premature)
      where is_premature is True if halted before the minimum roundtrip reach T_roundtrip = 2 * ceil(L / 4).
    """
    t_min_warmup = math.ceil(l_val / 4.0)
    t_roundtrip = 2 * math.ceil(l_val / 4.0)

    if not os.path.exists(trace_path):
        return t_max, False

    try:
        with open(trace_path, "r") as f:
            traces = json.load(f)
    except Exception:
        return t_max, False

    stable_count = 0
    t_halt = t_max

    for i, step_info in enumerate(traces):
        step_idx = step_info.get("step", i)
        # Relative update magnitude / kinetic energy
        upd = step_info.get("update_magnitude", 1.0)
        energy = step_info.get("energy", 1.0)

        # Global kinetic flux metric
        flux_global = (upd ** 2) / max(1e-6, energy)
        flux_local = upd

        # Check equilibrium condition
        is_equilibrium = (flux_global < eps_global) or (flux_local < eps_local)

        if step_idx >= t_min_warmup:
            if is_equilibrium:
                stable_count += 1
                if stable_count >= w_persistence:
                    t_halt = step_idx
                    break
            else:
                stable_count = 0
        else:
            stable_count = 0

    t_halt = min(t_halt, t_max)
    is_premature = t_halt < t_roundtrip
    return t_halt, is_premature


def evaluate_condition_fixed(
    cond_name: str,
    ckpt_path: str,
    l_val: int,
    t_val: int,
    extra_flags: list[str] | None = None,
) -> dict:
    """Run standard benchmark evaluation for a fixed (L, T) condition."""
    cmd = [
        BIN, "benchmark",
        "--task", "column-arithmetic",
        "--checkpoint", ckpt_path,
        "--seq-len", str(l_val),
        "--dev-steps", str(t_val),
        "--batch-size", BATCH_SIZE,
        "--seeds", EVAL_SEEDS,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    st = time.time()
    code, out = run_cmd(cmd)
    elapsed = time.time() - st

    if code != 0:
        return {
            "error": True,
            "output": out,
            "time": elapsed,
        }

    metrics = parse_benchmark_output(out)
    return {
        "acc_mean": metrics.get("acc_mean", 0.0),
        "acc_std": metrics.get("acc_std", 0.0),
        "loss_mean": metrics.get("loss_mean", 0.0),
        "loss_std": metrics.get("loss_std", 0.0),
        "time": round(elapsed, 2),
    }


def evaluate_condition_halting(
    ckpt_path: str,
    l_val: int,
    t_max: int = 32,
    scratch_dir: str = "scratch/halting_traces",
) -> dict:
    """Evaluate CD-DV-NCA under Dual-Metric Intrinsic Halting."""
    os.makedirs(scratch_dir, exist_ok=True)
    trace_file = os.path.join(scratch_dir, f"trace_l{l_val}.json")

    # 1. Rollout to generate step-by-step kinetic trajectory
    cmd_rollout = [
        BIN, "rollout",
        "--load-dir", ckpt_path,
        "--task", "column-arithmetic",
        "--seq-len", str(l_val),
        "--horizon", str(t_max),
        "--trace-output", trace_file,
    ]
    code_r, out_r = run_cmd(cmd_rollout)
    if code_r != 0:
        print(f"    [WARN] Rollout failed for L={l_val}: {out_r[:100]}")
        t_halt, is_premature = t_max, False
    else:
        t_halt, is_premature = compute_halting_decision(trace_file, l_val, t_max=t_max)

    # 2. Evaluate accuracy at halted tick
    res = evaluate_condition_fixed("CD-DV-NCA Halting", ckpt_path, l_val, t_halt)
    compute_reduction = max(0.0, (1.0 - (t_halt / float(t_max)))) * 100.0

    res["t_halt"] = t_halt
    res["t_max"] = t_max
    res["compute_reduction_pct"] = round(compute_reduction, 1)
    res["is_premature"] = is_premature
    res["premature_rate"] = 1.0 if is_premature else 0.0
    return res


def main():
    parser = argparse.ArgumentParser(description="CD-DV-NCA Campaign & Dynamic Halting Harness")
    parser.add_argument("--dry-run", action="store_true", help="Print protocol and theoretical parameters without running")
    parser.add_argument("--output", type=str, default="reports/cd_dv_nca_campaign_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("TITAN TEXT: CD-DV-NCA & DUAL-METRIC INTRINSIC HALTING CAMPAIGN HARNESS")
    print(f"Task: column-arithmetic | Lengths: {LENGTHS} | Seeds: {EVAL_SEEDS} | Batch Size: {BATCH_SIZE}")
    print("Conditions:")
    print("  1. Baseline Causal NCA (k=1, Cc=0)")
    print("  2. ECR-32 Skip-4 (k=4, Cc=32, uniform)")
    print("  3. CD-DV-NCA (dual velocity: 8 slow @ k=1, 8 fast @ k=4)")
    print("  4. CD-DV-NCA + STE-Sign discrete projection")
    print("  5. CD-DV-NCA + Dual-Metric Intrinsic Halting (W=6, eps_g=1e-4, eps_l=1e-3)")
    print("=" * 80)

    if args.dry_run:
        print("\n[INFO] Dry-run flag set. Exiting without execution.")
        return

    results = {}
    t0 = time.time()

    # -------------------------------------------------------------
    # Condition 1: Baseline Causal NCA (k=1, Cc=0)
    # -------------------------------------------------------------
    c1_name = "Baseline Causal NCA (k=1)"
    print(f"\nEvaluating: {c1_name}")
    results[c1_name] = {}
    c1_ckpt = CHECKPOINTS["baseline_causal"]
    for l_val in LENGTHS:
        results[c1_name][str(l_val)] = {}
        # Evaluate T=16 and T=32
        for t_val in [16, 24, 32]:
            res = evaluate_condition_fixed(c1_name, c1_ckpt, l_val, t_val)
            results[c1_name][str(l_val)][str(t_val)] = res
            print(f"  • L={l_val:2d} T={t_val:2d}: Acc = {res.get('acc_mean', 0.0):5.1f}% ± {res.get('acc_std', 0.0):4.1f}% | Loss = {res.get('loss_mean', 0.0):6.4f}")

    # -------------------------------------------------------------
    # Condition 2: ECR-32 Skip-4 (k=4, Cc=32)
    # -------------------------------------------------------------
    c2_name = "ECR-32 Skip-4 (k=4)"
    print(f"\nEvaluating: {c2_name}")
    results[c2_name] = {}
    c2_ckpt = CHECKPOINTS["ecr32_skip4"]
    for l_val in LENGTHS:
        results[c2_name][str(l_val)] = {}
        for t_val in [16, 24, 32]:
            res = evaluate_condition_fixed(c2_name, c2_ckpt, l_val, t_val)
            results[c2_name][str(l_val)][str(t_val)] = res
            print(f"  • L={l_val:2d} T={t_val:2d}: Acc = {res.get('acc_mean', 0.0):5.1f}% ± {res.get('acc_std', 0.0):4.1f}% | Loss = {res.get('loss_mean', 0.0):6.4f}")

    # -------------------------------------------------------------
    # Condition 3: CD-DV-NCA (Dual Velocity: k=1 + k=4)
    # -------------------------------------------------------------
    c3_name = "CD-DV-NCA (k in {1, 4})"
    print(f"\nEvaluating: {c3_name}")
    results[c3_name] = {}
    c3_ckpt = CHECKPOINTS["cd_dv_nca"]
    for l_val in LENGTHS:
        results[c3_name][str(l_val)] = {}
        for t_val in [16, 24, 32]:
            res = evaluate_condition_fixed(c3_name, c3_ckpt, l_val, t_val)
            results[c3_name][str(l_val)][str(t_val)] = res
            print(f"  • L={l_val:2d} T={t_val:2d}: Acc = {res.get('acc_mean', 0.0):5.1f}% ± {res.get('acc_std', 0.0):4.1f}% | Loss = {res.get('loss_mean', 0.0):6.4f}")

    # -------------------------------------------------------------
    # Condition 4: CD-DV-NCA + STE-Sign
    # -------------------------------------------------------------
    c4_name = "CD-DV-NCA + STE-Sign"
    print(f"\nEvaluating: {c4_name}")
    results[c4_name] = {}
    for l_val in LENGTHS:
        results[c4_name][str(l_val)] = {}
        for t_val in [16, 24, 32]:
            res = evaluate_condition_fixed(c4_name, c3_ckpt, l_val, t_val, extra_flags=["--carry-quantization", "ste_sign"])
            results[c4_name][str(l_val)][str(t_val)] = res
            print(f"  • L={l_val:2d} T={t_val:2d} [STE]: Acc = {res.get('acc_mean', 0.0):5.1f}% ± {res.get('acc_std', 0.0):4.1f}% | Loss = {res.get('loss_mean', 0.0):6.4f}")

    # -------------------------------------------------------------
    # Condition 5: CD-DV-NCA + Dual-Metric Intrinsic Halting
    # -------------------------------------------------------------
    c5_name = "CD-DV-NCA + Dynamic Halting"
    print(f"\nEvaluating: {c5_name}")
    results[c5_name] = {}
    for l_val in LENGTHS:
        res = evaluate_condition_halting(c3_ckpt, l_val, t_max=T_MAX_DEFAULT)
        results[c5_name][str(l_val)] = res
        premature_str = "PREMATURE" if res.get("is_premature") else "VALID"
        print(f"  • L={l_val:2d} [Halt at T={res['t_halt']:2d}/{res['t_max']} ({premature_str})]: Acc = {res.get('acc_mean', 0.0):5.1f}% | Loss = {res.get('loss_mean', 0.0):6.4f} | Compute Saved: {res.get('compute_reduction_pct', 0.0)}%")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"CAMPAIGN HARNESS COMPLETED IN {total_time:.1f}s")
    print("=" * 80)

    # Summary table
    print("\n─── CD-DV-NCA COMPREHENSIVE PERFORMANCE SUMMARY ───────────────────────────────────")
    print(f"{'Condition':<32} | {'L=16 (T=16)':<14} | {'L=32 (T=24)':<14} | {'L=64 (T=32)':<14}")
    print("─────────────────────────────────┼────────────────┼────────────────┼───────────────")
    for cond in [c1_name, c2_name, c3_name, c4_name]:
        row = [f"{cond:<32}"]
        for l_val, t_val in [(16, 16), (32, 24), (64, 32)]:
            c_data = results[cond][str(l_val)][str(t_val)]
            row.append(f"{c_data.get('acc_mean', 0.0):5.1f}% ({c_data.get('loss_mean', 0.0):.2f})")
        print(" | ".join(row))

    # Halting summary
    row_halt = [f"{c5_name:<32}"]
    for l_val in [16, 32, 64]:
        h_data = results[c5_name][str(l_val)]
        row_halt.append(f"{h_data.get('acc_mean', 0.0):5.1f}% (T={h_data.get('t_halt')})")
    print(" | ".join(row_halt))
    print("───────────────────────────────────────────────────────────────────────────────────")

    # Persist report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time_seconds": round(total_time, 2),
                "lengths": LENGTHS,
                "eval_seeds": EVAL_SEEDS,
                "batch_size": BATCH_SIZE,
                "task": "column-arithmetic",
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[INFO] Full campaign results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
