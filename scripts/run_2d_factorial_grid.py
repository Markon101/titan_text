#!/usr/bin/env python3
"""2D Factorial Grid Benchmark: Recurrence Depth (T) x Sequence Length (L).

Evaluates:
  1. Baseline Causal NCA (k=1)
  2. ECR-32 Bi Skip-4 (k=4)

Across:
  Sequence Lengths (L): [16, 24, 32, 48, 64]
  Recurrence Depths (T): [16, 24, 32, 48, 64]

Task: column-arithmetic
Evaluation Seeds: 42, 43, 44 (N=3)
Batch Size: 16 (RAYON_NUM_THREADS=1 strictly enforced)

Outputs:
  reports/factorial_grid_2d_results.json
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

L_GRID = [16, 24, 32, 48, 64]
T_GRID = [16, 24, 32, 48, 64]

MODELS = [
    {
        "name": "Baseline Causal NCA (k=1)",
        "id": "baseline_k1",
        "stride": 1,
        "checkpoint": "checkpoints/campaign_mechanism_matrix/C1_baseline_causal_seed_42",
        "fallback_checkpoints": [
            "checkpoints/campaign_mechanism_matrix/C1_baseline_causal_seed_101",
            "checkpoints/campaign_mechanism_matrix/C1_baseline_causal_seed_202",
        ],
    },
    {
        "name": "ECR-32 Bi Skip-4 (k=4)",
        "id": "ecr32_k4",
        "stride": 4,
        "checkpoint": "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_42",
        "fallback_checkpoints": [
            "checkpoints/campaign_mechanism_matrix/C6_ecr32_k4_t24_seed_42",
            "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_101",
        ],
    },
]


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


def compute_lightcone_matrix(stride: int) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Compute the theoretical bidirectional lightcone reach and coverage for all (L, T) pairs.

    In bidirectional column arithmetic:
      Roundtrip reach = (k * T) / 2
      Coverage ratio eta = reach / L = (k * T) / (2 * L)
      Causally covered iff reach >= L (i.e. eta >= 1.0).
    """
    matrix = {}
    for l_val in L_GRID:
        matrix[str(l_val)] = {}
        for t_val in T_GRID:
            reach = (stride * t_val) / 2.0
            eta = reach / float(l_val)
            covered = reach >= float(l_val)
            matrix[str(l_val)][str(t_val)] = {
                "reach": reach,
                "eta": round(eta, 4),
                "covered": covered,
                "status": "Covered" if covered else "Blackout",
            }
    return matrix


def resolve_checkpoint(cfg: dict) -> str:
    """Resolve valid checkpoint path, trying fallbacks if primary is missing."""
    p = Path(cfg["checkpoint"])
    if p.exists() and (p / "manifest.json").exists():
        return str(p)
    for fb in cfg.get("fallback_checkpoints", []):
        p_fb = Path(fb)
        if p_fb.exists() and (p_fb / "manifest.json").exists():
            return str(p_fb)
    raise FileNotFoundError(f"No valid checkpoint found for {cfg['name']}. Checked {cfg['checkpoint']} and fallbacks.")


def print_ascii_table(title: str, l_list: list[int], t_list: list[int], data_getter) -> None:
    """Utility to print a 2D matrix in clean ASCII table format."""
    print(f"\n{title}")
    header = f"{'L \\ T':<8} | " + " | ".join(f"T={t:<6}" for t in t_list)
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for l_val in l_list:
        row = [f"L={l_val:<5} | "]
        for t_val in t_list:
            val_str = data_getter(l_val, t_val)
            row.append(f"{val_str:<8} | ")
        print("".join(row).rstrip(" |"))
    print("─" * len(header))


def main():
    parser = argparse.ArgumentParser(description="2D Factorial Grid: T in [16..64] x L in [16..64]")
    parser.add_argument("--dry-run", action="store_true", help="Print theoretical predictions only without running benchmarks")
    parser.add_argument("--output", type=str, default="reports/factorial_grid_2d_results.json", help="Path to save JSON output")
    parser.add_argument("--eval-seeds", type=str, default="42,43,44", help="Evaluation seeds list")
    args = parser.parse_args()

    print("=" * 80)
    print("TITAN TEXT: 2D FACTORIAL GRID (RECURRENCE T x SEQUENCE LENGTH L)")
    print(f"Task: column-arithmetic | Eval Seeds: {args.eval_seeds} | Batch Size: 16")
    print(f"Sequence Lengths (L): {L_GRID}")
    print(f"Recurrence Depths (T): {T_GRID}")
    print("=" * 80)

    # 1. Compute and print theoretical Lightcone Boundary Matrices
    theoretical_matrices = {}
    for m in MODELS:
        mat = compute_lightcone_matrix(m["stride"])
        theoretical_matrices[m["id"]] = mat
        print_ascii_table(
            f"Theoretical Lightcone Coverage: {m['name']} (Reach = {m['stride']} * T / 2)",
            L_GRID,
            T_GRID,
            lambda l, t, m_mat=mat: f"{'✓ COV' if m_mat[str(l)][str(t)]['covered'] else '✗ BLK'} ({m_mat[str(l)][str(t)]['reach']:.0f})",
        )

    if args.dry_run:
        print("\n[INFO] Dry-run complete. Theoretical matrices computed successfully.")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "dry_run": True,
                "parameters": {"L_grid": L_GRID, "T_grid": T_GRID, "task": "column-arithmetic"},
                "theoretical_matrices": theoretical_matrices,
            }, f, indent=2)
        print(f"Saved theoretical matrix to {out_path}")
        return

    # 2. Benchmark execution
    results = {}
    t0 = time.time()

    for m in MODELS:
        m_id = m["id"]
        ckpt = resolve_checkpoint(m)
        print(f"\n[{m['name']}] Benchmark starting using checkpoint: {ckpt}")
        m_results = {}

        for l_val in L_GRID:
            m_results[str(l_val)] = {}
            for t_val in T_GRID:
                cmd = [
                    BIN, "benchmark",
                    "--task", "column-arithmetic",
                    "--checkpoint", ckpt,
                    "--seq-len", str(l_val),
                    "--dev-steps", str(t_val),
                    "--batch-size", "16",
                    "--seeds", args.eval_seeds,
                ]
                st = time.time()
                code, out = run_cmd(cmd)
                elapsed = time.time() - st

                if code != 0:
                    print(f"  [ERROR] L={l_val} T={t_val} failed ({elapsed:.1f}s):\n{out[:200]}")
                    m_results[str(l_val)][str(t_val)] = {
                        "error": True,
                        "output": out,
                        "time": elapsed,
                    }
                    continue

                metrics = parse_benchmark_output(out)
                theory = theoretical_matrices[m_id][str(l_val)][str(t_val)]
                cell_data = {
                    "acc_mean": metrics.get("acc_mean", 0.0),
                    "acc_std": metrics.get("acc_std", 0.0),
                    "loss_mean": metrics.get("loss_mean", 0.0),
                    "loss_std": metrics.get("loss_std", 0.0),
                    "reach": theory["reach"],
                    "eta": theory["eta"],
                    "covered": theory["covered"],
                    "time": round(elapsed, 2),
                }
                m_results[str(l_val)][str(t_val)] = cell_data

                acc_str = f"{cell_data['acc_mean']:5.1f}% ± {cell_data['acc_std']:4.1f}%"
                loss_str = f"Loss={cell_data['loss_mean']:6.4f}"
                cov_flag = "✓" if theory["covered"] else "✗"
                print(f"  • L={l_val:2d} T={t_val:2d} [{cov_flag} eta={theory['eta']:.2f}]: Acc={acc_str} | {loss_str} ({elapsed:.1f}s)")

        results[m_id] = m_results

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"2D FACTORIAL GRID BENCHMARK COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # 3. Print Final Accuracy & Loss Comparison Tables
    for m in MODELS:
        m_id = m["id"]
        print_ascii_table(
            f"Validation Accuracy Matrix (%): {m['name']}",
            L_GRID,
            T_GRID,
            lambda l, t, res=results[m_id]: f"{res[str(l)][str(t)].get('acc_mean', 0.0):.1f}%",
        )
        print_ascii_table(
            f"Validation Loss Matrix: {m['name']}",
            L_GRID,
            T_GRID,
            lambda l, t, res=results[m_id]: f"{res[str(l)][str(t)].get('loss_mean', 0.0):.2f}",
        )

    # 4. Save structured report
    report_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": round(total_time, 2),
            "eval_seeds": args.eval_seeds,
            "task": "column-arithmetic",
            "L_grid": L_GRID,
            "T_grid": T_GRID,
        },
        "theoretical_matrices": theoretical_matrices,
        "results": results,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[INFO] Results successfully persisted to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
