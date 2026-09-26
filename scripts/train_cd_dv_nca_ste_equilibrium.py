#!/usr/bin/env python3
"""Train CD-DV-NCA End-to-End with Native STE-Sign + Tail Equilibrium.

Executes:
1. Training Phase:
   - Task: column-arithmetic
   - Architecture: CD-DV-NCA (Cc=32, k=4 skip stride, bidirectional)
   - Quantization: Native STE-Sign (from epoch 0)
   - Regularization: Tail Equilibrium (weight=0.1, ticks=4)
   - Epochs: 50, Dev Steps: 20, Seq Len: 16, Batch Size: 16, LR: 0.003
   - Seeds: N=3 (42, 43, 44)

2. Evaluation Phase (L in {16, 32, 64}):
   - Accuracy & Loss scaling (check if L=64 reaches/exceeds 55%)
   - Kinetic Velocity Decay (check if Delta < 0.005 absorbing fixed point)
   - Dual-Metric Intrinsic Halting (warmup ceil(L/4), persistence W=6)
   - Compute Reduction Percentage & Premature Halting Rate

Safety: RAYON_NUM_THREADS=1 enforced.
Output: reports/cd_dv_nca_ste_equilibrium_results.json
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
TRAIN_SEEDS = [42, 43, 44]
EVAL_SEEDS = "42,43,44"
LENGTHS = [16, 32, 64]
BATCH_SIZE = "16"
EPOCHS = "50"
DEV_STEPS = "20"
LR = "0.003"
T_MAX = 32

BASE_SAVE_DIR = "checkpoints/campaign_cd_dv_ste_equilibrium"
REPORT_FILE = "reports/cd_dv_nca_ste_equilibrium_results.json"
SCRATCH_DIR = "scratch/ste_equilibrium_traces"


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


def analyze_kinetic_trajectory(trace_path: str, l_val: int) -> dict:
    """Analyze step-by-step kinetic velocity Delta and halting convergence.

    Checks:
    - Delta decay: Does tail velocity Delta < 0.005?
    - Dual-metric halting with warmup T_min = ceil(L/4), persistence W=6.
    """
    t_min_warmup = math.ceil(l_val / 4.0)
    t_roundtrip = 2 * math.ceil(l_val / 4.0)

    if not os.path.exists(trace_path):
        return {
            "initial_delta": 0.0,
            "tail_delta": 0.0,
            "is_fixed_point": False,
            "t_halt": T_MAX,
            "is_premature": False,
        }

    try:
        with open(trace_path, "r") as f:
            traces = json.load(f)
    except Exception:
        return {
            "initial_delta": 0.0,
            "tail_delta": 0.0,
            "is_fixed_point": False,
            "t_halt": T_MAX,
            "is_premature": False,
        }

    deltas = [t.get("update_magnitude", 0.0) for t in traces]
    energies = [t.get("energy", 0.0) for t in traces]

    initial_delta = sum(deltas[:4]) / max(1, len(deltas[:4])) if deltas else 0.0
    tail_window = deltas[-4:] if len(deltas) >= 4 else deltas
    tail_delta = sum(tail_window) / max(1, len(tail_window)) if tail_window else 0.0

    is_fixed_point = tail_delta < 0.005

    # Dual-Metric Intrinsic Halting evaluation
    stable_count = 0
    t_halt = T_MAX

    for i, step_info in enumerate(traces):
        step_idx = step_info.get("step", i)
        upd = step_info.get("update_magnitude", 1.0)
        energy = step_info.get("energy", 1.0)
        flux_global = (upd ** 2) / max(1e-6, energy)
        flux_local = upd

        # Halting thresholds: eps_global = 1e-4, eps_local = 1e-3 (or upd < 0.005)
        is_equilibrium = (flux_global < 1e-4) or (flux_local < 0.005)

        if step_idx >= t_min_warmup:
            if is_equilibrium:
                stable_count += 1
                if stable_count >= 6:
                    t_halt = step_idx
                    break
            else:
                stable_count = 0
        else:
            stable_count = 0

    t_halt = min(t_halt, T_MAX)
    is_premature = t_halt < t_roundtrip

    return {
        "initial_delta": round(initial_delta, 5),
        "tail_delta": round(tail_delta, 5),
        "is_fixed_point": is_fixed_point,
        "t_halt": t_halt,
        "is_premature": is_premature,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate native CD-DV-NCA with STE-sign + tail equilibrium")
    parser.add_argument("--dry-run", action="store_true", help="Print protocol without executing training")
    args = parser.parse_args()

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: NATIVE CD-DV-NCA (STE-SIGN + TAIL EQUILIBRIUM)")
    print("Task: column-arithmetic | Architecture: CD-DV-NCA (Cc=32, k=4, Bi)")
    print("Training Config: 50 Epochs | Dev Steps: 20 | Seq Len: 16 | Batch: 16 | LR: 0.003")
    print("Inductive Biases: Native STE-Sign + Tail Equilibrium (weight=0.1, ticks=4)")
    print(f"Seeds: {TRAIN_SEEDS}")
    print("=" * 80)

    if args.dry_run:
        print("[INFO] Dry-run complete. Exiting.")
        return

    # -------------------------------------------------------------
    # Step 1: Training Phase across Seeds [42, 43, 44]
    # -------------------------------------------------------------
    t0_all = time.time()
    train_results = {}

    for s in TRAIN_SEEDS:
        s_dir = os.path.join(BASE_SAVE_DIR, f"seed_{s}")
        os.makedirs(s_dir, exist_ok=True)
        print(f"\n[TRAIN] Starting training for Seed {s} -> {s_dir}")

        cmd_train = [
            BIN, "train",
            "--task", "column-arithmetic",
            "--save-dir", s_dir,
            "--epochs", EPOCHS,
            "--seq-len", "16",
            "--dev-steps", DEV_STEPS,
            "--batch-size", BATCH_SIZE,
            "--lr", LR,
            "--seed", str(s),
            "--carry-channels", "32",
            "--carry-skip-stride", "4",
            "--carry-bidirectional",
            "--carry-quantization", "ste_sign",
            "--tail-eq-weight", "0.1",
            "--tail-eq-ticks", "4",
        ]

        st = time.time()
        code, out = run_cmd(cmd_train)
        dur = time.time() - st

        if code != 0:
            print(f"  [ERROR] Training failed for seed {s} ({dur:.1f}s):\n{out}")
            continue

        metrics = parse_train_output(out)
        train_results[s] = {
            "time": round(dur, 1),
            "train_acc": metrics.get("train_acc", 0.0),
            "train_loss": metrics.get("train_loss", 0.0),
            "val_acc": metrics.get("val_acc", 0.0),
            "val_loss": metrics.get("val_loss", 0.0),
            "checkpoint_dir": s_dir,
        }
        print(f"  ✓ Seed {s} trained in {dur:.1f}s | Val Acc = {metrics.get('val_acc', 0.0):.1f}%, Val Loss = {metrics.get('val_loss', 0.0):.4f}")

    # -------------------------------------------------------------
    # Step 2: Evaluation Phase on L in {16, 32, 64}
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION PHASE: ZERO-SHOT SCALING, KINETIC STABILITY & DYNAMIC HALTING")
    print("=" * 80)

    eval_results = {str(l): {} for l in LENGTHS}
    kinetic_results = {str(l): [] for l in LENGTHS}
    halting_results = {str(l): [] for l in LENGTHS}

    # Recommended evaluation depth per length:
    # L=16: T=20; L=32: T=24; L=64: T=24 and T=32
    depth_map = {
        16: [16, 20],
        32: [24],
        64: [24, 32],
    }

    for l_val in LENGTHS:
        print(f"\n--- Sequence Length L = {l_val} ---")
        eval_results[str(l_val)]["depths"] = {}

        # 2a. Fixed Recurrence Benchmark (evaluate across seeds)
        for t_val in depth_map[l_val]:
            accs = []
            losses = []
            for s in TRAIN_SEEDS:
                ckpt = train_results[s]["checkpoint_dir"]
                cmd_bench = [
                    BIN, "benchmark",
                    "--task", "column-arithmetic",
                    "--checkpoint", ckpt,
                    "--seq-len", str(l_val),
                    "--dev-steps", str(t_val),
                    "--batch-size", BATCH_SIZE,
                    "--seeds", EVAL_SEEDS,
                    "--carry-quantization", "ste_sign",
                ]
                code_b, out_b = run_cmd(cmd_bench)
                if code_b == 0:
                    metrics = parse_benchmark_output(out_b)
                    accs.append(metrics.get("acc_mean", 0.0))
                    losses.append(metrics.get("loss_mean", 0.0))

            mean_acc = sum(accs) / max(1, len(accs))
            mean_loss = sum(losses) / max(1, len(losses))
            eval_results[str(l_val)]["depths"][str(t_val)] = {
                "acc_mean": round(mean_acc, 2),
                "loss_mean": round(mean_loss, 4),
                "raw_accs": accs,
            }
            print(f"  • L={l_val:2d} T={t_val:2d} (STE-Sign Native): Acc = {mean_acc:5.1f}% | Loss = {mean_loss:6.4f}")

        # 2b. Kinetic Trajectory Analysis & Dual-Metric Halting
        for s in TRAIN_SEEDS:
            ckpt = train_results[s]["checkpoint_dir"]
            trace_file = os.path.join(SCRATCH_DIR, f"trace_s{s}_l{l_val}.json")
            cmd_rollout = [
                BIN, "rollout",
                "--load-dir", ckpt,
                "--task", "column-arithmetic",
                "--seq-len", str(l_val),
                "--horizon", str(T_MAX),
                "--trace-output", trace_file,
            ]
            code_r, out_r = run_cmd(cmd_rollout)
            if code_r == 0:
                k_info = analyze_kinetic_trajectory(trace_file, l_val)
                k_info["seed"] = s
                kinetic_results[str(l_val)].append(k_info)

                # Evaluate accuracy at stopped tick
                t_h = k_info["t_halt"]
                cmd_h = [
                    BIN, "benchmark",
                    "--task", "column-arithmetic",
                    "--checkpoint", ckpt,
                    "--seq-len", str(l_val),
                    "--dev-steps", str(t_h),
                    "--batch-size", BATCH_SIZE,
                    "--seeds", EVAL_SEEDS,
                    "--carry-quantization", "ste_sign",
                ]
                code_h, out_h = run_cmd(cmd_h)
                m_h = parse_benchmark_output(out_h) if code_h == 0 else {}
                h_acc = m_h.get("acc_mean", 0.0)
                h_loss = m_h.get("loss_mean", 0.0)

                compute_reduction = max(0.0, 1.0 - (t_h / float(T_MAX))) * 100.0
                halting_results[str(l_val)].append({
                    "seed": s,
                    "t_halt": t_h,
                    "acc": h_acc,
                    "loss": h_loss,
                    "compute_reduction_pct": round(compute_reduction, 1),
                    "is_premature": k_info["is_premature"],
                    "tail_delta": k_info["tail_delta"],
                    "is_fixed_point": k_info["is_fixed_point"],
                })

        # Summary of kinetic & halting metrics for this L
        k_list = kinetic_results[str(l_val)]
        h_list = halting_results[str(l_val)]
        mean_tail_delta = sum(k["tail_delta"] for k in k_list) / max(1, len(k_list))
        fixed_point_fraction = sum(1 for k in k_list if k["is_fixed_point"]) / max(1, len(k_list))
        mean_t_halt = sum(h["t_halt"] for h in h_list) / max(1, len(h_list))
        mean_compute_saved = sum(h["compute_reduction_pct"] for h in h_list) / max(1, len(h_list))
        premature_count = sum(1 for h in h_list if h["is_premature"])
        mean_h_acc = sum(h["acc"] for h in h_list) / max(1, len(h_list))

        eval_results[str(l_val)]["kinetic_summary"] = {
            "mean_tail_delta": round(mean_tail_delta, 5),
            "fixed_point_fraction": round(fixed_point_fraction, 2),
            "mean_t_halt": round(mean_t_halt, 1),
            "mean_compute_reduction_pct": round(mean_compute_saved, 1),
            "premature_rate": round(premature_count / max(1, len(h_list)), 2),
            "halting_acc_mean": round(mean_h_acc, 2),
        }

        print(f"  • Kinetic Equilibrium: Tail Delta = {mean_tail_delta:.5f} (Fixed Point: {fixed_point_fraction*100:.0f}%)")
        print(f"  • Dual-Metric Halting: Mean T_halt = {mean_t_halt:.1f}/{T_MAX} | Compute Saved = {mean_compute_saved:.1f}% | Halting Acc = {mean_h_acc:.1f}% (Premature: {premature_count}/{len(h_list)})")

    total_time = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"CAMPAIGN COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 3: Synthesis & Comparison Summary
    # -------------------------------------------------------------
    print("\n─── NATIVE CD-DV-NCA (STE-SIGN + EQUILIBRIUM): FINAL SCALING RESULTS ───────────")
    print(f"{'Sequence Length':<16} | {'Fixed Recurrence Acc':<22} | {'Halting Acc (T_halt)':<22} | {'Compute Saved':<14} | {'Tail Delta'}")
    print("─────────────────┼────────────────────────┼────────────────────────┼───────────────┼───────────")
    for l_val in LENGTHS:
        l_str = str(l_val)
        best_fixed = max(eval_results[l_str]["depths"].values(), key=lambda x: x["acc_mean"])
        ks = eval_results[l_str]["kinetic_summary"]
        print(f"L={l_val:<14} | {best_fixed['acc_mean']:5.1f}% (Loss {best_fixed['loss_mean']:.2f})  | {ks['halting_acc_mean']:5.1f}% (T={ks['mean_t_halt']:.0f})    | {ks['mean_compute_reduction_pct']:5.1f}%         | {ks['mean_tail_delta']:.5f}")
    print("────────────────────────────────────────────────────────────────────────────────")

    # Persist structured output
    full_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": round(total_time, 2),
            "task": "column-arithmetic",
            "epochs": EPOCHS,
            "dev_steps": DEV_STEPS,
            "lengths": LENGTHS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "model_config": {
                "carry_channels": 32,
                "carry_skip_stride": 4,
                "carry_bidirectional": True,
                "carry_quantization": "ste_sign",
                "tail_eq_weight": 0.1,
                "tail_eq_ticks": 4,
            },
        },
        "training": train_results,
        "evaluation": eval_results,
        "kinetic_details": kinetic_results,
        "halting_details": halting_results,
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[INFO] Complete results and telemetry persisted to: {Path(REPORT_FILE).resolve()}")


if __name__ == "__main__":
    main()
