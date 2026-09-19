#!/usr/bin/env python3
"""Automated Dual-Track Research Campaign Runner for Titan Text.

Executes both scientific tracks authorized by user and DeepSeek Council:
  - Track 1 (Path A): Character-Matched Cellular Discrete Latching on iterated-parity-dense
    (Walsh-Hadamard downsampling, state-derivative coupling, emergent bistable potential)
  - Track 2 (Path B): Continuous Lipschitz Invariant Transport on iterated-sum-dense
    (Testing multiscale coarsening s=2 vs s=1 on low-frequency non-attenuating modes)

Pre-Registered Decision Gates (N=5 seeds [42, 101, 202, 303, 404]):
  - Gate G-T1 (Track 1 / Parity): Arm W (Walsh + State-Derivative) exceeds pool_mean by >= 10.0%
    on interior query slots (Slots 1 & 2 reaching >= 55.0% with G_identity >= +5.0%).
  - Gate G-T2 (Track 2 / Sum): Arm H-Sum (s=2) beats Arm C-Sum (s=1) by >= 15.0% on query slots
    with paired Cohen's d >= 0.50, and beats local-only baseline (~33%) by reaching >= 70.0%.
"""

from __future__ import annotations

import argparse
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

def train_increment(
    load_dir: str | None,
    save_dir: str,
    task: str,
    epochs: int,
    dev_steps: int,
    seq_len: int,
    macro_stride: int = 2,
    macro_period: int = 2,
    macro_channels: int = 32,
    macro_downsampler: str = "walsh",
    macro_coupling: str = "state_derivative",
    macro_gamma: float = 0.2,
    macro_lambda: float = 0.1,
    feedback_mode: str = "hierarchy",
    zero_boundary: bool = True,
    seed: int = 42,
) -> bool:
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        BIN, "train",
        "--task", task,
        "--save-dir", save_dir,
        "--epochs", str(epochs),
        "--dev-steps", str(dev_steps),
        "--seq-len", str(seq_len),
        "--seed", str(seed),
        "--feedback-mode", feedback_mode,
        "--macro-stride", str(macro_stride),
        "--macro-period", str(macro_period),
        "--macro-channels", str(macro_channels),
        "--macro-downsampler", macro_downsampler,
        "--macro-coupling", macro_coupling,
        "--macro-gamma", str(macro_gamma),
        "--macro-lambda", str(macro_lambda),
    ]
    if load_dir and os.path.exists(os.path.join(load_dir, "manifest.json")):
        cmd.extend(["--load-dir", load_dir])
    if zero_boundary:
        cmd.append("--zero-boundary")

    code, out = run_cmd(cmd)
    if code != 0:
        print(f"  [Error] Training failed with code {code}:\n{out}")
        return False
    return True

def eval_checkpoint(
    ckpt_dir: str,
    seq_len: int,
    seeds: str = "42,101,202,303,404",
    zero_boundary: bool = True,
) -> dict:
    tmp_dir = tempfile.gettempdir()
    report_file = os.path.join(tmp_dir, f"sweep_eval_{os.getpid()}_{time.time_ns()}.json")

    cmd_intact = [
        BIN, "sweep",
        "--load-dir", ckpt_dir,
        "--budgets", "0,1,2,4,8,12,16",
        "--seq-len", str(seq_len),
        "--batch-size", "32",
        "--seeds", seeds,
        "--output", report_file,
    ]
    if zero_boundary:
        cmd_intact.append("--zero-boundary")

    code, out = run_cmd(cmd_intact)
    if code != 0:
        print(f"  [Error] Intact sweep failed on {ckpt_dir}:\n{out}")
        return {}

    data_intact = {}
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            data_intact = json.load(f)
        os.remove(report_file)

    target_budget = 16 if seq_len >= 16 else 8

    def run_condition(extra_args: list[str]) -> dict:
        c_file = os.path.join(tmp_dir, f"sweep_cond_{os.getpid()}_{time.time_ns()}.json")
        cmd = [
            BIN, "sweep",
            "--load-dir", ckpt_dir,
            "--budgets", str(target_budget),
            "--seq-len", str(seq_len),
            "--batch-size", "32",
            "--seeds", seeds,
            "--output", c_file,
        ] + extra_args
        if zero_boundary:
            cmd.append("--zero-boundary")
        code_cond, err = run_cmd(cmd)
        if os.path.exists(c_file):
            with open(c_file, "r") as f:
                d = json.load(f)
            os.remove(c_file)
            return d
        return {}

    data_lesion = run_condition(["--lesion-state"])
    data_invert = run_condition(["--lesion-gain", "-1.0"])
    data_shuffle = run_condition(["--lesion-shuffle"])
    data_sham = run_condition(["--lesion-noise", "0.05"])

    def extract_budget_stats(data: dict, b_val: int) -> dict:
        if not data:
            return {"acc": 0.0, "acc_std": 0.0, "acc_seeds": [], "loss": 0.0, "slots": []}
        if "reports" in data:
            reports = data["reports"]
        elif "budgets" in data:
            reports = [data]
        else:
            return {"acc": 0.0, "acc_std": 0.0, "acc_seeds": [], "loss": 0.0, "slots": []}

        accs = []
        losses = []
        slot_sums = None
        for r in reports:
            for pt in r.get("budgets", []):
                if pt["latent_ticks"] == b_val:
                    accs.append(pt["accuracy"] * 100.0)
                    losses.append(pt["loss"])
                    if pt.get("per_slot_accuracy"):
                        slots = pt["per_slot_accuracy"]
                        if slot_sums is None:
                            slot_sums = [0.0] * len(slots)
                        for si, sv in enumerate(slots):
                            slot_sums[si] += sv * 100.0
        n = max(1, len(accs))
        mean_acc = sum(accs) / n
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / max(1, n - 1) if n > 1 else 0.0
        std_acc = math.sqrt(var_acc)
        mean_slots = [s / n for s in slot_sums] if slot_sums else []
        return {
            "acc": mean_acc,
            "acc_std": std_acc,
            "acc_seeds": accs,
            "loss": sum(losses) / n,
            "slots": mean_slots,
        }

    sweep_curve = {}
    for b in [0, 1, 2, 4, 8, 12, 16]:
        sweep_curve[str(b)] = extract_budget_stats(data_intact, b)

    stats_intact_target = extract_budget_stats(data_intact, target_budget)
    stats_lesion = extract_budget_stats(data_lesion, target_budget)
    stats_invert = extract_budget_stats(data_invert, target_budget)
    stats_shuffle = extract_budget_stats(data_shuffle, target_budget)
    stats_sham = extract_budget_stats(data_sham, target_budget)

    def compute_paired_gap(cond_stats: dict) -> tuple[float, float, float, float, list[float]]:
        in_seeds = stats_intact_target["acc_seeds"]
        co_seeds = cond_stats["acc_seeds"]
        if len(in_seeds) != len(co_seeds) or len(in_seeds) == 0:
            return 0.0, 0.0, 0.0, 0.0, []
        diffs = [a - b for a, b in zip(in_seeds, co_seeds)]
        n = len(diffs)
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, n - 1) if n > 1 else 0.0
        sem_d = math.sqrt(var_d / n) if n > 0 else 0.0
        std_d = math.sqrt(var_d)
        cohen_d = mean_d / std_d if std_d > 1e-9 else 0.0
        return mean_d, sem_d, std_d, cohen_d, diffs

    g_id, g_id_sem, _, cohen_d_id, _ = compute_paired_gap(stats_shuffle)

    per_slot_g_id = []
    if stats_intact_target["slots"] and stats_shuffle["slots"]:
        for s_in, s_sh in zip(stats_intact_target["slots"], stats_shuffle["slots"]):
            per_slot_g_id.append(s_in - s_sh)

    # Latent representation probe (runs on parity tasks)
    probe_reps = {}
    cmd_probe = [
        BIN, "influence",
        "--load-dir", ckpt_dir,
        "--budgets", str(target_budget),
        "--seq-len", str(seq_len),
        "--zero-boundary",
    ]
    code_probe, out_probe = run_cmd(cmd_probe)
    if code_probe == 0:
        for line in out_probe.splitlines():
            if "Linear Chunk0 Parity Acc" in line:
                probe_reps["linear_chunk0"] = line.split(":")[-1].strip()
            elif "Linear Local Parity Acc" in line:
                probe_reps["linear_local"] = line.split(":")[-1].strip()
            elif "Linear Cumulative Acc" in line:
                probe_reps["linear_cumulative"] = line.split(":")[-1].strip()
            elif "MLP Chunk0 Parity Acc" in line:
                probe_reps["mlp_chunk0"] = line.split(":")[-1].strip()
            elif "MLP Local Parity Acc" in line:
                probe_reps["mlp_local"] = line.split(":")[-1].strip()
            elif "MLP Cumulative Acc" in line:
                probe_reps["mlp_cumulative"] = line.split(":")[-1].strip()

    return {
        "target_budget": target_budget,
        "intact": stats_intact_target,
        "lesion_state": stats_lesion,
        "invert_gain": stats_invert,
        "shuffle_batch": stats_shuffle,
        "sham_noise": stats_sham,
        "g_identity": g_id,
        "g_identity_sem": g_id_sem,
        "cohen_d_identity": cohen_d_id,
        "per_slot_g_identity": per_slot_g_id,
        "sweep_curve": sweep_curve,
        "probe_representations": probe_reps,
    }

def run_arm(
    arm_name: str,
    task: str,
    macro_stride: int,
    macro_period: int,
    macro_channels: int,
    macro_downsampler: str,
    macro_coupling: str,
    macro_gamma: float,
    macro_lambda: float,
    total_epochs: int,
    increment: int,
    base_dir: str,
    seeds: list[int],
    seq_len: int = 16,
    dev_steps: int = 16,
) -> dict:
    print(f"\n{'='*80}")
    print(f"STARTING {arm_name.upper()} (task={task}, stride={macro_stride}, down={macro_downsampler}, coup={macro_coupling}, epochs={total_epochs})")
    print(f"{'='*80}\n")

    dataset_records = []
    seeds_str = ",".join(str(s) for s in seeds)

    for ep in range(increment, total_epochs + 1, increment):
        print(f"\n--- Epoch {ep} Progression ---")
        for seed in seeds:
            seed_dir = os.path.join(base_dir, f"seed_{seed}")
            prev_ep = ep - increment
            load_dir = os.path.join(seed_dir, f"epoch_{prev_ep}") if prev_ep > 0 else None
            save_dir = os.path.join(seed_dir, f"epoch_{ep}")

            success = train_increment(
                load_dir=load_dir,
                save_dir=save_dir,
                task=task,
                epochs=increment,
                dev_steps=dev_steps,
                seq_len=seq_len,
                macro_stride=macro_stride,
                macro_period=macro_period,
                macro_channels=macro_channels,
                macro_downsampler=macro_downsampler,
                macro_coupling=macro_coupling,
                macro_gamma=macro_gamma,
                macro_lambda=macro_lambda,
                feedback_mode="hierarchy",
                zero_boundary=True,
                seed=seed,
            )
            if not success:
                print(f"Fatal: Failed training seed {seed} at epoch {ep}")
                sys.exit(1)

        eval_dir = os.path.join(base_dir, f"seed_{seeds[0]}", f"epoch_{ep}")
        print(f"  Evaluating N={len(seeds)} multi-seed ensemble at epoch {ep}...")
        eval_metrics = eval_checkpoint(
            ckpt_dir=eval_dir,
            seq_len=seq_len,
            seeds=seeds_str,
            zero_boundary=True,
        )

        intact = eval_metrics.get("intact", {})
        slots = intact.get("slots", [])
        slots_fmt = [f"{s:.2f}%" for s in slots]
        g_id = eval_metrics.get("g_identity", 0.0)

        print(f"  [Epoch {ep} Result] Intact: {intact.get('acc', 0.0):.2f}% ± {intact.get('acc_std', 0.0):.2f}% | G_identity: {g_id:+.2f}%")
        print(f"    Per-Slot: {slots_fmt}")

        record = {
            "epoch": ep,
            "metrics": eval_metrics,
        }
        dataset_records.append(record)

    return {
        "arm": arm_name,
        "task": task,
        "macro_stride": macro_stride,
        "macro_period": macro_period,
        "macro_channels": macro_channels,
        "macro_downsampler": macro_downsampler,
        "macro_coupling": macro_coupling,
        "macro_gamma": macro_gamma,
        "macro_lambda": macro_lambda,
        "total_epochs": total_epochs,
        "seeds": seeds,
        "checkpoints": dataset_records,
    }

def main():
    parser = argparse.ArgumentParser(description="Run Dual-Track Research Campaign")
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs per seed")
    parser.add_argument("--increment", type=int, default=100, help="Epoch increment between evaluations")
    parser.add_argument("--track", choices=["all", "track1", "track2"], default="all", help="Which track to run")
    args = parser.parse_args()

    seeds = [42, 101, 202, 303, 404]
    os.makedirs("reports", exist_ok=True)

    # TRACK 1: Path A (Walsh + State-Derivative + Bistable on iterated-parity-dense)
    if args.track in ["all", "track1"]:
        res_t1 = run_arm(
            arm_name="Arm W (Walsh + State-Derivative Latching)",
            task="iterated-parity-dense",
            macro_stride=2,
            macro_period=2,
            macro_channels=32,
            macro_downsampler="walsh",
            macro_coupling="state_derivative",
            macro_gamma=0.2,
            macro_lambda=0.1,
            total_epochs=args.epochs,
            increment=args.increment,
            base_dir="checkpoints/campaign_track1_walsh",
            seeds=seeds,
        )
        out_t1 = "reports/campaign_track1_walsh.json"
        with open(out_t1, "w") as f:
            json.dump(res_t1, f, indent=2)
        print(f"\n✓ Track 1 dataset written to {out_t1}")

    # TRACK 2: Path B (Continuous Lipschitz Transport on iterated-sum-dense)
    if args.track in ["all", "track2"]:
        # Arm H-Sum: s=2
        res_h_sum = run_arm(
            arm_name="Arm H-Sum (Multiscale Hierarchy Stride 2)",
            task="iterated-sum-dense",
            macro_stride=2,
            macro_period=2,
            macro_channels=32,
            macro_downsampler="walsh",
            macro_coupling="state_derivative",
            macro_gamma=0.2,
            macro_lambda=0.0,
            total_epochs=args.epochs,
            increment=args.increment,
            base_dir="checkpoints/campaign_track2_h_sum",
            seeds=seeds,
        )
        out_h_sum = "reports/campaign_track2_h_sum.json"
        with open(out_h_sum, "w") as f:
            json.dump(res_h_sum, f, indent=2)
        print(f"\n✓ Track 2 Arm H-Sum dataset written to {out_h_sum}")

        # Arm C-Sum: s=1 (Degenerate Control)
        res_c_sum = run_arm(
            arm_name="Arm C-Sum (Degenerate Stride 1 Control)",
            task="iterated-sum-dense",
            macro_stride=1,
            macro_period=2,
            macro_channels=32,
            macro_downsampler="walsh",
            macro_coupling="state_derivative",
            macro_gamma=0.2,
            macro_lambda=0.0,
            total_epochs=args.epochs,
            increment=args.increment,
            base_dir="checkpoints/campaign_track2_c_sum",
            seeds=seeds,
        )
        out_c_sum = "reports/campaign_track2_c_sum.json"
        with open(out_c_sum, "w") as f:
            json.dump(res_c_sum, f, indent=2)
        print(f"\n✓ Track 2 Arm C-Sum dataset written to {out_c_sum}")

if __name__ == "__main__":
    main()
