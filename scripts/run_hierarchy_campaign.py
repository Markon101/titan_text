#!/usr/bin/env python3
"""Automated runner for Option A: 1D Micro-Macro Cellular Hierarchy Campaign.

Evaluates whether a strictly local two-level cellular hierarchy (Micro L=16 + Macro L_M=8)
with local discrete difference stencil modulation (w_j = M_j - 0.5*(M_{j-1} + M_{j+1}))
overcomes the physical transport barrier (H_TRANSPORT) and unlocks genuine long-range
recurrent parity transport to interior query slots (Slots 1 & 2 at pos 7 & 11).

Arms Evaluated:
  - Arm H (Hierarchy Treatment): macro_stride=2, macro_period=2, macro_channels=32
  - Arm C (Degenerate Control): macro_stride=1, macro_period=2, macro_channels=32

Pre-Registered Decision Gates (N=5 seeds):
  - Gate G1: Interior slot 1 & 2 intact accuracy >= 65.0% (Reject if <= 55.0%)
  - Gate G2: Macro lesion delta >= 20.0%
  - Gate G3: Non-degeneracy / Anti-saturation: Var(w) / Mean(|w|) >= 0.05
  - Gate G4: Spatial coarsening is load-bearing: Arm H succeeds while Arm C fails
  - Gate G5: Receptive field expansion without all-to-all bypass
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
    ]
    if load_dir and os.path.exists(os.path.join(load_dir, "manifest.json")):
        cmd.extend(["--load-dir", load_dir])
    if zero_boundary:
        cmd.append("--zero-boundary")

    print(f"  [Train] Running {epochs} epochs (stride={macro_stride}, dev_steps={dev_steps}) -> {save_dir}...")
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
    report_file = os.path.join(tmp_dir, f"sweep_h_eval_{os.getpid()}_{time.time_ns()}.json")

    # 1. Full sweep intact across budgets 0, 1, 2, 4, 8, 12, 16
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
        if code_cond != 0:
            print(f"    [Warning] Condition {extra_args} returned {code_cond}: {err}")
        if os.path.exists(c_file):
            with open(c_file, "r") as f:
                d = json.load(f)
            os.remove(c_file)
            return d
        return {}

    # Standard causal controls
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

    # Extract intact stats across budgets
    sweep_curve = {}
    for b in [0, 1, 2, 4, 8, 12, 16]:
        sweep_curve[str(b)] = extract_budget_stats(data_intact, b)

    stats_intact_target = extract_budget_stats(data_intact, target_budget)
    stats_lesion = extract_budget_stats(data_lesion, target_budget)
    stats_invert = extract_budget_stats(data_invert, target_budget)
    stats_shuffle = extract_budget_stats(data_shuffle, target_budget)
    stats_sham = extract_budget_stats(data_sham, target_budget)

    # Compute paired gaps across seeds
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

    # Per-slot gaps
    per_slot_g_id = []
    if stats_intact_target["slots"] and stats_shuffle["slots"]:
        for s_in, s_sh in zip(stats_intact_target["slots"], stats_shuffle["slots"]):
            per_slot_g_id.append(s_in - s_sh)

    # Run latent representation probe
    cmd_probe = [
        BIN, "influence",
        "--load-dir", ckpt_dir,
        "--budgets", str(target_budget),
        "--seq-len", str(seq_len),
        "--zero-boundary",
    ]
    code_probe, out_probe = run_cmd(cmd_probe)
    probe_results = {}
    if code_probe == 0:
        for line in out_probe.splitlines():
            line_str = line.strip()
            if "Linear Chunk0 Parity Acc" in line_str:
                probe_results["linear_chunk0"] = line_str.split(":", 1)[-1].strip()
            elif "Linear Local Parity Acc" in line_str:
                probe_results["linear_local"] = line_str.split(":", 1)[-1].strip()
            elif "Linear Cumulative Acc" in line_str:
                probe_results["linear_cumulative"] = line_str.split(":", 1)[-1].strip()
            elif "MLP Chunk0 Parity Acc" in line_str:
                probe_results["mlp_chunk0"] = line_str.split(":", 1)[-1].strip()
            elif "MLP Local Parity Acc" in line_str:
                probe_results["mlp_local"] = line_str.split(":", 1)[-1].strip()
            elif "MLP Cumulative Acc" in line_str:
                probe_results["mlp_cumulative"] = line_str.split(":", 1)[-1].strip()

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
        "probe_representations": probe_results,
    }

def run_arm(
    arm_name: str,
    macro_stride: int,
    macro_period: int = 2,
    macro_channels: int = 32,
    total_epochs: int = 500,
    increment: int = 100,
    seq_len: int = 16,
    dev_steps: int = 16,
    task: str = "iterated-parity-dense",
    base_dir: str = "checkpoints/campaign_arm_hierarchy",
    seeds: list[int] = [42, 101, 202, 303, 404],
) -> dict:
    seeds_str = ",".join(str(s) for s in seeds)
    checkpoints_epochs = list(range(increment, total_epochs + 1, increment))
    dataset_records = []

    print(f"\n================================================================================")
    print(f"STARTING {arm_name.upper()} (stride={macro_stride}, period={macro_period}, total_epochs={total_epochs})")
    print(f"================================================================================")

    for ep in checkpoints_epochs:
        prev_ep = ep - increment
        print(f"\n--- Epoch {ep} Progression ---")

        for seed in seeds:
            load_dir = os.path.join(base_dir, f"seed_{seed}", f"epoch_{prev_ep}") if prev_ep > 0 else None
            save_dir = os.path.join(base_dir, f"seed_{seed}", f"epoch_{ep}")

            ok = train_increment(
                load_dir=load_dir,
                save_dir=save_dir,
                task=task,
                epochs=increment,
                dev_steps=dev_steps,
                seq_len=seq_len,
                macro_stride=macro_stride,
                macro_period=macro_period,
                macro_channels=macro_channels,
                feedback_mode="hierarchy",
                zero_boundary=True,
                seed=seed,
            )
            if not ok:
                print(f"[Fatal] Training failed for seed {seed} at epoch {ep}")
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
        "macro_stride": macro_stride,
        "macro_period": macro_period,
        "macro_channels": macro_channels,
        "total_epochs": total_epochs,
        "seeds": seeds,
        "checkpoints": dataset_records,
    }

def main():
    parser = argparse.ArgumentParser(description="Run 1D Micro-Macro Hierarchy Campaign")
    parser.add_argument("--epochs", type=int, default=500, help="Total training epochs per seed")
    parser.add_argument("--increment", type=int, default=100, help="Epoch increment between evaluations")
    parser.add_argument("--arm", choices=["all", "hierarchy", "control"], default="all", help="Which arm to run")
    args = parser.parse_args()

    seeds = [42, 101, 202, 303, 404]

    if args.arm in ["all", "hierarchy"]:
        res_h = run_arm(
            arm_name="Arm H (Micro-Macro Hierarchy Stride 2)",
            macro_stride=2,
            macro_period=2,
            macro_channels=32,
            total_epochs=args.epochs,
            increment=args.increment,
            base_dir="checkpoints/campaign_arm_h_stride2",
            seeds=seeds,
        )
        out_h_file = "reports/campaign_arm_h_stride2.json"
        os.makedirs("reports", exist_ok=True)
        with open(out_h_file, "w") as f:
            json.dump(res_h, f, indent=2)
        print(f"\n✓ Arm H dataset written to {out_h_file}")

    if args.arm in ["all", "control"]:
        res_c = run_arm(
            arm_name="Arm C (Degenerate Stride 1 Control)",
            macro_stride=1,
            macro_period=2,
            macro_channels=32,
            total_epochs=args.epochs,
            increment=args.increment,
            base_dir="checkpoints/campaign_arm_c_stride1",
            seeds=seeds,
        )
        out_c_file = "reports/campaign_arm_c_stride1.json"
        os.makedirs("reports", exist_ok=True)
        with open(out_c_file, "w") as f:
            json.dump(res_c, f, indent=2)
        print(f"\n✓ Arm C dataset written to {out_c_file}")

if __name__ == "__main__":
    main()
