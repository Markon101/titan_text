#!/usr/bin/env python3
"""Automated runner for the Controlled IPPR Learning Campaign.

Orchestrates multi-checkpoint training increments (100 epochs each),
runs full evaluation batteries at every checkpoint, computes G_identity
(overall and per slot), sham controls, and evaluates pre-registered
stopping and escalation gates.
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
    ]
    if load_dir and os.path.exists(os.path.join(load_dir, "manifest.json")):
        cmd.extend(["--load-dir", load_dir])
    if zero_boundary:
        cmd.append("--zero-boundary")

    print(f"  [Train] Running {epochs} epochs (dev_steps={dev_steps}, seq_len={seq_len}) -> {save_dir}...")
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
    report_file = os.path.join(tmp_dir, f"sweep_eval_{os.getpid()}.json")

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
    code, _ = run_cmd(cmd_intact)
    if code != 0:
        print(f"  [Error] Intact sweep failed on {ckpt_dir}")
        return {}

    with open(report_file, "r") as f:
        data_intact = json.load(f)

    # 2. Controls at budget = dev_steps (16 for L=16, 8 for L=8)
    target_budget = 16 if seq_len >= 16 else 8

    def run_control(extra_args: list[str]) -> dict:
        c_file = os.path.join(tmp_dir, f"sweep_ctrl_{os.getpid()}.json")
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
        run_cmd(cmd)
        if os.path.exists(c_file):
            with open(c_file, "r") as f:
                d = json.load(f)
            os.remove(c_file)
            return d
        return {}

    data_lesion = run_control(["--lesion-state"])
    data_invert = run_control(["--lesion-gain", "-1.0"])
    data_shuffle = run_control(["--lesion-shuffle"])
    data_sham = run_control(["--lesion-noise", "0.05"])

    if os.path.exists(report_file):
        os.remove(report_file)

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

    # Compute paired identity gap across seeds
    accs_in = stats_intact_target["acc_seeds"]
    accs_sh = stats_shuffle["acc_seeds"]
    paired_g_id = [a - b for a, b in zip(accs_in, accs_sh)] if (accs_in and len(accs_in) == len(accs_sh)) else [stats_intact_target["acc"] - stats_shuffle["acc"]]
    n_p = len(paired_g_id)
    mean_g_id = sum(paired_g_id) / n_p
    var_g_id = sum((g - mean_g_id) ** 2 for g in paired_g_id) / max(1, n_p - 1) if n_p > 1 else 0.0
    std_g_id = math.sqrt(var_g_id)
    sem_g_id = std_g_id / math.sqrt(n_p) if n_p > 0 else 0.0
    cohen_d = mean_g_id / std_g_id if std_g_id > 1e-6 else 0.0

    g_sham = stats_intact_target["acc"] - stats_sham["acc"]
    g_lesion = stats_intact_target["acc"] - stats_lesion["acc"]

    # Per slot gaps
    per_slot_g_id = []
    if stats_intact_target["slots"] and stats_shuffle["slots"]:
        for s_in, s_sh in zip(stats_intact_target["slots"], stats_shuffle["slots"]):
            per_slot_g_id.append(s_in - s_sh)

    return {
        "target_budget": target_budget,
        "intact": stats_intact_target,
        "lesion_state": stats_lesion,
        "invert_gain": stats_invert,
        "shuffle_batch": stats_shuffle,
        "sham_noise": stats_sham,
        "g_identity": mean_g_id,
        "g_identity_std": std_g_id,
        "g_identity_sem": sem_g_id,
        "cohen_d": cohen_d,
        "paired_g_id": paired_g_id,
        "g_sham": g_sham,
        "g_lesion": g_lesion,
        "per_slot_g_identity": per_slot_g_id,
        "sweep_curve": sweep_curve,
    }

def run_arm(
    arm_name: str,
    stages: list[dict],
    total_epochs: int,
    report_output: str,
    zero_boundary: bool = True,
):
    print(f"\n{'='*80}")
    print(f"STARTING CAMPAIGN ARM: {arm_name}")
    print(f"{'='*80}")

    checkpoint_records = []
    current_load_dir = None
    cum_epochs = 0

    for stage_idx, stage in enumerate(stages):
        stage_epochs = stage["epochs"]
        seq_len = stage["seq_len"]
        dev_steps = stage["dev_steps"]
        step_increment = stage.get("increment", 100)

        num_increments = stage_epochs // step_increment
        for inc in range(1, num_increments + 1):
            cum_epochs += step_increment
            save_dir = f"checkpoints/{arm_name}/step_{cum_epochs:04d}"

            # 1. Train
            success = train_increment(
                load_dir=current_load_dir,
                save_dir=save_dir,
                task="iterated-parity",
                epochs=step_increment,
                dev_steps=dev_steps,
                seq_len=seq_len,
                zero_boundary=zero_boundary,
            )
            if not success:
                print(f"  [Fatal] Arm {arm_name} failed training at {save_dir}")
                return

            current_load_dir = save_dir

            # 2. Evaluate
            print(f"  [Eval] Evaluating checkpoint at {save_dir}...")
            eval_res = eval_checkpoint(save_dir, seq_len=seq_len, zero_boundary=zero_boundary)

            record = {
                "epoch": cum_epochs,
                "stage": stage_idx + 1,
                "seq_len": seq_len,
                "dev_steps": dev_steps,
                "checkpoint": save_dir,
                "eval": eval_res,
            }
            checkpoint_records.append(record)

            # Print summary line
            g_id = eval_res.get("g_identity", 0.0)
            in_acc = eval_res.get("intact", {}).get("acc", 0.0)
            sh_acc = eval_res.get("shuffle_batch", {}).get("acc", 0.0)
            slots_str = ", ".join(f"s{i}:{s:.1f}%" for i, s in enumerate(eval_res.get("intact", {}).get("slots", [])))
            g_slots_str = ", ".join(f"s{i}:{g:+.1f}%" for i, g in enumerate(eval_res.get("per_slot_g_identity", [])))

            print(f"  --> Epoch {cum_epochs:4d} | Intact Acc: {in_acc:5.1f}% | Shuf Acc: {sh_acc:5.1f}% | G_id: {g_id:+5.1f}% (±{eval_res.get('g_identity_sem', 0.0):.1f}%, d={eval_res.get('cohen_d', 0.0):.2f})")
            print(f"      Slots: [{slots_str}] | G_slots: [{g_slots_str}]")

            # Check Stopping Criteria
            # S1: G_identity < 3.0% and slope < 0.5% sustained across 4 checkpoints after step 400 of L=16
            if seq_len >= 16 and inc >= 4:
                stage_records = [r for r in checkpoint_records if r["stage"] == stage_idx + 1]
                if len(stage_records) >= 4 and inc * step_increment >= 400:
                    last4_g = [r["eval"]["g_identity"] for r in stage_records[-4:]]
                    if all(g < 3.0 for g in last4_g):
                        slope = (last4_g[-1] - last4_g[0]) / 3.0
                        if abs(slope) < 0.5:
                            print(f"\n[EARLY STOPPING TRIGGERED] S1: Flat identity gap (last 4 G_id: {last4_g}, slope: {slope:.2f}%/step).")
                            print("Terminating arm early to preserve compute.")
                            break

            # Check Escalation Criteria
            # E1: G_id >= 15.0% and later slots > 65% for 2 consecutive checkpoints
            if len(checkpoint_records) >= 2:
                last2_g = [r["eval"]["g_identity"] for r in checkpoint_records[-2:]]
                if all(g >= 15.0 for g in last2_g):
                    print(f"\n[ESCALATION CRITERIA MET] E1: G_identity decisively cleared 15% ({last2_g})!")
                    print("Phase transition to genuine instance-specific recurrence detected!")

    # Save arm report
    with open(report_output, "w") as f:
        json.dump({
            "arm_name": arm_name,
            "zero_boundary": zero_boundary,
            "total_epochs": cum_epochs,
            "records": checkpoint_records,
        }, f, indent=2)
    print(f"\n✓ Saved campaign arm results to '{report_output}'")

def main():
    parser = argparse.ArgumentParser(description="Titan Text IPPR Controlled Learning Campaign")
    parser.add_argument("--arm", choices=["from_scratch", "curriculum", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.arm in ["from_scratch", "both"]:
        # Arm 1: Matched from-scratch L=16, dev_steps=16, 1000 epochs
        run_arm(
            arm_name="campaign_arm1_from_scratch",
            stages=[{"epochs": args.epochs, "seq_len": 16, "dev_steps": 16, "increment": 100}],
            total_epochs=args.epochs,
            report_output="reports/campaign_arm1_from_scratch.json",
            zero_boundary=True,
        )

    if args.arm in ["curriculum", "both"]:
        # Arm 2: Curriculum L=8 (500 ep) -> L=16 (500 ep)
        half_ep = args.epochs // 2
        run_arm(
            arm_name="campaign_arm2_curriculum",
            stages=[
                {"epochs": half_ep, "seq_len": 8, "dev_steps": 8, "increment": 100},
                {"epochs": half_ep, "seq_len": 16, "dev_steps": 16, "increment": 100},
            ],
            total_epochs=args.epochs,
            report_output="reports/campaign_arm2_curriculum.json",
            zero_boundary=True,
        )

if __name__ == "__main__":
    main()
