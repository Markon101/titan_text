#!/usr/bin/env python3
"""Automated runner for the Pre-Registered Coordinate-Channel Learning Campaign.

Executes 1,000 epochs of training on L=16 with --coord-channel and --zero-boundary,
evaluating at each 100-epoch checkpoint:
  1. Latent-tick budget sweep (tau in [0, 1, 2, 4, 8, 12, 16])
  2. Standard causal controls: zero tick, state lesion, inverted gain, batch-state shuffle, sham noise
  3. Full pre-registered coordinate counterfactual battery:
     - coord_zeroed
     - coord_shuffled
     - coord_reversed
     - coord_constant
  4. Per-query-slot resolution across all conditions
  5. Statistical gaps (G_identity, G_coord_ablation, G_coord_shuffle, G_coord_reversed)
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
    coord_channel: bool = True,
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
    if coord_channel:
        cmd.append("--coord-channel")

    print(f"  [Train] Running {epochs} epochs (dev_steps={dev_steps}, seq_len={seq_len}, coord={coord_channel}) -> {save_dir}...")
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
    coord_channel: bool = True,
) -> dict:
    tmp_dir = tempfile.gettempdir()
    report_file = os.path.join(tmp_dir, f"sweep_coord_eval_{os.getpid()}.json")

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
    if coord_channel:
        cmd_intact.append("--coord-channel")

    code, out = run_cmd(cmd_intact)
    if code != 0:
        print(f"  [Error] Intact sweep failed on {ckpt_dir}:\n{out}")
        return {}

    with open(report_file, "r") as f:
        data_intact = json.load(f)

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
        if coord_channel:
            cmd.append("--coord-channel")
        code, err = run_cmd(cmd)
        if code != 0:
            print(f"    [Warning] Condition {extra_args} returned {code}: {err}")
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

    # Coordinate Counterfactuals
    data_coord_zeroed = run_condition(["--coord-mode", "zeroed"])
    data_coord_shuffled = run_condition(["--coord-mode", "shuffled"])
    data_coord_reversed = run_condition(["--coord-mode", "reversed"])
    data_coord_constant = run_condition(["--coord-mode", "constant"])

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

    stats_coord_zeroed = extract_budget_stats(data_coord_zeroed, target_budget)
    stats_coord_shuffled = extract_budget_stats(data_coord_shuffled, target_budget)
    stats_coord_reversed = extract_budget_stats(data_coord_reversed, target_budget)
    stats_coord_constant = extract_budget_stats(data_coord_constant, target_budget)

    # Compute paired gaps across seeds
    def compute_paired_gap(cond_stats: dict) -> tuple[float, float, float, float, list[float]]:
        in_seeds = stats_intact_target["acc_seeds"]
        co_seeds = cond_stats["acc_seeds"]
        if in_seeds and len(in_seeds) == len(co_seeds):
            diffs = [a - b for a, b in zip(in_seeds, co_seeds)]
        else:
            diffs = [stats_intact_target["acc"] - cond_stats["acc"]]
        n_p = len(diffs)
        mean_g = sum(diffs) / n_p
        var_g = sum((g - mean_g) ** 2 for g in diffs) / max(1, n_p - 1) if n_p > 1 else 0.0
        std_g = math.sqrt(var_g)
        sem_g = std_g / math.sqrt(n_p) if n_p > 0 else 0.0
        d = mean_g / std_g if std_g > 1e-6 else 0.0
        return mean_g, std_g, sem_g, d, diffs

    g_id, g_id_std, g_id_sem, d_id, paired_g_id = compute_paired_gap(stats_shuffle)
    g_cz, g_cz_std, g_cz_sem, d_cz, paired_g_cz = compute_paired_gap(stats_coord_zeroed)
    g_cs, g_cs_std, g_cs_sem, d_cs, paired_g_cs = compute_paired_gap(stats_coord_shuffled)
    g_cr, g_cr_std, g_cr_sem, d_cr, paired_g_cr = compute_paired_gap(stats_coord_reversed)
    g_cc, g_cc_std, g_cc_sem, d_cc, paired_g_cc = compute_paired_gap(stats_coord_constant)

    # Per slot gaps for identity
    per_slot_g_id = []
    if stats_intact_target["slots"] and stats_shuffle["slots"]:
        for s_in, s_sh in zip(stats_intact_target["slots"], stats_shuffle["slots"]):
            per_slot_g_id.append(s_in - s_sh)

    # Per slot gaps for coordinate zeroed
    per_slot_g_cz = []
    if stats_intact_target["slots"] and stats_coord_zeroed["slots"]:
        for s_in, s_cz in zip(stats_intact_target["slots"], stats_coord_zeroed["slots"]):
            per_slot_g_cz.append(s_in - s_cz)

    return {
        "target_budget": target_budget,
        "intact": stats_intact_target,
        "lesion_state": stats_lesion,
        "invert_gain": stats_invert,
        "shuffle_batch": stats_shuffle,
        "sham_noise": stats_sham,
        "coord_zeroed": stats_coord_zeroed,
        "coord_shuffled": stats_coord_shuffled,
        "coord_reversed": stats_coord_reversed,
        "coord_constant": stats_coord_constant,
        "g_identity": g_id,
        "g_identity_sem": g_id_sem,
        "cohen_d_identity": d_id,
        "g_coord_zeroed": g_cz,
        "g_coord_zeroed_sem": g_cz_sem,
        "g_coord_shuffled": g_cs,
        "g_coord_shuffled_sem": g_cs_sem,
        "g_coord_reversed": g_cr,
        "g_coord_reversed_sem": g_cr_sem,
        "g_coord_constant": g_cc,
        "g_coord_constant_sem": g_cc_sem,
        "per_slot_g_identity": per_slot_g_id,
        "per_slot_g_coord_zeroed": per_slot_g_cz,
        "sweep_curve": sweep_curve,
    }

def run_coordinate_campaign(
    total_epochs: int = 1000,
    increment: int = 100,
    report_output: str = "reports/campaign_arm2_coordinate.json",
):
    print(f"\n{'='*88}")
    print("STARTING TITAN TEXT IPPR COORDINATE-CHANNEL LEARNING CAMPAIGN")
    print(f"{'='*88}")
    print(f"  Configuration : L=16, dev_steps=16, --coord-channel, --zero-boundary")
    print(f"  Total Epochs  : {total_epochs} (in {increment}-epoch increments)")
    print(f"  Checkpoints   : checkpoints/campaign_coord_l16/step_XXXX")
    print(f"  Report Output : {report_output}\n")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("checkpoints/campaign_coord_l16", exist_ok=True)

    records = []
    current_load_dir = None
    cum_epochs = 0
    num_increments = total_epochs // increment

    for inc in range(1, num_increments + 1):
        cum_epochs += increment
        save_dir = f"checkpoints/campaign_coord_l16/step_{cum_epochs:04d}"

        # 1. Train increment
        success = train_increment(
            load_dir=current_load_dir,
            save_dir=save_dir,
            task="iterated-parity",
            epochs=increment,
            dev_steps=16,
            seq_len=16,
            zero_boundary=True,
            coord_channel=True,
            seed=42,
        )
        if not success:
            print(f"  [Fatal] Training failed at {save_dir}")
            break

        current_load_dir = save_dir

        # 2. Evaluate checkpoint
        print(f"  [Eval] Running comprehensive evaluation battery on {save_dir}...")
        eval_res = eval_checkpoint(save_dir, seq_len=16, zero_boundary=True, coord_channel=True)

        rec = {
            "epoch": cum_epochs,
            "checkpoint": save_dir,
            "eval": eval_res,
        }
        records.append(rec)

        in_acc = eval_res.get("intact", {}).get("acc", 0.0)
        sh_acc = eval_res.get("shuffle_batch", {}).get("acc", 0.0)
        cz_acc = eval_res.get("coord_zeroed", {}).get("acc", 0.0)
        cs_acc = eval_res.get("coord_shuffled", {}).get("acc", 0.0)
        g_id = eval_res.get("g_identity", 0.0)
        g_cz = eval_res.get("g_coord_zeroed", 0.0)
        slots = eval_res.get("intact", {}).get("slots", [])
        slots_str = ", ".join(f"s{i}:{s:.1f}%" for i, s in enumerate(slots))

        print(f"  --> Epoch {cum_epochs:4d} | Intact: {in_acc:5.1f}% | Shuf: {sh_acc:5.1f}% | CoordZero: {cz_acc:5.1f}% | CoordShuf: {cs_acc:5.1f}%")
        print(f"      G_id: {g_id:+5.1f}% (±{eval_res.get('g_identity_sem', 0.0):.1f}%) | G_coord_ablation: {g_cz:+5.1f}% (±{eval_res.get('g_coord_zeroed_sem', 0.0):.1f}%)")
        print(f"      Slots: [{slots_str}]")

        # Check pre-registered criteria
        if len(slots) >= 3:
            s1, s2 = slots[1], slots[2]
            if s1 >= 60.0 and s2 >= 60.0 and g_cz >= 8.0 and g_id >= 10.0:
                print(f"\n  ★ [PRE-REGISTERED SUCCESS CRITERIA MET at Epoch {cum_epochs}] ★")
                print(f"    Interior slots rescued: s1={s1:.1f}%, s2={s2:.1f}% >= 60%!")
                print(f"    Coordinate sensitivity: G_coord_zeroed={g_cz:+.1f}% >= +8%!")
                print(f"    Instance recurrence: G_identity={g_id:+.1f}% >= +10%!")

        if cum_epochs >= 500 and len(slots) >= 3:
            s1, s2 = slots[1], slots[2]
            if s1 < 53.0 and s2 < 53.0:
                print(f"    [Interim Observation] At epoch {cum_epochs}, interior slots remain pinned at chance (s1={s1:.1f}%, s2={s2:.1f}%).")

    # Save report
    with open(report_output, "w") as f:
        json.dump({
            "campaign_name": "campaign_arm2_coordinate",
            "seq_len": 16,
            "dev_steps": 16,
            "zero_boundary": True,
            "coord_channel": True,
            "total_epochs": cum_epochs,
            "records": records,
        }, f, indent=2)
    print(f"\n✓ Saved coordinate campaign results to '{report_output}'")

def main():
    parser = argparse.ArgumentParser(description="Titan Text Coordinate Campaign")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--increment", type=int, default=100)
    args = parser.parse_args()

    run_coordinate_campaign(total_epochs=args.epochs, increment=args.increment)

if __name__ == "__main__":
    main()
