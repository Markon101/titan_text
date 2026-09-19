#!/usr/bin/env python3
"""Automated runner for Escalation B: Auxiliary Intermediate Supervision Campaign.

Evaluates whether providing auxiliary supervision to intermediate carrier cells
allows the 1D NCA to escape the Local Receptive Field Trap (H_OPT) and establish
horizontal state transport, or confirms an intrinsic dynamical transport barrier (H_TRANSPORT).

Executes 1,000 epochs of training on L=16 with task 'iterated-parity-dense' and --zero-boundary,
evaluating at each 100-epoch checkpoint:
  1. Latent-tick budget sweep (tau in [0, 1, 2, 4, 8, 12, 16])
  2. Standard causal controls: zero tick, state lesion, inverted gain, batch-state shuffle, sham noise
  3. Identity-dependent recurrence gap G_identity across N=5 seeds
  4. Per-query-slot resolution (s0, s1, s2, s3)
  5. Closed-form latent representation probes (local, chunk 0 prefix, cumulative parity)
  6. Empirical causal influence mapping at intermediate and final checkpoints
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
    report_file = os.path.join(tmp_dir, f"sweep_b_eval_{os.getpid()}.json")

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

def run_campaign(
    total_epochs: int = 1000,
    increment: int = 100,
    seq_len: int = 16,
    dev_steps: int = 16,
    task: str = "iterated-parity-dense",
    out_dir: str = "checkpoints/campaign_b_dense_l16",
    report_path: str = "reports/campaign_arm_b_dense.json",
):
    print("╔══════════════════════════════════════════════════════════════════════════════════════╗")
    print("║ TITAN TEXT · ESCALATION B CAMPAIGN (Auxiliary Intermediate Supervision)              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════════╝")
    print(f"  Task                  : {task}")
    print(f"  Sequence Length (L)   : {seq_len}")
    print(f"  Recurrence Ticks (tau): {dev_steps}")
    print(f"  Total Epochs          : {total_epochs} (in increments of {increment})")
    print(f"  Checkpoint Directory  : {out_dir}")
    print(f"  Report Destination    : {report_path}\n")

    os.makedirs(out_dir, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    records = []
    current_load_dir = None

    for epoch_target in range(increment, total_epochs + 1, increment):
        ckpt_dir = os.path.join(out_dir, f"step_{epoch_target:04d}")
        print(f"\n─── EPOCH {epoch_target}/{total_epochs} ─────────────────────────────────────────────────────────────")

        # Train increment
        success = train_increment(
            load_dir=current_load_dir,
            save_dir=ckpt_dir,
            task=task,
            epochs=increment,
            dev_steps=dev_steps,
            seq_len=seq_len,
            zero_boundary=True,
            seed=42,
        )
        if not success:
            print(f"Aborting campaign at epoch {epoch_target} due to training failure.")
            break

        current_load_dir = ckpt_dir

        # Evaluate checkpoint across full battery
        print(f"  [Eval] Evaluating checkpoint {ckpt_dir} across N=5 seeds & controls...")
        eval_res = eval_checkpoint(ckpt_dir, seq_len=seq_len, zero_boundary=True)

        if eval_res:
            intact = eval_res["intact"]
            slots = intact.get("slots", [])
            slots_str = ", ".join(f"s{i}:{v:.1f}%" for i, v in enumerate(slots))
            g_slots_str = ", ".join(f"s{i}:{g:+.1f}%" for i, g in enumerate(eval_res.get("per_slot_g_identity", [])))
            probes = eval_res.get("probe_representations", {})

            print(f"  [Result @ τ={eval_res['target_budget']}]")
            print(f"    • Intact Acc     : {intact['acc']:.2f}% ± {intact['acc_std']:.2f}%  [{slots_str}]")
            print(f"    • Shuffle Acc    : {eval_res['shuffle_batch']['acc']:.2f}% (G_id = {eval_res['g_identity']:+.2f}% ± {eval_res['g_identity_sem']:.2f}%, Cohen's d = {eval_res['cohen_d_identity']:.2f})")
            print(f"    • Per-Slot G_id  : [{g_slots_str}]")
            print(f"    • Zero-Tick (τ=0): {eval_res['sweep_curve']['0']['acc']:.2f}%")
            print(f"    • State Lesion   : {eval_res['lesion_state']['acc']:.2f}%")
            if probes:
                print(f"    • Probes: Lin Local={probes.get('linear_local')}, Lin C0={probes.get('linear_chunk0')}, Lin Cumul={probes.get('linear_cumulative')}")

        records.append({
            "epoch": epoch_target,
            "checkpoint": ckpt_dir,
            "eval": eval_res,
        })

        # Save cumulative progress
        campaign_data = {
            "campaign_name": "campaign_arm_b_dense",
            "task": task,
            "seq_len": seq_len,
            "dev_steps": dev_steps,
            "zero_boundary": True,
            "total_epochs": total_epochs,
            "records": records,
        }
        with open(report_path, "w") as f:
            json.dump(campaign_data, f, indent=2)
        print(f"  [Saved] Updated report JSON -> '{report_path}'")

    print("\n✓ Escalation B Campaign execution finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Escalation B Campaign")
    parser.add_argument("--epochs", type=int, default=1000, help="Total epochs (default: 1000)")
    parser.add_argument("--increment", type=int, default=100, help="Checkpoint increment (default: 100)")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length (default: 16)")
    parser.add_argument("--dev-steps", type=int, default=16, help="Dev recurrence steps (default: 16)")
    parser.add_argument("--task", type=str, default="iterated-parity-dense", help="Task name")
    parser.add_argument("--out-dir", type=str, default="checkpoints/campaign_b_dense_l16")
    parser.add_argument("--report", type=str, default="reports/campaign_arm_b_dense.json")
    args = parser.parse_args()

    run_campaign(
        total_epochs=args.epochs,
        increment=args.increment,
        seq_len=args.seq_len,
        dev_steps=args.dev_steps,
        task=args.task,
        out_dir=args.out_dir,
        report_path=args.report,
    )
