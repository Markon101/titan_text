#!/usr/bin/env python3
"""Causal Recurrence Horizon Campaign (CHCD).

Evaluates whether extending the developmental recurrence depth from tau=16 to tau=48
enables sequential state transport across 4 chunks on L=16 under the Causal Cellular Automaton.

Protocol:
  1. Train CCA on iterated-parity-dense with --causal-stencil --zero-boundary
     at dev_steps=48 for 100 epochs across N=5 seeds [42, 101, 202, 303, 404].
  2. Evaluate intact accuracy across latent budgets [0, 4, 8, 12, 16, 24, 32, 40, 48].
  3. Evaluate state lesion and batch-shuffle ablation (G_identity).
  4. Run latent representation and causal influence probes at tau=48.
  5. Compare against tau=16 baseline from checkpoints/campaign_cca_parity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

BIN = "./target/release/titan_text"

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, proc.stdout

def main():
    seeds = [42, 101, 202, 303, 404]
    base_dir = "checkpoints/campaign_horizon_48"
    os.makedirs(base_dir, exist_ok=True)
    report_file = Path("reports/campaign_horizon_48.json")

    print("=" * 80)
    print("STARTING CAUSAL RECURRENCE HORIZON CAMPAIGN (CHCD): TAU = 48 ON L = 16")
    print("Task: iterated-parity-dense | Stencil: Causal DAG | Boundary: Zero Dirichlet")
    print(f"Seeds: {seeds} | Epochs: 100 | Dev Steps: 48")
    print("=" * 80)

    # 1. Training loop across 5 seeds
    t0 = time.time()
    train_results = {}
    for s in seeds:
        s_dir = os.path.join(base_dir, f"seed_{s}")
        os.makedirs(s_dir, exist_ok=True)
        print(f"\n[Training] Seed {s} for 100 epochs with tau=48...")
        cmd_train = [
            BIN, "train",
            "--task", "iterated-parity-dense",
            "--save-dir", s_dir,
            "--epochs", "100",
            "--dev-steps", "48",
            "--seq-len", "16",
            "--seed", str(s),
            "--causal-stencil",
            "--zero-boundary",
            "--state-norm", "bounded",
        ]
        st = time.time()
        code, out = run_cmd(cmd_train)
        dur = time.time() - st
        if code != 0:
            print(f"  [ERROR] Training failed on seed {s}:\n{out}")
            sys.exit(1)
        print(f"  ✓ Seed {s} completed in {dur:.1f}s")
        train_results[str(s)] = {"train_time": dur}

    total_train_time = time.time() - t0
    print(f"\nAll 5 seeds trained successfully in {total_train_time:.1f}s")

    # 2. Evaluation across latent budgets
    budgets = [0, 4, 8, 12, 16, 24, 32, 40, 48]
    budgets_str = ",".join(str(b) for b in budgets)

    eval_results = {}
    sweep_curves = {b: {"accs": [], "slots": [[] for _ in range(4)]} for b in budgets}
    g_identities = []

    print("\n" + "=" * 80)
    print("EVALUATING LATENT COMPUTE BUDGET SWEEPS & CAUSAL LESIONS (B=32)")
    print("=" * 80)

    for s in seeds:
        s_dir = os.path.join(base_dir, f"seed_{s}")

        # Intact sweep
        cmd_sweep = [
            BIN, "sweep",
            "--load-dir", s_dir,
            "--task", "iterated-parity-dense",
            "--seq-len", "16",
            "--budgets", budgets_str,
            "--batch-size", "32",
            "--causal-stencil",
            "--zero-boundary",
            "--seeds", str(s),
        ]
        code, out_intact = run_cmd(cmd_sweep)
        if code != 0:
            print(f"  [ERROR] Sweep failed on seed {s}:\n{out_intact}")
            continue

        # Shuffle ablation at tau=48
        cmd_shuffle = [
            BIN, "sweep",
            "--load-dir", s_dir,
            "--task", "iterated-parity-dense",
            "--seq-len", "16",
            "--budgets", "48",
            "--batch-size", "32",
            "--causal-stencil",
            "--zero-boundary",
            "--lesion-shuffle",
            "--seeds", str(s),
        ]
        _, out_shuffle = run_cmd(cmd_shuffle)

        # Parse intact sweep
        current_tick = None
        seed_budget_data = {}
        for line in out_intact.splitlines():
            line_s = line.strip()
            if line_s.startswith("|") and any(f" {b} " in line_s or f"| {b} " in line_s for b in budgets):
                parts = [p.strip() for p in line_s.split("|")]
                if len(parts) >= 3 and parts[1].isdigit():
                    current_tick = int(parts[1])
                    acc_val = float(parts[2])
                    seed_budget_data[current_tick] = {"acc": acc_val, "slots": []}
            elif "Query Slots:" in line_s and current_tick is not None:
                parts = line_s.split("Query Slots:")[1].split(",")
                for p in parts:
                    slot_acc = float(p.split(":")[1].strip().replace("%", ""))
                    seed_budget_data[current_tick]["slots"].append(slot_acc)
                current_tick = None

        # Parse shuffle accuracy
        shuffle_acc = 50.0
        for line in out_shuffle.splitlines():
            line_s = line.strip()
            if line_s.startswith("|") and " 48 " in line_s:
                parts = [p.strip() for p in line_s.split("|")]
                if len(parts) >= 3:
                    shuffle_acc = float(parts[2])

        intact_48 = seed_budget_data.get(48, {}).get("acc", 50.0)
        g_id = intact_48 - shuffle_acc
        g_identities.append(g_id)

        eval_results[str(s)] = {
            "intact_48": intact_48,
            "shuffle_48": shuffle_acc,
            "g_identity": g_id,
            "budgets": seed_budget_data,
        }

        for b in budgets:
            if b in seed_budget_data:
                sweep_curves[b]["accs"].append(seed_budget_data[b]["acc"])
                for si in range(min(4, len(seed_budget_data[b]["slots"]))):
                    sweep_curves[b]["slots"][si].append(seed_budget_data[b]["slots"][si])

        print(f"  Seed {s} Intact(tau=48): {intact_48:.1f}% | Shuffled: {shuffle_acc:.1f}% | G_identity: {g_id:+.1f}%")
        slots_48 = seed_budget_data.get(48, {}).get("slots", [])
        print(f"    └─ Slots at tau=48: {[round(x, 1) for x in slots_48]}")

    # 3. Latent Representation & Influence Probe at tau=48 (Seed 42)
    print("\n" + "=" * 80)
    print("RUNNING LATENT REPRESENTATION & CAUSAL INFLUENCE PROBES AT TAU = 48")
    print("=" * 80)
    cmd_influence = [
        BIN, "influence",
        "--load-dir", os.path.join(base_dir, "seed_42"),
        "--budgets", "48",
        "--seq-len", "16",
        "--causal-stencil",
        "--zero-boundary",
    ]
    _, out_probe = run_cmd(cmd_influence)
    probe_results = {}
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

    print(f"  Probes at tau=48:")
    for k, v in probe_results.items():
        print(f"    {k:<20}: {v}")

    # Aggregate trajectory across seeds
    print("\n" + "=" * 80)
    print("AGGREGATE PROGRESSION ACROSS LATENT TICKS (N=5 SEEDS)")
    print("=" * 80)
    summary_progression = []
    print(f"{'Tick':<6} | {'Intact Acc':<16} | {'Slot 0':<10} | {'Slot 1':<10} | {'Slot 2':<10} | {'Slot 3':<10}")
    print("-" * 75)
    for b in budgets:
        accs = sweep_curves[b]["accs"]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        slot_means = []
        for si in range(4):
            s_vals = sweep_curves[b]["slots"][si]
            m = sum(s_vals) / len(s_vals) if s_vals else 0.0
            slot_means.append(m)
        summary_progression.append({
            "tick": b,
            "mean_acc": mean_acc,
            "slots": slot_means,
        })
        slots_str = " | ".join(f"{s:7.1f}%" for s in slot_means)
        print(f"{b:<6d} | {mean_acc:6.1f}%          | {slots_str}")

    mean_g_id = sum(g_identities) / len(g_identities) if g_identities else 0.0

    output_data = {
        "campaign": "Causal Recurrence Horizon Campaign (CHCD)",
        "dev_steps_trained": 48,
        "total_epochs": 100,
        "seeds": seeds,
        "mean_g_identity_tau48": mean_g_id,
        "progression": summary_progression,
        "probe_results_tau48": probe_results,
        "seed_evals": eval_results,
        "total_train_time": total_train_time,
    }

    with open(report_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Saved campaign results to '{report_file}'")

if __name__ == "__main__":
    main()
