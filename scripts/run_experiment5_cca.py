#!/usr/bin/env python3
"""Decisive Experiment 5: Causal Cellular Automata (CCA) Campaign Runner.

Tests the DAG Total-Order Hypothesis derived by the DeepSeek Council:
  - Replacing the symmetric radius-1 stencil N(i) = {i-1, i, i+1} with the causal stencil N_→(i) = {i-1, i}
  - Converts isotropic relaxation (which is mathematically incapable of sequential composition due to symmetric fixed-point reflection and exponential attenuation) into an exact spatial fold.

Executes both tracks across N=5 pre-registered seeds [42, 101, 202, 303, 404]:
  - Track 1 (CCA-Parity): iterated-parity-dense with --causal-stencil
  - Track 2 (CCA-Sum):    iterated-sum-dense with --causal-stencil

Evaluates at Epochs 100, 200, 300:
  - Intact sweep across latent budgets [0, 1, 2, 4, 8, 12, 16]
  - Interventions: zero-tick (b=0), state lesion, inverted gain (-1.0), batch-state shuffle, sham noise (0.05)
  - Identity gap G_identity = intact - batch_shuffle
  - Per-query-slot accuracy for slots 0, 1, 2, 3 (cells 3, 7, 11, 15)
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
    causal_stencil: bool = True,
    feedback_mode: str = "none",
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
    ]
    if causal_stencil:
        cmd.append("--causal-stencil")
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
    causal_stencil: bool = True,
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
    if causal_stencil:
        cmd_intact.append("--causal-stencil")
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
        if causal_stencil:
            cmd.append("--causal-stencil")
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

    g_id_mean, g_id_sem, g_id_std, g_id_d, g_id_seeds = compute_paired_gap(stats_shuffle)
    g_lesion_mean, g_lesion_sem, _, g_lesion_d, _ = compute_paired_gap(stats_lesion)
    g_invert_mean, g_invert_sem, _, g_invert_d, _ = compute_paired_gap(stats_invert)
    g_sham_mean, g_sham_sem, _, g_sham_d, _ = compute_paired_gap(stats_sham)

    b0_acc = sweep_curve.get("0", {}).get("acc", 0.0)
    g_recurrence = stats_intact_target["acc"] - b0_acc

    return {
        "sweep_curve": sweep_curve,
        "target_budget": target_budget,
        "intact": stats_intact_target,
        "zero_tick": sweep_curve.get("0", {}),
        "lesion": stats_lesion,
        "invert": stats_invert,
        "shuffle": stats_shuffle,
        "sham": stats_sham,
        "G_recurrence": g_recurrence,
        "G_identity": {
            "mean": g_id_mean,
            "sem": g_id_sem,
            "std": g_id_std,
            "cohen_d": g_id_d,
            "per_seed": g_id_seeds,
        },
        "G_lesion": {
            "mean": g_lesion_mean,
            "sem": g_lesion_sem,
            "cohen_d": g_lesion_d,
        },
        "G_invert": {
            "mean": g_invert_mean,
            "sem": g_invert_sem,
            "cohen_d": g_invert_d,
        },
        "G_sham": {
            "mean": g_sham_mean,
            "sem": g_sham_sem,
            "cohen_d": g_sham_d,
        },
    }

def run_campaign_arm(
    arm_name: str,
    task: str,
    base_dir: str,
    seeds: list[int],
    milestones: list[int],
    dev_steps: int = 16,
    seq_len: int = 16,
    causal_stencil: bool = True,
    feedback_mode: str = "none",
) -> dict:
    print(f"\n{'='*80}")
    print(f"STARTING CAMPAIGN ARM: {arm_name.upper()} (Task: {task}, Causal Stencil: {causal_stencil})")
    print(f"Seeds: {seeds} | Milestones: {milestones} epochs | Recurrence T: {dev_steps}")
    print(f"{'='*80}")

    seeds_str = ",".join(str(s) for s in seeds)
    milestone_results = {}
    prev_epoch = 0

    for m_epoch in milestones:
        inc_epochs = m_epoch - prev_epoch
        print(f"\n--- Training Milestone {m_epoch} Epochs (+{inc_epochs} epochs across {len(seeds)} seeds) ---")
        t0 = time.time()

        for s in seeds:
            s_dir = os.path.join(base_dir, f"seed_{s}")
            prev_ckpt = os.path.join(s_dir, f"epoch_{prev_epoch}") if prev_epoch > 0 else None
            curr_ckpt = os.path.join(s_dir, f"epoch_{m_epoch}")

            ok = train_increment(
                load_dir=prev_ckpt,
                save_dir=curr_ckpt,
                task=task,
                epochs=inc_epochs,
                dev_steps=dev_steps,
                seq_len=seq_len,
                causal_stencil=causal_stencil,
                feedback_mode=feedback_mode,
                zero_boundary=True,
                seed=s,
            )
            if not ok:
                print(f"[Fatal] Training failed for seed {s} at milestone {m_epoch}")
                sys.exit(1)

        t_train = time.time() - t0
        print(f"  Training completed in {t_train:.1f}s. Evaluating all seeds...")

        t_eval_0 = time.time()
        seed_evals = {}
        for s in seeds:
            ckpt_path = os.path.join(base_dir, f"seed_{s}", f"epoch_{m_epoch}")
            res = eval_checkpoint(
                ckpt_dir=ckpt_path,
                seq_len=seq_len,
                seeds=seeds_str,
                causal_stencil=causal_stencil,
                zero_boundary=True,
            )
            seed_evals[str(s)] = res

        t_eval = time.time() - t_eval_0

        # Aggregate across seeds
        intact_accs = [seed_evals[str(s)]["intact"]["acc"] for s in seeds if str(s) in seed_evals]
        mean_intact = sum(intact_accs) / len(intact_accs) if intact_accs else 0.0
        var_intact = sum((a - mean_intact)**2 for a in intact_accs) / max(1, len(intact_accs) - 1) if len(intact_accs) > 1 else 0.0
        std_intact = math.sqrt(var_intact)

        g_id_means = [seed_evals[str(s)]["G_identity"]["mean"] for s in seeds if str(s) in seed_evals]
        mean_gid = sum(g_id_means) / len(g_id_means) if g_id_means else 0.0

        # Slot accuracies
        all_slots = [seed_evals[str(s)]["intact"]["slots"] for s in seeds if str(s) in seed_evals and seed_evals[str(s)]["intact"]["slots"]]
        avg_slots = []
        if all_slots:
            n_slots = len(all_slots[0])
            for i in range(n_slots):
                avg_slots.append(sum(s[i] for s in all_slots) / len(all_slots))

        print(f"  [Milestone {m_epoch}] Intact: {mean_intact:.2f}% ± {std_intact:.2f}% | G_identity: {mean_gid:+.2f}% | Slots: {[round(x, 1) for x in avg_slots]} | Eval time: {t_eval:.1f}s")

        milestone_results[str(m_epoch)] = {
            "epoch": m_epoch,
            "intact_mean": mean_intact,
            "intact_std": std_intact,
            "G_identity_mean": mean_gid,
            "avg_slots": avg_slots,
            "seed_evals": seed_evals,
            "t_train": t_train,
            "t_eval": t_eval,
        }

        prev_epoch = m_epoch

    return {
        "arm_name": arm_name,
        "task": task,
        "seeds": seeds,
        "milestones": milestones,
        "dev_steps": dev_steps,
        "seq_len": seq_len,
        "causal_stencil": causal_stencil,
        "milestone_results": milestone_results,
    }

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: CCA Campaign")
    parser.add_argument("--arm", choices=["all", "cca-parity", "cca-sum"], default="all")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    seeds = [42, 101, 202, 303, 404]
    milestones = [100, 200, 300] if args.epochs == 300 else [args.epochs]

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arm in ["all", "cca-parity"]:
        res_parity = run_campaign_arm(
            arm_name="cca_parity",
            task="iterated-parity-dense",
            base_dir="checkpoints/campaign_cca_parity",
            seeds=seeds,
            milestones=milestones,
            dev_steps=16,
            seq_len=16,
            causal_stencil=True,
            feedback_mode="none",
        )
        parity_path = out_dir / "campaign_exp5_cca_parity.json"
        with open(parity_path, "w") as f:
            json.dump(res_parity, f, indent=2)
        print(f"\nSaved CCA Parity results to: {parity_path}")

    if args.arm in ["all", "cca-sum"]:
        res_sum = run_campaign_arm(
            arm_name="cca_sum",
            task="iterated-sum-dense",
            base_dir="checkpoints/campaign_cca_sum",
            seeds=seeds,
            milestones=milestones,
            dev_steps=16,
            seq_len=16,
            causal_stencil=True,
            feedback_mode="none",
        )
        sum_path = out_dir / "campaign_exp5_cca_sum.json"
        with open(sum_path, "w") as f:
            json.dump(res_sum, f, indent=2)
        print(f"\nSaved CCA Sum results to: {sum_path}")

if __name__ == "__main__":
    main()
