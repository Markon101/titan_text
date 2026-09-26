#!/usr/bin/env python3
"""Phase 5: Causal Double Dissociation and Channel Lesion Matrix.

Evaluates the causal necessity of Carry Channels vs Hidden Channels across
5 seeds (42, 101, 202, 303, 404) on column-arithmetic for:
1. Baseline Causal NCA (Cc=0)
2. ECR-16 Bi (Cc=16)
3. ECR-32 Bi (Cc=32)

Four Intervention Conditions:
- Intact: Unperturbed forward evaluation.
- Carry Lesion: Ablate exact carry channel partition (Cc zeroed at eval).
- Hidden Lesion: Ablate first 32 hidden channels (Ch zeroed at eval).
- Shuffled Recurrence: Permute state across batch at each tick (destroy temporal/spatial continuity).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BIN = "./target/release/titan_text"
CHECKPOINT_BASE = "checkpoints/campaign_mechanism_matrix"
SEEDS = [42, 101, 202, 303, 404]
EVAL_SEEDS = "42,43,44,45,46"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, proc.stdout


def parse_benchmark(output: str) -> dict[str, float]:
    metrics = {}
    m = re.search(r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+)(?:\s*±\s*[0-9.]+)?,\s*Acc\s*=\s*([0-9.]+)%", output)
    if m:
        metrics["loss"] = float(m.group(1))
        metrics["acc"] = float(m.group(2))
    else:
        # Fallback to any loss and acc found in line
        m_loss = re.search(r"Loss\s*=\s*([0-9.]+)", output)
        m_acc = re.search(r"Acc\s*=\s*([0-9.]+)%", output)
        if m_loss and m_acc:
            metrics["loss"] = float(m_loss.group(1))
            metrics["acc"] = float(m_acc.group(2))
        else:
            print(f"  [WARN] Failed to parse benchmark output: {output[:150]}...")
    return metrics


def evaluate_condition(model_prefix: str, carry_indices: str | None, hidden_indices: str, condition_type: str) -> dict:
    print(f"\nEvaluating Condition: {model_prefix} -> {condition_type}...")
    seed_accs = []
    seed_losses = []

    for s in SEEDS:
        ckpt_dir = f"{CHECKPOINT_BASE}/{model_prefix}_seed_{s}"
        if not os.path.exists(ckpt_dir):
            print(f"  [WARN] Checkpoint {ckpt_dir} not found, skipping...")
            continue

        cmd = [
            BIN, "benchmark",
            "--task", "column-arithmetic",
            "--checkpoint", ckpt_dir,
            "--seq-len", "16",
            "--dev-steps", "16",
            "--batch-size", "32",
            "--seeds", EVAL_SEEDS,
        ]

        if condition_type == "carry_lesion":
            if carry_indices:
                cmd.extend(["--lesion-channels", carry_indices])
            else:
                # No carry channels in baseline; lesioning dummy channels 48..63
                cmd.extend(["--lesion-channels", ",".join(str(i) for i in range(48, 64))])
        elif condition_type == "hidden_lesion":
            cmd.extend(["--lesion-channels", hidden_indices])
        elif condition_type == "shuffled":
            cmd.append("--lesion-shuffle")

        code, out = run_cmd(cmd)
        if code != 0:
            print(f"  [ERROR] Benchmark failed for seed {s}:\n{out[:200]}")
            continue

        m = parse_benchmark(out)
        acc = m.get("acc", 0.0)
        loss = m.get("loss", 0.0)
        seed_accs.append(acc)
        seed_losses.append(loss)
        print(f"  Seed {s}: Acc = {acc:5.1f}%, Loss = {loss:6.4f}")

    n = len(seed_accs)
    mean_acc = sum(seed_accs) / n if n else 0.0
    var_acc = sum((x - mean_acc) ** 2 for x in seed_accs) / max(1, n - 1) if n else 0.0
    std_acc = var_acc ** 0.5
    mean_loss = sum(seed_losses) / n if n else 0.0

    return {
        "condition": condition_type,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_loss": mean_loss,
        "seed_accs": seed_accs,
        "seed_losses": seed_losses,
    }


def main():
    print("=" * 80)
    print("TITAN TEXT: PHASE 5 CAUSAL DOUBLE DISSOCIATION CAMPAIGN")
    print("Multi-seed evaluation across Intact, Carry Lesion, Hidden Lesion, Shuffled")
    print("=" * 80)

    carry_16_indices = ",".join(str(i) for i in range(48, 64))
    carry_32_indices = ",".join(str(i) for i in range(32, 64))
    hidden_indices = ",".join(str(i) for i in range(0, 32))

    configs = [
        {"id": "C1_baseline_causal", "name": "Baseline Causal NCA (Cc=0)", "carry_indices": None},
        {"id": "C2_ecr16_k1", "name": "ECR-16 Bi (Cc=16)", "carry_indices": carry_16_indices},
        {"id": "C3_ecr32_k1", "name": "ECR-32 Bi (Cc=32)", "carry_indices": carry_32_indices},
    ]

    conditions = ["intact", "carry_lesion", "hidden_lesion", "shuffled"]

    full_results = {}

    for cfg in configs:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n{'=' * 40}\nMODEL: {cname}\n{'=' * 40}")
        full_results[cid] = {"name": cname, "conditions": {}}

        for cond in conditions:
            res = evaluate_condition(cid, cfg["carry_indices"], hidden_indices, cond)
            full_results[cid]["conditions"][cond] = res

    # Compute Causal Double Dissociation Metrics
    print("\n" + "=" * 80)
    print("CAUSAL DOUBLE DISSOCIATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model Architecture':<28} | {'Intact Acc':<12} | {'Carry Lesion':<14} | {'Hidden Lesion':<14} | {'Delta Carry':<12}")
    print("-" * 88)

    for cid, data in full_results.items():
        cname = data["name"]
        conds = data["conditions"]
        intact_acc = conds["intact"]["mean_acc"]
        intact_std = conds["intact"]["std_acc"]
        carry_acc = conds["carry_lesion"]["mean_acc"]
        carry_std = conds["carry_lesion"]["std_acc"]
        hidden_acc = conds["hidden_lesion"]["mean_acc"]
        hidden_std = conds["hidden_lesion"]["std_acc"]
        delta_carry = intact_acc - carry_acc

        print(
            f"{cname:<28} | {intact_acc:5.1f}±{intact_std:4.1f}% | "
            f"{carry_acc:5.1f}±{carry_std:4.1f}% | "
            f"{hidden_acc:5.1f}±{hidden_std:4.1f}% | "
            f"-{delta_carry:5.1f}%"
        )

    out_file = Path("reports/causal_double_dissociation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved complete results to {out_file}")


if __name__ == "__main__":
    main()
