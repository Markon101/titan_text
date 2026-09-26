#!/usr/bin/env python3
"""Phase 3: ECR Mechanism Matrix across 10 Controlled Configurations.

Isolates:
- Capacity: Cc=0 (Chidden=64) vs Cc=16 (Chidden=48) vs Cc=32 (Chidden=32)
- Carry Semantics: Stationary local features vs hyperbolic directional advection
- Topology: Stride k=1 vs k=2 vs k=4
- Recurrence Depth: T=16 vs T=24
- Numerical Stability: Continuous vs STE Sign
- Causal Necessity: Intact vs Carry Zeroed vs Carry Batch-Shuffled at Evaluation

Task: column-arithmetic (L=16, 100 epochs, B=32, AdamW lr=0.003)
Seeds: N=5 seeds [42, 101, 202, 303, 404]
Evaluation: 5-batch pooled test set [42, 43, 44, 45, 46]
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

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc.returncode, proc.stdout

def parse_benchmark_output(out: str) -> dict[str, float]:
    res = {}
    m = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%", out)
    if m:
        res["val_loss_mean"] = float(m.group(1))
        res["val_loss_std"] = float(m.group(2))
        res["val_acc_mean"] = float(m.group(3))
        res["val_acc_std"] = float(m.group(4))
    return res

def parse_train_output(out: str) -> dict[str, float]:
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

def main():
    seeds = [42, 101, 202, 303, 404]
    eval_seeds_str = "42,43,44,45,46"
    base_dir = "checkpoints/campaign_mechanism_matrix"
    os.makedirs(base_dir, exist_ok=True)
    report_file = Path("reports/ecr_mechanism_matrix.json")
    os.makedirs(report_file.parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: ECR MECHANISM MATRIX (Phase 3)")
    print(f"Task: column-arithmetic | Lattice: L=16 | Seeds: {seeds} | Epochs: 100")
    print("=" * 80)

    # 1. Base configurations to train
    train_configs = [
        {
            "id": "C1_baseline_causal",
            "name": "1. Baseline Causal NCA (Cc=0, T=16)",
            "carry_channels": 0,
            "skip_stride": 1,
            "bidirectional": False,
            "dev_steps": 16,
            "quantization": "none",
            "causal": True,
        },
        {
            "id": "C2_ecr16_k1",
            "name": "2. ECR-16 Bi (k=1, T=16)",
            "carry_channels": 16,
            "skip_stride": 1,
            "bidirectional": True,
            "dev_steps": 16,
            "quantization": "none",
            "causal": False,
        },
        {
            "id": "C3_ecr32_k1",
            "name": "3. ECR-32 Bi (k=1, T=16)",
            "carry_channels": 32,
            "skip_stride": 1,
            "bidirectional": True,
            "dev_steps": 16,
            "quantization": "none",
            "causal": False,
        },
        {
            "id": "C4_ecr32_k2",
            "name": "4. ECR-32 Bi Skip-2 (k=2, T=16)",
            "carry_channels": 32,
            "skip_stride": 2,
            "bidirectional": True,
            "dev_steps": 16,
            "quantization": "none",
            "causal": False,
        },
        {
            "id": "C5_ecr32_k4",
            "name": "5. ECR-32 Bi Skip-4 (k=4, T=16)",
            "carry_channels": 32,
            "skip_stride": 4,
            "bidirectional": True,
            "dev_steps": 16,
            "quantization": "none",
            "causal": False,
        },
        {
            "id": "C6_ecr32_k4_t24",
            "name": "6. ECR-32 Bi Skip-4 Deep (k=4, T=24)",
            "carry_channels": 32,
            "skip_stride": 4,
            "bidirectional": True,
            "dev_steps": 24,
            "quantization": "none",
            "causal": False,
        },
        {
            "id": "C7_ecr32_k4_ste",
            "name": "7. ECR-32 Bi Skip-4 + STE-Sign (k=4, T=16)",
            "carry_channels": 32,
            "skip_stride": 4,
            "bidirectional": True,
            "dev_steps": 16,
            "quantization": "ste_sign",
            "causal": False,
        },
    ]

    t0 = time.time()
    trained_checkpoints = {}

    # Train all base configs across 5 seeds
    for cfg in train_configs:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n[Training] {cname}...")
        trained_checkpoints[cid] = []

        for s in seeds:
            s_dir = os.path.join(base_dir, f"{cid}_seed_{s}")
            os.makedirs(s_dir, exist_ok=True)
            trained_checkpoints[cid].append(s_dir)

            cmd_train = [
                BIN, "train",
                "--task", "column-arithmetic",
                "--save-dir", s_dir,
                "--epochs", "100",
                "--dev-steps", str(cfg["dev_steps"]),
                "--seq-len", "16",
                "--batch-size", "32",
                "--lr", "0.003",
                "--seed", str(s),
                "--carry-channels", str(cfg["carry_channels"]),
                "--carry-skip-stride", str(cfg["skip_stride"]),
                "--carry-quantization", cfg["quantization"],
            ]
            if cfg["bidirectional"]:
                cmd_train.append("--carry-bidirectional")
            if cfg["causal"]:
                cmd_train.append("--causal-stencil")

            st = time.time()
            code, out = run_cmd(cmd_train)
            dur = time.time() - st
            if code != 0:
                print(f"  [ERROR] Training failed for {cid} seed {s}:\n{out}")
                continue
            tm = parse_train_output(out)
            train_acc = tm.get("train_acc", 0.0)
            print(f"  ✓ Seed {s} trained in {dur:.1f}s (Train Acc: {train_acc:5.1f}%)")

    # Evaluation Matrix: 10 Experimental Conditions
    # Indices 32..64 correspond to carry channels in C=64, Cc=32
    carry_indices_str = ",".join(str(i) for i in range(32, 64))

    eval_conditions = [
        {"name": "1. Baseline Causal NCA (Cc=0, T=16)", "source_cfg": "C1_baseline_causal", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "2. ECR-16 Bi (Cc=16, k=1, T=16)", "source_cfg": "C2_ecr16_k1", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "3. ECR-32 Bi (Cc=32, k=1, T=16)", "source_cfg": "C3_ecr32_k1", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "4. ECR-32 Bi Skip-2 (Cc=32, k=2, T=16)", "source_cfg": "C4_ecr32_k2", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "5. ECR-32 Bi Skip-4 (Cc=32, k=4, T=16)", "source_cfg": "C5_ecr32_k4", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "6. ECR-32 Bi Skip-4 Deep (Cc=32, k=4, T=24)", "source_cfg": "C6_ecr32_k4_t24", "dev_steps": 24, "lesion": None, "shuffle": False},
        {"name": "7. ECR-32 Bi Skip-4 + STE-Sign (k=4, T=16)", "source_cfg": "C7_ecr32_k4_ste", "dev_steps": 16, "lesion": None, "shuffle": False},
        {"name": "8. ECR-32 Skip-4 Carry Lesion (Ablate Cc=32 at Eval)", "source_cfg": "C5_ecr32_k4", "dev_steps": 16, "lesion": carry_indices_str, "shuffle": False},
        {"name": "9. ECR-32 Skip-4 Carry Shuffle (Shuffle Cc across Batch)", "source_cfg": "C5_ecr32_k4", "dev_steps": 16, "lesion": None, "shuffle": True},
        {"name": "10. ECR-32 Deep Carry Lesion (Ablate Cc=32 at T=24)", "source_cfg": "C6_ecr32_k4_t24", "dev_steps": 24, "lesion": carry_indices_str, "shuffle": False},
    ]

    print("\n" + "=" * 80)
    print("EVALUATING MECHANISM MATRIX ACROSS 5 SEEDS & 5 EVAL BATCHES")
    print("=" * 80)

    matrix_results = []

    for ec in eval_conditions:
        ename = ec["name"]
        scfg = ec["source_cfg"]
        dsteps = ec["dev_steps"]
        lesion = ec["lesion"]
        shuffle = ec["shuffle"]

        print(f"\n[Evaluating] {ename}...")
        seed_accs = []
        seed_losses = []

        for idx, s in enumerate(seeds):
            s_dir = trained_checkpoints[scfg][idx]
            cmd_bench = [
                BIN, "benchmark",
                "--task", "column-arithmetic",
                "--checkpoint", s_dir,
                "--seq-len", "16",
                "--dev-steps", str(dsteps),
                "--batch-size", "32",
                "--seeds", eval_seeds_str,
            ]
            if lesion:
                cmd_bench.extend(["--lesion-channels", lesion])
            if shuffle:
                cmd_bench.append("--lesion-shuffle")

            code, out = run_cmd(cmd_bench)
            if code != 0:
                print(f"  [ERROR] Benchmark failed for {ename} seed {s}:\n{out}")
                continue
            metrics = parse_benchmark_output(out)
            acc = metrics.get("val_acc_mean", 0.0)
            loss = metrics.get("val_loss_mean", 0.0)
            seed_accs.append(acc)
            seed_losses.append(loss)
            print(f"  Seed {s}: Acc = {acc:5.1f}%, Loss = {loss:6.4f}")

        n = len(seed_accs)
        mean_acc = sum(seed_accs) / n if n else 0.0
        var_acc = sum((x - mean_acc) ** 2 for x in seed_accs) / max(1, n - 1) if n else 0.0
        std_acc = var_acc ** 0.5

        mean_loss = sum(seed_losses) / n if n else 0.0

        res_entry = {
            "name": ename,
            "seed_accs": seed_accs,
            "seed_losses": seed_losses,
            "mean_val_acc": mean_acc,
            "std_val_acc": std_acc,
            "mean_val_loss": mean_loss,
        }
        matrix_results.append(res_entry)
        print(f"  --> {ename} OVERALL: {mean_acc:5.1f}% ± {std_acc:4.1f}%")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"MECHANISM MATRIX EVALUATION COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # Print Final Mechanism Matrix Table
    print("\n─── ECR MECHANISM MATRIX: FINAL EXPERIMENTAL RESULTS (N=5 SEEDS) ────────────")
    print(f"{'Condition':<46} | {'Val Acc (Mean ± Std)':<22} | {'Val Loss':<10} | {'Delta vs Baseline'}")
    print("───────────────────────────────────────────────┼────────────────────────┼────────────┼──────────────────")
    base_acc = matrix_results[0]["mean_val_acc"]
    for r in matrix_results:
        cname = r["name"]
        m = r["mean_val_acc"]
        s = r["std_val_acc"]
        l = r["mean_val_loss"]
        delta = m - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"{cname:<46} | {m:5.1f}% ± {s:4.1f}%          | {l:6.4f}     | {sign}{delta:5.1f}%")
    print("───────────────────────────────────────────────┴────────────────────────┴────────────┴──────────────────")

    with open(report_file, "w") as f:
        json.dump({"matrix_results": matrix_results, "total_time": total_time}, f, indent=2)
    print(f"Results saved to {report_file}")

if __name__ == "__main__":
    main()
