#!/usr/bin/env python3
"""Phase 8: Zero-Shot Out-of-Distribution (OOD) Length Generalization.

Evaluates frozen-weight models trained at L=16 on test lengths L in {16, 24, 32, 48, 64}
under two operational regimes:
1. Fixed Recurrence Depth: T = 16 (tests causal lightcone breakdown boundary d_max = k * T)
2. Dynamically Scaled Recurrence Depth: T(L) = 4 + ceil(L / k) (tests true cellular scalability)

Tasks evaluated:
- column-arithmetic
- iterated-parity
"""

from __future__ import annotations

import json
import math
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


def main():
    eval_lengths = [16, 24, 32, 48, 64]
    eval_seeds_str = "42,43,44,45,46"
    report_file = Path("reports/ood_length_scaling_results.json")
    os.makedirs(report_file.parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: ZERO-SHOT OOD LENGTH GENERALIZATION CAMPAIGN (Phase 8)")
    print(f"Test Lengths L: {eval_lengths} | Evaluation Seeds: {eval_seeds_str}")
    print("=" * 80)

    # Models to benchmark:
    # 1. Baseline Causal NCA (k=1)
    # 2. ECR-16 Bi (k=1)
    # 3. ECR-16 Bi Skip-2 (k=2)
    # 4. ECR-32 Bi Skip-4 Deep (k=4)
    # 5. ECR-16 Parity Baseline
    # 6. ECR-16 Parity STE-Sign
    models = [
        {
            "id": "baseline_causal",
            "name": "Baseline Causal NCA (k=1, Cc=0)",
            "checkpoint": "checkpoints/campaign_mechanism_matrix/C1_baseline_causal_seed_42",
            "task": "column-arithmetic",
            "k": 1,
        },
        {
            "id": "ecr16_k1",
            "name": "ECR-16 Bi (k=1, Cc=16)",
            "checkpoint": "checkpoints/campaign_column_arithmetic/ecr-16_bi_k1_t16_seed_42",
            "task": "column-arithmetic",
            "k": 1,
        },
        {
            "id": "ecr16_k2",
            "name": "ECR-16 Bi Skip-2 (k=2, Cc=16)",
            "checkpoint": "checkpoints/campaign_column_arithmetic/ecr-16_bi_skip-2_k2_t16_seed_42",
            "task": "column-arithmetic",
            "k": 2,
        },
        {
            "id": "ecr32_k4_deep",
            "name": "ECR-32 Bi Skip-4 Deep (k=4, Cc=32)",
            "checkpoint": "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_42",
            "task": "column-arithmetic",
            "k": 4,
        },
        {
            "id": "parity_continuous",
            "name": "ECR-16 Parity Continuous (k=1)",
            "checkpoint": "checkpoints/campaign_drift_mitigation/none_seed_42",
            "task": "iterated-parity",
            "k": 1,
        },
        {
            "id": "parity_ste_sign",
            "name": "ECR-16 Parity STE-Sign (k=1)",
            "checkpoint": "checkpoints/campaign_drift_mitigation/ste_sign_seed_42",
            "task": "iterated-parity",
            "k": 1,
        },
    ]

    all_results = []
    t0 = time.time()

    for m in models:
        mid = m["id"]
        mname = m["name"]
        ck = m["checkpoint"]
        task = m["task"]
        k = m["k"]

        if not os.path.exists(ck):
            print(f"[SKIP] Checkpoint '{ck}' does not exist yet.")
            continue

        print(f"\nEvaluating: {mname} (Task: {task}, k={k})")
        model_entry = {"id": mid, "name": mname, "task": task, "stride_k": k, "regimes": {}}

        # 1. Regime 1: Fixed T = 16
        print("  [Regime 1: Fixed T=16]")
        fixed_results = []
        for L in eval_lengths:
            bs = 16 if L >= 64 else 32
            cmd = [
                BIN, "benchmark",
                "--task", task,
                "--checkpoint", ck,
                "--seq-len", str(L),
                "--dev-steps", "16",
                "--batch-size", str(bs),
                "--seeds", eval_seeds_str,
            ]
            code, out = run_cmd(cmd)
            if code != 0:
                print(f"    L={L}: [ERROR] {out}")
                continue
            metrics = parse_benchmark_output(out)
            acc = metrics.get("val_acc_mean", 0.0)
            loss = metrics.get("val_loss_mean", 0.0)
            reach = k * 16
            causal_status = "Complete" if reach >= L else f"Blackout (Reach={reach}<L={L})"
            fixed_results.append({"L": L, "T": 16, "acc": acc, "loss": loss, "causal_status": causal_status})
            print(f"    L={L:2d} (T=16): Acc = {acc:5.1f}%, Loss = {loss:6.4f} [{causal_status}]")
        model_entry["regimes"]["fixed_t16"] = fixed_results

        # 2. Regime 2: Dynamically Scaled Recurrence Depth T(L) = 4 + ceil(L / k)
        print("  [Regime 2: Dynamically Scaled T(L)]")
        dynamic_results = []
        for L in eval_lengths:
            bs = 16 if L >= 64 else 32
            # Slack formula from Information Velocity Specialist
            t_dynamic = 4 + math.ceil(L / k)
            cmd = [
                BIN, "benchmark",
                "--task", task,
                "--checkpoint", ck,
                "--seq-len", str(L),
                "--dev-steps", str(t_dynamic),
                "--batch-size", str(bs),
                "--seeds", eval_seeds_str,
            ]
            code, out = run_cmd(cmd)
            if code != 0:
                print(f"    L={L}: [ERROR] {out}")
                continue
            metrics = parse_benchmark_output(out)
            acc = metrics.get("val_acc_mean", 0.0)
            loss = metrics.get("val_loss_mean", 0.0)
            dynamic_results.append({"L": L, "T": t_dynamic, "acc": acc, "loss": loss, "causal_status": "Complete"})
            print(f"    L={L:2d} (T={t_dynamic:2d}): Acc = {acc:5.1f}%, Loss = {loss:6.4f} [Complete]")
        model_entry["regimes"]["dynamic_t"] = dynamic_results

        all_results.append(model_entry)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"ZERO-SHOT OOD LENGTH GENERALIZATION COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # Print Summary Table
    print("\n─── ZERO-SHOT OOD GENERALIZATION SUMMARY TABLE ─────────────────────────────")
    print(f"{'Model':<34} | {'L=16 (T=16)':<12} | {'L=32 (T=16)':<12} | {'L=32 (Dyn T)':<12} | {'L=64 (T=16)':<12} | {'L=64 (Dyn T)'}")
    print("───────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────")
    for r in all_results:
        mname = r["name"]
        f_res = {x["L"]: x["acc"] for x in r["regimes"].get("fixed_t16", [])}
        d_res = {x["L"]: x["acc"] for x in r["regimes"].get("dynamic_t", [])}
        acc_16_f = f_res.get(16, 0.0)
        acc_32_f = f_res.get(32, 0.0)
        acc_32_d = d_res.get(32, 0.0)
        acc_64_f = f_res.get(64, 0.0)
        acc_64_d = d_res.get(64, 0.0)
        print(f"{mname:<34} | {acc_16_f:5.1f}%       | {acc_32_f:5.1f}%       | {acc_32_d:5.1f}%       | {acc_64_f:5.1f}%       | {acc_64_d:5.1f}%")
    print("───────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────")

    with open(report_file, "w") as f:
        json.dump({"results": all_results, "total_time": total_time}, f, indent=2)
    print(f"Results saved to {report_file}")


if __name__ == "__main__":
    main()
