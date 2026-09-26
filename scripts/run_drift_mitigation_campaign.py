#!/usr/bin/env python3
"""Carry Drift Mitigation Campaign.

Evaluates discrete carry quantization (STE Round, STE Sign, Bistable) against continuous
floating-point baseline (None) on OOD length generalization (L=16 -> 32 -> 64) for iterated-parity.
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
    """Parse Titan NCA loss and accuracy from benchmark output."""
    res = {}
    # Look for: "Titan NCA          : Loss = 0.6452 ± 0.0773, Acc =  68.8% ± 10.8%"
    # or single seed: "Titan NCA          : Loss = 0.8067, Acc =  31.2%"
    m_multi = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%", out)
    if m_multi:
        res["loss_mean"] = float(m_multi.group(1))
        res["loss_std"] = float(m_multi.group(2))
        res["acc_mean"] = float(m_multi.group(3))
        res["acc_std"] = float(m_multi.group(4))
        return res
    m_single = re.search(r"Titan NCA\s+:\s+Loss\s*=\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%", out)
    if m_single:
        res["loss_mean"] = float(m_single.group(1))
        res["loss_std"] = 0.0
        res["acc_mean"] = float(m_single.group(2))
        res["acc_std"] = 0.0
        return res
    return res

def main():
    modes = ["none", "ste_sign", "bistable", "ste_round"]
    train_seeds = [42, 101, 202]
    eval_lengths = [16, 32, 64]
    eval_seeds_str = "42,43,44,45,46"
    
    base_dir = "checkpoints/campaign_drift_mitigation"
    os.makedirs(base_dir, exist_ok=True)
    report_file = Path("reports/drift_mitigation_campaign.json")
    os.makedirs(report_file.parent, exist_ok=True)

    print("=" * 80)
    print("TITAN TEXT: CARRY DRIFT MITIGATION EXPERIMENTAL CAMPAIGN")
    print(f"Modes: {modes}")
    print(f"Train Seeds: {train_seeds} | Epochs: 50 | Dev Steps: 16 | Carry Channels: 16")
    print(f"OOD Length Generalization: {eval_lengths}")
    print("=" * 80)

    results = {mode: {L: [] for L in eval_lengths} for mode in modes}

    t0 = time.time()
    for mode in modes:
        print(f"\n[{mode.upper()}] Starting evaluation...")
        for s in train_seeds:
            s_dir = os.path.join(base_dir, f"{mode}_seed_{s}")
            os.makedirs(s_dir, exist_ok=True)
            
            # Train model
            cmd_train = [
                BIN, "train",
                "--task", "iterated-parity",
                "--save-dir", s_dir,
                "--epochs", "50",
                "--dev-steps", "16",
                "--seq-len", "16",
                "--batch-size", "32",
                "--lr", "0.003",
                "--seed", str(s),
                "--carry-channels", "16",
                "--carry-quantization", mode,
            ]
            st = time.time()
            code, out = run_cmd(cmd_train)
            dur = time.time() - st
            if code != 0:
                print(f"  [ERROR] Training failed for mode {mode} seed {s}:\n{out}")
                continue
            print(f"  ✓ [{mode}] Seed {s} trained in {dur:.1f}s")

            # Evaluate on L in [16, 32, 64]
            for L in eval_lengths:
                # Use batch size 16 for L=64 to avoid memory spikes
                bs = "16" if L >= 64 else "32"
                cmd_bench = [
                    BIN, "benchmark",
                    "--task", "iterated-parity",
                    "--checkpoint", s_dir,
                    "--seq-len", str(L),
                    "--dev-steps", str(L),
                    "--batch-size", bs,
                    "--seeds", eval_seeds_str,
                ]
                code, out = run_cmd(cmd_bench)
                if code != 0:
                    print(f"  [ERROR] Benchmark failed for mode {mode} seed {s} L={L}:\n{out}")
                    continue
                metrics = parse_benchmark_output(out)
                if "acc_mean" in metrics:
                    acc = metrics["acc_mean"]
                    results[mode][L].append(acc)
                    print(f"    L={L:2d} (T={L:2d}): Acc = {acc:5.1f}%")
                else:
                    print(f"    [WARN] Could not parse metrics for mode {mode} seed {s} L={L}")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"CAMPAIGN COMPLETE IN {total_time:.1f}s")
    print("=" * 80)

    # Summary table
    print("\n─── CARRY DRIFT MITIGATION: OOD LENGTH GENERALIZATION RESULTS ─────────────")
    print(f"{'Quantization Mode':<18} | {'L=16 (T=16)':<16} | {'L=32 (T=32)':<16} | {'L=64 (T=64)':<16}")
    print("───────────────────┼──────────────────┼──────────────────┼─────────────────")
    
    summary = {}
    for mode in modes:
        row = [f"{mode:<18}"]
        mode_summary = {}
        for L in eval_lengths:
            accs = results[mode][L]
            if accs:
                mean = sum(accs) / len(accs)
                variance = sum((x - mean) ** 2 for x in accs) / max(1, len(accs) - 1)
                std = variance ** 0.5
                row.append(f"{mean:5.1f}% ± {std:4.1f}%")
                mode_summary[str(L)] = {"mean": mean, "std": std, "raw": accs}
            else:
                row.append("N/A")
                mode_summary[str(L)] = None
        print(" | ".join(row))
        summary[mode] = mode_summary
    print("──────────────────────────────────────────────────────────────────────────")

    with open(report_file, "w") as f:
        json.dump({"summary": summary, "results": results, "total_time": total_time}, f, indent=2)
    print(f"Results saved to {report_file}")

if __name__ == "__main__":
    main()
