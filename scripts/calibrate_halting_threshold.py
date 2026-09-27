#!/usr/bin/env python3
"""Halting Threshold Calibration Sweep for Titan Text.

Sweeps halting thresholds theta across the relative_delta metric to map:
1. Mean recurrent depth tau(theta)
2. Latent compute variance / dynamic range across tokens
3. Output quality: Nearest Edit Similarity and Horizontal Symmetry
4. Character-class compute allocation: boundaries vs symbols vs newlines
5. Pareto efficiency curve

Identifies the calibrated threshold theta* operating in the optimal
recurrent depth regime (tau in [5, 8]).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional
import numpy as np


SEEDS = [42, 101, 202, 303, 404]
PROMPTS = [
    ("<BOX>\n", "box"),
    ("<MAZE>\n", "maze"),
    ("<DIAMOND>\n", "diamond"),
]
THRESHOLDS = [0.10, 0.14, 0.17, 0.20, 0.25, 0.30, 0.35]


def run_generate(
    cmd: List[str],
    scratch_dir: Path,
    timeout: int = 90,
) -> Dict[str, Any]:
    tmp_out = scratch_dir / f"tmp_calib_{os.getpid()}_{random.randint(10000, 99999)}.json"
    full_cmd = list(cmd) + ["--output", str(tmp_out), "--format", "json"]

    proc = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    if tmp_out.exists():
        with open(tmp_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            tmp_out.unlink()
        except OSError:
            pass
        return data

    raw = proc.stdout.strip()
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON found in output:\n{raw}")
    return json.loads(raw[s : e + 1])


def run_calibration(
    checkpoint: str = "checkpoints/ascii_v1",
    output_dir: str = "reports/raw/halting_calibration",
    thresholds: Optional[List[float]] = None,
    patience: int = 2,
    metric: str = "relative_delta",
    max_len: int = 48,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    thresholds = thresholds or THRESHOLDS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    bin_path = "./target/release/titan_text"
    if not Path(bin_path).exists():
        bin_path = "./target/debug/titan_text"

    print(f"=== Titan Text Halting Calibration Sweep ===")
    print(f"Binary: {bin_path}")
    print(f"Metric: {metric}, Patience: {patience}, Max Len: {max_len}")
    print(f"Thresholds: {thresholds}")
    print(f"Seeds: {SEEDS}, Prompts: {[p[1] for p in PROMPTS]}")
    print(f"Total runs: {len(thresholds) * len(PROMPTS) * len(SEEDS)}\n")

    results_by_thresh: Dict[str, List[Dict[str, Any]]] = {}

    for th in thresholds:
        th_key = f"{th:.2f}"
        results_by_thresh[th_key] = []
        print(f"--- Testing Threshold theta = {th:.2f} ---")

        for p_str, p_name in PROMPTS:
            for seed in SEEDS:
                cmd = [
                    bin_path,
                    "generate",
                    "--load-dir", checkpoint,
                    "--prompt", p_str,
                    "--max-len", str(max_len),
                    "--temperature", str(temperature),
                    "--halting", "adaptive",
                    "--halting-metric", metric,
                    "--halting-threshold", str(th),
                    "--halting-patience", str(patience),
                    "--seed", str(seed),
                ]
                sample = run_generate(cmd, scratch_dir)
                results_by_thresh[th_key].append(sample)
                print(f"  [{p_name:7s} | seed {seed:3d}] tau_mean: {sample['mean_tau']:5.2f} | edit_sim: {sample['metrics']['nearest_edit_similarity']:.4f} | sym: {sample['metrics']['horizontal_symmetry']:.4f}")

    # Analyze and aggregate
    summary_table = []
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Theta':>7s} | {'Mean Tau':>8s} | {'Tau SD':>7s} | {'Edit Sim (mean±se)':>18s} | {'H-Sym':>7s} | {'Eff Ratio':>9s} | {'Bound Tau':>9s} | {'Sym Tau':>7s}")
    print("-" * 80)

    analysis_data = {}

    for th in thresholds:
        th_key = f"{th:.2f}"
        samples = results_by_thresh[th_key]
        n = len(samples)

        taus = [s["mean_tau"] for s in samples]
        edit_sims = [s["metrics"]["nearest_edit_similarity"] for s in samples]
        h_syms = [s["metrics"]["horizontal_symmetry"] for s in samples]

        # Extract all token ticks across all samples
        all_token_ticks = []
        bound_ticks = []
        sym_ticks = []
        for s in samples:
            for t in s.get("token_traces", []):
                all_token_ticks.append(t["ticks_used"])
                if t.get("char_class") == "boundary":
                    bound_ticks.append(t["ticks_used"])
                elif t.get("char_class") == "symbol":
                    sym_ticks.append(t["ticks_used"])

        mean_tau = float(np.mean(taus))
        sd_tau = float(np.std(all_token_ticks)) if all_token_ticks else 0.0
        mean_sim = float(np.mean(edit_sims))
        se_sim = float(np.std(edit_sims) / np.sqrt(n)) if n > 1 else 0.0
        mean_hsym = float(np.mean(h_syms))
        eff_ratio = mean_sim / mean_tau if mean_tau > 0 else 0.0

        mean_bound = float(np.mean(bound_ticks)) if bound_ticks else 0.0
        mean_sym = float(np.mean(sym_ticks)) if sym_ticks else 0.0

        analysis_data[th_key] = {
            "threshold": th,
            "n_samples": n,
            "mean_tau": mean_tau,
            "token_tau_sd": sd_tau,
            "mean_edit_sim": mean_sim,
            "se_edit_sim": se_sim,
            "mean_h_symmetry": mean_hsym,
            "efficiency_ratio": eff_ratio,
            "mean_boundary_tau": mean_bound,
            "mean_symbol_tau": mean_sym,
        }

        print(f"{th:7.2f} | {mean_tau:8.2f} | {sd_tau:7.2f} | {mean_sim:8.4f} ± {se_sim:5.4f} | {mean_hsym:7.4f} | {eff_ratio:9.4f} | {mean_bound:9.2f} | {mean_sym:7.2f}")

    # Identify optimal threshold theta* (targeting sweet spot tau in [5, 7] and highest edit sim)
    sweet_candidates = [v for v in analysis_data.values() if 4.0 <= v["mean_tau"] <= 8.0]
    if sweet_candidates:
        best_candidate = max(sweet_candidates, key=lambda x: x["mean_edit_sim"])
    else:
        best_candidate = max(analysis_data.values(), key=lambda x: x["mean_edit_sim"])

    optimal_theta = best_candidate["threshold"]
    print("\n" + "=" * 80)
    print(f"CALIBRATED OPTIMAL THRESHOLD: theta* = {optimal_theta:.2f}")
    print(f"Operating Regime: Mean Tau = {best_candidate['mean_tau']:.2f}, Edit Sim = {best_candidate['mean_edit_sim']:.4f} ± {best_candidate['se_edit_sim']:.4f}")
    print("=" * 80)

    # Save artifacts
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "metric": metric,
        "patience": patience,
        "max_len": max_len,
        "temperature": temperature,
        "thresholds_tested": thresholds,
        "optimal_threshold": optimal_theta,
        "optimal_metrics": best_candidate,
        "analysis": analysis_data,
    }

    with open(out_path / "calibration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(out_path / "calibration_samples.json", "w", encoding="utf-8") as f:
        json.dump(results_by_thresh, f, indent=2)

    print(f"\nArtifacts saved to {output_dir}/")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Halting Threshold Calibration Sweep")
    parser.add_argument("--checkpoint", default="checkpoints/ascii_v1")
    parser.add_argument("--output-dir", default="reports/raw/halting_calibration")
    parser.add_argument("--metric", default="relative_delta")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thresholds", type=str, default="0.10,0.14,0.17,0.20,0.25,0.30,0.35")

    args = parser.parse_args()
    th_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    run_calibration(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        thresholds=th_list,
        patience=args.patience,
        metric=args.metric,
        max_len=args.max_len,
        temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
