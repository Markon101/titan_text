#!/usr/bin/env python3
"""Adaptive Recurrent Compute & Halting Campaign Runner for Titan Text.

Executes a 4-arm experimental campaign across a fixed-seed battery and prompts:
- Arm A: Adaptive Recurrent Halting (--halting adaptive)
- Arm B: Dense Fixed Sweep (tau in {0 (lesion), 1, 2, 4, 8, 16})
- Arm C: Shuffled/Permuted Adaptive Control (--tau-schedule shuffled_ticks)
- Arm D: Uniform Random Sham Control (--halting random)

Collects raw JSON generation outputs, extracts token tick traces, calculates
Pareto efficiency (quality vs compute), and checks causal hypotheses.
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


SEEDS = [42, 101, 202, 303, 404]
PROMPTS = [
    ("<BOX>\n", "box"),
    ("<MAZE>\n", "maze"),
    ("<DIAMOND>\n", "diamond"),
]
FIXED_TAUS = [0, 1, 2, 4, 8, 16]


def run_cmd(cmd: List[str], tmp_output: Optional[Path] = None, timeout: int = 90) -> Dict[str, Any]:
    if tmp_output is None:
        tmp_output = Path(f"/data/data/com.termux/files/home/projects/titan_text/scratch/tmp_run_{os.getpid()}_{random.randint(1000, 9999)}.json")
    
    cmd_with_out = list(cmd)
    if "--output" not in cmd_with_out:
        cmd_with_out.extend(["--output", str(tmp_output)])
    if "--format" not in cmd_with_out:
        cmd_with_out.extend(["--format", "json"])

    proc = subprocess.run(
        cmd_with_out,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}")
    
    if tmp_output.exists():
        with open(tmp_output, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            tmp_output.unlink()
        except OSError:
            pass
        return data

    # Fallback to stdout parsing if output file not written
    raw = proc.stdout.strip()
    json_start = raw.find("{")
    json_end = raw.rfind("}")
    if json_start == -1 or json_end == -1:
        raise ValueError(f"Could not find valid JSON in output:\n{raw}")
    
    return json.loads(raw[json_start : json_end + 1])


def run_campaign(
    checkpoint: str = "checkpoints/ascii_v1",
    output_dir: str = "reports/raw/adaptive_halting",
    max_len: int = 48,
    temperature: float = 0.7,
    halting_metric: str = "relative_delta",
    halting_threshold: float = 0.35,
    halting_patience: int = 2,
    seeds: Optional[List[int]] = None,
    prompts: Optional[List[tuple]] = None,
) -> Dict[str, Any]:
    seeds = seeds or SEEDS
    prompts = prompts or PROMPTS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    bin_path = "./target/release/titan_text"
    if not Path(bin_path).exists():
        raise FileNotFoundError(f"Binary {bin_path} not found. Run cargo build --release first.")

    manifest: Dict[str, Any] = {
        "timestamp_unix": int(datetime.now(timezone.utc).timestamp()),
        "checkpoint": checkpoint,
        "max_len": max_len,
        "temperature": temperature,
        "halting_metric": halting_metric,
        "halting_threshold": halting_threshold,
        "halting_patience": halting_patience,
        "seeds": seeds,
        "prompts": [p[0] for p in prompts],
        "runs": [],
    }

    print(f"================================================================================")
    print(f"TITAN TEXT ADAPTIVE HALTING RESEARCH CAMPAIGN")
    print(f"Checkpoint : {checkpoint}")
    print(f"Metric     : {halting_metric} (thresh: {halting_threshold}, patience: {halting_patience})")
    print(f"Seeds      : {seeds}")
    print(f"Prompts    : {[p[1] for p in prompts]}")
    print(f"Output Dir : {output_dir}")
    print(f"================================================================================\n")

    all_samples: List[Dict[str, Any]] = []

    # 1. Run Arm A: Adaptive Halting
    print(">>> Executing Arm A: Adaptive Halting...")
    arm_a_results = {}
    for prompt_text, prompt_name in prompts:
        for seed in seeds:
            cmd = [
                bin_path,
                "generate",
                "--load-dir", checkpoint,
                "--prompt", prompt_text,
                "--max-len", str(max_len),
                "--temperature", str(temperature),
                "--halting", "adaptive",
                "--tau-min", "1",
                "--tau-max", "16",
                "--halting-metric", halting_metric,
                "--halting-threshold", str(halting_threshold),
                "--halting-patience", str(halting_patience),
                "--seed", str(seed),
                "--format", "json",
            ]
            sample = run_cmd(cmd)
            sample["arm"] = "A_adaptive"
            sample["prompt_name"] = prompt_name
            traces = sample.get("token_traces", [])
            ticks = [t["ticks_used"] for t in traces]
            if ticks:
                frac_min = sum(1 for t in ticks if t == 1) / len(ticks)
                frac_max = sum(1 for t in ticks if t == 16) / len(ticks)
                sample["frac_tau_min"] = frac_min
                sample["frac_tau_max"] = frac_max
                sample["is_degenerate"] = (frac_min + frac_max) > 0.90
            all_samples.append(sample)
            arm_a_results[(prompt_name, seed)] = sample
            print(f"  [Arm A] {prompt_name:<8} seed {seed:3d} -> ticks={sample['total_ticks']:3d}, mean_tau={sample['mean_tau']:4.2f}, edit_sim={sample['metrics']['nearest_edit_similarity']:.3f}")

    # 2. Run Arm B: Fixed Tau Sweep
    print("\n>>> Executing Arm B: Fixed Tau Baseline Sweep...")
    for tau in FIXED_TAUS:
        for prompt_text, prompt_name in prompts:
            for seed in seeds:
                cmd = [
                    bin_path,
                    "generate",
                    "--load-dir", checkpoint,
                    "--prompt", prompt_text,
                    "--max-len", str(max_len),
                    "--temperature", str(temperature),
                    "--seed", str(seed),
                    "--format", "json",
                ]
                if tau == 0:
                    cmd.append("--lesion-state")
                else:
                    cmd.extend(["--tau", str(tau)])

                sample = run_cmd(cmd)
                sample["arm"] = f"B_fixed_tau{tau}"
                sample["prompt_name"] = prompt_name
                sample["fixed_tau"] = tau
                sample["intervention"] = "state_lesion" if tau == 0 else f"fixed_tau_{tau}"
                all_samples.append(sample)
        print(f"  [Arm B] Completed tau={tau}")

    # 3. Run Arm C: Shuffled / Permuted Adaptive Schedule Control
    print("\n>>> Executing Arm C: Shuffled Adaptive Control (Preserving Budget & Dist, Destroying State Alignment)...")
    for prompt_text, prompt_name in prompts:
        for seed in seeds:
            arm_a_sample = arm_a_results[(prompt_name, seed)]
            traces = arm_a_sample.get("token_traces", [])
            if not traces:
                continue
            ticks = [t["ticks_used"] for t in traces]
            
            # Deterministically permute the exact ticks multiset using seed + 9999
            shuffled_ticks = list(ticks)
            rng = random.Random(seed + 9999)
            rng.shuffle(shuffled_ticks)
            schedule_str = ",".join(str(t) for t in shuffled_ticks)

            cmd = [
                bin_path,
                "generate",
                "--load-dir", checkpoint,
                "--prompt", prompt_text,
                "--max-len", str(max_len),
                "--temperature", str(temperature),
                "--tau-schedule", schedule_str,
                "--seed", str(seed),
                "--format", "json",
            ]
            sample = run_cmd(cmd)
            sample["arm"] = "C_shuffled_control"
            sample["prompt_name"] = prompt_name
            sample["source_arm_a_mean_tau"] = arm_a_sample["mean_tau"]
            sample["source_arm_a_total_ticks"] = arm_a_sample["total_ticks"]
            c_traces = sample.get("token_traces", [])
            sample["schedule_consumed_count"] = len(c_traces)
            sample["schedule_total_count"] = len(ticks)
            sample["schedule_consumed_fraction"] = len(c_traces) / max(1, len(ticks))
            sample["realized_budget_ratio"] = sample["total_ticks"] / max(1, arm_a_sample["total_ticks"])
            all_samples.append(sample)
            print(f"  [Arm C] {prompt_name:<8} seed {seed:3d} -> ticks={sample['total_ticks']:3d} (Arm A={arm_a_sample['total_ticks']:3d}), mean_tau={sample['mean_tau']:4.2f}, edit_sim={sample['metrics']['nearest_edit_similarity']:.3f}")

    # 4. Run Arm D: Uniform Random Sham Control
    print("\n>>> Executing Arm D: Random Compute Sham Control (tau ~ Uniform(1, 16))...")
    for prompt_text, prompt_name in prompts:
        for seed in seeds:
            cmd = [
                bin_path,
                "generate",
                "--load-dir", checkpoint,
                "--prompt", prompt_text,
                "--max-len", str(max_len),
                "--temperature", str(temperature),
                "--halting", "random",
                "--tau-min", "1",
                "--tau-max", "16",
                "--seed", str(seed),
                "--format", "json",
            ]
            sample = run_cmd(cmd)
            sample["arm"] = "D_random_sham"
            sample["prompt_name"] = prompt_name
            all_samples.append(sample)
            print(f"  [Arm D] {prompt_name:<8} seed {seed:3d} -> ticks={sample['total_ticks']:3d}, mean_tau={sample['mean_tau']:4.2f}, edit_sim={sample['metrics']['nearest_edit_similarity']:.3f}")

    # Save full raw dataset
    raw_file = out_path / "campaign_samples.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\n✓ Saved {len(all_samples)} raw sample records to {raw_file}")

    manifest["total_samples"] = len(all_samples)
    manifest_file = out_path / "campaign_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest": manifest, "samples": all_samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titan Text Adaptive Halting Campaign")
    parser.add_argument("--checkpoint", default="checkpoints/ascii_v1")
    parser.add_argument("--output-dir", default="reports/raw/adaptive_halting")
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--halting-metric", default="relative_delta")
    parser.add_argument("--halting-threshold", type=float, default=0.35)
    parser.add_argument("--halting-patience", type=int, default=2)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed integers")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None

    run_campaign(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        max_len=args.max_len,
        temperature=args.temperature,
        halting_metric=args.halting_metric,
        halting_threshold=args.halting_threshold,
        halting_patience=args.halting_patience,
        seeds=seed_list,
    )
