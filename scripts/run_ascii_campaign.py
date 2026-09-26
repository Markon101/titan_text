#!/usr/bin/env python3
"""Titan Text ASCII Generation Research Campaign Runner.

Orchestrates:
1. Untrained baseline raw generation and causal sweep (seeds 42, 101, 202, 303, 404).
2. Procedural training on the ASCII curriculum.
3. Post-training generation across fixed prompt battery and latent tick sweep (tau in 0, 1, 2, 4, 8, 16).
4. Causal state-lesion ablation (--lesion-state).
5. Immutable raw artifact preservation (raw/, metadata/, manifest.jsonl, config.json, metrics.json).
6. Export to external Android shared storage (/sdcard/Download/TitanText/ascii_runs/<RUN_ID>).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_binary() -> str:
    release_bin = Path("target/release/titan_text")
    if release_bin.exists() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    debug_bin = Path("target/debug/titan_text")
    if debug_bin.exists() and os.access(debug_bin, os.X_OK):
        return str(debug_bin)
    raise FileNotFoundError("Could not find compiled titan_text binary in target/release or target/debug")


def run_sampling_step(
    bin_path: str,
    checkpoint: str | None,
    prompt: str,
    max_len: int = 64,
    tau: int = 4,
    temp: float = 0.7,
    top_k: int = 0,
    seed: int = 42,
    lesion_state: bool = False,
) -> dict:
    cmd = [
        bin_path,
        "generate",
        "--prompt",
        prompt,
        "--max-len",
        str(max_len),
        "--tau",
        str(tau),
        "--temperature",
        str(temp),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--format",
        "json",
    ]
    if checkpoint:
        cmd.extend(["--load-dir", checkpoint])
    if lesion_state:
        cmd.append("--lesion-state")

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stdout.strip()
    start_idx = out.find("{")
    end_idx = out.rfind("}")
    if start_idx != -1 and end_idx != -1:
        json_str = out[start_idx : end_idx + 1]
        return json.loads(json_str)
    return json.loads(out)


def record_sample(run_dir: Path, sample_idx: int, sample_data: dict, manifest_file) -> None:
    raw_dir = run_dir / "raw"
    meta_dir = run_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    fam_tag = sample_data["prompt"].replace("<", "").replace(">", "").strip()
    tau = sample_data["config"]["tau"]
    seed = sample_data["config"]["seed"]
    lesion = "_lesion" if sample_data["config"]["lesion_state"] else ""
    filename_base = f"sample_{sample_idx:03d}_{fam_tag}_seed{seed}_tau{tau}{lesion}"

    raw_path = raw_dir / f"{filename_base}.txt"
    raw_path.write_text(sample_data["full_text"])

    meta_path = meta_dir / f"{filename_base}.json"
    meta_path.write_text(json.dumps(sample_data, indent=2))

    manifest_record = {
        "sample_id": sample_idx,
        "filename": f"{filename_base}.txt",
        "prompt": sample_data["prompt"],
        "seed": seed,
        "tau": tau,
        "temperature": sample_data["config"]["temperature"],
        "top_k": sample_data["config"]["top_k"],
        "lesion_state": sample_data["config"]["lesion_state"],
        "steps_generated": sample_data["steps_generated"],
        "stop_reason": sample_data["stop_reason"],
        "metrics": sample_data["metrics"],
    }
    manifest_file.write(json.dumps(manifest_record) + "\n")
    manifest_file.flush()


def compute_aggregate_metrics(samples: list[dict]) -> dict:
    if not samples:
        return {}
    n = len(samples)
    metrics_list = [s["metrics"] for s in samples]

    def avg(key):
        return sum(m[key] for m in metrics_list) / n

    return {
        "sample_count": n,
        "mean_valid_char_ratio": avg("valid_char_ratio"),
        "mean_line_count": avg("line_count"),
        "mean_line_width": avg("mean_line_width"),
        "mean_non_whitespace_density": avg("non_whitespace_density"),
        "mean_row_diversity": avg("row_diversity"),
        "mean_col_diversity": avg("col_diversity"),
        "mean_repeated_char_collapse_rate": avg("repeated_char_collapse_rate"),
        "mean_repeated_line_collapse_rate": avg("repeated_line_collapse_rate"),
        "mean_horizontal_symmetry": avg("horizontal_symmetry"),
        "mean_vertical_symmetry": avg("vertical_symmetry"),
        "mean_bigram_entropy": avg("bigram_entropy"),
        "mean_nearest_edit_similarity": avg("nearest_edit_similarity"),
        "exact_training_match_count": sum(1 for m in metrics_list if m["exact_training_match"]),
    }


def run_campaign_battery(
    run_id: str,
    checkpoint: str | None,
    bin_path: str,
    description: str,
) -> Path:
    run_dir = Path("runs/ascii") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.jsonl"
    all_samples = []

    prompts = ["<BOX>\n", "<CHECKER>\n", "<DIAMOND>\n", "<MAZE>\n", "<BANNER>\n", "<FACE>\n", "<MOUNTAIN>\n", "<ABSTRACT>\n"]
    fixed_seeds = [42, 101, 202, 303, 404]
    tau_sweep = [0, 1, 2, 4, 8, 16]

    print(f"\n================================================================================")
    print(f"STARTING ASCII CAMPAIGN BATTERY: {run_id}")
    print(f"Checkpoint: {checkpoint or '[Untrained Random Weights]'}")
    print(f"================================================================================")

    sample_counter = 0
    with open(manifest_path, "w") as manifest_file:
        # 1. Fixed prompt & seed battery across tau levels
        for p in prompts:
            fam = p.replace("<", "").replace(">", "").strip()
            print(f"\n--- Category: {fam} ---")
            for seed in fixed_seeds:
                # Test intact model at tau = 4
                sample_data = run_sampling_step(
                    bin_path, checkpoint, prompt=p, tau=4, temp=0.7, seed=seed, lesion_state=False
                )
                sample_counter += 1
                record_sample(run_dir, sample_counter, sample_data, manifest_file)
                all_samples.append(sample_data)

            # For seed 42, run full tau sweep
            print(f"  Latent tau sweep for {fam} (seed 42)...")
            for tau in tau_sweep:
                sample_data = run_sampling_step(
                    bin_path, checkpoint, prompt=p, tau=tau, temp=0.7, seed=42, lesion_state=False
                )
                sample_counter += 1
                record_sample(run_dir, sample_counter, sample_data, manifest_file)
                all_samples.append(sample_data)

            # Causal ablation: Lesion state at tau = 4
            lesion_data = run_sampling_step(
                bin_path, checkpoint, prompt=p, tau=4, temp=0.7, seed=42, lesion_state=True
            )
            sample_counter += 1
            record_sample(run_dir, sample_counter, lesion_data, manifest_file)
            all_samples.append(lesion_data)

    # Save run configuration and aggregate metrics
    config_record = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "description": description,
        "timestamp_unix": int(time.time()),
        "prompts": prompts,
        "fixed_seeds": fixed_seeds,
        "tau_sweep": tau_sweep,
        "total_samples": sample_counter,
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2))

    agg_metrics = compute_aggregate_metrics(all_samples)
    (run_dir / "metrics.json").write_text(json.dumps(agg_metrics, indent=2))

    # Gallery creation: select clean and illustrative samples
    gallery_dir = run_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    seen_prompts = set()
    for s in all_samples:
        p = s["prompt"]
        if p not in seen_prompts and s["config"]["tau"] == 4 and not s["config"]["lesion_state"]:
            seen_prompts.add(p)
            fam = p.replace("<", "").replace(">", "").strip().lower()
            gallery_file = gallery_dir / f"gallery_{fam}.txt"
            gallery_file.write_text(
                f"# Titan Text ASCII Gallery Sample\n# Prompt: {p.strip()}\n# Seed: {s['config']['seed']} | Tau: {s['config']['tau']}\n# Nearest Edit Sim: {s['metrics']['nearest_edit_similarity']:.3f}\n\n{s['full_text']}\n"
            )

    # Write README.md
    readme_content = f"""# Titan Text ASCII Run: `{run_id}`

**Description**: {description}  
**Checkpoint**: `{checkpoint or "Untrained random weights"}`  
**Total Samples**: {sample_counter}  

## Aggregate Quantitative Metrics
- **Mean Valid Char Ratio**: {agg_metrics.get('mean_valid_char_ratio', 0):.4f}
- **Mean Lines Emitted**: {agg_metrics.get('mean_line_count', 0):.2f}
- **Mean Line Width**: {agg_metrics.get('mean_line_width', 0):.2f}
- **Mean Horizontal Symmetry**: {agg_metrics.get('mean_horizontal_symmetry', 0):.4f}
- **Mean Vertical Symmetry**: {agg_metrics.get('mean_vertical_symmetry', 0):.4f}
- **Mean Non-Whitespace Density**: {agg_metrics.get('mean_non_whitespace_density', 0):.4f}
- **Repeated Char Collapse Rate**: {agg_metrics.get('mean_repeated_char_collapse_rate', 0):.4f}
- **Repeated Line Collapse Rate**: {agg_metrics.get('mean_repeated_line_collapse_rate', 0):.4f}
- **Mean Nearest Edit Similarity**: {agg_metrics.get('mean_nearest_edit_similarity', 0):.4f}
- **Exact Training Match Count**: {agg_metrics.get('exact_training_match_count', 0)} / {sample_counter}

All raw samples are preserved in `raw/` with matching JSON metadata in `metadata/`.
"""
    (run_dir / "README.md").write_text(readme_content)

    print(f"\n✓ Completed Battery: {sample_counter} raw generations saved to '{run_dir}'")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run complete Titan Text ASCII generation campaign.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stage and run only sampling")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--export", action="store_true", default=True, help="Export to Android shared storage")

    args = parser.parse_args()
    bin_path = find_binary()
    print(f"Using Titan Text binary: {bin_path}")

    # 1. Untrained baseline run
    baseline_run_id = f"baseline_untrained_{int(time.time())}"
    baseline_dir = run_campaign_battery(
        run_id=baseline_run_id,
        checkpoint=None,
        bin_path=bin_path,
        description="Untrained random weights baseline across fixed seeds, prompt battery, and tau sweep.",
    )

    if args.export:
        subprocess.run(["python3", "scripts/export_ascii_run.py", str(baseline_dir)], check=False)

    # 2. Training stage
    checkpoint_dir = "checkpoints/ascii_v1"
    if not args.skip_train:
        print(f"\n================================================================================")
        print(f"TRAINING TITAN TEXT ASCII MODEL: {checkpoint_dir}")
        print(f"================================================================================")
        train_cmd = [
            bin_path,
            "train",
            "--task",
            "ascii",
            "--epochs",
            str(args.epochs),
            "--dev-steps",
            "4",
            "--seq-len",
            "48",
            "--batch-size",
            "8",
            "--lr",
            "0.003",
            "--save-dir",
            checkpoint_dir,
            "--causal-stencil",
            "--zero-boundary",
        ]
        train_res = subprocess.run(train_cmd, capture_output=True, text=True)
        print(train_res.stdout)
        if train_res.returncode != 0:
            print(f"[ERROR] Training failed:\n{train_res.stderr}")
            sys.exit(1)

    # 3. Post-training campaign battery
    trained_run_id = f"ascii_v1_trained_{int(time.time())}"
    trained_dir = run_campaign_battery(
        run_id=trained_run_id,
        checkpoint=checkpoint_dir,
        bin_path=bin_path,
        description=f"Trained Titan Text ASCII v1 model ({args.epochs} epochs, T=4, C=64) evaluated on fixed battery.",
    )

    if args.export:
        subprocess.run(["python3", "scripts/export_ascii_run.py", str(trained_dir)], check=False)

    print(f"\n================================================================================")
    print(f"CAMPAIGN EXECUTION COMPLETE")
    print(f"  Baseline Run : {baseline_dir}")
    print(f"  Trained Run  : {trained_dir}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
