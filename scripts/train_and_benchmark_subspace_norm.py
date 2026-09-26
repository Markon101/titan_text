#!/usr/bin/env python3
"""Adversarial Subspace Normalization & OCPD Validation Campaign.

Evaluates the Orthogonal Contractive-Pushdown Decomposition (OCPD) hypothesis
against the Skeptical Reviewer's deflationary saturation-as-gating null hypothesis:
  H_OCPD: Restricting bounded normalization to the continuous subspace H (channels 0..32)
          contracts continuous drift (preventing loss explosion) while preserving
          unattenuated discrete pushdown stack dynamics in carry channels 32..64.
  H_Null: Normalization harms accuracy simply by removing activation saturation,
          and any recovery is a scale/temperature artifact rather than a subspace property.

Four Experimental Arms (Trained for 35 epochs, B=16, dev_steps=20, LR=0.003, seeds 42, 43, 44):
  - Arm A: No Norm (state_norm="none") -> Unbounded continuous field drift
  - Arm B: Subspace H-Only Norm (state_norm="bounded_h_only") -> OCPD contractive decoupling
  - Arm C: Full-64 Norm (state_norm="bounded", carry_quant="none") -> All channels squashed
  - Arm D: Full-64 Norm + STE-Sign (state_norm="bounded", carry_quant="ste_sign") -> Squashed then re-quantized

Evaluated on Dyck-4 across:
  - L=16 (In-Distribution, T=20)
  - L=64 (Extrapolation, T=32)
  - L=128 (Extreme OOD, T=64)

Safety: RAYON_NUM_THREADS=1 strictly enforced.
Output: reports/subspace_normalization_ocpd_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BIN = "./target/release/titan_text"
SEEDS = [42, 43, 44]
BATCH_SIZE = "16"
EPOCHS = "35"
DEV_STEPS = "20"
LR = "0.003"
REPORT_FILE = "reports/subspace_normalization_ocpd_results.json"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Execute command with strict single-thread environment."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


def parse_train_output(out: str) -> dict[str, float]:
    """Parse final train and val metrics from training output."""
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


def parse_benchmark_output(out: str) -> dict[str, float]:
    """Extract Titan NCA loss and accuracy metrics from benchmark CLI output."""
    res = {}
    m = re.search(
        r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+)\s*±\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
        out,
    )
    if m:
        res["loss_mean"] = float(m.group(1))
        res["loss_std"] = float(m.group(2))
        res["acc_mean"] = float(m.group(3))
        res["acc_std"] = float(m.group(4))
        return res

    m_simple = re.search(r"Titan NCA\s*:\s*Loss\s*=\s*([0-9.]+),\s*Acc\s*=\s*([0-9.]+)%", out)
    if m_simple:
        res["loss_mean"] = float(m_simple.group(1))
        res["loss_std"] = 0.0
        res["acc_mean"] = float(m_simple.group(2))
        res["acc_std"] = 0.0
        return res
    return res


def train_arm(arm_name: str, state_norm: str, carry_quant: str, seed: int) -> str:
    """Train a specific normalization arm."""
    save_dir = f"checkpoints/subspace_norm/{arm_name}_seed_{seed}"
    manifest_path = Path(save_dir) / "manifest.json"
    if manifest_path.exists():
        print(f"    Reusing existing checkpoint at {save_dir}")
        return save_dir

    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        BIN, "train",
        "--task", "dyck-pushdown",
        "--epochs", EPOCHS,
        "--dev-steps", DEV_STEPS,
        "--seq-len", "16",
        "--batch-size", BATCH_SIZE,
        "--lr", LR,
        "--seed", str(seed),
        "--save-dir", save_dir,
        "--causal-stencil",
        "--zero-boundary",
        "--carry-channels", "32",
        "--carry-bidirectional",
        "--carry-skip-stride", "4",
        "--carry-quantization", carry_quant,
        "--tail-eq-weight", "0.1",
        "--tail-eq-ticks", "4",
        "--state-norm", state_norm,
    ]

    t0 = time.time()
    code, out = run_cmd(cmd)
    dur = time.time() - t0
    if code != 0:
        print(f" FAILED in {dur:.1f}s")
        print(out[-800:])
        raise RuntimeError(f"Training failed for {arm_name} seed {seed}")

    metrics = parse_train_output(out)
    acc = metrics.get("val_acc", 0.0)
    loss = metrics.get("val_loss", 99.0)
    print(f"    Trained ({dur:.1f}s) | Val Acc: {acc:.1f}%, Val Loss: {loss:.3f}")
    return save_dir


def evaluate_arm(arm_name: str, state_norm: str, carry_quant: str, cp_dirs: list[str], seq_len: int, dev_steps: int) -> dict:
    """Benchmark arm across multiple seeds."""
    seeds_arg = ",".join(str(s) for s in SEEDS)
    cp_arg = cp_dirs[0]  # First checkpoint for config loading
    cmd = [
        BIN, "benchmark",
        "--task", "dyck-pushdown",
        "--seq-len", str(seq_len),
        "--batch-size", BATCH_SIZE,
        "--dev-steps", str(dev_steps),
        "--seeds", seeds_arg,
        "--causal-stencil",
        "--zero-boundary",
        "--carry-channels", "32",
        "--carry-bidirectional",
        "--carry-skip-stride", "4",
        "--carry-quantization", carry_quant,
        "--state-norm", state_norm,
        "--checkpoint", cp_arg,
    ]

    code, out = run_cmd(cmd)
    if code != 0:
        print(f"Benchmark error for {arm_name} (L={seq_len}, T={dev_steps}):\n{out[-600:]}")
        return {"acc_mean": 0.0, "acc_std": 0.0, "loss_mean": 99.0, "loss_std": 0.0}

    return parse_benchmark_output(out)


def main():
    print("=" * 80)
    print("TITAN TEXT: ADVERSARIAL SUBSPACE NORMALIZATION & OCPD VALIDATION")
    print("Four-Arm Decisive Falsification Campaign across Seeds {42, 43, 44}")
    print("=" * 80)

    arms = [
        {
            "name": "arm_a_no_norm",
            "desc": "Arm A: No Norm (Unbounded Continuous Drift)",
            "state_norm": "none",
            "carry_quant": "ste_sign",
            "existing_prefix": "checkpoints/dyck4_pushdown/cd_dv_nca_seed_",
        },
        {
            "name": "arm_b_subspace_h_only",
            "desc": "Arm B: Subspace H-Only Norm (OCPD Clean Decoupling)",
            "state_norm": "bounded_h_only",
            "carry_quant": "ste_sign",
            "existing_prefix": None,
        },
        {
            "name": "arm_c_full64_no_quant",
            "desc": "Arm C: Full-64 Norm (All Channels Squashed, No STE)",
            "state_norm": "bounded",
            "carry_quant": "none",
            "existing_prefix": None,
        },
        {
            "name": "arm_d_full64_ste_sign",
            "desc": "Arm D: Full-64 Norm + STE-Sign (Squashed then Re-quantized)",
            "state_norm": "bounded",
            "carry_quant": "ste_sign",
            "existing_prefix": None,
        },
    ]

    results = {
        "benchmark": "subspace_normalization_ocpd",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEEDS,
        "arms": {},
    }

    for arm in arms:
        arm_name = arm["name"]
        print(f"\n[{arm['desc']}]")
        cp_dirs = []
        for seed in SEEDS:
            if arm["existing_prefix"]:
                cp_dir = f"{arm['existing_prefix']}{seed}"
            else:
                cp_dir = train_arm(arm_name, arm["state_norm"], arm["carry_quant"], seed)
            cp_dirs.append(cp_dir)

        # Evaluations
        print(f"  Evaluating {arm_name} across L in {{16, 64, 128}}...")
        eval_16 = evaluate_arm(arm_name, arm["state_norm"], arm["carry_quant"], cp_dirs, seq_len=16, dev_steps=20)
        print(f"    L=16,  T=20: Acc = {eval_16.get('acc_mean', 0):5.2f}% ± {eval_16.get('acc_std', 0):4.2f}% | Loss = {eval_16.get('loss_mean', 99):6.4f}")

        eval_64 = evaluate_arm(arm_name, arm["state_norm"], arm["carry_quant"], cp_dirs, seq_len=64, dev_steps=32)
        print(f"    L=64,  T=32: Acc = {eval_64.get('acc_mean', 0):5.2f}% ± {eval_64.get('acc_std', 0):4.2f}% | Loss = {eval_64.get('loss_mean', 99):6.4f}")

        eval_128 = evaluate_arm(arm_name, arm["state_norm"], arm["carry_quant"], cp_dirs, seq_len=128, dev_steps=64)
        print(f"    L=128, T=64: Acc = {eval_128.get('acc_mean', 0):5.2f}% ± {eval_128.get('acc_std', 0):4.2f}% | Loss = {eval_128.get('loss_mean', 99):6.4f}")

        results["arms"][arm_name] = {
            "description": arm["desc"],
            "state_norm": arm["state_norm"],
            "carry_quantization": arm["carry_quant"],
            "eval_l16_t20": eval_16,
            "eval_l64_t32": eval_64,
            "eval_l128_t64": eval_128,
        }

    os.makedirs(Path(REPORT_FILE).parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Saved Subspace Normalization OCPD Results to {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
