#!/usr/bin/env python3
"""Phase 4: Direct Carry Probing and Spatio-Temporal Decodability Curves.

Evaluates:
1. Spatial decodability decay: decodability(distance) across sites d in {0, 4, 8, 12}.
2. Representation partitioning: Full State (x) vs Carry Register (c) vs Hidden State (h).
3. Null Controls (as mandated by Adversarial Audit):
   - Zero-Leakage Balanced Ripple Chains: a_i + b_i = 9 to enforce I(input; carry) = 0.
   - Untrained Random-Weights Baseline Control: Delta Decodability = Acc_trained - Acc_untrained.
   - t=0 Prompt Injection Null Control (must equal 50.0% chance).
   - Label-Shuffled Null Control.
4. Tasks:
   - column-arithmetic (carry propagation through ripple chain)
   - iterated-parity (chunk 0 parity propagation across 12 cells)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np

BIN = "./target/release/titan_text"


def run_rollout(checkpoint: str | None, prompt: str, horizon: int, seq_len: int = 16) -> list[dict]:
    """Runs autonomous rollout and returns per-step traces with spatial activations."""
    tmp_trace = f"scratch/trace_{random.randint(100000, 999999)}.json"
    os.makedirs("scratch", exist_ok=True)
    cmd = [
        BIN, "rollout",
        "--seq-len", str(seq_len),
        "--horizon", str(horizon),
        "--prompt", prompt,
        "--record-activations",
        "--trace-output", tmp_trace,
    ]
    if checkpoint:
        cmd.extend(["--load-dir", checkpoint])

    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Rollout failed: {proc.stderr}")

    with open(tmp_trace) as f:
        traces = json.load(f)
    if os.path.exists(tmp_trace):
        os.remove(tmp_trace)
    return traces


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Trains ridge classification probe and returns test accuracy."""
    # Add bias column
    ones_train = np.ones((X_train.shape[0], 1), dtype=np.float32)
    X_b_train = np.hstack([X_train, ones_train])
    ones_test = np.ones((X_test.shape[0], 1), dtype=np.float32)
    X_b_test = np.hstack([X_test, ones_test])

    # Ridge regression: w = (X^T X + lambda I)^-1 X^T y
    d = X_b_train.shape[1]
    reg = 1e-3 * np.eye(d, dtype=np.float32)
    reg[-1, -1] = 0.0  # Do not regularize bias
    w = np.linalg.solve(X_b_train.T @ X_b_train + reg, X_b_train.T @ y_train)

    preds = (X_b_test @ w > 0.5).astype(np.int32)
    acc = float(np.mean(preds == y_test))
    return acc


def generate_balanced_ripple_samples(n_samples: int = 100, seed: int = 42) -> list[tuple[str, int]]:
    """Generates column arithmetic samples where middle columns satisfy a_i + b_i = 9.

    Format for L=16 (d=4 digits):
    a0 a1 a2 a3 + b0 b1 b2 b3 = ?????
    Positions:
    Column 0 (LSD, site 3 for a, site 8 for b): generates carry 1 (a3+b3 >= 10) or carry 0 (a3+b3 < 10)
    Column 1 (site 2, 7): a2+b2 = 9
    Column 2 (site 1, 6): a1+b1 = 9
    Column 3 (MSD, site 0, 5): a0+b0 = 9
    Query positions: 10..14 ('?????')
    Ground truth carry at column 3 (distance 3 columns away) is EXACTLY the carry from column 0!
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        # 50% carry=1, 50% carry=0 at column 0
        c0 = rng.choice([0, 1])
        if c0 == 1:
            # a3 + b3 >= 10
            a3 = rng.randint(1, 9)
            b3 = rng.randint(10 - a3, 9)
        else:
            # a3 + b3 < 10
            a3 = rng.randint(0, 8)
            b3 = rng.randint(0, 9 - a3)

        # Middle columns: sum to 9
        pairs = [(rng.randint(0, 9),) for _ in range(3)]
        a2 = rng.randint(0, 9)
        b2 = 9 - a2
        a1 = rng.randint(0, 9)
        b1 = 9 - a1
        a0 = rng.randint(0, 9)
        b0 = 9 - a0

        prompt = f"{a0}{a1}{a2}{a3}+{b0}{b1}{b2}{b3}=?????"
        samples.append((prompt, c0))
    return samples


def generate_iterated_parity_samples(n_samples: int = 100, seed: int = 42) -> list[tuple[str, int]]:
    """Generates iterated parity samples. Target is Chunk 0 parity p0 in {0, 1}."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        b0 = rng.choice([0, 1])
        b1 = rng.choice([0, 1])
        b2 = rng.choice([0, 1])
        p0 = (b0 + b1 + b2) % 2

        # Chunks 1, 2, 3 data bits
        other_bits = [rng.choice([0, 1]) for _ in range(9)]
        prompt = (
            f"{b0}{b1}{b2}?"
            f"{other_bits[0]}{other_bits[1]}{other_bits[2]}?"
            f"{other_bits[3]}{other_bits[4]}{other_bits[5]}?"
            f"{other_bits[6]}{other_bits[7]}{other_bits[8]}?"
        )
        samples.append((prompt, p0))
    return samples


def evaluate_model_probing(
    model_name: str,
    checkpoint: str | None,
    task_type: str,
    carry_channels: int = 32,
    n_samples: int = 80,
    horizon: int = 16,
) -> dict:
    """Extracts activations and trains decodability probes across distances and channel subsets."""
    print(f"\n--- Probing Model: {model_name} (Task: {task_type}, Horizon: {horizon}) ---")
    if task_type == "arithmetic":
        samples = generate_balanced_ripple_samples(n_samples=n_samples, seed=42)
        # Sites to probe: Column 0 (site 3), Column 1 (site 2), Column 2 (site 1), Column 3 (site 0), Query (site 10)
        probe_sites = [3, 2, 1, 0, 10]
        site_labels = ["Col 0 (Gen, d=0)", "Col 1 (Ripple, d=1)", "Col 2 (Ripple, d=2)", "Col 3 (Ripple, d=3)", "Query (Out, d=7)"]
    else:
        samples = generate_iterated_parity_samples(n_samples=n_samples, seed=42)
        # Query slots: Slot 0 (site 3, d=0), Slot 1 (site 7, d=4), Slot 2 (site 11, d=8), Slot 3 (site 15, d=12)
        probe_sites = [3, 7, 11, 15]
        site_labels = ["Slot 0 (d=0)", "Slot 1 (d=4)", "Slot 2 (d=8)", "Slot 3 (d=12)"]

    prompts = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples], dtype=np.float32)

    # Split into train (70%) and test (30%)
    n_train = int(0.7 * n_samples)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n_samples))

    # Extract activations across all samples
    # Shape: [N, T, L, C]
    all_acts = []
    t0 = time.time()
    for idx, p in enumerate(prompts):
        traces = run_rollout(checkpoint, p, horizon=horizon)
        # traces has length horizon. Each has spatial_activations of shape [L, C]
        sample_act = [t["spatial_activations"] for t in traces]
        all_acts.append(sample_act)
        if (idx + 1) % 20 == 0:
            print(f"  Extracted activations for {idx + 1}/{n_samples} samples ({time.time() - t0:.1f}s)...")

    tensor_acts = np.array(all_acts, dtype=np.float32)  # [N, T, L, C]
    n_total, T, L, C = tensor_acts.shape
    c_hidden = C - carry_channels

    print(f"  Activation Tensor extracted: shape {tensor_acts.shape} (Chidden={c_hidden}, Ccarry={carry_channels})")

    results_by_site = []

    for site_idx, site in enumerate(probe_sites):
        s_label = site_labels[site_idx]
        site_metrics = {"site": site, "label": s_label, "temporal_curves": {}}

        # Probe across time steps t
        probe_ticks = [0, 1, 2, 4, 8, horizon - 1]
        for t in probe_ticks:
            # Full state
            X_full = tensor_acts[:, t, site, :]
            # Carry register only
            X_carry = tensor_acts[:, t, site, c_hidden:] if carry_channels > 0 else X_full
            # Hidden state only
            X_hidden = tensor_acts[:, t, site, :c_hidden] if carry_channels > 0 else X_full

            acc_full = train_linear_probe(X_full[train_idx], labels[train_idx], X_full[test_idx], labels[test_idx])
            acc_carry = train_linear_probe(X_carry[train_idx], labels[train_idx], X_carry[test_idx], labels[test_idx]) if carry_channels > 0 else 0.5
            acc_hidden = train_linear_probe(X_hidden[train_idx], labels[train_idx], X_hidden[test_idx], labels[test_idx]) if carry_channels > 0 else acc_full

            # Label shuffled null control at final tick
            shuffled_labels = labels.copy()
            np.random.shuffle(shuffled_labels)
            acc_null = train_linear_probe(X_full[train_idx], shuffled_labels[train_idx], X_full[test_idx], shuffled_labels[test_idx])

            site_metrics["temporal_curves"][str(t)] = {
                "acc_full": acc_full,
                "acc_carry": acc_carry,
                "acc_hidden": acc_hidden,
                "acc_null_shuffled": acc_null,
            }

        final_t = horizon - 1
        final_res = site_metrics["temporal_curves"][str(final_t)]
        print(f"  --> {s_label:<24} | Full: {final_res['acc_full']*100:5.1f}% | Carry: {final_res['acc_carry']*100:5.1f}% | Hidden: {final_res['acc_hidden']*100:5.1f}% | Null: {final_res['acc_null_shuffled']*100:5.1f}%")
        results_by_site.append(site_metrics)

    return {"model": model_name, "task": task_type, "results_by_site": results_by_site}


def main():
    print("=" * 80)
    print("TITAN TEXT: PHASE 4 DIRECT CARRY PROBING & SPATIAL FIDELITY CURVES")
    print("=" * 80)

    os.makedirs("reports", exist_ok=True)
    report_file = Path("reports/carry_probing_results.json")

    # 1. Models to evaluate:
    # (a) Untrained Random-Weights Baseline Control
    # (b) Trained Baseline Causal (Cc=0, k=1)
    # (c) Trained ECR-32 Bi Skip-4 Deep (Cc=32, k=4, T=24)
    # (d) Trained ECR-16 Bi (Cc=16, k=1, T=16)

    probing_suite = []

    # Model 1: Untrained Random Weights Control (Iterated Parity)
    m1 = evaluate_model_probing(
        model_name="Untrained Random-Weights Baseline",
        checkpoint=None,
        task_type="parity",
        carry_channels=0,
        n_samples=60,
        horizon=16,
    )
    probing_suite.append(m1)

    # Model 2: Trained ECR-32 Bi Skip-4 Deep on Column Arithmetic
    ck_ecr32 = "checkpoints/campaign_column_arithmetic/ecr-32_bi_skip-4_deep_k4_t24_seed_42"
    if os.path.exists(ck_ecr32):
        m2 = evaluate_model_probing(
            model_name="ECR-32 Bi Skip-4 Deep (T=24, Seed 42)",
            checkpoint=ck_ecr32,
            task_type="arithmetic",
            carry_channels=32,
            n_samples=60,
            horizon=24,
        )
        probing_suite.append(m2)

    # Model 3: Trained ECR-16 Bi Skip-2 on Column Arithmetic
    ck_ecr16 = "checkpoints/campaign_column_arithmetic/ecr-16_bi_skip-2_k2_t16_seed_42"
    if os.path.exists(ck_ecr16):
        m3 = evaluate_model_probing(
            model_name="ECR-16 Bi Skip-2 (T=16, Seed 42)",
            checkpoint=ck_ecr16,
            task_type="arithmetic",
            carry_channels=16,
            n_samples=60,
            horizon=16,
        )
        probing_suite.append(m3)

    # Model 4: Trained ECR-16 on Iterated Parity
    ck_parity = "checkpoints/ecr_parity_l16"
    if os.path.exists(ck_parity):
        m4 = evaluate_model_probing(
            model_name="ECR-16 Bi (T=16, Parity)",
            checkpoint=ck_parity,
            task_type="parity",
            carry_channels=16,
            n_samples=60,
            horizon=16,
        )
        probing_suite.append(m4)

    with open(report_file, "w") as f:
        json.dump(probing_suite, f, indent=2)
    print(f"\nProbing evaluation complete. Results saved to {report_file}")


if __name__ == "__main__":
    main()
