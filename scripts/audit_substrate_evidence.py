#!/usr/bin/env python3
"""Independent, dependency-free audit of historical IPPR substrate evidence.

Exhaustive unique-row scores are diagnostics of the existing split, not a new
dataset. The local lookup and majority baselines are fitted on train rows only.
Historical confidence calculations concern evaluation batches of one model,
not independent training replicates or independent unique observations.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1


def observation_hash(text):
    value = 0xCBF29CE484222325
    for character in text:
        value = ((value ^ ord(character)) * 0x100000001B3) & MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def rows(length):
    assert length % 4 == 0
    result = {"train": [], "validation": []}
    for bits in itertools.product((0, 1), repeat=3 * (length // 4)):
        text = "".join("".join(map(str, bits[i:i + 3])) + "?"
                       for i in range(0, len(bits), 3))
        labels = tuple(sum(bits[:3 * (slot + 1)]) % 2
                       for slot in range(length // 4))
        split = "validation" if observation_hash(text) % 5 == 0 else "train"
        result[split].append((text, bits, labels))
    return result


def majority(labels):
    # Explicit deterministic tie rule, including unseen local contexts.
    return int(2 * sum(labels) > len(labels))


def metrics(dataset, predictions):
    count = len(dataset)
    slots = len(dataset[0][2])
    correct = [sum(pred[slot] == row[2][slot]
                   for row, pred in zip(dataset, predictions)) for slot in range(slots)]
    complete = sum(tuple(pred) == row[2] for row, pred in zip(dataset, predictions))
    return {"unique_rows": count, "correct_by_slot": correct,
            "accuracy_by_slot": [c / count for c in correct],
            "complete_correct": complete, "complete_accuracy": complete / count}


def audit_split(length):
    data = rows(length)
    train, validation = data["train"], data["validation"]
    slots = length // 4
    priors = [majority([row[2][slot] for row in train]) for slot in range(slots)]
    result = {"length": length, "train_unique": len(train),
              "validation_unique": len(validation), "overlap_unique": 0,
              "train_only_majority_labels": priors,
              "train_only_majority": metrics(validation, [priors] * len(validation)),
              "train_only_local_suffix_lookup": {}}
    for width in (1, 2, 3, 6):
        tables = []
        for slot in range(slots):
            end = 3 * (slot + 1)
            table = {}
            for _, bits, labels in train:
                table.setdefault(bits[max(0, end - width):end], []).append(labels[slot])
            tables.append({key: majority(values) for key, values in table.items()})
        predictions = []
        for _, bits, _ in validation:
            predictions.append([tables[slot].get(bits[max(0, 3 * (slot + 1) - width):3 * (slot + 1)],
                                                       priors[slot]) for slot in range(slots)])
        result["train_only_local_suffix_lookup"][str(width)] = metrics(validation, predictions)
    result["fixed_last_two_parity"] = metrics(validation, [
        [sum(bits[3 * slot + 1:3 * slot + 3]) % 2 for slot in range(slots)]
        for _, bits, _ in validation])
    result["fixed_last_two_parity_complement"] = metrics(validation, [
        [1 - sum(bits[3 * slot + 1:3 * slot + 3]) % 2 for slot in range(slots)]
        for _, bits, _ in validation])
    # Pairs remaining wholly within validation permit a source-bit intervention
    # without inadvertently evaluating a training observation.
    by_text = {text: labels for text, _, labels in validation}
    pairs = []
    for text, _, labels in validation:
        if text[0] == "0" and "1" + text[1:] in by_text:
            assert all(a != b for a, b in zip(labels, by_text["1" + text[1:]]))
            pairs.append(text)
    result["heldout_first_bit_counterfactual_pairs"] = len(pairs)
    assert len(train) + len(validation) == 2 ** (3 * slots)
    assert not ({r[0] for r in train} & {r[0] for r in validation})
    return result


def historical_stats():
    source = "reports/campaign_arm2_curriculum.json"
    campaign = json.loads((ROOT / source).read_text())
    outputs = []
    for record in campaign["records"]:
        if record["epoch"] not in (200, 500):
            continue
        evaluation = record["eval"]
        intact = evaluation["intact"]["acc_seeds"]
        shuffled = evaluation["shuffle_batch"]["acc_seeds"]
        differences = [a - b for a, b in zip(intact, shuffled)]
        assert len(intact) == len(shuffled) == 5
        mean = statistics.mean(differences)
        sd = statistics.stdev(differences)
        sem = sd / math.sqrt(len(differences))
        observed = abs(sum(differences))
        extreme = sum(abs(sum(sign * value for sign, value in zip(signs, differences)))
                      >= observed - 1e-12 for signs in itertools.product((-1, 1), repeat=5))
        t = mean / sem
        # Exact two-sided Student-t tail with four degrees of freedom.
        u = abs(t) / math.sqrt(t * t + 4)
        outputs.append({"epoch": record["epoch"], "checkpoint": record["checkpoint"],
                        "intact_batches_pct": intact, "shuffle_batches_pct": shuffled,
                        "intact_mean_pct": statistics.mean(intact),
                        "intact_sample_sd_pct": statistics.stdev(intact),
                        "shuffle_mean_pct": statistics.mean(shuffled),
                        "shuffle_sample_sd_pct": statistics.stdev(shuffled),
                        "paired_gaps_pct": differences, "gap_mean_pct": mean,
                        "gap_sample_sd_pct": sd, "gap_sem_pct": sem,
                        "paired_cohen_d": mean / sd, "paired_t_df4": t,
                        "paired_t_two_sided_p": 1 - 1.5 * u + 0.5 * u ** 3,
                        "exact_signflip_two_sided_p": extreme / 32,
                        "statistical_scope": "Five evaluation batches of one trained model; overlapping unique rows; no training-replicate inference."})
    return {"source": source, "records": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/vnext_evidence_audit.json")
    args = parser.parse_args()
    source_paths = ["src/tasks.rs", "src/nca.rs", "src/train.rs", "src/dataset.rs",
                    "scripts/run_campaign.py", "reports/campaign_arm2_curriculum.json",
                    "scripts/audit_substrate_evidence.py"]
    audit = {
        "schema": "titan-substrate-evidence-audit-v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "split_implementation": "Independent Python transcription of tasks.rs observation_hash and IPPR rows, exhaustively enumerated; no RNG resampling.",
        "local_baseline_scope": "Suffix width counts data bits preceding each query; per-slot lookup uses position and may be more powerful than translation-invariant readout; fit only on train rows; ties/unseen contexts use documented rule.",
        "splits": [audit_split(8), audit_split(16)],
        "historical": historical_stats(),
        "source_findings": [
            {"finding": "Legacy STE backward cancellation", "status": "SOURCE_ALGEBRA_VERIFIED",
             "source": "src/nca.rs quantize_carry ste_round/ste_sign",
             "historical_expression": "carry + (quantized.detach() - carry)",
             "historical_derivative": 0,
             "intended_expression": "carry + (quantized - carry).detach()",
             "intended_derivative": 1,
             "scope": "Forward projection can work while recurrence gradient through quantized carry is zero. This script does not execute Candle backward."},
            {"finding": "Legacy train_step repeats algorithmic batch", "status": "SOURCE_VERIFIED",
             "source": "src/train.rs train_step -> src/dataset.rs sample_train_batch_with_mask -> sample_batch_internal_with_mask seed=0",
             "scope": "Historical Trainer path; an explicit-batch harness may avoid this."},
            {"finding": "Legacy --seed does not initialize Trainer weights", "status": "SOURCE_VERIFIED",
             "source": "src/train.rs Trainer::new; no initialize_seeded or set_seed call",
             "scope": "Fresh historical Trainer path; not the independently seeded baseline harness."},
            {"finding": "Historical campaign train resumes every 100 epochs", "status": "SOURCE_VERIFIED",
             "source": "scripts/run_campaign.py run_campaign_arm and train_increment",
             "scope": "Weights-only resume resets Adam; five evaluation seeds are not five training seeds."},
            {"finding": "Radius-1 per tick is conditional", "status": "SOURCE_VERIFIED",
             "source": "src/nca.rs neighbors, dissipate, step_field; src/config.rs FieldConfig::default",
             "scope": "Zero boundary, viscosity=0, feedback=none, carry_channels=0 needed for simple radius-1 bound. Nonzero viscosity adds ceil(viscosity*step_size/.25) hops; default boundary wraps."},
            {"finding": "Symmetric spatial support does not require symmetric learned transport", "status": "SOURCE_ALGEBRA_VERIFIED",
             "source": "src/nca.rs perceive_with_feedback_and_seed",
             "scope": "Basis [x, (right-left)/2, right-2*x+left] spans [left,x,right]; learned linear weights can distinguish direction. Audit narrative of mandatory diffusion is unsupported."}
        ],
        "interpretation": "Historical identity dependence is supported for a limited repeated evaluation domain; full sequential parity, multi-training-seed robustness, and optimization impossibility are not established. This audit changes no task or historical output."
    }
    assert [s["validation_unique"] for s in audit["splits"]] == [11, 789]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; L8/L16 validation unique=11/789; historical p values recomputed.")


if __name__ == "__main__":
    main()
