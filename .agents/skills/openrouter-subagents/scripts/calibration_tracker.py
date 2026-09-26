#!/usr/bin/env python3
"""Empirical Probability Calibration Tracker and Evaluator.

Implements Part 16 of the Cognitive Control Architecture:
- Records (predicted_distribution, observed_outcome) pairs to .agents/telemetry/calibration_records.jsonl.
- Evaluates Brier score, Expected Calibration Error (ECE), and reliability bins.
- Guardrail: Explicitly flags when sample size is too small to claim statistical calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


DEFAULT_CALIBRATION_FILE = Path(".agents/telemetry/calibration_records.jsonl")
MIN_CALIBRATION_SAMPLES = 25


class CalibrationTracker:
    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or DEFAULT_CALIBRATION_FILE

    def record_prediction_outcome(
        self,
        predicted_distribution: dict[str, float],
        observed_outcome: str,
        *,
        decision_type: str = "THINKING_BUDGET",
        task_category: str = "general",
        outcome_source: str,
    ) -> None:
        record_calibration_event(
            predicted_distribution,
            observed_outcome,
            decision_type=decision_type,
            task_category=task_category,
            log_path=self.log_path,
            outcome_source=outcome_source,
        )

    def load_records(self) -> list[dict[str, Any]]:
        return load_calibration_records(self.log_path)

    def generate_calibration_report(self, num_bins: int = 5) -> dict[str, Any]:
        records = self.load_records()
        return compute_brier_and_ece(records, num_bins=num_bins)


def calculate_multiclass_brier_score(records: list[dict[str, Any]]) -> float | None:
    res = compute_brier_and_ece(records)
    return res.get("mean_brier_score", 0.0)


def calculate_expected_calibration_error(records: list[dict[str, Any]], num_bins: int = 5) -> float | None:
    res = compute_brier_and_ece(records, num_bins=num_bins)
    return res.get("expected_calibration_error", 0.0)


def record_calibration_event(
    predicted_distribution: dict[str, float],
    observed_outcome: str,
    *,
    decision_type: str = "THINKING_BUDGET",
    task_category: str = "general",
    log_path: Path | None = None,
    outcome_source: str,
) -> None:
    """Record a prediction-outcome pair for empirical probability calibration."""
    if not isinstance(outcome_source, str) or not outcome_source.strip():
        raise ValueError("An independent outcome artifact/review reference is required")
    if not _valid_distribution(predicted_distribution) or not isinstance(observed_outcome, str) or not observed_outcome.strip():
        raise ValueError("Invalid prediction or observed outcome")
    target = log_path or DEFAULT_CALIBRATION_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "schema_version": 2,
        "outcome_source": outcome_source,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision_type": decision_type,
        "task_category": task_category,
        "predicted_distribution": predicted_distribution,
        "observed_outcome": observed_outcome,
    }

    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_calibration_records(log_path: Path | None = None) -> list[dict[str, Any]]:
    target = log_path or DEFAULT_CALIBRATION_FILE
    if not target.is_file():
        return []
    records = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def _valid_distribution(probs: Any) -> bool:
    return (isinstance(probs, dict) and bool(probs)
            and all(isinstance(k, str) and k for k in probs)
            and all(not isinstance(v, bool) and isinstance(v, (int, float))
                    and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
            and math.isclose(sum(probs.values()), 1.0, abs_tol=1e-3))


def _eligible_record(record: Any) -> bool:
    # Legacy rows include self-labelled model predictions. Preserve them on disk,
    # but require explicit independent outcome provenance for scoring.
    return (isinstance(record, dict) and record.get("schema_version") == 2
            and isinstance(record.get("outcome_source"), str) and bool(record["outcome_source"].strip())
            and isinstance(record.get("observed_outcome"), str) and bool(record["observed_outcome"].strip())
            and _valid_distribution(record.get("predicted_distribution")))


def compute_brier_and_ece(
    records: list[dict[str, Any]],
    num_bins: int = 5,
) -> dict[str, Any]:
    """Compute multi-class Brier score and Expected Calibration Error (ECE)."""
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError("num_bins must be positive")
    input_count = len(records)
    records = [r for r in records if _eligible_record(r)]
    excluded_count = input_count - len(records)
    if not records:
        return {
            "status": "no_data",
            "sample_count": 0,
            "excluded_count": excluded_count,
            "has_minimum_samples": False,
            "is_statistically_calibrated": False,
            "calibration_status_note": f"Preliminary/Uncalibrated: 0/{MIN_CALIBRATION_SAMPLES} samples. Do NOT treat probabilities as ground truth.",
            "mean_brier_score": None,
            "expected_calibration_error": None,
            "reliability_bins": [],
        }

    n = len(records)
    total_brier = 0.0

    # For top-1 confidence ECE
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bins = [{"confs": [], "corrects": []} for _ in range(num_bins)]

    for r in records:
        probs = r.get("predicted_distribution", {})
        actual = r.get("observed_outcome")
        if not probs or not actual:
            continue

        # Multi-class Brier component: sum((p_k - y_k)^2)
        all_options = set(probs.keys())
        all_options.add(actual)
        record_brier = sum((probs.get(opt, 0.0) - (1.0 if opt == actual else 0.0)) ** 2 for opt in all_options)
        total_brier += record_brier

        # Top-1 confidence for ECE
        top_opt, top_conf = max(probs.items(), key=lambda kv: kv[1]) if probs else (actual, 0.0)
        is_correct = 1.0 if top_opt == actual else 0.0

        # Bin assignment
        for b_idx in range(num_bins):
            low = bin_boundaries[b_idx]
            high = bin_boundaries[b_idx + 1]
            if (low <= top_conf < high) or (b_idx == num_bins - 1 and top_conf == high):
                bins[b_idx]["confs"].append(top_conf)
                bins[b_idx]["corrects"].append(is_correct)
                break

    mean_brier = total_brier / n

    # Calculate ECE
    ece = 0.0
    reliability_bins = []
    for b_idx, b in enumerate(bins):
        bin_count = len(b["confs"])
        if bin_count > 0:
            avg_conf = sum(b["confs"]) / bin_count
            avg_acc = sum(b["corrects"]) / bin_count
            ece += (bin_count / n) * abs(avg_acc - avg_conf)
            reliability_bins.append({
                "bin_range": f"{bin_boundaries[b_idx]:.2f}-{bin_boundaries[b_idx+1]:.2f}",
                "count": bin_count,
                "avg_confidence": round(avg_conf, 3),
                "empirical_accuracy": round(avg_acc, 3),
                "calibration_gap": round(abs(avg_acc - avg_conf), 3),
            })

    has_minimum_samples = n >= MIN_CALIBRATION_SAMPLES

    return {
        "status": "ok",
        "sample_count": n,
        "excluded_count": excluded_count,
        "has_minimum_samples": has_minimum_samples,
        "is_statistically_calibrated": False,
        "calibration_status_note": (
            "Descriptive metrics only; sample count does not establish calibration. Evaluate held-out outcomes and uncertainty." if has_minimum_samples
            else f"Preliminary/Uncalibrated: {n}/{MIN_CALIBRATION_SAMPLES} samples. Do NOT treat probabilities as ground truth."
        ),
        "mean_brier_score": round(mean_brier, 4),
        "expected_calibration_error": round(ece, 4),
        "reliability_bins": reliability_bins,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probability Calibration Tracker CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # report
    subparsers.add_parser("report", help="Compute Brier score and ECE calibration report")

    # record
    rec_p = subparsers.add_parser("record", help="Record a prediction-outcome pair")
    rec_p.add_argument("decision_type", help="e.g. THINKING_BUDGET, PLAUSIBILITY")
    rec_p.add_argument("predicted_json", help='e.g. \'{"minimal": 0.2, "normal": 0.8}\'')
    rec_p.add_argument("observed_outcome", help="e.g. normal")
    rec_p.add_argument("--category", default="general")
    rec_p.add_argument("--outcome-source", required=True, help="Independent measurement or adjudicated review reference; never Jev itself")

    args = parser.parse_args(argv)

    if args.subcommand == "record":
        probs = json.loads(args.predicted_json)
        record_calibration_event(probs, args.observed_outcome, decision_type=args.decision_type, task_category=args.category, outcome_source=args.outcome_source)
        print(f"Recorded calibration event for {args.decision_type} -> observed: {args.observed_outcome}")

    elif args.subcommand == "report":
        records = load_calibration_records()
        res = compute_brier_and_ece(records)
        print("=== PROBABILITY CALIBRATION REPORT ===")
        print(f"Total Samples:       {res['sample_count']}")
        print(f"Excluded Records:    {res['excluded_count']}")
        print(f"Calibration Status:  {res.get('calibration_status_note', '')}")
        if res["sample_count"] > 0:
            print(f"Mean Brier Score:    {res['mean_brier_score']:.4f} (lower is better, 0.0 is perfect)")
            print(f"ECE (Top-1):         {res['expected_calibration_error']:.4f}")
            print("\nReliability Bins:")
            for b in res.get("reliability_bins", []):
                print(f"  [{b['bin_range']}] N={b['count']:<3} | Conf: {b['avg_confidence']:.2f} | Acc: {b['empirical_accuracy']:.2f} | Gap: {b['calibration_gap']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
