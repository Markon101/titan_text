#!/usr/bin/env python3
"""Scientific Analysis of Adaptive Recurrent Compute in Titan Text.

Processes raw campaign records from reports/raw/adaptive_halting/campaign_samples.json:
- Quality vs Compute Pareto Frontier (Arm A vs Arm B fixed sweep)
- Causal hypothesis test: Arm A (Adaptive) vs Arm C (Shuffled)
- Sham control test: Arm A vs Arm D (Random uniform)
- Recurrence allocation across character classes
- Correlation between predictive entropy and recurrent ticks
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def bootstrap_ci_mean(data: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    if not data:
        return 0.0, 0.0
    import random
    rng = random.Random(seed)
    boot_means = []
    n = len(data)
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower_idx = int((alpha / 2.0) * n_boot)
    upper_idx = int((1.0 - alpha / 2.0) * n_boot)
    return boot_means[lower_idx], boot_means[min(upper_idx, n_boot - 1)]


def mean_std_err(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n > 1:
        variance = sum((x - m) ** 2 for x in values) / (n - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(n)
    else:
        sd = 0.0
        se = 0.0
    return m, sd, se


def pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    var_x = sum((xi - mx) ** 2 for xi in x)
    var_y = sum((yi - my) ** 2 for yi in y)
    if var_x < 1e-12 or var_y < 1e-12:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(var_x * var_y)


def analyze_campaign(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)

    # 1. Group by arm
    arm_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        arm_groups[s["arm"]].append(s)

    report_data: Dict[str, Any] = {"arms": {}, "head_to_head": {}, "character_classes": {}, "entropy_corr": 0.0}

    print("================================================================================")
    print("TITAN TEXT ADAPTIVE HALTING SCIENTIFIC EVALUATION")
    print(f"Total generation samples analyzed: {len(samples)}")
    print("================================================================================\n")

    print("--- 1. SUMMARY OF EXPERIMENTAL ARMS -------------------------------------------")
    print(f"{'Arm':<20} | {'Mean Tau':<10} | {'Edit Sim':<15} | {'H-Symmetry':<12} | {'Exact Matches'}")
    print("-" * 80)

    arm_order = ["A_adaptive"] + [f"B_fixed_tau{t}" for t in [0, 1, 2, 4, 8, 16]] + ["C_shuffled_control", "D_random_sham"]
    arm_summaries = {}

    for arm in arm_order:
        if arm not in arm_groups:
            continue
        group = arm_groups[arm]
        edit_sims = [s["metrics"]["nearest_edit_similarity"] for s in group]
        mean_taus = [s["mean_tau"] for s in group]
        syms = [s["metrics"]["horizontal_symmetry"] for s in group]
        exact_cnt = sum(1 for s in group if s["metrics"]["exact_training_match"])

        m_edit, _, se_edit = mean_std_err(edit_sims)
        m_tau, _, _ = mean_std_err(mean_taus)
        m_sym, _, _ = mean_std_err(syms)

        arm_summaries[arm] = {
            "mean_tau": m_tau,
            "mean_edit_sim": m_edit,
            "se_edit_sim": se_edit,
            "mean_sym": m_sym,
            "exact_matches": exact_cnt,
            "n": len(group),
        }
        print(f"{arm:<20} | {m_tau:<10.2f} | {m_edit:.4f} ± {se_edit:.4f} | {m_sym:<12.4f} | {exact_cnt}/{len(group)}")

    report_data["arms"] = arm_summaries

    # 2. Causal Pairwise Tests (Arm A vs Arm C, Arm A vs Arm D)
    print("\n--- 2. CAUSAL COMPARISON: ADAPTIVE vs MATCHED-BUDGET CONTROLS -----------------")
    # Match Arm A and Arm C by (prompt_name, seed)
    paired_a_c = []
    a_dict = {(s["prompt_name"], s["config"]["seed"]): s for s in arm_groups.get("A_adaptive", [])}
    c_dict = {(s["prompt_name"], s["config"]["seed"]): s for s in arm_groups.get("C_shuffled_control", [])}
    d_dict = {(s["prompt_name"], s["config"]["seed"]): s for s in arm_groups.get("D_random_sham", [])}

    diffs_a_c = []
    for key, sample_a in a_dict.items():
        if key in c_dict:
            sample_c = c_dict[key]
            sim_a = sample_a["metrics"]["nearest_edit_similarity"]
            sim_c = sample_c["metrics"]["nearest_edit_similarity"]
            diff = sim_a - sim_c
            diffs_a_c.append(diff)
            paired_a_c.append((key, sample_a["mean_tau"], sim_a, sim_c, diff))

    if diffs_a_c:
        m_diff, sd_diff, se_diff = mean_std_err(diffs_a_c)
        ci_lo, ci_hi = bootstrap_ci_mean(diffs_a_c)
        t_stat = m_diff / (se_diff + 1e-12) if se_diff > 0 else 0.0
        win_a = sum(1 for d in diffs_a_c if d > 1e-5)
        tie = sum(1 for d in diffs_a_c if abs(d) <= 1e-5)
        win_c = sum(1 for d in diffs_a_c if d < -1e-5)

        c_samples = list(c_dict.values())
        mean_consumed_frac = sum(s.get("schedule_consumed_fraction", 1.0) for s in c_samples) / max(1, len(c_samples))
        mean_budget_ratio = sum(s.get("realized_budget_ratio", 1.0) for s in c_samples) / max(1, len(c_samples))

        print(f"Arm A (Adaptive) vs Arm C (Shuffled Adaptive Control - Matched Budget):")
        print(f"  Paired N                  : {len(diffs_a_c)}")
        print(f"  Mean Delta Sim (A - C)    : {m_diff:+.4f} ± {se_diff:.4f} (95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}])")
        print(f"  Paired t-statistic        : t = {t_stat:.2f}")
        print(f"  Win / Tie / Loss          : {win_a} / {tie} / {win_c}")
        print(f"  Arm C Schedule Parity     : mean consumed fraction = {mean_consumed_frac:.3f}, realized budget ratio = {mean_budget_ratio:.3f}")
        report_data["head_to_head"]["A_vs_C"] = {
            "mean_delta": m_diff,
            "se_delta": se_diff,
            "ci_95": [ci_lo, ci_hi],
            "t_stat": t_stat,
            "win_tie_loss": [win_a, tie, win_c],
            "mean_consumed_fraction": mean_consumed_frac,
            "mean_budget_ratio": mean_budget_ratio,
        }

    diffs_a_d = []
    for key, sample_a in a_dict.items():
        if key in d_dict:
            sample_d = d_dict[key]
            sim_a = sample_a["metrics"]["nearest_edit_similarity"]
            sim_d = sample_d["metrics"]["nearest_edit_similarity"]
            diff = sim_a - sim_d
            diffs_a_d.append(diff)

    if diffs_a_d:
        m_diff_d, _, se_diff_d = mean_std_err(diffs_a_d)
        ci_lo_d, ci_hi_d = bootstrap_ci_mean(diffs_a_d)
        print(f"\nArm A (Adaptive) vs Arm D (Uniform Random Sham [1, 16] - Unmatched):")
        print(f"  Paired N                  : {len(diffs_a_d)}")
        print(f"  Mean Delta Sim (A - D)    : {m_diff_d:+.4f} ± {se_diff_d:.4f} (95% CI: [{ci_lo_d:+.4f}, {ci_hi_d:+.4f}])")
        report_data["head_to_head"]["A_vs_D"] = {
            "mean_delta": m_diff_d,
            "se_delta": se_diff_d,
            "ci_95": [ci_lo_d, ci_hi_d],
        }

    # Arm A Degeneracy Audit
    a_samples = list(a_dict.values())
    if a_samples:
        frac_min = sum(s.get("frac_tau_min", 0.0) for s in a_samples) / len(a_samples)
        frac_max = sum(s.get("frac_tau_max", 0.0) for s in a_samples) / len(a_samples)
        num_deg = sum(1 for s in a_samples if s.get("is_degenerate", False))
        print(f"\nArm A Halting Distribution & Degeneracy Audit:")
        print(f"  Mean fraction at tau_min=1 : {frac_min:.3f}")
        print(f"  Mean fraction at tau_max=16: {frac_max:.3f}")
        print(f"  Degenerate runs (>90% extremes): {num_deg}/{len(a_samples)}")
        report_data["adaptive_degeneracy"] = {
            "frac_tau_min": frac_min,
            "frac_tau_max": frac_max,
            "degenerate_runs": num_deg,
            "total_runs": len(a_samples),
        }

    # 3. Dynamic Allocation across Character Classes
    print("\n--- 3. RECURRENT DYNAMICS: TAU ALLOCATION BY CHARACTER CLASS ------------------")
    adaptive_samples = arm_groups.get("A_adaptive", [])
    class_ticks: Dict[str, List[int]] = defaultdict(list)
    entropy_list: List[float] = []
    ticks_list: List[float] = []

    for s in adaptive_samples:
        for t in s.get("token_traces", []):
            cls_name = t["char_class"]
            ticks = t["ticks_used"]
            class_ticks[cls_name].append(ticks)
            entropy_list.append(t["entropy"])
            ticks_list.append(float(ticks))

    print(f"{'Class':<15} | {'Tokens':<8} | {'Mean Tau':<10} | {'Median':<8} | {'Min':<5} | {'Max'}")
    print("-" * 60)
    for cls_name, t_list in sorted(class_ticks.items(), key=lambda kv: -len(kv[1])):
        m_t, _, se_t = mean_std_err([float(x) for x in t_list])
        sorted_t = sorted(t_list)
        med = sorted_t[len(sorted_t) // 2] if sorted_t else 0
        min_t = sorted_t[0] if sorted_t else 0
        max_t = sorted_t[-1] if sorted_t else 0
        report_data["character_classes"][cls_name] = {
            "count": len(t_list),
            "mean_tau": m_t,
            "se_tau": se_t,
            "median": med,
            "min": min_t,
            "max": max_t,
        }
        print(f"{cls_name:<15} | {len(t_list):<8} | {m_t:<10.2f} | {med:<8} | {min_t:<5} | {max_t}")

    # 4. Correlation between predictive entropy and ticks
    if entropy_list and ticks_list:
        r = pearson_corr(entropy_list, ticks_list)
        report_data["entropy_corr"] = r
        print(f"\nPearson correlation between predictive entropy H(P_t) and allocated tau: r = {r:+.4f} (N={len(entropy_list)} tokens)")

    # 5. Quality-Compute Pareto Frontier Summary
    print("\n--- 4. PARETO EFFICIENCY: QUALITY vs COMPUTE -----------------------------------")
    if "B_fixed_tau0" in arm_summaries:
        lesion_sim = arm_summaries["B_fixed_tau0"]["mean_edit_sim"]
        print(f"Lesion Baseline (tau=0): EditSim = {lesion_sim:.4f} (Reference ablation, uncoupled from recurrent sweep)")
        report_data["lesion_sim"] = lesion_sim

    fixed_curve = []
    for t in [1, 2, 4, 8, 16]:
        arm_name = f"B_fixed_tau{t}"
        if arm_name in arm_summaries:
            fixed_curve.append((arm_summaries[arm_name]["mean_tau"], arm_summaries[arm_name]["mean_edit_sim"]))

    adaptive_point = (arm_summaries["A_adaptive"]["mean_tau"], arm_summaries["A_adaptive"]["mean_edit_sim"]) if "A_adaptive" in arm_summaries else (0, 0)
    print(f"Continuous Fixed Tau Curve (Tau >= 1): {', '.join(f'({t:.0f}t -> {s:.3f})' for t, s in fixed_curve)}")
    print(f"Adaptive Halting Point                : ({adaptive_point[0]:.2f}t -> {adaptive_point[1]:.3f})")

    sorted_fixed = sorted(fixed_curve, key=lambda x: x[0])
    interp_sim = None
    for i in range(len(sorted_fixed) - 1):
        x0, y0 = sorted_fixed[i]
        x1, y1 = sorted_fixed[i + 1]
        if x0 <= adaptive_point[0] <= x1 and x1 > x0:
            interp_sim = y0 + (adaptive_point[0] - x0) / (x1 - x0) * (y1 - y0)
            break

    if interp_sim is not None:
        gain = adaptive_point[1] - interp_sim
        print(f"Interpolated Fixed Quality at same budget ({adaptive_point[0]:.2f}t): {interp_sim:.3f}")
        print(f"Adaptive Quality Advantage: {gain:+.4f} ({gain/interp_sim*100:+.2f}%)")
        report_data["pareto_gain"] = gain

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Adaptive Halting Campaign")
    parser.add_argument("--json", default="reports/raw/adaptive_halting/campaign_samples.json")
    parser.add_argument("--output", default="reports/raw/adaptive_halting/campaign_analysis.json")
    args = parser.parse_args()

    results = analyze_campaign(args.json)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Analysis summary written to {args.output}")
