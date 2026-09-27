#!/usr/bin/env python3
"""Reproducible analysis for the qualification battery (frozen protocol).

Implements the statistical analysis section of
reports/ascii_halting_qualification_protocol.md:
- per-arm summary;
- paired A-X differences (all pairs = primary; truncation-excluded = sensitivity);
- seed-cluster bootstrap CI (10 clusters, 10k resamples);
- seed-level exact sign-flip permutation (2^10; the VALID inferential unit);
- token-level 30-pair permutation (DESCRIPTIVE ONLY, anti-conservative —
  reported for comparison and flagged);
- compute-matching quality (realized tick ratios, truncation counts);
- cluster-aware entropy/tau correlation;
- positional/structural R^2 of a char-class-mean rule.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

SAMPLES = Path("reports/raw/qualification_battery/qualification_samples.json")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else float("nan")


def main() -> int:
    samples = json.load(open(SAMPLES))
    key = lambda s: (s["prompt_name"], s["config"]["seed"])
    arms: dict[str, dict] = {}
    for s in samples:
        if s["arm"] in ("A_adaptive_rerun_check", "sanity_lesion"):
            continue
        arms.setdefault(s["arm"], {})[key(s)] = s

    edit = lambda s: s["metrics"]["nearest_edit_similarity"]
    ticks = lambda s: s["total_ticks"]
    length = lambda s: s["steps_generated"]

    out = {"per_arm": {}, "paired": {}, "entropy_cluster": {}, "positional": {}}

    print("== per-arm summary ==")
    for arm, d in arms.items():
        es = [edit(s) for s in d.values()]
        ts = [ticks(s) for s in d.values()]
        toks = [t["ticks_used"] for s in d.values() for t in s["token_traces"]]
        out["per_arm"][arm] = {
            "edit_mean": statistics.mean(es), "edit_se": statistics.stdev(es) / math.sqrt(len(es)),
            "ticks_per_run_mean": statistics.mean(ts), "mean_tau": statistics.mean(toks),
            "tick_histogram": dict(sorted(Counter(toks).items())),
        }
        print(f"  {arm:14s} edit={statistics.mean(es):.4f}+-{out['per_arm'][arm]['edit_se']:.4f} "
              f"ticks/run={statistics.mean(ts):.1f} mean_tau={statistics.mean(toks):.3f} "
              f"hist={out['per_arm'][arm]['tick_histogram']}")

    def paired(armname, exclude_trunc=False):
        A, X = arms["A_adaptive"], arms[armname]
        pairs = [k for k in A if k in X and not
                 (exclude_trunc and length(X[k]) < 0.8 * length(A[k]))]
        return [edit(A[k]) - edit(X[k]) for k in pairs], pairs

    def cluster_ci(diffs, pairs, nboot=10000, seed=20260926):
        by_seed = defaultdict(list)
        for d, (_, sd) in zip(diffs, pairs):
            by_seed[sd].append(d)
        seeds = sorted(by_seed)
        rng = random.Random(seed)
        boots = []
        for _ in range(nboot):
            sample = [d for _ in seeds for d in by_seed[seeds[rng.randrange(len(seeds))]]]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))], by_seed

    def seed_perm_p(by_seed):
        seeds = sorted(by_seed)
        means = [statistics.mean(by_seed[s]) for s in seeds]
        obs = statistics.mean(means)
        count = 0
        for flips in product([1, -1], repeat=len(seeds)):
            if sum(f * d for f, d in zip(flips, means)) / len(seeds) >= obs - 1e-12:
                count += 1
        return count / 2 ** len(seeds)

    def token_perm_p(diffs, n=30000, seed=20260927):
        # DESCRIPTIVE ONLY: treats 30 nested pairs as exchangeable.
        rng = random.Random(seed)
        obs = statistics.mean(diffs)
        c = sum(1 for _ in range(n)
                if statistics.mean(d if rng.random() < 0.5 else -d for d in diffs) >= obs)
        return (c + 1) / (n + 1)

    for label, excl in (("primary", False), ("sensitivity", True)):
        print(f"\n== paired A-X ({label}) ==")
        out["paired"][label] = {}
        for armname in ["B_fixed_tau4", "B_fixed_tau5", "B_fixed_tau8", "C_matched", "D_matched"]:
            diffs, pairs = paired(armname, exclude_trunc=excl)
            lo, hi, by_seed = cluster_ci(diffs, pairs)
            rec = {
                "mean_diff": statistics.mean(diffs), "n_pairs": len(diffs),
                "cluster_ci": [lo, hi],
                "seed_perm_p": seed_perm_p(by_seed),
                "token_perm_p_descriptive": token_perm_p(diffs),
                "seeds_negative": sum(1 for v in by_seed.values() if statistics.mean(v) < 0),
                "per_pair_diffs": {f"{k[0]}:{k[1]}": round(d, 4) for d, k in zip(diffs, pairs)},
            }
            out["paired"][label][armname] = rec
            print(f"  A-{armname:14s} mean={rec['mean_diff']:+.4f} n={rec['n_pairs']} "
                  f"clusterCI=[{lo:+.4f},{hi:+.4f}] seedPermP={rec['seed_perm_p']:.4f} "
                  f"tokenPermP(desc)={rec['token_perm_p_descriptive']:.4f} seeds_neg={rec['seeds_negative']}/10")

    print("\n== compute matching (C/D) ==")
    out["compute_matching"] = {}
    for armname in ["C_matched", "D_matched"]:
        A, X = arms["A_adaptive"], arms[armname]
        ratios = [ticks(X[k]) / max(1, ticks(A[k])) for k in A]
        lr = [length(X[k]) / max(1, length(A[k])) for k in A]
        rec = {"tick_ratio_mean": statistics.mean(ratios),
               "within_10pct": sum(1 for r in ratios if abs(r - 1) <= 0.1),
               "n": len(ratios), "truncated_lt_0.8": sum(1 for r in lr if r < 0.8)}
        out["compute_matching"][armname] = rec
        print(f"  {armname}: ratio mean={rec['tick_ratio_mean']:.3f} within10%={rec['within_10pct']}/{rec['n']} truncated={rec['truncated_lt_0.8']}")

    # cluster-aware entropy/tau
    A = arms["A_adaptive"]
    seq_rs = []
    for s in A.values():
        t = [x["ticks_used"] for x in s["token_traces"]]
        e = [x["entropy"] for x in s["token_traces"]]
        seq_rs.append(pearson(t, e))
    valid = [r for r in seq_rs if not math.isnan(r)]
    tok_t = [x["ticks_used"] for s in A.values() for x in s["token_traces"]]
    tok_e = [x["entropy"] for s in A.values() for x in s["token_traces"]]
    pairs_all = [(t, e) for s in A.values() for t, e in
                 zip([x["ticks_used"] for x in s["token_traces"]],
                     [x["entropy"] for x in s["token_traces"]])]
    rng = random.Random(20260926)
    boots = []
    seqs = [[p for p in seq] for seq in
            [[(x["ticks_used"], x["entropy"]) for x in s["token_traces"]] for s in A.values()]]
    for _ in range(10000):
        pool = [p for _ in seqs for p in seqs[rng.randrange(len(seqs))]]
        r = pearson([p[0] for p in pool], [p[1] for p in pool])
        if not math.isnan(r):
            boots.append(r)
    boots.sort()
    out["entropy_cluster"] = {
        "token_r_descriptive": pearson(tok_t, tok_e), "n_tokens": len(tok_t),
        "n_sequences": len(valid), "n_seeds": 10,
        "seq_r_mean": statistics.mean(valid), "seq_r_sd": statistics.stdev(valid),
        "seq_r_negative": sum(1 for r in valid if r < 0),
        "cluster_bootstrap_ci": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]],
    }
    e = out["entropy_cluster"]
    print(f"\n== entropy/tau (cluster-aware) ==\n  token r={e['token_r_descriptive']:.4f} (descriptive, N={e['n_tokens']}) "
          f"seq r mean={e['seq_r_mean']:.4f} ({e['seq_r_negative']}/{e['n_sequences']} neg) "
          f"clusterCI=[{e['cluster_bootstrap_ci'][0]:.4f},{e['cluster_bootstrap_ci'][1]:.4f}]")

    # positional: char-class-mean rule R^2
    cls = defaultdict(list)
    for s in A.values():
        for t in s["token_traces"]:
            cls[t["char_class"]].append(t["ticks_used"])
    cm = {c: statistics.mean(v) for c, v in cls.items()}
    obs = [t["ticks_used"] for s in A.values() for t in s["token_traces"]]
    pred = [cm[t["char_class"]] for s in A.values() for t in s["token_traces"]]
    m = statistics.mean(obs)
    r2 = 1 - sum((o - p) ** 2 for o, p in zip(obs, pred)) / sum((o - m) ** 2 for o in obs)
    out["positional"] = {"class_mean_r2": r2, "class_means": cm}
    print(f"\n== positional ==\n  char-class-mean rule R^2={r2:.4f}  class means={ {k: round(v,3) for k,v in cm.items()} }")

    Path("reports/raw/qualification_battery/qualification_analysis.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nSaved analysis to reports/raw/qualification_battery/qualification_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
