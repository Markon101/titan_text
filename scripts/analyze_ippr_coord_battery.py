#!/usr/bin/env python3
"""Seed-level analysis for the IPPR coordinate-channel battery (prereg 6670875).

Statistical unit: the seed/model run (prereg sec 6). Paired by seed across arms.
Primary endpoints: Slot 1, Slot 2, InteriorMean = mean(Slot1, Slot2) at tau=16.
Inference: per-seed paired differences, mean, exact sign-flip permutation (2^n),
seed-level bootstrap CI. Token/example counts are descriptive only.
Diverged runs (null/nonfinite diagnostics) are recorded per prereg sec 7 and
excluded pairwise (a seed must be valid in BOTH compared arms; §7).
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

SEEDS = [801, 802, 803, 804, 805]
ARMS = ["A_baseline", "C_sham", "B_coord"]


def load_run(out: str, arm: str, seed: int) -> dict | None:
    p = os.path.join(out, f"{arm}_seed_{seed}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    mpath = os.path.join(r["checkpoint"], "manifest.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else {}
    diag = [m.get(k) for k in ("train_loss", "train_accuracy", "grad_norm", "state_energy")]
    nan_div = any(v is None or (isinstance(v, float) and v != v) for v in diag) \
        or m.get("train_accuracy") == 0.0
    if r.get("diverged") or nan_div:
        return {"diverged": True, "arm": arm, "seed": seed,
                "reason": "NaN diag" if nan_div else "flagged"}
    slots = (r["conditions"].get("intact") or {}).get("slots") or []
    if len(slots) < 4:
        return {"diverged": True, "arm": arm, "seed": seed, "reason": "missing slots"}
    return {
        "diverged": False, "arm": arm, "seed": seed,
        "slots": slots,
        "slot1": slots[1], "slot2": slots[2],
        "interior": (slots[1] + slots[2]) / 2.0,
        "overall": sum(slots) / len(slots),
        "lesion_state": (r["conditions"].get("lesion_state") or {}).get("slots"),
        "shuffle": (r["conditions"].get("shuffle") or {}).get("slots"),
    }


def signflip(diffs: list[float]) -> float:
    n = len(diffs)
    obs = abs(sum(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=n):
        s = abs(sum(d * g for d, g in zip(diffs, signs)))
        hits += s >= obs - 1e-12
        total += 1
    return hits / total


def boot_ci(diffs: list[float], iters: int = 10000) -> tuple[float, float]:
    rng = random.Random(20260927)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/ippr_coordinate"
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            runs[(arm, seed)] = load_run(out, arm, seed)

    print("== Per-seed results (tau=16, L=16) ==")
    for arm in ARMS:
        for seed in SEEDS:
            r = runs[(arm, seed)]
            if r is None:
                print(f"{arm} {seed}: MISSING")
            elif r["diverged"]:
                print(f"{arm} {seed}: DIVERGED ({r.get('reason','NaN diag')})")
            else:
                print(f"{arm} {seed}: slots {r['slots']} interior {r['interior']:.2f}")

    # baseline replication gate: converged baselines only
    a_ok = [runs[("A_baseline", s)] for s in SEEDS
            if runs[("A_baseline", s)] and not runs[("A_baseline", s)]["diverged"]]
    n_ok = len(a_ok)
    print(f"\n== Baseline replication gate: {n_ok}/5 converged baselines ==")
    gate_fail = False
    if n_ok:
        over = sum(1 for r in a_ok if r["interior"] > 58.0)
        gate_fail = over > 1
        print("baseline InteriorMean per seed:", [round(r["interior"], 2) for r in a_ok],
              "| >chance+8pp count:", over)
        print("GATE:", "FAIL -> STOP, do not interpret coordinate arms" if gate_fail
              else "PASS (interior at chance) -> proceed to coordinate contrast")
    else:
        print("GATE: CANNOT EVALUATE (no converged baseline) -> STOP interpretation")

    if n_ok and not gate_fail:
        print("\n== Paired contrasts (valid seeds in both arms) ==")
        for (a1, a2) in [("B_coord", "A_baseline"), ("B_coord", "C_sham"),
                         ("C_sham", "A_baseline")]:
            valid = [s for s in SEEDS
                     if runs[(a1, s)] and not runs[(a1, s)]["diverged"]
                     and runs[(a2, s)] and not runs[(a2, s)]["diverged"]]
            if not valid:
                print(f"{a1} - {a2}: no pairwise-valid seeds")
                continue
            diffs_int = [runs[(a1, s)]["interior"] - runs[(a2, s)]["interior"] for s in valid]
            diffs_s1 = [runs[(a1, s)]["slot1"] - runs[(a2, s)]["slot1"] for s in valid]
            diffs_s2 = [runs[(a1, s)]["slot2"] - runs[(a2, s)]["slot2"] for s in valid]
            mean = sum(diffs_int) / len(diffs_int)
            p = signflip(diffs_int) if len(valid) >= 2 else float("nan")
            lo, hi = boot_ci(diffs_int)
            dir_count = sum(1 for d in diffs_int if d > 0)
            print(f"{a1} - {a2} InteriorMean: mean {mean:+.2f} "
                  f"[boot {lo:+.2f},{hi:+.2f}] sign-flip p={p:.3f} "
                  f"(n={len(valid)}, +diffs {dir_count}/{len(valid)})")
            print(f"   Slot1 diffs {[round(d,2) for d in diffs_s1]}")
            print(f"   Slot2 diffs {[round(d,2) for d in diffs_s2]}")

        # identity gap for converged runs
        print("\n== Identity gap (intact - shuffle) & lesion, converged runs ==")
        for arm in ARMS:
            for seed in SEEDS:
                r = runs[(arm, seed)]
                if r and not r["diverged"] and r.get("shuffle") and len(r["shuffle"]) >= 4:
                    g = (r["slots"][0] - r["shuffle"][0], r["slots"][1] - r["shuffle"][1],
                         r["slots"][2] - r["shuffle"][2], r["slots"][3] - r["shuffle"][3])
                    print(f"{arm} {seed}: G_slot {tuple(round(x,1) for x in g)} "
                          f"lesion {r.get('lesion_state')}")

    with open(os.path.join(out, "analysis_summary.json"), "w") as fh:
        json.dump({f"{a}/{s}": runs[(a, s)] for a in ARMS for s in SEEDS},
                  fh, indent=1, default=str)
    print("\nsaved:", os.path.join(out, "analysis_summary.json"))


if __name__ == "__main__":
    main()
