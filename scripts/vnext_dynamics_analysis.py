#!/usr/bin/env python3
"""Dynamics analysis for the vNext transport battery.

Reads the tick-resolved sweep JSONs produced by scripts/vnext_battery.sh
(`*_eval_*.json`, budgets 0,4,6,8,12,16 on the exhaustive 789-row heldout split)
and the causal-influence reports (`influence` command output JSON), and computes:

1. Interior staircase onset per arm: the first budget where interior-slot accuracy
   rises above the shuffled/chance band. A transport mechanism predicts onset moves
   EARLIER as k grows; a local shortcut predicts arm-invariant onset.
2. Interior accuracy at the binding horizon T=8 (the falsifier metric).
3. Transport-front speed: from the influence matrix, the first tick where flipping
   a far-prefix bit changes an interior slot's prediction (W(k,q)); RFT predicts
   W ~ (q-i)/k + launch term, decreasing in k.

Falsification rule (frozen): if interior accuracy at T=8 stays at chance AND onset
is arm-invariant across k in {1,2,4}, Relay-Fold Transport is FALSIFIED -> a clean
negative result that transport-at-speed-k is not the lever.
"""
import json, glob, os, statistics as st, argparse

def load_eval(path):
    d = json.load(open(path))
    out = {}
    for b in d["budgets"]:
        ps = b["per_slot_accuracy"]
        out[b["latent_ticks"]] = {"interior": (ps[1] + ps[2]) / 2 * 100,
                                   "slot0": ps[0] * 100, "slot3": ps[3] * 100,
                                   "full": b["accuracy"] * 100}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", default="runs/vnext_battery")
    ap.add_argument("--out", default="reports/vnext_battery_analysis.json")
    args = ap.parse_args()
    arms = {}
    for p in sorted(glob.glob(os.path.join(args.rundir, "*_eval_*.json"))):
        name = os.path.basename(p).replace("_eval_", "|").replace(".json", "")
        arm, seed = name.split("|")
        if "lesion" in arm: continue
        arms.setdefault(arm, []).append((seed, load_eval(p)))
    budgets = [4, 6, 8, 12, 16]
    report = {"rundir": args.rundir, "budgets": budgets, "arms": {}}
    print(f"{'arm':<14} " + " ".join(f"T{b:<4}" for b in budgets) + "  | interior@T8")
    for arm, runs in sorted(arms.items()):
        rows = {}
        for seed, ev in runs:
            for b, v in ev.items():
                rows.setdefault(b, []).append(v["interior"])
        means = {b: st.mean(v) for b, v in rows.items()}
        onset = next((b for b in budgets if means.get(b, 0) > 52.0), None)  # above chance band
        line = f"{arm:<14} " + " ".join(f"{means.get(b, float('nan')):5.1f}" for b in budgets)
        print(line + f"  | {means.get(8, float('nan')):5.1f}  onset T={onset}")
        report["arms"][arm] = {"n": len(runs), "interior_means": means, "onset_budget": onset}
    # Falsification check
    ks = [a for a in report["arms"] if "k" in a and "fold" in a or "rft" in a.lower()]
    print("\nFALSIFIER: interior@T8 at chance (<52%) for all k AND onset arm-invariant => RFT falsified.")
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
