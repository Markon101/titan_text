#!/usr/bin/env python3
"""RD-018 tick-resolved staircase diagnostic (forensic, post-battery).

Runs the EXISTING `sweep` command per converged checkpoint at budgets
0,3,7,11,15,16 with the full 789-row heldout split (batch-size 789, seeds 0),
then aggregates per-slot accuracy per tick. Requires no source change.

What it tests: whether a genuinely computed prefix-parity solution produces the
causal light-cone staircase (slot q at cell 3+4q becomes solvable at t>=3+4q),
versus gradual/memorized onset. NOTE: the terminal readout projection is only
trained at t=16 in Arm A, so early-tick readout failure conflates "information
absent" with "readout untrained at that tick" — this diagnostic reports the
curve; it does NOT by itself prove absence of information (see the frozen-state
probe design). Use it to compare arms and onsets, not as a proof of attenuation.
"""
import json, os, statistics as st, math

ROOT = "/data/data/com.termux/files/home/projects/titan_text"
RUNDIR = os.path.join(ROOT, "runs/rd018_aux_supervision")
ARMS = {"A_baseline": [901, 903, 904], "B_aux": [906, 908, 909, 910], "C_sham": [911, 912, 913, 914, 915]}
BUDGETS = "0,3,7,11,15,16"
OUT = os.path.join(ROOT, "reports/rd018_staircase_analysis.json")

def ci(vals):
    n = len(vals); m = st.mean(vals)
    if n < 2: return [m, m]
    h = 1.96 * st.stdev(vals) / math.sqrt(n)
    return [m - h, m + h]

def main():
    out = {"budgets": [int(b) for b in BUDGETS.split(",")], "eval_rows": 789,
           "eval_seed": 0, "unit_of_analysis": "trained run", "per_slot_t16": {}, "traj": {}}
    slots_t16 = {}
    traj_slot0, traj_int = {}, {}
    for a, seeds in ARMS.items():
        s0 = [[] for _ in range(4)]
        t0, ti = {}, {}
        for s in seeds:
            p = os.path.join(RUNDIR, f"forensic_staircase_{a}_{s}.json")
            d = json.load(open(p))
            for b in d["budgets"]:
                t = b["latent_ticks"]; ps = b["per_slot_accuracy"]
                if t == 16:
                    for i, v in enumerate(ps): s0[i].append(v * 100)
                t0.setdefault(t, []).append(ps[0] * 100)
                ti.setdefault(t, []).append((ps[1] + ps[2]) / 2 * 100)
        slots_t16[a] = [{"mean": st.mean(v), "ci95": ci(v), "values": v} for v in s0]
        traj_slot0[a] = {t: st.mean(v) for t, v in sorted(t0.items())}
        traj_int[a] = {t: st.mean(v) for t, v in sorted(ti.items())}
    out["per_slot_t16"] = slots_t16
    out["traj_slot0"] = traj_slot0
    out["traj_interior"] = traj_int
    out["interior_mean_t16"] = {
        a: (slots_t16[a][1]["mean"] + slots_t16[a][2]["mean"]) / 2 for a in ARMS}
    im = out["interior_mean_t16"]
    out["contrasts_t16"] = {"B-A": im["B_aux"] - im["A_baseline"],
                            "C-A": im["C_sham"] - im["A_baseline"],
                            "B-C": im["B_aux"] - im["C_sham"]}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)
    print(json.dumps({"interior_mean_t16": {a: round(v, 2) for a, v in im.items()},
                      "contrasts_t16": {k: round(v, 2) for k, v in out["contrasts_t16"].items()},
                      "traj_slot0": traj_slot0, "traj_interior": traj_int}, indent=1))

if __name__ == "__main__":
    main()
