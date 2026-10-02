#!/usr/bin/env python3
"""RD-018 reproducible analysis from immutable raw artifacts.

Reads ONLY runs/rd018_aux_supervision/ (manifests, eval JSONs, train logs) and
writes reports/rd018_battery_analysis.json. Deterministic: fixed RNG seeds.

Design constraints encoded here (see docs/VNEXT_RECONSTRUCTION.md:40-47 and
src/train.rs:159-166 for provenance):
  * Independent statistical unit = a trained RUN (seed label), never a token.
  * Arms use DISJOINT seed pools (901-905 / 906-910 / 911-915): no preregistered
    pairing exists, so no paired test is reported as confirmatory. Paired-by-index
    is emitted only as an explicitly labelled arbitrary comparator.
  * Diverged runs have no endpoint outcome. Prereg specifies NO exclusion rule.
    Converged-run analysis is reported as primary-for-description; failure-coding
    (zero) is reported as a labelled sensitivity assumption, never as "ITT".
  * A failed decoder/probe is not evidence of absent information; that claim is
    out of scope for this script.
"""
import json, os, random, re, hashlib, itertools, statistics as st

ROOT = "/data/data/com.termux/files/home/projects/titan_text"
RUNDIR = os.path.join(ROOT, "runs/rd018_aux_supervision")
OUT = os.path.join(ROOT, "reports/rd018_battery_analysis.json")
ARMS = {"A_baseline": [901, 902, 903, 904, 905],
        "B_aux": [906, 907, 908, 909, 910],
        "C_sham": [911, 912, 913, 914, 915]}
INTERIOR = (1, 2)  # slots of 4; endpoint = mean of slots 1,2

def sha8(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except FileNotFoundError:
        return None

def load_runs():
    runs = {}
    for arm, seeds in ARMS.items():
        runs[arm] = {}
        for s in seeds:
            d = os.path.join(RUNDIR, arm, f"seed_{s}")
            rec = {"arm": arm, "seed": s}
            mpath = os.path.join(d, "manifest.json")
            rec["manifest_sha"] = sha8(mpath)
            if not os.path.exists(mpath):
                rec["manifest_missing"] = True
                runs[arm][s] = rec
                continue
            man = json.load(open(mpath))
            acc, loss, grad = man.get("train_accuracy"), man.get("train_loss"), man.get("grad_norm")
            finite = [v for v in (acc, loss, grad) if v is not None and v == v]
            rec["train_accuracy"] = acc
            rec["train_loss"] = loss
            rec["grad_norm"] = grad
            rec["converged"] = bool(acc not in (None, 0) and len(finite) == 3)
            rec["manifest_seed_field"] = man.get("random_seed")
            for kind, fn in (("intact", f"{arm}_eval_{s}.json"),
                             ("lesion", f"{arm}_eval_{s}_lesion.json")):
                p = os.path.join(RUNDIR, fn)
                rec[f"{kind}_sha"] = sha8(p)
                if not os.path.exists(p):
                    rec[f"{kind}_missing"] = True
                    continue
                e = json.load(open(p))
                b = e["budgets"][0]
                ps = b["per_slot_accuracy"]
                rec[f"{kind}_ticks"] = b["latent_ticks"]
                rec[f"{kind}_full"] = b["accuracy"]
                rec[f"{kind}_per_slot"] = ps
                rec[f"{kind}_interior"] = st.mean(ps[INTERIOR[0]:INTERIOR[1] + 1])
                rec[f"{kind}_ce"] = b["loss"]
                rec[f"{kind}_conf"] = b["mean_confidence"]
            tl = os.path.join(RUNDIR, f"{arm}_train_{s}.log")
            rec["train_log_sha"] = sha8(tl)
            if os.path.exists(tl):
                aux = re.findall(r"aux=([0-9.]+|NaN)", open(tl).read())
                fin = [float(a) for a in aux if a != "NaN"]
                rec["aux_series_n"] = len(aux)
                rec["aux_first"] = fin[0] if fin else None
                rec["aux_last"] = fin[-1] if fin else None
                rec["first_nan_log_row"] = next((i + 1 for i, a in enumerate(aux) if a == "NaN"), None)
            runs[arm][s] = rec
    return runs

def exact_perm_two_sided(x, y, max_combos=200000):
    """Exact pooled-label permutation on the mean difference. Enumerates all
    label assignments when feasible, else Monte-Carlo with a fixed seed."""
    obs = abs(st.mean(x) - st.mean(y))
    pool = list(x) + list(y)
    n = len(x)
    total = itertools.combinations(range(len(pool)), n)
    k = 0
    cnt = 0
    for idx in total:
        k += 1
        if k > max_combos:
            break
        sel = set(idx)
        a = [pool[i] for i in sel]
        b = [pool[i] for i in range(len(pool)) if i not in sel]
        if abs(st.mean(a) - st.mean(b)) >= obs - 1e-12:
            cnt += 1
    if k <= max_combos:
        return {"kind": "exact", "p": cnt / k, "combos": k}
    rng = random.Random(20261002)
    N = 40000
    c = 0
    for _ in range(N):
        r = rng.sample(pool, len(pool))
        if abs(st.mean(r[:n]) - st.mean(r[n:])) >= obs - 1e-12:
            c += 1
    return {"kind": "monte_carlo", "p": c / N, "draws": N, "seed": 20261002}

def bootstrap_diff(x, y, n=20000, seed=20261002):
    """Independent (unpaired) run resampling CI for mean(x)-mean(y)."""
    rng = random.Random(seed)
    d = sorted(st.mean([rng.choice(x) for _ in x]) - st.mean([rng.choice(y) for _ in y]) for _ in range(n))
    return {"lo": d[int(0.025 * n)], "hi": d[int(0.975 * n)], "draws": n, "seed": seed}

def main():
    runs = load_runs()
    out = {"artifact": "runs/rd018_aux_supervision", "generated_by": os.path.basename(__file__),
           "unit_of_analysis": "trained run", "interior_slots": list(INTERIOR), "runs": runs, "analyses": {}}

    conv = {a: [runs[a][s] for s in ARMS[a] if runs[a][s].get("converged")] for a in ARMS}
    div = {a: [s for s in ARMS[a] if not runs[a][s].get("converged")] for a in ARMS}
    out["diverged_seeds"] = div
    out["converged_n"] = {a: len(conv[a]) for a in ARMS}

    ep = {}
    for a in ARMS:
        vals = [r["intact_interior"] for r in conv[a]]
        ep[a] = {"per_run_interior": vals, "mean": st.mean(vals) if vals else None,
                 "n": len(vals), "seeds": [r["seed"] for r in conv[a]]}
    out["analyses"]["endpoint_converged"] = ep
    # full-lattice endpoint too (descriptive)
    out["analyses"]["endpoint_converged_full"] = {
        a: {"mean": st.mean([r["intact_full"] for r in conv[a]]) if conv[a] else None,
            "values": [r["intact_full"] for r in conv[a]]} for a in ARMS}

    A, B, C = (ep["A_baseline"]["per_run_interior"], ep["B_aux"]["per_run_interior"], ep["C_sham"]["per_run_interior"])
    contrasts = {}
    for name, (x, y) in {"B-A": (B, A), "C-A": (C, A), "B-C": (B, C)}.items():
        if x and y:
            contrasts[name] = {
                "mean_diff_pp": (st.mean(x) - st.mean(y)) * 100,
                "perm": exact_perm_two_sided(x, y),
                "bootstrap_ci95_pp": {k: v * 100 for k, v in bootstrap_diff(x, y).items()
                                      if k in ("lo", "hi")},
                "cartesian_win_frac": sum(1 for b in x for a in y if b > a) / (len(x) * len(y)),
                "cartesian_ties": sum(1 for b in x for a in y if b == a),
                "n_x": len(x), "n_y": len(y),
            }
    out["analyses"]["contrasts_converged"] = contrasts

    # explicitly labelled arbitrary comparator (NOT preregistered pairing)
    if A and B:
        m = min(len(A), len(B))
        out["analyses"]["arbitrary_index_paired_B_minus_A_pp"] = {
            "values": [(B[i] - A[i]) * 100 for i in range(m)],
            "note": "index pairing across disjoint seed pools is arbitrary; descriptive only",
        }

    # failure-coding sensitivity: diverged runs counted as 0.0 endpoint
    fc = {}
    for a in ARMS:
        v = [(r["intact_interior"] if r.get("converged") else 0.0) for r in
             [runs[a][s] for s in ARMS[a]]]
        fc[a] = {"mean": st.mean(v) if v else None, "assumption": "diverged==0.0 (no endpoint outcome)", "values": v}
    out["analyses"]["failure_coding_sensitivity"] = fc
    if fc["A_baseline"]["mean"] is not None and fc["B_aux"]["mean"] is not None:
        out["analyses"]["failure_coding_sensitivity"]["B_minus_A_pp"] = (
            fc["B_aux"]["mean"] - fc["A_baseline"]["mean"]) * 100

    # lesion aggregates (arm-categorical behaviour recorded, not interpreted)
    les = {}
    for a in ARMS:
        rows = []
        for r in conv[a]:
            if r.get("lesion_interior") is not None:
                rows.append({"seed": r["seed"], "intact": r["intact_interior"], "lesion": r["lesion_interior"],
                             "intact_full": r["intact_full"], "lesion_full": r["lesion_full"],
                             "lesion_ce": r["lesion_ce"]})
        les[a] = rows
    out["analyses"]["lesion_rows"] = les

    # aux trajectories
    out["analyses"]["aux_trajectories"] = {
        a: {str(r["seed"]): {"first": r.get("aux_first"), "last": r.get("aux_last"), "n": r.get("aux_series_n")}
            for r in conv[a]} for a in ARMS}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    print(json.dumps({"diverged": div, "converged_n": out["converged_n"],
                      "endpoint": {a: (round(ep[a]["mean"] * 100, 2) if ep[a]["mean"] else None) for a in ARMS},
                      "mindiff": {k: round(v["mean_diff_pp"], 2) for k, v in contrasts.items()},
                      "p": {k: round(v["perm"]["p"], 4) for k, v in contrasts.items()},
                      "failure_coding_mindiff_B_minus_A_pp":
                          round(out["analyses"]["failure_coding_sensitivity"].get("B_minus_A_pp", float("nan")), 2)},
                     indent=1))

if __name__ == "__main__":
    main()
