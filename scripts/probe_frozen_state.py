#!/usr/bin/env python3
"""Frozen-state linear probe for RD-018b (offline).

Consumes a state export produced by the `export-states` subcommand:
  {
    "rows": N, "query_cells": [3,7,11,15], "ticks": [3,4,7,8,11,12,15,16],
    "channels": C,
    "is_train": [bool x N],                  # True = train split, False = heldout
    "targets": {"<cell>": [0/1 x N]},        # parity target at each query cell
    "states": {"<tick>": {"<cell>": [[x_{row,channel}] ...]}},   # [N][C]
    "oracle_bits": {"<cell>": [[bit x K] x N]}                   # true light-cone prefix bits
  }

For each (tick, cell) we fit a LOW-CAPACITY linear probe on the TRAIN rows and
evaluate on the HELDOUT rows. Controls:
  * oracle      : probe on oracle_bits (true light-cone inputs) -> should be ~100%
  * shuffled    : probe with permuted train labels -> should be chance (~50%)
  * leakage     : probe at t < distance(cell) -> should be chance; above chance
                  means leakage/memorization and invalidates that (tick,cell).
A failed probe is NOT proof the information is absent (failed decoder != absence);
it only bounds recoverability by THIS probe class. Interpreted together with the
oracle/shuffle/leakage controls and the within-model slot0-vs-interior dissociation.
"""
import json, sys, argparse
import numpy as np

def linear_probe_fit(X, y):
    """Least-squares linear probe to labels in {-1,+1}. Returns (w, b)."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    # ridge for numerical stability
    lam = 1e-3
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    return w

def linear_probe_acc(w, X, y):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    pred = np.sign(Xb @ w)
    pred[pred == 0] = 1
    return float((pred == y).mean())

def fit_eval(Xtr, ytr, Xte, yte):
    w = linear_probe_fit(Xtr, ytr)
    return linear_probe_acc(w, Xte, yte)

def to_pm1(v):
    return np.where(np.asarray(v) > 0.5, 1.0, -1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_json")
    ap.add_argument("--shuffle-seed", type=int, default=20261002)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D = json.load(open(args.export_json))
    cells = D["query_cells"]; ticks = D["ticks"]; is_train = np.array(D["is_train"], bool)
    dist = {c: c for c in cells}  # radius-1: cell q depends on input 0 after q ticks
    rng = np.random.default_rng(args.shuffle_seed)
    result = {"export": args.export_json, "unit": "row (heldout observation)",
              "n_train": int(is_train.sum()), "n_heldout": int((~is_train).sum()), "cells": {}}
    for c in cells:
        y = to_pm1(D["targets"][str(c)])
        ytr, yte = y[is_train], y[~is_train]
        cell_res = {"distance": dist[c], "ticks": {}}
        for t in ticks:
            X = np.asarray(D["states"][str(t)][str(c)], dtype=np.float64)
            Xtr, Xte = X[is_train], X[~is_train]
            acc = fit_eval(Xtr, ytr, Xte, yte)
            # shuffled-label control
            ytr_shuf = rng.permutation(ytr)
            acc_shuf = fit_eval(Xtr, ytr_shuf, Xte, yte)
            entry = {"linear_acc": acc, "shuffled_acc": acc_shuf,
                     "leakage_expected_chance": t < dist[c],
                     "n_features": int(X.shape[1])}
            # oracle control (only where oracle_bits provided)
            if "oracle_bits" in D and str(c) in D["oracle_bits"]:
                O = np.asarray(D["oracle_bits"][str(c)], dtype=np.float64)
                Otr, Ote = O[is_train], O[~is_train]
                entry["oracle_acc"] = fit_eval(Otr, ytr, Ote, yte)
            cell_res["ticks"][str(t)] = entry
        # constant majority baseline on heldout
        maj = max(float((yte > 0).mean()), float((yte < 0).mean()))
        cell_res["heldout_majority_baseline"] = maj
        result["cells"][str(c)] = cell_res

    # Compact table
    print(f"rows: train={result['n_train']} heldout={result['n_heldout']}")
    header = "cell dist | " + " ".join(f"t={t:<3}" for t in ticks)
    print(header)
    for c in cells:
        cr = result["cells"][str(c)]
        row = f"  {c}    {cr['distance']:>2}  | "
        row += " ".join(f"{cr['ticks'][str(t)]['linear_acc']*100:5.1f}" for t in ticks)
        print(row)
    print("\n(shuffled-label control should sit ~50%; oracle should be ~100%)")
    for c in cells:
        cr = result["cells"][str(c)]
        sh = " ".join(f"{cr['ticks'][str(t)]['shuffled_acc']*100:.0f}" for t in ticks)
        print(f"cell {c} shuffled: {sh}")
        if any("oracle_acc" in cr["ticks"][str(t)] for t in ticks):
            orc = " ".join(f"{cr['ticks'][str(t)].get('oracle_acc', float('nan'))*100:.0f}" for t in ticks)
            print(f"cell {c} oracle:   {orc}")
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
