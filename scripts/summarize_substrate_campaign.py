#!/usr/bin/env python3
"""Portable evidence summary; raw predictions/checkpoints stay under runs/."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in {"predictions", "exact_outcomes", "git_diff"}}
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, float):
        assert math.isfinite(value), "nonfinite report value"
    return value


def mean_sd(values):
    return {"mean": statistics.mean(values),
            "sd_across_training_seeds": statistics.stdev(values) if len(values) > 1 else None,
            "n_training_seeds": len(values), "values": values}


def graph_arrivals(length, scale_count):
    sizes = [length]
    while len(sizes) < scale_count and sizes[-1] > 1:
        sizes.append((sizes[-1]+1)//2)
    times = [[None]*n for n in sizes]
    times[0][0] = 0
    queue = [(0, 0)]
    for scale, pos in queue:
        neighbors = [(scale, pos-1), (scale, pos+1)]
        if scale+1 < len(sizes):
            neighbors.append((scale+1, pos//2))
        if scale:
            neighbors.extend([(scale-1, 2*pos), (scale-1, 2*pos+1)])
        for s, p in neighbors:
            if 0 <= p < sizes[s] and times[s][p] is None:
                times[s][p] = times[scale][pos]+1
                queue.append((s, p))
    return times


def summarize(roots):
    campaigns, receipts, aggregates, impulses = [], [], {}, []
    for root in roots:
        config = json.loads((root/"config.json").read_text())
        provenance = json.loads((root/"provenance.json").read_text())
        results = json.loads((root/"summary.json").read_text())
        assert len(results) == len(config["arms"])*len(config["seeds"]), "incomplete campaign"
        campaigns.append({"root": str(root), "config": config,
                          "provenance": compact(provenance), "results": compact(results)})
        for path in sorted(root.rglob("*")):
            if path.is_file():
                receipts.append({"path": str(path), "bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for r in results:
            run_dir = root / f'{r["arm"]}_seed_{r["seed"]}'
            assert r["roundtrip_max_abs_error"] == 0
            for e in r["evaluations"]:
                actual = sum(e["intact"]["exact_outcomes"])/e["intact"]["examples"]
                assert actual == e["intact"]["exact_accuracy"]
                key = (config["name"], r["arm"], e["length"], e["ticks"])
                aggregates.setdefault(key, []).append({"seed": r["seed"], "e": e})
            for probe in json.loads((run_dir/"transport.json").read_text()):
                theory = graph_arrivals(probe["length"], r["substrate"]["scales"])
                observed = [[None]*len(x) for x in theory]
                max_outside = 0.0
                for row in probe["trace"]:
                    tick = row["tick"]
                    for s, scale in enumerate(row["scales"]):
                        families = scale["family_max_abs_by_position"]
                        for pos in range(len(theory[s])):
                            effect = max(f[pos] for f in families)
                            if tick < theory[s][pos]:
                                max_outside = max(max_outside, effect)
                            if effect > 1e-7 and observed[s][pos] is None:
                                observed[s][pos] = tick
                assert max_outside == 0, f"causal shortcut in {run_dir}"
                impulses.append({"run": str(run_dir), "length": probe["length"],
                                 "threshold": 1e-7, "graph_earliest": theory,
                                 "empirical_earliest": observed,
                                 "max_effect_outside_graph": max_outside})
    groups = []
    for key, rows in sorted(aggregates.items()):
        seeds = [r["seed"] for r in rows]
        assert len(seeds) == len(set(seeds)), "duplicate training seed/arm"
        name, arm, length, ticks = key
        groups.append({"campaign": name, "arm": arm, "length": length, "ticks": ticks,
                       "seeds": seeds,
                       "query_accuracy": mean_sd([r["e"]["intact"]["query_accuracy"] for r in rows]),
                       "exact_accuracy": mean_sd([r["e"]["intact"]["exact_accuracy"] for r in rows]),
                       "identity_gap": mean_sd([r["e"]["identity_gap"] for r in rows])})
    return {"schema": "titan-substrate-evidence-v1", "campaigns": campaigns,
            "aggregates": groups, "impulse_arrivals": impulses, "artifact_receipts": receipts,
            "limitations": ["Exploratory training seeds are the statistical units; common heldout rows are not independent model replicas.",
                            "Raw checkpoint/prediction files are local ignored artifacts; receipts do not make them remotely available.",
                            "Arrival above a numerical threshold is not useful information transport.",
                            "Fixed trained horizon with post-training sweeps does not estimate optimal training tau_required."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    report = summarize(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"wrote {args.output}: {len(report['campaigns'])} campaigns, {len(report['artifact_receipts'])} verified receipts")
