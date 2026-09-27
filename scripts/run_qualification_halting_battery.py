#!/usr/bin/env python3
"""Qualification control battery for adaptive halting (frozen protocol).

Implements reports/ascii_halting_qualification_protocol.md:
Arms: A_adaptive, B_fixed_tau4, B_fixed_tau5, B_fixed_tau8,
      C_matched (per-(prompt,seed) multiset permutation of same-battery Arm A),
      D_matched (per-(prompt,seed) i.i.d. draws from the frozen adaptive tick
      distribution {4:641, 5:529, 6:30}, length = Arm A length, rng seed+7777).
Plus: one null-lesion sanity run and one Arm A determinism rerun.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SEEDS = [501, 502, 503, 504, 505, 601, 602, 603, 604, 605]
PROMPTS = [("<BOX>\n", "box"), ("<MAZE>\n", "maze"), ("<DIAMOND>\n", "diamond")]
CHECKPOINT = "checkpoints/ascii_v1"
CHECKPOINT_SHA256 = "97d00bd0085052e0ddd82799ffc0fb9451e523d7bd6243e1db7898a62ec13e3e"
MAX_LEN = 48
TEMPERATURE = 0.7
HALTING_METRIC = "relative_delta"
THETA = 0.25
PATIENCE = 2
TAU_MIN = 1
TAU_MAX = 16
# Frozen adaptive tick distribution (held-out Arm A, N=1200 tokens):
TICK_DIST = {4: 641, 5: 529, 6: 30}


def run_generate(bin_path: str, args: List[str], tag: str) -> Dict[str, Any]:
    tmp = Path(f"scratch/qual_tmp_{tag}.json")
    tmp.parent.mkdir(exist_ok=True)
    cmd = [bin_path, "generate", "--load-dir", CHECKPOINT,
           "--format", "json", "--output", str(tmp)] + args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"{tag}: exit {proc.returncode}: {proc.stderr[:500]}")
    data = json.loads(tmp.read_text())
    tmp.unlink()
    return data


def base_args(prompt: str, seed: int) -> List[str]:
    return ["--prompt", prompt, "--max-len", str(MAX_LEN),
            "--temperature", str(TEMPERATURE), "--seed", str(seed)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="reports/raw/qualification_battery")
    ap.add_argument("--bin", default="./target/release/titan_text")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seed list; default = protocol seeds 501-605")
    args = ap.parse_args()
    global SEEDS
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = args.bin

    # Verify checkpoint identity (protocol failure condition)
    import hashlib
    h = hashlib.sha256(Path(CHECKPOINT, "model.safetensors").read_bytes()).hexdigest()
    if h != CHECKPOINT_SHA256:
        print(f"FATAL: checkpoint hash mismatch: {h}", file=sys.stderr)
        return 2

    manifest = {
        "timestamp_unix": int(datetime.now(timezone.utc).timestamp()),
        "protocol": "reports/ascii_halting_qualification_protocol.md",
        "checkpoint": CHECKPOINT, "checkpoint_sha256": h,
        "max_len": MAX_LEN, "temperature": TEMPERATURE,
        "halting_metric": HALTING_METRIC, "theta": THETA, "patience": PATIENCE,
        "tau_min": TAU_MIN, "tau_max": TAU_MAX,
        "tick_dist_frozen": TICK_DIST,
        "seeds": SEEDS, "prompts": [p[0] for p in PROMPTS],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
    }

    samples: List[Dict[str, Any]] = []

    # ---- Arm A (first: schedules are paired to it) ----
    print(">>> Arm A: adaptive (frozen theta)")
    arm_a: Dict[tuple, Dict[str, Any]] = {}
    for prompt_text, prompt_name in PROMPTS:
        for seed in SEEDS:
            s = run_generate(bin_path, base_args(prompt_text, seed) + [
                "--halting", "adaptive", "--tau-min", str(TAU_MIN),
                "--tau-max", str(TAU_MAX), "--halting-metric", HALTING_METRIC,
                "--halting-threshold", str(THETA),
                "--halting-patience", str(PATIENCE)], f"a_{prompt_name}_{seed}")
            s.update(arm="A_adaptive", prompt_name=prompt_name)
            samples.append(s)
            arm_a[(prompt_name, seed)] = s
        print(f"  done {prompt_name}")

    # Determinism check: rerun one cell, must match bit-for-bit
    key = ("box", SEEDS[0])
    rerun = run_generate(bin_path, base_args("<BOX>\n", SEEDS[0]) + [
        "--halting", "adaptive", "--tau-min", str(TAU_MIN), "--tau-max", str(TAU_MAX),
        "--halting-metric", HALTING_METRIC, "--halting-threshold", str(THETA),
        "--halting-patience", str(PATIENCE)], f"rerun_{SEEDS[0]}")
    det_ok = (rerun["raw_output"] == arm_a[key]["raw_output"]
              and rerun["token_traces"] == arm_a[key]["token_traces"])
    manifest["determinism_rerun_ok"] = det_ok
    rerun.update(arm="A_adaptive_rerun_check", prompt_name="box")
    samples.append(rerun)
    print(f"  determinism rerun: {'OK' if det_ok else 'MISMATCH -> FATAL'}")
    if not det_ok:
        json.dump({"manifest": manifest, "samples": samples},
                  open(out / "qualification_samples.json", "w"), indent=2)
        return 3

    # Null-lesion sanity run
    les = run_generate(bin_path, base_args("<BOX>\n", SEEDS[0]) + [
        "--tau", "4", "--lesion-state"], "lesion_sanity")
    les.update(arm="sanity_lesion", prompt_name="box")
    samples.append(les)
    lesion_ok = les["total_ticks"] == 0
    manifest["lesion_sanity_ok"] = lesion_ok
    print(f"  lesion sanity: {'OK' if lesion_ok else 'UNEXPECTED TICKS'}")

    # ---- Fixed arms ----
    for tau in (4, 5, 8):
        print(f">>> Arm B: fixed tau={tau}")
        for prompt_text, prompt_name in PROMPTS:
            for seed in SEEDS:
                s = run_generate(bin_path, base_args(prompt_text, seed) + [
                    "--tau", str(tau)], f"b{tau}_{prompt_name}_{seed}")
                s.update(arm=f"B_fixed_tau{tau}", prompt_name=prompt_name, fixed_tau=tau)
                samples.append(s)
        print(f"  done tau={tau}")

    # ---- C_matched: per-(prompt,seed) multiset permutation (rng seed+9999) ----
    print(">>> Arm C_matched: per-sequence multiset permutation")
    for prompt_text, prompt_name in PROMPTS:
        for seed in SEEDS:
            a = arm_a[(prompt_name, seed)]
            ticks = [t["ticks_used"] for t in a["token_traces"]]
            shuffled = list(ticks)
            random.Random(seed + 9999).shuffle(shuffled)
            sched = ",".join(map(str, shuffled))
            s = run_generate(bin_path, base_args(prompt_text, seed) + [
                "--tau-schedule", sched], f"c_{prompt_name}_{seed}")
            s.update(arm="C_matched", prompt_name=prompt_name,
                     schedule_seed=seed + 9999, schedule_len=len(ticks),
                     source_arm_a_total_ticks=a["total_ticks"],
                     schedule=sched)
            samples.append(s)
        print(f"  done {prompt_name}")

    # ---- D_matched: i.i.d. draws from frozen distribution (rng seed+7777) ----
    print(">>> Arm D_matched: distribution-matched random schedule")
    pop = [t for t, w in TICK_DIST.items() for _ in range(w)]
    for prompt_text, prompt_name in PROMPTS:
        for seed in SEEDS:
            a = arm_a[(prompt_name, seed)]
            L = len(a["token_traces"])
            rng = random.Random(seed + 7777)
            sched = [rng.choice(pop) for _ in range(L)]
            s = run_generate(bin_path, base_args(prompt_text, seed) + [
                "--tau-schedule", ",".join(map(str, sched))], f"d_{prompt_name}_{seed}")
            s.update(arm="D_matched", prompt_name=prompt_name,
                     schedule_seed=seed + 7777, schedule_len=L,
                     source_arm_a_total_ticks=a["total_ticks"],
                     schedule=",".join(map(str, sched)))
            samples.append(s)
        print(f"  done {prompt_name}")

    json.dump(samples, open(out / "qualification_samples.json", "w"), indent=2)
    manifest["total_samples"] = len(samples)
    json.dump(manifest, open(out / "qualification_manifest.json", "w"), indent=2)
    print(f"\nSaved {len(samples)} samples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
