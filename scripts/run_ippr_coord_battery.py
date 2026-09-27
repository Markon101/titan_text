#!/usr/bin/env python3
"""IPPR coordinate-channel confirmatory battery (preregistration commit 6670875).

Three arms x five fresh seeds (801-805), L=16, iterated-parity, 1000 epochs:
  Arm A: baseline (no coordinate channel)
  Arm C: SHAM  (--coord-channel --coord-train-mode constant)   [run before B]
  Arm B: COORD (--coord-channel, intact coordinates)
Predetermined execution order: A -> C -> B. Baseline replication gate after A.

Per 100-epoch increment: intact eval at tau=16 (training curve).
At completion: full battery (intact, lesion-state, batch-state shuffle,
coordinate counterfactuals for Arm B) at tau=16 with paired eval seed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

BIN = "./target/release/titan_text"
PREREG_COMMIT = "6670875"
SEEDS = [801, 802, 803, 804, 805]
ARMS = {
    "A_baseline": [],
    "C_sham": ["--coord-channel", "--coord-train-mode", "constant"],
    "B_coord": ["--coord-channel"],
}
COUNTERFACTUALS = {
    "B_coord": [
        ("coord_zeroed", ["--coord-mode", "zeroed"]),
        ("coord_shuffled", ["--coord-mode", "shuffled"]),
        ("coord_reversed", ["--coord-mode", "reversed"]),
        ("coord_constant", ["--coord-mode", "constant"]),
    ],
    "C_sham": [("coord_intact_none", [])],
}


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def train_run(arm: str, seed: int, out_dir: str, epochs: int = 1000, chunk: int = 0):
    """Single-shot training (chunk=0): NO incremental resume. The chunked-resume
    procedure was found to destabilize training (Adam optimizer state is not
    persisted across resume; all 5 fresh-seed baseline runs diverged under
    chunking while an identical single-shot run converged cleanly — archived at
    runs/ippr_coordinate_chunked_invalid). Divergence is still detected
    explicitly from the final manifest and recorded per prereg sec 7."""
    save_dir = os.path.join(out_dir, arm, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        BIN, "train", "--task", "iterated-parity", "--save-dir", save_dir,
        "--epochs", str(epochs), "--dev-steps", "16", "--seq-len", "16",
        "--seed", str(seed), "--zero-boundary",
    ] + ARMS[arm]
    code, out = run_cmd(cmd)
    if code != 0:
        print(f"[FAIL] train {arm} seed {seed}:\n{out[-2000:]}")
        sys.exit(2)
    with open(os.path.join(save_dir, "manifest.json")) as fh:
        m = json.load(fh)
    diag_vals = [m.get(k) for k in
                 ("train_loss", "train_accuracy", "grad_norm", "state_energy")]
    diverged = any(v is None or (isinstance(v, float) and v != v) for v in diag_vals) \
        or m.get("train_accuracy") == 0.0
    if diverged:
        print(f"[DIVERGED] {arm} seed {seed} (null/nonfinite diag: {diag_vals}; "
              f"recorded per prereg sec 7)")
    return save_dir, diverged


def sweep_eval(ckpt: str, seed: int, extra: list[str], out_root: str, seq_len: int = 16) -> dict:
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "sweep.json")
        cmd = [
            BIN, "sweep", "--load-dir", ckpt, "--budgets", "16",
            "--seq-len", str(seq_len), "--batch-size", "32", "--seeds", str(seed),
            "--output", f, "--zero-boundary",
        ] + extra
        code, out = run_cmd(cmd)
        if code != 0:
            print(f"[WARN] sweep {ckpt} {extra} -> {code}: {out[-800:]}")
            return {}
        with open(f) as fh:
            return json.load(fh)


def budget16(d: dict) -> dict:
    if not d:
        return {}
    for entry in d.get("budgets", []):
        if entry.get("latent_ticks") == 16:
            return entry
    return {}


def slots_of(b16: dict) -> list[float]:
    acc = b16.get("per_slot_accuracy")
    if acc is not None:
        return [round(100.0 * a, 2) for a in acc]
    a = b16.get("accuracy")
    return [round(100.0 * a, 2)] if a is not None else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ippr_coordinate")
    ap.add_argument("--arms", default="A_baseline,C_sham,B_coord")
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {
        "prereg_commit": PREREG_COMMIT,
        "seeds": SEEDS,
        "arms": list(ARMS),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": {},
    }
    git = run_cmd(["git", "rev-parse", "HEAD"])[1].strip()
    manifest["git_head"] = git

    for arm in args.arms.split(","):
        if arm not in ARMS:
            print(f"unknown arm {arm}")
            sys.exit(1)
        for seed in SEEDS:
            t0 = time.time()
            ckpt, diverged = train_run(arm, seed, args.out, epochs=args.epochs)
            final = {}
            if not diverged:
                conds = {"intact": [], "lesion_state": ["--lesion-state"],
                         "shuffle": ["--lesion-shuffle"]}
                for name, extra in list(conds.items()):
                    d = sweep_eval(ckpt, seed, extra, args.out)
                    b16 = budget16(d)
                    final[name] = {"budget16": b16, "slots": slots_of(b16)}
                for name, extra in COUNTERFACTUALS.get(arm, []):
                    d = sweep_eval(ckpt, seed, extra, args.out)
                    b16 = budget16(d)
                    final[name] = {"budget16": b16, "slots": slots_of(b16)}
            record = {
                "arm": arm, "seed": seed, "checkpoint": ckpt, "diverged": diverged,
                "train_seconds": round(time.time() - t0, 1),
                "conditions": final,
            }
            path = os.path.join(args.out, f"{arm}_seed_{seed}.json")
            with open(path, "w") as fh:
                json.dump(record, fh, indent=1)
            manifest["runs"][f"{arm}/{seed}"] = {
                "json": path, "diverged": diverged,
                "intact_slots": final.get("intact", {}).get("slots"),
                "seconds": record["train_seconds"],
            }
            if diverged:
                print(f"[DIVERGED-RECORDED] {arm} seed {seed}")
            else:
                print(f"[OK] {arm} seed {seed}: intact {final['intact']['slots']} "
                      f"({record['train_seconds']}s)")
            # Baseline replication gate (prereg sec 5): check after EVERY baseline seed
            if arm == "A_baseline" and not diverged:
                s = final["intact"]["slots"]
                if len(s) >= 4:
                    interior = (s[1] + s[2]) / 2.0
                    if interior > 58.0:
                        print(f"[GATE] Arm A seed {seed} InteriorMean={interior:.2f} "
                              f"exceeds chance+8pp -> baseline replication gate may FAIL. "
                              f"Continuing battery (decision enforced in analysis).")
            with open(os.path.join(args.out, "battery_manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=1)

    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "battery_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("[DONE] battery manifest:", os.path.join(args.out, "battery_manifest.json"))


if __name__ == "__main__":
    main()
