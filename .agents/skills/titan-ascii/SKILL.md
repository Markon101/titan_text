---
name: titan-ascii
description: Train, sample, evaluate, archive, and export Titan Text ASCII-generation experiments, including untouched raw generations and reproducible sampling metadata.
---

# Titan Text ASCII Generation Skill

This skill governs training, autoregressive sampling, objective structural metrics evaluation, artifact preservation, and external Android export for the Titan Text ASCII generation campaign.

## 1. Directory Structure Standards

Every sampling and evaluation run must be archived immutably under:
`runs/ascii/<RUN_ID>/`

Contents:
- `raw/`: Untouched `.txt` raw model generations (naming convention: `sample_<INDEX>_seed<SEED>_tau<TAU>.txt`).
- `metadata/`: Per-sample JSON metadata (exact prompt, seed, tau, temperature, token IDs, metrics).
- `manifest.jsonl`: One JSON line per generated sample for indexing and search.
- `config.json`: Run configuration (checkpoint, vocabulary version, hyperparameters, hardware environment).
- `metrics.json`: Aggregate quantitative metrics (collapse rates, diversity, symmetry, nearest-neighbor edit distance).
- `stdout.log`: Console logging and execution traces.
- `README.md`: Summary of the run, findings, and sample inspection notes.
- `gallery/` (optional): Presentation copies for visual review. Gallery samples must reference their source raw sample ID.

## 2. Quantitative Structural Metrics

ASCII generation is evaluated on empirical measurements:
- **Valid Character Ratio**: Fraction of characters within allowed vocabulary.
- **Dimensional Properties**: Line count, mean line width, max line width, non-whitespace density.
- **Structural Diversity**: Row diversity, column diversity, unigram and bigram entropy.
- **Collapse Detection**: Consecutive identical character repetition rate, identical line repetition rate.
- **Symmetry**: Rough horizontal and vertical reflection symmetry scores.
- **Memorization vs Novelty**: Levenshtein edit similarity to nearest training corpus item, exact training match count.

## 3. Causal Latent Recurrence Sweep

For every generation battery, compare outputs under:
- Latent tick sweep: $\tau \in \{0, 1, 2, 4, 8, 16\}$
- Recurrent ablation / state lesion
- Identical prompts and fixed seeds (`42, 101, 202, 303, 404`)

## 4. Adaptive Recurrent Compute & Halting

Evaluate per-token dynamic recurrence allocation against matched controls:
- **Adaptive Halting**:
  `cargo run --release -- generate --load-dir checkpoints/ascii_v1 --prompt "<BOX>\n" --halting adaptive --tau-min 1 --tau-max 16 --halting-metric relative_delta --halting-threshold 0.35 --halting-patience 2`
- **Supported Halting Metrics**:
  - `relative_delta`: relative state velocity $\|x_t - x_{t-1}\|_2 / (\|x_t\|_2 + \epsilon)$ (recommended).
  - `state_delta`: unnormalized $L_2$ state norm.
  - `logit_delta`: maximum coordinate change across vocabulary logits.
  - `entropy_delta`: predictive entropy delta $|H(P_t) - H(P_{t-1})|$.
  - `cosine`: cosine distance $1 - \cos(x_t, x_{t-1})$.
- **Controls**:
  - **Arm B (Fixed Sweep)**: `--tau 1`, `--tau 2`, `--tau 4`, `--tau 8`, `--tau 16`, and `--lesion-state`.
  - **Arm C (Shuffled Schedule)**: `--tau-schedule "2,4,3,2,1..."` (preserves multiset budget, breaks state alignment).
  - **Arm D (Random Sham)**: `--halting random --tau-min 1 --tau-max 16`.

## 5. External Android Shared Storage Export

Export completed runs to external shared storage for accessible viewing outside Termux:
```bash
python3 scripts/export_ascii_run.py runs/ascii/<RUN_ID> --dest /sdcard/Download/TitanText/ascii_runs
```

Requirements:
- Preserves raw files exactly.
- Verifies file counts and SHA-256 hashes.
- Leaves repository originals intact.
- Fails clearly if Termux storage permission is absent.
