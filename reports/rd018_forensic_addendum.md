# RD-018 Forensic Addendum (2026-10-02)

This file CORRECTS and EXTENDS [rd018_battery_report.md](rd018_battery_report.md).
The historical report and its raw runs are preserved unchanged; nothing here
retroactively edits the executed result. Executed source is tagged `rd018-executed`
(= `3f83688`). All findings below are VERIFIED against source and immutable raw
artifacts; interpretation is labelled separately.

## 0. Reproducible tooling (new, no source change)
- `scripts/rd018_analysis.py` -> `reports/rd018_battery_analysis.json`
  (run table, artifact SHA-256, converged/failure-coding contrasts, exact
  permutation + independent-run bootstrap; unit = trained run).
- `scripts/rd018_staircase_analysis.py` -> `reports/rd018_staircase_analysis.json`
  (tick-resolved per-slot accuracy on the exhaustive 789-row heldout split).

## 1. Corrections to the historical report (VERIFIED)
1. **Aux "decodable intermediate parity" overclaim.** The report (line 28) says
   dedicated aux readouts learn true intermediate parity. FALSE: `aux_interior_loss`
   (`src/train.rs:117-129`) calls the SAME `interface.masked_cross_entropy_loss`
   on `interface.logits(...)` — the shared terminal projection (`src/vocab.rs:180`).
   There is NO dedicated aux head. The aux CE column is a TRAINING-batch quantity.
2. **Training is memorization, not generalization.** `train_step` ->
   `sample_train_batch_with_mask` -> `sample_batch_internal_with_mask` hardcodes
   stream seed 0 (`src/dataset.rs:159-167`): the SAME 8 observations every epoch.
   Train acc 0.94-1.0 with heldout interior at chance is the memorization signature.
3. **Seeds are not seeds.** `Trainer::new` (`src/train.rs:45-46`) builds a bare
   `VarMap`/`VarBuilder`; no `initialize_seeded`, and no `set_seed` anywhere in
   `src/main.rs`. `--seed` only feeds `horizon_mode=='randomized'` (unused here) and
   the manifest `random_seed` field. So 901-915 are UNCONTROLLED inits, not
   reproducible replicates. Run variation is candle init entropy.
4. **Eval batch was 32 rows, shared, seed 0** (`src/main.rs:1317-1324`): all arms
   evaluated the same 32 rows. Exhaustive 789-row heldout changes slot 0 from
   72-78% (32-row) to 65-70% and interiors from 40-50% to ~49% (see 2.3).
5. **STE claim in the prereg is wrong.** Protocol lines 18-22 say the zero-gradient
   claim was "REFUTED at source ... identity-gradient STE". The code is
   `carry + (round.detach() - carry)` (`src/nca.rs:406-411`): the carry terms cancel,
   gradient = 0. In-repo test `legacy_quantization_backward_is_zero_not_identity_ste`
   (`src/nca.rs:1735`) asserts exactly `[0.0, 0.0, 0.0]`. MOOT for RD-018
   (`carry_channels=0`) but a live documentation error.
6. **Prereg control (b) is impossible as written.** "frozen-model linear probe on
   final-state interior carry channels" — RD-018 has `carry_channels=0`
   (`src/config.rs:194`); there are no carry channels to probe.
7. **Sham target is a label-space mismatch.** `aux_sham` uses token id 1
   (`src/train.rs:123-125`) = `<bos>` in `Vocab::new_ascii` (`src/vocab.rs:22-43`),
   NOT digit '1' (id 21). It is a constant label, but not the one the prose names.

## 2. New empirical findings (VERIFIED, exhaustive 789-row heldout)
2.1 **Primary null holds on solid ground.** InteriorMean at t=16:
    A 49.75, B 48.99 (both ~= chance) -> **B-A = -0.76pp**. Aux does not help.
2.2 **The sham arm is CONTAMINATED.** C_sham interior = **44.04** (CI 43-45,
    excludes chance): the constant-BOS aux actively CORRUPTS the interior readout
    below chance. Therefore **B-C = +4.94pp is driven by C's damage, not B's
    benefit.** The prereg "bias-absorbable sham" premise is violated empirically.
2.3 **Slot 0 is the only generalizing slot** (65-70% exhaustive) and it rises
    GRADUALLY (A: t3=38 -> t7=56 -> t11=69), NOT in the causal light-cone staircase
    (which would be slot0@t3, slot1@t7, slot2@t11, slot3@t15). No staircase ->
    no evidence of clean prefix-parity-by-propagation.
2.4 **Readout-class artifact at t=0.** A and C readout emits a non-parity class at
    query cells (exactly 0% acc, balanced targets), while B emits a parity bit (50%)
    — the shared projection was trained to emit parity labels.
2.5 **Lesion is arm-categorical (anomaly).** Under `--lesion-state` (sweep path
    zeroes the update delta every tick -> state frozen at the token embedding):
    A and C give exactly 0.00 interior (predicting a non-target class, high CE),
    B gives 0.40-0.53 (at/around chance). The "collapse to chance" shortcut guard
    is therefore NOT satisfied for B and the counterfactual is ambiguous.

## 3. Statistical validity (independent blind reconstruction, consistent)
- Converged contrasts reproduce exactly: B-A -0.13pp (perm p=1.0000), C-A -4.90pp
  (p=0.2679), B-C +4.77pp (p=0.119). (32-row eval basis.)
- **No principled pairing exists** (disjoint seed pools 901-905/906-910/911-915):
  the preregistered "seed-level sign-flip + paired bootstrap" is NOT computable.
  Any paired number is arbitrary; the 4/5 paired win gate cannot be claimed.
- Failure-coding (diverged=0) yields B-A = **+8.75pp**, but this is produced solely
  by differential divergence counts (A 2, B 1), NOT by a treatment effect. It is an
  explicitly labelled sensitivity assumption, never an intention-to-treat analysis.
- p-values are exploratory/conditional: endpoints are missing differentially by arm
  (treatment-dependent missingness) and labels were never randomized to runs.

## 4. Cross-path lesion inconsistency (VERIFIED in frozen source)
`apply_to_delta` (which implements `disable_recurrent`) is called ONLY in the
latent/sweep path (`src/latent.rs:237`). `develop_with_intervention`
(`src/nca.rs:828`, used by rollout/benchmark) applies only `apply_to_state`
(reset/noise/ablate/shuffle) and never `apply_to_delta`. So the same flag is a true
(delta-zeroing) lesion in `sweep` and a NO-OP in `rollout`. Not fixed (executed tree
frozen); recorded as a defect to version explicitly.

## 5. What remains defensible / suspended
DEFENSIBLE:
- The executed RD-018 result is a real null on the converged 32-row endpoint and is
  confirmed on the exhaustive 789-row endpoint (B-A ~ -0.8pp).
- The model memorizes one fixed 8-row batch and generalizes only slot 0.
- Aux deep supervision, as implemented (shared head, partly unrecoverable targets),
  does not improve the terminal interior.
SUSPENDED (do NOT claim):
- H_OPT "not supported" as a mechanistic conclusion. The design cannot separate
  H_OPT from H_ATTENUATION: both predict heldout chance under memorization.
- H_ATTENUATION in any form. Positive controls (a) aux-head grad norm, (b) a valid
  final-state probe are unmet/impossible as specified; a failed decoder is never
  proof of absent information.
- Any xi~4-6 transport-limit claim. At t=16 every query target is geometrically
  reachable (distances 3/7/11/15 <= 16); the substrate is not information-bounded.
- DISC_2 (unitary recurrence) stays DEFERRED: the null cannot select the mechanism.

## 6. Next experiment (pending research-team decision)
Ranked, cheapest first:
1. **Tick-resolved staircase on exhaustive heldout** (DONE, 2.3): no staircase;
   treat as the geometric diagnostic baseline.
2. **Frozen-state probe** (new read-only subcommand or offline export): record
   x_t at query cells {3,7,11,15} for t in {3,4,7,8,11,12,15,16} on all 789 heldout
   rows; fit constant / linear / 2-layer MLP probes FIT ON TRAIN, EVAL ON HELDOUT,
   with (i) oracle positive control (probe on true light-cone bits -> ~100%),
   (ii) shuffled-label control (-> chance), (iii) leakage validator (probe at
   t < distance(q) must be chance). Distinguishes "info present but unused by the
   trained decoder" (H_OPT-flavored) from "not recoverable by this probe class".
3. **Corrected RD-018b** (new prereg, new identity, own seeds): explicit seeded
   init; advancing data stream (or a representative balanced subset of the 3307
   train rows); aux only at light-cone-recoverable (tick,cell) targets; dedicated
   aux head; aux-head grad-norm telemetry; exhaustive heldout eval; drop or
   explicitly arm the carry-channel probe.

## Literature verification (2026-10-02)
All cited arXiv IDs independently verified present with matching topics (two
sources: direct web fetch of each abs page + local `agy`/Gemini cross-check). No
fabrication. See `docs/lit_review_interior_stagnation_2026.md`.
