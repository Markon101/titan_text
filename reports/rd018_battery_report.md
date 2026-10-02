# RD-018 Battery Report: Auxiliary Deep Supervision at Interior Slots (L=16 Iterated Parity)

**Date**: 2026-10-01. Battery: `runs/rd018_aux_supervision/` (this directory).
**Preregistration**: FROZEN at commit `260878c` BEFORE launch (`reports/rd018_aux_supervision_protocol_draft.md`).
**Config**: exact replication of coordinate-campaign training (plain NCA, L=16, dev-steps 16, batch 8, 1000 epochs single-shot, `--zero-boundary`); seeds 901-915 (unused pool); alpha=0.3 frozen; sham targets constant through identical machinery.

## Divergence (recorded, never hidden)
- Arm A: 2/5 diverged (seeds 902, 905) — manifest train_accuracy=0.0, loss/grad JSON-null (NaN).
- Arm B: 1/5 diverged (seed 907).
- Arm C: 0/5 diverged.
- Diverged seeds are INVALID batteries, excluded from means, reported as data. Asymmetry (A worst, C clean) is itself observed and unexplained; small n, no claim.

## Primary endpoint (frozen): InteriorMean (slots 1,2 of 4), tau=16, batch 32
Converged seeds only:
- Arm A (n=3: 901,903,904): 42.2, 40.6, 50.0 → mean 44.27 pp
- Arm B (n=4: 906,908,909,910): 42.2, 45.3, 40.6, 48.4 → mean 44.14 pp
- Arm C (n=5: 911-915): 43.8, 32.8, 40.6, 40.6, 39.1 → mean 39.38 pp

**B − A = −0.13 pp** (frozen decision threshold: ≥ +8 pp). Permutation p = 1.0000 (two-sided).
C − A = −4.90 pp (p = 0.269, exploratory). B − C = +4.77 pp (exploratory, does not reach the A reference).
Full-lattice accuracy: A 54.7, B 53.3, C 50.6 — all near-chance regimes, replication of prior evidence.

## Verdict per frozen decision rule
- **H_OPT: NOT supported.** B does not exceed A (direction is null/negative); no +8pp; no seed-level win pattern (2/3 valid pairs). Carry-lesion counterfactual (contingent on a B win) is therefore NOT triggered.
- **H_ATTENUATION: PRELIMINARY DIRECTION ONLY — NOT YET CLAIMABLE.** B ≈ A ≈ C at chance is satisfied, but the mandatory positive controls are incomplete (see Gaps). No H_ATTENUATION claim is made this session.

## Key observed evidence (OBSERVED, not claim-level)
- Aux trajectories (Arm B, epoch-trace `aux=` column, mean interior-slot aux CE): declines sharply — seed 906: 0.693→0.104; 908: 1.296→0.214; 910: 0.851→0.159. The dedicated aux readouts learn TRUE intermediate parity from intermediate states.
- Arm C sham aux CE also declines (constant targets learnable, machinery active): 911: 1.939→0.461; 915: 1.111→1.482 (this one does not converge to the constant within 1000 epochs — OBSERVED).
- Interpretation boundary: aux decode success shows INTERMEDIATE-state parity is linearly decodable by a dedicated readout; it does NOT show the FINAL-state interior cells carry the information. That question is exactly control (b).

## Telemetry / implementation gaps vs prereg (honest record)
1. Prereg item 3 required per-arm manifest fields: aux loss trajectory + aux-head grad norm. NOT implemented; only the epoch-trace aux column exists. Control (a) (aux-head grad norms > 1e-4) is therefore only partially evaluable (behavioral aux-CE decline substitutes weakly; the numeric gate is unmet).
2. Prereg pitfall "grad clip 1.0 on aux terms" NOT implemented (global grad norm telemetry exists only).
3. Arm A replication gate ("interior at chance in ≥4/5 seeds") is compromised: only 3/5 seeds converged. The 3 converged seeds DO show interior at chance (40.6-50.0), but the gate as frozen is not cleanly met.

## Mandatory next steps (before any H_ATTENUATION claim)
1. Control (b): frozen-model linear probe on FINAL-state interior cells (L=16), 5 seeds/arm — does the information exist but go unexploited (H_OPT-flavored routing failure) or is it absent (H_ATTENUATION)?
2. Implement aux-head grad-norm telemetry (Option<f32> finite-filtered) and rerun 2 seeds as the cheap positive control (a), OR amend the record to note the control stands unevaluated.
3. DISC_2 (unitary/skew-symmetric recurrence) remains queued as the mechanistic follow-up if H_ATTENUATION survives controls.

## Artifacts
- Training: `runs/rd018_aux_supervision/{A_baseline,B_aux,C_sham}/seed_NNNN/` (manifest.json + model.safetensors)
- Evals: `{arm}_eval_{seed}.json` (intact) and `{arm}_eval_{seed}_lesion.json` (state lesion)
- Driver log: `driver.log`; per-run train logs with aux= columns: `{arm}_train_{seed}.log`
- Analysis: in-session (execute_code); statistics: exact permutation (pooled valid seeds, two-sided), no token-level pooling anywhere.
