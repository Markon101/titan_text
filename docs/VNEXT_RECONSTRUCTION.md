# Titan Text vNext: reconstruction and evidence audit

Date: 2026-09-27. Archaeology starting point: `97f9ec4` on
`exp/ippr-coordinate-channel`. This document separates recovered measurements
from their historical interpretations. It does not claim to reproduce training.
Source code, saved results, campaign runners, reports, recent Git history, and
one relevant prior Codex session summary were inspected. Existing artifacts are
preserved; their presence is not a blanket validation of every reported claim.

## What the repository already implements

The baseline is a token-aligned continuous field with shared local recurrent
updates, pointwise embeddings, and pointwise token projection. The ordinary
noncausal stencil has radius one. With zero boundaries, information cannot
traverse more than one cell per ordinary tick. The default configuration is
periodic, so sequence experiments must explicitly select zero boundaries.
See [config.rs](../src/config.rs), [nca.rs](../src/nca.rs),
[latent.rs](../src/latent.rs), and the
[causal geometry audit](ippr_causal_geometry_audit.md).

The inherited model is already a family of interventions:

- `feedback_mode=hierarchy`: a fine field plus one coarse field, configurable
  stride, update period, and channels; mean or Walsh downsampling; local macro
  recurrence; perception or state-derivative coupling; optional bistable drift.
- Explicit Carry Registers: stationary channels plus deterministic transported
  channels, bidirectional splitting, and slow/fast skip strides. This already
  separates transport from learned reaction.
- Carry quantization: `ste_round`, `ste_sign`, and continuous bistable modes.
- Global pooling and dual-timescale pooling: explicitly nonlocal controls.
- State normalization, damping, viscosity, causal stencils, coordinate-channel
  interventions, state lesions, batch shuffles, traces, and budget sweeps.

A new arbitrary-depth directional hierarchy is therefore an extension of this
research, not the first hierarchical field or ballistic transport implementation.
Its discriminating features should be explicit graph timing, odd-length-safe
scale mappings, transport on every scale, checkpointed configuration, and
scale-specific causal interventions.

Independent source review also verified two legacy training limitations:
`Trainer::train_step` uses `sample_train_batch_with_mask`, whose default stream
seed is always zero, so algorithmic training repeats one batch. `Trainer::new`
does not call `initialize_seeded`; CLI seed labels do not reproduce weight
initialization. Fresh processes may still produce independent random weights,
but the recorded seed alone is insufficient provenance. New substrate runs use
explicit initialization and advancing data streams; their absolute accuracies
must not be compared directly to the older training protocol.

## Compact-horizon IPPR: actual numbers and limits

The historical headline is recoverable from
[campaign_arm2_curriculum.json](../reports/campaign_arm2_curriculum.json),
epoch 200, L=8, tau=8 (the same headline recurs at epoch 500):

| Quantity | Raw artifact value | Statistical unit |
| --- | ---: | --- |
| Intact accuracy | 68.75%, SD 7.9672 percentage points | Five evaluation batches/seeds |
| Batch-shuffled accuracy | 50.00%, SD 6.6291 percentage points | Same evaluation seed battery |
| Paired identity gap | 18.75 percentage points | Paired evaluation results |
| Gap SD | 9.8821 percentage points | Five paired gaps |
| Gap SEM | 4.4194 percentage points | SD divided by sqrt(5) |
| Paired standardized effect | 1.8974 | Mean gap divided by gap SD |
| Per-slot intact accuracy | 68.75%, 68.75% | Slots 0 and 1 |
| Per-slot identity gaps | 17.50, 20.00 percentage points | Slots 0 and 1 |

The shuffled SD is **not zero**, contrary to the historical final report.
The paired gaps are `[3.125, 21.875, 15.625, 28.125, 25.0]`. Recalculation gives
t(4)=4.24264, two-sided Student-t p=0.0132356, and exhaustive two-sided sign-flip
p=0.0625. These are descriptive reanalyses of the saved batch outcomes, not
independent model-replication inference. The report's p=0.0006 is not reproduced
by either calculation.

[run_campaign.py](../scripts/run_campaign.py) trains one model using its default
seed 42 and evaluates seeds 42,101,202,303,404. Five evaluation seeds must not be
described as five independently trained models.

More importantly, the later [causal geometry audit](ippr_causal_geometry_audit.md)
reports only **11 distinct validation sequences** among the 64 possible L=8
binary observations. A local two-bit heuristic attains about 71.3% on that
validation support. At tau=2, slot 1 depends on positions 5 and 6 while all
earlier positions have zero measured influence. At tau=8, its sensitivity from
position 0 rounds to 0.000 with 0% prediction flips; positions 1 and 2 have
reported sensitivities 0.001 and 0.007. Rounded zero sensitivity is not a proof
of mathematical independence, but the evidence does not establish robust
prefix-wide computation.

**Qualified conclusion:** the checkpoint exhibits sample-dependent processing
and an intact-versus-shuffled gap. The reported accuracy does not distinguish
full prefix parity from a local shortcut on a tiny correlated validation set.
The strong claim that this result establishes sequential parity generalization
requires a balanced/exhaustive and counterfactual evaluation. This limitation
does not establish that a continuous NCA is incapable of parity computation.

## L=16 geometry and failed interventions

IPPR query cells are 3,7,11,15. Each prefix depends on input cell 0, giving
minimum radius-one arrival times 3,7,11,15 ticks under zero boundaries.
Insufficient-tick failure is a geometry fact. Failure after enough ticks is
about the learned computation and training procedure, not automatically a
theorem about the substrate.

| Recovered campaign | Observations | Limits on interpretation |
| --- | --- | --- |
| Original L16 learning | Near slot about 71–77%; interior slots around chance | One training seed; multiple evaluation seeds; chunk resumes |
| Dense intermediate supervision | At epoch1000, slots 71.88/44.38/50.00/56.88%; overall identity gap 7.19pp | Does not eliminate all optimization explanations; single trained seed42 and chunk resumes |
| Two-scale hierarchy, s=2 vs s=1 | 56.72±7.15% vs56.25±4.85%; difference +0.47pp, d=.07; s2 interiors45.62/45.62% | Genuine multiple training seeds in runner, but chunk-resume risk; only this hierarchy tested |
| Confirmatory scalar coordinate | Baseline interior45.31%; sham41.25%; coordinate45.63% | Rejects specified scalar-coordinate rescue, not every positional encoding |

Sources: [learning report](../reports/ippr_learning_campaign_final_report.md),
[dense-supervision report](../reports/escalation_b_final_report.md),
[hierarchy report](../reports/hierarchy_campaign_final_report.md), and
[confirmatory coordinate report](../reports/ippr_coordinate_channel_campaign.md).
The first two reports' wide mechanism claims exceed these observations.

The newest coordinate experiment is stronger evidence: five fresh training
seeds 801–805 for each of three arms, single-shot 1000-epoch runs, L16/tau16,
zero boundaries, 15/15 converged. Coordinate minus baseline is +0.31pp,
bootstrap CI[-3.12,+3.44], exact sign-flip p=.938. Coordinate minus constant
sham is +4.38pp, below the frozen +8pp threshold, p=.0625. The latter is the
two-sided exact-test floor at n=5, not evidence that there can be no effect.
Constant-sham underperformance complicates that contrast. Coordinate
counterfactual effects are inconsistent.

The same campaign uncovered a procedural confound: all five baseline runs
diverged under chained 100-epoch resumes; an identical single-shot seed801 run
converged. Adam state is not persisted. Invalid artifacts remain under
`runs/ippr_coordinate_chunked_invalid/`. Older incremental campaigns carry
unquantified resume-perturbation risk. A converged state lesion gives about
48–52% accuracy here, so historical zero-percent lesion results cannot be
generalized across checkpoints.

## Existing transport and discrete-field evidence

[The ECR report](../reports/ecr_campaign_final_report.md) already describes
upwind transport, bidirectional channels, and skip strides 2 and 4. It reports
L16 parity validation76.6% versus baseline71.9%; running-sum three-training-seed
validation57.8±11.8% (best70.3%); and bidirectional arithmetic58.7% versus49.4%.
These are historical measurements, mostly token accuracy, not newly reproduced
complete-answer or carry-specific results. The report's parameter table is
internally inconsistent for the Transformer (41,859 versus45,987). The claim
that long-length degradation is specifically floating-point rounding drift is
not discriminated from representation, optimization, or algorithm errors.

The discrete drift report and packet report L16 sign-quantized accuracy about
58.0±1.4%, and L64 about50.5±0.1%. Remaining at chance does not demonstrate useful
memory. The Dyck artifacts contain distance/depth-stratified carry lesions and
scrambles, useful controls to retain. They do not by themselves establish an
unbounded pushdown machine. Metric units, local baselines, and claimed universal
roundtrip bounds need independent audit before reuse as strong premises.

**Source-level concern discovered during this reconstruction:**
`NeuralCellularAutomaton::quantize_carry` computes the legacy sign mode as
`carry + (sign.detach() - carry)`, and rounding analogously. Algebraically,
the two carry derivatives cancel; the difference itself is not detached. This
is not the usual identity-gradient STE (`carry + (sign-carry).detach()`).
A focused backward test is needed to empirically verify this concern. Forward
quantization observations remain meaningful; historical STE optimization
interpretations require qualification. Any correction must preserve historical
mode/checkpoint behavior or be explicitly versioned and opt-in.

Legacy hierarchy uses floor division `l / stride` and truncates input before
restriction. Its prolongation path does not pad a remaining odd tail. New
scale mappings must explicitly test odd lengths and one-cell terminal scales.
This inspection is not a completed audit of all legacy hierarchy shapes.

## Interpretations that must not be promoted

- The hierarchy report equates Boolean parity with the spatial Nyquist mode.
  Boolean-cube Fourier degree and sequence-position spatial frequency are
  different notions. Its statement that Hamming weight is blind to parity is
  false: parity is Hamming weight modulo two. The measured null does not prove
  a universal spectral impossibility of continuous fields.
- Failed linear/MLP probes show failure of those fitted decoders, not absence
  of information from every possible state representation.
- A vanishing prediction-flip rate does not prove zero analog sensitivity;
  conversely, nonzero sensitivity does not prove successful computation.
- Norm invariance under a spatial roll is not bitwise state equality or
  invariance of task computation.
- More ticks failing to rescue a particular trained model does not rule out
  every learning algorithm or optimization regime.
- Finite-precision, finite-state hardware cannot establish an unbounded DPDA.

## Latest agent corrections and repository history

| Commit | Substantive change |
| --- | --- |
| `97f9ec4` | Confirmatory coordinate null report |
| `57f2a88` | Single-shot correction after chunk-resume divergence |
| `6670875` | Coordinate preregistration and training-time sham modes |
| `b8e58da` | Fresh-seed adaptive-halting negative replication |
| `c1a30d4` | Git-anchored halting confirmatory specification/correction |
| `0aa77dd` | Matched-compute halting qualification report |
| `1648306` | Delegated context-file/stdin contents transmitted verbatim |
| `5e4fc3d` | Semantic context condenser machinery |

The [halting qualification report](../reports/ascii_adaptive_halting_qualification.md)
overrides the earlier adaptive-superiority narrative. In its fresh-seed
701–710 confirmatory battery, adaptive edit similarity is .5904, fixed tau5
.6342, matched random .6343, and shuffled schedule .6104. Adaptive minus fixed5
is −.0438, seed-cluster CI[−.1144,+.0215]; adaptive minus shuffled is −.0200
with CI crossing zero. No detectable state-dependent advantage survives the
registered controls. Earlier significant inferiority of adaptive in the first
battery must not be relabeled statistically significant in the confirmatory
battery where its CI crosses zero. EOS truncation and compute matching matter.

The canonical packet still places old halting superiority in ESTABLISHED while
appending its qualification later. The claim ledger has a newer C-HALT-014
refutation, but older prose remains. Likewise, the latest coordinate report
correctly leaves optimization versus attenuation unresolved, conflicting with
the older dense-supervision report's claim of decisive optimization falsification.

## Earlier independent reviews and session provenance

Repository reviews were inspected, including
[2026-09-18 system audit](../reviews/2026-09-18-research-audit.md),
[2026-09-15 Codex audit](../reviews/2026-09-15-codex/review.md), and
[2026-09-11 audit](../reviews/2026-09-11/review.md). They document task-generator
shortcuts, right-copy confounds, misleading digit accuracy, incorrect halting
telemetry, untrained baseline evaluation, and instability. These are reasons
to preserve exact-answer and independently specified baseline checks.

A stored Codex session summary was also read:
`01a0c7bf-a9ec-7692-ad5e-48591798fb4c`, dated2026-09-22, available at
`~/.codex/memories/rollout_summaries/2026-09-22T06-13-41-Na8s-deepseek_gemini_jev_harness_hardening_self_audit.md`.
It points to the session JSONL under `~/.codex/sessions/2026/09/22/` and reports
bounded evidence delegation, advisory-only Jev judgments, and an incomplete live
harness self-audit. The session summary, not the entire trajectory, was read.
Its historical claim that the revised 1M-context/16,384-output retry did not
complete is not a claim about the current harness. Current transport fixes are
visible in later Git history. Model review completion requires nonempty final
content and `finish_reason=stop`; transport success is not scientific evidence.

## Highest-information intervention and gates

Compare A: legacy radius-one NCA, B: single-scale directional field (informed
by existing ECR), C: arbitrary-depth directional hierarchy, with the same task
generator, embeddings/readout, optimizer, and independent training seeds.
Expose parameter/state/FLOP differences rather than calling unlike budgets
matched. Use single-shot training.

Before interpreting task accuracy:

1. Verify geometry, odd lengths, boundaries, deterministic initialization,
   gradient flow, manifest serialization, and checkpoint round trips.
2. Measure impulse arrival against the actual synchronous graph. Report
   possible reach separately from trained useful transport.
3. For short IPPR, enumerate the input universe or use explicit balanced
   counterfactual pairs. Flip each required distant prefix bit while keeping
   the local suffix fixed. Include suffix-only and train-fitted majority controls.
4. Report per-slot/distance accuracy, complete-answer accuracy, identity gap,
   and state/scale diagnostics. Preserve failure artifacts.
5. Contrast C with B and with upward/downward exchange disabled. A task gain
   lost under a coarse-scale lesion is stronger evidence of scale use than
   decodability or a faster theoretical cone.

If tiny training is negative, retain structural transport evidence and leave
H-S2 through H-S8 unresolved. A shorter graph diameter alone is not evidence
of superior reasoning, memory, stability, or length generalization.

## Canonical state changes recommended

Replace top-level unsupported certainty rather than adding another contradictory
footer. Qualify compact IPPR as instance dependence with unresolved prefix
computation; retain the actual raw SD/SEM and evaluation-seed distinction.
Mark empirical local-influence limits as checkpoint/procedure-specific,
optimization versus attenuation unresolved, and old chunk-resumed campaigns
procedure-confounded. Replace old adaptive superiority entries with the
qualification/confirmatory negative. Add a separate discrete-gradient concern
pending a backward test. Preserve old reports with links to these corrections.
Register new vNext claims as hypotheses until corresponding controls pass.
