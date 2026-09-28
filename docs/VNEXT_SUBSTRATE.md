# Directional multiscale substrate v1

Protocol frozen before training on 2026-09-27. Exploratory, not confirmatory.
Read [reconstruction](VNEXT_RECONSTRUCTION.md) before interpreting older claims.

## Intervention and its limits

The repository already has two-scale NCA feedback, Explicit Carry Registers,
bidirectional skip transport, and quantized carry. This fork adds arbitrary-depth
synchronous dyadic geometry and separately switchable transport, reaction, and
cross-scale exchange. It preserves every legacy update and parameter name.

At scale s, shape is `[batch, ceil(L/2^s), C]`, stopping at length one or the
configured maximum scale count. The C channels divide equally into M,L,R,X.
M and X stay put; L and R shift exactly one cell per tick with zero inflow and
absorbing outflow. A shared pointwise tanh MLP supplies a gated residual reaction
after that shift. The labels M/X describe transport allocation, not learned
memory/function claims. There is no hard separation preventing reaction from
rewriting transport or memory channels.

Upward exchange concatenates ordered child states, pads an absent odd child
with zeros, and applies a shared gated C-channel projection. Downward exchange
maps a parent to 2C channels and reshapes into ordered children, cropping only
the absent odd child. Each exchange reads the OLD states: no instantaneous
bottom-to-top-to-bottom cascade. Coarse states start at zero, with no free input
pooling. All output cells are clamped elementwise to ±4 after each tick. Initial
embeddings are not clamped. There are no spatial/batch norms or global summaries.
Restriction is a learned bottleneck, not a lossless encoder.

One tick crosses at most one same-scale edge or one parent/child edge. A graph
with sufficiently many scales has logarithmic diameter, but a fixed scale cap
eventually returns to linear growth. The protocol cap is 16, adequate for the
tested lengths. Dyadic child order introduces a spatial reference and breaks
one-cell translation equivariance. This is part of the topology intervention,
and a future shifted-origin control is needed to separate alignment benefits.
Bidirectional mode is suitable for masked algorithmic tasks; this version does
NOT implement causal streaming or autoregressive hierarchy caches.

The directional shifts are a discrete advection control, not a proven PDE solver
or a learned conservative wave equation. Discrete state is deliberately excluded
from this first topology comparison. The legacy `ste_sign`/`ste_round` backward
regression records their actual zero gradient; it does not silently repair them.
An eventual continuous/discrete factorial needs a newly versioned correct STE.

## Matched arms and budget disclosure

All use the existing ASCII vocabulary (99 tokens), 32-channel embedding/readout,
CPU f32, the same task generator, data streams, AdamW settings, and readout.

| Arm | Recurrent rule | Allocated / active parameters | L16 state scalars | Dense MACs/tick/example |
|---|---|---:|---:|---:|
| A | Legacy radius-one NCA, hidden80, step0.5, GELU, zero boundaries, no viscosity/carry/feedback | 19,379 / 19,379 | 512 | 204,800 |
| B | Directional, one scale, hidden48, reaction0.2 | 19,539 / 11,155 | 512 | 73,728 |
| C | B plus learned exchange0.2 at every adjacent scale | 19,539 / 19,539 | 992 | 265,728 |

B retains dormant exchange parameters so B/C initial tensors are identical.
Those 8,384 dormant parameters are not active capacity. MACs exclude nonlinear
operations, biases, readout, transport, backward, and optimizer; FLOPs=2*MACs.
A/B differs in reaction as well as transport. Thus B>A is a family comparison,
not exclusive proof that shifting caused the gain. A matched trained
transport-disabled arm remains needed. C/B differs in active capacity and state;
runtime and causal lesions must accompany accuracy. No fairness claim rests
solely on allocated parameter counts.

## Registered hypotheses and stop gates

* H-S1: hierarchy shortens causal paths. Structural test only; useful transport
  requires later sample-specific task evidence.
* H-S2/S3: stagnation partly reflects transport and ballistic channels preserve
  useful identity. Unresolved; loss/accuracy plus bit-pair controls required.
* H-S4: lower required ticks. Report tau_required only if a model reaches both
  90% query and 80% complete-answer accuracy on the fixed heldout panel. No
  threshold crossing means unmeasured, not infinity or an inferred exponent.
* H-S5: coarse states are used. Examine no-up, no-down, no-exchange, scale-reset
  and scale-shuffle effects. Distribution-shift damage alone is insufficient.
* H-S6: unseen-length performance. Report geometry extrapolation separately from
  instance generalization. Delayed-recall length changes can repeat the same
  source payload; these are not necessarily unseen algorithms or instances.
* H-S7: discrete state. Deferred to a separate versioned factorial.
* H-S8: reduced long-horizon drift. Clamping guarantees a bound after one tick;
  measured boundedness is not evidence of learned stability or useful memory.

1. Structural gate: shapes including odd lengths; exact shifts and no wrap;
   synchronous graph cone; finite nonzero gradients in every transport tensor;
   seeded repeatability; exact checkpoint and legacy trajectory equivalence.
2. Tiny optimization gate: four fixed L8 training rows, seed42, 200 uninterrupted
   steps, tau8, lr.003, weight decay.01. Target ≥95% exact on that training batch.
   Heldout L8 has only11 unique rows, so cannot establish generalization.
3. Single-seed smoke: L16, tau16, 300 steps, batch16; inspect before expanding.
4. Bounded replication: seeds42,101,202 only when useful and affordable. Five-seed
   confirmatory claims are explicitly outside this first exploratory result.
5. IPPR is primary. DelayedRecall is the second, different task; all arms train
   at tau16, sufficient for its source-to-query distance12. Evaluate L16/L32 at
   tau0,1,2,4,8,16,32. Fixed train horizon is not optimization over tau_required.

Every run uses explicit `baselines::initialize_seeded` initialization and the
unchanged TaskEngine content-hash split. Training seed10000+step advances the
data stream; tiny mode fixes seed10000. Unique heldout panels use seed9000001.
Report train diversity and verify no full-input overlap. Do not compare the
new fresh-batch harness numbers directly to legacy repeated-batch training.
Prefix-bit counterfactuals require BOTH members in heldout evaluation and report
their actual denominators; no pair at L8 means no measurement. Query scoring
uses mask>0.5, excluding dense auxiliary labels. Deranged final-state/readout
identity controls measure sample dependence, not necessarily prefix computation.
Repeated scale derangements can cycle information; null effects do not prove
non-use. Up/down/reset controls are complementary.

## Reproduction and provenance

`cargo build --release --locked --offline --bin titan_substrate`

`target/release/titan_substrate experiments/substrate_v1/tiny.json runs/substrate_v1/tiny`

Use the IPPR/recall JSON files for subsequent campaigns. The runner refuses an
existing output directory. All training is single-shot; Adam moments are not
resumed. Existing ModelManifest/SafeTensors storage is reused, with typed optional
`config.substrate` metadata. Old manifests default to None. Legacy commands
reject new substrate manifests rather than silently selecting a different model.
Every checkpoint is reloaded and required to reproduce final logits exactly.

Each run stores initial/final checkpoint hashes, configs, full input panels,
predictions, per-query and complete outcomes, identity gaps, lesion outputs,
bit-pair controls, gradient/scale diagnostics, impulse traces, wall time and
budget estimates. Git commit/status and backend are stored at campaign start.
Commit implementation before measurements so untracked source cannot escape
the provenance record. Raw output/checkpoints remain under ignored `runs/`;
tracked report summaries and content hashes provide the portable record.

## Relevant literature (background only)

[Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) studies gated
convolutional recurrent algorithm learning. It is a precedent for separating
algorithm learning from attention, not evidence for Titan's competence.
[MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612) explicitly addresses
message-passing distance through coarse meshes. Its physical simulation results
motivate a geometry control; they do not establish that hierarchy solves parity,
preserves discrete state, or generalizes in this repository. These source-checked
abstracts support background scope only; no theoretical result is imported into
the Titan claim ledger.
