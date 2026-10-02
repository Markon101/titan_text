# RD-018 Research-Team Decision Record (2026-10-02)

7-role council (Arbiter, Designer, Dynamics, Rust-Audit, Statistical, Ideation,
Skeptical) deliberated candidates A-E with the forensic context packet. PI synthesis.

## Decision: run **B first (frozen-state probe, read-only)**, and begin **A's
protocol fixes in parallel** (they are needed regardless of B's outcome).

### Why B is the correct FIRST move
- It is nearly free (1 read-only export + offline probes) and requires **no
  retraining and no change to the frozen tree**.
- It is **asymmetrically informative**: a POSITIVE result is decisive. If interior
  parity is decodable from the frozen state at t=16 while the trained decoder
  outputs chance, that is direct evidence of "information present but unused" —
  an H_OPT-flavored routing/decoder failure, not attenuation. That single result
  would reframe the entire thread.
- A negative result is weak but cheap: it does not block A and costs only a few
  minutes.

### The strong objection the team raised (must be handled)
The RD-018 checkpoints **memorized 8 fixed rows**, so their states on the 789
heldout rows are **off-manifold**. A probe failure there is doubly ambiguous
(memorized model AND failed-decoder-is-not-absence). Therefore B is ONLY
interpretable if run with:
1. **Oracle positive control** — probe on the true light-cone input bits must reach
   ~100% (validates the harness).
2. **Shuffled-label control** — must be at chance (validates no leakage).
3. **Leakage validator** — probe at t < distance(q) must be at chance; above-chance
   there implies leakage/memorization and invalidates the run.
4. **Within-model dissociation** — compare slot 0 (which DOES generalize, 65-70%)
   against interiors (chance) in the SAME model. Even off-manifold, "slot0 decodable
   & interior not" is a meaningful dissociation.
5. Probes **fit on train rows, evaluated on the 789 heldout rows** (never fit/eval
   on the same rows), low-capacity first (constant/linear), then a 2-layer MLP.

### Why A is the necessary SECOND move (started in parallel)
Without explicit seeded initialization and an advancing data stream over the ~3307
train rows, **no run is reproducible** and memorization confounds every downstream
claim. A is the foundation that makes any mechanistic test interpretable. Its
implementation (seeded init + advancing stream + dedicated aux head + aux grad-norm
telemetry + exhaustive heldout eval) is worth starting now because B's outcome only
sharpens, never removes, the need for it.

### Explicitly NOT chosen now
- **C (tick sweep)** — already done for slot0/interiors (see staircase analysis); the
  absence of a light-cone staircase is recorded. Marginal further value.
- **D (dedicated aux head + light-cone-gated targets)** — presupposes we know what
  intervention we are testing; folded into A's corrected design instead.
- **E (DISC_2 unitary recurrence)** — remains **DEFERRED**. The null cannot select a
  mechanism, so changing the recurrence operator now would be premature.

## Claim licensing (frozen before running B)
- B positive (interior decodable at t=16, controls pass): licenses "the final state
  carries linearly-decodable interior parity that the trained terminal decoder does
  not exploit." Does NOT license "H_OPT is true" (that needs an intervention showing
  exploiting it fixes accuracy) nor "H_ATTENUATION is false".
- B negative (interior not decodable, controls pass): licenses only "not recoverable
  by THIS probe class from THIS memorized model's states." Does NOT license "the
  information is absent" or any xi/transport claim.
- Either way, the only route to a real H_OPT vs H_ATTENUATION separation is A's
  clean, seeded, non-memorizing RD-018b — and even that requires a valid final-state
  probe (A must add one; the prereg's carry-channel probe is impossible as written).

## Housekeeping
- Executed source + original runs stay immutable (tag `rd018-executed`=3f83688).
- Probes/export run on existing checkpoints read-only; any code change is versioned
  (new subcommand) and never overwrites the historical training path.
- Run-level statistics only; the 4/5 paired-win gate stays unclaimed (disjoint pools).
