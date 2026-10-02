# Titan Text vNext — Final Report (2026-10-02)

Branch `exp/titan-vnext`, forked from `rd018-audited` (RD-018 + audited lineage
fully preserved: tags `rd018-executed`=3f83688, `rd018-audited`; original runs
untouched). Three commits: `99c6245` substrate+Run0, `8235446` RFT mechanism,
`328daf1` stability fix.

## Outcome
**A clean negative that reframes the program** — but not the negative we expected.
Relay-Fold Transport did not rescue interior slots, and the reason is more
fundamental than transport: **the tiny NCA cannot learn iterated parity from a
genuine data stream at all**, so the transport hypothesis cannot yet be tested on
a working baseline.

## What was BUILT (all committed, all tested)
1. **Clean experimental substrate** (201 tests pass): seeded weight init (same
   seed -> bit-identical weights), advancing data stream (no memorization),
   exhaustive 789-row heldout eval with per-slot accuracy, gradient clipping,
   per-tick aux telemetry with correct post-update timing, dedicated aux readout
   head. This removes every confound the RD-018 forensic audit identified.
2. **Relay-Fold Transport (RFT) mechanism** (NCA-native): carry channels with a
   learned relay-FOLD write `c_i = g*c_{i-k} + (1-g)*tanh(z)`, g=sigmoid(Wg.[c_{i-k};h_i]),
   z=tanh(Wc.[c_{i-k};h_i]) — each cell folds its local perception into the
   incoming carried accumulator (real fold, not advection). Closed-form unit test
   PROVES the fold accumulates far-field while pure additive transport is blind to
   interior bits. Plus: uniform-k transport speed (real speed knob), identity-STE
   modes (fixes the dy/dcarry=0 gradient-annihilation bug the critique found),
   launch dead-zone clamp. Drivable via `--carry-quantization fold`.
3. **Tooling**: `probe_frozen_state.py`, `vnext_battery.sh` (T=8 speed ablation),
   `vnext_dynamics_analysis.py` (onset + influence-front falsifier).

## What the EXPERIMENTS say
### Run 0 — frozen-state diagnostic (STRONG, decisive)
On the frozen RD-018 checkpoint (zero training): **cumulative parity is at chance
at interior cells for BOTH linear and MLP probes (0.43-0.55), while LOCAL parity
is decodable everywhere (0.9-1.0)**; the causal influence matrix shows interior
cells have ~0 influence from the early bits they need. => the accumulated value is
**ABSENT from interior state, not merely unread** — a real transport/relay failure,
readout failure ruled out. This vindicated building a transport mechanism.

### vNext battery — RFT falsified on the endpoint, but CONFOUNDED
T=8 speed ablation (k in {0,1,2,4}), 4 seeds/arm, clean substrate, exhaustive
heldout: interior accuracy at T=8 = baseline 50.1 / k1 50.1 / k2 50.0 / k4 49.9
— ALL CHANCE, no staircase onset at any budget, influence fronts show interior
far-prefix flip probability 0.00% for every arm. By the frozen falsifier RFT is
falsified. **BUT the baseline does not even learn slot 0 (53.4%, chance), so the
battery cannot cleanly attribute the null to transport** — nothing learns.

### Isolation — the deeper, reframing finding
The clean substrate (seeded + advancing data) yields all-chance predictions at
EVERY tick budget, even at T=16 and even at 5000 epochs and lr 0.001 (slot0 53.4%,
interior ~50). Yet the RD-018 memorization checkpoint (same 8 rows every epoch)
reached slot0 65-70% on heldout. Conclusion: **the 44k-param plain NCA memorizes
but cannot learn the iterated-parity rule (not even slot 0's 3-bit parity) from a
genuine advancing data stream.** The prior "slot0 generalizes" result was a
byproduct of the fixed-batch memorization regime, not evidence of learned local
computation.

## What is DEFENSIBLE vs SUSPENDED
DEFENSIBLE:
- Run 0: interior cumulative parity is absent from the frozen RD-018 state
  (transport/relay failure), readout failure ruled out. [strong]
- RFT mechanism is mechanically correct: the fold accumulates (unit-proved),
  uniform-k transport reaches k cells/tick, identity-STE has nonzero gradient.
- The clean substrate is real and reproducible (seeded, non-memorizing).
- Finding: this tiny NCA does not learn iterated parity from a data stream; the
  earlier slot0 result was a memorization artifact. [reframes the program]
SUSPENDED (do NOT claim):
- That transport is or is not the binding limit — UNTESTABLE until a baseline
  learns the base task. RFT's endpoint null is confounded by under-training.
- Any H_OPT/H_ATTENUATION selection (still unresolved; DISC_2 still deferred).
- That RFT "doesn't work" — it was never tested against a working baseline.

## Strongest next move
The bottleneck is now **genuine learnability, not transport**. Priority order:
1. **Make the base task learnable** on the clean substrate: a curriculum (master
   slot0 3-bit parity first, then cumulative), larger capacity, or a better recipe
   (the model needs to learn a 3-bit XOR before any transport question is live).
   This is the gate: without a baseline that learns slot0 AND stalls at interior,
   transport cannot be tested.
2. **Then** re-run the RFT speed ablation against that working baseline, measuring
   the influence-front onset (the transport signature) — the mechanism and battery
   are already built and waiting.
3. Run 0's frozen-state probe (linear/MLP cumulative-parity) should be re-run on
   any new baseline to confirm the transport/relay gap persists once learning
   happens.

## Honest caveat
We set out to show a transport mechanism changes how the NCA computes. We did not
get that. What we got instead is a sharper, more fundamental negative: the system
does not learn the task once memorization is removed, which invalidates the way the
whole prior thread interpreted "slot0 generalizes." That is a real result, and a
better one to have than a fake transport win.
