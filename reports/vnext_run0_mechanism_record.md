# Titan Text vNext — Run 0 Diagnostic & Mechanism Record

Branch `exp/titan-vnext` (forked from `rd018-audited`). Substrate rebuilt clean
(seeded init, advancing data stream, exhaustive 789-row heldout eval, grad clip,
per-tick aux telemetry, dedicated aux head; 185 tests pass).

## Run 0 — Frozen-state diagnostic (ZERO training, decisive)

Run on the frozen RD-018 checkpoint `A_baseline/seed_901` with existing tooling
(`influence`, `probe`) — no new code.

### Result A: latent representation probe (readout-vs-transport discriminator)
| probe | slot0 (c3) | slot1 (c7) | slot2 (c11) | slot3 (c15) |
|---|---|---|---|---|
| Linear **Local** parity | 0.984 | 0.945 | 0.906 | 1.000 |
| Linear **Cumulative** parity | 0.984 | 0.453 | 0.477 | 0.547 |
| MLP **Cumulative** parity | 0.969 | 0.430 | 0.484 | 0.539 |
| Linear Chunk0 parity | 0.984 | 0.516 | 0.539 | 0.508 |

**Interpretation (VERIFIED):** local parity is linearly decodable at EVERY cell
(0.9-1.0), so local bits are present in the interior state. But cumulative parity
is at CHANCE at interior cells for BOTH linear and MLP probes (0.43-0.55). An MLP
failing to recover it means this is NOT a linear-readout limitation — the
accumulated prefix value is genuinely ABSENT from the interior state.

### Result B: causal influence matrix (transport front)
Each query cell's numerical influence is confined to a ~±3-cell neighborhood:
- Query c7: influence at pos 4,5,6 (12.6, 24.4, 49.1) but ~0 at pos 0,1,2
  (0.004, 0.022, 1.005; FlipPr 0%, 0%, 1.6%). It needs bits 0,1,2 for cumulative
  parity — they NEVER reach it.
- Query c11: influence peaks at 8-13, ~0 at 0-7 (which it needs).
- Query c15: influence at 12-14 only, ~0 at 0-11.
- `Reach: YES` for all positions is GEOMETRIC reachability (T=16 allows it), but
  the actual learned numerical influence is ~0 at distance -> the transport is
  missing, not the light cone.

### Verdict
**Run 0 supports the TRANSPORT/RELAY diagnosis and rules out a pure readout
failure.** Cumulative parity information does not reach interior query cells
(absent from state, not merely unread). Therefore a transport/relay mechanism is
warranted. Critique's refinement stands: the target is an ACCUMULATOR that must be
**re-folded at every relay**, so the mechanism must be a relay-FOLD, not pure
advection of a stored value.

## Mechanism under test: Relay-Fold Transport (RFT)
NCA-native (local update + deterministic transport + local nonlinear fold). At
each cell, a dedicated carry channel is updated by folding locally-seen bits into
the incoming accumulated carry as it is advected past at speed k (cells/tick):
`c_i <- fold(local_bits_i, incoming_carry_{i-k})`, fold = learned tanh·sigmoid
gated combination. This accumulates prefix parity along the transport path instead
of ferrying a stale value.

### Design constraints from adversarial critique (must honor)
1. **Fix the STE gradient bug** (`src/nca.rs:401-412`): `carry + (rounded.detach() -
   carry)` gives dy/dcarry = 0 whenever carry_quantization != none. Use continuous
   carry (`none`, full gradient) for RFT, or add correct identity-STE modes
   `carry + (rounded - carry).detach()`. Never use legacy ste_sign/ste_round for
   anything that must learn.
2. **Uniform transport speed k**: legacy `carry_skip_stride` applies k to only HALF
   the carry channels (slow/fast split). RFT needs uniform k so the speed ablation
   is a real speed knob, not a mixture.
3. **Launch dead-zone**: `k_left_neighbor` zero-pads cells 0..k-1, so nothing
   launches from the first k cells. Predicted arrival ≈ k + ceil((q-k)/k). Account
   for this or the null is a geometry artifact.
4. **Test where transport BINDS**: at T=16 every s covers q=15 (s=1 reaches in 15),
   so T=16 cannot attribute gains to transport. Run the battery at **T=8** where
   slot2 (11 cells) needs k>=2 and slot3 (15 cells) needs k>=3 to arrive in time.
5. **Falsifier metric = influence-front onset**, not accuracy alone: W(k,q) = first
   tick where flipping far-prefix bits changes slot q's prediction. RFT predicts
   W moves ~1/k with the launch term; a local shortcut predicts W arm-invariant.
6. Keep k << L (NCA-native). If k scales with L, it is a disguised global copy —
   a category violation, not a result.

## Falsification battery (see scripts/vnext_battery.sh, updated for T=8)
Arms {no-carry baseline; RFT k=1; RFT k=2; RFT k=4} x seeds {921..}, single-shot
1000-epoch, seeded init + advancing stream + exhaustive 789-row heldout, eval at
budgets {4,6,8,12,16}. Single falsifier: interior staircase onset + interior
accuracy at T=8. If k=4 interior at T=8 stays at chance AND onset is
arm-invariant, RFT is falsified -> clean negative on transport-at-speed-k.
