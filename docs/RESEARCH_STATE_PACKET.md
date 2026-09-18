# Canonical Research-State Packet: Titan Text
**Laboratory for Recurrent Neural-Cellular Latent Dynamics**  
*Document Version: 2.0 · Date: 2026-09-18 · Repository: `titan_text`*

---

## 1. ESTABLISHED (Supported by Definite Evidence)

1. **Causal Necessity of Recurrent Updates for Output Generation**:
   - On `TaskKind::IteratedParity` ($B=32, L=16$), ablating latent updates entirely (Zero-Tick $\tau=0$ or State Lesion `--lesion-state`) collapses task accuracy from $53.28\% \pm 3.42\%$ directly to $0.00\% \pm 0.00\%$ and expands loss from $1.4249$ to $4.7768$ nats ($+3.35$ nats, $t(4) = 34.80, p = 4.07 \times 10^{-6}$, paired Cohen's $d = 15.56$).
   - Reversing the recurrence update vector ($\Delta x \to -\Delta x$ via `--lesion-gain -1.0`) destroys accuracy ($0.0\%$) and causes catastrophic loss divergence ($14.1282$ nats, $t(4) = -168.69, p = 7.41 \times 10^{-9}, d = -75.44$).
   - *Conclusion*: Recurrence is physically load-bearing; the readout head is calibrated strictly to transformed continuous hidden states.

2. **Genuine Instance-Specific Recurrence Proven on Compact Horizons ($L=8, \tau=8$)**:
   - In Arm 2 Stage 1 (Curriculum Stage 1 on $L=8, \tau=8, B=32$), Titan Text achieves:
     - Checkpoint 200: Intact Acc $68.8\% \pm 8.0\%$, Shuffled Acc $50.0\% \pm 0.0\%$.
     - **Identity Gap**: $G_{\text{identity}} = +18.8\% \pm 4.4\%$ (paired SEM), Cohen's $d = 1.90$.
     - Replicated at Checkpoint 500: Intact $68.8\%$, Shuffled $50.0\%$, $G_{\text{identity}} = +18.8\% \pm 4.4\%$, $d = 1.90$.
     - **Per-Slot De-composition on $L=8$**:
       - Slot 0 ($k=0$, cell 3): Intact $68.8\%$ vs Shuffled $51.3\%$ ($G_{\text{slot}}[0] = +17.5\%$).
       - Slot 1 ($k=1$, cell 7): Intact $68.8\%$ vs Shuffled $48.8\%$ ($G_{\text{slot}}[1] = +20.0\%$).
   - *Conclusion*: **Falsification of H2 (Parity Impossibility)**. 1D Neural Cellular Automata *can* and *do* perform genuine instance-specific sequential recurrent parity computation without specialized discrete memory gates. Both prefix-adjacent and distant slots carry sample-specific sequential information across latent ticks. Escalation criterion **E1** is decisively met on $L=8$.

3. **Bulk Interior Stagnation on Extended Horizons ($L=16, \tau=16$)**:
   - In both Arm 1 (From-Scratch 1000 epochs) and Arm 2 Stage 2 (Curriculum 500 epochs transfer from $L=8$ to $L=16$):
     - **Slot 0** ($k=0$, cell 3): Robustly solved! Intact accuracy reaches $71.2\% - 77.5\%$ (vs $50\%$ shuffled, $G_{\text{slot}}[0] = +16.2\% \text{ to } +25.0\%$). Curriculum transfer specifically enhances Slot 0 by $+6\% - +8\%$ over from-scratch.
     - **Slot 3** ($k=3$, cell 15): Moderately preserved at $55.6\% - 60.0\%$ (even with `--zero-boundary`), $G_{\text{slot}}[3] = +7.5\% \text{ to } +10.0\%$.
     - **Bulk Interior Slots (Slot 1 at cell 7, Slot 2 at cell 11)**: Complete collapse to chance across all 10 checkpoints in both arms:
       - Arm 1 Slot 1: $43.8\% - 48.1\%$ (mean $G_{\text{slot}} \approx -0.5\%$).
       - Arm 1 Slot 2: $45.0\% - 53.1\%$ (mean $G_{\text{slot}} \approx -1.0\%$).
       - Arm 2 Slot 1: $40.6\% - 43.1\%$ (mean $G_{\text{slot}} \approx -2.2\%$).
       - Arm 2 Slot 2: $46.9\% - 51.2\%$ (mean $G_{\text{slot}} \approx -3.6\%$).
     - Overall intact accuracy on $L=16$ remains capped at $52.8\% - 57.3\%$ ($G_{\text{identity}} \approx +3\% - +7\%$).
   - *Conclusion*: Extending spatial length from 8 to 16 causes a selective computational bottleneck in interior bulk cells, while boundary-adjacent cells retain strong recurrent parity capability.

4. **Hard Light-Cone Constraint of 1D NCA Radius-1 Stencil**:
   - In 1D NCA (`src/nca.rs:70-94`), perception is local (`roll_spatial(x, -1)` and `roll_spatial(x, 1)`).
   - In $\tau$ ticks, information propagates at most $\tau$ lattice positions in each direction.
   - For query at token index $4k+3$ requiring input from token 0, the dependency distance is $d = 4k+3$.
   - Causal reachability demands $\tau \ge 4k+3$. At $\tau=4$, slots 1 ($d=7$), 2 ($d=11$), and 3 ($d=15$) are physically unreachable from prefix bits.

5. **Trajectory Divergence Under Over-Unrolling in Fixed Horizon Training**:
   - Without multi-horizon supervision or contractive damping, unrolling beyond the training depth $\tau > \tau_{\text{train}}$ causes monotonic drift away from the trained readout distribution.

---

## 2. SUPPORTED BUT NOT ESTABLISHED (Plausible Interpretations)

1. **Positional Ambiguity in Interior Cells (Spatial Coordinate Deficit)**:
   - In 1D NCA, convolution weights are spatially invariant (shared across all lattice cells).
   - Boundary cells (cells 0..3 and 12..15) experience asymmetric padding from `--zero-boundary` ($x[-1]=0$ and $x[L]=0$), providing implicit absolute positional coordinates.
   - Bulk interior cells (cells 4..11) receive identical local environments and lack explicit positional encodings, making it ambiguous whether a cell is executing Chunk 1 ($k=1$) or Chunk 2 ($k=2$).
2. **Gradient Attenuation / Diffusion Dilution Over Long Spatial Chains**:
   - In $L=8$, information travels at most 7 cells in 8 ticks.
   - In $L=16$, propagating prefix parity to Chunk 2 and Chunk 3 requires multi-hop diffusion across 11 to 15 cells. Without skip-connections, macro feedback, or coordinate channels, continuous updates may suffer diffusion dilution or gradient vanishing.

---

## 3. MECHANISTIC HYPOTHESES (Awaiting Decisive Testing)

- **Hypothesis H4 (Positional Coordinate Indispensability for Bulk Recurrence)**:
   Adding an explicit coordinate channel $p_i = 2i/(L-1) - 1$ to the NCA perception vector will break translational invariance for interior cells, allowing cells 4..11 to differentiate their computational roles and solve Slot 1 and Slot 2.
- **Hypothesis H5 (Dual-Timescale / Global Feedback Coupling)**:
   Enabling macro recursive feedback (`--feedback-mode dual_timescale` or `global_pool`) provides a global recurrence shortcut, bypassing the $O(L)$ spatial light-cone delay and synchronizing distant bulk chunks.
- **Hypothesis H6 (Intermediate Curriculum Granularity)**:
   Stepping from $L=8$ directly to $L=16$ doubled the spatial problem. An intermediate step ($L=12, \tau=12$, 3 chunks) may allow inductive generalization before scaling to $L=16$.

---

## 4. FALSIFIED / SUPERSEDED (Explicitly Invalidated Claims)

1. **FALSIFIED: "1D NCA Cannot Learn Instance-Specific Sequential Parity"**:
   - *Falsification*: Decisively falsified by Arm 2 Stage 1 ($L=8, \tau=8$), where $G_{\text{identity}} = +18.8\% \pm 4.4\%$ ($d=1.90$) and both slots achieved $68.8\%$ intact vs $50\%$ shuffled.
2. **SUPERSEDED: "The $v_0$ Checkpoint Proves Instance-Specific Latent Computation"**:
   - *Falsification*: Shuffled batch controls on $v_0$ yielded $p = 0.388, d = 0.43$.
3. **FALSIFIED: "Extended Training Alone Resolves $L=16$ Interior Slots"**:
   - *Falsification*: 1000 epochs of training (Arm 1 and Arm 2) left Slot 1 and Slot 2 pinned strictly at chance ($40\% - 50\%$), with zero identity gap. Extended compute without architectural or curriculum modulation is insufficient.

---

## 5. OPEN QUESTIONS & IMMEDIATE NEXT STEPS

1. Why did Chunk 1 succeed in $L=8$ (where it was adjacent to the right boundary at cells 4..7) but completely fail in $L=16$ (where it was embedded in the bulk at cells 4..7)?
2. Does injecting a fixed spatial coordinate gradient into the input embedding or NCA perception solve Bulk Interior Stagnation?
3. Does `--feedback-mode dual_timescale` (macro recursive feedback) enable information transfer across bulk cells?
4. What does the DeepSeek Agent Council recommend as the most parsimonious, hypothesis-driven next experiment?
