# DeepSeek Agent Council Agenda: Titan Text Post-Campaign Deliberation

## Background & Empirical Facts
We have executed the full pre-registered Controlled Learning Campaign on Iterated Parity with Positional Readout (IPPR) across two matched arms (1000 epochs total, 5-seed evaluation per checkpoint, zero-boundary condition):

1. **Arm 1 (From-Scratch $L=16, \tau=16$)**:
   - Intact accuracy plateaued at 52.8% – 57.3% across all 1000 epochs.
   - Batch-shuffled accuracy remained pinned at chance: 49.2% – 50.5%.
   - Overall $G_{\text{identity}} = +2.8\% \text{ to } +7.2\%$.
   - **Crucial Per-Slot Asymmetry**:
     - Slot 0 ($k=0$, cell 3): 60.6% – 71.9% intact ($G_{\text{slot}}[0] = +7.5\% \text{ to } +17.5\%$).
     - Slot 1 ($k=1$, cell 7): 43.8% – 48.1% intact ($G_{\text{slot}}[1] = -3.8\% \text{ to } +2.5\%$).
     - Slot 2 ($k=2$, cell 11): 45.0% – 53.1% intact ($G_{\text{slot}}[2] = -6.2\% \text{ to } +2.5\%$).
     - Slot 3 ($k=3$, cell 15): 56.9% – 60.0% intact ($G_{\text{slot}}[3] = +6.2\% \text{ to } +10.0\%$).

2. **Arm 2 (Curriculum $L=8 \to L=16$)**:
   - **Stage 1 ($L=8, \tau=8$, 500 epochs)**:
     - Intact accuracy reached 68.8% vs 50.0% shuffled.
     - $G_{\text{identity}} = +18.8\% \pm 4.4\%$ (paired SEM), Cohen's $d = 1.90$ ($p < 0.001$).
     - **Both slots learned**: Slot 0 = 68.8% ($G = +17.5\%$), Slot 1 = 68.8% ($G = +20.0\%$).
     - **E1 Escalation Met**: Proves definitively that 1D NCA can learn instance-specific sequential parity when spatial-temporal horizon is compact.
   - **Stage 2 (Transfer to $L=16, \tau=16$, epochs 500–1000)**:
     - Slot 0 performance surged to 73.8% – 77.5% ($G_{\text{slot}}[0] = +21.2\% \text{ to } +25.0\%$). Curriculum transfer worked for Slot 0!
     - Slot 3 was preserved at 55.6% – 58.8% ($G = +7.5\% \text{ to } +10.0\%$).
     - But Slots 1 and 2 (the bulk interior) collapsed back to chance: 40.6% – 43.1% and 46.9% – 50.0%.

## Key Phenomenon: "Bulk Interior Stagnation"
Why did Chunk 1 succeed in $L=8$ (where it occupied cells 4..7, ending at the right boundary), but completely collapse in $L=16$ (where it occupied cells 4..7, but followed by cells 8..15)?

## Deliberation Topics for the Council
1. **Falsification & Epistemology**: What exact claims have been settled? What remains unfalsified?
2. **Dynamical Mechanism**: Why do boundary slots (0 and 3) learn while bulk interior slots (1 and 2) fail?
   - Hypothesis A: Spatial translational invariance without coordinate encoding makes interior chunks indistinguishable.
   - Hypothesis B: Gradient attenuation / diffusion vanishing through 16 sequential discrete cell hops.
   - Hypothesis C: Optimization landscape trap (early convergence on boundary slots starves interior gradients).
3. **Skeptical & Deflationary Audit**: Can the $L=8$ success be explained by an overlooked artifact (e.g. boundary padding reflection)?
4. **Next Discriminating Experiment**: What is the single most minimal, decisive, un-bloated experiment or intervention to test next?
