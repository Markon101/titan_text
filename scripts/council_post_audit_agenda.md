# Adversarial DeepSeek Council: Post-Audit Causal Reconstruction Agenda

## The Decisive Empirical Evidence from the Causal Influence Map

1. **Light-Cone Conformance**:
   - In `checkpoints/campaign_arm2_curriculum/step_0200` ($L=8$), at $\tau=2$, Query Slot 1 (cell 7) has strictly $0.0000$ numerical sensitivity to positions 0, 1, 2, 3, 4. There is no non-local shortcut or wrap-around.
   - The observed $71.3\%$ accuracy at $\tau=2$ was driven entirely by local chunk-1 bits ($b_5, b_6$), which exhibit finite-sample correlation with the cumulative target on the 11-sequence validation split of $L=8$.

2. **The Local Receptive Field Trap on $L=16$**:
   - In `checkpoints/campaign_arm1_from_scratch/step_1000` ($L=16$), even after 1,000 epochs and at $\tau=16$:
     - Each query slot $q_k = 4k+3$ has developed a symmetric local receptive field of radius $R \approx 3-4$ cells.
     - Slot 0 (cell 3) reads $[0..6]$, containing its prefix bits $b_0, b_1, b_2$ (solves Slot 0 at $77.5\%$).
     - Slot 1 (cell 7) reads $[4..10]$ (its own chunk and the next chunk to the right!). Its sensitivity to $b_0, b_1$ is $\le 0.012$ ($0.0\%$ flip probability). It has virtually zero connection to Chunk 0!
     - Slot 2 (cell 11) reads $[8..14]$. Its sensitivity to Chunks 0 and 1 is strictly $0.000$.
     - Slot 3 (cell 15) reads $[12..15]$.
   - The translation-invariant NCA update fails to establish directional left-to-right transport of prefix parity across chunks.

## Reformulated Hypotheses for Deliberation

- **$H_{\text{POS}}$ (Positional / Symmetry Deficit)**: Translation-invariant convolution without coordinate encoding makes interior chunks indistinguishable, preventing cells from establishing directed convection/transport from left to right.
- **$H_{\text{TRANSPORT}}$ (Signal Attenuation / Diffusion Dilution)**: The continuous NCA updates diffuse locally; transporting a 1-bit parity signal over 11–15 hops via continuous $\tanh$ integration decays below the noise floor without discrete latching or global feedback.
- **$H_{\text{GEOMETRY}}$ (Receptive Field Geometry)**: The model naturally minimizes loss by learning a symmetric local radius-$R$ smoother rather than a directed pipeline.
- **$H_{\text{READOUT}}$ (Readout & Loss Masking)**: Since readout is pointwise and identical across cells, and loss is only supervised at query positions $4k+3$, intermediate data cells are not explicitly incentivized to act as communication repeaters.
- **$H_{\text{OPT}}$ (Optimization Landscape)**: Early convergence on the immediate chunk bits creates a local minimum where extending the receptive field across chunk boundaries yields no immediate loss improvement.

## Council Directives
Each panelist must perform recursive self-questioning:
- "What assumption am I currently treating as fact?"
- "What observation is hardest for my hypothesis to explain?"
- "What competing mechanism generates the same spatial pattern?"
- "What experiment would make my interpretation fail?"
- "What result would distinguish positional information from signal transport?"
