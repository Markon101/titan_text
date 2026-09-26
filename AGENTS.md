# Titan Text: Agent Working Agreement & Operational Guidelines

*Repository*: `titan_text`  
*Core Framework*: Pure Rust with `candle-core` & `candle-nn` on Linux/ARM64 (Termux)  
*Primary Focus*: Recurrent neural-cellular sequence modeling, discrete pushdown dynamics, and structured generative text.

---

## 1. Core Operating Principles

1. **Evidence-First Development**:
   - Every claim must be supported by empirical artifacts, code inspections, or execution outputs.
   - Distinguish capability demonstrations (e.g. producing plausible ASCII) from mechanistic claims (e.g. recurrence is causally necessary for closure).
   - Negative results are valuable data; never hide or silently discard failed runs.
   - Never cherry-pick single favorable generations. Maintain both `raw/` (unfiltered evidence) and `gallery/` (presentation).

2. **Causal & Mechanistic Rigor**:
   - For any recurrent model claim, execute causal ablations:
     - Latent tick sweep ($\tau \in \{0, 1, 2, 4, 8, 16\}$).
     - Intact recurrence vs zero ticks vs state lesion (`--lesion-state`) vs gate ablation (`--lesion-gates`).
   - Retain matched baselines and fixed seed batteries (familiar seeds: `42, 101, 202, 303, 404`).

3. **Active Research Threads**:
   - **Thread 1 (Mechanistic)**: IPPR / coordinate-channel / formal pushdown dynamics. Unresolved bulk-interior stagnation at $L=16$ remains an active research thread (see [RESEARCH_STATE_PACKET.md](docs/RESEARCH_STATE_PACKET.md)). ASCII generation does NOT resolve this mechanistic question.
   - **Thread 2 (Generative Capability)**: Structured ASCII text generation using tiny recurrent NCA-style models with character-level causal autoregression.

4. **Environment & Git Hygiene**:
   - Working repository is strictly `titan_text`. Never touch sibling repositories (`titan_code`, etc.).
   - Preserve uncommitted user work.
   - Keep bulk checkpoints and massive generation logs out of git (enforced via `.gitignore`).
   - External exports go to `/sdcard/Download/TitanText/` when shared Android storage is writable. Never track `/sdcard/` copies in git.

---

## 2. Specialized Skills & Workflows

Detailed agent workflows are structured under `.agents/skills/`:

- [titan-research](.agents/skills/titan-research/SKILL.md): Run controlled Titan Text training, causal evaluations, latent tick sweeps, and hypothesis tests while preserving claims, ledgers, and reproducible artifacts.
- [titan-ascii](.agents/skills/titan-ascii/SKILL.md): Train, sample, evaluate, archive, and export Titan Text ASCII-generation experiments with raw output preservation.
- [openrouter-subagents](.agents/skills/openrouter-subagents/SKILL.md): Multi-agent auditing and adversarial review.
