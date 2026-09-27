---
name: research-team
description: >-
  Dedicated scientific research team skill led by Gemini as Principal Investigator (PI).
  Orchestrates parallel DeepSeek 4.1 Flash high-context research collaborators, 13-step recursive self-questioning,
  independent model reconstruction audits, multi-agent council meetings, high-temperature ideation with diversity bypass,
  Jev structured routing gates, Codex CLI implementation, Astra escalation, and 7-tier canonical state memory.
---

# Research Team (Principal Investigator Architecture)

This skill provides a rigorous multi-agent scientific research framework for Titan Text:
recurrent dynamics analysis, Chomsky pushdown evaluation, causal interventions, and adversarial falsification.
**Gemini acts as Research Lead / Principal Investigator (PI)**, coordinating high-context specialists,
synthesizing findings, preserving empirical disagreements, and maintaining ground-truth experimental state.

Context transport for delegated workers: pass condensed packets with `investigate --roles <role> --context-file <packet.txt>` (or pipe via `--context-file -` / `--context -`). Plain `--context` is literal text only — a file path passed there is sent as a path string, NOT read (this silently starved workers of context; fixed 2026-09 with sentinel-tested `resolve_context_argument`, `research_coordinator.py`). Verify worker prompt size via `prompt_tokens` in the JSON output before trusting an audit.

Nested-data statistics rule: when observations are nested (tokens within sequences, sequences within seeds/runs, cells within batches), do NOT infer significance by treating low-level observations as independent. Identify the independent unit first (usually seed or run), then use seed-level sign-flip permutation, cluster/hierarchical bootstrap preserving within-cluster pairing, or per-cluster summaries across clusters. Token-level p-values from pooled observations are descriptive only and typically anti-conservative; prefer cluster-level inference even when it costs power. Report both only if labeled.

Jev is an optional adviser for routing and identifying uncertainty. `verify` returns an advisory assessment, never a scientific certification: `model_supports_claim` describes the model's opinion, while `is_supported` remains false pending independent artifact verification. Positive assessments route to artifact inspection; negative assessments route to investigation rather than automatically falsifying a hypothesis.

Only explicit PI-verified immutable artifact/run references (`verify --evidence-id`, Python `evidence_ids`) count as new evidence; repeated IDs within a coordinator do not reset its stopping counter. Across CLI sessions the PI must track whether evidence is actually new. A changed model opinion does not count. Calibration outcomes are recorded separately with independent outcome provenance, never from Jev's own verdict. Failed/malformed calls are unavailable judgments, not confirming evidence; retain unscored ideas for manual review.

Historical controller benchmark tables used simulated task outcomes and do not demonstrate research quality, savings, or calibration. Inspect the actual provenance before carrying those claims into research records.


> [!IMPORTANT]
> **High-Context Collaboration Principle**:
> DeepSeek 4.1 Flash agents are used as **high-context research collaborators**, not short-response consultants.
> Token economy is secondary to scientific correctness, independent reconstruction of the problem, and discovery of things we have not thought to ask.
> Always supply agents with long, detailed context via `ContextBuilder` (`scripts/context_builder.py`) covering all 21 materially relevant dimensions.

---

## The 15-Step Empirical Research Loop

```
OBSERVATION
  ↓
COMPETING HYPOTHESES (Conservative vs Novel)
  ↓
RECURSIVE SELF-QUESTIONING (13-Step Interrogation)
  ↓
DISCRIMINATING EXPERIMENT DESIGN (Multi-Branch Separation)
  ↓
INSTRUMENTATION (Probes, Masks, Diagnostics)
  ↓
IMPLEMENTATION (Rust kernel / PyTorch harness)
  ↓
DETERMINISTIC TEST (Unit tests & CLI verification)
  ↓
REAL RUN (Multi-seed evaluation: N >= 3 seeds)
  ↓
ARTIFACT INSPECTION (Logit temperature, RNG seeding, masking)
  ↓
STATISTICAL ANALYSIS (Paired Cohen's d, TOST equivalence, p-values)
  ↓
CAUSAL INTERVENTION (Lesion, transplant, roll, conjugate)
  ↓
ADVERSARIAL REVIEW (Skeptical agent / Falsification arbiter)
  ↓
IDEATION CYCLE (Novel tasks, synthetic worlds, counterfactuals)
  ↓
UPDATED CANONICAL RESEARCH STATE (.agents/state/research_packet.json)
  ↓
NEXT DISCRIMINATING EXPERIMENT
```

---

## 1. High-Context Research Collaborators & Context Builder

For important consultations, generate full uncompressed context using:
```bash
python3 .agents/skills/research-team/scripts/context_builder.py \
  "Investigate pushdown stability under extreme unroll T >= 64" --out /tmp/context.txt
```

Gathers up to 21 materially relevant dimensions:
- Current research objective & architecture dynamics (CD-DV-NCA, ECR, causal stencil)
- Task definitions and selection rationale (Parity, Sum, Arithmetic, Dyck-4)
- Latent tick semantics ($\tau$ unrolling depth, roundtrip reach $T^* \ge 2\lceil L/k \rceil$)
- Stabilization mechanisms (STE sign quantization, OCPD subspace normalization)
- Current & historical metrics across stratified depths ($D=1..16$)
- Activation diagnostics & Lyapunov drift indicators
- Causal interventions and implementations (`src/intervention.rs`)
- Statistical protocols (paired seeds, Cohen's d, TOST equivalence)
- Active hypotheses vs falsified hypotheses
- **Historical failures & artifacts explicitly labeled** so agents never reason from superseded results
- Key source-code excerpts from `src/nca.rs`, `src/latent.rs`, `src/intervention.rs`
- Benchmark reports, git diffs, known confounds, and competing agent arguments.

---

## 2. The 7-Tier Canonical Research-State Packet

Maintained at `.agents/state/research_packet.json` and documented in [`docs/RESEARCH_STATE_PACKET.md`](file:///data/data/com.termux/files/home/projects/titan_text/docs/RESEARCH_STATE_PACKET.md):

1. **`ESTABLISHED`**: Results supported by deterministic quantitative evidence (e.g. recurrent necessity $d=15.56$, Dyck-4 pushdown $43.9\%$, transport conjugation invariance $\Delta \le 0.0023$).
2. **`SUPPORTED BUT NOT ESTABLISHED`**: Interpretations fitting current evidence with plausible competitors.
3. **`HYPOTHESES`**: Mechanistic causal ideas awaiting decisive testing.
4. **`FALSIFIED / SUPERSEDED`**: Earlier interpretations or measurements invalidated by subsequent experiments (explicitly tagging measurement artifacts!).
5. **`OPEN QUESTIONS`**: Unresolved scientific questions.
6. **`KNOWN CONFOUNDS`**: Alternative explanations, shallow-bracket domination, unconstrained continuous drift, static persistent input memorization.
7. **`NEXT DISCRIMINATING EXPERIMENTS`**: Minimal tests capable of separating remaining hypotheses.

CLI Operations:
```bash
# Render formal 7-tier canonical state packet
python3 .agents/skills/research-team/scripts/research_state.py canonical

# Add supported interpretation
python3 .agents/skills/research-team/scripts/research_state.py add-supported \
  "OCPD bounds loss growth via spectral damping" --alt "Contractive fixed-point projector"

# Add falsified finding with artifact flag
python3 .agents/skills/research-team/scripts/research_state.py add-falsified \
  "Canonical baselines collapse to 0.0% on Dyck-4" "Evaluated untrained checkpoint weights" --artifact

# Add known confound
python3 .agents/skills/research-team/scripts/research_state.py add-confound \
  "Shallow bracket density (D <= 2) allows local bigram shortcuts" --mitigation "Stratify by D >= 4"
```

---

## 3. Disciplined 13-Step Recursive Interrogation & 11-Point Decision Battery

For major consultations, require recursive self-questioning before finalizing recommendations:
```bash
# Run 13-step recursive self-questioning flow
python3 .agents/skills/research-team/scripts/recursive_interrogator.py \
  "Is discrete carry quantization strictly necessary for pushdown memory or can unitary continuous gating achieve it?" \
  --depth 2

# Run 11-point deep decision battery for pivotal forks
python3 .agents/skills/research-team/scripts/recursive_interrogator.py \
  "Should we adopt OCPD subspace normalization as the canonical baseline for all L=128 experiments?" \
  --deep-decision
```

The 11-Point Battery explicitly produces:
1. Strongest argument FOR
2. Strongest argument AGAINST
3. Strongest mundane explanation
4. Strongest interesting explanation
5. Easiest experiment that could falsify it
6. Most decisive experiment regardless of cost
7. Result that would cause substantial belief update
8. Hidden assumption most likely to invalidate experiment
9. Measurement currently missing
10. Surprising alternative hypothesis
11. Question nobody in the current research loop is asking
Followed by recursive interrogation of the most fragile assumption and missing measurement.

---

## 4. Independent Reconstruction Test

Before trusting a major consultation, audit the agent's internal model:
```bash
python3 .agents/skills/research-team/scripts/independent_reconstruction.py
```
Asks the agent:
> *"Given only this evidence, reconstruct your model of Titan Text. Explain what has actually been demonstrated, what has not been demonstrated, which historical conclusions are now invalid, what the strongest remaining alternative explanations are, and what experiment would most efficiently distinguish them."*

Audits whether the agent recognizes:
- Baseline collapse was an artifact of untrained weights.
- FC-4 lesion failure was an artifact of shallow bracket averaging.
- Continuous NCA failure was spatial correlation length ($\xi \le 6$), not settling latency.
- Physical grids are bounded DPDAs (FSA Type-3).

---

## 5. Dedicated Exploratory Ideation Agent

The Ideation Agent searches conceptual space broadly across neuroscience, dynamical systems, cellular automata, and control theory. Unconventional high-risk/high-information ideas are **unconditionally preserved** from being filtered out.

```bash
python3 .agents/skills/research-team/scripts/ideation_engine.py \
  "Construct counterfactual state-interventions to isolate the minimal pushdown subspace"
```

Explicitly prompted with the 11 exploratory inquiries:
1. *What is Titan doing that current experiments are incapable of detecting?*
2. *What behavior are we currently averaging away?*
3. *What variable treated as a nuisance contains the phenomenon?*
4. *What result would be surprising under every current hypothesis?*
5. *What minimal synthetic world would force Titan to reveal compositional latent computation?*
6. *What strange experiment produces highly discriminating evidence?*
7. *What can we measure from saved activation traces not yet considered?*
8. *What interventions distinguish storage, transport, transformation, attractor stabilization, and true recurrence?*
9. *Can we construct two inputs locally indistinguishable requiring different outputs solely from recurrent state?*
10. *Can we construct counterfactual state-transplant, state-rescue, state-swap, or trajectory-splicing experiments?*
11. *What is the smallest latent subspace carrying task-relevant information?*

---

## 6. Multi-Agent Research Meetings & Cross-Agent Questions

For major milestones, convene the 7-specialist research meeting:
```bash
python3 .agents/skills/research-team/scripts/research_meeting.py \
  "Decide whether to replace continuous residual H with unitary skew-symmetric updates" \
  --all-roles
```

Protocol:
- **Round 1 (Independent)**: Each specialist (Arbiter, Designer, Dynamics, Rust Audit, Statistical, Ideation, Skeptic) conducts an isolated domain analysis with zero cross-pollination to kill groupthink.
- **Round 2 (Synthesis)**: Specialists review competing findings and answer the 10 core questions (What is happening? What evidence weakens it? What should we absolutely NOT conclude yet? etc.).
- **Cross-Agent Consultations**: Specialists pose targeted follow-up inquiries to one another (e.g. Dynamics $\to$ Statistics, Arbiter $\to$ Rust Audit).
- **Gemini PI Synthesis**: Explicitly preserves empirical disagreements. Asks: *"What experiment would make this disagreement empirical rather than rhetorical?"*

---

## 7. Research Epistemology Reference

Governed strictly by [`references/EPISTEMOLOGY.md`](file:///data/data/com.termux/files/home/projects/titan_text/.agents/skills/research-team/references/EPISTEMOLOGY.md). Enforces precise distinctions between:
- interesting dynamics
- representation
- decodability
- correlation
- information storage
- information transport
- iterative transformation
- recurrence
- causal dependence on recurrence
- instance-specific causal dependence
- algorithmic generalization
- latent computation
- autonomous dynamical behavior.

Terms like *"reasoning"*, *"attractor manifold"*, *"algorithm"*, or *"emergent memory"* are forbidden without operational metrics and negative ablation controls.
