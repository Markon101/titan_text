# Codex-led DeepSeek research harness

Adapted from Gemini/Antigravity's Titan Text `research-team` and adaptive
`openrouter-subagents` skills, inspected on 2026-09-22. Their reusable ideas are
independent reconstruction, evidence-separated state, specialist criticism,
exploratory ideation, and experiments that resolve disagreements. Their project
results, routing benchmarks, credential configuration, and claims of calibrated
model confidence are not adopted as verified facts.

## Start with an evidence packet

State the decision, deliverable, compute/time budget, and what would change the
next action. Select enough evidence for an independent reconstruction:

- Objective, task definition, split construction, target masks, and trivial baselines.
- Architecture, update equations or source spans, causal visibility, tick/horizon
  semantics, initialization, optimizer, readout, and stabilization when relevant.
- Exact run/config/checkpoint identities, code revision and relevant dirty diff,
  seed pairing, sample counts, metrics with denominators, and uncertainty.
- Relevant interventions, activation summaries, diagnostics, and source locations.
- Historical artifacts explicitly marked invalidated or superseded, known
  confounds, missing evidence, and competing explanations where appropriate.

These are relevance checks, not a mandatory large packet for every request.
Do not automatically scrape the repository, attach binary checkpoints, or treat
old research notes as current measurements. Select precise UTF-8 spans and
sanitized tables; missing evidence must remain visible as missing.

The helper blocks `.agents`, `.codex`, and checkpoint directories. If a project's
research state lives there, locally prepare a minimal, reviewed research summary
in an ordinary project report file with its source paths and timestamps. Do not
copy protected files wholesale or loosen the selection filter to send them.
Inspect `--dry-run` metadata before a live request. Raise budgets explicitly only
when relevant evidence requires it; split genuinely separate questions rather
than silently truncating context. Increase output tokens when a complete review
needs them; an `incomplete` envelope is unfinished work, not a completed review.

## Independent reconstruction before synthesis

For a major claim, first give a collaborator raw selected evidence and task
semantics without the lead's preferred answer or other reviewers' conclusions:

> Reconstruct what this implementation and these measurements establish. Cite
> evidence locations. Identify what remains unproved, any explicitly superseded
> result, the strongest alternative explanation, and the cheapest separating
> test. List evidence you would need but have not received.

Use `--role researcher` for this pass. Judge it against actual sources, not
agreement with Codex. If the packet is missing decisive context, add that context
and request a targeted correction; do not coach the reviewer toward a conclusion.

For independent branches, each request receives the same relevant raw evidence
but no other branch's output. Parallel requests are optional when authorized and
useful; separate sequential calls also preserve independence. Keep the batch
bounded by a stated call/token/time budget. Do not spawn all roles by default.

## Specialist tasks using existing CLI roles

These are task specializations, not additional `--role` values:

| Research assignment | Supported role | Required deliverable |
| --- | --- | --- |
| Independent investigator | `researcher` | Reconstruction, evidence gaps, alternatives |
| Task integrity auditor | `task-auditor` | Visibility, shortcuts, masks, split and dummy controls |
| Rust/intervention auditor | `code-reviewer` | Whether code implements the intended lesion/update |
| Dynamics specialist | `dynamical-systems-critic` | Measured behavior, finite-time limits, separating controls |
| Statistical specialist | `researcher` | Unit of analysis, pairing, uncertainty, assumptions |
| Experimental designer | `experiment-designer` | Smallest decisive protocol and predicted outcomes |
| Exploratory ideation | `mechanistic-competitor` with `--mode hypothesis` | Diverse mechanisms and discriminators |
| Skeptic/synthesis challenger | `result-skeptic` or `adversarial-reviewer` | Strongest artifact explanation or unsupported leap |
| Falsification arbiter | `experiment-designer` | CLAIM A/B, why they differ, empirical decision rule |

For example, after preparing and inspecting a packet:

```sh
SKILL="${CODEX_HOME:-$HOME/.codex}/skills/deepseek-flash"
python "$SKILL/scripts/deepseek_flash.py" run \
  --role researcher \
  --task 'Independently reconstruct what the selected evidence establishes, what it does not, and the cheapest separating test. Cite evidence and missing inputs.' \
  --context-file reports/research-context.md \
  --max-tokens 4096 --dry-run
```

The packet path is an example to create locally, not a shipped artifact. Remove
`--dry-run` only when ready for the bounded external request. Save each useful
result's JSON envelope, including model, input hashes, status, and returned usage.
An API failure, truncated answer, or mock test is not a successful consultation.

## Synthesis and pivotal decisions

After independent reviews, supply selected competing findings to a synthesis
pass. Preserve each disagreement using **CLAIM A**, **CLAIM B**, **WHY THEY
DIFFER**, **DISCRIMINATING EXPERIMENT**. Ask the responsible specialist a targeted
follow-up when a conflict hinges on missing implementation or statistical detail.
Model agreement and model-generated probabilities are not empirical evidence.

For a consequential fork, request a concise decision brief covering the relevant
items below. Request conclusions and supporting evidence, not private reasoning
traces or a transcript of recursive internal deliberation:

1. Strongest evidence for and against the interpretation.
2. Strongest mundane explanation and strongest interesting alternative.
3. Cheapest falsifying test and most decisive test, with their cost difference.
4. Outcomes that would change the decision, including an inconclusive outcome.
5. Hidden assumption most likely to invalidate the test.
6. Missing measurement and an overlooked research question.

Two unproductive review passes trigger evidence gathering, a bounded experiment,
or closure of that branch. Do not recursively convene councils without new inputs.

## Preserve exploratory ideas without promoting them to findings

Use hypothesis mode for questions such as: what is an aggregate hiding; what
minimal synthetic world separates the mechanisms; can a state transplant, rescue,
swap, or trajectory splice distinguish storage from transformation; what unused
measurement is available from saved traces?

Retain at least one unconventional but testable candidate when it adds a distinct
prediction. Label it speculative and specify a discriminator. Novelty does not
earn execution priority by itself, and skepticism should not erase an idea solely
because it is unfamiliar. Codex chooses the next experiment from information
value, feasibility, and the user's research objective.

## Execute and audit the separating experiment

Before a run, specify hypotheses, controls, intervention timing, sample/seed
design, primary metric, expected outcomes, stopping condition, and compute bound.
Validate the implementation and provenance before interpreting numerical results.
Verify that baselines were actually trained and that interventions touch the
intended state. Preserve parent checkpoints and use isolated outputs when needed.

Match statistical analysis to the design. Pair seeds when the comparison is
paired; identify independent replication units rather than treating tokens or
ticks as independent samples. Report effect sizes and uncertainty. Equivalence
claims need a justified prespecified margin; a nonsignificant difference alone
does not establish equivalence. Choose seed counts for the claim and variance,
not because three seeds mechanically certify correctness. Single-seed diagnostics
can locate bugs but cannot establish robust learning gains.

For recurrent models, keep distinct: dynamics, representation, decodability,
storage, transport, transformation, implemented recurrence, causal necessity,
instance-specific dependence, and out-of-distribution generalization. Probe
accuracy alone does not show the readout uses the information. Lesion damage may
reflect generic distribution shift; matched controls and rescue/transplant tests
help distinguish it. Stratify difficulty when easy cases dominate. Finite-grid
results do not establish unbounded algorithmic capacity. Numerical parity and
test success remain implementation evidence, separate from learning quality.

## Seven categories of research state

Use the project's existing record and schema when present; otherwise maintain a
small research note in the authorized project workspace. These categories organize
research records, not Codex's personal memory. Do not modify assistant memory as
part of this workflow without an explicit user request to do so.

| Category | Admission/update rule |
| --- | --- |
| Established findings | Verified measurement with provenance and a bounded claim |
| Supported interpretations | Fits evidence; lists surviving alternatives |
| Hypotheses | Testable mechanism with falsification conditions |
| Falsified / superseded | Rejection evidence and whether the issue was a measurement artifact |
| Open questions | Missing knowledge that can change the research decision |
| Known confounds | Alternative cause and planned control or remaining limitation |
| Next discriminating experiments | Competing predictions, protocol, budget, decision criteria |

Each material entry should carry an ID, date, source artifact/run ID, code/config
and checkpoint identity when relevant, scope, and evidence status. Label text as
`MEASUREMENT`, `INFERENCE`, `HYPOTHESIS`, or `SPECULATION` where ambiguous. Keep
superseded entries and their replacement links; never silently rehabilitate an
invalidated result. Hashes establish artifact identity, not scientific validity.

Update state only after inspecting the supporting evidence. Preserve unresolved
disagreements and missing controls. Finish with what changed, what remains
uncertain, and the next separating test or concrete blocker.
