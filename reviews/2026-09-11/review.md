# TITAN TEXT review — September 11, 2026

**Recommendation: pause extended training, keep the small cellular-model core, and repair the task and evidence pipeline first.** The current system demonstrates trainability and nonlinear field evolution. It does not yet demonstrate causal text prediction, useful long-term memory, or a stable nontrivial attractor. The first substantial change should be the learning problem; state stabilization should then be evaluated in a separate, controlled experiment.

This review covers the current dirty checkout at `0b59e11`, all source modules and tests, the seven local commits, all three saved checkpoint manifests and weights, and `patterns.txt`. Existing source changes were preserved. New frozen diagnostics and isolated audit programs supplement the historical artifacts. No extended training was started and no saved checkpoint was updated.

## Current status and recent work

The model has 43,715 parameters, a 32-cell ring with 64 channels, a shared 96-unit local MLP, eight developmental updates during training, and a character embedding/output interface. Developmental recurrence occurs within an example; training reconstructs the initial field from tokens on every iteration. There is no persistent recurrent memory carried across examples.

| Saved checkpoint | Recorded optimizer steps | Viscosity | Train accuracy | Validation accuracy | Validation CE |
|---|---:|---:|---:|---:|---:|
| `m1_sanity` | 50 | 0 | 100% | 74.22% | 1.3150 |
| `v0_text` | 50 | 0 | 100% | 73.05% | 1.4745 |
| `v0_viscous` | 5,706 | 0.01 | 100% | 82.03% | 1.3089 |

These are different saved runs, not a controlled learning curve. The latest checkpoint records zero training CE and gradient norm approximately `9.33e-10`. Its 82.03% validation accuracy was reproduced by loading its weights into the current code. Reported “epochs” are individual optimizer iterations on the same deterministic batch, not full passes through a changing dataset.

The local history records: the minimal recurrent model; perturbation/context/trajectory diagnostics; fluid-inspired measurements and diffusion; a correction to diffusion substepping and Fourier energy normalization; checkpoint weight continuation and text-pattern export; then nested output directory, task-loading, and resume-test fixes. The current uncommitted work adds strict CLI parsing, help, config validation, and integration tests. `cargo test --locked --offline` passed **12 unit tests and 3 integration tests** during this review.

There are no retained historical probe JSON reports or training curves in this checkout. The exact sequence of prior diagnostic observations cannot be reconstructed from manifests and README assertions alone. `patterns.txt` contains eight identical decoded lines. Those lines are observations of output tokens, not measurements of hidden-state convergence.

## Findings that change the interpretation

### 1. The current next-token task exposes its answers

In [dataset.rs](../../src/dataset.rs), lines 102–105, `target[i]` equals `input[i+1]` at every interior text position. Dyck targets also wrap this relationship around the ring (lines 86–90). The entire input sequence is embedded before development ([train.rs](../../src/train.rs), lines 66–100).

The perception vector explicitly contains the right neighbor ([nca.rs](../../src/nca.rs), lines 60–72). In fact, its three components reconstruct it exactly:

`right = identity + gradient + 0.5 * laplacian`.

A zero-parameter rule that returns the next input cell achieves **248/256 = 96.875% on both current text splits**, and **100% on both Dyck splits**. Text misses only the last position in each window. This is an intentionally cheating control that quantifies target exposure; it is not a causal language-model baseline.

The isolated Rust audit also demonstrates the trained model actually depends on future input. In the latest checkpoint, changing only future position 16 from `p` to `z`, with the entire prefix through position 15 unchanged, changes the prediction at position 15 from `c` to `e`. The maximum absolute logit change is approximately **10.964**.

The existing context test does not catch this. It scrambles input positions while retaining the original targets and declares a large accuracy drop evidence of useful context ([experiment.rs](../../src/experiment.rs), lines 370–445). The simple right-copy rule drops from 96.875% to **8.59375%** under the exact training-batch shuffle and would receive the same favorable verdict. This directly falsifies the diagnostic's claimed ability to rule out that shortcut.

**Consequence:** current validation accuracy and context-sensitivity verdicts cannot establish causal prediction or linguistic memory, even though the phrase lists are disjoint.

### 2. Dataset diversity and the algorithmic task need replacement

Text training comprises four phrases totaling 193 characters; validation comprises two phrases totaling 98 characters. At the default batch size and length, every call returns the same eight windows per split. Further iterations repeatedly optimize the same 256 training targets.

The alleged Dyck depth generalization does not match the generated data. At length 32, there are only two distinct sequences in each split. Training sequences reach depths 1 and 2; validation sequences reach depths 3 and 4 and both end with two unmatched opens. They are not complete balanced Dyck strings. Even the rule “predict the opposite parenthesis” scores 96.875% on training and 90.625% on validation.

Use a real seeded generator and check prefix balance, final balance, actual depth distribution, duplicates, and split overlap. For a well-defined prediction test, use matching-delimiter recall, depth classification, or legal-continuation probability; generic balanced-string next-token prediction can have several valid answers. Dyck-2 with bracket-type recall is a stronger memory test than alternating Dyck-1 patterns.

### 3. The latest checkpoint does not exhibit stable hidden dynamics in the tested rollout

The new frozen `v0_viscous` probe uses its recorded viscosity 0.01 and the first training input. It warms up for eight steps, then records 256 states, from total age 8 through age 263:

| Quantity | First recorded state | Last recorded state |
|---|---:|---:|
| Field energy | 3.9530 | 1,721.7290 |
| Hidden-state RMS | 2.8118 | 58.6810 |
| Next-step RMS displacement | 0.3801 | 0.3057 |

No repeated cycle was detected; the minimum tested recurrence distance was 0.8872. There were 30 distinct decoded patterns. This is strong evidence of ongoing drift for this checkpoint/input/horizon. It does not characterize every possible initial condition, but it is enough to reject stability inferred from the short repeated pattern file.

The older `v0_text` checkpoint similarly increases energy from 2.9693 to 224.1407 in a 64-state probe, although its current classifier calls that “bounded wandering.” Those labels are heuristic finite thresholds, not established attractor properties.

The reason to investigate mean drift is structural: tanh and sigmoid bound each **increment**, not the accumulated state. The diffusion operator preserves each channel's spatial mean. It can smooth variation while allowing the mean to keep moving. In exact arithmetic with finite weights and bounded forcing, the implementation satisfies an at-most-linear amplitude bound over time; it supplies no uniform bound for arbitrarily long rollouts.

The isolated Rust audit confirms that mean drift matters here. Energy in the per-channel spatial means rises from **0.4699 to 972.187**, increasing from 11.9% to 56.5% of total energy. Centered spatial energy also grows, from 3.4831 to 749.542. At age 263 one channel's spatial mean has magnitude 115.68, although the single mean over all channels/cells is only 0.3503. A scalar global mean would hide this instability through cancellation.

### 4. Fluid-inspired measurements need narrower claims and consistent operators

Diffusion substepping and Parseval-normalized **energy** are useful numerical improvements. The learned 1D latent field is not a 3D incompressible Navier–Stokes solver. Its finite amplitude/roughness thresholds and growth fits do not demonstrate a mathematical singularity or satisfy the Beale–Kato–Majda theorem.

The September 8 external announcement referenced by the README exists: [OpenAI reports a proof and Lean formalization for forced 3D Navier–Stokes Cases C/D](https://openai.com/index/navier-stokes-solution/). That announcement does not transfer a theorem to TITAN's learned update. The [official problem formulation](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) specifies the physical equations and distinguishes the relevant vorticity criterion. The original [BKM paper](https://scholars.duke.edu/publication/759544) concerns 3D Euler solutions.

There is also a concrete diagnostic mismatch. Spatial enstrophy uses the centered-gradient multiplier `sin(theta)`, spectral enstrophy uses `theta²`, and actual Laplacian dissipation uses `4 sin²(theta/2)` ([field.rs](../../src/field.rs), lines 180–268; [experiment.rs](../../src/experiment.rs), line 581). They cannot be treated as the same discrete energy budget.

A decisive fixture is the alternating field `+1,-1,+1,-1,...`: energy is 0.5, centered-gradient enstrophy and the “BKM” proxy are zero, palinstrophy is 8, and the spectral-enstrophy sum is approximately 4.9348. At viscosity 0.05 and step size 0.5, the report predicts zero dissipation, while one diffusion step changes energy at rate −0.19.

In the new 64-step frozen viscosity sweep, 0.05 is again the first sampled positive value passing the program's regularization criterion. Yet energy remains 101.24 at 0.05 and rises to 191.11 at 0.4 while spatial roughness falls. This is not a universal critical viscosity or proof of whole-state stabilization. The sweep itself does not measure token accuracy.

A separate same-weights evaluation supplies that missing comparison: viscosity 0.01/0.05/0.10/0.20/0.40 yields current-task validation accuracy **82.03% / 82.42% / 82.42% / 76.17% / 32.42%**. Moderate smoothing can help this particular exposed-target task; sufficiently strong smoothing degrades it. This supports a real stability/information tradeoff and does not establish causal text performance or the outcome of retraining at another viscosity.

Replace singularity/ergodicity verdicts with measured growth, threshold crossings, tail drift, and observed recurrence. Separate spatial mean energy from centered spatial energy. Use Laplacian-consistent edge energy and measured diffusion energy loss. Freeze the forcing schedule independently of observation horizon: currently changing horizon also changes the forcing envelope.

### 5. Nonlinear response is measurable, but its meaning is overstated

The perturbation probe computes curvature `Q = even_response / epsilon²`, then compares `Q` directly to a normal update. The actual even perturbation displacement is `epsilon² * Q`. At epsilon 0.01, the latest checkpoint's high-frequency ratio is reported as 0.5326, while the realized even-displacement ratio is approximately **0.0000533**, or **0.00533%**.

The new epsilon sweep shows a fairly consistent high-frequency curvature near epsilon 0.003–0.03, but small-epsilon sensitivity in the weaker modes. For example, low-frequency Q is 0.01619 at 0.01 and 0.07898 at 0.001. This warrants precision/null controls, not a claim that every observed response is noise. A pointwise quadratic nonlinearity can turn an alternating perturbation into a constant response without learned spatial information transfer.

Before interpreting upward coupling, test a linear null and a known quadratic map, normalize perturbations across frequencies, test several channel directions, use exact low-frequency Fourier projection, and share random update masks across plus/minus/base evaluations. Correct odd-length zero-mean perturbations. Report curvature and actual finite displacement separately.

### 6. Reproducibility and checkpoint integrity need small infrastructure changes

`seed=42` is recorded but never applied to initialization or stochastic updates. Continuation deliberately creates a fresh AdamW optimizer, so the step count is cumulative but the optimizer trajectory restarts. This is documented and should stay explicit.

One encouraging result: a bounded frozen-checkpoint gradient audit found finite, nonzero validation-loss gradients in all nine parameter tensors. A central finite difference along the normalized full gradient approached the automatic derivative as epsilon decreased: relative error was 1.86%, 0.197%, and 0.0228% at epsilon 0.01, 0.003, and 0.001. This supports a working differentiable path through the present model. It is a directional spot check, not exhaustive derivative validation, but points toward the objective and dynamics as immediate priorities over an autograd rewrite.

Saving overwrites the weight file before replacing the manifest; the pair is not transactional. Loading does not verify the stored checksum ([checkpoint.rs](../../src/checkpoint.rs), lines 64–113). All three existing files currently match their recorded checksums, so this is a reliability gap, not evidence that the current checkpoints are corrupt.

Probe reports lack checkpoint/source identity and effective settings. Several probes hardcode eight warmup steps; `ns-probe` defaults viscosity to zero even when a loaded checkpoint records another value. The declared `periodic_boundary` option is not honored by the spatial operators. These issues can silently invalidate comparisons.

## Recommended direction and acceptance gates

### First: make the experiment trustworthy, preserving the legacy model

Add deterministic RNG streams; record dataset identity, actual seeds, dirty source identity, parent checkpoint hash, effective config, input split and warmup age in every report. Verify checksums on load. Save immutable checkpoint generations and atomically select the completed generation. Retain legacy reading and explicitly identify optimizer-reset continuation; exact continuation additionally needs optimizer/RNG/sampler state.

Add semantic tests, not more CLI-only checks: right-copy control; future-token intervention; valid dataset properties; diffusion energy identities including Nyquist and odd lengths; perturbation linear/quadratic controls; and finite-difference gradient checks through embedding, perception, several recurrent steps, gate/delta, diffusion, and output projection. Test multiple development lengths, including beyond eight. Existing tests passing is valuable but does not replace these gates.

These changes can preserve existing weights and legacy execution. Version corrected reports and new task semantics so historical scores remain interpretable.

### Second: fork the learning problem deliberately

For text prediction, the first new model should enforce **prefix-only information**. A causal spatial NCA needs one-sided perception and a consistent boundary/mixing operator; masking the loss or changing only the output head cannot remove the existing future path. The diffusion operation must also obey the information constraint. A prediction at position `i` must be invariant to every change in positions greater than `i`, including wraparound, after every developmental step.

A useful alternative retaining bidirectional internal dynamics is a prefix-only input plus a separate next-token query/output: withhold all target/future tokens from every input channel and score the withheld query. This changes batching and objectives but can keep an internal spatial workspace. Do not mask one global suffix and still score positions that can see their own targets.

For the original developmental-attractor question, use a separate **masked restoration** task: reveal a partial symbolic pattern, withhold the targets, and train recovery from randomized corruptions and ages. Bidirectional dynamics are appropriate here. Measure hidden-token reconstruction, damage recovery, and persistence. Keep its scores distinct from causal language modeling.

My preference is to establish the causal task first, with masked restoration as the companion experiment if stable developmental computation remains a primary research goal. Preserve current checkpoints as historical controls. A change of objective/boundary is scientifically incompatible with current scores even if some tensor shapes happen to load.

### Third: test one stabilization change at a time

Compare the corrected-task NCA with an opt-in weak restoring/leak term. For example, `(1-lambda)*x + alpha*bounded_update`, followed by nonexpansive mixing, can provide a uniform amplitude bound for fixed `0 < lambda <= 1`; it can also erase useful memory. A smoothly bounded state parameterization is another arm, with saturation/gradient costs that must be measured. Neither needs to be installed as a new default yet.

Train across randomized developmental ages and, for the restoration arm, damaged states; score several ages and held-out longer horizons. Stability should mean acceptable tail drift and task performance together. Penalizing all activity could produce an uninformative fixed point, while diffusion alone leaves mean drift unresolved.

Do not add attention, a large global mixer, more channels, or much longer backpropagation until the small model fails a valid task for an identified reason. Later, a persistent latent cellular workspace updated once per incoming token is a coherent redesign for streaming memory, but the present benchmark cannot yet justify its complexity.

### Fourth: run small comparisons before approving a longer run

| Question | Minimal experiment | Gate for continuing |
|---|---|---|
| Is prediction causal? | Same prefix, many altered futures; finite-difference influence map at several ages/boundaries | No future influence within a stated numerical tolerance |
| Does the task require memory? | Generated delayed copy, bracket-type recall, and held-out text; valid split/property checks | Local-copy/alternation shortcuts fail; genuine history matters |
| Does cellular recurrence help? | Smoothed n-gram, small recurrent baseline, radius-zero model, and causal local model with matched data/parameter and compute reporting | Repeatable held-out benefit or a measured recovery/stability advantage |
| Does a stability change preserve computation? | Baseline versus leak or bounded-state arm, initially 3 seeds | Better long-horizon state behavior without losing the task advantage |
| Is there an attractor or recovery? | Fixed forcing schedule; ages 8/32/128/512, initially bounded runtime; multiple starts, damage and recovery; repeated-cycle checks | Tail behavior and successful task recovery across inputs, not one close state pair |
| Does it generalize? | Held-out lengths, nesting depths, phrases/templates and seeds | Gains survive distribution changes and a final untouched test set |

Use three seeds for an inexpensive initial screen, then more seeds for the surviving comparison. Report uncertainty across independent sequences/runs, not 256 correlated token positions as independent experiments. Predeclare the score and stability gate before a long run. A negative result is useful if it identifies whether locality, available memory, objective, or state control is the limiting factor.

## Evidence and reproduction

The JSON files beside this report contain the raw frozen probes; `provenance.json` records the checkout, source/checkpoint hashes, and the current binaries. The isolated dataset and Rust audit source/results are included alongside them. Raw legacy reports retain their original overstrong verdict strings so this review can be checked against actual program output; those strings are not adopted as conclusions here.

The principal frozen commands were:

```sh
./target/debug/titan_text probe --load-dir checkpoints/v0_viscous --horizon 256 --output reviews/2026-09-11/v0_viscous-probe-256.json
./target/debug/titan_text probe --load-dir checkpoints/v0_text --horizon 64 --output reviews/2026-09-11/v0_text-probe-64.json
./target/debug/titan_text ns-probe --load-dir checkpoints/v0_viscous --viscosity 0.01 --horizon 64 --forcing-amp 0.2 --output reviews/2026-09-11/v0_viscous-fluid-64.json
```

Separate eight-state probes used epsilon 0.001, 0.003, 0.03 and 0.1; the main probe supplies epsilon 0.01. These frozen tests do not update weights. The context battery includes an unseeded random state-reset comparison, so that particular submetric is not exactly reproducible until RNG handling is fixed.
