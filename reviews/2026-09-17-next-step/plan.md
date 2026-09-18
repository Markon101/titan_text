# Arithmetic data gate implementation

Decision: implement gate 1 of `reviews/2026-09-15-codex/review.md` before spending compute on another training comparison. Existing 58 unit tests and 3 CLI tests passed at entry. The earlier memory-recorded task integrity failure is no longer current.

Scope: an opt-in corpus module and standalone audit example. Preserve all legacy generators, defaults, checkpoints, historical evidence, and dirty work. No model training in this change.

Acceptance:
- Diverse operands at exact maximum carry depths, including 0 and full width.
- Independent integer-addition labels and prefix-sum carry oracle.
- Stable canonical operand IDs and seed-independent split ownership, including swapped operands.
- Equal row counts per split/depth, explicit duplicate and overlap checks.
- Query-only scoring; separate overflow digit, complete answers, digit scores, and equal depth/position scores.
- Priors fitted only on training rows; separate held-out baseline reporting.
- Bounded failure for invalid or exhausted configurations; exclusive output creation.

Delegation: DeepSeek Flash authors candidate source; local inspection and independent tests decide acceptance. First draft rejected for duplicated final carry, shifted query labels, reversed overflow position, unbalanced split/depth cells, and depth-conditioned position priors. See `builder-task.txt`, `revision-task.txt`, and full provenance envelopes.

CLAIM A: the first delegated draft correctly implements exact carry and query alignment.
CLAIM B: inspection shows final outgoing carry is counted twice and padding receives a target.
WHY THEY DIFFER: the draft adds an extra overflow bit/step and derives query start from padded length.
DISCRIMINATING EXPERIMENT: exhaustive two-digit integer prefix-sum carry comparison and serialized observation/mask/label verification independent of the Rust generator.

After this gate: connect the accepted corpus to a separate bounded trainer, retain per-example predictions, and require held-out complete-answer competence above train-fitted priors before topology or adaptive-compute comparisons. Keep legacy MaxRipple explicitly a stress fixture.
