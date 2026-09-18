---
name: deepseek-flash
description: Delegate bounded research, coding, adversarial review, architecture critique, and experiment analysis to DeepSeek V4.1 Flash through OpenRouter. Use for an independent challenge or research checkpoint with explicitly selected context.
---

# DeepSeek Flash collaborator

Use DeepSeek periodically in the research program: audit a new task/objective, review a meaningful diff, challenge an impressive result, or design a discriminating experiment before scaling compute. Keep each request bounded to one purpose and a concrete deliverable. This skill uses a Python helper, not a registered MCP tool.

## Setup and configuration

Requires Python 3.10+ and curl 8.4+. No pip packages. The installed skill lives at `${CODEX_HOME:-$HOME/.codex}/skills/deepseek-flash`; its reusable source is `skills/deepseek-flash` in the originating repository. The `SKILL.md` frontmatter and `agents/openai.yaml` follow the installed Codex skill conventions. If a newer Codex installation scans only `.agents/skills`, install this same folder there instead; avoid duplicate copies in discovery paths.

`OPENROUTER_API_KEY` is required in the inherited process environment for live delegation. Never persist it in source, shell history, Markdown, logs, fixtures, or checkpoint metadata. The helper never reads credential files and has no `--env-file` option. Existing ignored local secret files are not used. Supply the environment through your trusted launcher or secret manager; see [README.md](README.md) for usage and troubleshooting.

- `DEEPSEEK_FLASH_MODEL`: optional model override. Default `deepseek/deepseek-v4.1-flash`, verified from OpenRouter's live catalog on 2026-09-14. Never silently switch versions or models.
- `OPENROUTER_BASE_URL`: optional HTTPS API root, default `https://openrouter.ai/api/v1`. An override receives the credential and selected input; use only a trusted endpoint. Redirects and URL credentials are refused.
- CLI `--model` and `--base-url` override those environment settings.
- Defaults: 2,048 output tokens, temperature 0.1, total network timeout 120 seconds, at most two retries with 1/2-second exponential backoff. Each attempt is capped at 45 seconds. CLI bounds: output 1–32,768 tokens, timeout 1–600 seconds, retries 0–4.
- `--mode hypothesis` uses temperature 0.8; `--mode review` uses 0.1. Explicit `--temperature` wins. The mechanistic-competitor role defaults to 0.8. Reasoning is disabled and excluded from returned fields; overrides requiring reasoning may reject the request, rather than silently changing parameters.

## Invocation

Set `SKILL` to the installed folder (or the source folder while developing):

```sh
SKILL="${CODEX_HOME:-$HOME/.codex}/skills/deepseek-flash"
python "$SKILL/scripts/deepseek_flash.py" discover
python "$SKILL/scripts/deepseek_flash.py" smoke
python "$SKILL/scripts/deepseek_flash.py" run \
  --task 'Audit target visibility and propose the cheapest leakage control.' \
  --role task-auditor --file src/dataset.rs:1-80 --dry-run
```

Inspect the dry-run selections, byte count, model, role, and purpose, then remove `--dry-run` to delegate. No prompt/file contents are printed by dry-run. For a review, select a diff yourself:

```sh
git diff -- src/dataset.rs > scratch/task-generator-review.diff
python "$SKILL/scripts/deepseek_flash.py" run \
  --task 'Review this diff for shortcuts and masking errors; cite concrete evidence.' \
  --role code-reviewer --context-file scratch/task-generator-review.diff \
  --max-tokens 1600
python "$SKILL/scripts/deepseek_flash.py" run \
  --task 'Give three incompatible explanations and cheap separating controls.' \
  --role mechanistic-competitor --mode hypothesis \
  --context 'A synthetic example: recurrent state norms stabilize while next-token accuracy rises.' \
  --json-answer
```

The Python module exposes:

```python
# Add the installed skill's scripts directory to sys.path before importing.
from deepseek_flash import deepseek_flash
result = deepseek_flash(
    task='Find the strongest mundane explanation and a falsifying control.',
    context='Aggregate observations and experimental constraints only.',
    files=['src/dataset.rs:1-80'], role='result-skeptic',
    temperature=0.1, max_tokens=1600,
)
```

The module returns a dictionary; the CLI always emits a JSON envelope. `--json-answer` additionally requests and validates a JSON object from the model. Results identify purpose, role, requested/returned model, endpoint, selected file spans and SHA-256 hashes, context hash, request bytes/hash, finish reason, attempts, approximate input token count (serialized bytes / 4), and returned numerical usage/cost. Missing usage stays null. A non-stop finish or invalid JSON answer is marked `incomplete` and exits nonzero; do not treat truncated answers as complete reviews. API failures exit nonzero with sanitized errors and safe request provenance; envelopes include UTC timestamps. No raw response or reasoning fields are persisted.

## Context and security

Never send the whole repository. Use explicit relevant files, inclusive 1-based `PATH:START-END` spans, narrow diffs, experiment tables, sanitized checkpoint metadata, aggregate statistics, and explicit research questions. The helper never enumerates the repository or runs git on its own. Resolve relative selections under `--root` (default current directory); symlink targets must remain within it.

At most 16 selections; each source is at most 2 MiB. Serialized requests default to 128 KiB with a configurable hard cap of 512 KiB (`--max-request-bytes`); oversized inputs fail without truncation. Response cap is 8 MiB. Byte limits are guardrails, not tokenizer estimates. Environment files, obvious credentials, protected directories, and binary checkpoints are refused. Known key/pattern scanning is defense in depth, not a comprehensive secret detector: inspect selected text, especially diffs, before sending it.

The key goes only in the authentication header through curl stdin, never argv, prompt content, temporary request files, or output. Curl configuration files and redirects are disabled; TLS verification remains enabled. The helper strips the key from the curl child environment. It does not execute model output, edit files, browse, or load checkpoints. OpenRouter/provider data-retention policies still govern sent content; this helper does not establish zero retention.

## Research roles and use of findings

| Role | Requested challenge |
| --- | --- |
| `adversarial-reviewer` | Leakage, confounds, wrong metrics, gradients, mundane explanations |
| `architectural-minimalist` | Smallest conventional competitor; components that do not earn complexity |
| `mechanistic-competitor` | Three incompatible mechanisms and cheapest separating experiments |
| `task-auditor` | Generator leakage, shortcuts, split/distribution mismatch, target artifacts |
| `dynamical-systems-critic` | Conservative evidence for contraction, instability, transients, oscillation, recurrence, fixed points |
| `code-reviewer` | State mutation, gradients, masks, shapes, silent no-ops, recurrence, nondeterminism |
| `experiment-designer` | Maximum information gain within compute constraints |
| `result-skeptic` | Strongest artifact explanation and decisive control |
| `researcher` / `coder` | Bounded evidence synthesis or a proposed minimal patch with verification |

Treat the returned answer as a candidate claim. Independently inspect evidence and test proposed code. Save useful conclusions, reviews, or experiment proposals with returned provenance/usage in an appropriate research note; do not save hidden reasoning. When Codex/Astra, DeepSeek, prior Gemini findings, or the Grumpy Reviewer disagree, record **CLAIM A**, **CLAIM B**, **WHY THEY DIFFER**, **DISCRIMINATING EXPERIMENT**. Then run the smallest authorized experiment, record its evidence, and update the claims; if blocked by compute/access, record that concrete blocker. Model reputation never settles disagreement. Keep CPU correctness/parity and learning-quality claims separate.

Builder creates. Codex reasons. DeepSeek challenges. Grumpy Reviewer complains. Minimalist deletes. Experiments decide. Then recurse.

## Validation and troubleshooting

```sh
python -m unittest discover -s "$SKILL/scripts" -p 'test_*.py'
python "$SKILL/scripts/deepseek_flash.py" smoke
```

Unit tests mock transport and never call external services; they are not connected to Cargo tests. `smoke` is a separately invoked tiny paid request (16 output tokens maximum, 30 seconds, no retries) and fails clearly without a network request when the environment key is absent. It passes only for a complete `OK` reply.

For 401, check the inherited environment credential and its validity; 402 means credits; 403 means access/policy; 404 means endpoint/model availability—run `discover`. For 429/5xx, bounded retries handle transient failures; timeout/network retries can duplicate billing, and returned usage describes only the successful response. No retries for authentication, credit, malformed input, or certificate failures. Narrow oversized requests. Empty/reasoning-only responses require checking output budget and model compatibility. `discover` queries the public catalog without authentication and never picks a substitute.

Sources: [OpenRouter API](https://openrouter.ai/docs/api/reference/overview), [model catalog](https://openrouter.ai/api/v1/models), [usage accounting](https://openrouter.ai/docs/guides/guides/usage-accounting), [reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [Codex skill layout](https://developers.openai.com/codex/skills).
