# DeepSeek Flash delegation

A reusable Codex skill and Python helper for bounded independent reviews through OpenRouter. Python 3.10+ and curl 8.4+ are required; no pip packages.

## Environment setup

Supply `OPENROUTER_API_KEY` in the inherited environment using a trusted launcher or secret manager. Do not paste a literal key into shell commands or repository files. The helper reads no credential files. Missing credentials fail clearly before any paid request; public discovery and dry runs require no key.

Optional non-secret configuration:

- `DEEPSEEK_FLASH_MODEL`: model ID override. Default `deepseek/deepseek-v4.1-flash`, verified against the public catalog on 2026-09-14.
- `OPENROUTER_BASE_URL`: trusted HTTPS API root; default `https://openrouter.ai/api/v1`. An override receives both authentication and selected input.

The supported local location is `${CODEX_HOME:-$HOME/.codex}/skills/deepseek-flash`, containing `SKILL.md`, `agents/openai.yaml`, and `scripts/`. The repository source is `skills/deepseek-flash`.

## Usage

Run from the repository you want to inspect:

```sh
SKILL="${CODEX_HOME:-$HOME/.codex}/skills/deepseek-flash"
python "$SKILL/scripts/deepseek_flash.py" discover
python "$SKILL/scripts/deepseek_flash.py" run \
  --task 'Find task leakage; give the cheapest falsifying control.' \
  --role task-auditor --file src/dataset.rs:1-80 --dry-run
```

Review selection metadata and size, then remove `--dry-run` to send only the selected span. Other useful delegations use `--role architectural-minimalist` with a concise architecture description, `--role result-skeptic` with an aggregate result table, or `--mode hypothesis` for explicit brainstorming. `--max-tokens`, `--temperature`, `--timeout`, and `--retries` bound output and transport. Use `--context` for short text or `--context-file` for a selected diff/table. Do not send a whole repository.

The importable `deepseek_flash(task, context=None, files=None, role=..., temperature=None, max_tokens=...)` returns the same JSON envelope as the CLI. See `SKILL.md` for the complete invocation and role guidance.

## Security and audit model

Authentication travels only in curl stdin, never argv, temporary files, or output. Curl's child environment excludes the key. Redirects, curl configuration files, URL credentials, binary checkpoints, obvious secrets, environment files, and paths escaping the selected root are refused. Limits: 16 explicit selections, 2 MiB per source, 128 KiB default serialized request with 512 KiB hard maximum, and 8 MiB response.

Each result identifies purpose, role, model, endpoint, hashes and byte counts of selected inputs, approximate input tokens (UTF-8 bytes / 4, not exact tokenization), numerical provider usage/cost when returned, attempts, and elapsed time. Transport/API failures also retain safe request metadata with null usage; timestamps are UTC. No hidden reasoning fields are saved. The helper emits JSON; callers choose which useful output to preserve. Treat external output as a hypothesis or proposed patch requiring independent verification. Provider retention policies still apply.

Existing `.env.openrouter*` files remain ignored but are never used by this helper. No new credential files are created.

## Validation and troubleshooting

```sh
python -m unittest discover -s "$SKILL/scripts" -p 'test_*.py'
python "$SKILL/scripts/deepseek_flash.py" smoke
```

Unit tests mock the network. The opt-in smoke makes a tiny paid request (16 output-token maximum, 30 seconds, zero retries) and requires a complete `OK` reply. It is never run by ordinary offline or Cargo tests.

Missing environment key: configure the launching process; files and `--env-file` are intentionally unsupported. HTTP 401: the inherited credential is invalid. HTTP 402: insufficient credits. HTTP 403: access/policy. HTTP 404: query `discover`; no model substitution occurs. HTTP 429/5xx and transient network errors receive bounded retries; prior POST attempts may still be billed. TLS/authentication/configuration failures are not retried. An `incomplete` result means truncation or invalid requested JSON and exits nonzero. Narrow oversized input rather than relying on truncation.

API references: [OpenRouter API](https://openrouter.ai/docs/api/reference/overview), [public model catalog](https://openrouter.ai/api/v1/models).
