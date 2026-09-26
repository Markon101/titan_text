#!/usr/bin/env python3
"""OpenRouter DeepSeek 4.1 Flash Subagent Engine.

Provides autonomous and interactive subagent delegation to DeepSeek V4.1 Flash via OpenRouter.
Permanently configured with API key and default parameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
from urllib.parse import urlsplit


DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
PERMANENT_OPENROUTER_KEY = ""
STALE_OPENROUTER_KEY = ""
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

ROLE_PROMPTS: dict[str, str] = {
    "task-auditor": (
        "You are an adversarial task auditor and algorithmic falsifier. "
        "Your duty is to detect generator leakage, trivial shortcuts, split contamination, "
        "constant-output heuristics, and distribution collapse in sequence modeling datasets. "
        "Identify exact line numbers, mathematical proofs of shortcuts, and propose the cheapest discriminating control."
    ),
    "adversarial-reviewer": (
        "You are a harsh adversarial reviewer and skeptic. "
        "Scrutinize all claims, metrics, gradients, and experimental narratives. "
        "Look for mundane non-recurrent explanations (e.g. feedforward depth specialization, "
        "temperature collapse, right-copy heuristics) masquerading as emergent dynamical phenomena. "
        "Propose concrete falsification experiments."
    ),
    "architectural-minimalist": (
        "You are an architectural minimalist following Occam's razor. "
        "Identify components, code paths, and configurations that do not earn their complexity. "
        "Spot dead code, pseudo-physics baggage (e.g. continuum hydrodynamic metrics on discrete 1D grids), "
        "and unneeded abstractions. Propose the simplest conventional competitor."
    ),
    "code-reviewer": (
        "You are a rigorous systems and deep learning code reviewer. "
        "Inspect code for state mutation bugs, gradient detachment, loss masking errors, "
        "shape mismatches, silent no-ops, unseeded RNGs, and boundary conditions."
    ),
    "experiment-designer": (
        "You are an empirical experiment designer. "
        "Design minimal, high-information-gain experiments that cleanly separate competing hypotheses "
        "within compute constraints. Specify baseline controls, seed sweeps, and ablation metrics."
    ),
    "falsification-arbiter": (
        "You are a scientific falsification arbiter. "
        "When two models or personas disagree, identify CLAIM A, CLAIM B, WHY THEY DIFFER, "
        "and formulate a DISCRIMINATING EXPERIMENT. Model reputation never settles disagreements; data does."
    ),
    "coder": (
        "You are an expert systems programmer. "
        "Provide minimal, robust, bug-free implementations and patches with comprehensive tests."
    ),
    "researcher": (
        "You are an evidence-driven AI research scientist. "
        "Synthesize theoretical formulations, audit experimental data, and report findings with strict precision."
    ),
    "ideation-agent": (
        "You are an exploratory AI research ideation scientist. "
        "Your purpose is broad, creative conceptual search across neuroscience, cellular automata, "
        "dynamical systems, information theory, control theory, recurrent computation, algorithm learning, "
        "and statistical mechanics. Propose novel tasks, new causal interventions, counterfactual state-transplants, "
        "subspace interventions, and minimal synthetic worlds. "
        "Generate both high-value practical experiments and high-risk/high-information unconventional ideas."
    ),
    "dynamics-agent": (
        "You are a mathematical physicist and dynamical systems specialist. "
        "Analyze phase space topologies, Lyapunov spectra, attractors, fixed points, energy dissipation, "
        "and continuum limits in discrete and recurrent cellular systems."
    ),
    "statistical-agent": (
        "You are a biostatistician and empirical measurement specialist. "
        "Formulate rigorous pre-registered hypothesis tests, power calculations, Cohen's d effect sizes, "
        "bootstrap confidence intervals, and distinguish signals from seed noise."
    ),
    "skeptical-agent": (
        "You are a relentless skeptical critic and deflationary analyst. "
        "Identify the most mundane, uninteresting explanation for any positive result: "
        "feedforward depth specialization, label imbalance, marginal preference, token leakage, "
        "or initialization artifacts."
    ),
}


class SubagentError(Exception):
    def __init__(self, message: str, audit: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit = audit or {}


def load_api_key() -> str:
    """Read environment variable or fall back to the embedded permanent key."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key == STALE_OPENROUTER_KEY:
        config_key_file = Path.home() / ".config" / "openrouter" / "api_key"
        if config_key_file.is_file():
            try:
                key = config_key_file.read_text().strip()
            except Exception:
                key = ""
        if not key:
            key = PERMANENT_OPENROUTER_KEY
    if key and (not re.fullmatch(r"[A-Za-z0-9._-]+", key) or len(key) < 16):
        raise SubagentError("OPENROUTER_API_KEY has an invalid format; value withheld.")
    return key


def sha256_digest(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def read_file_span(spec: str, root: Path) -> dict[str, Any]:
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", str(spec))
    if match:
        filename, start_line, end_line = match[1], int(match[2]), int(match[3])
    else:
        filename, start_line, end_line = str(spec), None, None

    path = (root / filename).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError as e:
        raise SubagentError(f"Selected path '{filename}' escapes workspace root.") from e

    if not path.is_file():
        raise SubagentError(f"Selected file '{rel}' does not exist.")

    if path.stat().st_size > 2 * 1024 * 1024:
        raise SubagentError(f"File '{rel}' exceeds 2 MiB selection limit.")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    if start_line is not None and end_line is not None:
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            raise SubagentError(f"Invalid line range {start_line}-{end_line} for '{rel}' ({len(lines)} lines).")
        selected_text = "".join(lines[start_line - 1 : end_line])
        span_str = f"{start_line}-{min(end_line, len(lines))}"
    else:
        selected_text = text
        span_str = f"1-{len(lines)}"

    digest_info = sha256_digest(selected_text)
    return {
        "file": str(rel),
        "span": span_str,
        "content": selected_text,
        "bytes": digest_info["bytes"],
        "sha256": digest_info["sha256"],
    }


def call_openrouter(
    payload: dict[str, Any],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 300,
    retries: int = 2,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/antigravity-ai/titan_text",
        "X-Title": "Titan Text Subagent",
    }
    last_error = ""

    for attempt in range(retries + 1):
        req = urllib.request.Request(endpoint, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_output = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw_output)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
                api_err = err_json.get("error", {}).get("message", err_body)
            except Exception:
                api_err = err_body
            last_error = f"HTTP {e.code}: {api_err}"
            if attempt < retries and e.code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            raise SubagentError(f"OpenRouter API error: {last_error}")
        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise SubagentError(f"OpenRouter network error: {last_error}")
        except TimeoutError:
            last_error = f"Timeout after {timeout}s"
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise SubagentError(f"OpenRouter API timed out after {timeout}s.")
        except json.JSONDecodeError as e:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise SubagentError(f"Invalid JSON returned by OpenRouter: {e}") from e

    raise SubagentError(f"OpenRouter API failed after {retries + 1} attempts: {last_error}")



def run_subagent(
    task: str,
    *,
    role: str = "researcher",
    files: list[str] | None = None,
    context: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    enable_reasoning: bool = False,
    json_answer: bool = False,
    root: str | Path = ".",
    timeout: int = 300,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    api_key = load_api_key()

    system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["researcher"])
    if json_answer:
        system_prompt += " You must respond strictly with a valid JSON object."

    selected_files = []
    file_contents = []
    if files:
        for f in files:
            sel = read_file_span(f, root_path)
            selected_files.append({"file": sel["file"], "span": sel["span"], "sha256": sel["sha256"]})
            file_contents.append(f"--- FILE: {sel['file']} ({sel['span']}) ---\n{sel['content']}\n")

    user_content_parts = []
    if file_contents:
        user_content_parts.append("\n".join(file_contents))
    if context:
        user_content_parts.append(f"--- CONTEXT ---\n{context}\n")
    user_content_parts.append(f"--- TASK ---\n{task}")

    user_prompt = "\n".join(user_content_parts)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if enable_reasoning:
        payload["reasoning"] = {"enabled": True}
    else:
        payload["reasoning"] = {"exclude": True, "enabled": False}
        payload["provider"] = {"require_parameters": True}

    if json_answer:
        payload["response_format"] = {"type": "json_object"}

    start_time = datetime.now(timezone.utc)
    response = call_openrouter(payload, api_key, timeout=timeout)
    end_time = datetime.now(timezone.utc)

    choices = response.get("choices", [])
    if not choices:
        raise SubagentError("OpenRouter response contained no choices.")

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning")
    if not content.strip() and reasoning:
        content = reasoning

    parsed_json = None
    if json_answer and content:
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as e:
            raise SubagentError(f"Model failed to produce valid JSON: {e}") from e

    usage = response.get("usage", {})

    return {
        "status": "ok",
        "model": response.get("model", model),
        "role": role,
        "purpose": task,
        "answer": content,
        "parsed_json": parsed_json,
        "reasoning": reasoning if enable_reasoning else None,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "files": selected_files,
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "created_at_utc": start_time.isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenRouter DeepSeek 4.1 Flash Subagent CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # smoke
    subparsers.add_parser("smoke", help="Perform a fast connectivity test")

    # discover
    subparsers.add_parser("discover", help="Verify DeepSeek 4.1 Flash availability on OpenRouter")

    # run
    run_parser = subparsers.add_parser("run", help="Run a single subagent task")
    run_parser.add_argument("--task", default="", help="Task or research instruction")
    run_parser.add_argument("--task-file", help="Path to file containing task or research instruction")
    run_parser.add_argument("--role", default="researcher", choices=list(ROLE_PROMPTS.keys()))
    run_parser.add_argument("--file", action="append", dest="files", default=[])
    run_parser.add_argument("--context", default="")
    run_parser.add_argument("--context-file")
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--max-tokens", type=int, default=4096)
    run_parser.add_argument("--timeout", type=int, default=300, help="Total request timeout in seconds")
    run_parser.add_argument("--reasoning", action="store_true", help="Enable DeepSeek reasoning tokens")
    run_parser.add_argument("--json-answer", action="store_true", help="Request structured JSON output")
    run_parser.add_argument("--output-json", action="store_true", help="Print entire result envelope as JSON")

    # batch
    batch_parser = subparsers.add_parser("batch", help="Run batch jobs from a JSON specification file")
    batch_parser.add_argument("--file", required=True, help="Path to jobs JSON file")
    batch_parser.add_argument("--output-dir", default="reports/subagents")

    # review-diff
    diff_parser = subparsers.add_parser("review-diff", help="Review current git diff")
    diff_parser.add_argument("--staged", action="store_true")
    diff_parser.add_argument("--target", help="Optional specific file or pattern")
    diff_parser.add_argument("--role", default="adversarial-reviewer", choices=list(ROLE_PROMPTS.keys()))

    args = parser.parse_args(argv)

    try:
        if args.subcommand == "smoke":
            res = run_subagent(
                "Reply strictly with 'SUBAGENT_READY'.",
                role="researcher",
                max_tokens=20,
                temperature=0.0,
            )
            print(f"Smoke test successful! Model: {res['model']}, Answer: {res['answer'].strip()}")
            return 0

        elif args.subcommand == "discover":
            api_key = load_api_key()
            cmd = [
                "curl", "-sS",
                "https://openrouter.ai/api/v1/models",
                "-H", f"Authorization: Bearer {api_key}",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            models = json.loads(proc.stdout.decode()).get("data", [])
            matches = [m for m in models if "deepseek" in m.get("id", "").lower() and "flash" in m.get("id", "").lower()]
            print(f"Found {len(matches)} DeepSeek Flash model(s) on OpenRouter:")
            for m in matches:
                print(f" - ID: {m.get('id')} | Name: {m.get('name')} | Context: {m.get('context_length')}")
            return 0

        elif args.subcommand == "run":
            task = args.task
            if args.task_file:
                task = Path(args.task_file).read_text().strip()
            if not task:
                raise SubagentError("Either --task or --task-file must be provided and non-empty.")

            ctx = args.context
            if args.context_file:
                ctx = (ctx + "\n" + Path(args.context_file).read_text()).strip()

            res = run_subagent(
                task,
                role=args.role,
                files=args.files,
                context=ctx,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_reasoning=args.reasoning,
                json_answer=args.json_answer,
                timeout=args.timeout,
            )
            if args.output_json:
                print(json.dumps(res, indent=2))
            else:
                print(f"=== [{res['role'].upper()}] (Model: {res['model']}, Time: {res['elapsed_seconds']:.2f}s) ===")
                print(res["answer"])
            return 0

        elif args.subcommand == "batch":
            jobs_file = Path(args.file)
            jobs = json.loads(jobs_file.read_text())
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running {len(jobs)} subagent job(s)...")
            results = []
            for i, job in enumerate(jobs, 1):
                name = job.get("name", f"job_{i}")
                print(f"[{i}/{len(jobs)}] Running {name} ({job.get('role', 'researcher')})...")
                res = run_subagent(
                    job["task"],
                    role=job.get("role", "researcher"),
                    files=job.get("files", []),
                    context=job.get("context", ""),
                    max_tokens=job.get("max_tokens", 2048),
                    enable_reasoning=job.get("reasoning", False),
                    json_answer=job.get("json_answer", False),
                )
                res["name"] = name
                results.append(res)
                job_out = out_dir / f"{name}.json"
                job_out.write_text(json.dumps(res, indent=2))
                print(f"Saved: {job_out}")

            summary_out = out_dir / "batch_summary.json"
            summary_out.write_text(json.dumps(results, indent=2))
            print(f"All jobs completed. Summary saved to {summary_out}")
            return 0

        elif args.subcommand == "review-diff":
            cmd = ["git", "diff"]
            if args.staged:
                cmd.append("--staged")
            if args.target:
                cmd.extend(["--", args.target])
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            diff_text = proc.stdout.decode("utf-8")
            if not diff_text.strip():
                print("No git diff detected to review.")
                return 0

            task = (
                "Review this git diff thoroughly. Identify potential bugs, answer leaks, "
                "shortcut vulnerabilities, regression risks, and architectural bloat. "
                "Cite concrete line changes and provide actionable recommendations."
            )
            res = run_subagent(
                task,
                role=args.role,
                context=diff_text,
                max_tokens=2500,
            )
            print(f"=== GIT DIFF REVIEW [{args.role.upper()}] ===")
            print(res["answer"])
            return 0

    except SubagentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
