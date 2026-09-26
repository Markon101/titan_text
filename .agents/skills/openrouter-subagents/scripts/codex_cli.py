#!/usr/bin/env python3
"""Non-interactive wrapper and helper for Codex CLI.

Provides focused, non-interactive code generation, patch review, debugging,
and test generation using the local Codex CLI installation (configured with gpt-6-astra).
Acts as an independent coding/review specialist when coding uncertainty is high.
Fails gracefully if Codex CLI is unavailable or unauthenticated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


def is_codex_available() -> bool:
    """Check if codex executable is present in PATH."""
    return shutil.which("codex") is not None


def run_codex_exec(
    prompt: str,
    *,
    files: list[str] | None = None,
    context: str = "",
    sandbox: str = "read-only",
    model: str | None = None,
    timeout: int = 180,
    cwd: str | Path | None = None,
    ephemeral: bool = True,
) -> dict[str, Any]:
    """Run Codex CLI non-interactively with focused context and prompt."""
    start_time = datetime.now(timezone.utc)
    if not is_codex_available():
        return {
            "status": "unavailable",
            "error": "Codex CLI executable ('codex') is not installed or not in PATH.",
            "answer": "",
            "elapsed_seconds": 0.0,
            "fallback_applied": True,
        }

    # Construct focused prompt with file snippets and context
    prompt_sections = []
    if files:
        for fspec in files:
            # Handle filename:start-end
            match = re.fullmatch(r"(.+):(\d+)-(\d+)", str(fspec))
            if match:
                fname, s_line, e_line = match[1], int(match[2]), int(match[3])
                fpath = Path(fname)
                if fpath.is_file():
                    lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                    snippet = "\n".join(lines[s_line - 1 : e_line])
                    prompt_sections.append(f"--- FILE: {fname} (lines {s_line}-{e_line}) ---\n{snippet}")
            else:
                fpath = Path(fspec)
                if fpath.is_file():
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    prompt_sections.append(f"--- FILE: {fspec} ---\n{text}")

    if context:
        prompt_sections.append(f"--- CONTEXT ---\n{context}")

    prompt_sections.append(f"--- INSTRUCTIONS ---\n{prompt}")
    full_prompt = "\n\n".join(prompt_sections)

    cmd = ["codex", "exec"]
    if ephemeral:
        cmd.append("--ephemeral")
    if sandbox:
        cmd.extend(["-s", sandbox])
    if model:
        cmd.extend(["-m", model])

    cmd.extend(["--", full_prompt])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Parse token usage and model metadata from stderr if available
        tokens_used = None
        m_tokens = re.search(r"tokens?\s+used\s*[:\n\s]+([0-9,]+)", proc.stderr, re.IGNORECASE)
        if m_tokens:
            tokens_used = int(m_tokens.group(1).replace(",", ""))

        m_model = re.search(r"model:\s*([a-zA-Z0-9._-]+)", proc.stderr, re.IGNORECASE)
        detected_model = m_model.group(1) if m_model else (model or "codex-default")
        requested_model = model or "gpt-6-astra"

        if proc.returncode == 0:
            return {
                "status": "ok",
                "answer": proc.stdout.strip(),
                "model": detected_model,
                "requested_model": requested_model,
                "tokens_used": tokens_used,
                "elapsed_seconds": elapsed,
                "fallback_applied": False,
            }
        else:
            return {
                "status": "error",
                "error": f"Codex exited with code {proc.returncode}: {proc.stderr.strip()}",
                "answer": proc.stdout.strip(),
                "model": detected_model,
                "requested_model": requested_model,
                "tokens_used": tokens_used,
                "elapsed_seconds": elapsed,
                "fallback_applied": True,
            }

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        return {
            "status": "timeout",
            "error": f"Codex CLI timed out after {timeout}s.",
            "answer": "",
            "elapsed_seconds": elapsed,
            "fallback_applied": True,
        }
    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        return {
            "status": "error",
            "error": f"Codex CLI invocation failed: {e}",
            "answer": "",
            "elapsed_seconds": elapsed,
            "fallback_applied": True,
        }


def review_uncommitted_diff(
    instructions: str = "Perform a strict code review on uncommitted changes. Identify defects, edge case bugs, regressions, and type safety issues.",
    *,
    timeout: int = 180,
) -> dict[str, Any]:
    """Run non-interactive code review against uncommitted git changes."""
    # First get diff via git
    try:
        proc = subprocess.run(["git", "diff", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        diff_text = proc.stdout.strip()
        if not diff_text:
            return {
                "status": "ok",
                "answer": "No uncommitted git changes found to review.",
                "elapsed_seconds": 0.0,
                "tokens_used": 0,
                "fallback_applied": False,
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to get git diff: {e}",
            "answer": "",
            "elapsed_seconds": 0.0,
            "fallback_applied": True,
        }

    return run_codex_exec(
        instructions,
        context=diff_text,
        sandbox="read-only",
        timeout=timeout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Codex CLI Non-Interactive Specialist Helper")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # smoke
    subparsers.add_parser("smoke", help="Verify Codex CLI installation and basic execution")

    # exec
    exec_p = subparsers.add_parser("exec", help="Run a focused prompt through Codex CLI")
    exec_p.add_argument("prompt", help="Instruction prompt for Codex")
    exec_p.add_argument("--file", action="append", dest="files", default=[], help="File path or FILE:START-END")
    exec_p.add_argument("--context", default="", help="Additional context or error string")
    exec_p.add_argument("--sandbox", default="read-only", choices=["read-only", "workspace-write"])
    exec_p.add_argument("--model", default=None, help="Codex model override")
    exec_p.add_argument("--timeout", type=int, default=180)
    exec_p.add_argument("--json", action="store_true", help="Print result as JSON envelope")

    # review-diff
    diff_p = subparsers.add_parser("review-diff", help="Review uncommitted changes")
    diff_p.add_argument("--prompt", default="Review uncommitted changes for correctness, edge cases, and regressions.")
    diff_p.add_argument("--json", action="store_true")

    # implement
    impl_p = subparsers.add_parser("implement", help="Propose minimal code implementation or patch")
    impl_p.add_argument("task", help="Implementation task description")
    impl_p.add_argument("--file", action="append", dest="files", default=[], help="Target file(s)")
    impl_p.add_argument("--context", default="")
    impl_p.add_argument("--json", action="store_true")

    # debug
    debug_p = subparsers.add_parser("debug", help="Debug an error trace or failing test")
    debug_p.add_argument("error", help="Error message, stack trace, or failing test output")
    debug_p.add_argument("--file", action="append", dest="files", default=[], help="Suspect file(s)")
    debug_p.add_argument("--json", action="store_true")

    # test-gen
    test_p = subparsers.add_parser("test-gen", help="Generate unit tests for a target module")
    test_p.add_argument("target", help="Module or function description to test")
    test_p.add_argument("--file", action="append", dest="files", default=[], help="Source file(s)")
    test_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.subcommand == "smoke":
        if not is_codex_available():
            print("Codex CLI is NOT available in PATH.")
            return 1
        res = run_codex_exec("Reply strictly with 'CODEX_READY'.", timeout=30)
        if res["status"] == "ok":
            print(f"Codex CLI smoke test successful! Model: {res.get('model')}, Answer: {res['answer'].strip()}")
            return 0
        else:
            print(f"Codex smoke test failed: {res.get('error')}", file=sys.stderr)
            return 1

    elif args.subcommand == "review-diff":
        res = review_uncommitted_diff(args.prompt)
    elif args.subcommand == "implement":
        task_prompt = (
            f"Provide a minimal, correct, robust implementation or patch for the following task:\n{args.task}\n"
            "Format the proposed patch clearly with exact file changes and explanation."
        )
        res = run_codex_exec(task_prompt, files=args.files, context=args.context)
    elif args.subcommand == "debug":
        debug_prompt = (
            f"Analyze the following failure/error and identify the root cause and minimal fix:\n{args.error}\n"
            "Be precise and cite exact line numbers."
        )
        res = run_codex_exec(debug_prompt, files=args.files)
    elif args.subcommand == "test-gen":
        test_prompt = (
            f"Generate rigorous, edge-case unit tests for:\n{args.target}\n"
            "Include boundary conditions, invalid inputs, and deterministic assertions."
        )
        res = run_codex_exec(test_prompt, files=args.files)
    elif args.subcommand == "exec":
        res = run_codex_exec(
            args.prompt,
            files=args.files,
            context=args.context,
            sandbox=args.sandbox,
            model=args.model,
            timeout=args.timeout,
        )
    else:
        parser.print_help()
        return 1

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
    else:
        if res["status"] == "ok":
            print(f"=== CODEX CLI [{res.get('model', 'codex')}] ===")
            print(res["answer"])
            if res.get("tokens_used"):
                print(f"\n[Tokens: {res['tokens_used']:,} | Time: {res.get('elapsed_seconds', 0.0):.2f}s]")
        else:
            print(f"Error ({res['status']}): {res.get('error')}", file=sys.stderr)
            if res.get("answer"):
                print(res["answer"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
