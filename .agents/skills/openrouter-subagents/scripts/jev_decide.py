#!/usr/bin/env python3
"""Jev Structured Decision Helper via OpenRouter.

Provides fast, cheap, typed System-1 decisions (classification, scoring, routing,
uncertainty decomposition, and evidence verification) using TypeSafe's Jev model on OpenRouter.

Jev does not generate conversational prose. It evaluates an input state against
explicit criteria and returns model probability distributions (not empirical calibration) and structured choices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request


PINNED_JEV_MODEL = "typesafe/jev-1.13"
LATEST_JEV_MODEL = "~typesafe/jev-latest"
DEFAULT_JEV_MODEL = PINNED_JEV_MODEL
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# Standard decision schemas required by the cognitive governor
DECISION_SCHEMAS: dict[str, dict[str, Any]] = {
    "THINKING_BUDGET": {
        "question_id": "thinking_budget",
        "type": "choice",
        "instructions": (
            "Assess the cognitive and thinking budget required for this engineering/research task. "
            "Consider task complexity, uncertainty, and blast radius."
        ),
        "criteria": {
            "minimal": "Trivial or mechanical task: one-step fix, typo, simple unit test, routine lookup, clear instructions.",
            "normal": "Standard coding or research task: multi-step implementation, standard refactoring, well-scoped analysis.",
            "deep": "Complex algorithmic work, subtle multi-component debugging, architecture redesign, or non-trivial experiment.",
            "escalate": "High ambiguity, critical cross-domain bug, profound architectural dilemma, or contradictory core mechanisms."
        },
        "default": "normal"
    },
    "NEXT_ACTION": {
        "question_id": "next_action",
        "type": "choice",
        "instructions": (
            "Given the current progress, evidence, and state, what is the single most productive next action? "
            "Prefer acting or testing over generic reflection if sufficient information exists."
        ),
        "criteria": {
            "act": "Execute the concrete implementation or edit directly; enough information and confidence already exist.",
            "test": "Run deterministic tests, benchmarks, or validation harness to get empirical feedback.",
            "continue_reasoning": "Perform one more focused conceptual or mathematical reasoning pass before acting.",
            "targeted_read": "Read a specific file span, test, or error log to fill a concrete information gap.",
            "broad_read": "Explore wider repository context or documentation to resolve fundamental unknown context.",
            "spawn_critic": "Spawn DeepSeek adversarial reviewer or critic to audit for subtle shortcuts, leaks, or flaws.",
            "spawn_coder": "Spawn Codex CLI to implement or review a self-contained code module or patch.",
            "escalate": "Escalate to a deeper/expensive reasoning model (e.g., Codex with Astra) due to intractable complexity.",
            "stop": "Goal is satisfied, task is complete, or further iterations yield diminishing returns."
        },
        "default": "act"
    },
    "UNCERTAINTY_TYPE": {
        "question_id": "uncertainty_type",
        "type": "choice",
        "instructions": "Identify the primary source of uncertainty in the current task or problem state.",
        "criteria": {
            "implementation": "Uncertainty is code-local: syntax, types, API details, boundary conditions, or edge cases.",
            "missing_evidence": "Uncertainty stems from lack of empirical data, test outputs, manifests, or benchmark traces.",
            "conceptual": "Uncertainty is theoretical: mechanism disputes, mathematical formulation, or architectural design.",
            "requirements": "Uncertainty is in task scope, user intent, or underspecified constraints.",
            "conflicting_evidence": "Empirical data, test outputs, or agent claims directly contradict each other.",
            "low_uncertainty": "State is well-understood, requirements are clear, and implementation path is straightforward."
        },
        "default": "low_uncertainty"
    },
    "CRITIC_REQUIRED": {
        "question_id": "critic_required",
        "type": "choice",
        "instructions": "Determine what external critic or reviewer, if any, is required before proceeding.",
        "criteria": {
            "none": "No critic required. The task is routine, verified, or low risk.",
            "DeepSeek": "Cheap independent critic: check for generator shortcuts, distribution collapse, or ideation.",
            "Codex": "Code-local reviewer: inspect patch for type errors, mutation bugs, or boundary logic.",
            "expensive_reviewer": "Deep architectural escalation required for high-risk system-level changes."
        },
        "default": "none"
    },
    "EVIDENCE_STATUS": {
        "question_id": "evidence_status",
        "type": "choice",
        "instructions": "Evaluate whether the supplied claims are supported by the provided empirical evidence or test logs.",
        "criteria": {
            "supported": "Claims are directly corroborated by empirical measurements, test passes, or mathematical proofs.",
            "partially_supported": "Plausible with partial evidence, but lacks full proof, controls, or comprehensive edge cases.",
            "insufficient": "Claims lack empirical measurements, traces, or valid controls. Pure speculation or assertion.",
            "contradicted": "Supplied empirical results, test runs, or logs directly contradict or falsify the claims."
        },
        "default": "insufficient"
    },
    "PLAUSIBILITY": {
        "question_id": "plausibility",
        "type": "choice",
        "instructions": "Evaluate the scientific and engineering plausibility of this hypothesis or claim.",
        "criteria": {
            "highly_plausible": "Directly aligned with established physics, mathematics, and empirical baselines.",
            "plausible": "Reasonable mechanism without obvious physical or mathematical contradictions.",
            "unlikely": "Relies on strained assumptions, coincidences, or unverified emerging phenomena.",
            "implausible": "Violates conservation laws, causality, or mathematical definitions."
        },
        "default": "plausible"
    },
    "CONTRADICTION_RISK": {
        "question_id": "contradiction_risk",
        "type": "choice",
        "instructions": "Evaluate the risk that this proposal contradicts existing empirical measurements, tests, or architecture invariants.",
        "criteria": {
            "low": "Consistent with all known test logs, invariants, and constraints.",
            "moderate": "Tension with some secondary observations or performance metrics.",
            "high": "Direct tension with primary benchmark results or architecture rules.",
            "critical": "Blatantly contradicts established ground truth measurements or manifests."
        },
        "default": "low"
    },
    "IMPLEMENTATION_RISK": {
        "question_id": "implementation_risk",
        "type": "choice",
        "instructions": "Assess the blast radius and regression risk of implementing this change.",
        "criteria": {
            "minimal": "Self-contained helper, doc change, or isolated unit test.",
            "low": "Local function modification with clear inputs and outputs.",
            "moderate": "Cross-module refactoring touching multiple components or traits.",
            "high": "Core tensor operations, loss masking, memory layout, or distributed harness."
        },
        "default": "low"
    },
    "NOVELTY": {
        "question_id": "novelty",
        "type": "choice",
        "instructions": "Assess the novelty and diversity contribution of this idea or mechanism.",
        "criteria": {
            "derivative": "Direct paraphrase or restatement of already known mechanisms.",
            "incremental": "Standard parameter tweak or slight variation of conventional approaches.",
            "novel": "Unconventional mechanism, new counterfactual probe, or cross-domain connection.",
            "radical": "Fundamentally new paradigm or counter-intuitive hypothesis."
        },
        "default": "incremental"
    },
    "TESTABILITY": {
        "question_id": "testability",
        "type": "choice",
        "instructions": "Evaluate how cheaply and decisively this hypothesis or change can be tested empirically.",
        "criteria": {
            "immediately_testable": "Testable via short deterministic test (< 5s) or existing harness.",
            "requires_harness": "Requires writing a short synthetic probe or targeted unit test (< 60s).",
            "expensive_test": "Requires multi-seed training run, GPU cluster sweep, or hours of compute.",
            "untestable": "Vague conjecture or unfalsifiable metaphysical claim."
        },
        "default": "immediately_testable"
    },
    "INFORMATION_GAIN": {
        "question_id": "information_gain",
        "type": "choice",
        "instructions": "Estimate the expected reduction in system uncertainty if this action or test is executed.",
        "criteria": {
            "negligible": "Confirms already known facts; minimal or no new information gained.",
            "moderate": "Narrows down candidate causes or eliminates one minor branch.",
            "high": "Decisively separates competing hypotheses or identifies root cause.",
            "decisive": "Resolves foundational architectural dilemma across entire system."
        },
        "default": "moderate"
    },
    "VALUE_OF_REASONING": {
        "question_id": "value_of_reasoning",
        "type": "choice",
        "instructions": "Evaluate the expected marginal value of purchasing another unit of reasoning versus acting or testing now.",
        "criteria": {
            "diminishing_returns": "Enough clarity exists; further reasoning yields sludge. Act or test immediately.",
            "worthwhile": "One more focused pass will likely clarify boundary conditions or reduce risk.",
            "high_value": "High ambiguity with multiple conflicting mechanisms; reasoning has strong expected ROI.",
            "critical": "Catastrophic blindspot or profound paradox requiring deep multi-perspective analysis."
        },
        "default": "diminishing_returns"
    },
    "CONTEXT_SUFFICIENCY": {
        "question_id": "context_sufficiency",
        "type": "choice",
        "instructions": "Judge whether the currently available context is sufficient to make a sound decision.",
        "criteria": {
            "sufficient": "Sufficient evidence and context available to proceed with confidence.",
            "needs_targeted_read": "Missing a specific function, line span, test log, or error trace.",
            "needs_broad_read": "Missing overall architectural context, design doc, or subsystem relationship.",
            "missing_external_data": "Requires running an external command or inspecting uncommitted git state."
        },
        "default": "sufficient"
    },
    "STOP_SUITABILITY": {
        "question_id": "stop_suitability",
        "type": "choice",
        "instructions": "Determine whether this line of inquiry or task execution should terminate.",
        "criteria": {
            "stop_now": "Objective satisfied or diminishing returns reached; terminate cleanly.",
            "test_first": "Run verification test first before terminating.",
            "continue_branching": "Uncertainty remains high with unexplored high-value branches.",
            "escalate": "Blocked on fundamental conceptual paradox requiring escalation."
        },
        "default": "stop_now"
    },
}


def load_api_key() -> str:
    """Read OpenRouter API key from env or standard config locations."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        config_key_file = Path.home() / ".config" / "openrouter" / "api_key"
        if config_key_file.is_file():
            try:
                key = config_key_file.read_text().strip()
            except Exception:
                key = ""
    if not key:
        # Fallback to key in subagent.py if available
        try:
            from subagent import PERMANENT_OPENROUTER_KEY, STALE_OPENROUTER_KEY
            if key != STALE_OPENROUTER_KEY:
                key = PERMANENT_OPENROUTER_KEY
        except Exception:
            pass
    return key


def call_jev(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, Any],
    *,
    model: str = DEFAULT_JEV_MODEL,
    api_key: str | None = None,
    timeout: int = 45,
    retries: int = 2,
) -> dict[str, Any]:
    """Call OpenRouter Decisions API with state and typed questions."""
    key = api_key or load_api_key()
    if not key:
        raise RuntimeError("No OpenRouter API key found for Jev decisions.")

    payload = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/antigravity-ai/titan_text",
        "X-Title": "Titan Text Jev Governor",
    }

    last_error = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(OPENROUTER_DECISIONS_URL, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {err_body}"
            if attempt < retries and e.code in (429, 500, 502, 503, 504):
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter Jev API error: {last_error}")
        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter Jev network error: {last_error}")
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter Jev failed: {last_error}")

    raise RuntimeError(f"OpenRouter Jev failed after {retries + 1} attempts: {last_error}")


def jev_decide(
    decision_type: str,
    state: str | dict[str, Any] | list[Any],
    *,
    model: str = DEFAULT_JEV_MODEL,
    custom_schema: dict[str, Any] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    """Execute a structured decision query via Jev.

    Preserves full probability distributions and returns a clean, structured envelope.
    On failure or network disruption, gracefully falls back to sensible heuristics
    so that Jev serves as an accelerator, never a blocker.
    """
    start_time = datetime.now(timezone.utc)
    decision_key = decision_type.upper()

    # Determine question schema
    if custom_schema:
        schema = custom_schema
        q_id = schema.get("question_id", "decision")
        q_def = {
            "type": schema.get("type", "choice"),
            "instructions": schema.get("instructions", "Make a decision based on state."),
            "criteria": schema.get("criteria", {}),
        }
        default_val = schema.get("default", next(iter(q_def["criteria"].keys()), "unknown"))
    elif decision_key in DECISION_SCHEMAS:
        schema = DECISION_SCHEMAS[decision_key]
        q_id = schema["question_id"]
        q_def = {
            "type": schema["type"],
            "instructions": schema["instructions"],
            "criteria": schema["criteria"],
        }
        default_val = schema["default"]
    else:
        # Fallback to generic choice if unknown type
        return {
            "status": "error",
            "decision_type": decision_type,
            "decision": "unknown",
            "error": f"Unknown decision type: {decision_type}. Available: {list(DECISION_SCHEMAS.keys())}",
            "fallback_applied": True,
        }

    questions_payload = {q_id: q_def}

    try:
        raw_res = call_jev(state, questions_payload, model=model, timeout=timeout)
        if not isinstance(raw_res, dict):
            raise ValueError("Invalid Jev response envelope")
        answers = raw_res.get("answers")
        ans = answers.get(q_id) if isinstance(answers, dict) else None
        if not isinstance(ans, dict):
            raise ValueError("Missing requested Jev answer")
        choice = ans.get("choice") or ans.get("label")
        if not isinstance(choice, str) or choice not in q_def["criteria"]:
            raise ValueError("Unknown or missing decision label")
        probabilities = ans.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError("Missing probability distribution")
        if any(k not in q_def["criteria"] for k in probabilities):
            raise ValueError("Unknown probability label")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Invalid probability value")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
            raise ValueError("Probability distribution must sum to one")
        if choice not in probabilities:
            raise ValueError("Decision missing from probability distribution")
        confidence = ans.get("confidence", probabilities[choice])
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        usage = raw_res.get("usage", {})
        entropy = calculate_entropy(probabilities)

        return {
            "status": "ok",
            "decision_type": decision_type,
            "decision": choice,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "entropy": entropy,
            "model": raw_res.get("model", model),
            "cost_usd": usage.get("cost", 0.0),
            "usage": usage,
            "elapsed_seconds": elapsed,
            "fallback_applied": False,
        }

    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        # Graceful fallback: Never crash the calling agent!
        # A transport/validation failure is not a model probability distribution.
        return {
            "status": "fallback",
            "decision_type": decision_type,
            "decision": "inconclusive" if custom_schema else default_val,
            "confidence": 0.0,
            "probabilities": {},
            "entropy": None,
            "model": model,
            "error": str(e),
            "elapsed_seconds": elapsed,
            "fallback_applied": True,
        }


def calculate_entropy(probabilities: dict[str, float]) -> float:
    """Calculate normalized Shannon entropy (0.0 = total certainty, 1.0 = maximum uncertainty)."""
    if not probabilities or len(probabilities) <= 1:
        return 0.0
    k = len(probabilities)
    max_entropy = math.log2(k)
    ent = 0.0
    for p in probabilities.values():
        if p > 0.0:
            ent -= p * math.log2(p)
    return round(ent / max_entropy if max_entropy > 0 else 0.0, 4)


def _summarize_ensemble(decision_type: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    # Failed calls cannot vote or manufacture unanimity.
    valid = [r for r in results if r.get("status") == "ok"
             and not r.get("fallback_applied") and r.get("probabilities")]
    keys = set().union(*(r["probabilities"] for r in valid))
    avg = {k: sum(r["probabilities"].get(k, 0.0) for r in valid) / len(valid) for k in keys}
    variance = {k: sum((r["probabilities"].get(k, 0.0) - avg[k]) ** 2 for r in valid) / len(valid) for k in keys}
    choice = max(sorted(avg), key=avg.get) if avg else "inconclusive"
    complete = bool(valid) and len(valid) == len(results)
    return {
        "status": "ok" if complete else ("partial" if valid else "unavailable"),
        "decision_type": decision_type,
        "decision": choice,
        "trials": len(results),
        "successful_trials": len(valid),
        "failed_trials": len(results) - len(valid),
        "fallback_applied": not complete,
        "confidence": avg.get(choice, 0.0),
        "ensemble_probabilities": avg,
        "variance_per_option": variance,
        "mean_variance": sum(variance.values()) / len(variance) if variance else None,
        "shannon_entropy": calculate_entropy(avg) if avg else None,
        "is_unanimous": complete and all(r["decision"] == choice for r in valid),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in results),
        "results": results,
    }


def jev_ensemble(
    decision_type: str,
    state: str | dict[str, Any] | list[Any],
    *,
    trials: int = 3,
    model: str = DEFAULT_JEV_MODEL,
    timeout: int = 45,
    custom_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare repeated judgments; agreement is not independent empirical evidence."""
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 8:
        raise ValueError("trials must be between 1 and 8")
    results = [jev_decide(decision_type, state, model=model, timeout=timeout,
                          custom_schema=custom_schema) for _ in range(trials)]
    return _summarize_ensemble(decision_type, results)


COGNITIVE_FRAMINGS: dict[str, str] = {
    "objective": "Objective engineering assessment: Evaluate technical requirements with balanced trade-offs.",
    "risk_averse": "Risk-averse verification lens: Prioritize empirical validation, fail-safes, boundary stress, and minimizing false positives.",
    "high_velocity": "High-velocity efficiency lens: Prioritize token frugality, speed, lean execution, and eliminating unnecessary cognitive bloat.",
}


def jev_framing_ensemble(
    decision_type: str,
    state: str | dict[str, Any] | list[Any],
    *,
    framings: list[str] | None = None,
    model: str = DEFAULT_JEV_MODEL,
    timeout: int = 45,
) -> dict[str, Any]:
    """Execute an ensemble of Jev judgments across distinct cognitive lenses."""
    selected_framings = list(COGNITIVE_FRAMINGS) if framings is None else framings
    if not 1 <= len(selected_framings) <= 8 or len(set(selected_framings)) != len(selected_framings):
        raise ValueError("Provide 1 to 8 distinct framings")
    state_str = state if isinstance(state, str) else json.dumps(state)
    results = []
    for key in selected_framings:
        prefix = COGNITIVE_FRAMINGS.get(key, f"Framing: {key}")
        result = dict(jev_decide(decision_type, f"[{prefix}]\n\n{state_str}", model=model, timeout=timeout))
        result["framing"] = key
        results.append(result)
    summary = _summarize_ensemble(decision_type, results)
    decisions = {r["framing"]: r["decision"] for r in results if r.get("status") == "ok" and not r.get("fallback_applied")}
    summary.update({
        "framings": selected_framings,
        "framing_decisions": decisions,
        "has_framing_divergence": len(set(decisions.values())) > 1,
    })
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev Structured Decision Helper via OpenRouter")
    parser.add_argument("decision_type", help=f"Type of decision: {', '.join(DECISION_SCHEMAS.keys())} or custom")
    parser.add_argument("state", help="State description, code context, evidence, or task summary")
    parser.add_argument("--model", default=DEFAULT_JEV_MODEL, help=f"Jev model ({PINNED_JEV_MODEL} or {LATEST_JEV_MODEL})")
    parser.add_argument("--schema-file", help="Path to JSON file containing custom decision schema")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of independent trials to run in an ensemble")
    parser.add_argument("--json", action="store_true", help="Print entire result envelope as JSON")
    parser.add_argument("--quiet", action="store_true", help="Print only the selected decision label")

    args = parser.parse_args(argv)

    custom_schema = None
    if args.schema_file:
        custom_schema = json.loads(Path(args.schema_file).read_text())

    if not 1 <= args.ensemble <= 8:
        parser.error("--ensemble must be between 1 and 8")
    if args.ensemble > 1:
        result = jev_ensemble(args.decision_type, args.state, trials=args.ensemble, model=args.model, custom_schema=custom_schema)
    else:
        result = jev_decide(
            args.decision_type,
            args.state,
            model=args.model,
            custom_schema=custom_schema,
        )

    if args.quiet:
        print(result["decision"])
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        status_tag = f"[{result['status'].upper()}]"
        if result.get("fallback_applied"):
            status_tag += " (FALLBACK APPLIED)"
        print(f"=== JEV DECISION: {result['decision_type']} {status_tag} ===")
        print(f"Decision:    {result['decision']}")
        print(f"Confidence:  {result.get('confidence', 0.0):.2f}")
        probs = result.get("probabilities")
        if probs:
            probs_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1]))
            print(f"Distribution: {probs_str}")
        if result.get("cost_usd"):
            print(f"Cost:        ${result['cost_usd']:.6f} USD ({result.get('elapsed_seconds', 0.0):.2f}s)")
        if result.get("error"):
            print(f"Error note:  {result['error']}")

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
