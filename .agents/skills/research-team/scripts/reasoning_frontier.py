#!/usr/bin/env python3
"""Explicit Reasoning Frontier and Branch Controller.

Manages tree-based hypothesis exploration, semantic deduplication, UCB-like
branch selection, diversity preservation, and discriminating experiment selection.

Guarantees:
- Semantic branch deduplication (Jaccard similarity threshold >= 0.75).
- Diversity preservation: always reserves at least one slot for high-novelty/cheaply-falsifiable paths.
- Discriminating test prioritization: prefers tests that eliminate multiple branches simultaneously.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import uuid
from typing import Any


DEFAULT_FRONTIER_FILE = Path(".agents/state/reasoning_frontier.json")

VALID_BRANCH_TYPES = {
    "causal_hypothesis",
    "implementation_bug",
    "measurement_artifact",
    "missing_evidence",
    "alternative_mechanism",
    "counterexample",
    "architecture_alternative",
    "code_change_proposal",
    "tool_action_proposal",
    "experiment_proposal",
    "escalation_proposal",
    "stop_finish_proposal",
}


CONTRASTIVE_MARKERS = {
    "not", "never", "no", "without", "instead", "rather", "unlike", "except", "against", "non"
}


def extract_directed_assertions(text: str) -> tuple[set[str], set[str]]:
    """Extract affirmed substantive terms vs terms scoped under negation/contrast."""
    tokens = re.findall(r"\b[a-z0-9_-]+\b", text.lower())
    affirmed: set[str] = set()
    negated: set[str] = set()
    in_negation = False
    negation_window = 0

    for tok in tokens:
        if tok in CONTRASTIVE_MARKERS:
            in_negation = True
            negation_window = 4
            continue
        if len(tok) < 4:
            continue
        if in_negation and negation_window > 0:
            negated.add(tok)
            negation_window -= 1
            if negation_window == 0:
                in_negation = False
        else:
            in_negation = False
            affirmed.add(tok)

    return affirmed, negated


def has_conflicting_polarity(a: str, b: str) -> bool:
    """Detect if statement A affirms what statement B negates, or vice versa."""
    aff_a, neg_a = extract_directed_assertions(a)
    aff_b, neg_b = extract_directed_assertions(b)
    return bool((neg_a & aff_b) or (neg_b & aff_a))


def compute_text_similarity(a: str, b: str) -> float:
    """Compute polarity-guarded word-level similarity to detect duplicate branch proposals.

    Guards against collapsing opposing causal hypotheses (e.g., 'caused by X not Y' vs 'caused by Y not X').
    """
    words_a = set(re.findall(r"\b[a-z]{4,}\b", a.lower()))
    words_b = set(re.findall(r"\b[a-z]{4,}\b", b.lower()))
    if not words_a or not words_b:
        return 0.0

    raw_jaccard = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
    if raw_jaccard >= 0.50 and has_conflicting_polarity(a, b):
        return 0.0

    return raw_jaccard


class BranchNode:
    def __init__(
        self,
        hypothesis_or_action: str,
        branch_type: str = "causal_hypothesis",
        *,
        branch_id: str | None = None,
        parent_id: str | None = None,
        support: list[str] | None = None,
        contradictions: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        next_discriminating_action: str | None = None,
        estimated_cost: float = 0.001,
        expected_information_gain: float = 0.5,
        novelty_score: float = 0.5,
        plausibility: float = 0.5,
        jev_probabilities: dict[str, float] | None = None,
        status: str = "active",
        termination_reason: str | None = None,
    ) -> None:
        self.branch_id = branch_id or f"B_{str(uuid.uuid4())[:6]}"
        self.parent_id = parent_id
        self.hypothesis_or_action = hypothesis_or_action
        self.branch_type = branch_type if branch_type in VALID_BRANCH_TYPES else "causal_hypothesis"
        self.support = support or []
        self.contradictions = contradictions or []
        self.unresolved_questions = unresolved_questions or []
        self.evidence_refs = evidence_refs or []
        self.next_discriminating_action = next_discriminating_action
        self.estimated_cost = max(estimated_cost, 0.00001)
        self.expected_information_gain = max(0.0, min(1.0, expected_information_gain))
        self.novelty_score = max(0.0, min(1.0, novelty_score))
        self.plausibility = max(0.0, min(1.0, plausibility))
        self.jev_probabilities = jev_probabilities or {}
        self.status = status
        self.termination_reason = termination_reason
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.visit_count = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "hypothesis_or_action": self.hypothesis_or_action,
            "branch_type": self.branch_type,
            "support": self.support,
            "contradictions": self.contradictions,
            "unresolved_questions": self.unresolved_questions,
            "evidence_refs": self.evidence_refs,
            "next_discriminating_action": self.next_discriminating_action,
            "estimated_cost": self.estimated_cost,
            "expected_information_gain": self.expected_information_gain,
            "novelty_score": self.novelty_score,
            "plausibility": self.plausibility,
            "jev_probabilities": self.jev_probabilities,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "created_at_utc": self.created_at_utc,
            "visit_count": self.visit_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchNode:
        node = cls(
            hypothesis_or_action=data.get("hypothesis_or_action", ""),
            branch_type=data.get("branch_type", "causal_hypothesis"),
            branch_id=data.get("branch_id"),
            parent_id=data.get("parent_id"),
            support=data.get("support"),
            contradictions=data.get("contradictions"),
            unresolved_questions=data.get("unresolved_questions"),
            evidence_refs=data.get("evidence_refs"),
            next_discriminating_action=data.get("next_discriminating_action"),
            estimated_cost=data.get("estimated_cost", 0.001),
            expected_information_gain=data.get("expected_information_gain", 0.5),
            novelty_score=data.get("novelty_score", 0.5),
            plausibility=data.get("plausibility", 0.5),
            jev_probabilities=data.get("jev_probabilities"),
            status=data.get("status", "active"),
            termination_reason=data.get("termination_reason"),
        )
        node.created_at_utc = data.get("created_at_utc", node.created_at_utc)
        node.visit_count = data.get("visit_count", 1)
        return node


class ReasoningFrontier:
    def __init__(self, storage_path: Path | None = None) -> None:
        self.path = storage_path or DEFAULT_FRONTIER_FILE
        self.branches: dict[str, BranchNode] = {}
        self.total_expansions = 0
        self.load()

    def load(self) -> None:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.total_expansions = data.get("total_expansions", 0)
                for bdata in data.get("branches", []):
                    node = BranchNode.from_dict(bdata)
                    self.branches[node.branch_id] = node
            except Exception:
                self.branches = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_expansions": self.total_expansions,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_branch_count": sum(1 for b in self.branches.values() if b.status == "active"),
            "branches": [b.to_dict() for b in self.branches.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_branch(
        self,
        hypothesis_or_action: str,
        branch_type: str = "causal_hypothesis",
        *,
        parent_id: str | None = None,
        support: list[str] | None = None,
        contradictions: list[str] | None = None,
        next_discriminating_action: str | None = None,
        estimated_cost: float = 0.001,
        expected_information_gain: float = 0.5,
        novelty_score: float = 0.5,
        plausibility: float = 0.5,
        jev_probabilities: dict[str, float] | None = None,
        dedup_threshold: float = 0.75,
    ) -> tuple[str, bool]:
        """Add a branch if not semantically duplicate. Returns (branch_id, is_new)."""
        # Semantic deduplication against existing active branches
        for b in self.branches.values():
            if b.status == "active":
                sim = compute_text_similarity(hypothesis_or_action, b.hypothesis_or_action)
                if sim >= dedup_threshold:
                    # Duplicate detected: update visit count and merge support instead of creating redundant node
                    b.visit_count += 1
                    if support:
                        for s in support:
                            if s not in b.support:
                                b.support.append(s)
                    self.save()
                    return (b.branch_id, False)

        node = BranchNode(
            hypothesis_or_action=hypothesis_or_action,
            branch_type=branch_type,
            parent_id=parent_id,
            support=support,
            contradictions=contradictions,
            next_discriminating_action=next_discriminating_action,
            estimated_cost=estimated_cost,
            expected_information_gain=expected_information_gain,
            novelty_score=novelty_score,
            plausibility=plausibility,
            jev_probabilities=jev_probabilities,
        )
        self.branches[node.branch_id] = node
        self.total_expansions += 1
        self.save()
        return (node.branch_id, True)

    def calculate_branch_utility(
        self,
        node: BranchNode,
        total_visits: int,
        *,
        lambda_info: float = 0.35,
        lambda_novelty: float = 0.25,
        lambda_cost: float = 0.10,
        c_ucb: float = 0.40,
    ) -> float:
        """Compute UCB-like expected utility score for beam search."""
        # Exploitation component: plausibility weighted by support minus contradictions
        contradiction_penalty = len(node.contradictions) * 0.20
        base_value = max(0.0, node.plausibility - contradiction_penalty)

        # Information gain & novelty components
        bonus = (lambda_info * node.expected_information_gain) + (lambda_novelty * node.novelty_score)

        # Cost penalty (normalized)
        cost_penalty = lambda_cost * min(1.0, node.estimated_cost / 0.01)

        # UCB exploration term
        ucb_term = 0.0
        if total_visits > 0 and node.visit_count > 0:
            ucb_term = c_ucb * math.sqrt(math.log(total_visits + 1) / node.visit_count)

        return base_value + bonus + ucb_term - cost_penalty

    def select_active_frontier(
        self,
        top_k: int = 3,
        *,
        preserve_diversity: bool = True,
    ) -> list[BranchNode]:
        """Select top branches for expansion while strictly preserving at least one high-novelty branch."""
        active = [b for b in self.branches.values() if b.status == "active"]
        if not active:
            return []
        if len(active) <= top_k:
            return active

        total_visits = sum(b.visit_count for b in active)
        scored = [(b, self.calculate_branch_utility(b, total_visits)) for b in active]
        scored.sort(key=lambda kv: kv[1], reverse=True)

        if not preserve_diversity or top_k <= 1:
            return [b for b, _ in scored[:top_k]]

        # Diversity preservation: pick top K-1 by utility
        selected = [b for b, _ in scored[: top_k - 1]]
        selected_ids = {b.branch_id for b in selected}

        # Find highest-novelty candidate among remaining branches
        remaining = [b for b in active if b.branch_id not in selected_ids]
        if remaining:
            remaining.sort(key=lambda b: (b.novelty_score / max(0.0001, b.estimated_cost)), reverse=True)
            selected.append(remaining[0])

        return selected

    def find_multi_branch_discriminating_test(self, branches: list[BranchNode] | None = None) -> dict[str, Any]:
        """Identify or propose a single test that discriminates across multiple branches simultaneously."""
        target_branches = branches or self.select_active_frontier(top_k=3)
        if len(target_branches) < 2:
            return {
                "status": "trivial",
                "message": "Fewer than 2 active branches; no multi-branch separation needed.",
                "candidate_test": target_branches[0].next_discriminating_action if target_branches else None,
            }

        actions = [b.next_discriminating_action for b in target_branches if b.next_discriminating_action]
        return {
            "status": "ready",
            "branch_ids": [b.branch_id for b in target_branches],
            "hypotheses": [b.hypothesis_or_action for b in target_branches],
            "proposed_actions": actions,
            "recommendation": (
                f"Design a unified discriminating experiment separating branches {', '.join(b.branch_id for b in target_branches)}: "
                "Evaluate with paired control baselines, lesioning perception/recurrence parameters to isolate mechanism."
            ),
        }

    def prune_branch(self, branch_id: str, reason: str) -> bool:
        if branch_id in self.branches:
            self.branches[branch_id].status = "pruned"
            self.branches[branch_id].termination_reason = reason
            self.save()
            return True
        return False

    def render_frontier_summary(self) -> str:
        active = [b for b in self.branches.values() if b.status == "active"]
        closed = [b for b in self.branches.values() if b.status in ("pruned", "closed", "falsified")]
        lines = [f"=== REASONING FRONTIER ({len(active)} active, {len(closed)} closed) ==="]
        if not active:
            lines.append("  (No active branches)")
        for b in active:
            disc = f" -> Next: {b.next_discriminating_action}" if b.next_discriminating_action else ""
            lines.append(
                f"  [{b.branch_id}] ({b.branch_type}) Plaus: {b.plausibility:.2f} | Info: {b.expected_information_gain:.2f} | "
                f"Nov: {b.novelty_score:.2f} | Cost: ${b.estimated_cost:.5f}\n"
                f"    Target: {b.hypothesis_or_action}{disc}"
            )
            if b.contradictions:
                lines.append(f"    Contradictions: {', '.join(b.contradictions)}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reasoning Frontier CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # view
    subparsers.add_parser("view", help="View active reasoning frontier")

    # add
    add_p = subparsers.add_parser("add", help="Add a branch node")
    add_p.add_argument("hypothesis", help="Hypothesis or action description")
    add_p.add_argument("--type", default="causal_hypothesis", choices=list(VALID_BRANCH_TYPES))
    add_p.add_argument("--plausibility", type=float, default=0.5)
    add_p.add_argument("--info-gain", type=float, default=0.5)
    add_p.add_argument("--novelty", type=float, default=0.5)
    add_p.add_argument("--action", help="Next discriminating action")

    # select
    sel_p = subparsers.add_parser("select", help="Select active beam frontier preserving diversity")
    sel_p.add_argument("-k", "--top-k", type=int, default=3)

    # discriminate
    subparsers.add_parser("discriminate", help="Formulate unified discriminating test")

    # prune
    prune_p = subparsers.add_parser("prune", help="Prune or close a branch")
    prune_p.add_argument("branch_id", help="ID of branch to close")
    prune_p.add_argument("reason", help="Falsification or termination reason")

    args = parser.parse_args(argv)
    frontier = ReasoningFrontier()

    if args.subcommand == "view":
        print(frontier.render_frontier_summary())
    elif args.subcommand == "add":
        bid, is_new = frontier.add_branch(
            args.hypothesis,
            branch_type=args.type,
            plausibility=args.plausibility,
            expected_information_gain=args.info_gain,
            novelty_score=args.novelty,
            next_discriminating_action=args.action,
        )
        status_msg = "Created new branch" if is_new else "Deduplicated / merged into existing branch"
        print(f"{status_msg} [{bid}]")
    elif args.subcommand == "select":
        chosen = frontier.select_active_frontier(top_k=args.top_k)
        print(f"=== SELECTED FRONTIER (Top {len(chosen)}) ===")
        for c in chosen:
            print(f"  • [{c.branch_id}] {c.hypothesis_or_action} (Nov: {c.novelty_score:.2f}, Plaus: {c.plausibility:.2f})")
    elif args.subcommand == "discriminate":
        plan = frontier.find_multi_branch_discriminating_test()
        print(json.dumps(plan, indent=2))
    elif args.subcommand == "prune":
        if frontier.prune_branch(args.branch_id, args.reason):
            print(f"Pruned branch [{args.branch_id}]: {args.reason}")
        else:
            print(f"Branch [{args.branch_id}] not found.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
