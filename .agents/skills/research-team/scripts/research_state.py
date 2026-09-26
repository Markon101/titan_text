#!/usr/bin/env python3
"""Canonical Research-State Packet Manager.

Maintains compact, persistent task state for long-running scientific research,
Titan experiments, causal interventions, and architecture analysis.

Prevents the research lead (Gemini) from repeatedly rediscovering facts or
re-running dead hypotheses across multi-agent turns.
References raw evidence externally rather than bloating context.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_STATE_FILE = Path(".agents/state/research_packet.json")
DEFAULT_ARCHIVE_FILE = Path(".agents/state/research_packet_archive.json")
SOFT_BUDGET_BYTES = 8192
HARD_BUDGET_BYTES = 16384
AUTO_PRUNE_THRESHOLD_BYTES = SOFT_BUDGET_BYTES


def load_state(path: Path | None = None) -> dict[str, Any]:
    target = path or DEFAULT_STATE_FILE
    if not target.is_file():
        return {
            "topic": "Untitled Research Task",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_hypotheses": [],
            "confirmed_facts": [],
            "supported_interpretations": [],
            "rejected_hypotheses": [],
            "unresolved_questions": [],
            "known_confounds": [],
            "important_experimental_results": [],
            "next_candidate_experiments": [],
            "agent_disagreements": [],
            "uncertainty_structure": {},
            "state_insufficiency_events": [],
        }
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        data.setdefault("supported_interpretations", [])
        data.setdefault("known_confounds", [])
        data.setdefault("next_candidate_experiments", [])
        return data
    except Exception as e:
        return {"error": f"Failed to parse research state: {e}"}


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    target = path or DEFAULT_STATE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(state, indent=2), encoding="utf-8")


class ResearchState:
    def __init__(self, path: Path | None = None, archive_path: Path | None = None) -> None:
        self.path = path or DEFAULT_STATE_FILE
        self.archive_path = archive_path or DEFAULT_ARCHIVE_FILE
        self.data = load_state(self.path)

    def save(self, auto_prune: bool = True) -> None:
        save_state(self.data, self.path)
        if auto_prune and self.path.is_file() and self.path.stat().st_size > AUTO_PRUNE_THRESHOLD_BYTES:
            self.archive_stale()

    def set_topic(self, topic: str) -> None:
        self.data["topic"] = topic
        self.save()

    def add_hypothesis(self, statement: str, confidence: float = 0.5) -> str:
        hid = f"H{len(self.data['current_hypotheses']) + len(self.data['rejected_hypotheses']) + 1}"
        self.data["current_hypotheses"].append({
            "id": hid,
            "statement": statement,
            "confidence": confidence,
            "status": "active",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return hid

    def confirm_fact(self, statement: str, source_measurement: str = "") -> str:
        fid = f"F{len(self.data['confirmed_facts']) + 1}"
        self.data["confirmed_facts"].append({
            "id": fid,
            "statement": statement,
            "source_measurement": source_measurement,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return fid

    def reject_hypothesis(self, hypothesis_id: str, falsified_by: str) -> bool:
        for i, h in enumerate(self.data["current_hypotheses"]):
            if h["id"] == hypothesis_id:
                rejected = self.data["current_hypotheses"].pop(i)
                rejected["falsified_by"] = falsified_by
                rejected["status"] = "rejected"
                rejected["rejected_at_utc"] = datetime.now(timezone.utc).isoformat()
                self.data["rejected_hypotheses"].append(rejected)
                self.save()
                return True
        return False

    def add_established(self, statement: str, source_measurement: str = "") -> str:
        """Register an empirically established result (alias for confirm_fact)."""
        return self.confirm_fact(statement, source_measurement)

    def add_supported(self, statement: str, competing_alternatives: str = "") -> str:
        """Register an interpretation that fits current evidence but has plausible alternatives."""
        self.data.setdefault("supported_interpretations", [])
        sid = f"SUP_{len(self.data['supported_interpretations']) + 1}"
        self.data["supported_interpretations"].append({
            "id": sid,
            "statement": statement,
            "competing_alternatives": competing_alternatives,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return sid

    def add_falsified(self, statement: str, falsified_by: str, is_artifact: bool = False) -> str:
        """Register a falsified or superseded finding, explicitly labeling artifacts."""
        rid = f"FAL_{len(self.data['rejected_hypotheses']) + 1}"
        self.data["rejected_hypotheses"].append({
            "id": rid,
            "statement": statement,
            "falsified_by": falsified_by,
            "status": "artifact_debunked" if is_artifact else "falsified",
            "is_artifact": is_artifact,
            "rejected_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return rid

    def add_confound(self, confound: str, mitigation_or_weakness: str = "") -> str:
        """Register a known confound, leakage possibility, or weakness of current controls."""
        self.data.setdefault("known_confounds", [])
        cid = f"CONF_{len(self.data['known_confounds']) + 1}"
        self.data["known_confounds"].append({
            "id": cid,
            "confound": confound,
            "mitigation_or_weakness": mitigation_or_weakness,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return cid

    def add_discriminating_experiment(self, name: str, target_hypotheses: list[str], protocol: str) -> str:
        """Register a next discriminating experiment capable of separating remaining hypotheses."""
        self.data.setdefault("next_candidate_experiments", [])
        eid = f"DISC_{len(self.data['next_candidate_experiments']) + 1}"
        self.data["next_candidate_experiments"].append({
            "id": eid,
            "name": name,
            "target_hypotheses": target_hypotheses,
            "protocol": protocol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()
        return eid

    def add_question(self, question: str, priority: str = "normal") -> str:
        qid = f"Q{len(self.data['unresolved_questions']) + 1}"
        self.data["unresolved_questions"].append({
            "id": qid,
            "question": question,
            "priority": priority,
            "status": "open",
        })
        self.save()
        return qid

    def resolve_question(self, question_id: str, resolution: str) -> bool:
        for q in self.data["unresolved_questions"]:
            if q["id"] == question_id:
                q["status"] = "resolved"
                q["resolution"] = resolution
                self.save()
                return True
        return False

    def record_experiment_result(
        self,
        experiment_id: str,
        metric: str,
        value: Any,
        control_value: Any = None,
        notes: str = "",
    ) -> None:
        eid = f"EXP_{experiment_id}"
        self.data["important_experimental_results"].append({
            "id": eid,
            "metric": metric,
            "value": value,
            "control_value": control_value,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()

    def record_disagreement(
        self,
        claim_a: str,
        claim_b: str,
        agents: list[str],
        discriminating_test: str,
    ) -> str:
        did = f"D{len(self.data['agent_disagreements']) + 1}"
        self.data["agent_disagreements"].append({
            "id": did,
            "claim_a": claim_a,
            "claim_b": claim_b,
            "agents": agents,
            "discriminating_test": discriminating_test,
            "status": "unresolved",
        })
        self.save()
        return did

    def archive_stale(
        self,
        *,
        max_rejected_kept: int = 5,
        max_resolved_questions_kept: int = 0,
        max_experiments_kept: int = 10,
        archive_path: Path | None = None,
    ) -> dict[str, int]:
        """Move stale/resolved entities into an archive file to guarantee context compactness.

        Salience-aware retention guarantees:
        - NEVER prunes active hypotheses (current_hypotheses).
        - NEVER prunes high-priority unresolved questions.
        - Prunes low-priority questions when exceeding SOFT_BUDGET_BYTES.
        - Prunes normal-priority questions only when exceeding HARD_BUDGET_BYTES.
        """
        target_archive = archive_path or self.archive_path
        target_archive.parent.mkdir(parents=True, exist_ok=True)

        archive_data: dict[str, Any] = {"topic": self.data.get("topic", ""), "archived_batches": []}
        if target_archive.is_file():
            try:
                archive_data = json.loads(target_archive.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 1. Prune questions with salience awareness
        current_size = self.path.stat().st_size if self.path.is_file() else 0
        open_questions = []
        archived_questions = []

        for q in self.data.get("unresolved_questions", []):
            if q.get("status") == "resolved":
                archived_questions.append(q)
            elif current_size > HARD_BUDGET_BYTES and q.get("priority") != "high":
                # Under hard budget, preserve ONLY high-priority
                archived_questions.append(q)
            elif current_size > SOFT_BUDGET_BYTES and q.get("priority") == "low":
                # Under soft budget, prune low-priority
                archived_questions.append(q)
            else:
                open_questions.append(q)

        # Determine effective limits based on budget pressure
        if current_size > HARD_BUDGET_BYTES:
            effective_max_rejected = min(max_rejected_kept, 2)
            effective_max_experiments = min(max_experiments_kept, 3)
        elif current_size > SOFT_BUDGET_BYTES:
            effective_max_rejected = min(max_rejected_kept, 4)
            effective_max_experiments = min(max_experiments_kept, 7)
        else:
            effective_max_rejected = max_rejected_kept
            effective_max_experiments = max_experiments_kept

        # 2. Prune rejected hypotheses beyond effective_max_rejected
        all_rejected = self.data.get("rejected_hypotheses", [])
        if effective_max_rejected == 0:
            to_archive_rejected = list(all_rejected)
            kept_rejected = []
        elif len(all_rejected) > effective_max_rejected:
            to_archive_rejected = all_rejected[:-effective_max_rejected]
            kept_rejected = all_rejected[-effective_max_rejected:]
        else:
            to_archive_rejected = []
            kept_rejected = all_rejected

        # 3. Prune old experiments beyond effective_max_experiments
        all_experiments = self.data.get("important_experimental_results", [])
        if effective_max_experiments == 0:
            to_archive_experiments = list(all_experiments)
            kept_experiments = []
        elif len(all_experiments) > effective_max_experiments:
            to_archive_experiments = all_experiments[:-effective_max_experiments]
            kept_experiments = all_experiments[-effective_max_experiments:]
        else:
            to_archive_experiments = []
            kept_experiments = all_experiments

        total_pruned = len(archived_questions) + len(to_archive_rejected) + len(to_archive_experiments)
        if total_pruned > 0:
            batch = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "archived_questions": archived_questions,
                "archived_rejected_hypotheses": to_archive_rejected,
                "archived_experiments": to_archive_experiments,
            }
            archive_data.setdefault("archived_batches", []).append(batch)
            target_archive.write_text(json.dumps(archive_data, indent=2), encoding="utf-8")

            # Update working state (active hypotheses are never modified or pruned)
            self.data["unresolved_questions"] = open_questions
            self.data["rejected_hypotheses"] = kept_rejected
            self.data["important_experimental_results"] = kept_experiments
            self.data["archived_summary"] = {
                "archive_file": str(target_archive),
                "last_archived_utc": datetime.now(timezone.utc).isoformat(),
                "total_items_archived": sum(
                    len(b.get("archived_questions", [])) + len(b.get("archived_rejected_hypotheses", [])) + len(b.get("archived_experiments", []))
                    for b in archive_data.get("archived_batches", [])
                ),
            }
            # Save without triggering infinite auto_prune loop
            self.save(auto_prune=False)

        return {
            "questions_archived": len(archived_questions),
            "hypotheses_archived": len(to_archive_rejected),
            "experiments_archived": len(to_archive_experiments),
            "total_pruned": total_pruned,
        }

    def retrieve_from_archive(self, item_id: str, archive_path: Path | None = None) -> dict[str, Any] | None:
        """Retrieve an archived item (question, rejected hypothesis, or experiment) by ID.

        Tracks state insufficiency events to monitor whether context budget limits are too aggressive.
        """
        target_archive = archive_path or self.archive_path
        found_item: dict[str, Any] | None = None

        if target_archive.is_file():
            try:
                archive_data = json.loads(target_archive.read_text(encoding="utf-8"))
                for batch in archive_data.get("archived_batches", []):
                    for q in batch.get("archived_questions", []):
                        if q.get("id") == item_id:
                            found_item = q
                            break
                    if found_item:
                        break
                    for h in batch.get("archived_rejected_hypotheses", []):
                        if h.get("id") == item_id:
                            found_item = h
                            break
                    if found_item:
                        break
                    for e in batch.get("archived_experiments", []):
                        if e.get("id") == item_id:
                            found_item = e
                            break
                    if found_item:
                        break
            except Exception:
                pass

        # Track state insufficiency event
        self.data.setdefault("state_insufficiency_events", []).append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "item_id": item_id,
            "found": found_item is not None,
            "source": "archive" if found_item else "none",
        })
        self.save(auto_prune=False)
        return found_item

    def get_budget_status(self) -> dict[str, Any]:
        """Return quantitative context budget status, size ratios, and insufficiency counts."""
        current_size = self.path.stat().st_size if self.path.is_file() else 0
        return {
            "current_bytes": current_size,
            "soft_budget_bytes": SOFT_BUDGET_BYTES,
            "hard_budget_bytes": HARD_BUDGET_BYTES,
            "soft_budget_ratio": round(current_size / SOFT_BUDGET_BYTES, 3) if SOFT_BUDGET_BYTES else 0.0,
            "hard_budget_ratio": round(current_size / HARD_BUDGET_BYTES, 3) if HARD_BUDGET_BYTES else 0.0,
            "exceeds_soft": current_size > SOFT_BUDGET_BYTES,
            "exceeds_hard": current_size > HARD_BUDGET_BYTES,
            "active_hypotheses_count": len(self.data.get("current_hypotheses", [])),
            "high_priority_questions_count": sum(
                1 for q in self.data.get("unresolved_questions", []) if q.get("priority") == "high"
            ),
            "insufficiency_events_count": len(self.data.get("state_insufficiency_events", [])),
        }

    def render_compact_summary(self) -> str:
        """Render a high-density, context-minimized summary for agent ingestion."""
        lines = [f"=== RESEARCH STATE PACKET: {self.data.get('topic', 'Untitled')} ==="]
        lines.append(f"Last updated: {self.data.get('updated_at_utc', '')[:19]}")

        # Archived Summary Note if present
        if self.data.get("archived_summary"):
            asum = self.data["archived_summary"]
            lines.append(f"[ARCHIVE: {asum.get('total_items_archived', 0)} stale items offloaded to {asum.get('archive_file')}]")

        # Confirmed Facts
        lines.append("\n[CONFIRMED FACTS (Ground Truth / Measurements)]")
        if not self.data.get("confirmed_facts"):
            lines.append("  (None recorded yet)")
        for f in self.data.get("confirmed_facts", []):
            src = f" [via {f['source_measurement']}]" if f.get("source_measurement") else ""
            lines.append(f"  • {f['id']}: {f['statement']}{src}")

        # Active Hypotheses
        lines.append("\n[ACTIVE HYPOTHESES]")
        if not self.data.get("current_hypotheses"):
            lines.append("  (None active)")
        for h in self.data.get("current_hypotheses", []):
            lines.append(f"  • {h['id']} (conf: {h.get('confidence', 0.5):.2f}): {h['statement']}")

        # Rejected Hypotheses
        if self.data.get("rejected_hypotheses"):
            lines.append("\n[FALSIFIED / REJECTED HYPOTHESES]")
            for r in self.data["rejected_hypotheses"]:
                lines.append(f"  • {r['id']} (REJECTED): {r['statement']} (Falsified by: {r.get('falsified_by', 'experiment')})")

        # Disagreements
        if self.data.get("agent_disagreements"):
            lines.append("\n[OPEN AGENT DISAGREEMENTS]")
            for d in self.data["agent_disagreements"]:
                if d.get("status") == "unresolved":
                    lines.append(f"  • {d['id']} [{', '.join(d.get('agents', []))}]: '{d['claim_a']}' vs '{d['claim_b']}' -> Test: {d.get('discriminating_test')}")

        # Unresolved Questions
        open_qs = [q for q in self.data.get("unresolved_questions", []) if q.get("status") == "open"]
        lines.append("\n[OPEN QUESTIONS]")
        if not open_qs:
            lines.append("  (None)")
        for q in open_qs:
            lines.append(f"  • {q['id']} [{q.get('priority', 'normal')}]: {q['question']}")

        # Key Results
        if self.data.get("important_experimental_results"):
            lines.append("\n[KEY EXPERIMENTAL RESULTS]")
            for e in self.data["important_experimental_results"][-5:]:
                ctrl = f" (ctrl: {e['control_value']})" if e.get("control_value") is not None else ""
                lines.append(f"  • {e['id']}: {e['metric']} = {e['value']}{ctrl} - {e.get('notes', '')}")

        return "\n".join(lines)

    def render_canonical_packet(self) -> str:
        """Render the formal 7-tier canonical scientific research state packet."""
        lines = [
            f"# CANONICAL RESEARCH-STATE PACKET: {self.data.get('topic', 'Untitled')}",
            f"**Last Updated**: {self.data.get('updated_at_utc', '')[:19]} UTC",
            "",
            "---",
            "",
            "## 1. ESTABLISHED (Results Supported by Current Evidence)",
        ]
        facts = self.data.get("confirmed_facts", [])
        if not facts:
            lines.append("*(No established results recorded yet)*")
        for f in facts:
            src = f" [Source: `{f['source_measurement']}`]" if f.get("source_measurement") else ""
            lines.append(f"- **`{f['id']}`**: {f['statement']}{src}")

        lines.extend([
            "",
            "## 2. SUPPORTED BUT NOT ESTABLISHED (Plausible Interpretations With Competing Alternatives)",
        ])
        supported = self.data.get("supported_interpretations", [])
        if not supported:
            lines.append("*(No supported interpretations recorded)*")
        for s in supported:
            alt = f"\n  - *Competing Alternative*: {s['competing_alternatives']}" if s.get("competing_alternatives") else ""
            lines.append(f"- **`{s.get('id', 'SUP')}`**: {s['statement']}{alt}")

        lines.extend([
            "",
            "## 3. HYPOTHESES (Mechanistic Ideas Awaiting Decisive Testing)",
        ])
        hypotheses = self.data.get("current_hypotheses", [])
        if not hypotheses:
            lines.append("*(No active hypotheses)*")
        for h in hypotheses:
            lines.append(f"- **`{h['id']}`** (conf: {h.get('confidence', 0.5):.2f}): {h['statement']}")

        lines.extend([
            "",
            "## 4. FALSIFIED / SUPERSEDED (Explicitly Labeled Historical Failures & Artifacts)",
            "> [!WARNING]",
            "> Agents must never unknowingly reason from a superseded result. Historical failures and artifacts are useful context, but labeled explicitly.",
        ])
        rejected = self.data.get("rejected_hypotheses", [])
        if not rejected:
            lines.append("*(None)*")
        for r in rejected:
            tag = "[ARTIFACT]" if r.get("is_artifact") or "artifact" in r.get("falsified_by", "").lower() else "[FALSIFIED]"
            lines.append(f"- **`{r['id']}` {tag}**: {r['statement']}\n  - *Reason / Disproven By*: {r.get('falsified_by', 'experiment')}")

        lines.extend([
            "",
            "## 5. OPEN QUESTIONS (Important Unresolved Scientific Questions)",
        ])
        open_qs = [q for q in self.data.get("unresolved_questions", []) if q.get("status") == "open"]
        if not open_qs:
            lines.append("*(None)*")
        for q in open_qs:
            lines.append(f"- **`{q['id']}`** [{q.get('priority', 'normal')} priority]: {q['question']}")

        lines.extend([
            "",
            "## 6. KNOWN CONFOUNDS (Alternative Explanations, Measurement Artifacts, Leakage, Optimization Weaknesses)",
        ])
        confounds = self.data.get("known_confounds", [])
        if not confounds:
            lines.append("*(None recorded)*")
        for c in confounds:
            mit = f"\n  - *Weakness / Control*: {c['mitigation_or_weakness']}" if c.get("mitigation_or_weakness") else ""
            lines.append(f"- **`{c.get('id', 'CONF')}`**: {c['confound']}{mit}")

        lines.extend([
            "",
            "## 7. NEXT DISCRIMINATING EXPERIMENTS (Experiments Capable of Separating Remaining Hypotheses)",
        ])
        candidate_exps = self.data.get("next_candidate_experiments", [])
        if not candidate_exps:
            lines.append("*(None currently queued)*")
        for e in candidate_exps:
            th = f" (Targets: {', '.join(e.get('target_hypotheses', []))})" if e.get("target_hypotheses") else ""
            lines.append(f"- **`{e.get('id', 'EXP')}`**: {e.get('name', 'Experiment')}{th}\n  - *Protocol*: {e.get('protocol', '')}")

        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research State Packet CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # init
    init_p = subparsers.add_parser("init", help="Initialize or reset research state")
    init_p.add_argument("topic", help="Topic or title of the research campaign")

    # view
    view_p = subparsers.add_parser("view", help="Print compact summary of research state")
    view_p.add_argument("--json", action="store_true")

    # add-fact
    fact_p = subparsers.add_parser("add-fact", help="Add a confirmed measurement/fact")
    fact_p.add_argument("statement", help="Confirmed factual statement")
    fact_p.add_argument("--source", default="", help="Experimental source or trace")

    # add-hypothesis
    hyp_p = subparsers.add_parser("add-hypothesis", help="Add a candidate hypothesis")
    hyp_p.add_argument("statement", help="Hypothesis statement")
    hyp_p.add_argument("--conf", type=float, default=0.5)

    # reject-hypothesis
    rej_p = subparsers.add_parser("reject-hypothesis", help="Reject a hypothesis")
    rej_p.add_argument("id", help="Hypothesis ID (e.g. H1)")
    rej_p.add_argument("reason", help="Evidence or test that falsified it")

    # add-question
    q_p = subparsers.add_parser("add-question", help="Add an open question")
    q_p.add_argument("question", help="Question text")
    q_p.add_argument("--priority", default="normal", choices=["high", "normal", "low"])

    # record-exp
    exp_p = subparsers.add_parser("record-exp", help="Record experimental metric")
    exp_p.add_argument("id", help="Experiment identifier")
    exp_p.add_argument("metric", help="Metric name (e.g. accuracy, lyapunov_exp)")
    exp_p.add_argument("value", help="Observed metric value")
    exp_p.add_argument("--control", help="Control baseline value")
    exp_p.add_argument("--notes", default="")

    # archive
    arch_p = subparsers.add_parser("archive", help="Archive resolved questions and stale hypotheses")
    arch_p.add_argument("--max-rejected", type=int, default=5)
    arch_p.add_argument("--max-experiments", type=int, default=10)

    # retrieve
    ret_p = subparsers.add_parser("retrieve", help="Retrieve an item from archive by ID")
    ret_p.add_argument("id", help="Item ID (e.g. Q1, H2, EXP_3)")

    # status
    stat_p = subparsers.add_parser("status", help="Show context budget status and insufficiency count")
    stat_p.add_argument("--json", action="store_true")

    # canonical
    can_p = subparsers.add_parser("canonical", help="Print the 7-tier canonical scientific research state packet")
    can_p.add_argument("--markdown", action="store_true", help="Format as markdown")

    # add-supported
    sup_p = subparsers.add_parser("add-supported", help="Add a supported interpretation with competing alternatives")
    sup_p.add_argument("statement", help="Plausible interpretation supported by evidence")
    sup_p.add_argument("--alt", default="", help="Plausible competing alternative explanation")

    # add-falsified
    fal_p = subparsers.add_parser("add-falsified", help="Add a falsified or superseded finding")
    fal_p.add_argument("statement", help="Falsified or superseded claim")
    fal_p.add_argument("reason", help="Evidence or experiment that invalidated it")
    fal_p.add_argument("--artifact", action="store_true", help="Explicitly label as measurement/code artifact")

    # add-confound
    conf_p = subparsers.add_parser("add-confound", help="Add a known confound or control weakness")
    conf_p.add_argument("confound", help="Known confound, leakage possibility, or artifact risk")
    conf_p.add_argument("--mitigation", default="", help="Control condition or mitigation")

    # add-discriminating
    disc_p = subparsers.add_parser("add-discriminating", help="Add a discriminating experiment")
    disc_p.add_argument("name", help="Experiment name")
    disc_p.add_argument("protocol", help="Experimental protocol and measurement")
    disc_p.add_argument("--targets", default="", help="Comma-separated hypothesis IDs targeted")

    args = parser.parse_args(argv)
    state = ResearchState()

    if args.subcommand == "init":
        state.data = {
            "topic": args.topic,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_hypotheses": [],
            "confirmed_facts": [],
            "rejected_hypotheses": [],
            "unresolved_questions": [],
            "important_experimental_results": [],
            "next_candidate_experiments": [],
            "agent_disagreements": [],
            "uncertainty_structure": {},
            "state_insufficiency_events": [],
        }
        state.save()
        print(f"Initialized research state for: '{args.topic}'")

    elif args.subcommand == "view":
        if args.json:
            print(json.dumps(state.data, indent=2))
        else:
            print(state.render_compact_summary())

    elif args.subcommand == "add-fact":
        fid = state.confirm_fact(args.statement, args.source)
        print(f"Recorded fact {fid}: {args.statement}")

    elif args.subcommand == "add-hypothesis":
        hid = state.add_hypothesis(args.statement, args.conf)
        print(f"Recorded hypothesis {hid}: {args.statement}")

    elif args.subcommand == "reject-hypothesis":
        if state.reject_hypothesis(args.id, args.reason):
            print(f"Rejected {args.id}: {args.reason}")
        else:
            print(f"Hypothesis {args.id} not found among active hypotheses.", file=sys.stderr)
            return 1

    elif args.subcommand == "add-question":
        qid = state.add_question(args.question, args.priority)
        print(f"Added question {qid}: {args.question}")

    elif args.subcommand == "record-exp":
        state.record_experiment_result(args.id, args.metric, args.value, args.control, args.notes)
        print(f"Recorded experiment result for {args.id}")

    elif args.subcommand == "archive":
        res = state.archive_stale(
            max_rejected_kept=args.max_rejected,
            max_experiments_kept=args.max_experiments,
        )
        print(f"Archive completed: {res['total_pruned']} total items archived "
              f"({res['questions_archived']} questions, {res['hypotheses_archived']} rejected hypotheses, "
              f"{res['experiments_archived']} experiments).")

    elif args.subcommand == "retrieve":
        item = state.retrieve_from_archive(args.id)
        if item:
            print(json.dumps(item, indent=2))
        else:
            print(f"Item {args.id} not found in archive.", file=sys.stderr)
            return 1

    elif args.subcommand == "status":
        status = state.get_budget_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("=== RESEARCH STATE: BUDGET STATUS ===")
            print(f"Current Size:      {status['current_bytes']} bytes")
            print(f"Soft Budget:       {status['soft_budget_bytes']} bytes ({status['soft_budget_ratio']*100:.1f}%)")
            print(f"Hard Budget:       {status['hard_budget_bytes']} bytes ({status['hard_budget_ratio']*100:.1f}%)")
            print(f"Exceeds Soft:      {status['exceeds_soft']}")
            print(f"Exceeds Hard:      {status['exceeds_hard']}")
            print(f"Active Hypotheses: {status['active_hypotheses_count']}")
            print(f"High-Priority Qs:  {status['high_priority_questions_count']}")
            print(f"Insufficiency Events: {status['insufficiency_events_count']}")

    elif args.subcommand == "canonical":
        print(state.render_canonical_packet())

    elif args.subcommand == "add-supported":
        sid = state.add_supported(args.statement, args.alt)
        print(f"Recorded supported interpretation {sid}: {args.statement}")

    elif args.subcommand == "add-falsified":
        rid = state.add_falsified(args.statement, args.reason, is_artifact=args.artifact)
        tag = "artifact" if args.artifact else "falsified"
        print(f"Recorded {tag} result {rid}: {args.statement}")

    elif args.subcommand == "add-confound":
        cid = state.add_confound(args.confound, args.mitigation)
        print(f"Recorded known confound {cid}: {args.confound}")

    elif args.subcommand == "add-discriminating":
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        eid = state.add_discriminating_experiment(args.name, targets, args.protocol)
        print(f"Recorded discriminating experiment {eid}: {args.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
