"""Offline unit tests for the Research Team skill scripts."""

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any
from unittest.mock import patch, MagicMock

import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
OPENROUTER_SCRIPTS = SCRIPTS_DIR.parent.parent / "openrouter-subagents" / "scripts"
sys.path.insert(0, str(OPENROUTER_SCRIPTS))

import research_coordinator as research_coordinator_module
from research_coordinator import (
    ResearchCoordinator,
    resolve_context_argument,
    compose_worker_context,
)
from research_state import ResearchState, SOFT_BUDGET_BYTES, HARD_BUDGET_BYTES
from recursive_interrogator import is_repetitive, run_recursive_interrogation
from reasoning_frontier import ReasoningFrontier, BranchNode
from ideation_engine import filter_and_score_candidates, run_ideation_cycle


class TestResearchState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = Path(self.tmp.name) / "test_state.json"
        self.state = ResearchState(path=self.state_file)

    def test_state_lifecycle(self):
        self.state.set_topic("Test Research")
        hid = self.state.add_hypothesis("Hypothesis 1", confidence=0.7)
        self.assertEqual(hid, "H1")
        self.assertEqual(len(self.state.data["current_hypotheses"]), 1)

        fid = self.state.confirm_fact("Measurement A", source_measurement="run_01")
        self.assertEqual(fid, "F1")
        self.assertEqual(len(self.state.data["confirmed_facts"]), 1)

        success = self.state.reject_hypothesis("H1", "Falsified by experiment 2")
        self.assertTrue(success)
        self.assertEqual(len(self.state.data["current_hypotheses"]), 0)
        self.assertEqual(len(self.state.data["rejected_hypotheses"]), 1)
        self.assertEqual(self.state.data["rejected_hypotheses"][0]["falsified_by"], "Falsified by experiment 2")

        summary = self.state.render_compact_summary()
        self.assertIn("Test Research", summary)
        self.assertIn("F1: Measurement A", summary)
        self.assertIn("H1 (REJECTED)", summary)

    def test_state_archival(self):
        archive_path = Path(self.tmp.name) / "test_archive.json"
        state = ResearchState(path=self.state_file, archive_path=archive_path)
        qid = state.add_question("Question 1")
        state.resolve_question(qid, "Resolved answer")
        state.add_question("Question 2 (open)")

        # Add multiple rejected hypotheses
        for i in range(7):
            hid = state.add_hypothesis(f"Hypothesis {i}")
            state.reject_hypothesis(hid, f"Falsified {i}")

        res = state.archive_stale(max_rejected_kept=3, archive_path=archive_path)
        self.assertEqual(res["questions_archived"], 1)
        self.assertEqual(res["hypotheses_archived"], 4)
        self.assertEqual(len(state.data["unresolved_questions"]), 1)
        self.assertEqual(len(state.data["rejected_hypotheses"]), 3)
        self.assertTrue(archive_path.is_file())

    def test_salience_aware_retention_and_retrieval(self):
        archive_path = Path(self.tmp.name) / "test_salience_archive.json"
        state = ResearchState(path=self.state_file, archive_path=archive_path)

        hid_active = state.add_hypothesis("Active continuous recurrent dynamics", confidence=0.85)
        qid_high = state.add_question("Critical invariant check", priority="high")
        qid_low = state.add_question("Minor formatting question", priority="low")
        qid_res = state.add_question("Old question", priority="normal")
        state.resolve_question(qid_res, "Old answer")

        for i in range(6):
            h = state.add_hypothesis(f"Hypothesis {i}")
            state.reject_hypothesis(h, f"Falsified reason {i}")

        state.record_experiment_result("test_run", "loss", 0.042)

        res = state.archive_stale(max_rejected_kept=2, archive_path=archive_path)
        self.assertGreater(res["total_pruned"], 0)

        # Salience invariant: active hypothesis and high-priority question must remain
        active_ids = [h["id"] for h in state.data["current_hypotheses"]]
        self.assertIn(hid_active, active_ids)
        open_q_ids = [q["id"] for q in state.data["unresolved_questions"]]
        self.assertIn(qid_high, open_q_ids)

        # Retrieval from archive
        retrieved_q = state.retrieve_from_archive(qid_res, archive_path=archive_path)
        self.assertIsNotNone(retrieved_q)
        self.assertEqual(retrieved_q["id"], qid_res)

        status = state.get_budget_status()
        self.assertGreaterEqual(status["insufficiency_events_count"], 1)
        self.assertEqual(status["active_hypotheses_count"], 1)
        self.assertEqual(status["high_priority_questions_count"], 1)


class TestRecursiveInterrogator(unittest.TestCase):
    def test_repetition_detection(self):
        t1 = "The model learns an internal continuous manifold representing the state derivative."
        t2 = "The model learns an internal continuous manifold representing the state derivative."
        t3 = "A completely different discrete lookahead mechanism operating on token parity."
        self.assertTrue(is_repetitive(t1, t2))
        self.assertFalse(is_repetitive(t1, t3))

    @patch("recursive_interrogator.run_subagent")
    def test_recursive_interrogation_early_stop_on_no_new_info(self, mock_subagent):
        mock_subagent.return_value = {
            "status": "ok",
            "answer": "NO_NEW_INFORMATION after pass 1",
            "usage": {"total_tokens": 100},
        }
        res = run_recursive_interrogation("Test claim", max_depth=3)
        self.assertEqual(res["passes_completed"], 1)
        self.assertTrue(res["stopped_early"])
        self.assertIn("NO_NEW_INFORMATION", res["stop_reason"])


class TestResearchCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @patch("research_coordinator.jev_decide")
    def test_plan_task_routing(self, mock_jev):
        def mock_decision(qtype, *args, **kwargs):
            if qtype == "THINKING_BUDGET":
                return {"decision": "deep", "probabilities": {"deep": 0.8}, "confidence": 0.8}
            elif qtype == "UNCERTAINTY_TYPE":
                return {"decision": "conceptual", "probabilities": {"conceptual": 0.9}, "confidence": 0.9}
            return {"decision": "normal"}

        mock_jev.side_effect = mock_decision

        coord = ResearchCoordinator(state_directory=Path(self.tmp.name))
        plan = coord.plan_task("Investigate recurrent mechanism")
        self.assertEqual(plan["thinking_budget"], "deep")
        self.assertEqual(plan["uncertainty_type"], "conceptual")
        self.assertEqual(plan["recommended_branch"], "deepseek_parallel_research")
        self.assertIn("counter-hypothesis-generator", plan["recommended_roles"])

    @patch("research_coordinator.jev_decide")
    def test_verify_synthesis_gate_supported(self, mock_jev):
        def mock_decision(qtype, *args, **kwargs):
            if qtype == "EVIDENCE_STATUS":
                return {"status": "ok", "decision": "supported", "confidence": 0.95}
            elif qtype == "NEXT_ACTION":
                return {"decision": "act"}
            return {"decision": "normal"}

        mock_jev.side_effect = mock_decision

        coord = ResearchCoordinator(state_directory=Path(self.tmp.name))
        gate = coord.verify_synthesis_gate("Claim verified", "Evidence from 3 seeds")
        self.assertEqual(gate["evidence_status"], "supported")
        self.assertEqual(gate["action_branch"], "inspect_supporting_artifacts")
        self.assertFalse(gate["is_supported"])
        self.assertTrue(gate["model_supports_claim"])

    @patch("research_coordinator.run_ideation_cycle")
    def test_coordinator_ideation_and_discriminating_experiment(self, mock_ideation):
        mock_ideation.return_value = {
            "status": "ok",
            "raw_candidates_count": 3,
            "accepted_candidates_count": 2,
            "new_branches_added": ["B1", "B2"],
        }
        coord = ResearchCoordinator(state_directory=Path(self.tmp.name))
        res = coord.run_ideation("Investigate bifurcation")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["new_branches_added"]), 2)

        coord.add_hypothesis_branch("Branch Alpha", action="Test Alpha")
        coord.add_hypothesis_branch("Branch Beta", action="Test Beta")
        disc = coord.get_discriminating_experiment(top_k=2)
        self.assertEqual(disc["discriminating_test"]["status"], "ready")

    @patch("research_coordinator.jev_decide")
    def test_coordinator_overrides(self, mock_jev):
        mock_jev.return_value = {"decision": "normal", "confidence": 0.85, "probabilities": {"normal": 0.85}}
        coord = ResearchCoordinator(state_directory=Path(self.tmp.name))
        plan = coord.plan_task(
            "Task with override",
            overrides={"thinking_budget": "escalate", "recommended_branch": "escalation_astra", "should_branch": True},
        )
        self.assertEqual(plan["thinking_budget"], "escalate")
        self.assertEqual(plan["recommended_branch"], "escalation_astra")
        self.assertTrue(plan["branching_evaluation"]["should_branch"])
        self.assertEqual(plan["overrides_applied"]["thinking_budget"], "escalate")

    def test_coordinator_active_archive_retrieval(self):
        coord = ResearchCoordinator(state_directory=Path(self.tmp.name))
        hid = coord.state.add_hypothesis("Baseline linear attention model", confidence=0.7)
        coord.state.reject_hypothesis(hid, falsified_by="Gradient collapse test")
        coord.state.archive_stale(max_rejected_kept=0)

        # Confirm hid is in archive, not in active hypotheses
        self.assertNotIn(hid, [h["id"] for h in coord.state.data["current_hypotheses"]])

        # Test auto-restore when referenced in prompt
        restored = coord.check_and_restore_archived_context(f"Re-evaluating previous finding from {hid}")
        self.assertTrue(len(restored) > 0)
        self.assertIn(hid, restored[0])


class TestReasoningFrontier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frontier_file = Path(self.tmp.name) / "frontier.json"
        self.frontier = ReasoningFrontier(storage_path=self.frontier_file)

    def test_semantic_deduplication(self):
        bid1, is_new1 = self.frontier.add_branch(
            "The recurrent mechanism learns an internal continuous manifold representing state velocity",
            plausibility=0.8,
            support=["Observation A"],
        )
        self.assertTrue(is_new1)

        bid2, is_new2 = self.frontier.add_branch(
            "The recurrent mechanism learns an internal continuous manifold representing state velocity with parameters",
            plausibility=0.8,
            support=["Observation B"],
        )
        self.assertFalse(is_new2)
        self.assertEqual(bid1, bid2)
        self.assertEqual(self.frontier.branches[bid1].visit_count, 2)
        self.assertIn("Observation B", self.frontier.branches[bid1].support)

    def test_semantic_deduplication_polarity_guard(self):
        h1 = "The loss spike is caused by recurrence, not attention."
        h2 = "The loss spike is caused by attention, not recurrence."

        bid1, is_new1 = self.frontier.add_branch(h1, plausibility=0.7)
        self.assertTrue(is_new1)

        bid2, is_new2 = self.frontier.add_branch(h2, plausibility=0.7)
        self.assertTrue(is_new2)
        self.assertNotEqual(bid1, bid2)

    def test_ucb_and_diversity_preservation(self):
        # Add 3 distinct conventional branches (semantically distinct to avoid deduplication)
        self.frontier.add_branch(
            "Standard gradient descent optimization baseline",
            plausibility=0.90,
            expected_information_gain=0.30,
            novelty_score=0.10,
            estimated_cost=0.001,
        )
        self.frontier.add_branch(
            "Conventional weight decay regularization baseline",
            plausibility=0.90,
            expected_information_gain=0.30,
            novelty_score=0.10,
            estimated_cost=0.001,
        )
        self.frontier.add_branch(
            "Classical momentum acceleration dynamics benchmark",
            plausibility=0.90,
            expected_information_gain=0.30,
            novelty_score=0.10,
            estimated_cost=0.001,
        )

        bid_novel, _ = self.frontier.add_branch(
            "Radical non-local cellular automata topological soliton intervention",
            plausibility=0.25,
            expected_information_gain=0.90,
            novelty_score=0.95,
            estimated_cost=0.001,
        )

        no_div = self.frontier.select_active_frontier(top_k=2, preserve_diversity=False)
        self.assertNotIn(bid_novel, [b.branch_id for b in no_div])

        div = self.frontier.select_active_frontier(top_k=2, preserve_diversity=True)
        self.assertIn(bid_novel, [b.branch_id for b in div])

    def test_multi_branch_discriminating_test(self):
        self.frontier.add_branch(
            "Causal mechanism A depends on recurrent hidden state memory",
            next_discriminating_action="Zero-hidden-state ablation control",
        )
        self.frontier.add_branch(
            "Causal mechanism B depends on positional feedforward attention",
            next_discriminating_action="Positional embedding permutation sweep",
        )
        plan = self.frontier.find_multi_branch_discriminating_test()
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(len(plan["branch_ids"]), 2)
        self.assertIn("Zero-hidden-state ablation control", plan["proposed_actions"])


class TestIdeationEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.frontier_file = Path(self.tmp.name) / "ideation_frontier.json"
        self.frontier = ReasoningFrontier(storage_path=self.frontier_file)

    @patch("ideation_engine.jev_decide")
    def test_filter_and_score_diversity_bypass(self, mock_jev):
        def mock_decision(qtype, *args, **kwargs):
            if qtype == "PLAUSIBILITY":
                return {"decision": "unlikely", "probabilities": {"unlikely": 0.8}}
            elif qtype == "NOVELTY":
                return {"decision": "radical", "probabilities": {"radical": 0.9}}
            elif qtype == "TESTABILITY":
                return {"decision": "immediately_testable", "probabilities": {"immediately_testable": 0.9}}
            return {"decision": "normal"}

        mock_jev.side_effect = mock_decision

        candidate = {
            "name": "BistableHysteresisIntervention",
            "mechanism": "Nonlinear bistable state flips upon boundary crossing",
            "discriminating_test": "Single forward pass perturbation check",
        }
        accepted = filter_and_score_candidates([candidate], self.frontier)
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["is_diversity_pass"])
        self.assertEqual(accepted[0]["name"], "BistableHysteresisIntervention")


class TestContextBuilder(unittest.TestCase):
    def test_build_packet_contains_key_dimensions(self):
        from context_builder import ContextBuilder
        builder = ContextBuilder()
        packet = builder.build_packet("Test Research Objective: Dyck-4 Pushdown Stability")
        self.assertIn("1. CURRENT RESEARCH OBJECTIVE", packet)
        self.assertIn("2. ARCHITECTURE & RECURRENT DYNAMICS", packet)
        self.assertIn("3. TASK DEFINITIONS", packet)
        self.assertIn("4. LATENT TICK SEMANTICS", packet)
        self.assertIn("5. STABILIZATION MECHANISMS", packet)
        self.assertIn("7. HISTORICAL ARTIFACTS & FALSIFIED HYPOTHESES", packet)
        self.assertIn("8. UNRESOLVED INTERPRETATIONS & SUSPECTED CONFOUNDS", packet)
        self.assertIn("Dyck-4 Pushdown Stability", packet)


class TestCanonicalResearchState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = Path(self.tmp.name) / "test_canonical_state.json"
        self.state = ResearchState(path=self.state_file)

    def test_canonical_additions_and_rendering(self):
        self.state.add_established("Recurrence is causally required", "lesion_study.json")
        self.state.add_supported("OCPD bounds loss growth via spectral damping", "Strictly contractive fixed point")
        self.state.add_hypothesis("Continuous unitary operators achieve ballistic transport", 0.6)
        self.state.add_falsified("Canonical baselines collapse to 0.0%", "Untrained weights artifact", is_artifact=True)
        self.state.add_falsified("H1: Settling latency", "Horizon sweep tau=48", is_artifact=False)
        self.state.add_question("Can we construct a non-creeping fixed point?", priority="high")
        self.state.add_confound("Shallow bracket density dominates Dyck-4", "Stratify evaluation by D >= 4")
        self.state.add_discriminating_experiment("Adversarial N-gram Dyck Test", ["H1"], "Sample uniform bigrams")

        packet = self.state.render_canonical_packet()
        self.assertIn("## 1. ESTABLISHED", packet)
        self.assertIn("## 2. SUPPORTED BUT NOT ESTABLISHED", packet)
        self.assertIn("## 3. HYPOTHESES", packet)
        self.assertIn("## 4. FALSIFIED / SUPERSEDED", packet)
        self.assertIn("[ARTIFACT]", packet)
        self.assertIn("## 5. OPEN QUESTIONS", packet)
        self.assertIn("## 6. KNOWN CONFOUNDS", packet)
        self.assertIn("## 7. NEXT DISCRIMINATING EXPERIMENTS", packet)


class TestIndependentReconstructionAudit(unittest.TestCase):
    def test_audit_reconstruction_pass(self):
        from independent_reconstruction import audit_reconstruction
        high_quality_response = (
            "[ESTABLISHED] Recurrent updates are causally required for task execution (d=15.56).\n"
            "CD-DV-NCA achieves 43.9% bracket prediction on Dyck-4 via discrete STE carry channels.\n"
            "[ARTIFACT] Previous reports claiming canonical baselines collapse to 0.0% evaluated untrained random weights.\n"
            "[ARTIFACT] FC-4 carry lesion failure was an artifact of shallow bracket (D=1) averaging.\n"
            "[FALSIFIED] H1 settling latency is falsified by tau=48 horizon sweep.\n"
            "[BOUND] Finite physical lattices are strictly Chomsky Type-3 Finite State Automata (bounded DPDA).\n"
            "[CONFOUND] Shallow bigrams allow ~40% accuracy; continuous loss drift requires OCPD damping.\n"
            "Discriminating experiment: sample adversarial balanced n-grams with D in [4, 12]."
        )
        audit = audit_reconstruction(high_quality_response)
        self.assertIn(audit["grade"], ("PASS", "WARN"))
        self.assertGreater(audit["total_score"], 0.70)
        self.assertTrue(audit["flags"]["identified_baseline_artifact"])
        self.assertTrue(audit["flags"]["identified_fc4_shallow_artifact"])

    def test_audit_reconstruction_fail_on_hallucinations(self):
        from independent_reconstruction import audit_reconstruction
        poor_response = "Titan Text has fully emergent reasoning. Transformer gets 0.0% and is completely incapable."
        audit = audit_reconstruction(poor_response)
        self.assertEqual(audit["grade"], "FAIL")
        self.assertFalse(audit["flags"]["identified_baseline_artifact"])


class TestResearchMeeting(unittest.TestCase):
    def test_extract_cross_agent_questions(self):
        from research_meeting import ResearchMeeting
        meeting = ResearchMeeting("Test Milestone")
        round_2_results = {
            "dynamics-agent": {
                "answer": (
                    "My analysis of the Lyapunov spectrum shows drift.\n"
                    "TARGET_AGENT: statistical-agent\n"
                    "QUESTION: Can you compute the bootstrap confidence interval of field energy across seeds 42 and 43?\n"
                )
            },
            "falsification-arbiter": {
                "answer": (
                    "TARGET_AGENT: rust-audit-agent\n"
                    "QUESTION: Verify whether the carry lesion zeroing kernel is applied before or after perception.\n"
                )
            },
        }
        cross_q = meeting.extract_cross_agent_questions(round_2_results)
        self.assertEqual(len(cross_q), 2)
        self.assertEqual(cross_q[0]["from_role"], "dynamics-agent")
        self.assertEqual(cross_q[0]["to_role"], "statistical-agent")
        self.assertEqual(cross_q[1]["from_role"], "falsification-arbiter")
        self.assertEqual(cross_q[1]["to_role"], "rust-audit-agent")

    def test_synthesize_presidential_report(self):
        from research_meeting import ResearchMeeting
        meeting = ResearchMeeting("Test Milestone")
        round_1 = {"skeptical-agent": {"answer": "Analysis 1"}, "dynamics-agent": {"answer": "Analysis 2"}}
        round_2 = {
            "skeptical-agent": {
                "answer": "8. What should we absolutely NOT conclude yet? Do not conclude that pushdown memory is unbounded.\n5. What experiment would distinguish them? Run balanced bigram ablation."
            }
        }
        rep = meeting.synthesize_presidential_report(round_1, round_2, [])
        self.assertIn("Test Milestone", rep["objective"])
        self.assertGreaterEqual(len(rep["warnings_against_premature_conclusions"]), 1)
        self.assertIn("pushdown memory is unbounded", rep["warnings_against_premature_conclusions"][0])


class TestContextTransport(unittest.TestCase):
    """Deterministic tests: context packets must reach the worker prompt as
    actual text, not as a filesystem path. Sentinel-based, offline, no API."""

    SENTINEL = "SENTINEL-XYZZY-9137-DO-NOT-TRUNCATE"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _packet(self, size_chars: int) -> str:
        # Sentinel at both head and, critically, the END of the packet so any
        # silent truncation would remove it.
        filler = "".join(chr(97 + (i % 26)) for i in range(size_chars))
        return f"HEAD:{self.SENTINEL}\n{filler}\nTAIL:{self.SENTINEL}"

    def test_literal_context_passthrough(self):
        text, source = resolve_context_argument("plain literal context")
        self.assertEqual(text, "plain literal context")
        self.assertEqual(source, "literal")

    def test_empty_context_is_empty_literal(self):
        text, source = resolve_context_argument("", None)
        self.assertEqual(text, "")
        self.assertEqual(source, "empty")

    def test_file_context_inlines_sentinel_not_path(self):
        packet = self._packet(4000)
        path = Path(self.tmp.name) / "packet.txt"
        path.write_text(packet, encoding="utf-8")
        text, source = resolve_context_argument("", str(path))
        # Contents (both sentinels) transmitted; path string itself absent.
        self.assertIn(f"HEAD:{self.SENTINEL}", text)
        self.assertIn(f"TAIL:{self.SENTINEL}", text)
        self.assertNotIn(str(path), text)
        self.assertTrue(source.startswith("file:"))

    def test_large_packet_end_sentinel_preserved_no_truncation(self):
        packet = self._packet(300_000)
        path = Path(self.tmp.name) / "big_packet.txt"
        path.write_text(packet, encoding="utf-8")
        text, source = resolve_context_argument("", str(path))
        self.assertEqual(len(text), len(packet))
        self.assertTrue(text.endswith(f"TAIL:{self.SENTINEL}"))

    def test_stdin_transport(self):
        import io
        sentinel_text = f"STDIN:{self.SENTINEL}"
        with patch.object(sys, "stdin", io.StringIO(sentinel_text)):
            text, source = resolve_context_argument("-", None)
        self.assertEqual(text, sentinel_text)
        self.assertEqual(source, "stdin")

    def test_unicode_context_preserved(self):
        packet = "αβγδ — émigré — 日本語 ✓ " + self.SENTINEL
        path = Path(self.tmp.name) / "unicode.txt"
        path.write_text(packet, encoding="utf-8")
        text, _ = resolve_context_argument("", str(path))
        self.assertEqual(text, packet)

    def test_missing_context_file_raises(self):
        with self.assertRaises(ValueError):
            resolve_context_argument("", str(Path(self.tmp.name) / "nope.txt"))

    def test_directory_as_context_file_raises(self):
        with self.assertRaises(ValueError):
            resolve_context_argument("", self.tmp.name)

    def test_invalid_utf8_context_file_raises(self):
        path = Path(self.tmp.name) / "binary.bin"
        path.write_bytes(b"\xff\xfe\xfa\x00\x81")
        with self.assertRaises(ValueError):
            resolve_context_argument("", str(path))

    def test_empty_stdin_raises(self):
        import io
        with patch.object(sys, "stdin", io.StringIO("")):
            with self.assertRaises(ValueError):
                resolve_context_argument("-", None)

    def test_delegated_worker_prompt_receives_packet_contents(self):
        """End-to-end (offline): run_parallel_investigators must transmit the
        packet contents to the worker; a bare path alone is insufficient."""
        packet = self._packet(5000)
        path = Path(self.tmp.name) / "packet.txt"
        path.write_text(packet, encoding="utf-8")
        resolved, source = resolve_context_argument("", str(path))

        captured: dict[str, Any] = {}

        def fake_run_subagent(task, *, role="researcher", context="", **kwargs):
            captured["task"] = task
            captured["context"] = context
            return {
                "status": "ok", "model": "fake", "role": role,
                "purpose": task, "answer": "ok", "parsed_json": None,
                "reasoning": None, "finish_reason": "stop",
                "usage": {"prompt_tokens": 1234, "total_tokens": 1300},
                "files": [], "elapsed_seconds": 0.1,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }

        coordinator = ResearchCoordinator(state_directory=Path(self.tmp.name))

        with patch.object(research_coordinator_module, "run_subagent", fake_run_subagent):
            results = coordinator.run_parallel_investigators(
                "Question referencing the packet", ["researcher"],
                context=resolved, context_source=source,
            )

        self.assertEqual(results[0]["status"], "ok")
        sent = captured["context"]
        # Contents present in the actual worker prompt context...
        self.assertIn(f"HEAD:{self.SENTINEL}", sent)
        self.assertTrue(sent.rstrip().endswith(f"TAIL:{self.SENTINEL}"))
        # ...whereas the bare path alone could not contain the sentinel.
        self.assertNotIn(str(path), self.SENTINEL)
        self.assertNotIn(str(path), sent)
        self.assertGreater(len(sent), len(str(path)) * 10)
        # Provenance and prompt-token usage recorded.
        self.assertEqual(results[0]["context_source"], source)
        self.assertEqual(results[0]["context_chars_sent"], len(sent))
        self.assertEqual(results[0]["prompt_tokens"], 1234)


class TestRoleInstructionInWorkerContext(unittest.TestCase):
    def test_role_instruction_composed(self):
        ctx = compose_worker_context("skeptical-reviewer", "PACKET BODY")
        self.assertIn("PACKET BODY", ctx)
        self.assertIn("Role instruction:", ctx)


if __name__ == "__main__":
    unittest.main()

