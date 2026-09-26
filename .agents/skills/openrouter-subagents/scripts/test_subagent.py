"""Offline unit tests for subagent.py, jev_decide.py, codex_cli.py, telemetry.py, and cognitive_governor.py."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from subagent import (
    load_api_key,
    read_file_span,
    run_subagent,
    sha256_digest,
    PERMANENT_OPENROUTER_KEY,
    SubagentError,
)
from jev_decide import (
    jev_decide,
    DECISION_SCHEMAS,
    calculate_entropy,
    jev_ensemble,
    jev_framing_ensemble,
)
from codex_cli import run_codex_exec, is_codex_available
from telemetry import TelemetryTracker, append_event, read_events
from cognitive_governor import CognitiveGovernor, MarginalGainTracker
from policy_analyzer import analyze_policy_performance
from trajectory_sensor import (
    TrajectorySensor,
    BaseTrajectoryForecaster,
    HeuristicTrajectoryForecaster,
    TimesFMAdapter,
)
from calibration_tracker import (
    CalibrationTracker,
    calculate_multiclass_brier_score,
    calculate_expected_calibration_error,
)
from policy_replay import PolicyReplayHarness


class TestSubagent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_load_key_permanent_fallback(self):
        with patch.object(Path, "is_file", return_value=False):
            with patch.dict("os.environ", {}, clear=True):
                key = load_api_key()
                self.assertEqual(key, PERMANENT_OPENROUTER_KEY)

    def test_read_file_span_full(self):
        f = self.root / "sample.rs"
        f.write_text("line 1\nline 2\nline 3\n")
        res = read_file_span("sample.rs", self.root)
        self.assertEqual(res["file"], "sample.rs")
        self.assertEqual(res["span"], "1-3")
        self.assertEqual(res["content"], "line 1\nline 2\nline 3\n")

    def test_read_file_span_subset(self):
        f = self.root / "sample.rs"
        f.write_text("line 1\nline 2\nline 3\nline 4\n")
        res = read_file_span("sample.rs:2-3", self.root)
        self.assertEqual(res["file"], "sample.rs")
        self.assertEqual(res["span"], "2-3")
        self.assertEqual(res["content"], "line 2\nline 3\n")

    def test_read_file_span_escape_fails(self):
        with self.assertRaises(SubagentError):
            read_file_span("../escaped.rs", self.root)

    @patch("subagent.call_openrouter")
    def test_run_subagent_mocked(self, mock_call):
        mock_call.return_value = {
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Mocked audit finding"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 50},
        }

        f = self.root / "audit.rs"
        f.write_text("fn test() {}\n")

        res = run_subagent(
            "Audit this file",
            role="task-auditor",
            files=["audit.rs"],
            root=self.root,
        )

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["role"], "task-auditor")
        self.assertEqual(res["answer"], "Mocked audit finding")
        self.assertEqual(len(res["files"]), 1)


class TestJevDecide(unittest.TestCase):
    @patch("jev_decide.call_jev")
    def test_jev_decide_probabilities_preserved(self, mock_call):
        mock_call.return_value = {
            "model": "typesafe/jev-1.13",
            "answers": {
                "thinking_budget": {
                    "type": "choice",
                    "choice": "normal",
                    "probabilities": {"minimal": 0.2, "normal": 0.7, "deep": 0.1, "escalate": 0.0},
                    "confidence": 0.75,
                }
            },
            "usage": {"cost": 0.000015},
        }

        res = jev_decide("THINKING_BUDGET", "Sample task")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["decision"], "normal")
        self.assertEqual(res["confidence"], 0.75)
        self.assertEqual(res["probabilities"]["normal"], 0.7)
        self.assertFalse(res["fallback_applied"])

    @patch("jev_decide.call_jev")
    def test_jev_decide_graceful_fallback(self, mock_call):
        mock_call.side_effect = RuntimeError("Network error")

        res = jev_decide("EVIDENCE_STATUS", "Sample claim")
        self.assertEqual(res["status"], "fallback")
        self.assertEqual(res["decision"], "insufficient")
        self.assertTrue(res["fallback_applied"])
        self.assertIn("Network error", res["error"])
        # Contract integrity: probabilities must always be present as a dict
        self.assertIsInstance(res.get("probabilities"), dict)
        self.assertEqual(res["probabilities"], {})
        self.assertIsNone(res["entropy"])


class TestCodexCLI(unittest.TestCase):
    @patch("codex_cli.is_codex_available", return_value=False)
    def test_codex_unavailable_graceful(self, _):
        res = run_codex_exec("Test prompt")
        self.assertEqual(res["status"], "unavailable")
        self.assertTrue(res["fallback_applied"])

    @patch("codex_cli.subprocess.run")
    @patch("codex_cli.is_codex_available", return_value=True)
    def test_codex_exec_success(self, _, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Generated code patch\n",
            stderr="model: gpt-6-astra\ntokens used\n 3,450\n",
        )
        res = run_codex_exec("Implement feature")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["answer"], "Generated code patch")
        self.assertEqual(res["model"], "gpt-6-astra")
        self.assertEqual(res["tokens_used"], 3450)

    @patch("codex_cli.subprocess.run")
    @patch("codex_cli.is_codex_available", return_value=True)
    def test_codex_exec_missing_token_banner_graceful(self, _, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Answer without tokens\n",
            stderr="Custom stderr without token line\n",
        )
        res = run_codex_exec("Test")
        self.assertEqual(res["status"], "ok")
        self.assertIsNone(res["tokens_used"])

    @patch("codex_cli.subprocess.run")
    @patch("codex_cli.is_codex_available", return_value=True)
    def test_codex_exec_flexible_token_parser(self, _, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Generated code patch\n",
            stderr="Model: gpt-6-astra\nTokens used: 1,234\n",
        )
        res = run_codex_exec("Implement feature", model="gpt-6-astra")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["model"], "gpt-6-astra")
        self.assertEqual(res["requested_model"], "gpt-6-astra")
        self.assertEqual(res["tokens_used"], 1234)


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_file = Path(self.tmp.name) / "events.jsonl"

    def test_telemetry_tracker_and_sanitization(self):
        tracker = TelemetryTracker("coding", "moderate", task_id="test_001")
        tracker.record_budget("normal")
        tracker.record_routing("gemini_direct")
        tracker.record_jev("THINKING_BUDGET", "normal", confidence=0.8, cost_usd=0.00002)
        tracker.record_test("unit_test", True)

        event = tracker.finalize(status="success", log_path=self.log_file)
        self.assertEqual(event["task_id"], "test_001")
        self.assertEqual(event["thinking_budget"], "normal")
        self.assertEqual(event["tests_passed"], 1)

        # Direct test of recursive and regex sanitization: secrets must be scrubbed
        dirty_event = {
            "task_id": "clean_id",
            "api_key": "sk-or-v1-secret-123456",
            "auth_token": "bearer-token-999",
            "password": "secretpassword",
            "normal_field": "ok_value",
            "nested_headers": {
                "Authorization": "Bearer mock-token-placeholder-value",
                "content-type": "application/json",
            },
            "innocuous_note": "Here is key sk-or-v1-abcdef1234567890 embedded in text",
        }
        append_event(dirty_event, log_path=self.log_file)

        events = read_events(self.log_file)
        self.assertEqual(len(events), 2)
        sanitized = events[1]
        self.assertEqual(sanitized["task_id"], "clean_id")
        self.assertEqual(sanitized["normal_field"], "ok_value")
        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("auth_token", sanitized)
        self.assertNotIn("password", sanitized)
        self.assertNotIn("Authorization", sanitized.get("nested_headers", {}))
        self.assertEqual(sanitized["nested_headers"]["content-type"], "application/json")
        self.assertIn("[REDACTED_SECRET]", sanitized["innocuous_note"])
        self.assertNotIn("sk-or-v1-abcdef1234567890", sanitized["innocuous_note"])

    def test_telemetry_uncertainty_metrics_and_router_export(self):
        tracker = TelemetryTracker("research", "complex", task_id="task_router_01")
        probs = {"minimal": 0.1, "normal": 0.7, "deep": 0.2}
        tracker.record_jev("THINKING_BUDGET", "normal", probabilities=probs, confidence=0.7)

        call = tracker.jev_calls[0]
        self.assertEqual(call["top_1_prob"], 0.7)
        self.assertEqual(call["runner_up_prob"], 0.2)
        self.assertAlmostEqual(call["margin"], 0.5, places=4)
        self.assertGreater(call["entropy"], 0.0)

        router_file = Path(self.tmp.name) / "router_data.jsonl"
        samples = tracker.export_router_training_samples(
            "Investigate loss anomaly in Transformer",
            reward_signal=1.0,
            router_path=router_file,
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["task_id"], "task_router_01")
        self.assertEqual(samples[0]["reward_signal"], 1.0)
        self.assertTrue(router_file.is_file())


class TestCognitiveGovernor(unittest.TestCase):
    @patch("cognitive_governor.jev_decide")
    def test_stopping_rule_after_repeated_reflection(self, mock_decide):
        mock_decide.return_value = {"decision": "continue_reasoning", "probabilities": {"continue_reasoning": 0.9}}
        gov = CognitiveGovernor()
        gov.max_passes = 2

        # Step 1: consecutive = 1
        r1 = gov.check_next_step("Pondering 1", new_evidence_added=False)
        self.assertEqual(gov.consecutive_reasoning_passes, 1)
        self.assertFalse(r1.get("forced_stop_applied", False))

        # Step 2: consecutive = 2, Jev's continue_reasoning forced to test
        r2 = gov.check_next_step("Pondering 2", new_evidence_added=False)
        self.assertEqual(gov.consecutive_reasoning_passes, 2)
        self.assertEqual(r2["decision"], "test")

        # Step 3: consecutive = 3 (> max_passes 2) -> forced stop fires, mock_decide bypassed
        mock_decide.reset_mock()
        r3 = gov.check_next_step("Pondering 3", new_evidence_added=False)
        self.assertEqual(r3["decision"], "act")
        self.assertTrue(r3["forced_stop_applied"])
        self.assertIn("Cognitive Governor stopping rule triggered", r3["reason"])
        mock_decide.assert_not_called()

        # Step 4: evidence gathered -> resets counter to 0
        gov.check_next_step("Ran unit test", new_evidence_added=True)
        self.assertEqual(gov.consecutive_reasoning_passes, 0)

    @patch("cognitive_governor.jev_decide")
    def test_dynamic_confidence_gating_upgrades_shaky_minimal(self, mock_decide):
        def mock_dec(qtype, *args, **kwargs):
            if qtype == "THINKING_BUDGET":
                return {
                    "decision": "minimal",
                    "confidence": 0.55,  # below 0.70 threshold
                    "probabilities": {"minimal": 0.55, "normal": 0.45},
                }
            return {"decision": "low_uncertainty"}

        mock_decide.side_effect = mock_dec
        gov = CognitiveGovernor()
        res = gov.assess_initial_task("Sample ambiguous task")
        self.assertEqual(res["thinking_budget"], "normal")
        self.assertTrue(res["gating_adjusted"])
        self.assertIn("Upgraded minimal -> normal", res["gating_reason"])

    @patch("cognitive_governor.jev_decide")
    def test_dynamic_confidence_gating_upgrades_to_escalate_when_runner_up(self, mock_decide):
        def mock_dec(qtype, *args, **kwargs):
            if qtype == "THINKING_BUDGET":
                return {
                    "decision": "minimal",
                    "confidence": 0.55,
                    "probabilities": {"minimal": 0.55, "escalate": 0.40, "normal": 0.05},
                }
            return {"decision": "conceptual"}

        mock_decide.side_effect = mock_dec
        gov = CognitiveGovernor()
        res = gov.assess_initial_task("High-risk ambiguity")
        self.assertEqual(res["thinking_budget"], "escalate")
        self.assertTrue(res["gating_adjusted"])
        self.assertIn("Upgraded minimal -> escalate", res["gating_reason"])


class TestPolicyAnalyzer(unittest.TestCase):
    def test_policy_analyzer_guards_sample_size(self):
        # With fewer than MIN_TASKS_FOR_POLICY_REVISION (10), no policy recommendations should be emitted
        events = [
            {"task_id": "1", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "2", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "3", "thinking_budget": "minimal", "rework_needed": True},
        ]
        policy = {"governor": {"minimal_budget_confidence_threshold": 0.70}, "subagents": {}}
        res = analyze_policy_performance(events, policy)
        self.assertEqual(res["total_tasks"], 3)
        self.assertEqual(len(res["recommendations"]), 0)

    def test_policy_analyzer_recommendations_with_sufficient_samples(self):
        # 10 tasks, minimal budget has 4/5 rework (> 25%)
        events = [
            {"task_id": "1", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "2", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "3", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "4", "thinking_budget": "minimal", "rework_needed": True},
            {"task_id": "5", "thinking_budget": "minimal", "rework_needed": False},
            {"task_id": "6", "thinking_budget": "normal", "rework_needed": False},
            {"task_id": "7", "thinking_budget": "normal", "rework_needed": False},
            {"task_id": "8", "thinking_budget": "normal", "rework_needed": False},
            {"task_id": "9", "thinking_budget": "normal", "rework_needed": False},
            {"task_id": "10", "thinking_budget": "normal", "rework_needed": False},
        ]
        policy = {"governor": {"minimal_budget_confidence_threshold": 0.70}, "subagents": {}}
        res = analyze_policy_performance(events, policy)
        self.assertEqual(res["total_tasks"], 10)
        self.assertGreater(len(res["recommendations"]), 0)
        self.assertIn("minimal_budget_confidence_threshold", res["recommendations"][0]["target"])

    def test_policy_analyzer_role_probation(self):
        # 8 events of literature-scout in coding with 0 problems caught -> recommends probation
        events = [
            {"category": "coding", "deepseek_roles": ["literature-scout"], "reviewer_found_problem": False}
            for _ in range(8)
        ]
        policy = {"governor": {}, "subagents": {}}
        res = analyze_policy_performance(events, policy)
        self.assertIn("literature-scout", res["role_stats"])
        self.assertEqual(res["role_stats"]["literature-scout"]["problems_found"], 0)
        targets = [r["target"] for r in res["recommendations"]]
        self.assertIn("role_routing.prune_low_yield_role", targets)
        self.assertIn("probation", res["recommendations"][0]["proposed"])


class TestJevEntropyAndEnsemble(unittest.TestCase):
    def test_calculate_entropy_extremes(self):
        det = {"minimal": 1.0, "normal": 0.0, "deep": 0.0}
        self.assertAlmostEqual(calculate_entropy(det), 0.0, places=4)

        uniform = {"minimal": 0.25, "normal": 0.25, "deep": 0.25, "escalate": 0.25}
        self.assertAlmostEqual(calculate_entropy(uniform), 1.0, places=4)

    @patch("jev_decide.jev_decide")
    def test_jev_ensemble_variance_and_agreement(self, mock_dec):
        mock_dec.side_effect = [
            {"status": "ok", "probabilities": {"act": 0.8, "test": 0.2}, "confidence": 0.8, "decision": "act", "cost_usd": 0.00001},
            {"status": "ok", "probabilities": {"act": 0.7, "test": 0.3}, "confidence": 0.7, "decision": "act", "cost_usd": 0.00001},
        ]
        ens = jev_ensemble("NEXT_ACTION", "State description", trials=2)
        self.assertEqual(ens["decision"], "act")
        self.assertAlmostEqual(ens["ensemble_probabilities"]["act"], 0.75, places=4)
        self.assertTrue(ens["is_unanimous"])

    @patch("jev_decide.jev_decide")
    def test_jev_framing_ensemble(self, mock_dec):
        mock_dec.side_effect = [
            {"status": "ok", "probabilities": {"minimal": 0.8, "normal": 0.2}, "confidence": 0.8, "decision": "minimal", "cost_usd": 0.00001},
            {"status": "ok", "probabilities": {"minimal": 0.3, "normal": 0.7}, "confidence": 0.7, "decision": "normal", "cost_usd": 0.00001},
            {"status": "ok", "probabilities": {"minimal": 0.9, "normal": 0.1}, "confidence": 0.9, "decision": "minimal", "cost_usd": 0.00001},
        ]
        ens = jev_framing_ensemble("THINKING_BUDGET", "Fix binary search bug")
        self.assertEqual(ens["status"], "ok")
        self.assertEqual(len(ens["framings"]), 3)
        self.assertTrue(ens["has_framing_divergence"])
        self.assertFalse(ens["is_unanimous"])
        self.assertEqual(ens["framing_decisions"]["objective"], "minimal")
        self.assertEqual(ens["framing_decisions"]["risk_averse"], "normal")


class TestTrajectorySensor(unittest.TestCase):
    def test_compute_trend(self):
        sensor = TrajectorySensor()
        trend = sensor.compute_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(trend["mean"], 3.0)
        self.assertGreater(trend["drift"], 0.9)
        self.assertEqual(trend["last_value"], 5.0)

    def test_predict_diminishing_returns(self):
        sensor = TrajectorySensor()
        res_zero = sensor.predict_diminishing_returns(consecutive_passes=3, marginal_gain_history=[0.0, 0.0, 0.0])
        self.assertGreaterEqual(res_zero["prob_diminishing_returns"], 0.80)
        self.assertEqual(res_zero["recommended_action"], "stop_or_act")

        res_prog = sensor.predict_diminishing_returns(consecutive_passes=1, marginal_gain_history=[1.0, 1.5])
        self.assertLess(res_prog["prob_diminishing_returns"], 0.70)

    def test_predict_branch_explosion(self):
        sensor = TrajectorySensor()
        safe = sensor.predict_branch_explosion(active_branch_count=2)
        self.assertFalse(safe["needs_pruning_gate"])

        danger = sensor.predict_branch_explosion(active_branch_count=5)
        self.assertTrue(danger["needs_pruning_gate"])

    def test_pluggable_forecaster_and_timesfm_adapter(self):
        h_forecaster = HeuristicTrajectoryForecaster()
        h_res = h_forecaster.forecast([1.0, 2.0, 3.0, 4.0], horizon=2)
        self.assertEqual(h_res["point_forecast"], 6.0)
        self.assertEqual(h_res["status"], "ok")

        tfm_adapter = TimesFMAdapter()
        tfm_res = tfm_adapter.forecast([1.0, 2.0, 3.0, 4.0], horizon=1)
        self.assertIn("point_forecast", tfm_res)
        self.assertIn(tfm_res["adapter"], ("timesfm_adapter_stub", "timesfm_live"))

        sensor = TrajectorySensor(forecaster=tfm_adapter)
        trend = sensor.compute_trend([1.0, 2.0, 3.0])
        self.assertIn("point_forecast", trend)


class TestCalibrationTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_file = Path(self.tmp.name) / "test_calibration.jsonl"
        self.tracker = CalibrationTracker(log_path=self.log_file)

    def test_calibration_recording_and_brier_score(self):
        records = [
            {"schema_version": 2, "outcome_source": "independent-review/fixture", "predicted_distribution": {"act": 1.0, "test": 0.0}, "observed_outcome": "act"},
            {"schema_version": 2, "outcome_source": "independent-review/fixture", "predicted_distribution": {"act": 0.0, "test": 1.0}, "observed_outcome": "test"},
        ]
        brier = calculate_multiclass_brier_score(records)
        self.assertAlmostEqual(brier, 0.0, places=4)

        bad_records = [
            {"schema_version": 2, "outcome_source": "independent-review/fixture", "predicted_distribution": {"act": 1.0, "test": 0.0}, "observed_outcome": "test"},
        ]
        bad_brier = calculate_multiclass_brier_score(bad_records)
        self.assertAlmostEqual(bad_brier, 2.0, places=4)

    def test_calibration_report_sample_size_guard(self):
        for _ in range(5):
            self.tracker.record_prediction_outcome(
                {"minimal": 0.8, "normal": 0.2},
                "minimal",
                outcome_source="independent-review/fixture",
            )
        report = self.tracker.generate_calibration_report()
        self.assertEqual(report["sample_count"], 5)
        self.assertFalse(report["is_statistically_calibrated"])
        self.assertIn("Preliminary/Uncalibrated", report["calibration_status_note"])

    def test_calibration_missing_classes_in_prediction(self):
        records = [
            {"schema_version": 2, "outcome_source": "independent-review/fixture", "predicted_distribution": {"minimal": 0.7, "normal": 0.3}, "observed_outcome": "deep"},
        ]
        brier = calculate_multiclass_brier_score(records)
        # (0.7 - 0.0)^2 + (0.3 - 0.0)^2 + (0.0 - 1.0)^2 = 0.49 + 0.09 + 1.0 = 1.58
        self.assertAlmostEqual(brier, 1.58, places=2)


class TestPolicyReplayHarness(unittest.TestCase):
    def test_replay_and_ablations(self):
        policy = {
            "governor": {"minimal_budget_confidence_threshold": 0.70},
            "subagents": {},
        }
        events = [
            {
                "task_id": "T1",
                "thinking_budget": "normal",
                "total_cost_usd": 0.002,
                "rework_needed": False,
                "category": "coding",
            },
            {
                "task_id": "T2",
                "thinking_budget": "minimal",
                "total_cost_usd": 0.0005,
                "rework_needed": True,
                "category": "research",
            },
        ]
        harness = PolicyReplayHarness(policy=policy)
        eval_res = harness.evaluate(events)
        self.assertEqual(eval_res["evaluated_tasks"], 2)
        self.assertEqual(eval_res["status"], "ok")

        ablations = harness.run_all_ablations(events)
        self.assertIn("no_jev", ablations)
        self.assertIn("no_critic", ablations)
        self.assertIn("no_branching", ablations)
        self.assertIn("no_diversity", ablations)
        self.assertIn("no_confidence_gating", ablations)
        self.assertGreater(ablations["no_critic"]["mean_projected_rework_risk"], 0.0)


class TestCognitiveGovernorExtensions(unittest.TestCase):
    def test_calculate_utility(self):
        gov = CognitiveGovernor()
        u1 = gov.calculate_utility(
            expected_info_gain=0.9,
            tokens=500,
            cost_usd=0.0001,
            latency_sec=1.5,
            prob_rework=0.05,
            discovery_bonus=0.5,
        )
        self.assertGreater(u1, 0.5)

        u2 = gov.calculate_utility(
            expected_info_gain=0.1,
            tokens=10000,
            cost_usd=0.05,
            latency_sec=30.0,
            prob_rework=0.8,
            discovery_bonus=0.0,
        )
        self.assertLess(u2, u1)

    def test_evaluate_branching(self):
        gov = CognitiveGovernor()
        res_concept = gov.evaluate_branching("Explore anomaly", "conceptual", active_branches_count=2)
        self.assertTrue(res_concept["should_branch"])

        res_full = gov.evaluate_branching("Explore anomaly", "conceptual", active_branches_count=6)
        self.assertFalse(res_full["should_branch"])
        self.assertIn("Branch limit reached", res_full["reason"])

    def test_marginal_gain_tracker(self):
        tracker = MarginalGainTracker(patience=2, min_gain=0.1)
        g1 = tracker.record_step(new_evidence=1, new_hypotheses=1)
        self.assertGreater(g1, 0.0)
        halt, _ = tracker.should_halt()
        self.assertFalse(halt)

        tracker.record_step(new_evidence=0, new_hypotheses=0)
        tracker.record_step(new_evidence=0, new_hypotheses=0)
        halt, reason = tracker.should_halt()
        self.assertTrue(halt)
        self.assertIn("Marginal gain stagnated", reason)


if __name__ == "__main__":
    unittest.main()
