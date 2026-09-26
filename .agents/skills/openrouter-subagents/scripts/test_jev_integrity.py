"""Offline regressions for Jev failure handling and research evidence boundaries."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'research-team' / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts'))
import jev_decide as jev
from calibration_tracker import CalibrationTracker, compute_brier_and_ece
from cognitive_governor import CognitiveGovernor
from research_coordinator import ResearchCoordinator
from telemetry import TelemetryTracker
import jev_science_governor as science
import benchmark_suite as benchmark


def answer(choice='supported', probabilities=None, confidence=0.8):
    return {'answers': {'evidence_status': {
        'choice': choice, 'confidence': confidence,
        'probabilities': probabilities if probabilities is not None else {'supported': 0.8, 'insufficient': 0.2},
    }}}


def decision(kind, *args, **kwargs):
    label = {'EVIDENCE_STATUS': 'supported', 'NEXT_ACTION': 'continue_reasoning',
             'THINKING_BUDGET': 'normal', 'UNCERTAINTY_TYPE': 'missing_evidence'}[kind]
    return {'status': 'ok', 'decision': label, 'probabilities': {label: 1.0}, 'confidence': 1.0}


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, cwd)
        blocker = patch('urllib.request.urlopen', side_effect=AssertionError('Network prohibited in offline regressions'))
        self.network = blocker.start()
        self.addCleanup(blocker.stop)


class TestResponseIntegrity(OfflineCase):
    def test_malformed_responses_do_not_become_success(self):
        bad = [{}, [], {'answers': None}, {'answers': {'evidence_status': {}}},
               answer('outside_schema', {'outside_schema': 1.0}),
               answer(probabilities={'supported': float('nan')}),
               answer(probabilities={'supported': float('inf')}),
               answer(probabilities={'supported': True}),
               answer(probabilities={'supported': -0.2, 'insufficient': 1.2}),
               answer(probabilities={'supported': 0.4}),
               answer(probabilities={'insufficient': 1.0}),
               answer(confidence=float('nan')), answer(confidence=1.1)]
        for payload in bad:
            with self.subTest(payload=payload), patch.object(jev, 'call_jev', return_value=payload):
                res = jev.jev_decide('EVIDENCE_STATUS', 'unchanged evidence')
                self.assertEqual(res['status'], 'fallback')
                self.assertTrue(res['fallback_applied'])
                self.assertEqual(res['probabilities'], {})
                self.assertEqual(res['decision'], 'insufficient')

    def test_valid_sparse_distribution_and_missing_confidence(self):
        payload = answer()
        del payload['answers']['evidence_status']['confidence']
        with patch.object(jev, 'call_jev', return_value=payload):
            res = jev.jev_decide('EVIDENCE_STATUS', 'evidence')
        self.assertEqual(res['status'], 'ok')
        self.assertEqual(res['confidence'], 0.8)

    def test_all_failed_ensemble_and_framings_are_unavailable(self):
        with patch.object(jev, 'call_jev', side_effect=RuntimeError('offline failure')):
            for result in (jev.jev_ensemble('EVIDENCE_STATUS', '', trials=2),
                           jev.jev_framing_ensemble('EVIDENCE_STATUS', '')):
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['decision'], 'inconclusive')
                self.assertEqual(result['ensemble_probabilities'], {})
                self.assertFalse(result['is_unanimous'])
                self.assertEqual(result['successful_trials'], 0)

    def test_partial_ensemble_excludes_failed_votes(self):
        good = decision('EVIDENCE_STATUS')
        failed = {'status': 'fallback', 'decision': 'insufficient', 'probabilities': {'insufficient': 1.0}, 'fallback_applied': True}
        with patch.object(jev, 'jev_decide', side_effect=[good, failed]):
            res = jev.jev_ensemble('EVIDENCE_STATUS', '', trials=2)
        self.assertEqual(res['status'], 'partial')
        self.assertEqual(res['ensemble_probabilities'], {'supported': 1.0})
        self.assertFalse(res['is_unanimous'])
        self.assertEqual(res['failed_trials'], 1)

    def test_science_failures_are_inconclusive(self):
        with patch.object(jev, 'call_jev', side_effect=RuntimeError('offline failure')):
            results = [science.arbitrate_mechanism({}), science.audit_causal_transplant({}),
                       science.score_falsification_risk('claim', {}), science.recommend_next_phase({})]
        for res in results:
            self.assertEqual(res['decision'], 'inconclusive')
            self.assertEqual(res['status'], 'fallback')
            self.assertEqual(res['probabilities'], {})

    def test_science_missing_import_is_inconclusive(self):
        spec = importlib.util.spec_from_file_location('science_missing_dependency', ROOT / 'scripts/jev_science_governor.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'jev_decide': None}):
            spec.loader.exec_module(module)
        res = module.audit_causal_transplant({})
        self.assertEqual(res['decision'], 'inconclusive')
        self.assertEqual(res['confidence'], 0.0)
        self.assertTrue(res['fallback_applied'])

    def test_custom_schema_reaches_ensemble_cli_and_failed_exit(self):
        schema = {'question_id': 'custom', 'criteria': {'yes': 'yes', 'inconclusive': 'unknown'}, 'default': 'yes'}
        path = Path('schema.json')
        path.write_text(json.dumps(schema))
        with patch.object(jev, 'call_jev', return_value={'answers': {'custom': {'choice': 'yes', 'probabilities': {'yes': 1.0}}}}) as call, contextlib.redirect_stdout(io.StringIO()):
            status = jev.main(['custom', 'context', '--schema-file', str(path), '--ensemble', '2'])
        self.assertEqual(status, 0)
        self.assertEqual(call.call_count, 2)
        self.assertIn('custom', call.call_args.args[1])
        with patch.object(jev, 'call_jev', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(jev.main(['EVIDENCE_STATUS', 'context']), 1)

    def test_invalid_ensemble_size_does_not_call_model(self):
        with patch.object(jev, 'call_jev') as call:
            for n in (0, -1, 9, True):
                with self.assertRaises(ValueError):
                    jev.jev_ensemble('EVIDENCE_STATUS', '', trials=n)
            with self.assertRaises(ValueError):
                jev.jev_framing_ensemble('EVIDENCE_STATUS', '', framings=[])
            call.assert_not_called()


class TestResearchEvidence(OfflineCase):
    def test_model_support_is_not_evidence_outcome_or_completion(self):
        coord = ResearchCoordinator(decider_fn=decision)
        for _ in range(3):
            gate = coord.verify_synthesis_gate('claim', 'same evidence')
            self.assertFalse(gate['is_supported'])
            self.assertTrue(gate['model_supports_claim'])
            self.assertNotEqual(gate['action_branch'], 'conclude_and_report')
        self.assertEqual(coord.governor.evidence_gathered_count, 0)
        self.assertEqual(coord.governor.consecutive_reasoning_passes, 3)
        self.assertFalse(Path('.agents/telemetry/calibration_records.jsonl').exists())

    def test_only_new_explicit_artifact_ids_reset_stagnation(self):
        coord = ResearchCoordinator(decider_fn=decision)
        coord.verify_synthesis_gate('claim', 'evidence', evidence_ids=['run-1:sha256-abc'])
        self.assertEqual(coord.governor.evidence_gathered_count, 1)
        gate = coord.verify_synthesis_gate('claim', 'evidence', evidence_ids=['run-1:sha256-abc'])
        self.assertEqual(gate['new_evidence_ids'], [])
        self.assertEqual(coord.governor.consecutive_reasoning_passes, 1)
        self.assertEqual(coord.governor.evidence_gathered_count, 1)
        with self.assertRaises(ValueError):
            coord.verify_synthesis_gate('claim', 'evidence', evidence_ids=[''])

    def test_verify_cli_passes_explicit_evidence_ids(self):
        import research_coordinator
        with patch.object(research_coordinator, 'jev_decide', side_effect=decision), contextlib.redirect_stdout(io.StringIO()) as output:
            research_coordinator.main(['verify', 'claim', 'evidence', '--evidence-id', 'run:abc', '--json'])
        self.assertEqual(json.loads(output.getvalue())['new_evidence_ids'], ['run:abc'])

    def test_fallback_cannot_support_claim_even_with_optimistic_label(self):
        gov = CognitiveGovernor(decider_fn=lambda *a, **k: {'status': 'fallback', 'decision': 'supported', 'fallback_applied': True})
        res = gov.verify_evidence_status('unverified')
        self.assertEqual(res['evidence_status'], 'insufficient')
        self.assertFalse(res['model_supports_claim'])

    def test_fallback_not_exported_as_router_training_sample(self):
        tracker = TelemetryTracker('research', 'normal')
        tracker.record_jev('NEXT_ACTION', 'act', probabilities={'act': 1.0}, fallback=True)
        self.assertEqual(tracker.export_router_training_samples('task', router_path=Path('router.jsonl')), [])
        self.assertFalse(Path('router.jsonl').exists())


class TestCalibrationIntegrity(OfflineCase):
    def test_legacy_rows_excluded_without_mutating_log(self):
        path = Path('calibration.jsonl')
        legacy = {'predicted_distribution': {'supported': 1.0}, 'observed_outcome': 'supported'}
        path.write_text(json.dumps(legacy) + '\n')
        original = path.read_bytes()
        res = CalibrationTracker(path).generate_calibration_report()
        self.assertEqual(res['sample_count'], 0)
        self.assertEqual(res['excluded_count'], 1)
        self.assertIsNone(res['mean_brier_score'])
        self.assertEqual(path.read_bytes(), original)

    def test_many_wrong_predictions_never_claim_calibration(self):
        tracker = CalibrationTracker(Path('calibration.jsonl'))
        for i in range(30):
            tracker.record_prediction_outcome({'supported': 1.0, 'contradicted': 0.0}, 'contradicted', outcome_source=f'held-out/run-{i}')
        report = tracker.generate_calibration_report()
        self.assertTrue(report['has_minimum_samples'])
        self.assertFalse(report['is_statistically_calibrated'])
        self.assertEqual(report['mean_brier_score'], 2.0)
        self.assertEqual(report['expected_calibration_error'], 1.0)

    def test_bad_rows_do_not_dilute_metrics(self):
        row = {'schema_version': 2, 'outcome_source': 'independent-test/run-1', 'predicted_distribution': {'a': 1.0}, 'observed_outcome': 'b'}
        report = compute_brier_and_ece([row, {}, dict(row, predicted_distribution={'a': math.nan})])
        self.assertEqual(report['sample_count'], 1)
        self.assertEqual(report['excluded_count'], 2)
        self.assertEqual(report['mean_brier_score'], 2.0)

    def test_record_requires_provenance_and_valid_probability(self):
        tracker = CalibrationTracker(Path('calibration.jsonl'))
        with self.assertRaises(ValueError):
            tracker.record_prediction_outcome({'a': 1.0}, 'a', outcome_source='')
        with self.assertRaises(ValueError):
            tracker.record_prediction_outcome({'a': 2.0}, 'a', outcome_source='run-1')
        self.assertFalse(tracker.log_path.exists())


class TestSimulationIntegrity(OfflineCase):
    def test_default_simulation_is_offline_and_labeled(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(jev, 'call_jev') as call:
            res = benchmark.run_benchmark_suite(quick=True)
        call.assert_not_called()
        self.network.assert_not_called()
        self.assertEqual(res['measurement_kind'], 'simulation')
        self.assertFalse(res['empirical_performance'])
        self.assertIn('Synthetic simulation only', benchmark.render_comparison_table(res))

    def test_simulation_preserves_real_research_state(self):
        real = ResearchCoordinator(decider_fn=decision)
        real.state.add_question("real open question")
        real.frontier.add_branch("real hypothesis")
        files = [real.state.path, real.frontier.path]
        before = {path: path.read_bytes() for path in files}
        benchmark.run_single_task_controller(benchmark.BENCHMARK_TASKS[1], coordinator=real)
        for path in files:
            self.assertEqual(path.read_bytes(), before[path])
        self.assertEqual(real.governor.consecutive_reasoning_passes, 0)

    def test_no_jev_condition_never_calls_model_even_in_live_mode(self):
        with patch.object(jev, 'call_jev') as call:
            res = benchmark.run_single_task_controller(benchmark.BENCHMARK_TASKS[1], ablation='no_jev', live_routing=True)
        call.assert_not_called()
        self.network.assert_not_called()
        self.assertEqual(res['routing_mode'], 'simulation')


if __name__ == '__main__':
    unittest.main()
