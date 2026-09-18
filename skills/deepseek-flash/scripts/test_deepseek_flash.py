"""Offline regression suite; all network transport is mocked by default."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import deepseek_flash as d

FAKE_KEY = 'test-credential-not-real-1234567890'


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = self.enterContext(patch.object(d, 'curl_once', side_effect=AssertionError('Unexpected network attempt')))
        self.key = self.enterContext(patch.object(d, 'load_key', return_value=FAKE_KEY))

    def call(self, **kwargs):
        return d.deepseek_flash(kwargs.pop('task', 'Find a falsifying control.'), root=self.root, **kwargs)

    def success(self, **overrides):
        data = {'choices': [{'message': {'content': 'A testable conclusion.', 'reasoning': 'PRIVATE TRACE', 'reasoning_details': ['PRIVATE TRACE']}, 'finish_reason': 'stop'}], 'model': d.DEFAULT_MODEL, 'id': 'fake-id', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'cost': 0.001, 'text': 'PRIVATE TRACE', 'completion_tokens_details': {'reasoning_tokens': 0}}}
        data.update(overrides)
        self.transport.side_effect = None
        self.transport.return_value = (200, data, 0)
        return data

    def test_explicit_span_is_only_context_sent(self):
        (self.root / 'source.rs').write_text('UNSELECTED\nlet target = 2;\nUNSELECTED\n')
        self.success()
        result = self.call(files=['source.rs:2-2'])
        payload = self.transport.call_args.args[2]
        user = json.loads(payload['messages'][1]['content'])
        self.assertEqual(user['files'][0]['content'], 'let target = 2;\n')
        self.assertNotIn('UNSELECTED', json.dumps(payload))
        self.assertEqual(result['files'][0]['sha256'], d.digest('let target = 2;\n')['sha256'])

    def test_dry_run_has_no_network_or_context_echo(self):
        result = self.call(context='sensitive research context', dry_run=True)
        self.assertEqual(result['status'], 'dry_run')
        self.assertNotIn('sensitive research context', json.dumps(result))
        self.transport.assert_not_called()

    def test_usage_and_reasoning_filter(self):
        self.success()
        result = self.call()
        self.assertEqual(result['usage']['total_tokens'], 120)
        self.assertNotIn('PRIVATE TRACE', json.dumps(result))
        self.assertTrue(self.transport.call_args.args[2]['reasoning']['exclude'])

    def test_missing_usage_is_null(self):
        self.success(usage=None)
        self.assertIsNone(self.call()['usage'])

    def test_reasoning_in_content_is_removed(self):
        self.success(choices=[{'message': {'content': '<think>PRIVATE TRACE</think>Conclusion.'}, 'finish_reason': 'stop'}])
        self.assertEqual(self.call()['answer'], 'Conclusion.')

    def test_reasoning_only_is_error(self):
        for content in [None, '<think>PRIVATE TRACE', '<think>PRIVATE TRACE</think>']:
            self.success(choices=[{'message': {'content': content, 'reasoning': 'PRIVATE TRACE'}, 'finish_reason': 'stop'}])
            with self.assertRaises(d.DelegationError):
                self.call()

    def test_secret_refused_in_all_inputs(self):
        for kwargs in [{'task': FAKE_KEY}, {'context': FAKE_KEY}, {'model': FAKE_KEY}]:
            with self.assertRaises(d.DelegationError):
                self.call(**kwargs)
        (self.root / 'source.rs').write_text(FAKE_KEY)
        with self.assertRaises(d.DelegationError):
            self.call(files=['source.rs'])
        self.transport.assert_not_called()

    def test_protected_files_and_escape_refused(self):
        (self.root / '.env.openrouter').write_text('placeholder')
        (self.root / 'secret.txt').write_text('placeholder')
        (self.root / 'weights.bin').write_text('placeholder')
        (self.root / 'alias').symlink_to(self.root.parent)
        for selection in ['.env.openrouter', 'secret.txt', 'weights.bin', '../outside', 'alias/outside']:
            with self.assertRaises(d.DelegationError):
                self.call(files=[selection], dry_run=True)

    def test_span_validation(self):
        (self.root / 'a').write_text('one\ntwo\n')
        for selection in ['a:0-1', 'a:2-1', 'a:1-3']:
            with self.assertRaises(d.DelegationError):
                self.call(files=[selection], dry_run=True)

    def test_binary_and_large_source_refused(self):
        for content in [b'abc\0def', b'x' * (d.MAX_SOURCE_BYTES + 1), b'\xff']:
            (self.root / 'input').write_bytes(content)
            with self.assertRaises(d.DelegationError):
                self.call(files=['input'], dry_run=True)

    def test_limits_and_nonfinite_parameters(self):
        for kwargs in [{'context': 'x' * 140000}, {'files': ['x'] * 17}, {'max_tokens': 0}, {'max_tokens': 32769}, {'temperature': float('nan')}, {'timeout': float('inf')}, {'timeout': 0}, {'retries': 5}, {'max_request_bytes': 524289}]:
            with self.assertRaises(d.DelegationError):
                self.call(dry_run=True, **kwargs)
        self.transport.assert_not_called()

    def test_temperature_and_override(self):
        self.assertEqual(self.call(dry_run=True, role='mechanistic-competitor')['temperature'], 0.8)
        self.assertEqual(self.call(dry_run=True, mode='review')['temperature'], 0.1)
        self.assertEqual(self.call(dry_run=True, mode='hypothesis', temperature=0.3)['temperature'], 0.3)
        with patch.dict(os.environ, {'DEEPSEEK_FLASH_MODEL': 'deepseek/explicit-test'}):
            self.assertEqual(self.call(dry_run=True)['requested_model'], 'deepseek/explicit-test')
            self.assertEqual(self.call(dry_run=True, model=d.DEFAULT_MODEL)['requested_model'], d.DEFAULT_MODEL)

    def test_https_and_url_validation(self):
        for url in ['http://localhost/api/v1', 'https://user:pass@example.com', 'https://example.com/?token=x', 'https://example.com/#x', 'https://example.com/\nheader']:
            with self.assertRaises(d.DelegationError):
                self.call(base_url=url, dry_run=True)

    def test_transient_backoff_and_usage_scope(self):
        success = self.success()
        self.transport.side_effect = [(429, {'error': {'code': 429}}, 0), (503, {}, 0), (200, success, 0)]
        with patch.object(d.time, 'sleep') as sleep:
            result = self.call()
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])
        self.assertEqual(result['attempts'], 3)
        self.assertIn('earlier attempts', result['usage_scope'])

    def test_embedded_api_error_retries(self):
        success = self.success()
        self.transport.side_effect = [(200, {'error': {'code': 503, 'message': 'unavailable'}}, 0), (200, success, 0)]
        with patch.object(d.time, 'sleep'):
            self.assertEqual(self.call()['attempts'], 2)

    def test_auth_and_credit_failures_never_retry(self):
        for status in [400, 401, 402, 403, 404]:
            self.transport.reset_mock()
            self.transport.side_effect = None
            self.transport.return_value = (status, {'error': {'message': FAKE_KEY}}, 0)
            with self.assertRaises(d.DelegationError) as caught:
                self.call()
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.transport.assert_called_once()

    def test_http_failure_retains_safe_request_audit(self):
        self.transport.side_effect = None
        self.transport.return_value = (401, {'error': {'message': FAKE_KEY}}, 0)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(d.main(['run', '--task', 'Audit leakage']), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result['purpose'], 'Audit leakage')
        self.assertEqual(result['requested_model'], d.DEFAULT_MODEL)
        self.assertGreater(result['approximate_input_tokens'], 0)
        self.assertIn('created_at_utc', result)
        self.assertIsNone(result['usage'])
        self.assertNotIn(FAKE_KEY, output.getvalue())

    def test_timeout_and_network_retry_bounds(self):
        self.transport.side_effect = None
        self.transport.return_value = (0, None, 28)
        with patch.object(d.time, 'sleep'):
            with self.assertRaises(d.DelegationError):
                self.call(retries=1)
        self.assertEqual(self.transport.call_count, 2)
        self.transport.reset_mock()
        with patch.object(d.time, 'monotonic', side_effect=[0, 0, 121]):
            with self.assertRaises(d.DelegationError):
                d.request_json(d.DEFAULT_BASE, '/chat/completions', {}, FAKE_KEY)
        self.transport.assert_called_once()

    def test_tls_failure_does_not_retry(self):
        self.transport.side_effect = None
        self.transport.return_value = (0, None, 60)
        with self.assertRaises(d.DelegationError):
            self.call()
        self.transport.assert_called_once()

    def test_invalid_response_fails(self):
        for body in [None, {}, {'choices': []}]:
            self.transport.side_effect = None
            self.transport.return_value = (200, body, 0)
            with self.assertRaises(d.DelegationError):
                self.call()

    def test_truncation_and_json_validation(self):
        self.success(choices=[{'message': {'content': '{"conclusions": []}'}, 'finish_reason': 'stop'}])
        self.assertEqual(self.call(json_answer=True)['answer_json'], {'conclusions': []})
        self.success(choices=[{'message': {'content': 'partial'}, 'finish_reason': 'length'}])
        self.assertEqual(self.call()['status'], 'incomplete')
        self.assertEqual(self.call(json_answer=True)['status'], 'incomplete')

    def test_response_key_redacted(self):
        self.success(choices=[{'message': {'content': 'Echo: ' + FAKE_KEY}, 'finish_reason': 'stop'}])
        self.assertNotIn(FAKE_KEY, json.dumps(self.call()))

    def test_missing_key_smoke_fails_without_network(self):
        self.key.return_value = ''
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(d.main(['smoke']), 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'error')
        self.transport.assert_not_called()

    def test_invalid_cli_does_not_echo_values(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                d.main(['run', '--task', 'test', '--unknown', FAKE_KEY])
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn(FAKE_KEY, output.getvalue())
        self.transport.assert_not_called()

    def test_discovery_no_auth_and_no_substitution(self):
        self.transport.side_effect = None
        self.transport.return_value = (200, {'data': [{'id': 'deepseek/some-other-flash'}]}, 0)
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(d.main(['discover']), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result['target'], d.DEFAULT_MODEL)
        self.assertEqual(result['matches'], [])
        self.assertEqual(self.transport.call_args.args[3], '')


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.enterContext(patch.object(Path, 'cwd', return_value=self.root))
        self.enterContext(patch.object(d, '__file__', str(self.root / 'skill/scripts/deepseek_flash.py')))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_environment_is_only_credential_source(self):
        os.environ['OPENROUTER_API_KEY'] = FAKE_KEY
        with patch.object(Path, 'read_text', side_effect=AssertionError('No file reads permitted')):
            self.assertEqual(d.load_key(), FAKE_KEY)

    def test_absent_environment_ignores_local_secret(self):
        (self.root / '.env.openrouter').write_text('OPENROUTER_API_KEY=' + FAKE_KEY)
        with patch.object(Path, 'read_text', side_effect=AssertionError('No file reads permitted')):
            self.assertEqual(d.load_key(), d.PERMANENT_OPENROUTER_KEY)


    def test_invalid_environment_is_sanitized(self):
        for invalid in ['too-short', 'invalid value with spaces', 'a' * 20 + '\nheader']:
            os.environ['OPENROUTER_API_KEY'] = invalid
            with self.assertRaises(d.DelegationError) as caught:
                d.load_key()
            self.assertNotIn(invalid, str(caught.exception))

    def test_removed_env_file_option_rejected_without_echo(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                d.main(['smoke', '--env-file', 'unused'])
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn('unused', output.getvalue())


class CurlTransportTests(unittest.TestCase):
    def test_auth_only_in_stdin_and_no_redirect_or_curlrc(self):
        response = subprocess.CompletedProcess([], 0, b'{"ok":true}\n200', b'')
        with patch.object(d.subprocess, 'run', return_value=response) as run, patch.dict(os.environ, {'OPENROUTER_API_KEY': FAKE_KEY}):
            status, body, network = d.curl_once(d.DEFAULT_BASE, '/chat/completions', {'text': 'quotes " and newline\n and $HOME'}, FAKE_KEY, 3)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ['curl', '-q'])
        self.assertNotIn(FAKE_KEY, ' '.join(argv))
        self.assertNotIn('OPENROUTER_API_KEY', run.call_args.kwargs['env'])
        self.assertIn(FAKE_KEY.encode(), run.call_args.kwargs['input'])
        self.assertNotIn('--location', argv)
        self.assertIn('--max-filesize', argv)
        self.assertEqual((status, body, network), (200, {'ok': True}, 0))

    def test_transport_timeout_does_not_log_secret(self):
        with patch.object(d.subprocess, 'run', side_effect=subprocess.TimeoutExpired('curl', 1, output=FAKE_KEY)):
            self.assertEqual(d.curl_once(d.DEFAULT_BASE, '/models', None, FAKE_KEY, 1), (0, None, 28))


if __name__ == '__main__':
    unittest.main()
