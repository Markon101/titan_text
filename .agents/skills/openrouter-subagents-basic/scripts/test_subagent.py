"""Offline unit tests for subagent.py."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from subagent import (
    load_api_key,
    read_file_span,
    run_subagent,
    sha256_digest,
    PERMANENT_OPENROUTER_KEY,
    SubagentError,
)


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


if __name__ == "__main__":
    unittest.main()
