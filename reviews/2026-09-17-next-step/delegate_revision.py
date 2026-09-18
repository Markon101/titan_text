"""Bounded long-code delegation: one attempt, 180-second transport timeout."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/data/data/com.termux/files/home/.codex/skills/deepseek-flash/scripts')
import deepseek_flash as ds
original = ds.curl_once
def long_once(base, endpoint, payload, key, timeout):
    return original(base, endpoint, payload, key, 180)
ds.curl_once = long_once
r = ds.deepseek_flash(
    task="Write untested candidate source code, not a review or a plan. Return ONLY a JSON object with files:[{path,content}]. Do not claim execution. Codex verifies locally. " + Path('reviews/2026-09-17-next-step/revision-task.txt').read_text(),
    role='coder', context='',
    files=['reviews/2026-09-17-next-step/revision-task.txt', 'reviews/2026-09-17-next-step/builder-task.txt', 'reviews/2026-09-17-next-step/arithmetic_corpus.rs', 'reviews/2026-09-17-next-step/arithmetic_corpus_audit.rs'],
    max_tokens=14000, timeout=180, retries=0, json_answer=False,
)
Path('reviews/2026-09-17-next-step/deepseek-code.json').write_text(json.dumps(r,indent=2)+'\n')
print(r['status'])
