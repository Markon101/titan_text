"""Reproduce a bounded live self-review using Gemini's actual meeting runner."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / '.agents/skills/openrouter-subagents/scripts'))
sys.path.insert(0, str(REPO / '.agents/skills/research-team/scripts'))
import subagent
from research_meeting import ResearchMeeting
import research_meeting

# Bound transport retries for this audit; do not change the production helper.
original_transport = subagent.call_openrouter
def bounded_transport(payload, key, **kwargs):
    return original_transport(payload, key, **dict(kwargs, retries=0))
subagent.call_openrouter = bounded_transport

OBJECTIVE = '''Audit this agent research harness as production research tooling. Identify up to four actionable remaining improvements, ranked by impact. For each give exact supplied file/line evidence, a concrete triggering scenario, the smallest offline falsifying test, and a minimal fix. Distinguish reproduced facts from inspection-based inferences and design proposals. Also name one component to simplify or remove. Do not assume recent fixes establish overall reliability. Do not recommend more agents or more recursive reflection without a measurable benefit. Stay within 1200 words so the review finishes within its output budget.'''

def public_result(result):
    # Never persist private reasoning, including the helper's reasoning-only fallback.
    result = dict(result)
    reasoning = result.pop('reasoning', None)
    if reasoning and result.get('answer') == reasoning:
        result['answer'] = ''
        result['audit_status'] = 'reasoning_only'
    elif result.get('status') == 'ok' and (result.get('finish_reason') != 'stop' or not result.get('answer', '').strip()):
        result['audit_status'] = 'incomplete'
    else:
        result['audit_status'] = result.get('status', 'error')
    if result.get('error'):
        result['error'] = re.sub(r'sk-or-v1-[A-Za-z0-9_-]+', '[REDACTED]', str(result['error']))[:500]
    return result

if __name__ == '__main__':
    stage = sys.argv[1]
    final_only = '--final-only' in sys.argv[2:]
    prefix = 'final-only-' if final_only else ''
    if final_only:
        original_worker = research_meeting.run_subagent
        def final_answer_worker(*args, **kwargs):
            return original_worker(*args, **dict(kwargs, enable_reasoning=False))
        research_meeting.run_subagent = final_answer_worker
    context = (OUT / 'context.txt').read_text()
    with tempfile.TemporaryDirectory(prefix='harness-live-review-') as isolated:
        os.chdir(isolated)
        meeting = ResearchMeeting(OBJECTIVE, root=Path(isolated))
        if stage == 'round1':
            results = meeting.run_round_1_independent(['code-reviewer', 'statistical-agent', 'architectural-minimalist'], context=context, max_workers=3, timeout=180)
        elif stage == 'round2':
            first = json.loads((OUT / f'{prefix}round1.json').read_text())
            results = meeting.run_round_2_synthesis(first, roles_to_synthesize=['falsification-arbiter'], context=context, max_workers=1, timeout=180)
        else:
            raise SystemExit('Use round1 or round2')
        clean = {role: public_result(result) for role, result in results.items()}
        (OUT / f'{prefix}{stage}.json').write_text(json.dumps(clean, indent=2))
        for role, result in clean.items():
            print(json.dumps({'role': role, 'status': result.get('status'), 'audit_status': result['audit_status'], 'finish_reason': result.get('finish_reason'), 'answer_chars': len(result.get('answer', '')), 'usage': result.get('usage'), 'error': result.get('error')}), flush=True)
