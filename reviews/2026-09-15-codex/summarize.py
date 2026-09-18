#!/usr/bin/env python3
"""Recompute review tables from preserved agent results and the Rust audit."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
review = Path(__file__).resolve().parent
old = json.loads((root / 'reports/falsification_report.json').read_text())
audit = json.loads((review / 'research-heldout.json').read_text())
summary = {'historical_fixed_vs_zero': [], 'optimization_gate': [], 'historical_report_checks': {}}
for run in old['arms']:
    if run['arm'] != 'Arm1BlackboardFixed':
        continue
    panel = next(p for d in audit['data'] if d['seed'] == run['seed']
                 for p in d['panels'] if p['panel'] == 'validation')
    summary['historical_fixed_vs_zero'].append({
        'seed': run['seed'], 'fixed_digit_acc': run['val_acc'],
        'constant_zero_digit_acc': panel['constant_zero_digit_acc'],
        'fixed_exact_correct': sum(run['per_sample_correct']),
        'n': len(run['per_sample_correct']), 'unique_inputs': panel['unique_observations']})
checks = summary['historical_report_checks']
checks['all_15_runs_zero_exact_answers'] = all(not any(a['per_sample_correct']) for a in old['arms'])
checks['nonfinite_static_runs'] = sum(a['train_loss'] is None for a in old['arms'])
hist = old['adaptive_telemetry']['step_histogram']
checks['adaptive_batch_token_decision_mean'] = sum(i*n for i,n in enumerate(hist))/sum(hist)
def quantile(histogram, q):
    rank = max(1, __import__('math').ceil(sum(histogram)*q))
    cumulative = 0
    for steps, n in enumerate(histogram):
        cumulative += n
        if cumulative >= rank:
            return steps
checks['adaptive_median'] = quantile(hist, 0.5)
checks['adaptive_p95'] = quantile(hist, 0.95)
checks['original_reported_median'] = old['adaptive_telemetry']['median_steps']
checks['synthetic_extrap_plus2'] = old['extrapolation']['extrap_plus2_acc']
for run in audit['optimization_gate']:
    summary['optimization_gate'].append({
        'model': run['model'], 'seed': run['seed'], 'parameters': run['parameters'],
        'updates': run['updates'], 'final_training': run['curve'][-1],
        'held_out': run['held_out'], 'wall_seconds': run['wall_seconds'],
        'all_gradients_finite': run['all_gradients_finite']})
print(json.dumps(summary, indent=2))
