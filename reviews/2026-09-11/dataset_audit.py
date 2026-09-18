#!/usr/bin/env python3
"""Read-only reproduction of src/dataset.rs; writes JSON to stdout.

No training or checkpoint mutation. Source phrases are read from the actual
Rust file, and generator conditions are reproduced exactly in Python.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re

ROOT = Path('/data/data/com.termux/files/home/projects/titan_text')
BATCH_SIZE = 8
SEQ_LEN = 32
source = ROOT.joinpath('src/dataset.rs').read_text()

def phrases_from_source(name):
    block = re.search(r'let ' + name + r' = vec!\[(.*?)\];', source, re.S).group(1)
    return re.findall(r'"([^"\\]*)"\.to_string\(\)', block)

train_phrases = phrases_from_source('train_phrases')
val_phrases = phrases_from_source('val_phrases')

def text_batch(phrases, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    rows = []
    for b in range(batch_size):
        phrase = phrases[b % len(phrases)]
        offset = (b * 7) % len(phrase)
        inputs = ''.join(phrase[(i + offset) % len(phrase)] for i in range(seq_len))
        targets = ''.join(phrase[(i + offset + 1) % len(phrase)] for i in range(seq_len))
        rows.append(dict(batch_index=b, phrase_index=b % len(phrases), offset=offset,
                         input=inputs, target=targets))
    return rows

def dyck_batch(is_val, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    rows = []
    for b in range(batch_size):
        seq = []
        depth = 0
        depths = []
        max_depth = 6 if is_val else 3
        min_depth = 2 if is_val else 0
        for i in range(seq_len):
            if depth <= min_depth:
                depth += 1
                c = '('
            elif depth >= max_depth or i == seq_len - 1:
                depth -= 1
                c = ')'
            elif (b + i) % 2 == 0:
                depth += 1
                c = '('
            else:
                depth -= 1
                c = ')'
            seq.append(c)
            depths.append(depth)
        inputs = ''.join(seq)
        targets = ''.join(inputs[(i + 1) % seq_len] for i in range(seq_len))
        rows.append(dict(batch_index=b, input=inputs, target=targets,
                         final_depth=depth, peak_depth=max(depths), minimum_prefix_depth=min(depths),
                         valid_dyck_word=depth == 0 and min(depths) >= 0))
    return rows

def score(predictions, targets):
    correct = sum(a == b for pred, target in zip(predictions, targets) for a, b in zip(pred, target))
    total = sum(map(len, targets))
    return dict(correct=correct, total=total, accuracy=correct / total)

def ring_right_copy(inputs):
    return [s[1:] + s[:1] for s in inputs]

def falsification_shuffle(inputs):
    indices = list(range(SEQ_LEN))
    for i in reversed(range(1, SEQ_LEN)):
        j = (i * 7) % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return [''.join(s[j] for j in indices) for s in inputs]

text_train = text_batch(train_phrases)
text_val = text_batch(val_phrases)
bigram = defaultdict(Counter)
for row in text_train:
    for x, y in zip(row['input'], row['target']):
        bigram[x][y] += 1
# Counter resolves ties by first observation, in exact row/token order.
bigram_map = {x: ys.most_common(1)[0][0] for x, ys in bigram.items()}
majority = Counter(''.join(row['target'] for row in text_train)).most_common(1)[0][0]

report = dict(source_path=str(ROOT / 'src/dataset.rs'),
              source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              batch_size=BATCH_SIZE, sequence_length=SEQ_LEN,
              method='Exact Python reproduction of deterministic Rust loops; text phrases parsed from current source. No model execution or training.',
              text={}, dyck={})
for name, rows, phrases in [('train', text_train, train_phrases), ('validation', text_val, val_phrases)]:
    x = [r['input'] for r in rows]
    y = [r['target'] for r in rows]
    naive_falsification = score(ring_right_copy(falsification_shuffle(x)), y)
    base = score(ring_right_copy(x), y)
    report['text'][name] = dict(phrase_count=len(phrases), phrase_characters=sum(map(len, phrases)),
        phrases=phrases, unique_input_windows=len(set(x)), unique_target_windows=len(set(y)),
        repeated_calls_identical=rows == text_batch(phrases), rows=rows,
        ring_right_copy=base,
        interior_right_copy=score([s[1:] for s in x], [s[:-1] for s in y]),
        training_fitted_bigram=score([''.join(bigram_map.get(c, majority) for c in s) for s in x], y),
        training_majority=score([majority * len(s) for s in x], y),
        ring_right_copy_after_falsification_shuffle=naive_falsification,
        copy_shortcut_passes_context_verdict=(naive_falsification['accuracy'] < base['accuracy'] * 0.7 and base['accuracy'] > 0.4))
for is_val in [False, True]:
    name = 'validation' if is_val else 'train'
    rows = dyck_batch(is_val)
    x = [r['input'] for r in rows]
    y = [r['target'] for r in rows]
    report['dyck'][name] = dict(unique_input_windows=len(set(x)),
        repeated_calls_identical=rows == dyck_batch(is_val),
        valid_dyck_words=sum(row['valid_dyck_word'] for row in rows),
        sample_count=len(rows), peak_depths=sorted(set(row['peak_depth'] for row in rows)),
        ending_depths=sorted(set(row['final_depth'] for row in rows)), rows=rows,
        ring_right_copy=score(ring_right_copy(x), y),
        opposite_current_token=score([''.join(')' if c == '(' else '(' for c in s) for s in x], y))
report['text']['train_validation_input_intersection'] = sorted(set(r['input'] for r in text_train) & set(r['input'] for r in text_val))
report['limitations'] = [
    'This diagnoses available data and shortcuts; it does not establish which strategy any trained checkpoint uses.',
    'Exact deterministic dataset replay is intentional code behavior; no sampling RNG is present.',
    'Token-level baselines are calculated over all 256 positions per split unless explicitly named interior.',
    'Bigram tie breaking uses first observed target; this baseline is a low-cost reference only.',
]
print(json.dumps(report, indent=2))
