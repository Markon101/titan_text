"""Independent saved-corpus oracle; run with the audit JSON path."""
import json, sys
from collections import Counter

def carry_depth(a, b, width):
    # Detect carry leaving each prefix via integer prefix sums (not digit recurrence).
    bits = [int(a % (10 ** j) + b % (10 ** j) >= 10 ** j) for j in range(1, width + 1)]
    return max(map(len, "".join(map(str, bits)).split("0")))

# Report schema is finalized after reviewing the delegated implementation.

def verify_records(records):
    ids = set()
    cells = Counter()
    for r in records:
        a, b, w = r['a'], r['b'], r['width']
        assert 1 <= w <= 9 and 0 <= a < 10**w and 0 <= b < 10**w
        ident = f'w{w}_a{min(a,b)}_b{max(a,b)}'
        assert r['id'] == ident and ident not in ids
        ids.add(ident)
        h = 0xcbf29ce484222325
        for c in ident.encode():
            h = ((h ^ c) * 0x100000001b3) & ((1 << 64) - 1)
        split = 'train' if h % 100 < 80 else 'validation' if h % 100 < 90 else 'test'
        assert r['split'] == split
        assert r['depth'] == carry_depth(a, b, w)
        obs, target, mask = r['observation'], r['targets'], r['mask']
        expected = f'{a:0{w}d}+{b:0{w}d}=' + '?' * (w+1) + '.'
        assert obs == expected
        start = 2*w + 2
        assert target == '.'*start + f'{a+b:0{w+1}d}' + '.'
        assert mask == '0'*start + '1'*(w+1) + '0'
        cells[(split,r['depth'])] += 1
    assert len(cells) == 15 and set(cells.values()) == {64}
    return {'rows':len(records), 'unique_canonical_ids':len(ids), 'split_depth_cells':len(cells), 'rows_per_cell':64, 'oracle_and_masks':'passed'}

if __name__ == '__main__':
    doc=json.load(open(sys.argv[1]))
    records=doc['corpus']['records']
    print(json.dumps(verify_records(records), indent=2))
    train = [r for r in records if r['split'] == 'train']
    w = doc['config']['width']
    start = 2*w+2
    truths = lambda r: [int(c) for c,m in zip(r['targets'],r['mask']) if m == '1']
    mode = lambda values: min(range(10), key=lambda d: (-values.count(d),d))
    global_prior = mode([d for r in train for d in truths(r)])
    position_prior = [mode([truths(r)[i] for r in train]) for i in range(w+1)]
    for split in ['train','validation','test']:
        rows = [r for r in records if r['split'] == split]
        for name,pred in [('constant_zero',[0]*(w+1)),('global_digit_prior',[global_prior]*(w+1)),('position_prior',position_prior)]:
            correct = [[int(t==p) for t,p in zip(truths(r),pred)] for r in rows]
            cells = {}
            for r, cc in zip(rows,correct):
                for i,c in enumerate(cc): cells.setdefault((r['depth'],i),[]).append(c)
            expected = dict(digit_micro=sum(map(sum,correct))/(len(rows)*(w+1)),
                equal_depth_position_macro=sum(sum(v)/len(v) for v in cells.values())/len(cells),
                complete_answer=sum(all(c) for c in correct)/len(rows),
                leading_overflow_digit=sum(c[0] for c in correct)/len(rows),
                remaining_digit=sum(sum(c[1:]) for c in correct)/(len(rows)*w))
            actual=doc['audit']['by_split'][split][name]
            for key,value in expected.items(): assert abs(value-actual[key]) < 1e-12, (split,name,key)
    print('All 45 per-split baseline metrics independently verified.')
