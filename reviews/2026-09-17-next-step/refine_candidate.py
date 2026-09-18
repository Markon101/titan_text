from pathlib import Path
base=Path('reviews/2026-09-17-next-step')
s=(base/'revised-arithmetic_corpus.rs').read_text()
# Reject candidates from other splits without consuming their IDs; one shared retry budget per row.
s=s.replace('    target_depth: usize,\n    used_ids:', '    target_depth: usize,\n    requested_split: Split,\n    used_ids:')
s=s.replace('        if used_ids.contains(&id) {', '        if split_for(width, a, b) != requested_split || used_ids.contains(&id) {')
a=s.index('            let mut made = 0usize;');b=s.index('\n        }\n    }\n    Ok(Corpus',a)
s=s[:a]+'''            for _ in 0..config.count_per_depth {
                records.push(build_record(&mut rng, config.width, depth, split, &mut used_ids)?);
            }'''+s[b:]
s=s.replace('BTreeMap<(usize, usize), u8>', 'BTreeMap<usize, u8>')
s=s.replace('let mut counts: BTreeMap<(usize, usize), [usize; 10]>', 'let mut counts: BTreeMap<usize, [usize; 10]>')
s=s.replace('let key = (r.depth, i);','let key = i;').replace('position_prior.get(&(r.depth, i))','position_prior.get(&i)').replace('prior.get(&(r.depth, i))','prior.get(&i)')
s=s.replace('/// Fit a position prior over train targets only: for each (depth, position)\n/// cell, the most frequent digit. Uses ONLY position and depth, never the\n/// ground-truth target of the row being predicted.', '/// Fit the most frequent digit at each position using training rows only.')
s=s.replace('    pub examples: Vec<CorpusRecord>,','    pub by_split: BTreeMap<String, SplitMetrics>,')
idx=s.index('/// Fit a global digit prior')
s=s[:idx]+'''#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub rows: usize,
    pub constant_zero: BaselineMetrics,
    pub global_digit_prior: BaselineMetrics,
    pub position_prior: BaselineMetrics,
}

'''+s[idx:]
s=s.replace('    if corpus.records.is_empty() {\n        bail!("corpus has no records");\n    }', '''    if corpus.records.len() != 3 * corpus.config.depths.len() * corpus.config.count_per_depth {
        bail!("record count does not match configuration");
    }''')
s=s.replace('        if r.depth > width {','        if !corpus.config.depths.contains(&r.depth) {')
needle='        let expected = oracle_sum_string(r.a, r.b, r.width);'
s=s.replace(needle,'''        let (obs, targets, mask) = make_observation(r.a, r.b, r.width);
        if r.observation != obs || r.targets != targets || r.mask != mask {
            bail!("record {} has invalid observation, targets, or mask", r.id);
        }
'''+needle)
s=s.replace('let key = format!("{}:{}", r.depth, i);','let key = format!("{}:{}:{}", r.split.as_str(), r.depth, i);')
a=s.index('    let examples: Vec<CorpusRecord>');b=s.index('\n    Ok(Audit',a)
s=s[:a]+'''    let global_digit = (0..10).max_by(|&a, &b| global_prior[a].partial_cmp(&global_prior[b]).unwrap().then_with(|| b.cmp(&a))).unwrap() as u8;
    let mut by_split = BTreeMap::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let mut panel = corpus.clone();
        panel.records.retain(|r| r.split == split);
        by_split.insert(split.as_str().to_string(), SplitMetrics {
            rows: panel.records.len(),
            constant_zero: evaluate(&panel, |_, _| 0)?,
            global_digit_prior: evaluate(&panel, |_, _| global_digit)?,
            position_prior: evaluate(&panel, |_, i| position_prior[&i])?,
        });
    }
'''+s[b:]
s=s.replace('        examples,','        by_split,').replace('evaluate(corpus, |r, i| {','evaluate(corpus, |_r, i| {')
# Replace tautological prior test with actual leakage interventions.
a=s.index('        // The position prior must be a function');b=s.index('\n    #[test]\n    fn default_config_shape',a)
s=s[:a]+'''        let cfg = CorpusConfig { seed: 23, width: 3, depths: vec![0,1,2,3], count_per_depth: 8 };
        let mut corpus = build_corpus(&cfg).unwrap();
        let prior = fit_position_prior(&corpus);
        let global = fit_global_digit_prior(&corpus);
        assert_eq!(prior.len(), cfg.width + 1);
        for r in &mut corpus.records {
            r.depth = 99;
            if r.split != Split::Train {
                r.targets = r.targets.chars().map(|c| if c.is_ascii_digit() { '9' } else { c }).collect();
            }
        }
        assert_eq!(prior, fit_position_prior(&corpus));
        assert_eq!(global, fit_global_digit_prior(&corpus));
    }
'''+s[b:]
Path('src/arithmetic_corpus.rs').write_text(s)
Path('examples/arithmetic_corpus_audit.rs').write_text((base/'revised-arithmetic_corpus_audit.rs').read_text())
