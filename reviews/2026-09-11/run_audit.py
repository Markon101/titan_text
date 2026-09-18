#!/usr/bin/env python3
"""Compile and run audit.rs using this repository's existing debug dependencies.
Prerequisite: cargo test --locked --offline in the repository.
No model or checkpoint files are saved; only audit and results.jsonl beside this script.
The Rust source imports repository modules by absolute path, so it audits current code.
"""
from pathlib import Path
import subprocess

artifact_dir = Path(__file__).resolve().parent
repository = Path('/data/data/com.termux/files/home/projects/titan_text')
dependencies = repository / 'target/debug/deps'
command = ['rustc', '--edition=2021', '--crate-name', 'titan_architecture_audit',
           str(artifact_dir / 'audit.rs'), '-L', 'dependency=' + str(dependencies),
           '-o', str(artifact_dir / 'audit')]
for name in ('anyhow', 'candle_core', 'candle_nn', 'serde', 'serde_json', 'rand'):
    matches = sorted(dependencies.glob('lib' + name + '-*.rlib'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one cached debug rlib for {name}; found {matches}')
    command.extend(['--extern', name + '=' + str(matches[0])])
subprocess.run(command, check=True)
with (artifact_dir / 'results.jsonl').open('w') as output:
    subprocess.run([str(artifact_dir / 'audit')], stdout=output, check=True)
print((artifact_dir / 'results.jsonl').read_text(), end='')
