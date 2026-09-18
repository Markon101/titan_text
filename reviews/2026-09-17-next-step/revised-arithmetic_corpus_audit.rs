//! Runnable audit example for the arithmetic corpus.
//!
//! Usage: cargo run --example arithmetic_corpus_audit -- OUTPUT.json
//!
//! Creates OUTPUT.json exclusively (never overwrites). Rejects malformed CLI
//! before any compute. No training is performed.

#[path = "../src/arithmetic_corpus.rs"]
mod arithmetic_corpus;

use anyhow::{bail, Context, Result};
use arithmetic_corpus::{audit_corpus, build_corpus, CorpusConfig, ARITHMETIC_CORPUS_SCHEMA};
use serde::Serialize;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Serialize)]
struct Output {
    schema: String,
    seed: u64,
    config: CorpusConfig,
    corpus: arithmetic_corpus::Corpus,
    audit: arithmetic_corpus::Audit,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        bail!(
            "usage: {} OUTPUT.json",
            args.first().map(|s| s.as_str()).unwrap_or("program")
        );
    }
    let out_path = &args[1];
    if out_path.is_empty() {
        bail!("output path must be non-empty");
    }
    if out_path.starts_with('-') {
        bail!("unknown option: {}", out_path);
    }

    let config = CorpusConfig {
        seed: 20260917,
        width: 4,
        depths: vec![0, 1, 2, 3, 4],
        count_per_depth: 64,
    };
    config.validate().context("invalid default config")?;

    let corpus = build_corpus(&config).context("build corpus")?;
    let audit = audit_corpus(&corpus).context("audit corpus")?;

    let output = Output {
        schema: ARITHMETIC_CORPUS_SCHEMA.to_string(),
        seed: config.seed,
        config: config.clone(),
        corpus,
        audit,
    };

    let json = serde_json::to_string_pretty(&output).context("serialize output")?;

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)
        .with_context(|| format!("create output exclusively: {}", out_path))?;
    file.write_all(json.as_bytes()).context("write output")?;
    file.write_all(b"\n").context("write newline")?;
    file.flush().context("flush output")?;

    println!("wrote {} records to {}", output.corpus.records.len(), out_path);
    Ok(())
}
