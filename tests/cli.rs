use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "titan-text-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_titan_text"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }

    fn success(&self, args: &[&str]) -> Output {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out
    }

    fn manifest(&self, dir: &str) -> serde_json::Value {
        serde_json::from_slice(&fs::read(self.0.join(dir).join("manifest.json")).unwrap()).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn help_never_starts_work_or_touches_checkpoints() {
    let scratch = Scratch::new();
    for args in [vec![], vec!["--help"], vec!["-h"], vec!["help"]] {
        let out = scratch.success(&args);
        assert!(String::from_utf8_lossy(&out.stdout).contains("COMMANDS:"));
    }
    for (command, canonical) in [
        ("train", "train"),
        ("probe", "probe"),
        ("diagnostics", "probe"),
        ("ns-probe", "ns-probe"),
        ("navier-stokes", "ns-probe"),
        ("blowup", "ns-probe"),
        ("fluid-probe", "ns-probe"),
        ("rollout", "rollout"),
        ("falsify", "falsify"),
    ] {
        for args in [
            vec![command, "--help"],
            vec![command, "-h"],
            vec!["help", command],
            vec![command, "--load-dir", "missing", "--help"],
        ] {
            let out = scratch.success(&args);
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(stdout.contains(&format!("titan_text {canonical} [OPTIONS]")));
            assert!(stdout.contains("--output"));
            assert!(!stdout.contains("Loading checkpoint weights"));
            assert!(!stdout.contains("Loading trained weights"));
            assert!(!stdout.contains("TRAINING SESSION"));
            assert!(out.stderr.is_empty());
        }
    }
    assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 0);
}

#[test]
fn rejects_bad_arguments_before_io() {
    let scratch = Scratch::new();
    let cases: &[(&[&str], &str)] = &[
        (&["trane"], "unknown command"),
        (&["train", "--epohs", "1"], "--epohs"),
        (&["train", "--epochs"], "missing value for --epochs"),
        (
            &["train", "--save-dir", "--epochs", "1"],
            "missing value for --save-dir",
        ),
        (&["train", "--epochs", "nope"], "--epochs"),
        (&["train", "--epochs", "0"], "positive"),
        (&["train", "--epochs", "-1"], "--epochs"),
        (&["train", "--epochs", "1.5"], "--epochs"),
        (
            &["train", "--epochs", "9999999999999999999999999999999999999"],
            "--epochs",
        ),
        (&["train", "--dev-steps=0"], "positive"),
        (&["rollout", "--seq-len=0"], "positive"),
        (&["rollout", "--seq-len=2147483648"], "indexing"),
        (&["rollout", "--horizon=0"], "positive"),
        (&["probe", "--horizon", "0"], "positive"),
        (&["ns-probe", "--horizon", "0"], "positive"),
        (&["train", "--lr=NaN"], "finite"),
        (&["train", "--lr=inf"], "finite"),
        (&["train", "--lr=-0.01"], "greater than zero"),
        (&["train", "--lr=0"], "greater than zero"),
        (&["train", "--viscosity=NaN"], "finite"),
        (&["rollout", "--viscosity=inf"], "finite"),
        (&["ns-probe", "--viscosity=-0.1"], "non-negative"),
        (&["ns-probe", "--forcing-amp=-1"], "non-negative"),
        (&["ns-probe", "--forcing-amp=NaN"], "finite"),
        (&["probe", "--eps=0"], "greater than zero"),
        (&["probe", "--eps=-0.01"], "greater than zero"),
        (&["probe", "--eps=inf"], "finite"),
        (&["probe", "--eps=1e-30"], "squared range"),
        (&["probe", "--eps=1e30"], "squared range"),
        (&["train", "--task", "typo"], "--task"),
        (&["rollout", "--epochs=1"], "unexpected argument"),
        (&["probe", "--prompt=hello"], "unexpected argument"),
        (&["falsify", "--forcing-amp=1"], "unexpected argument"),
        (&["train", "extra"], "unexpected argument"),
        (&["rollout", "--raw", "true"], "unexpected argument"),
        (&["rollout", "--raw=true"], "does not take a value"),
        (&["train", "--patterns-only"], "requires --output"),
        (&["rollout", "--horizon=1", "--steps=2"], "duplicate"),
        (&["rollout", "--steps=1", "--dev-steps=1"], "duplicate"),
        (&["rollout", "--raw", "--patterns-only"], "duplicate"),
        (&["train", "--epochs=1", "--epochs=2"], "duplicate"),
        (&["train", "--output="], "must not be empty"),
        (&["rollout", "--load-dir", "missing", "--bogus"], "--bogus"),
        (&["train", "--help", "--bogus"], "--bogus"),
        (&["help", "train", "extra"], "unexpected arguments"),
        (&["--help", "extra"], "unexpected arguments"),
        (&["train", "--help=true"], "does not take a value"),
    ];
    for (args, error) in cases {
        let out = scratch.run(args);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stdout.is_empty(), "work started for {args:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(error),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(fs::read_dir(&scratch.0).unwrap().count(), 0);
}

#[test]
fn checkpoint_continuation_probes_and_pattern_exports() {
    let scratch = Scratch::new();
    scratch.success(&[
        "train",
        "--task=paren",
        "--epochs=1",
        "--dev-steps=1",
        "--seq-len=8",
        "--horizon=2",
        "--viscosity=0.05",
        "--lr=0.002",
        "--save-dir",
        "parent",
        "--output",
        "nested/train.txt",
        "--raw",
    ]);
    let manifest = scratch.manifest("parent");
    assert_eq!(manifest["task"], "dyck");
    assert_eq!(manifest["cumulative_step"], 1);
    assert_eq!(
        fs::read_to_string(scratch.0.join("nested/train.txt"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    let weights = fs::read(scratch.0.join("parent/model.safetensors")).unwrap();
    let metadata = fs::read(scratch.0.join("parent/manifest.json")).unwrap();

    let rollout = scratch.success(&[
        "rollout",
        "--load-dir=parent",
        "--task=paren",
        "--steps=3",
        "--prompt",
        "(()🙂)",
        "--seq-len=7",
        "--raw",
        "--output=nested/rollout.txt",
    ]);
    assert_eq!(
        rollout.stdout,
        fs::read(scratch.0.join("nested/rollout.txt")).unwrap()
    );
    assert_eq!(String::from_utf8_lossy(&rollout.stdout).lines().count(), 3);
    // Former aliases and values containing hyphens/equals remain usable.
    scratch.success(&[
        "rollout",
        "--dev-steps=1",
        "--prompt=--hello=world",
        "--raw",
    ]);
    scratch.success(&["rollout", "--horizon=1", "--prompt=", "--raw"]);
    for command in ["probe", "falsify", "ns-probe"] {
        let output = format!("nested/{command}/report.json");
        scratch.success(&[
            command,
            "--load-dir=parent",
            "--horizon=2",
            "--output",
            &output,
        ]);
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(scratch.0.join(output)).unwrap()).unwrap();
        assert!(report.is_object());
    }
    for command in ["train", "rollout", "probe", "ns-probe"] {
        let out = scratch.run(&[command, "--load-dir=parent", "--task=text"]);
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("conflicts with checkpoint task"));
    }
    scratch.success(&[
        "train",
        "--load-dir=parent",
        "--save-dir=child",
        "--epochs=1",
    ]);
    let child = scratch.manifest("child");
    assert_eq!(child["cumulative_step"], 2);
    assert_eq!(child["task"], "dyck");
    assert_eq!(child["config"], manifest["config"]);
    assert_ne!(
        fs::read(scratch.0.join("child/model.safetensors")).unwrap(),
        weights
    );
    assert_eq!(
        fs::read(scratch.0.join("parent/model.safetensors")).unwrap(),
        weights
    );
    assert_eq!(
        fs::read(scratch.0.join("parent/manifest.json")).unwrap(),
        metadata
    );
    // With no save-dir, continuation still uses load-dir.
    scratch.success(&["train", "--load-dir=child", "--epochs=1"]);
    assert_eq!(scratch.manifest("child")["cumulative_step"], 3);

    // Schema-1 manifests without task metadata can still select their vocabulary.
    let mut legacy = manifest.clone();
    legacy.as_object_mut().unwrap().remove("task");
    fs::write(
        scratch.0.join("parent/manifest.json"),
        serde_json::to_vec(&legacy).unwrap(),
    )
    .unwrap();
    scratch.success(&[
        "rollout",
        "--load-dir=parent",
        "--task=dyck",
        "--horizon=1",
        "--raw",
    ]);

    for (section, key) in [
        ("train", "log_every"),
        ("train", "save_every"),
        ("field", "seq_len"),
    ] {
        let mut invalid = manifest.clone();
        invalid["config"][section][key] = 0.into();
        fs::write(
            scratch.0.join("parent/manifest.json"),
            serde_json::to_vec(&invalid).unwrap(),
        )
        .unwrap();
        let out = scratch.run(&["train", "--load-dir=parent"]);
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("positive"));
    }
    let mut overflow = manifest;
    overflow["cumulative_step"] = usize::MAX.into();
    fs::write(
        scratch.0.join("parent/manifest.json"),
        serde_json::to_vec(&overflow).unwrap(),
    )
    .unwrap();
    let out = scratch.run(&["train", "--load-dir=parent", "--epochs=1"]);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("overflow"));
    assert_eq!(
        fs::read(scratch.0.join("parent/model.safetensors")).unwrap(),
        weights
    );
}
