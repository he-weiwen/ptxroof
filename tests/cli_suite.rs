use std::process::Command;

fn python(script: &[&str]) {
    let status = Command::new("python3")
        .args(script)
        .args(["--bin", env!("CARGO_BIN_EXE_ptxroof")])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("python3 runs");
    assert!(status.success(), "python3 {script:?} failed");
}

#[test]
#[ignore = "ci.sh runs it directly; .cargo/mutants.toml includes it"]
fn cli_and_acceptance_cases() {
    python(&["tests/run.py"]);
}

#[test]
#[ignore = "ci.sh runs it directly; .cargo/mutants.toml includes it"]
fn generated_loops_match_the_simulator() {
    python(&["tests/gen_loops.py", "--seed", "1", "--count", "200"]);
}
