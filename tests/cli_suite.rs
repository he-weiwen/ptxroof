use std::process::Command;

#[test]
#[ignore = "ci.sh runs it directly; .cargo/mutants.toml includes it"]
fn lit_tests() {
    let status = Command::new("tests/lit.sh")
        .args(["-q", "tests/lit"])
        .env("PTXROOF", env!("CARGO_BIN_EXE_ptxroof"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("tests/lit.sh runs");
    assert!(status.success(), "lit failed");
}
