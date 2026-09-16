#!/usr/bin/env bash
# CI: Rust checks, CLI/acceptance cases, and generated loops.
# Requires a rustup-managed toolchain (pinned by
# rust-toolchain.toml) and python3 >= 3.11. No LLVM, no C++; the ptxas
# round trip of every fixture runs when the CUDA toolkit is on PATH and
# is skipped otherwise (PTXROOF_REQUIRE_CUDA=1 turns the skip into a
# failure, for the machine that has it).
set -euo pipefail
cd "$(dirname "$0")"

# --locked: refuse to run if Cargo.lock disagrees with Cargo.toml, so the
# committed lockfile is always the one actually being built.
cargo fmt --check
# CARGO_INCREMENTAL=0: clippy replays cached incremental results and
# can silently skip lints for items already in the shared check
# cache (observed here: PR 10 lints surfacing two PRs late).
CARGO_INCREMENTAL=0 cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --locked # the test runner drives target/debug/ptxroof

python3 tests/run.py --self-test
python3 tests/run.py
# Compare recognized trip counts with a simulator for generated loop kernels.
python3 tests/gen_loops.py --self-test
python3 tests/gen_loops.py --seed 1 --count 200

echo "ci.sh: all green"
