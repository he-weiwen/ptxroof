#!/usr/bin/env bash
# CI: Rust checks, lit tests, and generated loops.
# Requires a rustup-managed toolchain (pinned by
# rust-toolchain.toml), python3, and lit and FileCheck from
# LLVM's tools (tests/lit.sh). No C++; the ptxas
# round trip of every fixture (tests/lit/roundtrip) runs when the CUDA
# toolkit is on PATH and is unsupported otherwise
# (PTXROOF_REQUIRE_CUDA=1 turns that into a failure, for the machine
# that has it). Likewise every Nsight Compute
# record is measured again where a GPU, ncu, the toolkit and gcc are
# present (PTXROOF_REQUIRE_GPU=1).
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
cargo build --locked # the lit tests run target/debug/ptxroof

tests/lit.sh -sv tests/lit
# Compare recognized trip counts with a simulator for generated loop kernels.
python3 tests/gen_loops.py --self-test
python3 tests/gen_loops.py --seed 1 --count 200

gpu_ready() {
    for tool in ncu ptxas cuobjdump gcc nvidia-smi; do
        command -v "$tool" > /dev/null || return 1
    done
    nvidia-smi -L > /dev/null 2>&1
}
if gpu_ready; then
    # The B200 records come from reports, not measure.py: tests/lit/b200-*.test check them.
    find tests/fixtures tests/lit/Inputs -name '*.ncu.json' -not -path 'tests/fixtures/b200/*' -print0 |
        sort -z | xargs -0 -n1 tools/ncu/measure.py --check
elif [ "${PTXROOF_REQUIRE_GPU:-0}" = 1 ]; then
    echo "ci.sh: PTXROOF_REQUIRE_GPU=1, but a GPU or ncu, ptxas, cuobjdump or gcc is missing" >&2
    exit 1
fi

echo "ci.sh: all green"
