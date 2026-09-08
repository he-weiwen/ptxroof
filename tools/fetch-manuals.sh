#!/usr/bin/env bash
# Fetches the official NVIDIA references  into refs/, which
# is untracked: NVIDIA copyright — cite, don't commit. The PTX ISA
# manual is the grammar reference PRs 03-04 transcribe against.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p refs
curl -fL --retry 3 -o refs/ptx-isa.html \
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
curl -fL --retry 3 -o refs/cuda-binary-utilities.html \
    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
echo "fetched into refs/ (untracked; cite, don't commit)"
