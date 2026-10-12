#!/usr/bin/env bash
# Regenerates the B200 fixtures (PLAN.md, fixture policy) from the sol-study
# Nsight Compute reports reports.txt selects (github.com/he-weiwen/sol-study,
# private; SOL_STUDY, default ~/compilers/sol-study). record_launches.py
# replays each report's command with the study's Triton and torch
# (SOL_PYTHON) as if this machine were the B200, recording each launch
# without running it, one process per report; import_reports.py then writes
# each selected kernel's PTX and record. Needs the triton kernel corpus
# (CORPUS) for the source header, a CUDA GPU for the commands' tensors, ncu,
# cuobjdump and readelf.
set -euo pipefail
cd "$(dirname "$0")"
sol=${SOL_STUDY:-$HOME/compilers/sol-study}
python=${SOL_PYTHON:-$HOME/compilers/triton-jitspec-base/.venv/bin/python}
corpus=${CORPUS:-$HOME/compilers/triton_kernel_corpus}
here=$(pwd)
tmp=$(mktemp -d)
python3 import_reports.py --commands "$sol" "$tmp" | while read -r cmd; do
    (cd "$sol" && RCA_OUT="$tmp" PYTHONPATH="$sol/vendor" "$python" "$here/record_launches.py" "$tmp/launches.jsonl" "$cmd")
done
rm -f ./*.sm_100a.ptx ./*.sm_100a.ncu.json
python3 import_reports.py "$sol" "$corpus" "$tmp/launches.jsonl" .
rm -rf "$tmp"
