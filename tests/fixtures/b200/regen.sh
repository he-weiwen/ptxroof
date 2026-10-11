#!/usr/bin/env bash
# Regenerates the B200 fixtures (PLAN.md, fixture policy) from the Nsight
# Compute reports of sol-study's b200_ncu_triton run
# (github.com/he-weiwen/sol-study, private; SOL_STUDY, default
# ~/compilers/sol-study). record_launches.py replays each profiled case with
# the study's Triton and torch (SOL_PYTHON) as if this machine were the
# B200, recording each launch without running it, one process per case;
# import_reports.py then writes each profiled kernel's PTX and record. Needs
# the triton kernel corpus (CORPUS) for the source header, a CUDA GPU for
# the cases' tensors, ncu, cuobjdump and readelf.
set -euo pipefail
cd "$(dirname "$0")"
sol=${SOL_STUDY:-$HOME/compilers/sol-study}
python=${SOL_PYTHON:-$HOME/compilers/triton-jitspec-base/.venv/bin/python}
corpus=${CORPUS:-$HOME/compilers/triton_kernel_corpus}
here=$(pwd)
tmp=$(mktemp -d)
python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))))' \
    "$sol/results/b200_ncu_triton/ncu_summary.json" |
while read -r case; do
    (cd "$sol" && PYTHONPATH="$sol/vendor" "$python" "$here/record_launches.py" "$tmp/launches.jsonl" "$case")
done
rm -f ./*.sm_100a.ptx ./*.sm_100a.ncu.json
python3 import_reports.py "$sol" "$corpus" "$tmp/launches.jsonl" .
rm -rf "$tmp"
