#!/bin/sh
# Runs LLVM's lit: $LIT (a lit executable or a lit.py) when set, else `lit`
# or `llvm-lit` on PATH (`uv tool install lit`, `pip install lit`), else the
# newest Ubuntu llvm-N-tools' lit.py. tests/lit/lit.cfg.py finds FileCheck.
if [ -n "$LIT" ]; then
    case $LIT in
    *.py) exec python3 "$LIT" "$@" ;;
    *) exec "$LIT" "$@" ;;
    esac
fi
for lit in lit llvm-lit; do
    if command -v "$lit" >/dev/null 2>&1; then
        exec "$lit" "$@"
    fi
done
lit=$(ls /usr/lib/llvm-*/build/utils/lit/lit.py 2>/dev/null | sort -V | tail -n 1)
if [ -n "$lit" ]; then
    exec python3 "$lit" "$@"
fi
echo "tests/lit.sh: no lit found; install one (uv tool install lit, pip install lit," \
    "apt install llvm-18-tools) or set LIT" >&2
exit 127
