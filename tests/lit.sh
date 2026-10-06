#!/bin/sh
# lit ships with LLVM's tools (Ubuntu: llvm-18-tools), beside FileCheck.
exec python3 "${LIT:-$(ls /usr/lib/llvm-*/build/utils/lit/lit.py | sort -V | tail -n 1)}" "$@"
