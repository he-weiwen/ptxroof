#!/usr/bin/env python3
"""Fixture lint and ptxas round trip.

  1. Fixture lint. Every committed .ptx under tests/fixtures/ declares
     where it came from (PLAN.md, fixture policy): a "Provenance:"
     header plus a sibling regen.sh (generated), a "HAND-WRITTEN:"
     header (authored for this suite), or a "HAND-EDITED:" header
     (negative case derived from a named base fixture).

  2. ptxas round trip. Every fixture's --dump-ast must assemble to the
     fixture's own SASS, except those listed in
     tests/roundtrip-allowlist.txt, which must still fail. Needs ptxas
     and cuobjdump; skipped without them unless PTXROOF_REQUIRE_CUDA is
     set.

Usage:
  run.py                  run both (binary: target/debug/ptxroof)
  run.py --bin PATH       use a different binary
  run.py --self-test      run the runner's own unit tests (no binary)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_BIN = REPO_ROOT / "target" / "debug" / "ptxroof"
ROUNDTRIP_ALLOWLIST = REPO_ROOT / "tests" / "roundtrip-allowlist.txt"

# --------------------------------------------------------------------------
# Fixture lint (PLAN.md, fixture policy)


def lint_fixtures(fixtures_dir):
    """Every .ptx declares its origin in a leading comment header."""
    errs = []
    for ptx in sorted(fixtures_dir.rglob("*.ptx")):
        head = ptx.read_text()[:2000]
        rel = ptx.relative_to(fixtures_dir)
        if "Provenance:" in head:
            if not (ptx.parent / "regen.sh").is_file():
                errs.append(f"{rel}: Provenance header but no regen.sh beside it")
        elif "HAND-WRITTEN:" not in head and "HAND-EDITED:" not in head:
            errs.append(
                f"{rel}: no Provenance / HAND-WRITTEN / HAND-EDITED header"
            )
    return errs


# --------------------------------------------------------------------------
# ptxas round trip: the dump must assemble to the fixture's own SASS.
# Needs the CUDA toolkit; skipped without it.


def sass_text(cubin):
    out = subprocess.run(
        ["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    return "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("/* 0x"))


def roundtrip_fixture(binary, ptx, tmp):
    """None when the dump assembles to the same SASS as the fixture; else why not."""
    lines = ptx.read_text().splitlines()
    arch = next((l.split()[1] for l in lines if l.startswith(".target")), None)
    if arch is None:
        return "no .target line"
    orig, dump, redo = tmp / "orig.cubin", tmp / "dump.ptx", tmp / "dump.cubin"
    r = subprocess.run(
        ["ptxas", f"-arch={arch}", str(ptx), "-o", str(orig)], capture_output=True, text=True
    )
    if r.returncode:
        return f"the fixture itself does not assemble: {r.stderr.strip()[:160]}"
    d = subprocess.run(
        [str(binary), "analyze", "--dump-ast", str(ptx)], capture_output=True, text=True
    )
    dump.write_text(d.stdout)
    r = subprocess.run(
        ["ptxas", f"-arch={arch}", str(dump), "-o", str(redo)], capture_output=True, text=True
    )
    if r.returncode:
        first = (r.stderr.strip().splitlines() or [""])[0]
        return f"ptxas rejects the dump: {first[-160:]}"
    if sass_text(orig) != sass_text(redo):
        return "the dump assembles to different SASS"
    return None


def read_allowlist(path):
    if not path.is_file():
        return []
    return [
        l.strip()
        for l in path.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]


def run_roundtrips(binary):
    """Return (results, failures): results are (label, rel, detail) rows."""
    rows, failures = [], 0
    allow = read_allowlist(ROUNDTRIP_ALLOWLIST)
    with tempfile.TemporaryDirectory() as tmp:
        for ptx in sorted(FIXTURES_DIR.rglob("*.ptx")):
            rel = str(ptx.relative_to(FIXTURES_DIR))
            reason = roundtrip_fixture(binary, ptx, Path(tmp))
            if reason is None and rel in allow:
                rows.append(("FAIL", rel, "round-trips now: remove it from roundtrip-allowlist.txt"))
                failures += 1
            elif reason is None:
                rows.append(("PASS", rel, ""))
            elif rel in allow:
                rows.append(("XFAIL", rel, reason))
            else:
                rows.append(("FAIL", rel, reason))
                failures += 1
    return rows, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args()

    if opts.self_test:
        return self_test()

    if not opts.bin.is_file():
        sys.exit(f"run.py: binary {opts.bin} not found — run `cargo build` first")

    failures = 0
    for err in lint_fixtures(FIXTURES_DIR):
        print(f"FAIL  lint    {err}")
        failures += 1

    n_roundtrips = 0
    if shutil.which("ptxas") and shutil.which("cuobjdump"):
        rows, rt_failures = run_roundtrips(opts.bin)
        for label, rel, detail in rows:
            print(f"{label:5} ptxas   {rel}" + (f": {detail}" if detail else ""))
        failures += rt_failures
        n_roundtrips = len(rows)
    elif os.environ.get("PTXROOF_REQUIRE_CUDA"):
        print("FAIL  ptxas   CUDA toolkit not on PATH and PTXROOF_REQUIRE_CUDA is set")
        failures += 1
    else:
        print("SKIP  ptxas   CUDA toolkit not on PATH (set PTXROOF_REQUIRE_CUDA=1 to fail instead)")

    print(f"run.py: {n_roundtrips} round trip(s), {failures} failure(s)")
    return 1 if failures else 0


def self_test():
    n = 0

    def ok(cond, what):
        nonlocal n
        n += 1
        assert cond, f"self-test failed: {what}"

    # fixture lint: each origin category, and each way to violate it
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "gen").mkdir()
        (root / "gen" / "a.ptx").write_text("// Provenance: regen.sh\n")
        ok(lint_fixtures(root) != [], "Provenance without regen.sh fails")
        (root / "gen" / "regen.sh").write_text("#!/bin/sh\n")
        ok(lint_fixtures(root) == [], "Provenance with regen.sh passes")
        (root / "micro").mkdir()
        (root / "micro" / "b.ptx").write_text("// no header at all\n")
        ok(any("no Provenance" in e for e in lint_fixtures(root)),
           "headerless fixture named")
        (root / "micro" / "b.ptx").write_text("// HAND-WRITTEN: minimal\n")
        ok(lint_fixtures(root) == [], "HAND-WRITTEN passes without regen.sh")
        (root / "micro" / "c.ptx").write_text("// HAND-EDITED: base k2, edit X\n")
        ok(lint_fixtures(root) == [], "HAND-EDITED passes without regen.sh")

    # round-trip allowlist: comments and blanks ignored, paths as written
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "allow.txt"
        f.write_text("# why\n\nk14/k14.sm_80.ptx\n  micro/x.ptx  \n")
        ok(read_allowlist(f) == ["k14/k14.sm_80.ptx", "micro/x.ptx"], "allowlist reader")
        ok(read_allowlist(Path(tmp) / "missing.txt") == [], "missing allowlist is empty")

    print(f"run.py --self-test: {n} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
