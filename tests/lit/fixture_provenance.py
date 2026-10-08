#!/usr/bin/env python3
"""Every committed .ptx under FIXTURES declares where it came from
(PLAN.md, fixture policy): a "Provenance:" header plus a sibling
regen.sh (generated), a "HAND-WRITTEN:" header (authored for this
suite), or a "HAND-EDITED:" header (negative case derived from a named
base fixture).

Usage: fixture_provenance.py FIXTURES
       fixture_provenance.py --self-test
"""

import sys
import tempfile
from pathlib import Path


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

    print(f"fixture_provenance.py --self-test: {n} assertions passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    errors = lint_fixtures(Path(sys.argv[1]))
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1 if errors else 0)
