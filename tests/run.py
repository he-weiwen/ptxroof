#!/usr/bin/env python3
"""CLI and acceptance test runner (historically called tiers T2 and T3).

Deliberately implementation-language-neutral and dependency-free:
stdlib only, Python >= 3.11 (for tomllib). If ptxroof were ever
rewritten again, this harness and its committed test inputs would
survive unchanged.

Pipeline, in order:

  0. Fixture lint. Every committed .ptx under tests/fixtures/ declares
     where it came from (PLAN.md, fixture policy): a "Provenance:"
     header plus a sibling regen.sh (generated), a "HAND-WRITTEN:"
     header (authored for this suite), or a "HAND-EDITED:" header
     (negative case derived from a named base fixture).

  1. CLI tests (T2). Every directory under tests/cli/ with a case.toml
     is run unconditionally and must pass. A case declares:
       args         = ["analyze", "{case}/in.ptx"]  argv after the
                      binary; "{case}" expands to the case dir,
                      "{fixtures}" to tests/fixtures
       exit         = 0         expected exit code (default 0)
       check_stream = "stdout"  stream expected.checks applies to
                                (or "stderr")
       json         = false     force-parse stdout as JSON even without
                                expected.json (so the report verifier
                                runs)
     plus optional committed expected-output files:
       expected.json    compared field-by-field against stdout parsed
                        as JSON (partial comparison, see below)
       expected.checks  CHECK lines: substrings that must appear in
                        order — the matching semantics of LLVM
                        FileCheck's CHECK: directives. '#'-prefixed
                        and blank lines are comments.

  2. Report verifier. Internal-consistency checks run on EVERY case's
     parsed JSON output — in the spirit of LLVM's IR verifier:
     identities that must hold for any correct implementation, with no
     per-case expectations and no human review. The registry below is
     empty in PR 01; checks register here as their analyses land
     .

  3. Acceptance scenarios (T3). tests/acceptance/status.toml lists the
     acceptance scenarios, each referencing a case by name with status
     pass | xfail ("expected failure" — the LLVM lit / pytest term).
     Enforced in both directions: a pass scenario that fails blocks CI,
     and an xfail scenario that passes is ALSO an error — it forces the
     status.toml edit that records progress.

  4. Minimum-coverage thresholds. status.toml [[min_coverage]] entries.
     Each JSON report may carry a "coverage" object of
     {metric: {"num": N, "den": D}} fraction pairs (counts, not
     percentages — percentages cannot be aggregated without weights).
     The runner sums num/den across all reports; for each enforced
     entry, achieved% below the recorded minimum fails, and achieved%
     above it is also an error until --raise-min rewrites the minimum
     up to the achieved value — a visible status.toml diff. Minimums
     may only rise (a ratchet: it turns one way).

Partial-comparison semantics for expected.json (the T2 contract):
  - objects: every expected key must exist in actual and match; EXTRA
    actual keys are ignored (new report fields never break old tests).
  - arrays: same length, element-wise match (arrays are ordered data;
    partial comparison applies to object keys only).
  - scalars: exact equality, including floats — analysis is exact
    arithmetic on static counts, so expected outputs store exact values.

Usage:
  run.py                  run everything (binary: target/debug/ptxroof)
  run.py --bin PATH       use a different binary
  run.py --raise-min      also record beaten coverage minimums in
                          status.toml
  run.py --self-test      run the runner's own unit tests (no binary)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("run.py needs python3 >= 3.11 (stdlib tomllib)")

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_TESTS_DIR = REPO_ROOT / "tests" / "cli"
ACCEPTANCE_DIR = REPO_ROOT / "tests" / "acceptance"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
STATUS_FILE = ACCEPTANCE_DIR / "status.toml"
DEFAULT_BIN = REPO_ROOT / "target" / "debug" / "ptxroof"
ROUNDTRIP_ALLOWLIST = REPO_ROOT / "tests" / "roundtrip-allowlist.txt"
CASE_TIMEOUT_S = 120

# --------------------------------------------------------------------------
# Report verifier : (name, fn(report) -> [violation, ...]).
# Each fn receives one case's parsed JSON report and returns human-readable
# violation strings — identities any correct implementation satisfies, with
# no per-case expectations. (The IR-level identities that need data the
# JSON deliberately omits — sum of per-block tallies == kernel totals,
# measurement provenance resolves — live as Rust integration tests in
# tests/report_invariants.rs.)


def _verify_instruction_accounting(report):
    """Every instruction is accounted exactly once (the check v1's
    AsyncCopy silent-zero bug would have failed)."""
    out = []
    parts = ("flop", "non_flop_arith", "memory", "sync", "communication",
             "control", "ignore", "unknown")
    for k in report.get("kernels", []):
        c = k.get("instruction_classes", {})
        got = sum(c.get(p, 0) for p in parts)
        if got != c.get("total", -1):
            out.append(
                f"kernel {k.get('name')}: class counts sum to {got}, "
                f"total is {c.get('total')}"
            )
    return out


def _verify_unparsed_surfaced(report):
    """Statements the parser could not read are outside the instruction
    total, so the only place they can be seen is the unknowns list:
    the count there must equal instruction_classes.unparsed."""
    out = []
    for k in report.get("kernels", []):
        n = k.get("instruction_classes", {}).get("unparsed", 0)
        listed = [u.get("count", 0) for u in k.get("unknowns", [])
                  if u.get("what") == "unparsed statement"]
        if (sum(listed) if listed else 0) != n:
            out.append(
                f"kernel {k.get('name')}: {n} unparsed statement(s), "
                f"unknowns list says {listed}"
            )
    return out


def _verify_instruction_kinds_sum(report):
    """Wherever an aggregate carries numeric instruction counts, the
    per-kind rows sum to the total and each kind's opcodes sum to the
    kind."""
    out = []

    def check(agg, where):
        ins = agg.get("instructions")
        if ins is None:
            return
        try:
            total = int(ins["total"]["expr"])
            parts = sum(int(v["total"]["expr"]) for v in ins["by_kind"].values())
            opcodes = {
                k: (int(v["total"]["expr"]),
                    sum(int(c["expr"]) for c in v["opcodes"].values()))
                for k, v in ins["by_kind"].items()
            }
        except (KeyError, ValueError):
            return
        if total != parts:
            out.append(f"{where}: instruction kinds sum to {parts}, total is {total}")
        for kind, (kind_total, by_opcode) in opcodes.items():
            if kind_total != by_opcode:
                out.append(f"{where}: {kind} opcodes sum to {by_opcode}, kind is {kind_total}")

    def walk(nodes, prefix):
        for n in nodes:
            check(n.get("per_iteration", {}), f"{prefix}/{n.get('name')}")
            walk(n.get("loops", []), f"{prefix}/{n.get('name')}")

    for k in report.get("kernels", []):
        check(k.get("totals", {}), f"{k.get('name')}/totals")
        walk(k.get("loops", []), k.get("name", "?"))
    return out


def _verify_classification_coverage(report):
    """coverage.instructions_classified is derived from, and must agree
    with, the per-kernel class counts."""
    cov = report.get("coverage", {}).get("instructions_classified")
    if cov is None:
        return ["coverage.instructions_classified missing"]
    total = sum(k.get("instruction_classes", {}).get("total", 0)
                for k in report.get("kernels", []))
    unknown = sum(k.get("instruction_classes", {}).get("unknown", 0)
                  for k in report.get("kernels", []))
    out = []
    if cov.get("den") != total:
        out.append(f"classified.den {cov.get('den')} != instruction total {total}")
    if cov.get("num") != total - unknown:
        out.append(f"classified.num {cov.get('num')} != total - unknown {total - unknown}")
    return out


def _verify_trip_coverage(report):
    """coverage.loop_trips_resolved must agree with the loop tree."""
    def walk(nodes):
        resolved = total = 0
        for n in nodes:
            total += 1
            resolved += 1 if "expr" in n.get("trips", {}) else 0
            r, t = walk(n.get("loops", []))
            resolved, total = resolved + r, total + t
        return resolved, total

    resolved = total = 0
    for k in report.get("kernels", []):
        r, t = walk(k.get("loops", []))
        resolved, total = resolved + r, total + t
    cov = report.get("coverage", {}).get("loop_trips_resolved")
    if cov is None:
        return ["coverage.loop_trips_resolved missing"]
    if (cov.get("num"), cov.get("den")) != (resolved, total):
        return [
            f"loop_trips_resolved {cov.get('num')}/{cov.get('den')} does not "
            f"match the loop tree ({resolved}/{total})"
        ]
    return []


def _verify_ai_consistency(report):
    """Wherever AI(global) is printed alongside integer flop/byte
    counts, it must equal their ratio."""
    out = []

    def check(agg, where):
        ai = agg.get("ai_global")
        if ai is None:
            return
        ai = ai["value"]
        try:
            flops = sum(int(agg[t]["total"]["expr"])
                        for t in ("flops", "tensor_flops", "sfu_flops", "atomic_flops") if t in agg)
            by = agg["bytes"]["global"]
            bytes_total = int(by["load"]["expr"]) + int(by["store"]["expr"])
        except (KeyError, ValueError):
            return  # symbolic; AI should not have been emitted, but
            # that is a schema question, not this identity
        if bytes_total > 0 and abs(ai - flops / bytes_total) > 1e-9:
            out.append(f"{where}: ai_global {ai} != {flops}/{bytes_total}")

    def walk(nodes, prefix):
        for n in nodes:
            check(n.get("per_iteration", {}), f"{prefix}/{n.get('name')}")
            walk(n.get("loops", []), f"{prefix}/{n.get('name')}")

    for k in report.get("kernels", []):
        check(k.get("totals", {}), f"{k.get('name')}/totals")
        walk(k.get("loops", []), k.get("name", "?"))
    return out


VERIFIER_CHECKS = [
    ("instruction-accounting", _verify_instruction_accounting),
    ("unparsed-surfaced", _verify_unparsed_surfaced),
    ("classification-coverage", _verify_classification_coverage),
    ("trip-coverage", _verify_trip_coverage),
    ("ai-consistency", _verify_ai_consistency),
    ("instruction-kinds-sum", _verify_instruction_kinds_sum),
]


def verify_report(report):
    violations = []
    for name, fn in VERIFIER_CHECKS:
        violations += [f"verifier '{name}': {v}" for v in fn(report)]
    return violations


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
    return "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("/*"))


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


# --------------------------------------------------------------------------
# Partial JSON comparison (T2)


def subset_match(expected, actual, path="$"):
    """Return a list of mismatch strings; empty list means match."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        errs = []
        for key, val in expected.items():
            if key not in actual:
                errs.append(f"{path}.{key}: missing from actual output")
            else:
                errs += subset_match(val, actual[key], f"{path}.{key}")
        return errs
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected array, got {type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{path}: expected {len(expected)} elements, got {len(actual)}"]
        errs = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            errs += subset_match(e, a, f"{path}[{i}]")
        return errs
    # Scalar. bool is an int subclass in Python; distinguish explicitly so
    # expected `true` never matches actual `1`.
    if isinstance(expected, bool) != isinstance(actual, bool) or expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


# --------------------------------------------------------------------------
# CHECK lines (T2) — ordered substring matching, the semantics of LLVM
# FileCheck's CHECK: directives (without the prefix syntax).


def parse_check_lines(text):
    needles = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        needles.append(line)
    return needles


def match_check_lines(needles, text):
    """Each CHECK line must occur, after the previous line's match."""
    errs = []
    pos = 0
    for needle in needles:
        i = text.find(needle, pos)
        if i >= 0:
            pos = i + len(needle)
        elif text.find(needle) >= 0:
            errs.append(f"CHECK out of order: {needle!r}")
        else:
            errs.append(f"CHECK not found: {needle!r}")
    return errs


# --------------------------------------------------------------------------
# Minimum-coverage thresholds (T3 cross-cutting)


def aggregate_coverage(reports):
    """Sum {"coverage": {metric: {"num": N, "den": D}}} across reports."""
    totals = {}
    for report in reports:
        for metric, frac in report.get("coverage", {}).items():
            num, den = totals.get(metric, (0, 0))
            totals[metric] = (num + frac["num"], den + frac["den"])
    return totals


def check_min_coverage(minimums, totals):
    """Return (violations, raises) where raises maps metric -> new minimum
    percent for entries the corpus now exceeds."""
    violations, raises = [], {}
    for entry in minimums:
        if not entry.get("enforced", False):
            continue
        metric, minimum = entry["metric"], entry["percent"]
        if metric not in totals:
            violations.append(
                f"min_coverage '{metric}' is enforced but no case emitted "
                "coverage for it"
            )
            continue
        num, den = totals[metric]
        if den == 0:
            violations.append(f"min_coverage '{metric}': denominator 0 across corpus")
            continue
        achieved = round(100.0 * num / den, 2)
        if achieved < minimum:
            violations.append(
                f"min_coverage '{metric}': achieved {achieved}% < minimum {minimum}%"
            )
        elif achieved > minimum:
            raises[metric] = achieved
    return violations, raises


def raise_min_text(text, raises):
    """Rewrite `percent = X` lines to the new achieved values. Relies on the
    documented status.toml format contract: within a [[min_coverage]] block
    the `metric` line precedes its `percent` line."""
    out, current_metric = [], None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[[min_coverage]]"):
            current_metric = None
        elif stripped.startswith("metric"):
            current_metric = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("percent") and current_metric in raises:
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}percent = {raises[current_metric]}\n"
        out.append(line)
    return "".join(out)


# --------------------------------------------------------------------------
# Scenario status (T3)


def evaluate_scenario(status, case_passed):
    """Both directions enforced; returns an error string or None."""
    if status == "pass" and not case_passed:
        return "regressed: status is 'pass' but the case failed"
    if status == "xfail" and case_passed:
        return "passes now: flip status to 'pass' in status.toml"
    if status not in ("pass", "xfail"):
        return f"unknown status {status!r} (want 'pass' or 'xfail')"
    return None


# --------------------------------------------------------------------------
# Case execution


def find_case_dir(name):
    for base in (ACCEPTANCE_DIR, CLI_TESTS_DIR):
        d = base / name
        if (d / "case.toml").is_file():
            return d
    return None


def run_case(binary, case_dir):
    """Run one case; return (passed, details, report_json_or_None)."""
    spec = tomllib.loads((case_dir / "case.toml").read_text())
    args = [
        a.replace("{case}", str(case_dir)).replace("{fixtures}", str(FIXTURES_DIR))
        for a in spec["args"]
    ]
    try:
        proc = subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            timeout=CASE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, [f"timed out after {CASE_TIMEOUT_S}s"], None

    details = []
    expected_exit = spec.get("exit", 0)
    if proc.returncode != expected_exit:
        details.append(f"exit code: expected {expected_exit}, got {proc.returncode}")

    expected_json = case_dir / "expected.json"
    report = None
    if expected_json.is_file() or spec.get("json", False):
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            details.append(f"stdout is not valid JSON: {e}")
    if expected_json.is_file() and report is not None:
        details += subset_match(json.loads(expected_json.read_text()), report)

    expected_checks = case_dir / "expected.checks"
    if expected_checks.is_file():
        stream = (
            proc.stderr
            if spec.get("check_stream", "stdout") == "stderr"
            else proc.stdout
        )
        details += match_check_lines(
            parse_check_lines(expected_checks.read_text()), stream
        )

    if report is not None:
        details += verify_report(report)

    return not details, details, report


# --------------------------------------------------------------------------
# Driver


def load_status():
    if not STATUS_FILE.is_file():
        return {}
    return tomllib.loads(STATUS_FILE.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--raise-min", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args()

    if opts.self_test:
        return self_test()

    if not opts.bin.is_file():
        sys.exit(f"run.py: binary {opts.bin} not found — run `cargo build` first")

    status = load_status()
    failures = 0
    reports = []

    # 0: fixture lint
    if FIXTURES_DIR.is_dir():
        for err in lint_fixtures(FIXTURES_DIR):
            print(f"FAIL  lint    {err}")
            failures += 1

    # 1+2: CLI tests (+ report verifier inside run_case)
    case_dirs = (
        sorted(d for d in CLI_TESTS_DIR.iterdir() if (d / "case.toml").is_file())
        if CLI_TESTS_DIR.is_dir()
        else []
    )
    for case_dir in case_dirs:
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        print(f"{'PASS' if passed else 'FAIL'}  cli     {case_dir.name}")
        for d in details:
            print(f"        {d}")
        failures += 0 if passed else 1

    # 3: acceptance scenarios
    for scenario in status.get("scenario", []):
        sid, sc_status = scenario["id"], scenario["status"]
        case_dir = find_case_dir(scenario["case"])
        if case_dir is None:
            print(f"FAIL  {sid:6}  case '{scenario['case']}' not found")
            failures += 1
            continue
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        err = evaluate_scenario(sc_status, passed)
        label = "FAIL" if err else ("XFAIL" if sc_status == "xfail" else "PASS")
        print(f"{label:5} {sid:6}  {scenario['case']}")
        if err:
            print(f"        {err}")
            for d in details:
                print(f"        {d}")
            failures += 1

    # 4: minimum-coverage thresholds
    minimums = status.get("min_coverage", [])
    violations, raises = check_min_coverage(minimums, aggregate_coverage(reports))
    for v in violations:
        print(f"FAIL  mincov  {v}")
        failures += 1
    if raises:
        if opts.raise_min:
            STATUS_FILE.write_text(raise_min_text(STATUS_FILE.read_text(), raises))
            for metric, new in raises.items():
                print(
                    f"RAISED min_coverage '{metric}' -> {new}% "
                    "(commit the status.toml diff)"
                )
        else:
            for metric, new in raises.items():
                print(
                    f"FAIL  mincov  '{metric}' exceeded: achieved {new}% — "
                    "rerun with --raise-min and commit the status.toml diff"
                )
                failures += 1

    # 5: ptxas round trip of every fixture's dump
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

    n_cases = len(case_dirs) + len(status.get("scenario", []))
    print(f"run.py: {n_cases} case(s), {n_roundtrips} round trip(s), {failures} failure(s)")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Self-tests (PR 01): the runner's own logic, no binary needed.


def self_test():
    import tempfile

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

    # partial JSON comparison: the T2 contract
    ok(subset_match({"a": 1}, {"a": 1, "b": 2}) == [], "extra actual field passes")
    ok(subset_match({"a": 1, "b": 2}, {"a": 1}) != [], "missing field fails")
    ok(subset_match({"a": {"b": 3}}, {"a": {"b": 3, "c": 4}}) == [], "nested partial")
    ok(subset_match({"a": 1}, {"a": 2}) != [], "wrong scalar fails")
    ok(subset_match({"a": True}, {"a": 1}) != [], "bool is not int")
    ok(subset_match([1, 2], [1, 2]) == [], "array exact match")
    ok(subset_match([1], [1, 2]) != [], "array length mismatch fails")
    ok(subset_match({"a": [1]}, {"a": {"x": 1}}) != [], "type mismatch fails")

    # CHECK lines
    ok(match_check_lines(["aa", "bb"], "xx aa yy bb zz") == [], "CHECKs in order")
    ok(match_check_lines(["bb", "aa"], "xx aa yy bb zz") != [],
       "CHECKs out of order fail")
    ok(any("out of order" in e for e in match_check_lines(["bb", "aa"], "aa bb")),
       "ordering violation named")
    ok(any("not found" in e for e in match_check_lines(["cc"], "aa bb")),
       "absence named")
    ok(parse_check_lines("# comment\n\nneedle\n") == ["needle"], "comments stripped")

    # scenario status: both directions
    ok(evaluate_scenario("pass", True) is None, "pass+passes ok")
    ok(evaluate_scenario("pass", False) is not None, "pass+fails errors")
    ok(evaluate_scenario("xfail", False) is None, "xfail+fails ok")
    ok(evaluate_scenario("xfail", True) is not None, "xfail+passes errors")
    ok(evaluate_scenario("skip", True) is not None, "unknown status errors")

    # coverage aggregation + minimums + raising
    totals = aggregate_coverage(
        [
            {"coverage": {"m": {"num": 9, "den": 10}}},
            {"coverage": {"m": {"num": 0, "den": 10}}},
            {"other": 1},
        ]
    )
    ok(totals == {"m": (9, 20)}, "coverage sums counts across corpus")
    minimums = [{"metric": "m", "percent": 50.0, "enforced": True}]
    viol, rai = check_min_coverage(minimums, {"m": (9, 20)})
    ok(viol != [] and rai == {}, "below minimum violates")
    viol, rai = check_min_coverage(minimums, {"m": (10, 20)})
    ok(viol == [] and rai == {}, "exactly at minimum passes")
    viol, rai = check_min_coverage(minimums, {"m": (15, 20)})
    ok(viol == [] and rai == {"m": 75.0}, "beaten minimum demands --raise-min")
    viol, rai = check_min_coverage(
        [{"metric": "m", "percent": 99.0, "enforced": False}], {"m": (0, 20)}
    )
    ok(viol == [] and rai == {}, "unenforced minimum ignored")
    viol, _ = check_min_coverage(minimums, {})
    ok(viol != [], "enforced minimum with no data violates")

    status_text = (
        '[[min_coverage]]\nmetric = "m"\npercent = 50.0\nenforced = true\n'
        '[[min_coverage]]\nmetric = "n"\npercent = 10.0\nenforced = true\n'
    )
    new_text = raise_min_text(status_text, {"m": 75.0})
    ok("percent = 75.0" in new_text and "percent = 10.0" in new_text,
       "raise rewrites only the beaten minimum")
    ok(tomllib.loads(new_text)["min_coverage"][0]["percent"] == 75.0,
       "rewritten status file still parses")

    # report-verifier hook plumbing
    VERIFIER_CHECKS.append(
        ("totals", lambda r: [] if r.get("a") == r.get("b") else ["a != b"])
    )
    try:
        totals_only = lambda r: [
            v for v in verify_report(r) if v.startswith("verifier 'totals'")
        ]
        ok(totals_only({"a": 1, "b": 1}) == [], "holding identity is silent")
        ok(totals_only({"a": 1, "b": 2}) == ["verifier 'totals': a != b"],
           "violated identity is named")
    finally:
        VERIFIER_CHECKS.pop()

    # the registered identities: each must hold on a good report and
    # name the violation on a doctored one
    good = {
        "kernels": [
            {
                "name": "k",
                "instruction_classes": {
                    "total": 5, "flop": 1, "non_flop_arith": 1, "memory": 1,
                    "sync": 0, "control": 1, "ignore": 0, "unknown": 1,
                },
                "loops": [
                    {"name": "l", "trips": {"expr": "8"},
                     "per_iteration": {
                         "ai_global": {"value": 0.5, "bound": "exact"},
                         "flops": {"total": {"expr": "8"}},
                         "bytes": {"global": {"load": {"expr": "16"},
                                              "store": {"expr": "0"}}},
                     },
                     "loops": []},
                ],
                "totals": {},
            }
        ],
        "coverage": {
            "instructions_classified": {"num": 4, "den": 5},
            "loop_trips_resolved": {"num": 1, "den": 1},
        },
    }
    ok(verify_report(good) == [], f"all identities hold: {verify_report(good)}")
    import copy

    bad = copy.deepcopy(good)
    bad["kernels"][0]["instruction_classes"]["flop"] = 2
    ok(any("instruction-accounting" in v for v in verify_report(bad)),
       "class-sum violation named")
    bad = copy.deepcopy(good)
    bad["coverage"]["instructions_classified"]["num"] = 5
    ok(any("classification-coverage" in v for v in verify_report(bad)),
       "classified-coverage violation named")
    bad = copy.deepcopy(good)
    bad["kernels"][0]["loops"][0]["trips"] = {"unknown": "reason"}
    ok(any("trip-coverage" in v for v in verify_report(bad)),
       "trip-coverage violation named")
    bad = copy.deepcopy(good)
    bad["kernels"][0]["loops"][0]["per_iteration"]["ai_global"]["value"] = 2.0
    ok(any("ai-consistency" in v for v in verify_report(bad)),
       "AI-ratio violation named")
    bad = copy.deepcopy(good)
    bad["kernels"][0]["totals"] = {"instructions": {
        "total": {"expr": "5"},
        "by_kind": {"control": {"total": {"expr": "1"}, "opcodes": {"bra": {"expr": "1"}}}}}}
    ok(any("instruction-kinds-sum" in v for v in verify_report(bad)),
       "instruction kinds not summing to the total is named")
    bad = copy.deepcopy(good)
    bad["kernels"][0]["totals"] = {"instructions": {
        "total": {"expr": "1"},
        "by_kind": {"control": {"total": {"expr": "1"}, "opcodes": {"bra": {"expr": "2"}}}}}}
    ok(any("opcodes sum" in v for v in verify_report(bad)),
       "opcodes not summing to their kind is named")
    bad = copy.deepcopy(good)
    bad["kernels"][0]["instruction_classes"]["unparsed"] = 1
    ok(any("unparsed-surfaced" in v for v in verify_report(bad)),
       "unparsed statement missing from unknowns is named")
    bad["kernels"][0]["unknowns"] = [{"what": "unparsed statement", "count": 1}]
    ok(verify_report(bad) == [], "unparsed statement listed in unknowns passes")

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
