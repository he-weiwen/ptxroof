#!/usr/bin/env python3
"""Check one ptxroof JSON report, read from stdin, against a lit test.

The test's expected JSON starts at its first line that begins with "{".
Partial comparison:
  - objects: every expected key must exist in actual and match; EXTRA
    actual keys are ignored (new report fields never break old tests).
  - arrays: same length, element-wise match (arrays are ordered data;
    partial comparison applies to object keys only).
  - scalars: exact equality, including floats — analysis is exact
    arithmetic on static counts, so expected outputs store exact values.

Every report must also satisfy the identities below, which any correct
implementation satisfies, with no per-test expectations.

Usage: check_report.py TEST < REPORT
       check_report.py --self-test
"""

import copy
import json
import sys
from pathlib import Path


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


def self_test():
    n = 0

    def ok(cond, what):
        nonlocal n
        n += 1
        assert cond, f"self-test failed: {what}"

    # partial JSON comparison
    ok(subset_match({"a": 1}, {"a": 1, "b": 2}) == [], "extra actual field passes")
    ok(subset_match({"a": 1, "b": 2}, {"a": 1}) != [], "missing field fails")
    ok(subset_match({"a": {"b": 3}}, {"a": {"b": 3, "c": 4}}) == [], "nested partial")
    ok(subset_match({"a": 1}, {"a": 2}) != [], "wrong scalar fails")
    ok(subset_match({"a": True}, {"a": 1}) != [], "bool is not int")
    ok(subset_match([1, 2], [1, 2]) == [], "array exact match")
    ok(subset_match([1], [1, 2]) != [], "array length mismatch fails")
    ok(subset_match({"a": [1]}, {"a": {"x": 1}}) != [], "type mismatch fails")

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
    print(f"check_report.py --self-test: {n} assertions passed")


def main(test):
    text = Path(test).read_text().split("\n")
    first = next(i for i, line in enumerate(text) if line.startswith("{"))
    expected = json.loads("\n".join(text[first:]))
    try:
        report = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        sys.exit(f"stdout is not valid JSON: {e}")
    errors = subset_match(expected, report) + verify_report(report)
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
