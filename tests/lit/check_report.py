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

Usage: check_report.py TEST < REPORT
       check_report.py --self-test
"""

import json
import sys
from pathlib import Path


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

    print(f"check_report.py --self-test: {n} assertions passed")


def main(test):
    text = Path(test).read_text().split("\n")
    first = next(i for i, line in enumerate(text) if line.startswith("{"))
    expected = json.loads("\n".join(text[first:]))
    try:
        report = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        sys.exit(f"stdout is not valid JSON: {e}")
    errors = subset_match(expected, report)
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
