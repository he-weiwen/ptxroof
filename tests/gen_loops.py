#!/usr/bin/env python3
"""Generated single-loop kernels with trip counts known by simulation.

Each kernel is one counted loop drawn from a grammar of shapes: up- or
down-counting by a constant step, initialised from a constant or the
parameter, compared (lt/le/gt/ge/ne, either operand order, either
branch polarity) against a constant, the parameter, or the parameter
plus a constant, read after the increment, before it through a copy,
or through LLVM's two-register form. A Python interpreter of the same
loop gives the trip count for each binding of the parameter; the tool's
symbolic count, bound with --bind, must equal it wherever the loop
actually iterates. A shape the tool refuses is counted as unsupported,
never as a failure: the property is that it is never wrong.

Usage: gen_loops.py [--seed N] [--count N] [--bin PATH] [--keep DIR]
"""

import argparse
import json
import operator
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / "target" / "debug" / "ptxroof"
BINDINGS = [2, 3, 7, 16, 64, 1000]
SIM_CAP = 20_000

OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "ne": operator.ne}


def draw(rng):
    """One loop shape as a dict of the choices below."""
    return {
        "step": rng.choice([1, 2, 3, 4, 8]) * rng.choice([1, -1]),
        "init": rng.choice(["0", "1", "5", "param"]),
        "bound": rng.choice(["param", "param+c", "const"]),
        "bound_c": rng.choice([1, 3, 8, 64]),
        "op": rng.choice(list(OPS)),
        "counter_left": rng.choice([True, False]),
        "negated": rng.choice([True, False]),
        "read": rng.choice(["post", "pre", "two-register"]),
    }


def emit(shape):
    step, op = shape["step"], shape["op"]
    init = "%r1" if shape["init"] == "param" else shape["init"]
    body = [
        "ld.param.u32 %r1, [gen_param_0];",
        f"mov.u32 %r2, {init};",
    ]
    if shape["bound"] == "param":
        body.append("mov.u32 %r3, %r1;")
    elif shape["bound"] == "param+c":
        body.append(f"add.s32 %r3, %r1, {shape['bound_c']};")
    else:
        body.append(f"mov.u32 %r3, {shape['bound_c']};")
    body.append("$L__L:")
    if shape["read"] == "post":
        body += [f"add.s32 %r2, %r2, {step};"]
        seen = "%r2"
    elif shape["read"] == "pre":
        body += ["mov.u32 %r5, %r2;", f"add.s32 %r2, %r2, {step};"]
        seen = "%r5"
    else:
        body += [f"add.s32 %r6, %r2, {step};"]
        seen = "%r6"
    lhs, rhs = (seen, "%r3") if shape["counter_left"] else ("%r3", seen)
    body.append(f"setp.{op}.s32 %p1, {lhs}, {rhs};")
    if shape["read"] == "two-register":
        body.append("mov.u32 %r2, %r6;")
    body.append(f"@{'!' if shape['negated'] else ''}%p1 bra $L__L;")
    body.append("ret;")
    return (
        ".version 8.7\n.target sm_80\n.address_size 64\n"
        ".visible .entry gen(\n\t.param .u32 gen_param_0\n)\n{\n"
        "\t.reg .pred %p<3>;\n\t.reg .b32 %r<8>;\n"
        + "".join(("" if l.endswith(":") else "\t") + l + "\n" for l in body)
        + "}\n"
    )


def simulate(shape, n):
    """Trips of the loop for parameter value n, or None past the cap."""
    step = shape["step"]
    i = n if shape["init"] == "param" else int(shape["init"])
    bound = {"param": n, "param+c": n + shape["bound_c"], "const": shape["bound_c"]}[shape["bound"]]
    cmp = OPS[shape["op"]]
    trips = 0
    while trips < SIM_CAP:
        trips += 1
        seen = i if shape["read"] == "pre" else i + step
        i += step
        a, b = (seen, bound) if shape["counter_left"] else (bound, seen)
        taken = cmp(a, b)
        if shape["negated"]:
            taken = not taken
        if not taken:
            return trips
    return None


def tool_trips(binary, ptx_path, n):
    """The tool's bound trip count as an int, or the unknown's reason."""
    out = subprocess.run(
        [str(binary), "analyze", "--json", "--bind", f"0:n={n}", str(ptx_path)],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return f"exit {out.returncode}: {out.stderr.strip()[:200]}"
    report = json.loads(out.stdout)
    loops = report["kernels"][0]["loops"]
    if len(loops) != 1:
        return f"{len(loops)} loops found"
    trips = loops[0]["trips"]
    if "unknown" in trips:
        return trips["unknown"]
    try:
        return int(trips["expr"])
    except ValueError:
        return f"symbolic after binding: {trips['expr']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--keep", type=Path, help="write every generated kernel here")
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args()
    if opts.self_test:
        return self_test()
    if not opts.bin.is_file():
        sys.exit(f"gen_loops: binary {opts.bin} not found — run `cargo build` first")
    rng = random.Random(opts.seed)
    compared = mismatches = 0
    unsupported = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(opts.count):
            shape = draw(rng)
            ptx = emit(shape)
            path = (opts.keep or Path(tmp)) / f"gen_{opts.seed}_{k}.ptx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ptx)
            # Only bindings where the loop really iterates: the tool assumes
            # a guarded, entered loop, and the guard is the kernel's business.
            cases = [(n, simulate(shape, n)) for n in BINDINGS]
            cases = [(n, t) for n, t in cases if t is not None and t >= 2]
            for n, expected in cases:
                got = tool_trips(opts.bin, path, n)
                if isinstance(got, int):
                    compared += 1
                    if got != expected:
                        mismatches += 1
                        print(f"MISMATCH seed {opts.seed} kernel {k} n={n}: tool {got}, simulated {expected}")
                        print(f"  shape {shape}")
                        print("  " + ptx.replace("\n", "\n  "))
                else:
                    unsupported[got] = unsupported.get(got, 0) + 1
    for reason, count in sorted(unsupported.items(), key=lambda kv: -kv[1]):
        print(f"unsupported x{count}: {reason}")
    print(
        f"gen_loops: {opts.count} kernels, {compared} bindings compared, "
        f"{sum(unsupported.values())} unsupported, {mismatches} mismatch(es)"
    )
    return 1 if mismatches else 0


def self_test():
    """The simulator on shapes with a known closed form."""
    base = {"bound_c": 8, "counter_left": True, "negated": False}
    up = {**base, "step": 1, "init": "0", "bound": "param", "op": "lt", "read": "post"}
    assert [simulate(up, n) for n in (1, 7, 64)] == [1, 7, 64]
    pre = {**up, "read": "pre"}
    assert simulate(pre, 7) == 8  # compares the previous value: one more trip
    down = {**base, "step": -3, "init": "param", "bound": "const", "op": "gt", "read": "two-register"}
    assert simulate(down, 20) == 4  # 17, 14, 11, 8: continue while > 8
    flipped = {**up, "counter_left": False, "op": "gt"}  # bound > counter
    assert simulate(flipped, 7) == 7
    inverted = {**up, "op": "ge", "negated": True}  # continue while not (i >= n)
    assert simulate(inverted, 7) == 7
    print("gen_loops --self-test: 5 assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
