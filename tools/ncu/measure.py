#!/usr/bin/env python3
"""Measure one kernel launch per PTX line with Nsight Compute.

RECORD names the launch: ptx (repository-relative), kernel, grid,
block, the dynamic shared bytes (shared, 0 when absent), and params in
launch.c's u32:/i32:/f32:/ptr: forms. Every instruction
gets a .loc naming its own line of the PTX file, which must assemble to
the original's SASS. The record is rewritten with the GPU and tools,
and per PTX line that has SASS: the line's text, the SASS opcodes ptxas
attributed to it, and their predicated-on thread instructions executed;
on a line with a global memory instruction also the L2 sectors they
request, the ideal number, each such instruction's access size in
bits, and their predicated-on thread instructions executed.
Lines of a loop that branches back on an mbarrier test_wait or try_wait
result are marked timing_dependent. With --check the record is measured
again on the GPU it names, if this is that GPU, and must not change but
for those lines' executions; on another GPU it is skipped, or with
PTXROOF_REQUIRE_GPU=1 fails. Either way it prints one line.

Usage: measure.py [--check] RECORD
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run(*cmd):
    done = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"measure.py: {Path(str(cmd[0])).name} failed:\n{done.stderr or done.stdout}")
    return done.stdout


def numbered(ptx, path):
    """A statement ends with a semicolon (PTX ISA 4.3), so a line after one
    ending in a comma, or after `.branchtargets`, continues a statement."""
    out, continued = [], False
    for n, line in enumerate(ptx.split("\n"), 1):
        code = line.split("//")[0].strip()
        if re.match(r"\.(loc|file)\b", code):
            line = ""
        elif code.startswith(".address_size"):
            line += f' .file 1 "{path}"'
        elif (not continued and code.endswith((";", "}")) and not code.startswith((".", "}"))
              and not re.match(r"[\w$]+:", code)):
            line = f".loc 1 {n} 0 {line}"
        if code:
            continued = code.endswith(",") or code == ".branchtargets"
        out.append(line)
    return "\n".join(out)


def timing_dependent(ptx):
    """Lines of the loops that branch back on an mbarrier test_wait or try_wait result. It says
    whether the phase has completed, or for try_wait whether a system-dependent time limit passed
    first (PTX ISA 9.7.14.16.19), so how often such a loop runs depends on timing."""
    lines, seen, out = [l.split("//")[0].strip() for l in ptx.split("\n")], {}, set()
    for n, code in enumerate(lines, 1):
        if label := re.match(r"([\w$]+):", code):
            seen[label[1]] = n
        elif (bra := re.match(r"@!?(\w+)\s+bra(?:\.uni)?\s+([\w$]+);", code)) and bra[2] in seen:
            span = range(seen[bra[2]], n + 1)
            wait = rf"mbarrier\.(test|try)_wait\S*\s+{bra[1]}\b"
            if any(re.match(wait, lines[k - 1]) for k in span):
                out.update(span)
    return out


def sass(cubin):
    return [l for l in run("cuobjdump", "-sass", cubin).splitlines() if not l.lstrip().startswith("/* 0x")]


def per_line(table):
    """Nsight Compute's memory_l2_theoretical_sectors_global is the "theoretical number of sectors
    requested in L2 from global memory instructions", its _ideal "assuming each not predicated-off
    thread performed the operation" (Profiling Guide, Source Counters)."""
    rows = list(csv.reader(table.splitlines()))
    head = next(i for i, r in enumerate(rows) if r[:1] == ["Line No"])
    column = {c: i for i, c in enumerate(rows[head])}
    executed = column["Predicated-On Thread Instructions Executed"]
    in_global = lambda r: "Address Space" in column and "Global" in re.findall(r"\w+", r[column["Address Space"]])
    lines = {}
    for r in rows[head + 1 :]:
        if r[0]:
            line = lines[r[0]] = {"ptx": " ".join(r[1].split()), "sass": {}, "executions": int(r[executed])}
            if in_global(r):
                line["l2_theoretical_sectors_global"] = int(r[column["L2 Theoretical Sectors Global"]])
                line["l2_theoretical_sectors_global_ideal"] = int(r[column["L2 Theoretical Sectors Global Ideal"]])
                line["global_access_sizes"] = []
                line["global_executions"] = 0
        elif r[2].startswith("0x"):
            op = re.sub(r"^@!?U?P\w+\s+", "", r[3].strip()).split()[0]
            line["sass"][op] = line["sass"].get(op, 0) + 1
            if in_global(r):
                line["global_access_sizes"].append(int(r[column["Access Size"]]))
                line["global_executions"] += int(r[executed])
    return lines


def main(path, check):
    record = json.loads(Path(path).read_text())
    ptx = REPO / record["ptx"]
    driver = run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        include = Path(shutil.which("ptxas")).resolve().parents[1] / "include"
        run("gcc", "-O2", Path(__file__).with_name("launch.c"), f"-I{include}", "-lcuda", "-o", tmp / "launch")
        gpu, arch = run(tmp / "launch", "--device").splitlines()
        if check and record.get("gpu") != gpu:
            skipped = f"measure.py: {path} was recorded on {record.get('gpu')}, not {gpu}"
            if os.environ.get("PTXROOF_REQUIRE_GPU") == "1":
                sys.exit(f"{skipped}, and PTXROOF_REQUIRE_GPU=1")
            print(f"{skipped}: skipped")
            return
        (tmp / "numbered.ptx").write_text(numbered(ptx.read_text(), ptx))
        for name, source in (("original", ptx), ("numbered", tmp / "numbered.ptx")):
            run("ptxas", f"-arch={arch}", "-lineinfo", source, "-o", tmp / f"{name}.cubin")
        if sass(tmp / "original.cubin") != sass(tmp / "numbered.cubin"):
            sys.exit(f"measure.py: {record['ptx']} assembles to different SASS with a .loc per line")
        dims = [",".join(map(str, record[k])) for k in ("grid", "block")]
        launch = [tmp / "launch", tmp / "numbered.cubin", record["kernel"], *dims, record.get("shared", 0)]
        launch += record["params"]
        run("ncu", "--section", "SourceCounters", "--import-source", "yes", "-o", tmp / "report", *launch)
        report = next(tmp.glob("report.*"))
        table = run("ncu", "--import", report, "--page", "source", "--print-source", "cuda,sass", "--csv")
    ptxas = next(l for l in run("ptxas", "--version").splitlines() if "release" in l)
    ncu = run("ncu", "--version").splitlines()[-1]
    lines = per_line(table)
    stored = json.loads(Path(path).read_text()).get("lines", {}) if check else {}
    for n in timing_dependent(ptx.read_text()) & set(map(int, lines)):
        lines[str(n)]["timing_dependent"] = True
        if str(n) in stored:
            lines[str(n)]["executions"] = stored[str(n)]["executions"]
    record.update(gpu=gpu, driver=driver, arch=arch, ptxas=ptxas, ncu=ncu, lines=lines)
    entry = lambda k, v, indent: f"{indent}{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
    fields = ",\n".join(entry(k, v, " ") for k, v in record.items() if k != "lines")
    lines = ",\n".join(entry(k, v, "  ") for k, v in record["lines"].items())
    text = "{\n" + fields + ',\n "lines": {\n' + lines + "\n }\n}\n"
    if check and text != Path(path).read_text():
        sys.exit(f"measure.py: {path} differs from a fresh measurement; rerun without --check, review the diff")
    Path(path).write_text(text)
    if check:
        print(f"measure.py: {path} measured again on {gpu}: unchanged")


if __name__ == "__main__":
    args = sys.argv[1:]
    check = args[:1] == ["--check"]
    if len(args) != 1 + check:
        sys.exit(__doc__)
    main(args[-1], check)
