#!/usr/bin/env python3
"""Measure one kernel launch per PTX line with Nsight Compute.

RECORD names the launch: ptx (repository-relative), kernel, grid,
block, and params in launch.c's u32:/f32:/ptr: forms. Every instruction
gets a .loc naming its own line of the PTX file, which must assemble to
the original's SASS. The record is rewritten with the GPU and tools,
and per PTX line that has SASS: the line's text, the SASS opcodes ptxas
attributed to it, and their predicated-on thread instructions executed.

Usage: measure.py RECORD
"""

import csv
import json
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
    out = []
    for n, line in enumerate(ptx.split("\n"), 1):
        code = line.split("//")[0].strip()
        if code.startswith((".loc", ".file")):
            line = ""
        elif code.startswith(".address_size"):
            line += f' .file 1 "{path}"'
        elif code.endswith((";", "}")) and not code.startswith((".", "}")) and not re.match(r"[\w$]+:", code):
            line = f".loc 1 {n} 0 {line}"
        out.append(line)
    return "\n".join(out)


def sass(cubin):
    return [l for l in run("cuobjdump", "-sass", cubin).splitlines() if not l.lstrip().startswith("/* 0x")]


def per_line(table):
    rows = list(csv.reader(table.splitlines()))
    head = next(i for i, r in enumerate(rows) if r[:1] == ["Line No"])
    executed = rows[head].index("Predicated-On Thread Instructions Executed")
    lines = {}
    for r in rows[head + 1 :]:
        if r[0]:
            line = lines[r[0]] = {"ptx": " ".join(r[1].split()), "sass": {}, "executions": int(r[executed])}
        elif r[2].startswith("0x"):
            op = re.sub(r"^@!?U?P\w+\s+", "", r[3].strip()).split()[0]
            line["sass"][op] = line["sass"].get(op, 0) + 1
    return lines


def main(path):
    record = json.loads(Path(path).read_text())
    ptx = REPO / record["ptx"]
    query = "--query-gpu=name,driver_version,compute_cap"
    gpu, driver, cc = run("nvidia-smi", query, "--format=csv,noheader").splitlines()[0].split(", ")
    arch = "sm_" + cc.replace(".", "")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "numbered.ptx").write_text(numbered(ptx.read_text(), ptx))
        for name, source in (("original", ptx), ("numbered", tmp / "numbered.ptx")):
            run("ptxas", f"-arch={arch}", "-lineinfo", source, "-o", tmp / f"{name}.cubin")
        if sass(tmp / "original.cubin") != sass(tmp / "numbered.cubin"):
            sys.exit(f"measure.py: {record['ptx']} assembles to different SASS with a .loc per line")
        include = Path(shutil.which("ptxas")).resolve().parents[1] / "include"
        run("gcc", "-O2", Path(__file__).with_name("launch.c"), f"-I{include}", "-lcuda", "-o", tmp / "launch")
        dims = [",".join(map(str, record[k])) for k in ("grid", "block")]
        launch = [tmp / "launch", tmp / "numbered.cubin", record["kernel"], *dims, *record["params"]]
        run("ncu", "--section", "SourceCounters", "--import-source", "yes", "-o", tmp / "report", *launch)
        report = next(tmp.glob("report.*"))
        table = run("ncu", "--import", report, "--page", "source", "--print-source", "cuda,sass", "--csv")
    ptxas = next(l for l in run("ptxas", "--version").splitlines() if "release" in l)
    ncu = run("ncu", "--version").splitlines()[-1]
    record.update(gpu=gpu, driver=driver, arch=arch, ptxas=ptxas, ncu=ncu, lines=per_line(table))
    entry = lambda k, v, indent: f"{indent}{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
    fields = ",\n".join(entry(k, v, " ") for k, v in record.items() if k != "lines")
    lines = ",\n".join(entry(k, v, "  ") for k, v in record["lines"].items())
    Path(path).write_text("{\n" + fields + ',\n "lines": {\n' + lines + "\n }\n}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
