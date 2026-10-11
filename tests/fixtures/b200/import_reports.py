#!/usr/bin/env python3
"""Writes a fixture for each kernel launch in sol-study's B200 Nsight Compute
reports: its PTX with a provenance header, and a tools/ncu/measure.py record of
the report's predicated-on thread instructions executed per PTX line.

A report's kernel is matched to the PTX it ran by SASS and block size among
the study's Triton caches, and to its launch by SASS and grid among the
launches record_launches.py recorded. The PTX must assemble to the report's
SASS with Triton's ptxas command, also with a .loc per line (measure.py),
whose line table attributes each SASS instruction to a PTX line.

Usage: import_reports.py SOL_STUDY CORPUS LAUNCHES OUT_DIR
       import_reports.py --check RECORD...
--check: each record's PTX still assembles to its recorded SASS.
"""

import csv
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "ncu"))
import sass  # noqa: E402
from measure import numbered, timing_dependent  # noqa: E402

RUN = "results/b200_ncu_triton"


def run(*cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stdout


def table(text):
    return list(csv.reader(io.StringIO(text)))


def report_kernels(rep):
    raw = table(run("ncu", "--import", rep, "--page", "raw", "--csv"))
    col = {name: raw[0].index(name) for name in ("ID", "Kernel Name", "Grid Size", "Block Size")}
    source = run("ncu", "--import", rep, "--page", "source", "--csv", "--print-source", "sass")
    for row, block in zip(raw[2:], re.split(r'^"Kernel Name",', source, flags=re.M)[1:]):
        name = row[col["Kernel Name"]]
        head, body = block.split("\n", 1)
        assert head.startswith(f'"{name}"'), (rep, name, head)
        rows = table(body)
        executed = rows[0].index("Predicated-On Thread Instructions Executed")
        instrs = [r for r in rows[1:] if r and r[0].startswith("0x")]
        yield {
            "id": row[col["ID"]],
            "kernel": name,
            "grid": [int(n) for n in re.findall(r"\d+", row[col["Grid Size"]])],
            "block": [int(n) for n in re.findall(r"\d+", row[col["Block Size"]])],
            "sass": sass.normalize([r[1] for r in instrs], int(instrs[0][0], 16)),
            "executed": [int(r[executed]) for r in instrs],
        }


def session(rep):
    rows = dict(r[:2] for r in table(run("ncu", "--import", rep, "--page", "session", "--csv")) if len(r) >= 2)
    return rows["Created"], rows["Nsight Compute Target"].split()[0]


def tool_note(cubin):
    """The ptxas name, release and command line in the cubin's .note.nv.tkinfo."""
    strings = re.findall(r"\]\s+(.*)", run("readelf", "-p", ".note.nv.tkinfo", cubin))
    tool = next(s for s in strings if s.startswith("ptxas"))
    release = next(s for s in strings if "release" in s).split(", ")[-1]
    return tool, release, strings[-1].split()


def assemble(ptxas, flags, ptx, cubin):
    run(ptxas, *flags, ptx, "-o", cubin)
    return sass.of_cubin(cubin)


def line_of_each_instruction(nvdisasm, cubin):
    lines, current = [], None
    for text in run(nvdisasm, "-g", "-c", cubin).splitlines():
        if m := re.search(r'//## File ".*", line (\d+)', text):
            current = int(m[1])
        elif re.match(r"\s+/\*[0-9a-f]{4,}\*/", text):
            lines.append(current)
    return lines


def write_record(path, record):
    entry = lambda k, v, indent: f"{indent}{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
    fields = ",\n".join(entry(k, v, " ") for k, v in record.items() if k != "lines")
    lines = ",\n".join(entry(k, v, "  ") for k, v in record["lines"].items())
    path.write_text("{\n" + fields + ',\n "lines": {\n' + lines + "\n }\n}\n")


def import_reports(sol, corpus, launches_path, out):
    bin_dir = sol / "vendor/triton/backends/nvidia/bin"
    sol_commit = run("git", "-C", sol, "describe", "--always", "--dirty", "--abbrev=12").strip()
    triton_commit = (sol / "vendor/TRITON_COMMIT").read_text().strip()[:12]
    manifest = {}
    for line in (corpus / "MANIFEST.jsonl").read_text().splitlines():
        row = json.loads(line)
        manifest[row["output_file"]] = row
    caches = {}
    for cubin in sorted(sol.glob("results/*/triton_cache/*/*.cubin")):
        caches.setdefault(sass.digest(sass.of_cubin(cubin)), []).append(cubin)
    launches = [json.loads(line) for line in launches_path.read_text().splitlines()]
    cases = json.loads((sol / RUN / "ncu_summary.json").read_text())
    for case in cases:
        rep = sol / RUN / "ncu" / (case.replace("/", "__") + ".ncu-rep")
        created, ncu = session(rep)
        for k in report_kernels(rep):
            digest = sass.digest(k["sass"])
            threads = k["block"][0] * k["block"][1] * k["block"][2]
            own_run_first = sorted(caches.get(digest, []), key=lambda c: RUN not in str(c))
            cubin = next(c for c in own_run_first
                         if re.search(rf"\.reqntid {threads}\b", next(c.parent.glob("*.ptx")).read_text()))
            params = {tuple(l["params"]) for l in launches
                      if l["case"] == case and l["sass"] == digest and l["grid"] == k["grid"]}
            assert len(params) == 1, (case, k["kernel"], len(params))
            shared = {l["shared"] for l in launches if l["case"] == case and l["sass"] == digest}.pop()
            tool, release, note_flags = tool_note(cubin)
            # Triton's make_cubin also passes --regAllocOptLevel=2, which the
            # cubin's note omits (vendor/triton/backends/nvidia/compiler.py).
            flags = [f for f in note_flags if f != "-v"] + ["--regAllocOptLevel=2"]
            original = next(cubin.parent.glob("*.ptx")).read_text()
            source = re.search(r'\.file\s+1\s+"/corpus/([^"]+)"', original)[1]
            row = manifest[source]
            name = f"{case.replace('/', '__')}.{k['kernel']}.sm_100a"
            header = [
                "//",
                "// Provenance: generated by regen.sh in this directory — do not edit.",
                f"//   source:   {row['origin_url']} ({row['license']})",
                f"//   launch:   sol-study {sol_commit} {case}, kernel {k['id']} of {rep.relative_to(sol)}",
                f"//   compiler: triton {json.loads(next(p for p in cubin.parent.glob('*.json') if not p.name.startswith('__grp')).read_text())['triton_version']} "
                f"({triton_commit}), {tool} {release} {' '.join(flags)}",
                f"//   measured: Nsight Compute {ncu}, NVIDIA B200, {created}",
                "//",
            ]
            ptx = out / f"{name}.ptx"
            ptx.write_text("\n".join(header) + "\n" + original)
            text = ptx.read_text()
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                (tmp / "numbered.ptx").write_text(numbered(text, ptx))
                for source_ptx in (ptx, tmp / "numbered.ptx"):
                    got = assemble(bin_dir / tool, flags, source_ptx, tmp / "out.cubin")
                    assert got == k["sass"], f"{source_ptx.name} of {name} assembles to other SASS than the report's"
                attributed = line_of_each_instruction(bin_dir / "nvdisasm", tmp / "out.cubin")
            lines, ptx_lines = {}, text.split("\n")
            for n, instr, executed in zip(attributed, k["sass"], k["executed"], strict=True):
                if n is None:
                    continue
                line = lines.setdefault(str(n), {"ptx": " ".join(ptx_lines[n - 1].split()), "sass": {}, "executions": 0})
                op = re.sub(r"^@!?U?P\w+\s+", "", instr).split()[0]
                line["sass"][op] = line["sass"].get(op, 0) + 1
                line["executions"] += executed
            for n in timing_dependent(text):
                if str(n) in lines:
                    lines[str(n)]["timing_dependent"] = True
            write_record(out / f"{name}.ncu.json", {
                "ptx": str(ptx.resolve().relative_to(REPO)), "kernel": k["kernel"], "grid": k["grid"],
                "block": k["block"], "shared": shared, "params": list(params.pop()), "gpu": "NVIDIA B200",
                "arch": "sm_100a", "ptxas": f"{tool} {release}", "ptxas_flags": flags, "ncu": ncu,
                "sass_sha256": digest, "lines": dict(sorted(lines.items(), key=lambda kv: int(kv[0]))),
            })
            print(name)


def check(records):
    with tempfile.TemporaryDirectory() as tmp:
        for path in records:
            record = json.loads(Path(path).read_text())
            got = sass.digest(assemble("ptxas", record["ptxas_flags"], REPO / record["ptx"], Path(tmp) / "out.cubin"))
            if got != record["sass_sha256"]:
                sys.exit(f"{record['ptx']} assembles to other SASS than {record['ptxas']} did")
    print(f"{len(records)} PTX files assemble to their recorded SASS")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--check"] and len(args) > 1:
        check(args[1:])
    elif len(args) == 4:
        import_reports(*map(Path, args))
    else:
        sys.exit(__doc__)
