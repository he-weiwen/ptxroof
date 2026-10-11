"""SASS as cuobjdump and Nsight Compute print it, reduced to one text so a
report's kernel can be matched to the cubin it ran."""

import hashlib
import re
import subprocess


def normalize(lines, base=None):
    """Drops encodings, register reuse flags and branch-target labels, and the
    comma Nsight Compute prints where cuobjdump prints a space. With base, the
    report's first instruction address, absolute branch targets become offsets."""
    out = []
    for line in lines:
        line = re.sub(r";?\s*/\*.*?\*/", "", line).replace(".reuse", "")
        line = re.sub(r"`\([^)]*\)\s*", "", line)
        if base is not None:
            line = re.sub(r"0x[0-9a-f]{9,}", lambda m: hex(int(m[0], 16) - base), line)
        out.append(re.sub(r"[,\s]+", " ", line).strip().rstrip(";").strip())
    return out


def of_cubin(path, cuobjdump="cuobjdump"):
    text = subprocess.run([str(cuobjdump), "-sass", str(path)], capture_output=True, text=True, check=True).stdout
    instr = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s*")
    return normalize(instr.sub("", l) for l in text.splitlines() if instr.match(l))


def digest(sass):
    return hashlib.sha256("\n".join(sass).encode()).hexdigest()
