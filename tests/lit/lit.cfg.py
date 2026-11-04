import glob
import os
import re
import shlex
import shutil
import subprocess

import lit.formats

config.name = "ptxroof"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)

repo = os.path.dirname(os.path.dirname(config.test_source_root))
config.test_exec_root = os.path.join(repo, "target", "lit")


def llvm_tools():
    """The directory of FileCheck and `not`: $LLVM_TOOLS_DIR; the LLVM whose
    lit this is (Ubuntu's llvm-N-tools); PATH; Homebrew's keg-only llvm; or
    the newest /usr/lib/llvm-N."""
    beside_lit = os.path.join(os.path.dirname(lit.__file__), "..", "..", "..", "..", "bin")
    on_path = shutil.which("FileCheck")
    ubuntu = glob.glob("/usr/lib/llvm-*/bin")
    ubuntu.sort(key=lambda d: int(re.search(r"llvm-(\d+)", d).group(1)), reverse=True)
    candidates = [
        os.environ.get("LLVM_TOOLS_DIR"),
        beside_lit,
        on_path and os.path.dirname(on_path),
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
        *ubuntu,
    ]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "FileCheck")):
            return os.path.realpath(d)
    lit_config.fatal(
        "no FileCheck: install LLVM's tools (brew install llvm, apt install llvm-18-tools)"
        " or set LLVM_TOOLS_DIR"
    )


config.environment["PATH"] = os.pathsep.join([llvm_tools(), os.environ["PATH"]])
config.environment["FILECHECK_OPTS"] = "--strict-whitespace"
config.ptxroof = os.environ.get("PTXROOF", os.path.join(repo, "target", "debug", "ptxroof"))
config.substitutions.append(("%ptxroof", shlex.quote(config.ptxroof)))
config.substitutions.append(("%fixtures", shlex.quote(os.path.join(repo, "tests", "fixtures"))))
if os.path.exists("/dev/full"):
    config.available_features.add("dev-full")
if shutil.which("ptxas") and shutil.which("cuobjdump"):
    release = subprocess.run(["ptxas", "--version"], capture_output=True, text=True).stdout
    config.available_features.update(f"ptxas-{r}" for r in re.findall(r"release (\S+),", release))
