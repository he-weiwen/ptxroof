"""One lit test per fixture: its --dump-ast must assemble to the
fixture's own SASS. Fixtures in the allowlist are expected to fail."""
import glob
import os
import subprocess
import tempfile

import lit.formats
import lit.Test


class RoundTrip(lit.formats.TestFormat):
    def __init__(self, ptxroof, fixtures, allowlist):
        self.ptxroof = ptxroof
        self.fixtures = fixtures
        with open(allowlist) as f:
            self.allowlist = {l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")}

    def getTestsInDirectory(self, suite, path_in_suite, lit_config, local_config):
        for ptx in sorted(glob.glob(os.path.join(self.fixtures, "**", "*.ptx"), recursive=True)):
            rel = os.path.relpath(ptx, self.fixtures)
            test = lit.Test.Test(suite, path_in_suite + tuple(rel.split(os.sep)), local_config, file_path=ptx)
            test.xfails = ["*"] if rel in self.allowlist else []
            yield test

    def execute(self, test, lit_config):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "needs ptxas and cuobjdump")
        ptx = test.getFilePath()
        with open(ptx) as f:
            arch = next((l.split()[1] for l in f if l.startswith(".target")), None)
        if arch is None:
            return lit.Test.Result(lit.Test.FAIL, "no .target line")
        env = test.config.environment
        run = lambda *cmd: subprocess.run(cmd, capture_output=True, text=True, env=env)
        with tempfile.TemporaryDirectory() as tmp:
            orig, dump, redo = (os.path.join(tmp, n) for n in ("orig.cubin", "dump.ptx", "dump.cubin"))
            r = run("ptxas", f"-arch={arch}", ptx, "-o", orig)
            if "Unsupported .version" in r.stderr:
                return lit.Test.Result(lit.Test.UNSUPPORTED, r.stderr)
            if r.returncode:
                return lit.Test.Result(lit.Test.FAIL, f"the fixture itself does not assemble:\n{r.stderr}")
            r = run(self.ptxroof, "analyze", "--dump-ast", ptx)
            if r.returncode:
                return lit.Test.Result(lit.Test.FAIL, f"--dump-ast failed:\n{r.stderr}")
            with open(dump, "w") as f:
                f.write(r.stdout)
            r = run("ptxas", f"-arch={arch}", dump, "-o", redo)
            if r.returncode:
                return lit.Test.Result(lit.Test.FAIL, f"ptxas rejects the dump:\n{r.stderr}")
            sass = [run("cuobjdump", "-sass", cubin) for cubin in (orig, redo)]
            if any(r.returncode for r in sass):
                return lit.Test.Result(lit.Test.FAIL, "cuobjdump failed:\n" + "".join(r.stderr for r in sass))
            orig_sass, redo_sass = ([l for l in r.stdout.splitlines() if not l.lstrip().startswith("/* 0x")] for r in sass)
            if orig_sass != redo_sass:
                return lit.Test.Result(lit.Test.FAIL, "the dump assembles to different SASS")
        return lit.Test.Result(lit.Test.PASS)
