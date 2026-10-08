import os
import shlex

import lit.formats

config.name = "ptxroof"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)

repo = os.path.dirname(os.path.dirname(config.test_source_root))
config.test_exec_root = os.path.join(repo, "target", "lit")
llvm_bin = os.path.join(os.path.dirname(lit.__file__), "..", "..", "..", "..", "bin")
config.environment["PATH"] = os.pathsep.join([os.path.realpath(llvm_bin), os.environ["PATH"]])
config.environment["FILECHECK_OPTS"] = "--strict-whitespace"
ptxroof = os.environ.get("PTXROOF", os.path.join(repo, "target", "debug", "ptxroof"))
config.substitutions.append(("%ptxroof", shlex.quote(ptxroof)))
config.substitutions.append(("%fixtures", shlex.quote(os.path.join(repo, "tests", "fixtures"))))
config.substitutions.append(("%check-report", "python3 " + shlex.quote(os.path.join(config.test_source_root, "check_report.py"))))
