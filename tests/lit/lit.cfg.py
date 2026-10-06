import os

import lit.formats

config.name = "ptxroof"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)

repo = os.path.dirname(os.path.dirname(config.test_source_root))
llvm_bin = os.path.join(os.path.dirname(lit.__file__), "..", "..", "..", "..", "bin")
config.environment["PATH"] = os.pathsep.join([os.path.realpath(llvm_bin), os.environ["PATH"]])
config.environment["FILECHECK_OPTS"] = "--strict-whitespace"
ptxroof = os.environ.get("PTXROOF", os.path.join(repo, "target", "debug", "ptxroof"))
config.substitutions.append(("%ptxroof", ptxroof))
config.substitutions.append(("%fixtures", os.path.join(repo, "tests", "fixtures")))
