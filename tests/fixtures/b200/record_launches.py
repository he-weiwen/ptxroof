#!/usr/bin/env python3
"""Replays sol-study cases as if this machine were the B200 their reports came
from, and records every Triton launch without running it: the kernel, the SASS
digest of its cubin, grid, dynamic shared bytes and PTX parameters in
tools/ncu/launch.c's forms. Run from the sol-study checkout with its Triton
(PYTHONPATH=<sol-study>/vendor) and the torch it ran with.
Usage: record_launches.py OUT.jsonl CASE...
"""

import importlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="b200-launches-")

import sass  # noqa: E402
import torch  # noqa: E402

# The reports' device attributes: multiprocessor_count, compute_capability,
# max_shared_memory_per_block_optin.
SMS, CAPABILITY, SHARED = 148, (10, 0), 232448
device_properties = torch.cuda.get_device_properties


class B200Properties:
    def __init__(self, props):
        self.props = props

    def __getattr__(self, name):
        b200 = {"multi_processor_count": SMS, "major": CAPABILITY[0], "minor": CAPABILITY[1], "name": "NVIDIA B200"}
        return b200[name] if name in b200 else getattr(self.props, name)


torch.cuda.get_device_properties = lambda *a, **k: B200Properties(device_properties(*a, **k))
torch.cuda.get_device_capability = lambda *a, **k: CAPABILITY

from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.backends.nvidia.driver import CudaDriver, CudaLauncher  # noqa: E402
from triton.compiler.compiler import CompiledKernel  # noqa: E402
from triton.runtime import driver  # noqa: E402

CudaDriver.get_current_target = lambda self: GPUTarget("cuda", CAPABILITY[0] * 10 + CAPABILITY[1], 32)
utils_properties = driver.active.utils.get_device_properties
driver.active.utils.get_device_properties = lambda d: {
    **utils_properties(d), "multiprocessor_count": SMS, "max_shared_mem": SHARED}
cubins = tempfile.mkdtemp(prefix="b200-cubins-")


def init_handles(self):
    """Builds the launcher but loads nothing: an sm_100a cubin does not load here.
    The function handle the launcher receives is the cubin's SASS digest, and the
    register count comes from the cubin, for wrappers that size grids by it."""
    if self.module is not None:
        return
    self._run = driver.active.launcher_cls(self.src, self.metadata)
    path = os.path.join(cubins, self.metadata.hash + ".cubin")
    with open(path, "wb") as f:
        f.write(self.kernel)
    usage = subprocess.run(["cuobjdump", "-res-usage", path], capture_output=True, text=True, check=True).stdout
    self.module, self.function, self.n_spills, self.n_max_threads = path, sass.digest(sass.of_cubin(path)), 0, 1024
    self.n_regs = int(re.search(r"REG:(\d+)", usage)[1])


CompiledKernel._init_handles = init_handles
launched = []


def param(ty, arg):
    if ty.startswith("*"):
        if arg is None:
            return "ptr:0"
        return f"ptr:{arg.untyped_storage().nbytes() - arg.storage_offset() * arg.element_size()}"
    if ty == "u32" and 0 <= arg < 2**32 or ty == "i32" and -(2**31) <= arg < 2**31:
        return f"{ty}:{arg}"
    if ty == "fp32":
        return f"f32:{float(arg)!r}"
    raise ValueError(f"no launch.c form for a {ty} parameter {arg!r}")


def flat(signature, args):
    for ty, arg in zip(signature, args):
        if isinstance(ty, tuple):
            yield from flat(ty, arg)
        elif ty != "constexpr":
            yield ty, arg


launcher_init = CudaLauncher.__init__


def init(self, src, metadata):
    launcher_init(self, src, metadata)
    self.recorded = metadata.name, metadata.shared, list(src.signature.values())


# The PTX parameters are the non-constexpr arguments, then the global and
# profile scratch pointers (CudaLauncher.__call__).
def call(self, gx, gy, gz, stream, function, kernel_metadata, launch_metadata, enter, exit_, *args):
    name, shared, signature = self.recorded
    params = [param(ty, arg) for ty, arg in flat(signature, args)]
    params += [f"ptr:{gx * gy * gz * self.num_ctas * size}" for size in (self.global_scratch_size, self.profile_scratch_size)]
    launched.append({"kernel": name, "sass": function, "grid": [gx, gy, gz], "shared": shared, "params": params})


CudaLauncher.__init__, CudaLauncher.__call__ = init, call

out_path, names = sys.argv[1], sys.argv[2:]
with open(out_path, "a") as out:
    for name in names:
        cases = {c.name: c for c in importlib.import_module(f"benches.{name.split('/')[0]}").cases()}
        launched.clear()
        for _ in range(4):
            cases[name].run()
        for launch in launched:
            out.write(json.dumps({"case": name, **launch}) + "\n")
