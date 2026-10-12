#!/usr/bin/env python3
"""Replays the command a sol-study report profiled, as if this machine were
the B200 the report came from, and records every Triton launch without
running it: the kernel, the SASS digest of its cubin, grid, dynamic shared
bytes and PTX parameters in tools/ncu/launch.c's forms, or, for parameters
that only a TMA kernel takes and launch.c does not, i64:, i1: and
tensormap:128 (a CUtensorMap's bytes). Run from the
sol-study checkout with its Triton (PYTHONPATH=<sol-study>/vendor) and the
torch it ran with.
Usage: record_launches.py OUT.jsonl '[LABEL, SCRIPT, ARG...]'
"""

import json
import os
import re
import runpy
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

from triton._utils import get_iterable_path  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.backends.driver import TensorDescABI, decompose_descriptor, get_kernel_argument_layout  # noqa: E402
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
    if ty in ("i64", "i1"):
        return f"{ty}:{int(arg)}"
    if ty == "nvTmaDesc":
        return "tensormap:128"
    raise ValueError(f"no launch.c form for a {ty} parameter {arg!r}")


def descriptor_fields(arg, abi):
    """A tensor descriptor's parameters in Triton's order (triton/backends/driver.py,
    _expand_descriptor): the TMA descriptor, or the decomposed base, then shape and strides."""
    if abi is TensorDescABI.CUDA_TMA:
        return [None, *arg.shape, *arg.strides]
    return decompose_descriptor(arg)


launcher_init = CudaLauncher.__init__


def init(self, src, metadata):
    launcher_init(self, src, metadata)
    signature = tuple(src.signature.values())
    abi = TensorDescABI.CUDA_TMA if getattr(metadata, "tensordesc_meta", None) else TensorDescABI.DECOMPOSED
    self.recorded = metadata.name, metadata.shared, signature, abi, get_kernel_argument_layout(signature, abi)


# The PTX parameters are the non-constexpr arguments, then the global and
# profile scratch pointers (CudaLauncher.__call__).
def call(self, gx, gy, gz, stream, function, kernel_metadata, launch_metadata, enter, exit_, *args):
    name, shared, signature, abi, layout = self.recorded
    params, descriptors = [], {}
    for path, ty in layout:
        arg = get_iterable_path(args, path)
        if str(get_iterable_path(signature, path)).startswith("tensordesc"):
            arg = next(descriptors.setdefault(path, iter(descriptor_fields(arg, abi))))
        params.append(param(ty, arg))
    params += [f"ptr:{gx * gy * gz * self.num_ctas * size}" for size in (self.global_scratch_size, self.profile_scratch_size)]
    launched.append({"kernel": name, "sass": function, "grid": [gx, gy, gz], "shared": shared, "params": params})


CudaLauncher.__init__, CudaLauncher.__call__ = init, call

out_path, (label, script, *args) = sys.argv[1], json.loads(sys.argv[2])
sys.argv = [script, *args]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
try:
    runpy.run_path(script, run_name="__main__")
finally:
    with open(out_path, "a") as out:
        for launch in launched:
            out.write(json.dumps({"label": label, **launch}) + "\n")
