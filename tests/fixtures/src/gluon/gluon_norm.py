"""nanochat's residual RMSNorm site as two Gluon kernels that hand the next GEMM a pre-quantized operand (sm_89).

    pass 1  x_new = a (x + z) + b x0 (z, x0 optional), rounded to bf16 as the residual stream is;
            rstd = rsqrt(mean(x_new^2) + eps) per row; amax = max_rows rstd * max|x_new| == max |x_new rstd|
            (the norm scales rows independently and fp32 multiplication is monotone, so the tensorwise amax of
            the normalized tensor is exact without materializing it)
    pass 2  y8 = e4m3(x_new rstd * scale) with scale = 448 / amax, and y8^T for the weight-gradient GEMM

So the GEMM reads one byte per element through cp.async instead of quantizing a bf16 operand in registers,
and no longer writes the transposed side output. A like-for-like bf16 replacement of Inductor's fused norm
kernel measured at parity (experiments/gluon_norm_fwd.py); this one changes what is written, not how fast.
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
from typing import Optional, Tuple
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from gluon_fp8 import FP8_MAX, scale_from_amax

NORM_EPS = torch.finfo(torch.float32).eps      # what F.rms_norm uses on a bf16 tensor (it upcasts first)


@gluon.jit
def _mix(x_ptr, z_ptr, x0_ptr, offs, cmask, a, b, HAS_Z: gl.constexpr, HAS_X0: gl.constexpr):
    v = gl.load(x_ptr + offs, mask=cmask, other=0.0).to(gl.float32)
    if HAS_Z:
        v = v + gl.load(z_ptr + offs, mask=cmask, other=0.0).to(gl.float32)
    v = v * a
    if HAS_X0:
        v = v + b * gl.load(x0_ptr + offs, mask=cmask, other=0.0).to(gl.float32)
    return v


@gluon.jit
def rmsnorm_stats_kernel(x_ptr, z_ptr, x0_ptr, xn_ptr, rstd_ptr, amax_ptr, a_ptr, b_ptr, M, eps,
                         C: gl.constexpr, C_PAD: gl.constexpr, ROWS: gl.constexpr, HAS_Z: gl.constexpr, HAS_X0: gl.constexpr,
                         WRITE_XN: gl.constexpr, num_warps: gl.constexpr):
    # a warp per row, 8-element (16-byte) vectors; C_PAD is the next power of two, the tail lanes are masked off
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [num_warps, 1], [1, 0])
    row0 = gl.program_id(0) * (num_warps * ROWS)
    cols = gl.arange(0, C_PAD, layout=gl.SliceLayout(0, layout))
    cmask = (cols < C)[None, :]
    a = gl.load(a_ptr)
    b = gl.load(b_ptr)
    tile_max = gl.zeros([num_warps], gl.float32, gl.SliceLayout(1, layout))
    rows = row0 + gl.arange(0, num_warps, layout=gl.SliceLayout(1, layout))
    v_next = _mix(x_ptr, z_ptr, x0_ptr, rows[:, None] * C + cols[None, :], cmask, a, b, HAS_Z, HAS_X0)
    for r in gl.static_range(ROWS):
        v = v_next
        offs = rows[:, None] * C + cols[None, :]
        if r + 1 < ROWS:                                        # next row's loads in flight before this row's stores
            v_next = _mix(x_ptr, z_ptr, x0_ptr, offs + num_warps * C, cmask, a, b, HAS_Z, HAS_X0)
        xn = v.to(gl.bfloat16)                                  # the residual stream is bf16
        if WRITE_XN:
            gl.store(xn_ptr + offs, xn, mask=cmask)
        xf = xn.to(gl.float32)
        rstd = gl.rsqrt(gl.sum(xf * xf, axis=1) / C + eps)
        gl.store(rstd_ptr + rows, rstd)
        tile_max = gl.maximum(tile_max, gl.max(gl.abs(xf), axis=1) * rstd)   # == max_j |xf_j * rstd|: fp32 * is monotone
        rows = rows + num_warps
    bits = tile_max.to(gl.int32, bitcast=True)                  # non-negative fp32 orders like int32; NaN sorts last
    gl.atomic_max(amax_ptr, gl.max(bits, axis=0))


@gluon.jit
def _quant_i8(x, scale):
    """fp32 -> e4m3 bits as int8 (the same convert as gluon_fp8._quant); int8 so the shared-memory tile below is typed
    as plain bytes. The corruption first seen here was a ptxas 12.9.86 miscompile of packed e4m3 convert results, not a
    Gluon bug (experiments/ptxas_129_e4m3_byte_store_repro.py); the kernels default to the CUDA 13.3 ptxas."""
    xf = x.to(gl.float32) * scale
    return gl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
        "=r,r,r,r,r", [xf], dtype=gl.int8, is_pure=True, pack=4)


@gluon.jit
def quantize_transpose_kernel(x_ptr, rstd_ptr, scale_ptr, y_ptr, yt_ptr, M, C: gl.constexpr, BM: gl.constexpr, BC: gl.constexpr,
                              num_warps: gl.constexpr):
    """y8 = e4m3(x rstd scale) tile by tile, row-major and transposed (M-contiguous). The transpose goes through shared
    memory explicitly: written row-major, read back with 16 consecutive rows per thread."""
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    tr_layout: gl.constexpr = gl.BlockedLayout([16, 1], [4, 8], [1, num_warps], [0, 1])
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=8, order=[1, 0])
    pid_m = gl.program_id(0)
    pid_c = gl.program_id(1)
    rows = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, layout))
    cols = pid_c * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, layout))
    x = gl.load(x_ptr + rows[:, None] * C + cols[None, :]).to(gl.float32)
    q = _quant_i8(x * gl.load(rstd_ptr + rows)[:, None], gl.load(scale_ptr))
    gl.store(y_ptr + rows[:, None] * C + cols[None, :], q)
    tile = gl.allocate_shared_memory(gl.int8, [BM, BC], layout=smem_layout)
    tile.store(q)
    gl.barrier()
    qt = tile.load(tr_layout)
    rows_t = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, tr_layout))
    cols_t = pid_c * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, tr_layout))
    gl.store(yt_ptr + cols_t[None, :] * M + rows_t[:, None], qt)


_consts = {}


def _scalar(s, device):
    if isinstance(s, torch.Tensor):
        return s.to(torch.float32).reshape(())
    key = (device, float(s))
    if key not in _consts:
        _consts[key] = torch.full((), float(s), dtype=torch.float32, device=device)
    return _consts[key]


def rmsnorm_stats(x, z=None, x0=None, a=1.0, b=0.0, eps=NORM_EPS, write_xn=True, rows=2, num_warps=4):
    """Pass 1. x, z, x0: bf16 [M, C] contiguous. Returns (x_new, rstd fp32 [M], amax of the normalized tensor as fp32 0-dim).
    With write_xn=False (no z, a == 1, b == 0) x_new is x itself and is not written."""
    M, C = x.shape
    assert x.is_contiguous() and (z is None or z.shape == x.shape) and (x0 is None or x0.shape == x.shape)
    xn = torch.empty_like(x) if write_xn else x
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    amax = torch.zeros((), dtype=torch.int32, device=x.device)
    per_cta = num_warps * rows
    assert M % per_cta == 0
    rmsnorm_stats_kernel[(M // per_cta,)](x, z if z is not None else x, x0 if x0 is not None else x, xn, rstd, amax,
                                          _scalar(a, x.device), _scalar(b, x.device), M, float(eps), C=C, C_PAD=triton.next_power_of_2(C),
                                          ROWS=rows, HAS_Z=z is not None, HAS_X0=x0 is not None, WRITE_XN=write_xn, num_warps=num_warps)
    return xn, rstd, amax.view(torch.float32)


def quantize_transpose(xn, rstd, scale, BM=128, BC=128, num_warps=4):
    """Pass 2. Returns (y8 [M, C], y8^T [C, M]), both float8_e4m3fn."""
    M, C = xn.shape
    assert M % BM == 0 and C % BC == 0
    y = torch.empty((M, C), dtype=torch.int8, device=xn.device)
    yt = torch.empty((C, M), dtype=torch.int8, device=xn.device)
    quantize_transpose_kernel[(M // BM, C // BC)](xn, rstd, _scalar(scale, xn.device), y, yt, M, C=C, BM=BM, BC=BC, num_warps=num_warps)
    return y.view(torch.float8_e4m3fn), yt.view(torch.float8_e4m3fn)


def rmsnorm_fp8(x, z=None, x0=None, a=1.0, b=0.0):
    """Both passes: (x_new, y8, y8^T, scale, rstd). scale is the FP8 scale of y8 (448 / amax), a 0-dim fp32 tensor.
    Without z and x0 the normalized tensor is x itself: x_new is returned as a 0-element tensor and not written."""
    write_xn = z is not None or x0 is not None
    xn, rstd, amax = rmsnorm_stats(x, z, x0, a, b, write_xn=write_xn)
    scale = scale_from_amax(amax)
    y, yt = quantize_transpose(xn, rstd, scale)
    return (xn if write_xn else torch.empty(0, dtype=x.dtype, device=x.device)), y, yt, scale, rstd


# ---- torch custom ops, so torch.compile keeps the kernels in the graph

@torch.library.custom_op("gluon_norm::rmsnorm_fp8", mutates_args=())
def rmsnorm_fp8_op(x: torch.Tensor, z: Optional[torch.Tensor], x0: Optional[torch.Tensor], a: float,
                   b: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_new or empty, y8, y8^T, scale, rstd); no autograd: the consumer differentiates through the norm with rstd."""
    xn, y, yt, scale, rstd = rmsnorm_fp8(x, z, x0, a, b)
    return xn, y, yt, scale, rstd


@rmsnorm_fp8_op.register_fake
def _(x, z, x0, a, b):
    M, C = x.shape
    return (torch.empty_like(x) if (z is not None or x0 is not None) else torch.empty(0, dtype=x.dtype, device=x.device),
            torch.empty((M, C), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((C, M), dtype=torch.float8_e4m3fn, device=x.device), torch.empty((), dtype=torch.float32, device=x.device),
            torch.empty(M, dtype=torch.float32, device=x.device))


def rmsnorm_backward(dy, xn, rstd):
    """dx of y = xn rstd given dy (any dtype), in fp32: rstd (dy - y mean(dy y)); returned in xn's dtype."""
    C = xn.shape[-1]
    y = xn.float() * rstd[:, None]
    dyf = dy.float()
    return (rstd[:, None] * (dyf - y * (dyf * y).sum(-1, keepdim=True) / C)).to(xn.dtype)


if __name__ == "__main__":
    import torch.nn.functional as F
    from triton.testing import do_bench
    from gluon_fp8 import quantize
    torch.manual_seed(0)
    M, C = 32768, 768
    x, z, x0 = (torch.randn(M, C, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    a, b = torch.tensor(0.9, device="cuda"), torch.tensor(0.2, device="cuda")
    rel = lambda p, q: ((p.float() - q.float()).norm() / q.float().norm()).item()
    for name, zz, xx0, aa, bb in (("site 1 (mix fused)", z, x0, a, b), ("site 2 (residual add)", z, None, 1.0, 0.0), ("plain norm of x", None, None, 1.0, 0.0)):
        xn, y8, yt, scale, rstd = rmsnorm_fp8(x, zz, xx0, aa, bb)
        xn_ref = (aa * (x + zz) + bb * xx0) if xx0 is not None else (x + zz if zz is not None else x)
        y_ref = F.rms_norm(xn_ref, (C,))
        s_ref = scale_from_amax(y_ref.float().abs().amax())
        xn_k = xn if xn.numel() else x
        y8_ref = quantize(xn_k.float() * rstd[:, None], scale)             # the kernel's own x_new and scale, exact math
        print(f"{name}: xn rel {rel(xn_k, xn_ref):.1e}  scale {scale.item():.4f} vs {s_ref.item():.4f}  y8 == exact quantize: "
              f"{torch.equal(y8.view(torch.int8), y8_ref.view(torch.int8))}  y8^T ok {torch.equal(yt.view(torch.int8), y8.t().contiguous().view(torch.int8))}  "
              f"dequantized y vs F.rms_norm {rel(y8.float() / scale, y_ref):.1e}")
        ms1 = do_bench(lambda: rmsnorm_stats(x, zz, xx0, aa, bb, write_xn=xx0 is not None or zz is not None))
        ms2 = do_bench(lambda: quantize_transpose(xn_k, rstd, scale))
        print(f"    pass 1 {ms1:.3f} ms, pass 2 {ms2:.3f} ms, total {ms1+ms2:.3f} ms")
    for BM, BC, nw in ((64, 128, 4), (128, 128, 4), (64, 128, 8), (128, 128, 8), (32, 128, 4)):
        ms2 = do_bench(lambda: quantize_transpose(xn_k, rstd, scale, BM=BM, BC=BC, num_warps=nw))
        print(f"    pass 2 tile {BM}x{BC} warps {nw}: {ms2:.3f} ms = {(2 + 1 + 1) * M * C / ms2 / 1e6:.0f} GB/s")
