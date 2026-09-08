"""Gluon fused rotary + QK-RMSNorm + 1.2x scale (nanochat CausalSelfAttention glue).

Per row (b, t, h) of D=128 channels:
  x1, x2 = x[:D/2], x[D/2:]
  y  = [x1*cos + x2*sin,  -x1*sin + x2*cos]      (apply_rotary_emb)
  y  = y * rsqrt(mean(y^2) + eps) * 1.2           (F.rms_norm in bf16, then the 1.2 sharpening)

Inductor already fuses this into one kernel; this is the same fusion written by hand to
measure whether a hand kernel beats the compiler on a purely memory-bound op on sm_89.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def rope_norm_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, n_rows, T, H, eps, scale,
                     D: gl.constexpr, ROWS: gl.constexpr, num_warps: gl.constexpr):
    HALF: gl.constexpr = D // 2
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    pid = gl.program_id(0)
    rows = pid * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, HALF, layout=gl.SliceLayout(0, layout))
    rmask = rows < n_rows
    mask = rmask[:, None] & (cols[None, :] < HALF)
    t = (rows // H) % T                               # position of each row
    x_off = rows[:, None] * D + cols[None, :]
    cs_off = t[:, None] * HALF + cols[None, :]        # cos/sin are [T, HALF]
    x1 = gl.load(x_ptr + x_off, mask=mask, other=0.0).to(gl.float32)
    x2 = gl.load(x_ptr + x_off + HALF, mask=mask, other=0.0).to(gl.float32)
    c = gl.load(cos_ptr + cs_off, mask=mask, other=0.0).to(gl.float32)
    s = gl.load(sin_ptr + cs_off, mask=mask, other=0.0).to(gl.float32)
    y1 = x1 * c + x2 * s
    y2 = x2 * c - x1 * s
    # nanochat rotates in bf16 then norms; round like the reference does
    y1 = y1.to(gl.bfloat16).to(gl.float32)
    y2 = y2.to(gl.bfloat16).to(gl.float32)
    ss = gl.sum(y1 * y1 + y2 * y2, axis=1)
    r = gl.rsqrt(ss / D + eps) * scale
    gl.store(y_ptr + x_off, (y1 * r[:, None]).to(gl.bfloat16), mask=mask)
    gl.store(y_ptr + x_off + HALF, (y2 * r[:, None]).to(gl.bfloat16), mask=mask)


def gluon_rope_norm(x, cos, sin, scale=1.2, ROWS=64, num_warps=4):
    """x: [B, T, H, D] bf16 contiguous; cos/sin: [1, T, 1, D/2] bf16."""
    B, T, H, D = x.shape
    y = torch.empty_like(x)
    n_rows = B * T * H
    eps = torch.finfo(x.dtype).eps  # F.rms_norm default eps for the input dtype
    grid = (triton.cdiv(n_rows, ROWS),)
    rope_norm_kernel[grid](x, cos.contiguous(), sin.contiguous(), y, n_rows, T, H, eps, scale,
                           D=D, ROWS=ROWS, num_warps=num_warps)
    return y


def ref_rope_norm(x, cos, sin, scale=1.2):
    import torch.nn.functional as F
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)
    return F.rms_norm(y, (y.size(-1),)) * scale


if __name__ == "__main__":
    import json, os
    from triton.testing import do_bench
    torch.manual_seed(0)
    B, T, H, D = 16, 2048, 6, 128
    x = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    ch = torch.arange(0, D, 2, dtype=torch.float32, device="cuda")
    inv = 1.0 / (100000 ** (ch / D))
    fr = torch.outer(torch.arange(T, dtype=torch.float32, device="cuda"), inv)
    cos, sin = fr.cos().to(torch.bfloat16)[None, :, None, :], fr.sin().to(torch.bfloat16)[None, :, None, :]

    ref = ref_rope_norm(x, cos, sin)
    out = gluon_rope_norm(x, cos, sin)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"max abs err vs eager reference: {err:.4f} (bf16 ulp at 1.0 = 0.0078)")
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-2)

    ref_c = torch.compile(ref_rope_norm)
    ref_c(x, cos, sin)
    nbytes = 2 * x.numel() * 2 + 2 * T * (D // 2) * 2
    res = {}
    for name, fn in [("eager (7 kernels)", lambda: ref_rope_norm(x, cos, sin)),
                     ("torch.compile fused", lambda: ref_c(x, cos, sin)),
                     ("gluon fused ROWS=64 warps=4", lambda: gluon_rope_norm(x, cos, sin)),
                     ("gluon fused ROWS=32 warps=4", lambda: gluon_rope_norm(x, cos, sin, ROWS=32)),
                     ("gluon fused ROWS=128 warps=8", lambda: gluon_rope_norm(x, cos, sin, ROWS=128, num_warps=8))]:
        ms = do_bench(fn)
        res[name] = dict(ms=ms, gbs=nbytes / ms / 1e6)
        print(f"  {name:32s} {ms:7.3f} ms   {nbytes/ms/1e6:7.0f} GB/s")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/gluon_rope_norm.json", "w"), indent=1)
