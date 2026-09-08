"""The cross-entropy of one chunk of logits as a Gluon kernel (sm_89), for the fused linear cross-entropy.

For a chunk of Mc rows of bf16 scores l (Mc x V, straight from the lm_head GEMM) and targets y (-1 = ignore):
    z = c tanh(l / c)                                   nanochat's logit softcap, c = 15
    loss_t = logsumexp(z_t) - z_{t, y_t}                 0 for ignored rows
    g = (softmax(z) - onehot(y)) (1 - (z/c)^2) / N       the gradient of the mean loss w.r.t. l, 0 for ignored rows
plus max |g| over the chunk, the FP8 scale of the two GEMMs that consume g. One CTA per row: pass one streams the
row for the online max and sum of exponentials, pass two streams it again and writes g in bf16. The chunk is sized
to sit in L2, so the second read is cheap.
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

SOFTCAP = 15.0
IGNORE_INDEX = -1


@gluon.jit
def _tanh(x):
    # exact tanh through exp, overflow-safe: sign(x) (1 - 2 / (exp(2|x|) + 1))
    a = gl.abs(x)
    t = 1.0 - 2.0 / (gl.exp(2.0 * a) + 1.0)
    return gl.where(x < 0, -t, t)


@gluon.jit
def ce_chunk_kernel(l_ptr, tgt_ptr, loss_ptr, g_ptr, amax_ptr, inv_n_ptr, softcap, V: gl.constexpr, BLOCK: gl.constexpr,
                    num_warps: gl.constexpr):
    """One row per CTA, warps along the row. The softcap bounds |z| < softcap, so the logsumexp needs no running maximum:
    every thread accumulates exp(z - softcap) over its own elements and the CTA reduces once per pass, two cross-warp
    reductions per row instead of two per block (Nsight: 45% of the stall cycles were warps at those barriers)."""
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, num_warps], [1, 0])
    row = gl.program_id(0)
    cols0 = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
    tgt = gl.load(tgt_ptr + row).to(gl.int32)
    valid = tgt != -1
    inv_n = gl.load(inv_n_ptr)
    inv_c = 1.0 / softcap
    base = l_ptr + row.to(gl.int64) * V
    lt = gl.load(base + gl.where(valid, tgt, 0)).to(gl.float32)
    zt = softcap * _tanh(lt * inv_c)
    s_v = gl.zeros([1, BLOCK], gl.float32, layout)                  # per-slot partial sums of exp(z - softcap)
    for b in range(0, V, BLOCK):
        z = softcap * _tanh(gl.load(base + b + cols0[None, :]).to(gl.float32) * inv_c)
        s_v += gl.exp(z - softcap)
    lse = softcap + gl.log(gl.sum(s_v, axis=1))                       # the one reduction of pass one
    loss = gl.where(valid, lse - zt, 0.0)
    gl.store(loss_ptr + row + gl.arange(0, 1, layout=gl.SliceLayout(1, layout)), loss)
    gmax_v = gl.zeros([1, BLOCK], gl.float32, layout)
    for b in range(0, V, BLOCK):
        cols = b + cols0
        z = softcap * _tanh(gl.load(base + cols[None, :]).to(gl.float32) * inv_c)
        p = gl.exp(z - lse[:, None])
        onehot = (cols[None, :] == tgt).to(gl.float32)
        zc = z * inv_c
        g = (p - onehot) * (1.0 - zc * zc) * inv_n
        g = gl.where(valid, g, 0.0).to(gl.bfloat16)
        gl.store(g_ptr + row.to(gl.int64) * V + cols[None, :], g)
        gmax_v = gl.maximum(gmax_v, gl.abs(g.to(gl.float32)))
    gmax = gl.max(gmax_v, axis=1)                                      # the one reduction of pass two
    gl.atomic_max(amax_ptr, gl.max(gmax.to(gl.int32, bitcast=True), axis=0))


def ce_chunk(logits, targets, inv_n, softcap=SOFTCAP, block=4096, num_warps=16):
    """logits: bf16 [Mc, V] contiguous; targets: int64 [Mc]; inv_n: 0-dim fp32 tensor, 1 / (valid tokens in the batch).
    Returns (loss per row fp32 [Mc], g bf16 [Mc, V], max |g| as a 0-dim fp32 tensor)."""
    Mc, V = logits.shape
    assert logits.is_contiguous() and V % block == 0
    loss = torch.empty(Mc, dtype=torch.float32, device=logits.device)
    g = torch.empty_like(logits)
    amax = torch.zeros((), dtype=torch.int32, device=logits.device)
    ce_chunk_kernel[(Mc,)](logits, targets, loss, g, amax, inv_n, float(softcap), V=V, BLOCK=block, num_warps=num_warps)
    return loss, g, amax.view(torch.float32)


def reference(logits, targets, inv_n, softcap=SOFTCAP):
    """nanochat's ops on one chunk: loss per row and the gradient of the mean loss w.r.t. the bf16 logits."""
    import torch.nn.functional as F
    l = logits.float().detach().requires_grad_()
    z = softcap * torch.tanh(l / softcap)
    per_row = F.cross_entropy(z, targets, ignore_index=IGNORE_INDEX, reduction="none")
    (per_row.sum() * inv_n).backward()
    return per_row.detach(), l.grad


if __name__ == "__main__":
    from triton.testing import do_bench
    torch.manual_seed(0)
    V = 32768
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    for Mc in (1024, 2048, 4096):
        logits = (torch.randn(Mc, V, device="cuda") * 4).to(torch.bfloat16)
        targets = torch.randint(0, V, (Mc,), device="cuda")
        targets[::97] = IGNORE_INDEX
        inv_n = torch.tensor(1.0 / 32768, device="cuda")
        loss_ref, g_ref = reference(logits, targets, inv_n)
        loss, g, amax = ce_chunk(logits, targets, inv_n)
        print(f"Mc {Mc}: loss rel {rel(loss, loss_ref):.1e} (ignored rows zero: {bool((loss[::97] == 0).all())}), g rel {rel(g, g_ref):.1e}, "
              f"amax {amax.item():.3e} vs {g.float().abs().amax().item():.3e}")
        ms = do_bench(lambda: ce_chunk(logits, targets, inv_n))
        for nw, blk in ((4, 2048), (8, 4096), (16, 4096), (8, 1024)):
            try:
                ms2 = do_bench(lambda: ce_chunk(logits, targets, inv_n, block=blk, num_warps=nw))
            except Exception as e:
                ms2 = float("nan")
            print(f"    warps 8 block 2048: {ms:.3f} ms ({3 * Mc * V * 2 / ms / 1e6:.0f} GB/s over the chunk's 3 passes)   warps {nw} block {blk}: {ms2:.3f} ms")
