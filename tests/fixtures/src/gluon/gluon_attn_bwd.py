"""Gluon flash-attention backward for sm_89 (RTX 4090).

Inputs q, k, v, o, do: [B, T, H, D] bf16; lse: [B, H, T] fp32 (natural log, from the forward). Causal with an
optional left window. Three kernels, as FA2/FA3 on this architecture:
  preprocess   D[b,h,t] = sum_d dO * O (fp32) and dq_accum = 0
  main         one CTA per (key block j, b, h), K_j and V_j kept in registers as MMA A operands for the whole
               query loop; per query block i (Q_i, dO_i double-buffered through cp.async):
                   S^T  = K Q^T          P^T  = exp2(S^T scale_log2 - lse log2e)   (masked)
                   dP^T = V dO^T         dS^T = P^T (dP^T - D)
                   dV  += P^T dO         dK  += dS^T Q
                   dQ_i += dS K          (dS read back transposed from shared memory; fp32 atomics into dq_accum)
               The transposed formulation puts P^T and dS^T in the accumulator layout, which is the A-operand
               layout of the next MMAs, so they never leave registers.
  postprocess  dq = (dq_accum * scale).bf16, back in [B, T, H, D]
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2

LOG2E = gl.constexpr(1.4426950408889634)


@gluon.jit
def attn_bwd_pre_kernel(o_ptr, do_ptr, d_ptr, stride_b, stride_t, stride_h, H, T, D: gl.constexpr, BLOCK_T: gl.constexpr,
                        num_warps: gl.constexpr):
    """D[b, h, t] = sum_d dO[b, t, h, d] * O[b, t, h, d]; one CTA per (BLOCK_T rows, b, h)."""
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    tb = gl.program_id(0)
    bh = gl.program_id(1)
    base = (bh // H) * stride_b + (bh % H) * stride_h
    rows = tb * BLOCK_T + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    offs = base + rows[:, None] * stride_t + cols[None, :]
    prod = gl.load(o_ptr + offs).to(gl.float32) * gl.load(do_ptr + offs).to(gl.float32)
    gl.store(d_ptr + bh * T + rows, gl.sum(prod, axis=1))


@gluon.jit
def attn_bwd_post_kernel(dqa_ptr, dq_ptr, stride_b, stride_t, stride_h, H, T, scale, D: gl.constexpr, BLOCK_T: gl.constexpr,
                         num_warps: gl.constexpr):
    """dq[b, t, h, :] = (dq_accum[b, h, t, :] * scale).bf16"""
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [num_warps, 1], [1, 0])
    tb = gl.program_id(0)
    bh = gl.program_id(1)
    rows = tb * BLOCK_T + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    acc = gl.load(dqa_ptr + (bh * T + rows)[:, None] * D + cols[None, :])
    base = (bh // H) * stride_b + (bh % H) * stride_h
    gl.store(dq_ptr + base + rows[:, None] * stride_t + cols[None, :], (acc * scale).to(gl.bfloat16))


@gluon.jit
def attn_bwd_kernel(q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, d_ptr, dqa_ptr, dk_ptr, dv_ptr,
                    stride_b, stride_t, stride_h, H, T, scale_log2, scale, window,
                    LOCAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr, num_warps: gl.constexpr):
    """Every operand tile is kept at [64, 64] (the head dimension split in two halves) so no MMA operand exceeds
    32 registers per thread; K and V are re-read from shared memory each iteration rather than pinned in
    registers. Q and dO are single-buffered and re-issued as soon as their last MMA of the iteration is done, so
    the copies overlap the rest of the iteration and the dQ atomics."""
    j = gl.program_id(0)                                          # key block; j = 0 is the heaviest under causal masking
    bh = gl.program_id(1)
    base = (bh // H) * stride_b + (bh % H) * stride_h
    k_start = j * BLOCK_N
    DH: gl.constexpr = D // 2

    ld_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, num_warps // 4], instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=2)
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])

    q_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, DH], smem_layout)        # [half]
    do_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M, DH], smem_layout)       # [half]
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_N, DH], smem_layout)
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_N, DH], smem_layout)
    ds_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_N, BLOCK_M], smem_layout)

    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld_layout))
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld_layout))
    offs_d = gl.arange(0, DH, layout=gl.SliceLayout(0, ld_layout))
    kv_ptrs = base + (k_start + offs_n)[:, None] * stride_t + offs_d[None, :]
    q_ptrs = q_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :]
    do_ptrs = do_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :]

    # query blocks i_lo..i_hi: from the diagonal to the end, or to the window's reach
    i_lo = k_start // BLOCK_M
    if LOCAL:
        i_hi = gl.minimum((k_start + BLOCK_N - 1 + window) // BLOCK_M, T // BLOCK_M - 1)
    else:
        i_hi = T // BLOCK_M - 1
    ni = i_hi - i_lo + 1

    for h in gl.static_range(2):
        cp.async_load(k_smem.index(h), k_ptr + kv_ptrs + h * DH)
        cp.async_load(v_smem.index(h), v_ptr + kv_ptrs + h * DH)
        cp.async_load(q_smem.index(h), q_ptrs + i_lo * BLOCK_M * stride_t + h * DH)
        cp.async_load(do_smem.index(h), do_ptrs + i_lo * BLOCK_M * stride_t + h * DH)
    cp.commit_group()

    dk0 = gl.zeros([BLOCK_N, DH], gl.float32, acc_layout)
    dk1 = gl.zeros([BLOCK_N, DH], gl.float32, acc_layout)
    dv0 = gl.zeros([BLOCK_N, DH], gl.float32, acc_layout)
    dv1 = gl.zeros([BLOCK_N, DH], gl.float32, acc_layout)
    keys = k_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, acc_layout))           # rows of the transposed tiles
    qcols = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, acc_layout))                    # their columns: queries
    dq_cols = gl.arange(0, DH, layout=gl.SliceLayout(0, acc_layout))
    red_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 8], [num_warps, 1], [1, 0])
    red_rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, red_layout))
    red_cols = gl.arange(0, DH, layout=gl.SliceLayout(0, red_layout))

    for it in range(ni):
        i = i_lo + it
        st = it % 2
        cp.wait_group(0)                                          # Q_i and dO_i landed for this thread
        gl.barrier()                                              # ... for all
        queries = i * BLOCK_M + qcols
        lse_i = gl.load(lse_ptr + bh * T + queries) * LOG2E
        d_i = gl.load(d_ptr + bh * T + queries)

        # S^T = K Q^T (two halves of the head dimension) and P^T
        s_t = mma_v2(k_smem.index(0).load(a_op), q_smem.index(0).permute((1, 0)).load(b_op),
                     gl.zeros([BLOCK_N, BLOCK_M], gl.float32, acc_layout))
        s_t = mma_v2(k_smem.index(1).load(a_op), q_smem.index(1).permute((1, 0)).load(b_op), s_t)
        p = gl.exp2(s_t * scale_log2 - lse_i[None, :])
        need_mask = i * BLOCK_M < k_start + BLOCK_N               # the diagonal block(s)
        if LOCAL:
            need_mask = need_mask | (i * BLOCK_M + BLOCK_M - 1 > k_start + window)
        if need_mask:
            keep = keys[:, None] <= queries[None, :]
            if LOCAL:
                keep = keep & (keys[:, None] >= queries[None, :] - window)
            p = gl.where(keep, p, 0.0)
        # dP^T = V dO^T and dS^T
        dp_t = mma_v2(v_smem.index(0).load(a_op), do_smem.index(0).permute((1, 0)).load(b_op),
                      gl.zeros([BLOCK_N, BLOCK_M], gl.float32, acc_layout))
        dp_t = mma_v2(v_smem.index(1).load(a_op), do_smem.index(1).permute((1, 0)).load(b_op), dp_t)
        ds = p * (dp_t - d_i[None, :])
        p16 = gl.convert_layout(p.to(gl.bfloat16), a_op)
        ds16 = ds.to(gl.bfloat16)
        ds_a = gl.convert_layout(ds16, a_op)
        # dV += P^T dO
        dv0 = mma_v2(p16, do_smem.index(0).load(b_op), dv0)
        dv1 = mma_v2(p16, do_smem.index(1).load(b_op), dv1)
        gl.barrier()                                              # dO_i fully consumed: fetch dO_{i+1}
        if it + 1 < ni:
            for h in gl.static_range(2):
                cp.async_load(do_smem.index(h), do_ptrs + (i + 1) * BLOCK_M * stride_t + h * DH)
        cp.commit_group()
        # dK += dS^T Q
        dk0 = mma_v2(ds_a, q_smem.index(0).load(b_op), dk0)
        dk1 = mma_v2(ds_a, q_smem.index(1).load(b_op), dk1)
        # dQ_i = dS K: dS^T through shared memory, read back transposed
        ds_smem.store(ds16)
        gl.barrier()                                              # dS visible; and Q_i fully consumed: fetch Q_{i+1}
        if it + 1 < ni:
            for h in gl.static_range(2):
                cp.async_load(q_smem.index(h), q_ptrs + (i + 1) * BLOCK_M * stride_t + h * DH)
        cp.commit_group()
        ds_q = ds_smem.permute((1, 0)).load(a_op)
        dq0 = mma_v2(ds_q, k_smem.index(0).load(b_op), gl.zeros([BLOCK_M, DH], gl.float32, acc_layout))
        dq1 = mma_v2(ds_q, k_smem.index(1).load(b_op), gl.zeros([BLOCK_M, DH], gl.float32, acc_layout))
        # re-tile so each red.f32 instruction covers whole 32-byte sectors (8 lanes on 8 consecutive floats)
        dq_ptrs = dqa_ptr + (bh * T + i * BLOCK_M + red_rows)[:, None] * D + red_cols[None, :]
        gl.atomic_add(dq_ptrs, gl.convert_layout(dq0, red_layout), sem="relaxed")
        gl.atomic_add(dq_ptrs + DH, gl.convert_layout(dq1, red_layout), sem="relaxed")
    cp.wait_group(0)

    out_ptrs = base + keys[:, None] * stride_t + dq_cols[None, :]
    gl.store(dk_ptr + out_ptrs, (dk0 * scale).to(gl.bfloat16))
    gl.store(dk_ptr + out_ptrs + DH, (dk1 * scale).to(gl.bfloat16))
    gl.store(dv_ptr + out_ptrs, dv0.to(gl.bfloat16))
    gl.store(dv_ptr + out_ptrs + DH, dv1.to(gl.bfloat16))


def attn_bwd(q, k, v, o, do, lse, window=None, sm_scale=None, BLOCK_M=64, BLOCK_N=64, num_warps=8):
    B, T, H, D = q.shape
    assert T % BLOCK_M == 0 and T % BLOCK_N == 0 and BLOCK_M == BLOCK_N
    scale = D ** -0.5 if sm_scale is None else sm_scale
    dev = q.device
    d = torch.empty(B, H, T, dtype=torch.float32, device=dev)
    dq_accum = torch.zeros(B, H, T, D, dtype=torch.float32, device=dev)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    attn_bwd_pre_kernel[(T // 64, B * H)](o, do, d, q.stride(0), q.stride(1), q.stride(2), H, T, D=D, BLOCK_T=64, num_warps=4)
    attn_bwd_kernel[(T // BLOCK_N, B * H)](q, k, v, do, lse, d, dq_accum, dk, dv, q.stride(0), q.stride(1), q.stride(2), H, T,
                                            scale * 1.4426950408889634, scale, -1 if window is None else window, LOCAL=window is not None,
                                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, num_warps=num_warps)
    attn_bwd_post_kernel[(T // 64, B * H)](dq_accum, dq, q.stride(0), q.stride(1), q.stride(2), H, T, scale, D=D, BLOCK_T=64, num_warps=4)
    return dq, dk, dv


def reference_bwd(q, k, v, do, window=None):
    """Gradients of the fp32 reference attention (mean-free, plain sum of o * do)."""
    from gluon_attn_fwd import reference
    qf, kf, vf = (x.float().detach().requires_grad_() for x in (q, k, v))
    o, _ = reference(qf, kf, vf, window)
    o.backward(do.float())
    return qf.grad, kf.grad, vf.grad


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/whe302/ml/nanochat")
    from nanochat.flash_attention import flash_attn
    from bench_fwd import hot
    from gluon_attn_fwd import attn_fwd
    torch.manual_seed(0)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    # correctness on a small problem against fp32 autograd
    B, T, H, D = 2, 512, 2, 128
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    do = torch.randn_like(q)
    for w in (None, 256):
        o, lse = attn_fwd(q, k, v, w)
        dq, dk, dv = attn_bwd(q, k, v, o, do, lse, w)
        rq, rk, rv = reference_bwd(q, k, v, do, w)
        print(f"small window {w}: dq {rel(dq, rq):.1e} dk {rel(dk, rk):.1e} dv {rel(dv, rv):.1e}")
    # full shape: against FA3's own backward, and timing
    B, T, H, D = 16, 2048, 6, 128
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    do = torch.randn_like(q)
    for name, w in (("L", None), ("S", 768)):
        o_fa = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(2048 if w is None else w, 0))
        gq, gk, gv = torch.autograd.grad(o_fa, (q, k, v), do, retain_graph=True)
        qd, kd, vd = q.detach(), k.detach(), v.detach()
        o, lse = attn_fwd(qd, kd, vd, w)
        dq, dk, dv = attn_bwd(qd, kd, vd, o, do, lse, w)
        t_fa = hot(lambda: torch.autograd.grad(o_fa, (q, k, v), do, retain_graph=True), iters=20)
        t_g = hot(lambda: attn_bwd(qd, kd, vd, o, do, lse, w), iters=20)
        print(f"{name}: vs FA3 grads dq {rel(dq, gq):.1e} dk {rel(dk, gk):.1e} dv {rel(dv, gv):.1e} | FA3 bwd {t_fa*1e3:7.1f} us, gluon bwd {t_g*1e3:7.1f} us ({t_fa/t_g:.2f}x)")
