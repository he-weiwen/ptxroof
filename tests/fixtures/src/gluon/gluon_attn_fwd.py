"""Gluon flash-attention forward for sm_89 (RTX 4090), warp-specialized into two 4-warp groups.

q, k, v, o: [B, T, H, D] bf16 (FA3's layout), causal with an optional left window. One persistent CTA per SM walks
a static schedule of (query block, b, h) tiles. Inside a CTA two partitions of 4 warps each own 64 of the tile's
128 query rows; they share the K/V stage buffers, each copying half of every block, and synchronize only through
mbarriers: `full[s]` completes when all 256 threads' cp.async copies of a stage landed, `empty[s]` when both groups
arrived after reading it. No CTA-wide barrier exists in the loop, so the two groups drift and one group's softmax
overlaps the other's MMAs on the tensor pipe they share. The refill of a stage is issued after the softmax, which
lets the groups drift by two thirds of an iteration. Q is held in registers per group and the next tile's Q is
prefetched into the group's Q buffer as soon as the current one is loaded.

Tile order (SCHED): 0 = query-block-major snake, heaviest blocks first, all heads concurrently; GH >= 2 = groups of
GH heads x all query blocks, odd groups mirrored, so the K/V working set of the 128 concurrent CTAs is GH heads and
every CTA still gets the same multiset of query blocks (balanced for any causal or windowed work profile).

ABL selects timing-only ablations (results are wrong): 1 no softmax, 2 no rescale, 4 no mask, 8 no waits,
16 no MMA, 32 no loads, 64 no stores.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2, mbarrier

LOG2E = 1.4426950408889634
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@gluon.jit
def _tile_of(j, pid, nprog, SNAKE: gl.constexpr):
    if SNAKE:
        if j % 2 == 0:
            t = j * nprog + pid
        else:
            t = (j + 1) * nprog - 1 - pid
    else:
        t = j * nprog + pid
    return t


@gluon.jit
def _params(tile, num_m, n_bh, H, stride_b, stride_h, window, LOCAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
            SCHED: gl.constexpr):
    if SCHED == 0:                                                # query-block-major: tile t -> (m rank, head)
        m_block = num_m - 1 - tile // n_bh
        bh = tile % n_bh
    else:                                                         # grouped: GH heads x all ranks per group (2 rows of CTAs),
        GH: gl.constexpr = SCHED                                  # odd groups mirrored so every CTA gets the same ranks
        gsz = GH * num_m
        grp = tile // gsz
        u = tile % gsz
        u = gl.where(grp % 2 == 1, gsz - 1 - u, u)
        m_block = num_m - 1 - u // GH
        bh = grp * GH + u % GH
    base = (bh // H) * stride_b + (bh % H) * stride_h
    m_start = m_block * BLOCK_M
    n_hi = (m_start + BLOCK_M - 1) // BLOCK_N
    if LOCAL:
        n_lo = gl.maximum(m_start - window, 0) // BLOCK_N
    else:
        n_lo = m_block * 0
    return base, m_start, n_lo, n_hi - n_lo + 1, bh


@gluon.jit
def _attn_partition(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, stride_b, stride_t, stride_h, H, T, n_tiles, scale_log2, window,
                    k_smem, v_smem, q_smem, full, empty, qbar,
                    PART: gl.constexpr, SNAKE: gl.constexpr, LOCAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                    D: gl.constexpr, STAGES: gl.constexpr, SKIP_MASKED: gl.constexpr, SCHED: gl.constexpr, ABL: gl.constexpr,
                    KW: gl.constexpr, num_warps: gl.constexpr):
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = BLOCK_N // 2
    ld_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])        # 16 B per thread along D
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8])
    q_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=KW)
    k_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=KW)
    p_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=2)
    v_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=2)

    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    num_m = T // BLOCK_M
    n_bh = n_tiles // num_m
    zero = pid * 0

    offs_qm = PART * HALF_M + gl.arange(0, HALF_M, layout=gl.SliceLayout(1, ld_layout))
    offs_kn = PART * HALF_N + gl.arange(0, HALF_N, layout=gl.SliceLayout(1, ld_layout))
    offs_d = gl.arange(0, D, layout=gl.SliceLayout(0, ld_layout))
    q_ptrs = q_ptr + offs_qm[:, None] * stride_t + offs_d[None, :]
    k_ptrs = k_ptr + offs_kn[:, None] * stride_t + offs_d[None, :]
    v_ptrs = v_ptr + offs_kn[:, None] * stride_t + offs_d[None, :]
    rows0 = PART * HALF_M + gl.arange(0, HALF_M, layout=gl.SliceLayout(1, acc_layout))
    cols0 = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, acc_layout))
    od = gl.arange(0, D, layout=gl.SliceLayout(0, acc_layout))

    # consumer cursor: first tile and its Q
    j = zero
    tile = _tile_of(j, pid, nprog, SNAKE)
    base, m_start, n_lo, nb, bh = _params(tile, num_m, n_bh, H, stride_b, stride_h, window, LOCAL, BLOCK_M, BLOCK_N, SCHED)
    cp.async_load(q_smem.index(PART), q_ptrs + base + m_start * stride_t)
    cp.mbarrier_arrive(qbar.index(PART), increment_count=False)
    qphase = zero
    # producer cursor: the stream of key blocks over this CTA's tiles, STAGES-1 blocks ahead of the consumer
    pj = zero
    ptile = tile
    pbase = base
    pn_lo = n_lo
    pnb = nb
    pblk = zero
    pg = zero
    for _s in gl.static_range(STAGES - 1):
        if ptile < n_tiles:
            pst = pg % STAGES
            cp.async_load(k_smem.index(pst).slice(PART * HALF_N, HALF_N, dim=0), k_ptrs + pbase + (pn_lo + pblk) * BLOCK_N * stride_t)
            cp.async_load(v_smem.index(pst).slice(PART * HALF_N, HALF_N, dim=0), v_ptrs + pbase + (pn_lo + pblk) * BLOCK_N * stride_t)
            cp.mbarrier_arrive(full.index(pst), increment_count=False)
            pg += 1
            pblk += 1
            if pblk == pnb:
                pj += 1
                ptile = _tile_of(pj, pid, nprog, SNAKE)
                pblk = zero
                if ptile < n_tiles:
                    pbase, _pm, pn_lo, pnb, _pb = _params(ptile, num_m, n_bh, H, stride_b, stride_h, window, LOCAL, BLOCK_M, BLOCK_N, SCHED)
    g = zero

    while tile < n_tiles:
        mbarrier.wait(qbar.index(PART), qphase)
        qphase = qphase ^ 1
        q = q_smem.index(PART).load(q_op)
        gl.barrier()                                              # this group's warps hold Q; refill for the next tile
        tile_n = _tile_of(j + 1, pid, nprog, SNAKE)
        if tile_n < n_tiles:
            base_n, m_start_n, _n1, _n2, _n3 = _params(tile_n, num_m, n_bh, H, stride_b, stride_h, window, LOCAL, BLOCK_M, BLOCK_N, SCHED)
            cp.async_load(q_smem.index(PART), q_ptrs + base_n + m_start_n * stride_t)
            cp.mbarrier_arrive(qbar.index(PART), increment_count=False)

        m_i = gl.full([HALF_M], float("-inf"), gl.float32, gl.SliceLayout(1, acc_layout))
        l_i = gl.zeros([HALF_M], gl.float32, gl.SliceLayout(1, acc_layout))
        acc = gl.zeros([HALF_M, D], gl.float32, acc_layout)
        rows = m_start + rows0
        row_lo = m_start + PART * HALF_M
        row_hi = row_lo + HALF_M - 1
        for i in range(nb):
            n = n_lo + i
            st = g % STAGES
            if ABL & 8 == 0:
                mbarrier.wait(full.index(st), (g // STAGES) & 1)
            # a block entirely above this group's rows (or entirely outside their window) is skipped; the
            # condition is uniform across the group, so the barriers inside the branch are safe
            compute = (n * BLOCK_N <= row_hi) | (not SKIP_MASKED)
            if LOCAL:
                compute = compute & ((n * BLOCK_N + BLOCK_N - 1 >= row_lo - window) | (not SKIP_MASKED))
            s = gl.zeros([HALF_M, BLOCK_N], gl.float32, acc_layout)
            p = s
            if compute and (ABL & 16 == 0):
                k = k_smem.index(st).permute((1, 0)).load(k_op)
                s = mma_v2(q, k, s)
            if compute:
                cols = n * BLOCK_N + cols0
                need_mask = n * BLOCK_N + BLOCK_N - 1 > row_lo
                if LOCAL:
                    need_mask = need_mask | (n * BLOCK_N < row_hi - window)
                if need_mask and (ABL & 4 == 0):
                    keep = cols[None, :] <= rows[:, None]
                    if LOCAL:
                        keep = keep & (cols[None, :] >= rows[:, None] - window)
                    s = gl.where(keep, s, float("-inf"))
                if ABL & 1:
                    p = s                                         # timing only: P := S, no max, exp, sum or rescale
                else:
                    m_new = gl.maximum(m_i, gl.max(s, axis=1) * scale_log2)
                    m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
                    alpha = gl.exp2(m_i - m_use)
                    p = gl.exp2(s * scale_log2 - m_use[:, None])
                    l_i = l_i * alpha + gl.sum(p, axis=1)
                    if ABL & 2 == 0:
                        acc = acc * alpha[:, None]
                    m_i = m_new
            # producer: block g+STAGES-1 goes into the stage block g-1 used, once both groups released it
            if g >= 1 and (ABL & 8 == 0):
                mbarrier.wait(empty.index((g - 1) % STAGES), ((g - 1) // STAGES) & 1)
            if ptile < n_tiles:
                pst = pg % STAGES
                if ABL & 32 == 0:
                    cp.async_load(k_smem.index(pst).slice(PART * HALF_N, HALF_N, dim=0), k_ptrs + pbase + (pn_lo + pblk) * BLOCK_N * stride_t)
                    cp.async_load(v_smem.index(pst).slice(PART * HALF_N, HALF_N, dim=0), v_ptrs + pbase + (pn_lo + pblk) * BLOCK_N * stride_t)
                cp.mbarrier_arrive(full.index(pst), increment_count=False)
                pg += 1
                pblk += 1
                if pblk == pnb:
                    pj += 1
                    ptile = _tile_of(pj, pid, nprog, SNAKE)
                    pblk = zero
                    if ptile < n_tiles:
                        pbase, _pm, pn_lo, pnb, _pb = _params(ptile, num_m, n_bh, H, stride_b, stride_h, window, LOCAL, BLOCK_M, BLOCK_N, SCHED)
            if compute and (ABL & 16 == 0):
                v = v_smem.index(st).load(v_op)
                acc = mma_v2(gl.convert_layout(p.to(gl.bfloat16), p_op), v, acc)
            mbarrier.arrive(empty.index(st))                      # group barrier, then one arrival: this group is done with the stage
            g += 1

        o = acc / l_i[:, None]
        if ABL & 64 == 0:
            gl.store(o_ptr + base + rows[:, None] * stride_t + od[None, :], o.to(gl.bfloat16))
            gl.store(lse_ptr + bh * T + rows, (m_i + gl.log2(l_i)) * 0.6931471805599453)
        j += 1
        tile = _tile_of(j, pid, nprog, SNAKE)
        if tile < n_tiles:
            base, m_start, n_lo, nb, bh = _params(tile, num_m, n_bh, H, stride_b, stride_h, window, LOCAL, BLOCK_M, BLOCK_N, SCHED)


@gluon.jit
def attn_fwd_ws_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, stride_b, stride_t, stride_h, H, T, n_tiles, scale_log2, window,
                       SNAKE: gl.constexpr, LOCAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr,
                       STAGES: gl.constexpr, SKIP_MASKED: gl.constexpr, SCHED: gl.constexpr, ABL: gl.constexpr, KW: gl.constexpr,
                       num_warps: gl.constexpr):
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_N, D], smem_layout)
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [STAGES, BLOCK_N, D], smem_layout)
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_M // 2, D], smem_layout)
    full = mbarrier.allocate_mbarrier(batch=STAGES)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    qbar = mbarrier.allocate_mbarrier(batch=2)
    for s in gl.static_range(STAGES):
        mbarrier.init(full.index(s), count=2 * 32 * num_warps)   # every thread of both groups arrives through its cp.async
        mbarrier.init(empty.index(s), count=2)                    # one arrival per group
    mbarrier.init(qbar.index(0), count=32 * num_warps)
    mbarrier.init(qbar.index(1), count=32 * num_warps)
    gl.barrier()
    gl.warp_specialize([
        (_attn_partition, (q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, stride_b, stride_t, stride_h, H, T, n_tiles, scale_log2, window,
                           k_smem, v_smem, q_smem, full, empty, qbar, 0, SNAKE, LOCAL, BLOCK_M, BLOCK_N, D, STAGES, SKIP_MASKED, SCHED, ABL, KW, num_warps)),
        (_attn_partition, (q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, stride_b, stride_t, stride_h, H, T, n_tiles, scale_log2, window,
                           k_smem, v_smem, q_smem, full, empty, qbar, 1, SNAKE, LOCAL, BLOCK_M, BLOCK_N, D, STAGES, SKIP_MASKED, SCHED, ABL, KW, num_warps)),
    ], [num_warps])


LAST = {}


def attn_fwd(q, k, v, window=None, sm_scale=None, BLOCK_M=128, BLOCK_N=64, STAGES=2, SKIP_MASKED=True, SCHED="grouped", ABL=0,
             KW=2, num_warps=4, num_ctas=None):
    """q, k, v: [B, T, H, D] bf16 contiguous, causal; window = left window in tokens (None = full causal).
    Returns o [B, T, H, D] bf16 and lse [B, H, T] fp32 (natural log)."""
    B, T, H, D = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and T % BLOCK_M == 0
    o = torch.empty_like(q)
    lse = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    scale = D ** -0.5 if sm_scale is None else sm_scale
    n_tiles = (T // BLOCK_M) * B * H
    grid = min(NUM_SMS if num_ctas is None else num_ctas, n_tiles)
    num_m = T // BLOCK_M
    if SCHED == "grouped":                                        # GH heads per group = two rows of CTAs; needs an even group count
        GH = (2 * grid) // num_m
        ok = GH >= 1 and (2 * grid) % num_m == 0 and (B * H) % GH == 0 and ((B * H) // GH) % 2 == 0
        SCHED = GH if ok else 0
    SNAKE = SCHED == 0 and n_tiles % (2 * grid) == 0
    LAST["h"] = attn_fwd_ws_kernel[(grid,)](q, k, v, o, lse, q.stride(0), q.stride(1), q.stride(2), H, T, n_tiles, scale * LOG2E,
                                            -1 if window is None else window, SNAKE=SNAKE, LOCAL=window is not None,
                                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, STAGES=STAGES, SKIP_MASKED=SKIP_MASKED, SCHED=SCHED,
                                            ABL=ABL, KW=KW, num_warps=num_warps)
    return o, lse


def reference(q, k, v, window=None):
    """fp32 reference: o [B, T, H, D] and the natural-log logsumexp [B, H, T]."""
    B, T, H, D = q.shape
    s = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * D ** -0.5
    t = torch.arange(T, device=q.device)
    keep = t[None, :] <= t[:, None]
    if window is not None:
        keep &= t[None, :] >= t[:, None] - window
    s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    return torch.einsum("bhts,bshd->bthd", p, v.float()), lse
