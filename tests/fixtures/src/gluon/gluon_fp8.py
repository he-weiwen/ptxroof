"""FP8 (e4m3) GEMM for sm_89 in Gluon:  C[M, N] = (A[M, K] @ W[N, K]^T) / (a_scale * b_scale)

Both operands are K-contiguous (nn.Linear layout). Each operand is fed by one of two modes,
chosen at compile time from its dtype:
  e4m3 tensor -> cp.async straight into the shared-memory ring (no ALU work)
  bf16 tensor -> LDG into registers, scale + convert to e4m3, STS into the same ring
Either way the mainloop is: swizzled fp8 tile -> ldmatrix (k_width 4) -> mma.sync m16n8k32 (QMMA).

Accumulation: the tensor core's fp32 accumulate keeps only ~14 bits once the running sum is large (the
same limit DeepSeek-V3 reports on Hopper), so over a long K a small tail is lost: the lm_head grad-input
GEMM (K = 32768, softmax gradients that cancel to a small result) comes out 19% wrong in one pass. With
PROMOTE = n the MMA accumulator is flushed into a separate fp32 register sum every n K tiles, which
restores fp32 quality at the cost of a second accumulator (so an 8-warp tile). _launch turns it on for
K >= PROMOTE_MIN_K.

Epilogues (EPILOGUE constexpr): the dequantized tile can pass through nanochat's MLP activation on its way
to the bf16 store, so relu^2 and its backward never run as separate memory-bound kernels:
  EPI_NONE        C = acc / (a_scale b_scale)
  EPI_RELU2       C = relu(acc / ..)^2                                            (c_fc forward)
  EPI_RELU2_BWD   C = acc / .. * 2 sqrt(aux),  aux = relu(pre)^2 saved by the forward   (c_proj grad-input)
The fused epilogues also reduce max|C| over the tile and atomic-max it into amax_ptr: the tensorwise
scale for whichever FP8 GEMM consumes C next, without another pass over C.

Weight gradients (fp8_linear_dw): dW[N, K] = g^T[N, M] @ a^T[K, M]^T reduces over the M tokens, the strided
dimension of both g and a, and sm_89 has no 8-bit ldmatrix.trans, so the kernel takes token-contiguous e4m3
copies g^T / a^T. Those are written by the GEMMs that already read g and a (WRITE_AT: the in-kernel quantization
pass over a bf16 A also stores its e4m3 tile transposed; of the N-tile CTAs sharing an A tile exactly one writes
each K tile), so no extra pass reads the tensor. The reduction is then the same GEMM with the roles renamed, run
split-K: grid axis 1 slices the tokens, each CTA reduces DW_K_PER_CTA of them (short enough to need no
promotion) and adds its fp32 partial with atomics (EPI_ADD_F32), which fills the GPU for the small [N, K] outputs.

Shapes must be multiples of the tile (nanochat's are). Scales are 0-dim fp32 device tensors or floats.
"""
import os
import torch
import triton
from triton.experimental import gluon

# The ptxas bundled with this Triton (12.9.86) miscompiles, at -O1 and above, a pattern our kernels can produce:
# a .b16 register holding two e4m3 values from cvt.rn.satfinite.e4m3x2 used both by mov.b32 {..} packing for a
# global store and as a .b8 element of a st.shared.v2.b8 (experiments/ptxas_129_e4m3_byte_store_repro.py). The
# CUDA 13.3 ptxas is correct at every level and yields the same kernel speed, so use it when it is installed.
if os.path.exists("/usr/local/cuda/bin/ptxas"):
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp, mma_v2
from triton.language.core import PropagateNan

FP8_MAX = 448.0
CPASYNC = gl.constexpr(0)   # operand already e4m3
QUANT = gl.constexpr(1)     # operand bf16, quantized on the way into shared memory
EPI_NONE = gl.constexpr(0)
EPI_RELU2 = gl.constexpr(1)
EPI_RELU2_BWD = gl.constexpr(2)
EPI_ADD_F32 = gl.constexpr(3)   # split-K partial: fp32 atomic add into a zeroed output, no amax
EPI_ROPE_NORM = gl.constexpr(4) # q / k projection: rotary, RMS norm over the head (= the 128-wide tile), scale; rstd kept
EPI_VGATE = gl.constexpr(5)     # v projection: + 3 sigmoid(x[:, :12] . Wg[head]) * ve, per token and head; gate kept
VE_GATE_CHANNELS = gl.constexpr(12)
NAN_ALL = gl.constexpr(PropagateNan.ALL)   # gl.maximum(.., propagate_nan=NAN_ALL) is arith.maximumf: NaN in, NaN out


@gluon.jit
def _quant(x, scale):
    # Direct, correctly rounded f32 -> e4m3 with saturation. Triton's own cast on sm_89 detours
    # through cvt.rz.f16.f32 and double-rounds ~1.4% of elements (see triton_sm89_fp8_cvt.patch).
    # Four elements per call into one 32-bit register: the 16-bit halves never leave the asm, which keeps
    # ptxas 12.9 from merging two of them and mis-tracking their byte uses (triton_ptxas129_fp8_widen.patch).
    xf = x.to(gl.float32) * scale
    return gl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
        "=r,r,r,r,r", [xf], dtype=gl.float8e4nv, is_pure=True, pack=4)


@gluon.jit
def _tile_coords(M, N, BM: gl.constexpr, BN: gl.constexpr, GROUP_M: gl.constexpr):
    # grouped ordering: GROUP_M row-blocks of A stay L2-resident while the N tiles stream by
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)
    group_size = GROUP_M * num_pid_n
    first_pid_m = (pid // group_size) * GROUP_M
    this_group = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size) % this_group
    pid_n = (pid % group_size) // this_group
    return pid_m, pid_n, num_pid_n


@gluon.jit
def _operand_ptrs(base, row0, stride_row, ROWS: gl.constexpr, BK: gl.constexpr, layout: gl.constexpr):
    rows = row0 + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    ks = gl.arange(0, BK, layout=gl.SliceLayout(0, layout))
    return base + rows[:, None] * stride_row + ks[None, :]

"""
    pointers:
        a, w, c: 
        aux  : used in relu2bwd epilogue: relu2 value saved in fwd
        at   : address for WRITE_AT
        amax : max|c| -> feed to the scale of the next user if needed
        a_scale, b_scale
    shapes & strides:
        M, N, K,
        stride_am, stride_wn, stride_cm, stride_at
    tile geometry:
        BM, BN, BK,
        STAGES (ring buffer size for pipelining)
        GROUP_M: allocate pid order to improve l2 reuse.
    operand modes:
        = CPASYNC(already in e4m3: cp.async global->shared) 
        | QUANT(in bf16: load(global->reg)->quant->store(reg->shared))

        forward:  x (bf16) @ W (e4m3)     -> (quant,   cpasync)
        backward: g^T (e4m3) @ a^T (e4m3) -> (cpasync, cpasync)
    others:
        PROMOTE:  period for add acc(accumulates in fp16) to total in fp32, then flush acc to 0
        EPILOGUE:
            EPI_NONE:      acc / (as * bs)
            EPI_RELU2:     relu(that)^2 to bf16, calculate & store amax
            EPI_RELU2_BWD: that * 2 * sqrt(aux) to bf16, calculate & store amax
            EPI_ADD_F32:   fp32 atromic add into a zeroed C, no amax (split-K weight gradient)
        WRITE_AT: store (quantized) A transposed
"""
@gluon.jit
def fp8_gemm_kernel(a_ptr, w_ptr, c_ptr, aux_ptr, at_ptr, amax_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
                    stride_am, stride_wn, stride_cm, stride_at,
                    p0_ptr, p1_ptr, side_ptr, gx_ptr, stride_gx, seq_len, eps, out_scale,
                    BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, STAGES: gl.constexpr,
                    GROUP_M: gl.constexpr, WARPS_M: gl.constexpr, A_MODE: gl.constexpr, B_MODE: gl.constexpr,
                    PROMOTE: gl.constexpr, EPILOGUE: gl.constexpr, WRITE_AT: gl.constexpr, GX_RSTD: gl.constexpr,
                    num_warps: gl.constexpr):
    # K is the reduction length of one CTA; with split-K (grid axis 1) this CTA owns the slice starting at k0
    pid_m, pid_n, num_pid_n = _tile_coords(M, N, BM, BN, GROUP_M)
    k0 = gl.program_id(1) * K

    # ---- layouts, all derived from the tile
    TPW_K: gl.constexpr = BK // 16                                   # 16 elements per thread along K
    ld_layout: gl.constexpr = gl.BlockedLayout([1, 16], [32 // TPW_K, TPW_K], [num_warps, 1], [1, 0])
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, num_warps // WARPS_M], instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=4)   # 32 bits / 8-bit elements
    b_op: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=4)
    PER_PHASE: gl.constexpr = 128 // BK if BK < 128 else 1            # bank-conflict-free ldmatrix for BK-byte rows
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=PER_PHASE, max_phase=8 // PER_PHASE, order=[1, 0])
    st_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])

    a_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BM, BK], layout=smem_layout)
    w_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, BK], layout=smem_layout)
    a_ptrs = _operand_ptrs(a_ptr, pid_m * BM, stride_am, BM, BK, ld_layout) + k0
    w_ptrs = _operand_ptrs(w_ptr, pid_n * BN, stride_wn, BN, BK, ld_layout) + k0
    if WRITE_AT:
        # A^T[k, m], token-contiguous: the same registers that feed the ring, stored with the axes swapped
        at_rows = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, ld_layout))
        at_ks = k0 + gl.arange(0, BK, layout=gl.SliceLayout(0, ld_layout))
        at_ptrs = at_ptr + at_ks[None, :] * stride_at + at_rows[:, None]
    a_scale = gl.load(a_scale_ptr)
    b_scale = gl.load(b_scale_ptr)
    acc = gl.zeros([BM, BN], gl.float32, acc_layout)
    if PROMOTE > 0:
        total = gl.zeros([BM, BN], gl.float32, acc_layout)   # fp32 sum of the flushed MMA accumulators
    n_k = gl.cdiv(K, BK)

    # ---- prologue: cp.async operands fill STAGES-1 slots; register operands hold tile 0
    for s in gl.static_range(STAGES - 1):
        if A_MODE == CPASYNC:
            cp.async_load(a_smem.index(s), a_ptrs + s * BK)
        if B_MODE == CPASYNC:
            cp.async_load(w_smem.index(s), w_ptrs + s * BK)
        cp.commit_group()
    if A_MODE == QUANT:
        a_reg = gl.load(a_ptrs)
    if B_MODE == QUANT:
        w_reg = gl.load(w_ptrs)

    for i in range(n_k):
        slot = i % STAGES
        cp.wait_group(STAGES - 2)
        # register operands: quantize tile i into the ring, then prefetch tile i+1 behind the MMAs
        if A_MODE == QUANT:
            a_q = _quant(a_reg, a_scale)
            a_smem.index(slot).store(a_q)
            if WRITE_AT:
                if i % num_pid_n == pid_n:      # of the N-tile CTAs sharing this A tile, exactly one writes K tile i
                    gl.store(at_ptrs + i * BK * stride_at, a_q)
            a_reg = gl.load(a_ptrs + (i + 1) * BK, mask=(i + 1) < n_k, other=0.0)
        if B_MODE == QUANT:
            w_smem.index(slot).store(_quant(w_reg, b_scale))
            w_reg = gl.load(w_ptrs + (i + 1) * BK, mask=(i + 1) < n_k, other=0.0)
        gl.barrier()
        a = a_smem.index(slot).load(a_op)
        b = w_smem.index(slot).permute((1, 0)).load(b_op)
        acc = mma_v2(a, b, acc)
        if PROMOTE > 0:
            if (i + 1) % PROMOTE == 0:                          # flush before the running sum swamps the products
                total += acc
                acc = gl.zeros([BM, BN], gl.float32, acc_layout)
        # cp.async operands: refill the slot consumed last iteration; always commit so wait_group counts stay right
        nxt = i + STAGES - 1
        if nxt < n_k:
            if A_MODE == CPASYNC:
                cp.async_load(a_smem.index(nxt % STAGES), a_ptrs + nxt * BK)
            if B_MODE == CPASYNC:
                cp.async_load(w_smem.index(nxt % STAGES), w_ptrs + nxt * BK)
        cp.commit_group()
    cp.wait_group(0)
    if PROMOTE > 0:
        acc += total

    if EPILOGUE == EPI_ADD_F32:
        # ---- split-K partial: fp32 atomics into the zeroed output; one warp instruction covers one 128-byte row segment
        red_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])
        out = gl.convert_layout(acc * (1.0 / (a_scale * b_scale)), red_layout)
        rows = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, red_layout))
        cols = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, red_layout))
        gl.atomic_add(c_ptr + rows[:, None] * stride_cm + cols[None, :], out, sem="relaxed")
    else:
        # ---- epilogue: dequantize, fused activation, bf16 store, amax of the stored tile
        out = gl.convert_layout(acc * (1.0 / (a_scale * b_scale)), st_layout)
        rows = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, st_layout))
        cols = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, st_layout))
        offs = rows[:, None] * stride_cm + cols[None, :]
        if EPILOGUE == EPI_RELU2:
            out = gl.maximum(out, 0.0, propagate_nan=NAN_ALL)   # max.NaN.f32: a NaN stays a NaN, as in torch.relu
            out = out * out
        elif EPILOGUE == EPI_RELU2_BWD:
            # aux = relu(pre)^2 from the forward, so d relu(pre)^2 / d pre = 2 relu(pre) = 2 sqrt(aux)
            out = out * (2.0 * gl.sqrt(gl.load(aux_ptr + offs).to(gl.float32)))
        elif EPILOGUE == EPI_ROPE_NORM:
            # nanochat's attention glue on one head: rotary over the two halves of the tile, then RMS norm over the
            # head and the 1.2 scale. Each thread holds both halves of its columns, so the partner element is a
            # register permutation: split the halves, join them swapped.
            gl.static_assert(BN == 128, "EPI_ROPE_NORM: a tile must be one head of 128")
            HALF: gl.constexpr = BN // 2
            x1, x2 = gl.split(gl.permute(gl.reshape(out, [BM, 2, HALF]), (0, 2, 1)))
            partner = gl.convert_layout(gl.reshape(gl.permute(gl.join(x2, x1), (0, 2, 1)), [BM, BN]), st_layout)
            pos = rows % seq_len                                         # token position within its sequence
            local = cols - pid_n * BN
            tab = pos[:, None] * HALF + (local % HALF)[None, :]           # cos / sin tables are [T, HALF]
            cs = gl.load(p0_ptr + tab).to(gl.float32)
            sn = gl.load(p1_ptr + tab).to(gl.float32)
            sn = gl.where(local[None, :] < HALF, sn, -sn)                # y1 = x1 cos + x2 sin;  y2 = x2 cos - x1 sin
            out = out * cs + partner * sn
            rstd = gl.rsqrt(gl.sum(out * out, axis=1) / BN + eps)
            out = out * (rstd * out_scale)[:, None]
            gl.store(side_ptr + rows * num_pid_n + pid_n, rstd)
        elif EPILOGUE == EPI_VGATE:
            # value residual: v += 3 sigmoid(y[:, :12] . Wg[head]) * ve, y the normalized bf16 input: gx_ptr holds either y
            # itself or the residual x_new with p1_ptr its rstd (when A is the e4m3 copy of y); aux is ve
            gate = gl.zeros([BM], gl.float32, gl.SliceLayout(1, st_layout))
            if GX_RSTD:
                gr = gl.load(p1_ptr + rows)
            for j in gl.static_range(VE_GATE_CHANNELS):
                xj = gl.load(gx_ptr + rows * stride_gx + j).to(gl.float32)
                if GX_RSTD:
                    xj = (xj * gr).to(gl.bfloat16).to(gl.float32)          # y = bf16(x_new rstd), as the bf16 path computes it
                gate += xj * gl.load(p0_ptr + pid_n * VE_GATE_CHANNELS + j).to(gl.float32)
            gate = 3.0 / (1.0 + gl.exp(-gate))
            out = out + gate[:, None] * gl.load(aux_ptr + offs).to(gl.float32)
            gl.store(side_ptr + rows * num_pid_n + pid_n, gate)
        out = out.to(gl.bfloat16)
        gl.store(c_ptr + offs, out)
        if EPILOGUE == EPI_RELU2 or EPILOGUE == EPI_RELU2_BWD:
            # max|C| of the values actually stored, reduced and atomic-maxed as int32: non-negative fp32 bit
            # patterns order like int32, and inf / NaN patterns sort above every finite value, so the result is
            # exact and a NaN in the tile makes the amax NaN, as torch's amax would
            bits = gl.abs(out.to(gl.float32)).to(gl.int32, bitcast=True)
            gl.atomic_max(amax_ptr, gl.max(gl.max(bits, axis=1), axis=0))


# tile configs: (BM, BN, BK, STAGES, num_warps, WARPS_M), measured on the 4090 at nanochat shapes
CFG_FP8_FP8 = (128, 128, 64, 3, 4, 2)     # both pre-quantized: ~310 TFLOPS at c_fc (cuBLASLt 266)
CFG_BF16_FP8 = (128, 128, 64, 3, 4, 2)    # bf16 activation quantized in-kernel, fp8 weight via cp.async: ~265 TFLOPS
CFG_PROMOTE = (128, 128, 64, 3, 8, 4)     # two accumulators need 8 warps (64 elements/thread each)
CFG_EPILOGUE = {EPI_RELU2.value: CFG_BF16_FP8, EPI_RELU2_BWD.value: CFG_BF16_FP8,   # 128x128 wins for both (check_epilogues)
                EPI_ROPE_NORM.value: CFG_BF16_FP8, EPI_VGATE.value: CFG_BF16_FP8}    # BN = 128 = one head
PROMOTE_TILES = 1                         # flush every K tile: most accurate, and no slower than every 2 or 4
PROMOTE_MIN_K = 8192                      # nanochat's only long-K GEMM is the lm_head grad-input (K = vocab)
DW_K_PER_CTA = 4096                       # tokens per CTA in the split-K weight-gradient GEMM (see fp8_linear_dw) ...
DW_K_PER_CTA_SMALL = 2048                 # ... and for outputs of fewer than DW_SMALL_TILES tiles, which need more CTAs
DW_SMALL_TILES = 64


def _scale_tensor(s, device):
    if isinstance(s, torch.Tensor):
        return s.to(torch.float32).reshape(())
    return torch.full((), float(s), dtype=torch.float32, device=device)


_dummy_amax = {}


def _launch(a, w, a_scale, b_scale, cfg, promote, epilogue, aux, split=1, write_at=False, p0=None, p1=None, seq_len=0, eps=0.0, out_scale=1.0,
            gate_x=None, gate_rstd=None, out=None):
    M, K = a.shape
    N, K2 = w.shape
    assert K == K2 and a.stride(1) == 1 and w.stride(1) == 1
    epilogue = int(getattr(epilogue, "value", epilogue))
    modes = {torch.float8_e4m3fn: 0, torch.bfloat16: 1}
    a_mode, b_mode = modes[a.dtype], modes[w.dtype]
    assert K % split == 0, (K, split)
    k_len = K // split                                    # the reduction length of one CTA
    if promote is None:
        promote = PROMOTE_TILES if k_len >= PROMOTE_MIN_K else 0
    if cfg is None:
        if promote:
            cfg = CFG_PROMOTE
        elif epilogue in CFG_EPILOGUE:
            cfg = CFG_EPILOGUE[epilogue]
        else:
            cfg = CFG_FP8_FP8 if (a_mode == 0 and b_mode == 0) else CFG_BF16_FP8
    BM, BN, BK, STAGES, num_warps, WARPS_M = cfg
    assert M % BM == 0 and N % BN == 0 and k_len % BK == 0, f"shape {(M, N, K)} / split {split} is not a multiple of the tile {cfg}"
    if epilogue == EPI_ADD_F32.value:
        if out is None:
            c = torch.zeros((M, N), device=a.device, dtype=torch.float32)   # the split-K partials are added into it
        else:
            assert out.shape == (M, N) and out.dtype == torch.float32 and out.stride(1) == 1, "out: fp32 [M, N], rows contiguous"
            c = out                                                       # accumulate into the caller's buffer
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    if epilogue in (EPI_NONE.value, EPI_ADD_F32.value):
        amax = _dummy_amax.get(a.device)
        if amax is None:
            amax = _dummy_amax[a.device] = torch.zeros((), device=a.device, dtype=torch.int32)
    else:
        amax = torch.zeros((), device=a.device, dtype=torch.int32)     # fp32 bits of max|C|, +0.0 == 0
    if aux is None:
        aux = c
    else:
        assert aux.shape == c.shape and aux.stride() == c.stride() and aux.dtype == torch.bfloat16
    if write_at:
        assert a_mode == 1, "A^T is written by the in-kernel quantization of a bf16 A"   # with split-K each split writes its K range
        at = torch.empty((K, M), device=a.device, dtype=torch.float8_e4m3fn)
    else:
        at = c
    if epilogue in (EPI_ROPE_NORM.value, EPI_VGATE.value):
        assert BN == 128 and p0 is not None and p0.is_contiguous(), "one tile per head; cos / sin or the gate weight required"
        assert epilogue != EPI_VGATE.value or a_mode == 1 or gate_x is not None, "the gate needs the bf16 normalized input"
        side = torch.empty((M, N // BN), device=a.device, dtype=torch.float32)   # rstd or gate per (token, head)
    else:
        side = c
    if gate_x is None:
        gate_x = a                                             # the bf16 A operand is the normalized input itself
    if gate_rstd is not None:
        assert gate_rstd.shape == (M,) and gate_rstd.dtype == torch.float32
        p1 = gate_rstd
    p0 = c if p0 is None else p0
    p1 = c if p1 is None else p1
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), split)
    fp8_gemm_kernel[grid](a, w, c, aux, at, amax, _scale_tensor(a_scale, a.device), _scale_tensor(b_scale, a.device), M, N, k_len,
                          a.stride(0), w.stride(0), c.stride(0), at.stride(0), p0, p1, side, gate_x, gate_x.stride(0), seq_len, float(eps), float(out_scale),
                          BM=BM, BN=BN, BK=BK, STAGES=STAGES, GROUP_M=8, WARPS_M=WARPS_M,
                          A_MODE=a_mode, B_MODE=b_mode, PROMOTE=promote, EPILOGUE=epilogue, WRITE_AT=write_at,
                          GX_RSTD=gate_rstd is not None, num_warps=num_warps)
    return c, amax.view(torch.float32), (at if write_at else None), side


def fp8_gemm(a, w, a_scale, b_scale, epilogue=EPI_NONE, aux=None, write_at=False, cfg=None, promote=None):
    """The general entry: C = A @ W^T (bf16 [M, N]) with an optional fused epilogue, its amax, and with write_at
    the e4m3 copy of A transposed to [K, M] for the weight-gradient GEMM, written by the same quantization pass
    that feeds the tensor core (A must be bf16). Returns (C, amax, A^T or None)."""
    return _launch(a, w, a_scale, b_scale, cfg, promote, epilogue, aux, write_at=write_at)[:3]


def fp8_gemm_rope_norm(a, w, a_scale, b_scale, cos, sin, seq_len, eps, out_scale, write_at=False):
    """q or k projection with nanochat's attention glue fused: C = rope(a @ w^T) per head, RMS-normalized over the head
    (eps as F.rms_norm on bf16: 2^-7) and scaled. cos, sin: [seq_len, 64] bf16. Returns (C bf16 [M, N], A^T or None,
    rstd fp32 [M, heads]), rstd being what the backward needs."""
    c, _, at, rstd = _launch(a, w, a_scale, b_scale, None, None, EPI_ROPE_NORM, None, write_at=write_at,
                             p0=cos, p1=sin, seq_len=seq_len, eps=eps, out_scale=out_scale)
    return c, at, rstd


def fp8_gemm_vgate(a, w, a_scale, b_scale, ve, gate_weight, write_at=False, gate_x=None, gate_rstd=None):
    """v projection with the value residual fused: C = a @ w^T + 3 sigmoid(y[:, :12] . gate_weight[head]) * ve, y the bf16
    normalized input: a itself when a is bf16, else gate_x (with gate_rstd: the residual x_new and its rstd, y = bf16(x_new rstd)).
    ve: bf16 [M, N] like C; gate_weight: [heads, 12] bf16. Returns (C, A^T or None, gate fp32 [M, heads])."""
    c, _, at, gate = _launch(a, w, a_scale, b_scale, None, None, EPI_VGATE, ve, write_at=write_at, p0=gate_weight,
                             gate_x=gate_x, gate_rstd=gate_rstd)
    return c, at, gate


def fp8_linear(a, w, a_scale, b_scale, cfg=None, promote=None):
    """a: [M, K], w: [N, K], each bf16 or float8_e4m3fn, K-contiguous. Returns bf16 [M, N].
    promote: K tiles per accumulator flush (0 = one MMA accumulator for all of K; None = PROMOTE_TILES for
    K >= PROMOTE_MIN_K, else 0)."""
    return _launch(a, w, a_scale, b_scale, cfg, promote, EPI_NONE, None)[0]


def fp8_linear_epilogue(a, w, a_scale, b_scale, epilogue, aux=None, cfg=None, promote=None):
    """fp8_linear with a fused EPI_RELU2 or EPI_RELU2_BWD epilogue (aux = the forward's relu^2 output for
    the latter). Returns (C bf16 [M, N], max|C| as a 0-dim fp32 device tensor)."""
    assert epilogue != EPI_NONE and (aux is not None) == (epilogue == EPI_RELU2_BWD)
    return _launch(a, w, a_scale, b_scale, cfg, promote, epilogue, aux)[:2]


def fp8_linear_dw(gt, at, g_scale, a_scale, cfg=None, split=None, out=None):
    """Weight gradient dW[N, K] = g^T[N, M] @ (a^T[K, M])^T, reducing over the M tokens: the same kernel with the
    token-contiguous e4m3 copies as operands. split-K: `split` CTAs per output tile (default M / DW_K_PER_CTA)
    each reduce their slice and add fp32 partials with atomics, so the bits can differ run to run. Returns fp32, or
    accumulates into `out` (fp32 [N, K]) when given."""
    if split is None:
        tiles = (gt.shape[0] // 128) * (at.shape[0] // 128)
        split = max(1, gt.shape[1] // (DW_K_PER_CTA_SMALL if tiles < DW_SMALL_TILES else DW_K_PER_CTA))
    return _launch(gt, at, g_scale, a_scale, cfg, None, EPI_ADD_F32, None, split=split, out=out)[0]


def scale_from_amax(amax, target=FP8_MAX):
    return target / amax.clamp(min=1e-12)


def scale_of(t, target=FP8_MAX):
    """Tensorwise dynamic scale mapping amax to `target`, as a 0-dim device tensor (no host sync)."""
    return scale_from_amax(t.float().abs().amax(), target)


def quantize(t, s):
    return (t.float() * s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def check_epilogues(M=32768, D=768, verbose=True):
    """Correctness + timing of the fused relu^2 epilogues against the unfused GEMM + torch.compile kernels,
    at nanochat's d12 MLP shapes (c_fc [M, 4D] forward, c_proj grad-input [M, 4D] backward)."""
    from triton.testing import do_bench
    torch.manual_seed(0)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    res = {}
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(4 * D, D, device="cuda", dtype=torch.bfloat16) * 0.05
    xs, w1s = scale_of(x), scale_of(w1)
    w1q = quantize(w1, w1s)
    pre_exact = (quantize(x, xs).float() @ w1q.float().t()) / (xs * w1s)
    h_exact = torch.relu(pre_exact).square()
    relu2 = torch.compile(lambda p: (torch.relu(p).square(), torch.relu(p).square().abs().amax()))
    h_unfused, amax_unfused = relu2(fp8_linear(x, w1q, xs, w1s))
    ms_gemm = do_bench(lambda: fp8_linear(x, w1q, xs, w1s))
    ms_unfused = do_bench(lambda: relu2(fp8_linear(x, w1q, xs, w1s)))
    if verbose:
        print(f"c_fc forward + relu^2, {M}x{4*D}x{D}: unfused GEMM {ms_gemm:.3f} ms, GEMM + relu^2/amax kernel {ms_unfused:.3f} ms")
    for cfg in [(128, 128, 64, 3, 4, 2), (128, 64, 64, 3, 4, 4), (128, 64, 64, 4, 4, 4), (64, 128, 64, 3, 4, 2)]:
        h, amax = fp8_linear_epilogue(x, w1q, xs, w1s, EPI_RELU2, cfg=cfg)
        ms = do_bench(lambda: fp8_linear_epilogue(x, w1q, xs, w1s, EPI_RELU2, cfg=cfg))
        ok = torch.equal(h, h_unfused) or rel(h, h_unfused) < 1e-6
        res[f"relu2 {cfg}"] = dict(ms=ms, ms_unfused=ms_unfused, relerr=rel(h, h_exact), amax_ok=bool(amax == amax_unfused))
        if verbose:
            print(f"  fused EPI_RELU2 {str(cfg):24s} {ms:.3f} ms  err vs exact {rel(h, h_exact):.1e} (unfused {rel(h_unfused, h_exact):.1e})"
                  f"  bitwise == unfused: {ok}  amax {amax.item():.4g} == {amax_unfused.item():.4g}: {bool(amax == amax_unfused)}")
    # backward: grad_h = grad_y @ W2 then * 2 relu(pre); the fused kernel uses 2 sqrt(h) from the saved h
    h = h_unfused
    gy = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(D, 4 * D, device="cuda", dtype=torch.bfloat16) * 0.05
    gys, w2s = scale_of(gy), scale_of(w2)
    w2tq = quantize(w2.t().contiguous(), w2s)
    gh_exact = (quantize(gy, gys).float() @ w2tq.float().t()) / (gys * w2s)
    gp_exact = gh_exact * 2 * torch.relu(pre_exact)
    relu2_bwd = torch.compile(lambda g, p: ((g * 2 * torch.relu(p)).to(torch.bfloat16), (g * 2 * torch.relu(p)).abs().amax()))
    pre_saved = fp8_linear(x, w1q, xs, w1s)                                  # what the unfused path keeps for backward
    gp_unfused, _ = relu2_bwd(fp8_linear(gy, w2tq, gys, w2s), pre_saved)
    ms_gemm = do_bench(lambda: fp8_linear(gy, w2tq, gys, w2s))
    ms_unfused = do_bench(lambda: relu2_bwd(fp8_linear(gy, w2tq, gys, w2s), pre_saved))
    if verbose:
        print(f"c_proj grad-input + relu^2 backward, {M}x{4*D}x{D}: unfused GEMM {ms_gemm:.3f} ms, GEMM + backward/amax kernel {ms_unfused:.3f} ms")
    for cfg in [(128, 128, 64, 3, 4, 2), (128, 64, 64, 3, 4, 4), (128, 64, 64, 4, 4, 4), (64, 128, 64, 3, 4, 2)]:
        gp, amax = fp8_linear_epilogue(gy, w2tq, gys, w2s, EPI_RELU2_BWD, aux=h, cfg=cfg)
        ms = do_bench(lambda: fp8_linear_epilogue(gy, w2tq, gys, w2s, EPI_RELU2_BWD, aux=h, cfg=cfg))
        res[f"relu2_bwd {cfg}"] = dict(ms=ms, ms_unfused=ms_unfused, relerr=rel(gp, gp_exact), relerr_unfused=rel(gp_unfused, gp_exact),
                                       amax_relerr=abs(amax.item() - gp.abs().amax().item()) / gp.abs().amax().item())
        if verbose:
            print(f"  fused EPI_RELU2_BWD {str(cfg):24s} {ms:.3f} ms  err vs exact {rel(gp, gp_exact):.1e} (unfused {rel(gp_unfused, gp_exact):.1e})"
                  f"  amax {amax.item():.4g} vs {gp.abs().amax().item():.4g}")
    return res


def check_weight_gradient(M=32768, verbose=True):
    """The weight-gradient path at nanochat's d12 shapes: cuBLAS bf16 (the previous path) against the FP8 split-K
    kernel over e4m3 transposes, and what writing those transposes costs: the GEMM's side output (write_at)
    against a standalone compiled quantize + transpose pass."""
    from triton.testing import do_bench
    torch.manual_seed(0)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    res = {}
    for name, N, K in [("c_fc / c_proj  dW [3072, 768]", 3072, 768), ("attention  dW [768, 768]", 768, 768)]:
        g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.05
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        gs, as_ = scale_of(g), scale_of(a)
        gt, at = quantize(g, gs).t().contiguous(), quantize(a, as_).t().contiguous()
        ref = (gt.float() @ at.float().t()) / (gs * as_)
        fl = 2 * M * N * K
        ms = do_bench(lambda: g.t() @ a)
        res[f"{name} cuBLAS bf16"] = dict(ms=ms, tflops=fl / ms / 1e9)
        if verbose:
            print(f"{name}, {M} tokens")
            print(f"  {'cuBLAS bf16 (g^T @ a)':44s} {ms:.3f} ms {fl/ms/1e9:6.1f} TF")
        for split in (1, 2, 4, 8, 16):
            dw = fp8_linear_dw(gt, at, gs, as_, split=split)
            ms = do_bench(lambda: fp8_linear_dw(gt, at, gs, as_, split=split))
            res[f"{name} fp8 split {split}"] = dict(ms=ms, tflops=fl / ms / 1e9, relerr=rel(dw, ref))
            if verbose:
                print(f"  {'fp8 split-K %2d (%5d tokens per CTA)' % (split, M // split):44s} {ms:.3f} ms {fl/ms/1e9:6.1f} TF  relerr {rel(dw, ref):.1e}")
    qt = torch.compile(lambda t, s: quantize(t, s).t().contiguous(), dynamic=False)
    for name, N, K in [("c_fc forward (writes x^T [768, M])", 3072, 768), ("c_proj forward (writes h^T [3072, M])", 768, 3072),
                       ("attention forward / grad-input (writes x^T [768, M])", 768, 768)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w8 = quantize(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05, 1.0)
        xs = scale_of(x)
        ms0 = do_bench(lambda: fp8_gemm(x, w8, xs, 1.0))
        ms1 = do_bench(lambda: fp8_gemm(x, w8, xs, 1.0, write_at=True))
        ms2 = do_bench(lambda: qt(x, xs))
        res[f"{name}"] = dict(ms_gemm=ms0, ms_gemm_write_at=ms1, ms_standalone_copy=ms2)
        if verbose:
            print(f"{name}: GEMM {ms0:.3f} ms, GEMM + A^T side output {ms1:.3f} ms (+{ms1-ms0:.3f}); standalone quantize+transpose {ms2:.3f} ms")
    return res


if __name__ == "__main__":
    import json, sys
    from triton.testing import do_bench
    if "--dw" in sys.argv:
        json.dump(check_weight_gradient(), open("results/gluon_fp8_dw.json", "w"), indent=1)
        sys.exit(0)
    if "--sweep" not in sys.argv:
        json.dump(check_epilogues(), open("results/gluon_fp8_epilogue.json", "w"), indent=1)
        sys.exit(0)
    torch.manual_seed(0)
    M, K, N = 32768, 768, 3072
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    xs, ws = scale_of(x), scale_of(w)
    x8, w8 = quantize(x, xs), quantize(w, ws)
    ref = torch._scaled_mm(x8, w8.t(), scale_a=1 / xs, scale_b=1 / ws, out_dtype=torch.bfloat16)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    fl = 2 * M * N * K
    res = {}
    print(f"c_fc forward {M}x{N}x{K}")
    print(f"  {'cuBLAS bf16':58s} {do_bench(lambda: x @ w.t()):.3f} ms")
    ms = do_bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=1 / xs, scale_b=1 / ws, out_dtype=torch.bfloat16))
    print(f"  {'cuBLASLt fp8, pre-quantized':58s} {ms:.3f} ms {fl/ms/1e9:6.1f} TF")
    cfgs = [(128, 128, 64, 3, 4, 2), (128, 64, 64, 4, 4, 4), (128, 128, 32, 4, 4, 2), (128, 128, 64, 3, 8, 4), (128, 256, 64, 3, 8, 4), (128, 128, 64, 2, 4, 2)]
    for label, A, W in [("fp8 x fp8 (cp.async both)", x8, w8), ("bf16 x fp8 (quantize A in kernel)", x, w8), ("bf16 x bf16 (quantize both)", x, w)]:
        for cfg in cfgs:
            try:
                c = fp8_linear(A, W, xs, ws, cfg)
                err = rel(c, ref)
                ms = do_bench(lambda: fp8_linear(A, W, xs, ws, cfg))
                res[f"{label} {cfg}"] = dict(ms=ms, tflops=fl / ms / 1e9, relerr=err)
                print(f"  {label:34s} {str(cfg):24s} {ms:.3f} ms {fl/ms/1e9:6.1f} TF relerr {err:.1e}")
            except Exception as e:
                print(f"  {label:34s} {str(cfg):24s} failed: {type(e).__name__} {str(e)[-120:]}")
    V, D = 32768, 768
    print(f"lm_head grad-input {M}x{D}x{V} (K = vocab): accumulator promotion, 8-warp tile")
    g = torch.randn(M, V, device="cuda", dtype=torch.bfloat16)          # grad_out [tokens, vocab]
    wt8 = quantize(torch.randn(D, V, device="cuda", dtype=torch.bfloat16), ws)   # W^T [dim, vocab]
    gs = scale_of(g)
    fl2 = 2 * M * D * V
    for label, promote, cfg in [("one accumulator (19% error on real logits gradients)", 0, CFG_BF16_FP8), ("8 warps, one accumulator", 0, CFG_PROMOTE),
                                ("promote every 4 tiles", 4, CFG_PROMOTE), ("promote every 2 tiles", 2, CFG_PROMOTE), ("promote every tile", 1, CFG_PROMOTE)]:
        ms = do_bench(lambda: fp8_linear(g, wt8, gs, ws, cfg=cfg, promote=promote))
        res[f"lm_head dgrad {label}"] = dict(ms=ms, tflops=fl2 / ms / 1e9)
        print(f"  {label:54s} {ms:.3f} ms {fl2/ms/1e9:6.1f} TF")
    json.dump(res, open("results/gluon_fp8.json", "w"), indent=1)
