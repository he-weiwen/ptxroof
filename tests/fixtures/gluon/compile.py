#!/usr/bin/env python3
"""Run each nanochat Gluon entry point once, at a nanochat shape, with a fresh Triton cache.

Usage: compile.py SRC_DIR OUT_DIR. Writes OUT_DIR/<kernel>[.<site>].ptx per launch and
prints one "name<TAB>module<TAB>wrapper" line per file for regen.sh's provenance header.
"""
import glob
import os
import sys
import tempfile

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.environ["TRITON_CACHE_DIR"] = cache = tempfile.mkdtemp(prefix="ptxroof-triton-")
sys.path.insert(0, src_dir)
import torch  # noqa: E402
import gluon_attn_bwd, gluon_attn_fwd, gluon_ce, gluon_fp8, gluon_norm, gluon_rope_norm  # noqa: E401,E402

torch.manual_seed(0)
dev = "cuda"
M, K, N = 4096, 768, 3072  # c_fc at an eighth of the tokens (tests/test_gluon_fp8.py)
C = 768  # model dim
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
xs, ws = gluon_fp8.scale_of(x), gluon_fp8.scale_of(w)
g = torch.randn(2048, 32768, device=dev, dtype=torch.bfloat16)  # lm_head grad-input: K = vocab
wt = torch.randn(C, 32768, device=dev, dtype=torch.bfloat16)
gs, wts = gluon_fp8.scale_of(g), gluon_fp8.scale_of(wt)
xr, z, x0 = (torch.randn(M, C, device=dev, dtype=torch.bfloat16) for _ in range(3))
T, H, D = 2048, 6, 128
xq = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
ang = torch.randn(1, T, 1, D // 2, device=dev)
cos, sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
B = 2  # attention at nanochat's T, H and D for two sequences
q, k, v = (torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
do = torch.randn_like(q)


def ce(V):
    logits = torch.randn(64, V, device=dev, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (64,), device=dev)
    return gluon_ce.ce_chunk(logits, targets, torch.full((), 1.0 / 64, device=dev))


launches = [
    ("c_fc", "gluon_fp8.py", "fp8_linear(e4m3 A [4096, 768], e4m3 W [3072, 768])",
     lambda: gluon_fp8.fp8_linear(gluon_fp8.quantize(x, xs), gluon_fp8.quantize(w, ws), xs, ws)),
    ("lm_head_dx", "gluon_fp8.py", "fp8_linear(bf16 g [2048, 32768], e4m3 W^T [768, 32768]): promoted, 8 warps",
     lambda: gluon_fp8.fp8_linear(g, gluon_fp8.quantize(wt, wts), gs, wts)),
    ("", "gluon_ce.py", "ce_chunk(bf16 [64, 32768]): V=32768, BLOCK=4096", lambda: ce(32768)),
    ("v8192", "gluon_ce.py", "ce_chunk(bf16 [64, 8192]): V=8192, BLOCK=4096, a two-trip loop", lambda: ce(8192)),
    ("", "gluon_norm.py", "rmsnorm_fp8(x, z, x0), bf16 [4096, 768]: both passes",
     lambda: gluon_norm.rmsnorm_fp8(xr, z, x0)),
    ("", "gluon_rope_norm.py", "gluon_rope_norm(bf16 [1, 2048, 6, 128])",
     lambda: gluon_rope_norm.gluon_rope_norm(xq, cos, sin)),
    ("", "gluon_attn_fwd.py", "attn_fwd(bf16 [2, 2048, 6, 128]): causal; persistent, two 4-warp partitions, BLOCK_M=128, BLOCK_N=64",
     lambda: gluon_attn_fwd.attn_fwd(q, k, v)),
    ("w768", "gluon_attn_fwd.py", "attn_fwd(bf16 [2, 2048, 6, 128], window=768)",
     lambda: gluon_attn_fwd.attn_fwd(q, k, v, 768)),
    ("", "gluon_attn_bwd.py", "attn_bwd(q, k, v, o, do, lse) causal, BLOCK 64, 8 warps: pre, main and post kernels",
     lambda: gluon_attn_bwd.attn_bwd(q, k, v, *gluon_attn_fwd.attn_fwd(q, k, v)[:1], do, gluon_attn_fwd.attn_fwd(q, k, v)[1])),
]
seen = set()
for site, module, wrapper, run in launches:
    run()
    torch.cuda.synchronize()
    for path in sorted(set(glob.glob(f"{cache}/*/*.ptx")) - seen):
        seen.add(path)
        kernel = os.path.basename(path)[: -len(".ptx")]
        name = f"{kernel}.{site}" if site else kernel
        with open(path) as f, open(os.path.join(out_dir, name + ".ptx"), "w") as out:
            out.write(f.read())
        print(f"{name}\t{module}\t{wrapper}")
