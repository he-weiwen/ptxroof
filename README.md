# ptxroof

Static analysis of PTX kernels for roofline questions: how many
instructions, flops and bytes a kernel's loops execute, which threads run
which code, and how each global access coalesces. It reads the PTX only:
nothing is run, and there is no cache or timing model. For measured
traffic and time, use Nsight Compute; ptxroof can compare its predictions
with a recorded launch.

Counts are symbolic in the kernel's parameters until they are bound; with
a block shape and a grid they become per-CTA and per-launch numbers. A
number is exact, an upper bound (`<=`), or a total with an unknown part
(`+ unknown`); what cannot be derived is reported as a named unknown.

## Quick start

```sh
cargo install --path .       # the toolchain pinned in rust-toolchain.toml
nvcc -ptx -lineinfo kernel.cu  # or the .ptx in Triton's kernel cache

ptxroof analyze kernel.ptx                    # symbolic, per loop and per kernel
ptxroof analyze kernel.ptx --bind 2:K=4096 --launch 128,1,1 --grid 64,32,1
                                              # parameter 2, block shape, grid
ptxroof analyze kernel.ptx --ncu record.json  # a launch tools/ncu/measure.py
                                              # recorded, against its counts
ptxroof analyze kernel.ptx --json             # the same result as JSON
```

Without `-lineinfo`, loops are named by their label instead of a source
line.

## Example

A 2D-tiled half-precision GEMM at M = N = K = 256, 64 threads per CTA and
4×4 CTAs, against its RTX 4090 record (`tests/fixtures/k5`). Abridged; the
output format is still changing.

```text
$ ptxroof analyze tests/fixtures/k5/k5.sm_89.ptx --ncu tests/fixtures/k5/k5.sm_89.256.ncu.json
kernel void hgemm_2d_blocktiling<64, 64, 8, 8, 8>(int, int, int, float, __half const*, __half const*, float, __half*)
  loop 5_2d_blocktiling.cuh:39 ($L__BB0_2)
    trips = 32
    per iteration:
      flops = 1024  (f32 1024)
      global bytes: load 32 B, store 0 B
      AI(global) = 32 flop/B
    accesses:
      5_2d_blocktiling.cuh:42  ld.global.u16  global load 2 B   warp: 4–8 sectors (ideal 2)    [32768 * %ctaid.y + 16 * k[5_2d_blocktiling.cuh:39] + 512 * ⌊%tid.x/8⌋ + 2 * (%tid.x mod 8) + param_4 - 16]
      5_2d_blocktiling.cuh:46  ld.global.u16  global load 2 B   warp: 2–3 sectors (ideal 2)    [128 * %ctaid.x + 4096 * k[5_2d_blocktiling.cuh:39] + 512 * ⌊%tid.x/64⌋ + 2 * (%tid.x mod 64) + param_5 - 4096]
    global bytes per CTA over the loop's own blocks: requested 65536 B, unique 65536 B, in sectors 98304–147456 B
  totals per launch [static]:
    flops = 33751040  (f32 33751040)
    global bytes: load 1179648 B, store 131072 B
    AI(global) = 25.75 flop/B
  measured L1TEX sectors of global loads over the launch: predicted 81920–131072, measured 81920
```

## Scope

- PTX from nvcc, Triton and Gluon; fixtures target sm_80, sm_89, sm_90
  and sm_100a.
- Trip counts of nvcc's and LLVM's loop shapes, guarded loops and loops
  whose counts depend on the CTA index; instructions, warp instructions,
  flops by pipe and precision, and bytes by state space, per loop
  iteration and per thread, CTA or launch.
- Which threads run each block and guarded instruction, from conditions
  on the thread and CTA indices, warp-specialized kernels included.
- Each memory access's address as an affine form, the sectors one warp's
  request touches, and per loop the bytes requested and touched.
- An instruction whose results nothing reads counts in no total.

Not covered, among others: `tcgen05` and `wgmma` flops, TMA bytes, loops
bounded by a loaded value or an mbarrier wait, a cache or DRAM model, and
kernels shipped only as SASS. `PLAN.md` lists every known limitation and
missing feature with its evidence.

## How the output is checked

- 64 Nsight Compute records (RTX 4090 launches of the Gluon kernels, k5
  and micro kernels, and imported B200 reports): `--ncu` compares each
  PTX line's predicted executions and L2 sectors and the launch's L1TEX
  sectors, and `tests/ncu_corpus.rs` keeps every verdict.
- Every fixture's re-printed PTX (`--dump-ast`) must assemble to the
  original's SASS, but for the cases `tests/roundtrip-allowlist.txt`
  lists.
- Generated loop kernels' trip counts are checked against a simulator
  (`tests/gen_loops.py`).

## Development

`./ci.sh` runs rustfmt, clippy, the unit and corpus tests, the lit tests
in `tests/lit` (LLVM's lit and FileCheck) and, with a GPU and Nsight
Compute, a fresh measurement of every RTX 4090 record. `cargo mutants`
runs the lit tests too, through `tests/cli_suite.rs`. `PLAN.md` lists the
known limitations and missing features; `docs/architecture.md` describes
the module layout.
