# ptxroof

Static analysis of PTX kernels for roofline questions: how many
instructions, flops and bytes a kernel's loops execute, which threads run
which code, and how each global access coalesces. It reads the PTX only:
nothing is run, and there is no cache or timing model. For measured
traffic and time, use Nsight Compute; ptxroof can compare its predictions
with a recorded launch.

It prints the PTX back, each line followed by what the analysis says of
it as a `//` comment. Counts are symbolic in the kernel's parameters until
they are bound; with a block shape and a grid they become per-CTA and
per-launch numbers. A number is exact, an upper bound (`<=`), or a lower
bound (`>=`) where a statement is not counted; what cannot be derived is
`?` and why, on its line.

## Quick start

```sh
cargo install --path .       # the toolchain pinned in rust-toolchain.toml
nvcc -ptx kernel.cu          # or the .ptx in Triton's kernel cache

ptxroof analyze kernel.ptx                    # symbolic, per thread
ptxroof analyze kernel.ptx --bind 2:K=4096 --launch 128,1,1 --grid 64,32,1
                                              # parameter 2, block shape, grid
ptxroof analyze kernel.ptx --ncu record.json  # the grid, block shape and
                                              # parameters of a launch
                                              # tools/ncu/measure.py recorded
```

## Example

A 2D-tiled half-precision GEMM at M = N = K = 256, 64 threads per CTA and
4×4 CTAs, the launch of its RTX 4090 record (`tests/fixtures/k5`). A few
of its lines; the `.entry` line's sums are cut short.

```text
$ ptxroof analyze tests/fixtures/k5/k5.sm_89.ptx --ncu tests/fixtures/k5/k5.sm_89.256.ncu.json
.visible .entry _Z20hgemm_2d_blocktilingILi64ELi64ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_( // grid 4×4×1 | block 64×1×1 | param_0 = 256, param_1 = 256, param_2 = 256 | pointer params assumed 32 B aligned | shared memory 2,048 B/CTA | per launch: flops f32 33,751,040; ...
$L__BB0_2: // loop: trips 32 | global per CTA: requested 65,536 B, unique 65,536 B, sectors 98,304 B | warp * × lane * | exec 32,768/launch
ld.global.u16  %rs1, [%rd15]; // global load [32768·%ctaid.y + 16·iv[$L__BB0_2] + 512·⌊%tid.x/8⌋ + 2·(%tid.x mod 8) + param_4] sectors/req 4 (ideal 2)
st.shared.u16  [%r20], %rs1; // shared store [2·%tid.x + As] wavefronts/req 1 (ideal 1)
ld.global.u16  %rs9, [%rd31]; // global load [128·%ctaid.x + 4096·iv[$L__BB0_2] + 512·⌊%tid.x/64⌋ + 2·(%tid.x mod 64) + param_5] sectors/req 2 (ideal 2)
$L__BB0_3: // loop: trips 8 | warp * × lane * | exec 262,144/launch
ld.shared.u16  %rs17, [%r110]; // shared load [2·iv[$L__BB0_3] + 128·⌊%tid.x/8⌋ + As] wavefronts/req 4 (ideal 1)
shl.b32  %r117, %r7, 3; // [<block 4>] warp * × lane * | exec 32,768/launch
```

A loop header gives its trips, an access its address with the loop's
iteration as LLVM's canonical induction variable, `iv[<header>]`, and the
sectors or shared wavefronts one warp's request needs, with pointer
parameters taken as 32 B aligned unless `--ptr-align` says otherwise.

## Scope

- PTX from nvcc, Triton and Gluon; fixtures target sm_80, sm_89, sm_90
  and sm_100a.
- Trip counts of nvcc's and LLVM's loop shapes, guarded loops and loops
  whose counts depend on the CTA index; how many times each block and
  instruction runs, and the kernel's flops by pipe and precision and
  bytes by state space, per thread, CTA or launch.
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

- 65 Nsight Compute records (RTX 4090 launches of the Gluon kernels, k5
  and micro kernels, and imported B200 reports): `tests/ncu_corpus.rs`
  and `tests/ncu_comparisons.rs` compare each PTX line's predicted
  executions and L2 sectors and the launch's L1TEX sectors with them
  (`report::build::compare`).
- Every fixture's re-printed PTX (`--dump-ast`) must assemble to the
  original's SASS, but for the cases `tests/roundtrip-allowlist.txt`
  lists.
- Generated loop kernels' trip counts are checked against a simulator
  (`tests/gen_loops.py`).

## Development

`./ci.sh` runs rustfmt, clippy, the unit and corpus tests, the lit tests
in `tests/lit` (LLVM's lit and FileCheck) and, with a GPU and Nsight
Compute, a fresh measurement of every RTX 4090 record. `tests/lit.sh`
runs `$LIT`, a `lit` on PATH (`uv tool install lit`) or Ubuntu's
`llvm-N-tools`; FileCheck comes from `$LLVM_TOOLS_DIR`, that LLVM, PATH
or Homebrew's `llvm`. `cargo mutants`
runs the lit tests too, through `tests/cli_suite.rs`. `PLAN.md` lists the
known limitations and missing features; `docs/architecture.md` describes
the module layout.
