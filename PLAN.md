# ptxroof: limitations, missing features, scope

What the tool does not do, with the evidence, and what is missing.
Edited in the same commit that changes either. The design history
before 2026-09-08 is in git: `git show 3ba19b5:PLAN.md`.

## What it does

Reads a PTX file. Per kernel: the block table (CFG), the loop forest
with trip counts as symbolic expressions over the kernel parameters,
instruction counts by kind and opcode per loop iteration and in total,
flops by pipe and precision, bytes by state space, AI(global), and per
memory operand its address as an affine form over the thread and CTA
indices, the loop counters and the parameters, with the 32-byte
sectors and 128-byte lines one warp's request touches once the block
shape and the parameters in the lane coefficients are bound; and per
block, which threads of the CTA run it when thread-index branches
select them, so per-CTA totals count each block on its own threads.
Text
and JSON views of the same tree. Every number is static, per thread,
as requested by the PTX; nothing is measured (README).

## Known limitations

### Wrong output

- **Addresses the tracer cannot read are unknown rows**, with the
  reason. Not read: xor swizzles of shared addresses (every ldmatrix
  and cp.async destination in the Gluon GEMM and attention kernels,
  "behind `xor.b32`"); divisions by a runtime value, such as the
  grouped tile schedule's `pid / num_pid_n` (the GEMM's global copies,
  a `min` of non-constants even with `--bind`); masks that are not
  one run of bits (the attention backward's dQ stores, `0xdc`); a
  counter's value after its loop (k1's remainder loop reads the main
  loop's final k: "more than one reaching definition"); `selp`.

### Reported as unknowns

- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, nvcc's
  unroll main+remainder pair, LLVM's two-register counter and its
  predicate-phi two-trip loop. Not recognised, reported as
  `trips = unknown`: grid-stride loops (special registers),
  data-dependent bounds, multi-exit loops, inner loops bounded by an
  enclosing loop's counter (`micro/triangular.ptx`), and the attention kernels'
  causal loops, bounded by the CTA index, the mbarrier spin-waits
  (`mbarrier.test_wait` defines the predicate), and the persistent
  tile loop, which has a latch and an exit per partition.
- **Instruction families.** 77 of the 232 rows in
  `docs/ptx-instruction-coverage.md` are `Unknown`: integer, sparse
  and block-scaled `mma`; `wgmma`; `tcgen05`; bulk/TMA copies;
  textures and surfaces; `multimem`; video instructions.
- **By-value aggregate parameters** are not field-resolvable.
- **Sibling loops on one source line** beyond the unroll pair are
  reported as variants and excluded from totals.

### Presentation

- Per-thread totals of a warp-specialized kernel are `<=` the sum of
  its partitions, since a thread runs one of them: the `threads` column
  of the block table says which threads run each block (`128
  (⌊%tid.x/32⌋ < 4)`), and the per-CTA totals count each block on its
  own threads, but there is no per-role view of the per-thread numbers.
- Parameter names are positional (`param_2`); the PTX carries no
  source names.
- Floor division prints as `/`.

### Producers and validation

- Fixture corpus: nvcc output for one CUDA header ladder (k1, k2, k5,
  k11, k12, k14, mma_demo), Triton 3.8.0 (Gluon) output for five
  nanochat kernels (`tests/fixtures/gluon`: twelve PTX files; the
  GEMM, the cross-entropy chunk and the warp-specialized attention
  forward at two shapes each, the attention backward's three kernels;
  every instruction classifies), and hand-written micro kernels.
  clang is untested.
- `--dump-ast` output reassembles (ptxas, cuobjdump -sass) to SASS
  identical to the original's for 29 of the 30 fixtures; k14's is
  rejected because the in-kernel `.local` depot declaration is
  discarded (`Unknown symbol '__local_depot0'`) and is allowlisted in
  `tests/roundtrip-allowlist.txt`. A `tests/run.py` stage when the
  toolkit is on PATH, skipped otherwise.
- Generated loops (`tests/gen_loops.py`, a CI step): single counted
  loops drawn from a grammar of steps, initial values, bounds,
  comparisons, operand orders, branch polarities and read positions,
  each simulated in Python and bound with `--bind`; the tool's count
  must equal the simulation wherever the loop iterates, and a refused
  shape only counts. 200 kernels per run; the only refused shape is a
  loop that continues while two values are equal.
- Hardware cross-check on an RTX 4090 with Nsight Compute: k5's
  counts, and the c_fc fp8 GEMM's launch (M=4096, N=3072, K=768):
  `sm__ops_path_tensor_src_fp8.sum` = 19,327,352,832 = 2·M·N·K, the
  tool's 16384 fp8 flops per thread per K iteration times 12
  iterations, 128 threads and 768 CTAs; `sm__inst_executed_pipe_tensor`
  = 2,359,296 = 64 `mma` per warp-iteration likewise; l1tex global-load
  bytes (151,191,552) = every `cp.async` byte (150,994,944: two
  prologue stages plus ten of the twelve iterations, the predicated
  prefetches off at the end) plus 256 B per CTA for the two scale
  loads at sector granularity, under the tool's `<=` bound of
  176,947,200. The attention launches at B=2, T=2048, H=6, D=128:
  `sm__inst_executed_pipe_tensor` = 3,244,032 for the forward, the
  3,264 key-block visits times the tool's 1,024 warp-`mma` per CTA
  visit minus the 192 masked half-blocks it skips at 512; 8,110,080
  for the backward, 6,336 visits times 160 per warp; the pre and post
  kernels' cuda-core flops (2·ffma + fadd + fmul) 6,684,672 and
  3,145,728 are the tool's per-CTA 17,408 and 8,192 times 384 CTAs.
  Sectors per warp request, k5 and k1 at M=N=K=256 (`tests/cli/
  analyze-k5-footprint`, `analyze-k1-footprint`): l1tex global load
  requests and sectors 18,432 / 81,920 (k5) and 1,050,624 / 1,576,960
  (k1), stores 2,048 / 32,768 and 2,048 / 4,096, each the rows'
  aligned counts times their execution counts. Nothing else.
- Counts are PTX, not SASS: ptxas removes most register moves, folds
  address arithmetic into addressing modes, expands `.rn` divides, and
  may add spills.
- No machine model. AI is printed without a ceiling; the PTX `.target`
  names a family, not a part.

## Missing features

One line each, with the trigger that would start it.

- clang fixtures with a regen script. Trigger: the first clang-built
  kernel.
- `diff` between two builds of one kernel, PTX and SASS spill counts.
  Trigger: the first spill regression hunt.
- Local memory per thread from the `.local` depot declaration (k14:
  `__local_depot0[512]`), where spills go; keeping the declaration
  also lets k14's dump reassemble. Trigger: the same hunt.
- Access patterns, in three steps on the tracer; the first two are
  done (the `accesses` rows: every fixture's global and shared
  operands except those listed above, including 2D tiles'
  `⌊%tid.x/8⌋` and `(%tid.x mod 8)` and counters stepped by `4·N`;
  sectors and lines per warp request by enumerating the lanes under
  every alignment of the uniform part, since pointer parameters carry
  no alignment in the PTX, so a coalesced 4-byte access reads
  `4–5 sectors`; needs `--launch` or `.reqntid` and `--bind` for a
  parameter in a lane coefficient; and the cache path from the state
  space and the cache operator, PTX ISA §9.7.9.1, so the GEMM's
  `cp.async.cg` copies read "L2 only"):
  3. per loop, whether a reference is invariant or strided, the
     per-thread reuse distance for self-reuse (the body's footprint),
     and the unique-byte span per scope: compulsory ≤ moved ≤ requested;
     oracle: ncu DRAM bytes ≥ compulsory.
  Same-line decisions are the difference of two affine forms: exact for
  constant differences (k1's `[%rd25]`, `[%rd25+2]`, `[%rd25+4]`),
  need `--bind` for symbolic strides, undecidable across base pointers
  (assume distinct, and say so). Trigger fired 2026-09-08; not started.
- `check` verb: CI gate on a kernel property. Trigger: the first gate.
- Nsight Compute import beside the static columns. Trigger: the first
  static-versus-measured comparison beyond k5.
- Per-line HTML view. Trigger: wanting it.
- `capabilities` verb that generates the audit doc's tables.
- sm_90 and sm_100 families (`wgmma`, TMA, `tcgen05`): need a fixture
  and a per-instruction issue scope, which the model lacks.

## Will not do

- Cache hit rates and any reuse distance across warps or CTAs: they
  depend on the schedule, so a model would have to guess. Divergence,
  bank conflicts, occupancy, latency: Nsight Compute's.
- Branch probabilities: conditional code is a `<=` bound.
- General scalar-evolution: trip shapes are a catalogue grown by
  fixtures.
- Closed-form series for triangular nests.
- Guard implication between loop variants.
- Data-dependent bounds.
- SASS semantics beyond the line join and spill counting.
- Machine peak ratios beside AI: the part is not in the PTX.

## Acceptance scenarios

Ids used by `tests/acceptance/status.toml`:

| id | question | fixture |
|---|---|---|
| S1.1, S1.2 | Is k5's design point what I computed on paper, at sm_80 and sm_89? | k5 |
| S6 | Where does the work go? (unroll pair ranked) | k2 |
| S7.1, S7.2 | Did tiling pay off? (AI 0.5 vs 32) | k1, k5 |
| S8 | Am I on the precision path I think? (0 f16 flops) | k2 |
| S9.1, S9.2, S9.3 | Does the tool admit what it cannot see? | micro/data_dep, branchy, no_loc |
| S10.1, S10.2 | Is my tensor-core work visible, and only the work? | mma_demo, k14 |

Planned, no case yet: S2 spill diff, S3 coalescing, S4 black-box
Triton kernel (fixtures and CLI cases exist, `tests/cli/analyze-gluon-*`;
a scenario case waits on the GEMM's flops and bytes), S5 CI gate.

## Fixture policy

Every committed PTX carries a provenance header and a `regen.sh`
beside it that rebuilds it from source; `tests/run.py` lints this.
Hand-written micro kernels say so in their header.

## Conventions

- Commits under 50 non-test lines. No comments except externally
  imposed invariants, with a link. Cite the PTX ISA from `refs/`
  (`tools/fetch-manuals.sh`), not memory.
- This file and `docs/ptx-instruction-coverage.md` change in the same
  commit as what they describe.
