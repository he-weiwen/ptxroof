# ptxroof: limitations, missing features, scope

What the tool does not do, with the evidence, and what is missing.
Edited in the same commit that changes either. The design history
before 2026-09-08 is in git: `git show 3ba19b5:PLAN.md`.

## What it does

Reads a PTX file. Per kernel: the block table (CFG), the loop forest
with trip counts as symbolic expressions over the kernel parameters,
instruction counts by kind and opcode per loop iteration and in total,
and per CTA also as warp instructions, one issue per warp with a
thread in the block; flops by pipe and precision (including atomic FP work), bytes by state space,
AI(global), and per
memory operand its address as an affine form over the thread and CTA
indices, the loop counters and the parameters, with the 32-byte
sectors one warp's request touches, over the lanes that execute it, once the block shape and the parameters in the lane
coefficients are bound; per
block and per guarded instruction, which threads of the CTA run it
when the selecting branches and the guard are thread-index
comparisons, `%laneid` in a one-dimensional block, combined with
`and`, `or` and `not`, or `elect.sync`'s one lane per warp, so
per-CTA totals count each on its own threads
(`micro/guards.ptx`, `micro/elect.ptx`); and per loop
with a numeric trip count, the global bytes one CTA
requests over the loop's own blocks and the distinct bytes it touches.
Text and JSON views of the same tree. Counts are static: per-thread
unless labeled per-CTA or per-warp, as requested by the PTX; nothing
is measured (README). Each instruction has one category and independent
contributions; text expands instructions with multiple contributions and JSON
retains variants for differing immediate sizes. Atomic FP arithmetic contributes
to `atomic_flops` and AI, with the existing atom read+write / red write-only byte
convention (`tests/instruction_contributions.rs`).

## Known limitations

### Wrong output

- **Register effects are incomplete.** The affine tracer records only a
  scalar first-operand register destination, misses tuple/pipe outputs and
  barrier reductions, and does not model guarded definitions. Its blacklist
  treats `stackrestore` and register-valued `nanosleep` inputs as definitions.
  The parser drops TMA coordinates, leaves bare register names as symbol
  references, and merges same-named registers in sibling scopes; negated
  source predicates and parenthesized call operands become `Unparsed`.
  The CFG treats `trap` as fallthrough. The `cp.async` classifier reports a
  fixed source-read size even with a runtime ignore-source predicate.
  Source locations, reproducible parser/classifier probes, and their outputs
  are in the [register-effects audit](docs/ptx-register-effects-audit.md#findings-in-the-current-implementation).
- **Addresses the tracer cannot read are unknown rows**, with the
  reason. Not read: xor swizzles of shared addresses (every ldmatrix
  and cp.async destination in the Gluon GEMM and attention kernels,
  "behind `xor.b32`"); divisions by a runtime value, such as the
  grouped tile schedule's `pid / num_pid_n` (the GEMM's global copies,
  a `min` of non-constants even with `--bind`); masks that are not
  one run of bits (the attention backward's dQ stores, `0xdc`); a
  counter's value after its loop (k1's remainder loop reads the main
  loop's final k: "more than one reaching definition"); `selp`.
- **Unique bytes per loop assume different pointer parameters are
  disjoint** (the cross-entropy's second pass, `tests/cli/
  analyze-gluon-ce-chunk`: the load and the store count 131,072 B
  unique for the 131,072 B requested; aliasing would halve it), count
  the loop's own blocks only, and need a numeric trip count, the block
  shape, and constant lane and counter coefficients. k5's K loop at
  256³ (`analyze-k5-footprint`): requested 65,536 B, unique 65,536 B;
  k1's (`analyze-k1-footprint`): 1,048,576 B, 32,768 B.

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
- **No cache model.** Cache operators are performance hints (PTX ISA
  §9.7.9.1) and are not reported. Sectors per warp request assume
  32-byte sectors, the granularity Nsight Compute confirmed on sm_89
  (the k5 and k1 footprint cross-checks below), and are not checked
  against `.target`; 128-byte lines are not reported, no metric having
  been compared to them. A loop's unique bytes are one
  CTA's compulsory footprint, ignoring lines another CTA on the same SM
  fetched, and `requested − unique` is the most an L1 could reuse, not
  what it does: k5 at 256³ is the one launch where L2 sectors equalled
  unique bytes times CTAs.

### Producers and validation

- **The frontend is not panic-free for arbitrary UTF-8.** An input file
  containing only `é` causes `ptxroof analyze` to exit with a panic:
  `end byte index 1 is not a char boundary`. The byte-oriented lexer
  slices an unexpected multibyte character after advancing one byte
  (`src/ptx/parse/lexer.rs`, `make` and `next_token`).

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
  discarded; assembler diagnostics can vary (the current round trip reports
  `Arguments mismatch for instruction 'mov'`). This is allowlisted in
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
  = 2,359,296 = 64 `mma` per warp-iteration likewise, the tool's
  `<= 3072` warp `mma` per CTA at K=768 times 768 CTAs
  (`analyze-gluon-fp8-gemm-c-fc-warps`); l1tex global-load
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
  aligned counts times their execution counts. Unique bytes, k5 at
  256³ and 16 CTAs: `lts__t_sectors_srcunit_tex_op_read.sum` = 36,868,
  the tool's 65,536 unique B per CTA over the K loop as 2,048 sectors
  plus the epilogue's 256-sector C tile, times 16, plus 4;
  `dram__bytes_read.sum` = 511,104 for the 524,288 B of A, B and C.
  Nothing else.
- Counts are PTX, not SASS: ptxas removes most register moves, folds
  address arithmetic into addressing modes, expands `.rn` divides, and
  may add spills.
- No machine model. AI is printed without a ceiling; the PTX `.target`
  names a family, not a part.

## Missing features

- SSA representation and migration of suitable dataflow analyses: planned,
  not implemented. The current tracer combines definition lookup and affine
  interpretation (`src/analysis/scalar/trace.rs`); construction semantics and
  migration boundaries are described in `docs/architecture.md`.

One line each, with the trigger that would start it.

- clang fixtures with a regen script. Trigger: the first clang-built
  kernel.
- `diff` between two builds of one kernel, PTX and SASS spill counts.
  Trigger: the first spill regression hunt.
- Local memory per thread from the `.local` depot declaration (k14:
  `__local_depot0[512]`), where spills go; keeping the declaration
  also lets k14's dump reassemble. Trigger: the same hunt.
- Access patterns: done 2026-09-15 (the `accesses` rows: every
  fixture's global and shared operands except those listed above,
  including 2D tiles' `⌊%tid.x/8⌋` and `(%tid.x mod 8)` and counters
  stepped by `4·N`; sectors per warp request by enumerating
  the lanes under every alignment of the uniform part, since pointer
  parameters carry no alignment in the PTX, so a coalesced 4-byte
  access reads `4–5 sectors`; needs `--launch` or `.reqntid` and
  `--bind` for a parameter in a lane coefficient; per enclosing loop how
  the address moves per iteration, `k[loop]: +16 B/iter` or
  `invariant`, the per-thread reuse class; and per loop the bytes one
  CTA requests and the distinct bytes it touches, the intervals of
  every executing thread at every iteration merged per base pointer).
  What remains is listed above.
- `check` verb: CI gate on a kernel property. Trigger: the first gate.
- Nsight Compute import beside the static columns. Trigger: the first
  static-versus-measured comparison beyond k5.
- Per-line HTML view. Trigger: wanting it.
- `capabilities` verb that generates the audit doc's tables.
- sm_90 and sm_100 families (`wgmma`, TMA, `tcgen05`): need a fixture
  and a per-instruction issue scope, which the model lacks.

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
