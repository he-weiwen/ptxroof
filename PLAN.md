# ptxroof: limitations, missing features, scope

What the tool does not do, with the evidence, and what is missing.
Edited in the same commit that changes either. The design history
before 2026-09-08 is in git: `git show 3ba19b5:PLAN.md`.

## What it does

Reads a PTX file. Per kernel: the block table (CFG), the loop forest
with trip counts as symbolic expressions over the kernel parameters,
instruction counts by kind and opcode per loop iteration and in total,
flops by pipe and precision, bytes by state space, AI(global). Text
and JSON views of the same tree. Every number is static, per thread,
as requested by the PTX; nothing is measured (README).

## Known limitations

### Wrong output

- **Triangular loop nests report the inner loop's trips as 0.** The
  latch tracer follows the outer induction variable to its pre-loop
  initial value. Found with a hand-written nest; no fixture yet.
- **Nested inlining is attributed to the intermediate file.** Only one
  `inlined_at` hop is followed. `ce_chunk_kernel.sm_89.ptx`: the
  reductions inlined from Triton's `standard.py` through
  `gluon_ce.py:50` land on `standard.py:293` (block rows, and
  `standard.py:191 x16, standard.py:293 x16` in the unrolled lines).

### Reported as unknowns

- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, and
  nvcc's unroll main+remainder pair. Not recognised, reported as
  `trips = unknown`: the predicate-controlled two-trip loop
  (`ce_chunk_kernel.v8192.sm_89.ptx`: a `mov.pred` phi, "latch
  predicate is not defined in the latch block"), LLVM's two-register
  counter (increment into a temporary, `mov` copy in the latch; seen
  in an earlier Triton build of the fp8 GEMM, not emitted by triton
  3.8.0 @ c3aa0c5, which the fixtures use), grid-stride loops (special
  registers), data-dependent bounds, multi-exit loops.
- **Instruction families.** 79 of the 232 rows in
  `docs/ptx-instruction-coverage.md` are `Unknown`: fp8, integer,
  sparse and block-scaled `mma`; `wgmma`; `tcgen05`; bulk/TMA copies;
  textures and surfaces; `multimem`; video instructions.
- **By-value aggregate parameters** are not field-resolvable.
- **Sibling loops on one source line** beyond the unroll pair are
  reported as variants and excluded from totals.

### Presentation

- Every label starts a block, so LLVM's `$L__tmpN` debug labels
  fragment the block table (`ce_chunk_kernel.sm_89.ptx`: 50 rows for
  straight-line code), and two zero-instruction blocks follow the
  final `ret` (`$L__tmp48`, `$L__func_end0`).
- Parameter names are positional (`param_2`); the PTX carries no
  source names.
- Floor division prints as `/`; `(param_10 + 63) / 64` is not folded
  to `⌈param_10/64⌉`, nor `(128 * x) / 128` to `x` (the two GEMM
  fixtures' trip counts).

### Producers and validation

- Fixture corpus: nvcc output for one CUDA header ladder (k1, k2, k5,
  k11, k12, k14, mma_demo), Triton 3.8.0 (Gluon) output for five
  nanochat kernels (`tests/fixtures/gluon`: seven PTX files, the GEMM
  and the cross-entropy chunk at two shapes each; classification 93
  to 100 percent), and hand-written micro kernels. clang is untested.
- `--dump-ast` output reassembles (ptxas, cuobjdump -sass) to SASS
  identical to the original's for 20 of the 21 fixtures; k14's is
  rejected because the in-kernel `.local` depot declaration is
  discarded (`Unknown symbol '__local_depot0'`). Not a CI step: it
  needs the CUDA toolkit.
- Hardware cross-check: k5's counts against Nsight Compute on an RTX
  4090, nothing else.
- Counts are PTX, not SASS: ptxas removes most register moves, folds
  address arithmetic into addressing modes, expands `.rn` divides, and
  may add spills.
- No machine model. AI is printed without a ceiling; the PTX `.target`
  names a family, not a part.

## Missing features

One line each, with the trigger that would start it.

- clang fixtures with a regen script. Trigger: the first clang-built
  kernel.
- fp8 dense `mma` (2·M·N·K over 32 lanes) with an NCU cross-check on
  sm_89. Trigger fired: `fp8_gemm_kernel.c_fc.sm_89.ptx`, 64
  unclassified `mma` per K iteration.
- `diff` between two builds of one kernel, PTX and SASS spill counts.
  Trigger: the first spill regression hunt.
- Local memory per thread from the `.local` depot declaration (k14:
  `__local_depot0[512]`), where spills go; keeping the declaration
  also lets k14's dump reassemble. Trigger: the same hunt.
- Access-pattern and coalescing analysis. Trigger: the first
  uncoalesced-access suspicion.
- `check` verb: CI gate on a kernel property. Trigger: the first gate.
- Nsight Compute import beside the static columns. Trigger: the first
  static-versus-measured comparison beyond k5.
- Per-line HTML view. Trigger: wanting it.
- `capabilities` verb that generates the audit doc's tables.
- sm_90 and sm_100 families (`wgmma`, TMA, `tcgen05`): need a fixture
  and a per-instruction issue scope, which the model lacks.

## Will not do

- Cache reuse, divergence, bank conflicts, occupancy, latency: Nsight
  Compute's.
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
