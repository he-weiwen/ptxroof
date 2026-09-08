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
- **Bounds ignore unknowns in their scope.** Flop and byte totals
  print as exact or `<=` even when the scope holds unclassified
  instructions or unquantified bytes. Gluon fp8 GEMM, K loop:
  `flops = 96` exact beside 32 unclassified `mma`; `global bytes:
  load <= 64 B` beside two `cp.async` whose bytes are missing, so the
  `<=` is wrong in direction. The unknowns are named; the numbers next
  to them are not marked.
- **Nested inlining is attributed to the intermediate file.** Only one
  `inlined_at` hop is followed. Gluon cross-entropy kernel: rows land
  on `standard.py:293` instead of `gluon_ce.py:50`.

### Reported as unknowns

- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, and
  nvcc's unroll main+remainder pair. Not recognised, reported as
  `trips = unknown`: LLVM's two-register counter (increment into a
  temporary, `mov` copy in the latch; Gluon GEMM K loop), the
  predicate-controlled two-trip loop (Gluon cross-entropy), grid-stride
  loops (special registers), data-dependent bounds, multi-exit loops.
- **Instruction families.** 79 of the 232 rows in
  `docs/ptx-instruction-coverage.md` are `Unknown`: fp8, integer,
  sparse and block-scaled `mma`; `wgmma`; `tcgen05`; bulk/TMA copies;
  textures and surfaces; `multimem`; video instructions.
- **`cp.async` sizes in hex** (`0x10`) are not parsed; the copy's bytes
  are then unknown. Gluon GEMM.
- **`.reg .b16 lo, hi;` inside inline-asm scopes** produces an
  "unparsed statement" entry per block; the instructions inside the
  block are counted. Gluon GEMM: 8 of 152 asm blocks.
- **By-value aggregate parameters** are not field-resolvable.
- **Sibling loops on one source line** beyond the unroll pair are
  reported as variants and excluded from totals.

### Presentation

- Every label starts a block, so LLVM's `$L__tmpN` debug labels
  fragment the block table (Gluon cross-entropy: 50 rows for
  straight-line code), and a zero-instruction block follows the final
  `ret`.
- Parameter names are positional (`param_2`); the PTX carries no
  source names.
- Floor division prints as `/`, and `(x + 63) / 64` is not folded to
  `⌈x/64⌉`.

### Producers and validation

- Fixture corpus: nvcc output for one CUDA header ladder (k1, k2, k5,
  k11, k12, k14, mma_demo) plus hand-written micro kernels. Triton
  3.8.0 (Gluon) output parses, with classification at 97 to 100
  percent on six kernels, but no Triton fixture is committed. clang
  is untested.
- Hardware cross-check: k5's counts against Nsight Compute on an RTX
  4090, nothing else.
- Counts are PTX, not SASS: ptxas removes most register moves, folds
  address arithmetic into addressing modes, expands `.rn` divides, and
  may add spills.
- No machine model. AI is printed without a ceiling; the PTX `.target`
  names a family, not a part.

## Missing features

One line each, with the trigger that would start it.

- Triton and clang fixtures with a generator script, and the fixes
  above that they pin. Trigger fired: the nanochat Gluon kernels.
- fp8 dense `mma` (2·M·N·K over 32 lanes) with an NCU cross-check on
  sm_89. Trigger fired: the same kernels.
- `diff` between two builds of one kernel, PTX and SASS spill counts.
  Trigger: the first spill regression hunt.
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
Triton kernel, S5 CI gate.

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
