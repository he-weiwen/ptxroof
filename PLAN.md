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
sectors one warp's request touches and the fewest its distinct bytes
need, over the lanes that execute it, once the block shape and the parameters in the lane
coefficients are bound; per
block and per guarded instruction, which threads of the CTA run it
when the selecting branches and the guard are thread-index
comparisons, `%laneid` in a one-dimensional block, combined with
`and`, `or` and `not`, or `elect.sync`'s one lane per warp, so
per-CTA totals count each on its own threads
(`micro/guards.ptx`, `micro/elect.ptx`), and which lanes of a warp
run it when the rest of the condition is the same for every thread
(rmsnorm's `mov.pred %p1, -1` guard: `warp: 16–17 sectors`); and per loop
with a numeric trip count, the global bytes one CTA
requests over the loop's own blocks, the bytes of the sectors those
requests touch, and the distinct bytes it touches.
Text and JSON views of the same tree. Counts are static: per-thread
unless labeled per-CTA or per-warp, as requested by the PTX; nothing
is measured (README). Each instruction has one category and independent
contributions; text expands instructions with multiple contributions and JSON
retains variants for differing immediate sizes. Atomic FP arithmetic contributes
to `atomic_flops` and AI, with the existing atom read+write / red write-only byte
convention (`tests/instruction_contributions.rs`). A `cp.async` with immediate
source size zero has zero global sectors per request and over the loop;
its shared writes and instruction counts remain counted
(`zero_source_copy_has_no_global_sectors_but_keeps_shared_writes` in that test).

## Known limitations

### Wrong output

- **Register effects are incomplete.** The affine tracer records the
  registers of the operands the register-effects decoder says an
  instruction writes, a register, a `p|q` pair or a `{…}` vector, as
  destinations (`micro/vector_def.ptx`: a guard on a register a vector
  load overwrote is a bound, `<= 9 sectors`), refuses an element of a
  multi-element `ld.param` vector rather than reading the parameter's
  bits, and does not model guarded definitions: a branch
  or guard comparing such a value is checked against the uniformity
  analysis only when the comparison is free of the thread index
  (`micro/guarded_def.ptx`: `<= 5 sectors`, not 64 threads exactly), and
  an address through one is read as the last write. A form the
  register-effects decoder does not decode, `call` or a sub-operation
  it does not list (none in the corpus,
  `tests/register_effects_corpus.rs`), is taken to write every register
  it names. The parser drops TMA coordinates (five B200 fixtures
  do not round-trip: `Arguments mismatch for instruction
  'cp.async.bulk.tensor'`) and a texture's coordinates and its
  module's `.texref` (`micro/texture_outputs.ptx`: `Arguments mismatch
  for instruction 'tex'`); negated source predicates, parenthesized
  call operands and an address followed by `.unified` (PTX ISA
  §9.7.9.8, `unified_address_is_unparsed`) become `Unparsed`.
  The CFG treats `trap` as fallthrough. The `cp.async` classifier reports a
  fixed source-read size even with a runtime ignore-source predicate.
  Source locations, reproducible parser/classifier probes, and their outputs
  are in the [register-effects audit](docs/ptx-register-effects-audit.md#findings-in-the-current-implementation).
- **Registers declared without `%` are symbols.** The parser reads `lo`
  in `{ .reg .b16 lo, hi; … }` (PTX ISA §4.4, §9.7.13.1) as a symbol
  reference, and the same name in sibling scopes as one, so an
  instruction writing it defines nothing and one reading it reads
  nothing: Gluon's fp8 conversions (`cvt.rn.satfinite.e4m3x2.f32 lo,
  …` then `mov.b32 %r106, {lo, hi}` in quantize_transpose and the
  lm_head_dx GEMM) and the attention forward's `mbarrier.test_wait`
  into `complete`, read by `@!complete` guards. In
  `micro/bare_registers.ptx` the packed value is taken as the same in
  every thread: a store only thread 0 runs is `warp: 4–5 sectors`.
  Likewise a `.reg` that reuses a special register's name in a nested
  scope takes the name in the whole kernel: in
  `micro/scoped_shadow.ptx` the `%laneid` read after the scope, which
  ptxas reads as `SR_LANEID`, is the scoped register's 0, so 32
  threads' stores are `warp: 1 sector` at `[param_0]`. And a video
  instruction's operand selector (`%r1.h0`, PTX ISA §9.7.19.1) is read
  as part of the name: in `micro/video_selector.ptx` the `vadd` that
  writes all of %r1 writes `%r1.h0` instead, so the address stays
  `[4 * %tid.x + param_0]`, where ptxas stores to param_0 from all 32
  threads.
- **`cvt` is read as its last operand unchanged** (PTX ISA §9.7.9.22):
  a two-source pack (`cvt.rn.bf16x2.f32 d, a, b` puts `a` in the
  upper half; 400 in seven Gluon fixtures), the bits of an
  integer-to-float conversion and a narrowing conversion are each read
  as the source's value (`micro/cvt_reads.ptx`: the three addresses are `4 * %tid.x +
  param_0`, the last plus `17179869184`; only the last is 4·%tid.x).
- **A parameter load at an offset reads the whole parameter** (PTX
  ISA §5.1.6.1): `ld.param.u32 %r3, [p+4]` reads as `param_1`, like
  the load at offset 0, so comparing two fields of a by-value
  aggregate is `0 == 0` (`micro/param_fields.ptx`: the store `on 64
  threads in 2 warps`, `store 256 B` per CTA). The parser also drops
  the parameter's `.align` and `[size]`.
- **A `setp` that selects threads is read as a signed comparison of
  its operands' values whatever its type, and its second output as the
  first's negation** (PTX ISA §9.7.6.2, §9.7.7.2). In
  `micro/setp_unsigned.ptx`, `setp.lt.u32` of `%tid.x - 16` against 32
  runs the store on threads 16 to 47 (ptxas: `ISETP.GE.U32`), read as
  `on 48 threads in 2 warps (%tid.x < 48)`. In `micro/setp_f16x2.ptx`
  the second output of `setp.lt.f16x2` compares the upper halves, 0 <
  0, so the store never runs; read as `on 64 threads in 2 warps
  (%tid.x >= 0)`.
- **Addresses the tracer cannot read are unknown rows**, with the
  reason. Not read: xor swizzles of shared addresses (every ldmatrix
  and cp.async destination in the Gluon GEMM and attention kernels,
  "behind `xor.b32`"); divisions by a runtime value, such as the
  grouped tile schedule's `pid / num_pid_n` (the GEMM's global copies,
  a `min` of non-constants even with `--bind`); masks that are not
  one run of bits (the attention backward's dQ stores, `0xdc`); a
  counter's value after its loop (k1's remainder loop reads the main
  loop's final k: "more than one reaching definition"); values loaded
  from memory (the cross-entropy's target gather: "depends on a value
  loaded from memory, behind `selp.b64`"); an
  `or` read as an add only when one side's largest value is below a
  power of two the other side is a multiple of, not by the bits each
  side can set (rope_norm's `or.b32 %r106, %r104, 16` on
  `64·%ctaid.x + (⌊%tid.x/8⌋ mod 16)`: "or with a value below 0x20";
  k14's C stores: "or of two values neither of known range").
- **Unique bytes per loop assume different pointer parameters are
  disjoint** (the cross-entropy's second pass, `tests/lit/
  gluon-ce-chunk.test`: the load and the store count 131,072 B
  unique for the 131,072 B requested; aliasing would halve it), count
  the loop's own blocks only, and need a numeric trip count, the block
  shape, and constant lane and counter coefficients. k5's K loop at
  256³ (`k5-footprint.test`): requested 65,536 B, unique 65,536 B;
  k1's (`k1-footprint.test`): 1,048,576 B, 32,768 B.

### Reported as unknowns

- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, nvcc's
  unroll main+remainder pair, LLVM's two-register counter and its
  predicate-phi two-trip loop. Not recognised, reported as
  `trips = unknown`: grid-stride loops (special registers),
  data-dependent bounds, multi-exit loops, inner loops bounded by an
  enclosing loop's counter (`micro/triangular.ptx`), a latch `setp`
  that combines its comparison with a predicate
  (`a_comparison_combined_with_a_predicate_is_not_a_trip_count`), and the attention kernels'
  causal loops, bounded by the CTA index, the mbarrier spin-waits
  (`mbarrier.test_wait` defines the predicate), and the persistent
  tile loop, which has a latch and an exit per partition. Bounds the
  tracer does not read, by the loop's source line: a `max` of
  non-constants (b200 `scaled_mm_kernel`, `:100`), a division by a
  non-constant (b200 `_chunk_scan_fwd_kernel`, `:331`), one half of a
  `mov.b64 {…}` unpack (b200 `_w8a8_block_fp8_matmul`, `:90`), a float
  immediate and an `and.pred` latch predicate (sol-study census
  `00511.ptx` `:121`, `00908.ptx` `:155`), and an `atom` result
  (`01542.ptx` `:223`).
  Over the census's 2,997 sm_100a loops (`missed-opt-census/art/sm100v3`,
  inferred arguments), 749 have a trip count: 1,437 are mbarrier waits
  and 408 read loaded values, 242 grid-stride, 50 `max`/`min`, 34
  unpacks, 20 atomics, 19 divisions, 15 multi-exit.
- **Instruction families.** 77 of the 232 rows in
  `docs/ptx-instruction-coverage.md` are `Unknown`: integer, sparse
  and block-scaled `mma`; `wgmma`; `tcgen05`; bulk/TMA copies;
  textures and surfaces; `multimem`; video instructions. `tcgen05`
  is in 12 of the B200 fixtures (`tests/fixtures/b200`), bulk tensor
  copies (`cp.async.bulk*`) in five.
- **Special registers PTX ISA §10 does not describe** — `%pm*`
  ("behavior is currently undefined"), `%envreg*` (driver-defined),
  `%globaltimer*` (target-specific), `%reserved_smem_offset_*` — read
  as divergent, "is not modelled"
  (`special_registers_by_the_threads_that_share_them`).
- **Instruction keywords after PTX ISA 9.3.** The parser knows the
  pinned manual's Table 2 and `fabric`; a keyword a later ISA adds
  leaves its statement unparsed (`mnemonics_outside_the_isa_are_unparsed`;
  nvcc 13.4 already emits `.version 9.4`).
- **Calls.** Module-scope `.func` declarations and definitions are
  discarded (`module_scope_functions_are_skipped`), so no callee's
  instructions are counted; nvcc's parenthesized `call.uni` is
  unparsed (above).
- **Sibling loops on one source line** beyond the unroll pair are
  reported as variants and excluded from totals.
- **In a kernel with a cycle of more than one entry**, the uniformity
  analysis takes every value defined on a cycle and read outside it as
  divergent, whatever the cycle's exits compare
  (`a_value_from_a_cycle_with_two_entries_is_divergent_after_it`).

### Presentation

- Per-thread totals of a warp-specialized kernel are `<=` the sum of
  its partitions, since a thread runs one of them: the `threads` column
  of the block table says which threads run each block (`128
  (⌊%tid.x/32⌋ < 4)`), and the per-CTA totals count each block on its
  own threads, but there is no per-role view of the per-thread numbers.
- Parameter names are positional (`param_2`); the PTX carries no
  source names.
- Floor division prints as `/`.
- **Pointer alignment is unknown.** Sectors per warp request and per
  loop are ranges over every alignment of the base under which the
  accesses are aligned (PTX ISA §6.4.1): nvcc declares no `.ptr
  .align`, Triton declares `.align 1`, and there is no flag to state
  one. The aligned end matched Nsight Compute on k5 and k1, whose
  pointers `cudaMalloc` aligned to 256 B. A 16-byte access on a
  pointer does not narrow the range of a 2-byte access on the same
  pointer.
- **A loop behind a branch on parameters is a bound even when
  `--bind` decides the branch.** k5 at K=256: per CTA,
  `fma.rn.f32 <= 1052672`, and its record measures exactly 16 CTAs
  times that; the K loop's blocks are `consistent`, not `confirmed`
  (`tests/lit/k5-ncu-verdicts.test`).
- **No arithmetic intensity over sector bytes.** A loop's sector
  bytes cover its own blocks while its flops sit in nested loops
  (k5's K loop: 98,304 B in sectors, no flops of its own).
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

- **The affine tracer can overflow the stack.** 14 of the census's
  2,837 sm_100a kernels abort (`missed-opt-census/art/sm100v3/00846.ptx`:
  `thread 'main' has overflowed its stack`): `obstacle` meets a merged
  register and calls `obstacle_of_any_def`, which starts a new visited
  set, so a loop-carried value recurses without end
  (`src/analysis/scalar/trace.rs`).

- **The frontend is not panic-free for arbitrary UTF-8.** An input file
  containing only `é` causes `ptxroof analyze` to exit with a panic:
  `end byte index 1 is not a char boundary`. The byte-oriented lexer
  slices an unexpected multibyte character after advancing one byte
  (`src/ptx/parse/lexer.rs`, `make` and `next_token`).
- **Opaque kernel parameters are a parse error.** `.texref`,
  `.samplerref` and `.surfref` parameters (PTX ISA §11.2.1) are not
  fundamental types (`opaque_params_are_a_parse_error`).

- Fixture corpus: nvcc output for one CUDA header ladder (k1, k2, k5,
  k11, k12, k14, mma_demo), Triton 3.8.0 (Gluon) output for five
  nanochat kernels (`tests/fixtures/gluon`: twelve PTX files; the
  GEMM, the cross-entropy chunk and the warp-specialized attention
  forward at two shapes each, the attention backward's three kernels;
  every instruction classifies), Triton 3.9.0 output for the 26 kernel
  launches of 20 production-shaped serving and training cases and eight
  kernels of its root-cause runs (`tests/fixtures/b200`, sm_100a, from
  sol-study's B200 Nsight Compute reports: among them Triton's
  warp-specialized SSD scan and tutorial attention, TMA attention and
  MLA, a block-scaled GEMM), and hand-written micro kernels. clang is
  untested.
- `--dump-ast` output reassembles (ptxas, cuobjdump -sass) to SASS
  identical to the original's for 61 of the 70 fixtures. k14's is
  rejected because the in-kernel `.local` depot declaration is
  discarded; assembler diagnostics can vary (the current round trip reports
  `Arguments mismatch for instruction 'mov'`). k1's and k2's assemble to
  different SASS because `.pragma "nounroll";` is discarded, so ptxas
  unrolls the loop it annotated. Five B200 TMA kernels lose their
  tensor coordinates (above). These are allowlisted in
  `tests/roundtrip-allowlist.txt`. One lit test per fixture
  (`tests/lit/roundtrip`), unsupported without the toolkit or when
  ptxas rejects the fixture's `.version` as newer than its own.
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
  (`gluon-fp8-gemm-c-fc-warps.test`); l1tex global-load
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
  Sectors per warp request, k5 and k1 at M=N=K=256 (`tests/lit/
  k5-footprint.test`, `k1-footprint.test`): l1tex global load
  requests and sectors 18,432 / 81,920 (k5) and 1,050,624 / 1,576,960
  (k1), stores 2,048 / 32,768 and 2,048 / 4,096, each the rows'
  aligned counts times their execution counts; per CTA, k5's K loop
  moves 98,304 B in sectors at the aligned end of its range and the
  epilogue's 64 loads 65,536 B, 5,120 sectors, times 16 CTAs. Unique bytes, k5 at
  256³ and 16 CTAs: `lts__t_sectors_srcunit_tex_op_read.sum` = 36,868,
  the tool's 65,536 unique B per CTA over the K loop as 2,048 sectors
  plus the epilogue's 256-sector C tile, times 16, plus 4;
  `dram__bytes_read.sum` = 511,104 for the 524,288 B of A, B and C.
  Nothing else.
- Counts are PTX, not SASS: ptxas removes most register moves, folds
  address arithmetic into addressing modes, expands `.rn` divides, and
  may add spills.
- Measurement records launch with every pointer parameter zero-filled
  (`tools/ncu/launch.c`), so a branch on loaded data takes its
  zero-data side. A line of a loop that branches back on an mbarrier
  wait's result runs a timing-dependent number of times: such lines
  are marked `timing_dependent` and not checked (the attention forward
  records: 30 lines each).
- Kernels that ship only SASS have no PTX to read: 18 of the 149
  launches in sol-study's B200 reports (cuBLAS, cuDNN, CUTLASS and CuTe
  DSL, flashinfer's trtllm-gen attention, PyTorch's elementwise).
- `--ncu` compares no counts in a kernel with irreducible control
  flow, whose blocks' executions the tool does not know
  (`tests/lit/ncu-irreducible.test`: every block `unchecked`), nor
  on a PTX line holding more than one instruction, which a record
  counts together (`tests/lit/ncu-two-per-line.test`).
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
- Nsight Compute import beside the static columns: started 2026-10-06.
  `tools/ncu/measure.py` records one launch per PTX line, the SASS
  ptxas attributes to the line and their predicated-on thread
  instructions executed (`tests/fixtures/k5/k5.sm_89.256.ncu.json`;
  every Gluon kernel's first launch, `tests/fixtures/gluon/*.ncu.json`).
  The B200 records (`tests/fixtures/b200/*.ncu.json`) come from
  sol-study's reports instead: `import_reports.py` attributes their
  per-SASS counts to PTX lines through the same `.loc` recompile under
  Triton's ptxas command, and takes each launch's parameters from a
  replay of the report's own command as the B200, matched by SASS and
  grid, with `i64:`, `i1:` and `tensormap:128` for the parameters of a
  tensor descriptor, which launch.c does not take; the PTX
  must still assemble to the recorded SASS (`b200-sass.test`, under
  ptxas 13.4 only). `ci.sh` does not re-measure them with
  `measure.py --check`, which would launch on zeroed inputs.
  `analyze --ncu RECORD` takes the record's block shape and `u32`
  and `i32` parameters for `--launch` and `--bind`, and refuses a record whose
  lines are not the PTX's (`tests/lit/k5-ncu-stale.test`), and
  gives each block a verdict (`kernels[].measured`): a line whose
  `CountConstraint` is not `Free` and whose SASS is one opcode must
  equal an exact prediction or stay within a bound. Each block also
  counts its instructions, those with SASS and those that agree, and
  lists every line whose measured count is not the predicted number,
  with its SASS opcodes (`differing`); the text report prints them
  under `measured`. `ci.sh` measures every record again on the GPU it
  names (`measure.py --check`) and fails if it changed.
- Per-line HTML view. Trigger: wanting it.
- `capabilities` verb that generates the audit doc's tables.
- sm_90 and sm_100 families (`wgmma`, TMA, `tcgen05`): need a
  per-instruction issue scope, which the model lacks; sm_100 fixtures
  exist (`tests/fixtures/b200`), sm_90 ones do not.

## Acceptance scenarios

Each id names a lit test, `tests/lit/s*.test`, in its first comment:

| id | question | fixture |
|---|---|---|
| S1.1, S1.2 | Is k5's design point what I computed on paper, at sm_80 and sm_89? | k5 |
| S6 | Where does the work go? (unroll pair ranked) | k2 |
| S7.1, S7.2 | Did tiling pay off? (AI 0.5 vs 32) | k1, k5 |
| S8 | Am I on the precision path I think? (0 f16 flops) | k2 |
| S9.1, S9.2, S9.3 | Does the tool admit what it cannot see? | micro/data_dep, branchy, no_loc |
| S10.1, S10.2 | Is my tensor-core work visible, and only the work? | mma_demo, k14 |

Planned, no case yet: S2 spill diff, S3 coalescing, S4 black-box
Triton kernel (fixtures and lit tests exist, `tests/lit/gluon-*`;
a scenario case waits on the GEMM's flops and bytes), S5 CI gate.

## Fixture policy

Every committed PTX carries a provenance header and a `regen.sh`
beside it that rebuilds it from source; `tests/lit/fixture-provenance.test` checks this.
Hand-written micro kernels say so in their header.

## Conventions

- Commits under 50 non-test lines. No comments except externally
  imposed invariants, with a link. Cite the PTX ISA from `refs/`
  (`tools/fetch-manuals.sh`), not memory.
- This file and `docs/ptx-instruction-coverage.md` change in the same
  commit as what they describe.
