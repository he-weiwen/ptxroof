# ptxroof: limitations, missing features, scope

What the tool does not do, with the evidence, and what is missing.
Edited in the same commit that changes either. The design history
before 2026-09-08 is in git: `git show 3ba19b5:PLAN.md`.

## What it does

Reads a PTX file and prints it back, each line followed by nothing or
by ` // ` and what the analysis says of it (`tests/report_invariants.rs`),
two statements' annotations on one line joined by ` ; `
(`statements_on_one_line_join_their_annotations`). Per kernel the
analysis finds the loop forest with trip counts as symbolic expressions
over the kernel parameters, and per
memory operand its address as an affine form over the thread and CTA
indices, the loop counters and the parameters, with the 32-byte
sectors one warp's request touches and the fewest its distinct bytes
need, over the lanes that execute it, once the block shape and the parameters in the lane
coefficients are bound, and for a shared operand the wavefronts one
warp's request takes and the number with no bank conflict, by how a
request's lanes were measured to be served on an RTX 4090
(`micro/shared_wavefronts.ptx`: the warp for 1- to 4-byte accesses,
half-warps for 8-byte, quarter-warps for a 16-byte store, the 8 lanes
of each matrix of an `ldmatrix`; on the census, 54,950 shared rows,
7,880 of them in 100 kernels above their ideal); per
block and per guarded instruction, which threads of the CTA run it
when the branches it is control dependent on in one trip of its loop
and the guard are thread-index comparisons
(`micro/or_entry.ptx`: under `%tid.x < 64`, `%tid.x < 16 || %tid.x >= 48`
stores on 32 threads), with the grid given comparisons with the CTA index as
well (a bound per CTA where CTAs differ), `%laneid` in a
one-dimensional block, or constants
(rmsnorm's loads under `mov.pred %p2, 0` run on no thread and count in
no total, per thread or per CTA; with `--bind 10:K=768` the c_fc fp8
GEMM's path for a short K, `gluon-fp8-gemm-c-fc-warps.test`), combined with
`and`, `or` and `not`, or `elect.sync`'s one lane per warp, so
per-CTA totals count each on its own threads
(`micro/guards.ptx`, `micro/elect.ptx`), and with the grid given the
totals per launch its threads summed over the CTAs
(`micro/cta_mask.ptx`: 4000 B stored by 1000 threads of 16 CTAs of
64), and which lanes of a warp
run it when the rest of the condition is the same for every thread
(`micro/carry_uniform.ptx`'s branch on a carry from two parameters:
`sectors/req 4`); and per loop
with a numeric trip count, the global bytes one CTA
requests over the loop's own blocks, the bytes of the sectors those
requests touch, and the distinct bytes it touches. An instruction that
runs but whose results nothing with an effect reads (LLVM's ADCE over
the register SSA, `src/analysis/dead_code.rs`) counts in no total and
no loop footprint, and is marked `unused`: ptxas emits no SASS for
any of the 954 such instructions in kernels with an Nsight Compute
record (`tests/dead_code_corpus.rs`). Counts are static; nothing is
measured (README). Each instruction has independent contributions, and
copies of one opcode with differing immediate sizes each count their own bytes
at their own addresses
(`copies_of_one_opcode_count_their_own_source_sizes_and_addresses`); an
ignored instruction has no contribution and
an unsupported one an unknown contribution
(`ignored_unknown_and_unquantified_contributions_stay_distinct`). A
warp-collective instruction (the tensor families) contributes its warp total
over the 32 lanes (`src/analysis/instruction_counts/measurement.rs`). Atomic FP
arithmetic contributes one flop per scalar element, packed and vector elements
included (`atomics_keep_byte_policy_and_count_fp_elements`), to the atomic
FLOPs, with the existing atom read+write / red write-only byte convention
(`tests/instruction_contributions.rs`). A `cp.async` with immediate
source size zero has zero global sectors per request and over the loop;
its shared writes and instruction counts remain counted
(`zero_source_copy_has_no_global_sectors_but_keeps_shared_writes` in that test).
A `cp.async` reads its source only on the threads whose ignore-src
predicate is false (PTX ISA §9.7.9.26.3.1), read as a guard is: its
sectors per warp, loop footprint, global load bytes and sector
comparison count those threads, and its shared store the threads that
execute it (`micro/cp_async_ignore_src.ptx`, RTX 4090: 2176 B over the
launch, its four lines' 68 measured L2 sectors). With a src-size
register it reads at most cp-size: `sectors/req <= 17 (ideal <= 16)`
(`micro/cp_async_src_size.ptx`).

Each
block's annotation, on its label's line or else on its first
instruction's after `[<block name>]` (`micro-vector-register.test`): a loop header's `loop: trips` and
the trip count, or `?` and why (`tests/lit/k5-annotate.test`), and
the global bytes one CTA requests over one run of the loop's own
blocks, the distinct bytes and the bytes of the sectors they touch,
`<=` for a bound, `?` where an address is not read
(`k5-annotate.test`, `k1-annotate.test`,
`micro-loop-bytes-annotate.test`), then,
given the block shape, the threads that run it as `warp <W> × lane
<L>` per group of warps with the same lanes, `<=` where the lanes are
a bound (`micro-guards-annotate.test`,
`gluon-attn-fwd-ws-annotate.test`); with the grid, where they read
the CTA index, the conjuncts that read it, the `not` of an `or` taken
as the `and` of its parts' negations, each as `runs when` prints a
comparison, and the CTAs that run it, `| CTAs %ctaid.x < 4 (4/8)`,
`| CTAs %ctaid.x ≥ 1 (3/4)` (`micro-cta-guards.test`,
`micro-cta-trips-annotate.test`, `b200-cta-condition-annotate.test`),
or where the lanes differ by CTA the most CTAs' lanes, `in 3844/3969
CTAs, other 125 CTAs differ`
(`k1-annotate.test`); and without the block shape the condition
that selects them, a loop's iteration k printed as
`(iv[<header>] + 1)`
(`without_a_block_shape_a_block_names_its_condition`). Then, where
past its immediate dominator the block runs under conditions each the
same for the whole CTA or a warp, `runs when` and each, a comparison
as `lhs cmp rhs`, the terms of negative coefficient and the constant
on the right,
an unknown value by its register and line (`%r1880@$L__BB0_39`), its
known multiple as the value, with the branch's polarity, joined by
`and`, one under an `or` or a `not` with every part, the `not` of an
`and` or an `or` by De Morgan's laws, an `and` under an `or` and an
`or` under an `and` parenthesized, a thread-index
comparison too; then `CTA-uniform` where each is the same for the
CTA, else `warp-uniform`, and the lines of the branches deciding them
(`gluon-attn-fwd-ws-annotate.test`, `micro-or-entry-annotate.test`,
`a_block_names_the_uniform_conditions_it_runs_under`); one that is
not a comparison read within its type as `predicate <register> (line
<L>) is true|false`: `micro/branchy.ptx`'s `setp.lt.f32` of a loaded
value (`s9-branchy.test`); a jump table's target as `jump table at
line <L> selects this target` (`b200-jump-table-annotate.test`), a
loop's exit as `leaves
loop <header> toward <block>`
(`micro-loop-guard-sibling-annotate.test`). A comparison of a loop's
counter, the same for the CTA, joins them, `runs when iv[$L__BB0_1] ≥
2` (`micro-counter-guard.test`). Then, where a block
dominating it has such a condition of its own, the nearest's line: after
`nested in` where the block has one, else after `under`
(`gluon-attn-fwd-ws-annotate.test`). Then how many times its
unguarded statements run, over the launch given the grid
and the block shape, else per CTA, else per thread but where every
thread runs them once
(`without_a_block_shape_a_block_names_its_condition`): `exec
32,768/launch`, `<=` for a bound, and under a loop whose trips are
not known `exec ? (trips of <header>)` (`k5-annotate.test`,
`gluon-attn-fwd-ws-annotate.test`); a statement whose count differs,
but a predicated `bra`, has its own on its line, and a guarded one
whose threads differ from its block's has theirs, as a block's, before
it (`micro-guards-annotate.test`). A memory operand but a parameter is
`<space> <direction> [<address>]`, a loop's iteration k printed as
`iv[<header>]` = k - 1, or `addr ?` and why, a counter there as
`(iv[<header>] + 1)` (`micro-guarded-exit-bound-annotate.test`),
and `c·⌊u/d⌋` of an unknown value u whose low bits every definition
leaves 0 up to d as `(c/d)·u` (`micro-uniform-aligned-annotate.test`),
then a global one's `sectors/req` and a shared one's
`wavefronts/req` with their ideal, `<=` before both where the lanes
are a bound, or `?` and why (`k5-annotate.test`,
`micro-vector-def-annotate.test`). An
instruction whose results nothing with an effect reads is `unused`
(the same test). What a statement leaves out of the totals is said on
its line: `not classified (its FLOPs and bytes not counted)`
(`micro-texture-outputs.test`), `not parsed (not counted)`
(`micro-unparsed-writes.test`), `bytes ? (not counted)` where an
access's size is not known (`micro-tensor-copies.test`), `call (its
callee not counted)` (`micro-call-returns.test`). On the kernel's `.entry` line, what the counts take
of the launch, those given: `grid X×Y×Z`, `block X×Y×Z` and each
bound parameter's value (`k5-annotate.test`); then the shared memory
each CTA declares, `shared memory 2,048 B/CTA`, `+ dynamic` where an
`.extern .shared` array's size is set at launch
(`gluon-attn-fwd-ws-annotate.test`); then what the counted statements
sum to at the counts' scale, `per launch:` and each pipe's FLOPs by
precision and each space's bytes loaded and stored, `>=` where a
statement not counted could add to it, `?` where none is counted,
`? B (trips of <header>)` under a loop whose trips are not known
(`k5-annotate.test`, `micro-totals-annotate.test`). Not printed: the
ncu comparisons, which test the analysis (`--ncu` gives the launch;
`tests/ncu_corpus.rs` keeps them); a branch target that matches no
label, whose edge the CFG drops
(`unresolved_target_is_recorded_never_panics`).

**Input contract.** The PTX is taken to come from a CUDA or Triton
program run without undefined behavior, whose integer arithmetic on
addresses, trip counts and thread-index comparisons does not wrap its
type, and whose signed comparisons read values in their signed range.
A value of unknown range is read in the integers on that premise; one
of known range keeps the value it computes, and an operand whose range
crosses an edge of its type's values is read past its multiples of
2^width where its form allows and else refused (the entries below). A
thread selector reads an unsigned comparison's operands as
unsigned where each has a known range lying in [0, 2^width) shifted
by a multiple of 2^width, or is proven nonnegative and so, of unknown
range, below 2^(width−1) by this contract (`n + %tid.x <u −16` reads
as n + %tid.x < 2^32 − 16,
`an_unsigned_setp_reads_an_operand_its_caller_takes_as_nonnegative`);
else as signed, every symbol taken as nonnegative, and x <u y as
0 ≤ x and x < y where y is proven nonnegative, since C's
`(unsigned)i < n` wraps a negative `i` on purpose
(`micro/setp_unsigned.ptx`: `%tid.x − 16 <u 32` stores on threads 16
to 47; `micro/setp_unknown_range.ptx`: `n + %tid.x <u n + 32` stores
on 32 threads, where n = −8 would leave out threads 0 to 7). C++ defines unsigned wrap and the conversion of an unsigned value
past 2^31 − 1 to `int`, so the premise is stronger than the absence
of undefined behavior where such a value reaches an address, a
comparison or a widening: that file's
third kernel, `add.u32` of n and `%tid.x` compared `setp.lt.s32` with
n + 32, reads `%tid.x < 32`, and n = 0x7fffffe0, which makes n + 32
the signed minimum and the store run on no thread, is outside the
contract (Codex's review of 2b2731b). A signed division, remainder or
sign-bit shift (PTX ISA §9.7.1.10: "The behavior for negative numbers
is machine-dependent") reads its dividend as nonnegative under the
contract only where no bound shows it negative, its range over the
launch nor its least value with every symbol nonnegative
(`micro/signed_division_symbol.ptx`: `div.s32` of n − 5 is refused,
where an RTX 4090 run rounds n = 2's −3/4 to 0); of no bound, as a
uniform value's, it is read so.

## How often each output is exact

Each item of the annotated PTX over the kernels of the Nsight Compute
records but `micro/`'s hand-written ones, each under its record's
launch. Exact: a count or formula with no `<=`, `>=`, `a–b` range or
`?`; bound: one of the first three; `?`: printed as not known; no
number: printed without one (an operand's `addr ?` or TMA copy, a
loop's sectors). `tests/output_tallies.rs` recomputes the table and
fails where it is stale; `PTXROOF_WRITE_PLAN=1 cargo test --test
output_tallies` rewrites it.

<!-- tests/output_tallies.rs -->
47 kernels.

| Output | Count | Exact | Bound | ? | No number |
|---|---|---|---|---|---|
| `.entry`: shared memory per CTA | 36 | 1 (3%) | 35 (97%) |  |  |
| `.entry`: FLOPs of a pipe it prints | 119 | 44 (37%) | 15 (13%) | 60 (50%) |  |
| `.entry`: global load bytes | 47 | 22 (47%) | 11 (23%) | 14 (30%) |  |
| `.entry`: global store bytes | 47 | 27 (57%) | 5 (11%) | 15 (32%) |  |
| `.entry`: shared load bytes | 36 | 16 (44%) | 9 (25%) | 11 (31%) |  |
| `.entry`: shared store bytes | 36 | 15 (42%) | 12 (33%) | 9 (25%) |  |
| loop: trips | 194 | 31 (16%) |  | 163 (84%), 135 mbarrier waits |  |
| loop: global bytes requested per CTA | 16 | 14 (88%) | 2 (13%) |  |  |
| loop: global bytes unique per CTA | 16 | 6 (38%) | 4 (25%) | 6 (38%) |  |
| loop: global sector bytes per CTA | 16 | 6 (38%) | 4 (25%) |  | 6 (38%) |
| block: threads | 854 | 825 (97%) | 29 (3%) |  |  |
| block: CTAs | 85 | 85 (100%) |  |  |  |
| block: `runs when` | 147 | 147 (100%) |  |  |  |
| block: exec | 818 | 286 (35%) | 112 (14%) | 420 (51%) |  |
| instruction: threads, where a guard narrows its block's | 952 | 778 (82%) | 174 (18%) |  |  |
| instruction: exec, where it differs from its block's | 651 | 556 (85%) | 91 (14%) | 4 (1%) |  |
| instruction: `unused` | 954 | | | | |
| instruction: `not classified` | 423 | | | | |
| instruction: `bytes ?` | 109 | | | | |
| operand: address | 4,399 | 4,050 (92%) |  | 349 (8%) |  |
| operand: global sectors/req | 2,221 | 1,491 (67%) | 447 (20%) |  | 283 (13%) |
| operand: global sectors/req, ideal | 1,938 | 1,772 (91%) | 166 (9%) |  |  |
| operand: shared wavefronts/req | 2,178 | 1,442 (66%) |  | 557 (26%) | 179 (8%) |
| operand: shared wavefronts/req, ideal | 1,442 | 1,442 (100%) |  |  |  |
<!-- end tests/output_tallies.rs -->

## Known limitations

### Wrong output

- **Static shared bytes leave out the padding alignment puts between
  declarations.** "The variable will be aligned to an address which is
  an integer multiple of byte-count" (PTX ISA §5.4.5), but the static
  bytes are the declarations' sizes summed: a kernel declaring
  `.shared .align 4 .b8 a[4];` then `.shared .align 8 .b8 b[8];` reads
  `shared memory 12 B/CTA`, where `ptxas -v` (13.4,
  sm_90) reports `16 bytes smem`.
- **A vector `.shared` variable is not read.** PTX ISA §5.4.2 declares
  one as `.shared .v2 .u16 uv;`. At module scope it is dropped:
  `--dump-ast` prints no declaration of `uv` and the `.entry` line no
  shared memory; in a kernel it is an unparsed statement. `ptxas
  -v` (13.4, sm_90) reports `4 bytes smem` for either.
- **A `cp.async`'s global sectors are counted as a load's.** Nsight
  Compute counts them otherwise where two lanes' shared destinations
  fall in one bank group or are one, or two lanes read one source
  (`micro/shared_wavefronts.ptx`'s `micro_cp_async_wavefronts`, RTX
  4090, `shared_wavefronts.cp_async.ncu.json`): a 16-byte `.cg` copy
  from `16·%tid.x` takes 32 L2 sectors to `32·%tid.x` (line 633,
  predicted 16) and 8 to `16·(%tid.x mod 16)`, where lanes t and
  t + 16 write one destination (line 685); one from
  `16·(%tid.x mod 16)` to `16·%tid.x` takes 16 (line 694, predicted
  8), and the record's launch load sectors differ with them. Every
  other `cp.async` line of the records agrees (`tests/ncu_corpus.rs`).
- **Register effects are incomplete.** Affine analysis records the
  registers of the operands the register-effects decoder says an
  instruction writes, a register, a `p|q` pair or a `{…}` vector, as
  destinations (`micro/vector_def.ptx`: a guard on a register a vector
  load overwrote is a bound, `<= 9 sectors`), refuses an element of a
  multi-element `ld.param` vector rather than reading the parameter's
  bits, and does not model guarded definitions: a branch
  or guard comparing a value written under a guard is an unknown
  selector (`micro/guarded_def.ptx`: `sectors/req <= 5`, not 64 threads
  exactly), even where every write agrees
  (`micro/guarded_selector.ptx`: in its second kernel 32 threads store,
  its `.entry` line says `global store <= 256 B`). A form the
  register-effects decoder does not decode, a sub-operation it does
  not list (none in the corpus,
  `tests/register_effects_corpus.rs`), is taken to write every register
  it names, and so is a statement the parser does not read, such as one
  indexing an array, `lbuf[1]` (PTX ISA §6.4.2;
  `micro/unparsed_writes.ptx`: the address after it is unknown, where
  ptxas stores to param_0 + 20). The parser drops a module's `.global`
  variable declarations (`micro/uniform_bases_global.ptx` does not
  round-trip: `Unknown symbol 'gbuf'`) and its `.texref` declaration
  (`micro/texture_outputs.ptx` does not round-trip: `Unknown symbol
  'micro_texture_outputs_tex'`).
  Source locations, reproducible parser/classifier probes, and their outputs
  are in the [register-effects audit](docs/ptx-register-effects-audit.md#findings-in-the-current-implementation).
- **A predicated `ret` or `exit` is read as falling through.** The CFG
  gives its block one successor, the next block, so what follows counts
  on every thread, exactly (`micro/predicated_exit.ptx`: the store
  after `@%p1 exit` counts 32 per CTA, where on an RTX 4090 at n = 20
  it runs on none).
- **`trap` is read as falling through, and a block ending in it as a
  path a launch takes.** PTX ISA §9.7.20.4: `trap` aborts the
  launch. The CFG gives a block ending in `trap` the next block as its
  successor, so each of LLVM's cold `call.uni __assertfail; trap;`
  blocks runs on the threads of the ones before it, and keeps every
  edge into one, so a loop holding an assert check exits there and a
  block a check can bypass is conditional. In the Blackwell corpus (the
  census's 2,837 kernels and sol-study's 2,601 PTX files, sm_100
  family, analyzed without launch arguments) 2,628 `trap`s sit in 23
  parsed files: reading `trap` as a terminator changes 2,498 bounds of
  assert blocks in 10 of them, and dropping the edges into those blocks
  changes 9,923 lines in 11. 20 of the corpus's 24 loops refused as
  "loop has N exit edges", in Triton concurrency-sanitizer builds of
  sol-study's Gluon `mla_decode_1cta_kernel` and `mma_rate_kernel`,
  have every exit but one into an assert block; without those edges the
  five `mma_rate_kernel` loops read `trips param_1`.
- **A block in an irreducible cycle counts as running once.** A cycle
  entered at two blocks is no natural loop, so no trip count multiplies
  its blocks: `micro/irreducible.ptx`'s `$L__BB0_A` reads `exec <=
  32/launch` where its record measures its `fma.rn.f32` 320 times
  (`ncu-irreducible.test`). None of the census's 2,837 kernels has such
  a cycle.
- **Vector registers are not modelled** (PTX ISA §6.4.3): the parser
  does not read `.reg .v2 .b32 %v;`, and an element `%v.x` is a register
  of its own, apart from `%v`. In `micro/vector_register.ptx`, after
  `ld.global.v2.b32 %v, [%rd1]` overwrites `%v`, `%v.x` is still the
  thread index an earlier `mov.b32 %v.x, %r1` wrote: the address is
  `[4·%tid.x + param_0]`, where ptxas takes it from the load.
- **Integer arithmetic is taken not to wrap where its range is
  unbounded**: a value through a parameter, `%ctaid` with no grid given
  or a symbolic trip count is read under the input contract above. An
  `add.sat` or `sub.sat` is read only where its operands and its sum
  lie in its type's values (`a_saturating_add_reads_a_sum_its_type_holds`).
  A conversion to or from floating point, or
  a pack, is refused (`micro/cvt_reads.ptx`).
- **An operand read as its type is refused where its range crosses the
  edge of the type's values.** An integer `cvt`'s source and result, a
  `.wide` product's, `shr`'s, `div`'s, `rem`'s, `min`'s and `max`'s
  operands and a signed `bfe` field from bit 0 are read as the integer
  their low bits are (PTX ISA §9.4, §9.4.1), the value less a multiple
  of 2^width, where its range over the same bounds lies in one window
  of the type's values (`micro/cvt_reads.ptx`: `cvt.u32.u64` of
  `%tid.x` + 2^32 is `%tid.x`; `micro/trip_max.ptx`: `min.u32` of n − 4
  under `--bind 1:n=2` is 8, the 8 trips its record measures), and as
  itself where the range is unknown, under the input contract. Where
  the range crosses, the value below 63 bits reads past its multiples
  of 2^width, the rest shifted into the values where it lies in one
  window and else as `divmod` reads its remainder (unit test
  `an_operand_reads_as_its_type_in_the_one_window_of_its_range`:
  2^32·%tid.y + %tid.x as a `u32` is %tid.x, 2^28·%tid.x is
  2^28·(%tid.x mod 16)); else each thread's value is one of two forms
  and the operand is refused (`micro/wide_operands.ptx`: `%tid.x` − 256 as
  `mul.wide.u32` reads it, 2^32 + `%tid.x` − 256 below `%tid.x` = 256,
  "reads a value its type does not hold"; `micro/division_operands.ptx`'s
  `div.u32` and `shr.u32`).
- **An and-mask that clears the sign bit keeps a value of unknown range
  as itself.** Up to the bit below the sign bit (nvcc's
  `& 0x7fffffc0`) the and is read as the value itself where its range
  lies in [0, 2^(width−1)), and refused where the range reaches outside
  it (`micro/and_sign_bit.ptx`: `(%tid.x − 256) & 0x7fffff00`, "clears
  bit 31 of a value that may set it"). A value through a parameter or
  `%ctaid` with no grid given has no range and is read under the input
  contract: `micro_and_sign_param`'s `(n − 256) &
  0x7fffff00` reads `[param_0 + 4 * param_1 - 4 * ((param_1 - 256) mod
  256) - 1024]`, where for n below 256 the and is 0x7fffff00 and the
  store 2^33 bytes higher; with `--bind 1:n=100` it is refused, and the
  address a uniform variable. A comparison of such a value with its
  mask then cancels: `micro/setp_unknown_range.ptx`'s fourth kernel
  reads `n < n & 0x7fffffff` as false, the store on no thread, where a
  negative n makes it true (Codex's review of 2b2731b; constructed). A
  16-bit and, and a 32-bit mask of more
  runs, take the remainder mod 2^(width−1) instead
  (`micro/and_sign_remainder.ptx`'s `n & 0x7fff`: `param_1 mod 32768`;
  `micro/and_runs.ptx`'s `micro_and_sign_runs`), and the runs' sum,
  its remainders ranged one by one, as reaching below 0 and read mod
  2^32: `4·((param_1 mod 16 + param_1 mod 2147483648 − param_1 mod
  256) mod 4294967296)`, the same value; a 64-bit mask of more
  runs is refused (`micro_and_sign_wide_runs`: a uniform variable).
- **`shfl.sync.idx` is read as if warps were rows of 32 `%tid.x`**,
  whatever the block shape: a warp is 32 consecutive
  `%tid.x + %tid.y·%ntid.x + %tid.z·%ntid.x·%ntid.y` (PTX ISA §3.1,
  §9.7.16.1), so in a block of more than one row whose `%ntid.x` is no
  multiple of 32 the lane read is another thread's. In
  `micro/shfl_2d_block.ptx` (`.reqntid 48, 2, 1`) every lane of a warp
  stores at lane 0's `%tid.x` (warp 2's lane 0 is thread (16, 1),
  param_0 + 64; ptxas 13.4); the store reads
  `[128·⌊%tid.x/32⌋ + param_0]`, `sectors/req 1–2`. `%laneid` is
  read only in a block of one row.
- **`shfl.sync.idx` reads one floor or mod of `%tid.x`**: a thread
  index under more, `⌊%tid.x/8⌋ mod 16` from a `bfe.u32` or
  `⌊(%tid.x mod 32)/4⌋` from `%laneid`, is no form, only the
  shuffle's value, one for a warp (`uniform(%r6@57)`), even where it is
  the same across a warp (`⌊⌊%tid.x/8⌋/4⌋`, whose shuffle ptxas drops);
  a fourth operand other than 31, which splits the warp into
  sub-segments or clamps the lane (PTX ISA §9.7.9.6), leaves it
  unknown: `micro/shfl_idx.ptx`.
- **The warp index is one value per warp only through a shuffle.** The
  uniformity analysis reads `⌊%tid.x/32⌋` as divergent, though in a
  block of one row, or whose `%ntid.x` is a multiple of 32, every lane
  of a warp holds the same (PTX ISA §3.1): a `brx.idx` on a load at
  `param_0 + 4·⌊%tid.x/32⌋` selects its targets' threads as an
  unknown, where with the index shuffled from lane 0 first, as Triton's
  warp specialization does, each target runs on whole warps
  (`a_jump_table_on_a_warp_s_index_selects_whole_warps`). So is an
  and-mask keeping bits of `%tid.x` above its lane bits, `and.b32 d,
  %tid.x, M` with M a multiple of 32, Triton's layout bits: b200
  `attn_lightllm__deepseek-v3__decode._fwd_kernel_flash_decode_stage1_padding`
  line 950's `ld.global.v4.b32` reads `addr ? (depends on a value
  loaded inside the loop)`, its index loaded at line 909 at an address
  built from line 98's `and.b32 %r2, %r1, 64`. In the Blackwell corpus
  (`trap` entry above) the shift, `bfe` and division forms of the warp
  index are 3,757 statements in 2,545 files and the masks 6,829 in
  3,746; reading both as one value per warp turns 8,162 `addr ?` into
  addresses.
- **A parameter load at an offset reads the whole parameter** (PTX
  ISA §5.1.6.1): `ld.param.u32 %r3, [p+4]` reads as `param_1`, like
  the load at offset 0, so comparing two fields of a by-value
  aggregate is always true (`micro/param_fields.ptx`: the store runs on
  every thread, `store 256 B` per CTA).
- **Integer literals are read in decimal and hexadecimal only**; PTX
  ISA §4.5.1 also writes them in octal and binary, and with the suffix
  `U`. In `micro/int_literals.ptx`, where ptxas folds each offset into
  the store, octal `020` reads as twenty (`[param_0 + 20]`, ptxas
  `[R2.64+0x10]`), and `0b10000` and `16U` as symbols (`[param_0 +
  0b10000]`). In the census the forms are 7,019 `prmt` selectors with
  a `U` (`0x3340U`, 119 kernels).
- **A `!=` latch with a symbolic distance is taken to meet its
  bound**: the trip solver divides the distance by the step, rounding
  down, as `src/analysis/affine/exit_limit.rs`'s `exit_limit` assumes the
  division exact (k1's and k2's `K − K mod 4`, the fp8 GEMMs,
  `micro/two_register_counter.ptx`). A constant distance that is no
  positive multiple of the step is refused: `micro/ne_latch_unreached.ptx`,
  stepping by 2 from 0 while `!= 5`, reads `loop: trips ?`, where the
  loop does not exit; a bound distance reads as the constant it is.
  A distance of 0 is refused even where a guard skips the loop: k2's
  remainder loop under `--bind 2:K=4096` reads `trips ? (latch
  condition never meets its != bound) | no thread`
  (`tests/lit/k2-bind.test`), and `tests/gen_loops.py`'s seed-1 kernel
  5 at n = 64, which its guard skips, is unsupported.
- **A loop's count drops its max with 0 or 1 on nonnegative
  parameters.** A loop is counted max(T, 0) per execution of a guard
  that skips it where its latch fails at k = 0 (`micro/loop_guard.ptx`),
  and max(T, 1) per entry otherwise (`micro/latch_exits_first.ptx`);
  the max is dropped where T is never below it under the bindings,
  every unbound parameter and every value a count reads as `uniform(…)`
  taken as nonnegative (`src/analysis/affine/sym_expr.rs`'s doc), and
  kept where a binding makes it below (`micro/negative_bind.ptx` under
  `--bind 1:n=-3`: 0 and 1 trips, as an RTX 4090 runs them):
  `micro/single_loop.ptx` reads its `ld.param.u32` n as `param_1`,
  with exact totals, where an n of 2^31 or more, below 1 as its
  `setp.lt.s32` compares it, runs no trips.
- **Addresses affine analysis cannot read are unknown rows**, with the
  reason, unless the lanes of a warp compute the part it cannot read
  alike: that part is then a variable, `uniform(%r54@90)`, or,
  where each of its definitions is an unguarded `shl` or `and` by a
  constant that leaves its k low bits 0, 2^k times its floor over 2^k,
  so an `or` with bits below 2^k adds (`micro/uniform_aligned.ptx`),
  but a floor by a divisor 2^k neither divides nor is divided by is
  then refused (`micro_uniform_aligned_div`: ⌊(3·%tid.x + b)/3⌋, which
  `uniform(…)` alone read)
  (the grouped tile schedule's
  `pid / num_pid_n` in the Gluon fp8 GEMM's global copies, k1's
  remainder loop reading the main loop's final k;
  `micro/uniform_address.ptx`). An xor reads when each side is an xor
  of thread index fields, `%tid.x` or a floor or remainder of it by a
  power of two, times a constant it carries nothing under, no two of a
  side's fields setting one bit, and constants: one variable, the xor of
  its runs of thread bits (`micro/xor_swizzle.ptx`, Triton's swizzle as
  `(16 * (⌊%tid.x/8⌋ mod 4) ⊕ 16 * (%tid.x mod 128))`, its offsets read
  back on the RTX 4090; census 02431.ptx:1769's 8208 times bit 3, a
  one-bit signed `bfe` masked, and 03874.ptx:2057's 2112 times bit 0, a
  `selp` of 0 or 2112 by it; 36,725 census rows). Not read: an xor with
  a side that is no such xor, "behind `xor.b32`" (237 of the census's
  21,154 unknown address rows, in 49 kernels, all but three shared, of
  its 800 shared rows with no address): a sum whose fields share a bit
  (`micro_xor_overlap`). Run on every `%tid.x` below its `.reqntid`, 12
  of the 88 xors not read in those kernels are an xor of `%tid.x`'s
  bits, 70 are one only within each warp, its bits differing by warp,
  and 6 read a register written more than once, a `prmt` or a load,
  which the evaluation does not follow. The first instruction not read
  on an xor's operand chains is an `or` whose sides set one bit on some
  thread (77 xors, 43 kernels; 56 `or`s, as `00196.ptx`'s line 934 a
  warp bit or'd onto a lane bit, `%tid.x | (⌊%tid.x/4⌋ & 8)` in 64
  threads) or a `selp` by no equality of a read thread bit (8, 8: an
  `lt`, a register chosen, a bit an unread `or` sets); an xor may stop
  at more than one.
  Not read either: divisions of the
  thread index by a runtime value (`micro/uniform_address.ptx`'s
  `micro_uniform_div_lanes`; 1,622 census rows "div by a non-constant
  divisor" and 3,525 "rem by a non-constant divisor": of 40 sampled,
  binding the kernel's 32-bit parameters reads 24 past the division,
  as census kernel `01155.ptx`'s `div.s32 %r11, %r8, %r9` of `%r9` =
  param_5, and 16 stop at a divisor of the CTA index, Triton's grouped
  tile schedule's `min(num_pid_m − first_pid_m, 8)`; none at a loaded
  divisor); an and-mask's run of bits or a `rem` whose remainder
  the affine form does not hold, of a form whose offset, past its
  multiple of the modulus, is not below the coefficient of the term the
  modulus does not divide and which that term's range, from the launch
  or as a remainder, can carry past the modulus
  (`micro/and_runs.ptx`'s `micro_and_offset`, `(%tid.x − 4) & 0x46`:
  "in a mod"; `micro/rem_offset.ptx`'s `micro_rem_offset_wraps`,
  `((⌊%tid.x/8⌋ mod 16) + 56) rem 64`: "in a division") or a negative
  coefficient, but for
  a value of 0 or −1 under a mask below the sign bit
  (`micro/and_bit_select.ptx`; refused with the sign bit,
  `micro_and_select_sign`, and for −(%tid.x & 3),
  `micro_and_select_two_bits`), or a 64-bit mask's run that starts at
  bit 63 (`micro/and_runs.ptx`'s `micro_and_bit63`); a mod or shift of
  a value read as a variable (898 census rows, "depends on
  ⌊uniform(…)/2^k⌋, in a mod" or "depends on uniform(…), in a
  shift"); a signed `bfe` whose field starts past the msb, which ptxas
  13.4 folds to −1 on every thread where the hardware takes the sign
  bit (`micro/bfe_fields.ptx`: "bfe at 32 of 1 bits", line 41's store
  1 L2 sector in `micro/bfe_fields.ncu.json`, line 46's, its position
  given at run time, 2), or of two bits or more from past bit 0 (no
  census `bfe`); values loaded
  from memory that may differ between threads (the fused MoE kernel's
  `cp.async` of the rows `sorted_token_ids` selects, its source line
  262, `moe_sglang__deepseek-v3__prefill__gate_up.fused_moe_kernel`:
  "depends on a value loaded from memory"); an `or` whose sides may
  both set a bit, by the bits the affine forms bound (`AffineExpr::may_set`):
  a bitwise or that is no add (`micro/or_disjoint.ptx`'s
  `128·⌊%tid.x/32⌋ | 16·%tid.x`: "or whose sides may both set bit 7"),
  or one side's bits unknown, a parameter not bound
  (`micro_or_param`), a value that may be negative
  (`micro/or_negative.ptx`) or a value every thread reads alike with a
  definition other than an unguarded `shl` or `and` by a constant
  (`micro/uniform_aligned.ptx`'s `micro_uniform_aligned_guarded` and
  `micro_uniform_aligned_count`); a division by a constant that is no power
  of two, which LLVM writes as `mul.hi` by a magic number and a shift
  (`micro/magic_div.ptx`: "`mul.hi.u32` keeps a product's upper half";
  sglang `bmm_kernel_persistent`'s `/ 148` leaves its trip count
  unknown); a floor or mod of a value that, past its multiple of the
  divisor, is neither a term c·v, c a divisor, plus a rest the ranges
  keep in [0, c), nor symbols whose factors share a divisor a of d plus
  a rest in [0, a), nor in one block of d over its range, such as a 0-based
  counter's `i % 2` read before its increment, `(k − 1) mod 2`, which
  is the same for every thread and so
  a uniform variable the loop defines (`micro/counter_mod_offset.ptx`:
  `requested 16 B, unique unknown` where ptxas stores 8 distinct bytes;
  k12's double buffer `tile % 2`, `uniform(%r10@365)`; census kernel
  `00337.ptx`'s signed remainder `x − ((x + ((x >> 31) >>> 26)) & −64)`
  of `x = param_4 + 1 + i` in its loop, its lines 60–66: "shift of an
  induction variable"); an element a
  `mov.b64 {…}` unpacks from a register no single `mov` packed, after a
  join of packs (`micro/unpack.ptx`'s `micro_unpack_join`; b200
  `bmm_kernel_persistent`'s `cp.async` sources, whose `%rd175` three
  `mov`s pack, lines 372, 540 and 711), with another element
  count (`micro_unpack_halves`), from 64-bit arithmetic
  (`micro_unpack_wide`) or from an unpacked element
  (`micro_unpack_twice`), or packed as an immediate wider than its
  element (`micro_unpack_imm`), or one a `mov.b128 {…}` unpacks, no
  form the analyses read (`selections_and_moves_take_one_type`; the
  Blackwell corpus's one file with `mov.b128` only packs). In the
  Blackwell corpus (`trap` entry above), not read either: an `or` in
  an `ldmatrix` or `stmatrix`
  `.x1`/`.x2` address whose sides share a bit only on lanes that give
  the instruction no address (PTX ISA §9.7.15.5.15–16: threads 0–7 or
  0–15), 2,962 of the 4,443 such rows refused "or whose sides may both
  set bit N", as b200 `attn_lightllm__llama3-8b__prefill` line 1300's
  `ldmatrix…x1`, its or at line 377; a `prmt` (89 rows, 43 files); a
  64-bit signed shift by 63 of a form with variables, the sign word of
  LLVM's signed division, "shift by 63 is out of range" (b200
  `sg_decode_BN32_w4_s2._fwd_grouped_kernel_stage1` line 1055; 59 rows,
  28 files); `min.relu` and `max.relu` (56 statements, 4 files).
- **Offsets past 2^62 bytes are not counted**: an access whose lane
  offsets over the block, or whose loop's shifts over its trips, reach
  2^62 bytes has unknown sectors per warp or unknown unique loop bytes
  (`micro/offsets_wide.ptx`: `sectors/req ? (lane offsets reach 2^62
  bytes)`; `requested 32 B, unique ?` for 8 distinct stores).
- **A loop footprint takes an address holding an unknown value the
  loop computes, or one that is one value per warp only, as unread**:
  requested bytes only, unique and sector bytes unknown
  (`micro/uniform_address.ptx`, `micro_uniform_in_loop`: requested
  2048 B, unique unknown; `micro/footprint_unknown.ptx`, a pointer the
  loop loads: requested 64 B, unique unknown).
- **A load is taken to give the lanes of a warp one value where its
  address is the same for them**: a weak `.global`, `.shared` or
  `.const` load, as LLVM's NVPTX and AMDGPU targets take it
  (`isSourceOfDivergence`, AMDGPU: "if threads issue loads with the same
  arguments, they will always get the same result"). PTX ISA §8.5 does
  not give it ("the warp is not a scope; the CTA is the smallest
  collection of threads that qualifies as a scope"): this assumes the
  hardware executes the load once for the lanes executing it together.
  A warp-specialized dispatch reads its state byte so
  (`micro/warp_dispatch.ptx`: the dispatched store's sectors read
  `sectors/req 4 (ideal 4)`, `micro-warp-dispatch.test`); over the
  65 records the access rows whose warp sectors or wavefronts are a
  bound go from 333 to 201, all in `gluon/attn_fwd_ws_kernel`'s two
  shapes. The records keep thread instructions only
  (`tools/ncu/measure.py`), not the warp instructions per line that
  would check it. The cross-entropy's target column, its index loaded
  through param_1, reads `[65536·%ctaid.x + 2·uniform(%rd14@86) +
  param_0]`, 1 sector, as its record measures (`gluon-ce-chunk.test`).
  A `.volatile`, `.relaxed`, `.acquire`, `.mmio` or generic load is
  never one value (`a_global_load_at_one_address_is_one_value_per_warp`).
  No load is one value for the CTA, so a loop bounded by a loaded value
  has no trip count: 9 counts over the 65 records
  (`attn_lightllm__llama3-8b__decode._fwd_kernel_flash_decode_stage2`'s
  `$L__BB0_2`: "uniform(%r4@79) is one value per warp only: depends on
  uniform(%r19@75), in a min", its sequence length), 240 of the
  census's 1,230.
- **Unique bytes per loop assume different pointer parameters are
  disjoint** (the cross-entropy's second pass, `tests/lit/
  gluon-ce-chunk.test`: the load and the store count 131,072 B
  unique for the 131,072 B requested; aliasing would halve it), and
  an offset from a pointer to stay in its allocation; bases on one
  pointer, or naming no single pointer, may overlap and give a range
  (`micro/one_pointer_two_bases.ptx`: a[k] and a[k + n] count
  16–32 B, 16 B with `--bind 1:n=0`; `micro/uniform_bases.ptx`), count
  the loop's own blocks only, and need a numeric trip count, the block
  shape, constant lane and counter coefficients, every global access's
  bytes (else no footprint, `micro/tensor_copies.ptx`) and address,
  its unknown values one for the CTA and no count of a loop the loop
  runs again on every trip (else `unique unknown`,
  `micro/footprint_unknown.ptx`; `micro/inner_exit_counter.ptx`, an
  inner loop's count k = 1, 1, 2, 3: requested 16 B, unique unknown,
  where an RTX 4090 run stores 12 distinct bytes), and
  are bounds where which threads run the loop is not known. k5's K loop at
  256³ (`k5-footprint.test`): requested 65,536 B, unique 65,536 B;
  k1's (`k1-footprint.test`): 1,048,576 B, 32,768 B, a bound until M and
  the grid are given, which its guard `x < M && y < N` reads.
- **A loop footprint takes the alignment of an access that may never
  run as its base's.** PTX ISA §6.4.1 makes only an access that runs
  aligned: `micro/optional_align.ptx` reads `in sectors <= 192 B`, its
  16 B store at p making each byte store one sector, where with n == 0
  and p ≡ 17 mod 32 (`--ptr-align 1`) the byte stores request 256
  (`micro-optional-align.test`).
- **The threads and sectors of an access under `elect.sync` read the
  lowest active lane as the elected one**, where its count tries every
  active lane (PTX ISA §9.7.14.15 does not say which is elected):
  `micro/elect_lane.ptx`'s store under `%tid.x == 1 and elected` runs
  `exec <= 1/CTA` but reads `no thread` and `sectors/req 0`, where lane
  1, if elected, requests one (`micro-elect-lane.test`). The census has no election
  under a condition that varies by lane.

### Reported as unknowns

- **Shared wavefronts are not counted for a `cp.async` destination, a
  shared atomic, or a 16-byte load two of whose lanes share a word**
  (13,489 and 6,723 census rows, 832 of the loads by name, every lane
  at one address). No NVIDIA document gives a copy's
  rule: the Nsight Compute Profiling Guide does not name LDGSTS, and
  the CUDA Programming Guide's LDGSTS section says only that 4- and
  8-byte copies go through L1, that 16-byte ones can bypass it, and
  that both addresses aligned to 128 bytes perform best. A copy's
  wavefronts follow its global source as well as its destination: to
  `16·%tid.x` 4 from `16·%tid.x`, 8 from
  eight 64-byte rows, 16 from sixteen 32-byte rows, 32 from
  `128·%tid.x`, and to `32·%tid.x` 32 from `16·%tid.x` by `.cg`; no
  record copies 16 bytes by `.ca` (`micro/shared_wavefronts.ptx`'s
  `micro_cp_async_wavefronts`, RTX 4090). One wavefront per 128-byte
  line the lanes' sources fall in, times the bank conflicts of those
  lanes' destinations, lanes writing one destination counted once
  (line 685: 8 L2 sectors, 2 wavefronts), gives 19 of that record's 20
  copies and 288 of the 312 copies whose two addresses read in the 22
  records, every term but the thread index's taken as 0 (the
  destination alone as a store: 240). It misses line 633, 32 where it
  gives 8, taking 32 L2 sectors where its 512 bytes span 16; a copy
  with an ignore-src predicate or a src-size register
  (`micro/cp_async_ignore_src.ptx` line 36,
  `micro/cp_async_src_size.ptx` line 30: 5 where 4); and 21 averages
  that are no integer, as the Gluon `attn_bwd_kernel`'s line 1914,
  5.70 per request with an ideal of 4.0, from 128-byte source rows.
  The 16-byte loads' measured counts follow no grouping found:
  `16·(%tid.x mod 8)` takes 4, `256·⌊%tid.x/16⌋` 2
  (`micro_shared_wavefronts`).
- **A block under a loop whose trip count is not read is unknown in
  the totals per launch**, as its trips may differ by CTA
  (`micro_loop_guard_sibling_exit`'s first loop, with two exit edges:
  `bar.sync <= 320 + unknown` under `micro/loop_guard_sibling.ncu.json`'s
  grid). Under a count that reads the CTA index
  a block is counted over the launch as the count summed over the
  grid, CTA by CTA where which of its threads run varies by CTA
  (`micro/cta_trips.ptx`: 192 and 224 barriers over 4 CTAs, as
  measured on an RTX 4090), and stays unknown where those threads are
  not counted at every CTA, past 2^24 CTAs on the axes the count
  reads, or where the count at a CTA is negative or not a constant.
- **A floor or mod can print a term that is always 0, or a remainder of
  a remainder**: where the rest of a term c·v lies in [0, c), the floor
  and the remainder keep ⌊v/(d/c)⌋ and v mod d/c even where v's range
  makes them 0 or v (census `01452.ptx`'s first load:
  `4096·⌊(%tid.x mod 128)/256⌋ + 16·((%tid.x mod 128) mod 128)`); the
  value is exact.
- **A product of two forms with variables is unknown**: `mul` and
  `mad` need one factor free of variables, after the bindings, which
  give a symbol its value where it is loaded (`micro_floor_bound_zero`
  reads `[param_0]` with `--bind 1:p=0`). 400 lines of 81 corpus files
  read "in a product" (`grep -c 'in a product'` over the 5,668-file
  run).
- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, nvcc's
  unrolled main loop and its remainder, LLVM's two-register counter and
  its predicate-phi two-trip loop, and a latch branching on a `setp` in
  a block of its loop that dominates it, as Triton's pipelined loops
  do (`tut_sg_prefixdense_tma_BM128_BN64_w8_s2_wsFalse.attn_fwd`'s
  `$L__BB0_8`: 31 trips, five of its blocks confirmed by its record).
  Not recognised, reported as
  `loop: trips ?` and why: grid-stride loops without `--grid`, whose step is
  `%nctaid.x` ("induction step is not a constant"), data-dependent
  bounds, multi-exit loops, inner loops bounded by an
  enclosing loop's counter (`micro/triangular.ptx`), a latch `setp`
  that combines its comparison with a predicate
  (`a_comparison_combined_with_a_predicate_is_not_a_trip_count`) or
  compares floats (`micro/latch_float.ptx`: "latch condition compares
  floating-point values") or packed halves (`micro/latch_forms.ptx`:
  "latch condition compares packed halves"), a bound one
  value per warp only (a load, or
  arithmetic on one), the mbarrier spin-waits
  (`mbarrier.test_wait` defines the predicate), and the persistent
  tile loop, which has a latch and an exit per partition. A bound
  affine analysis does not read, the same for every thread, is the value of the
  register whose trace fails, `uniform(%r3@30)` for a division by a
  parameter not bound (`micro/trip_div_param.ptx`, `param_1 / 4` with
  `--bind 2:d=4`), taken as one value per CTA, so the totals per launch
  under it are unknown (`micro-trip-div-param.test`), and unknown where
  a loop holding the latch defines it or threads leave the loop defining
  it at different trips (`micro/trip_uniform.ptx`). Bounds the affine
  analysis does not read, by the loop's source line: a float immediate and an
  `and.pred` latch predicate (sol-study census
  `00511.ptx` `:121`, `00908.ptx` `:155`), and an `atom` result
  (`01542.ptx` `:223`). In `gluon/attn_fwd_ws_kernel.sm_89.ptx` the
  persistent loops (`gluon_attn_fwd.py:126`, exit not at the latch;
  `:213`, several latches) and the causal loop (`:143`, its bound
  carried around the tile loop) are unknown. `:126` leaves at
  `@%p62 bra $L__BB0_62` (PTX 2552), source line 208's test, which
  reuses line 132's `setp` (502); its latch computes the next tile's
  parameters (209) and branches back unconditionally. `:213` is
  Triton's warp-specialize switch loop: the worker warps read their
  state byte (2617) from the word the default warps store, 0 at 121
  and `0x01010101`, return, at 5116, and each partition's end is a
  latch. b200 `tut_ll_BM128_BN128_w4_s2_wsTrue.attn_fwd` dispatches
  the same way through `brx.idx`, and `scan_tma_ws_best` too, whose
  worker k-loops read their bound from shared memory the default warps
  store (637, loaded at 1820 and 1917). Under the attn_fwd_ws record
  the FLOPs over the launch read `cuda-core, tensor, sfu flops ?`
  (`gluon-attn-fwd-ws-annotate.test`), where the record's
  `mma.m16n8k16` lines run 3,244,032 warp instructions,
  13,287,555,072 FLOPs.
  Over the census's 3,027 sm_100a loops (`missed-opt-census/art/sm100v3`,
  inferred arguments), 990 have a trip count: 1,445 are mbarrier
  waits, 315 read a value one per warp only (205 name a load), 95 a
  value an enclosing loop defines, 30 a loaded value lanes may see
  differently, 95 are grid-stride loops, 20 atomics, 18 multi-exit.
  Of the 96 loops whose step is not a constant (74 kernels), 83 have a
  count with `--grid 4,4,4`; the others step by a parameter, read with
  it bound: `02395.ptx`'s `$L__BB0_3` (line 80), "induction step is not
  a constant" under `--grid 4,4,4`, reads `max(⌈(param_10 −
  %ctaid.x)/16⌉, 1)` with `--bind 13:s=16` as well. A latch that holds
  no instruction is not read: the CFG starts a block at the statement
  after a branch, so an untargeted DWARF label there (`@!%p12 bra
  $L__BB0_18; $L__tmp28: $L__BB0_21:`) is a block of its own, the
  loop's latch, and the exit is "not at the latch" (sol-study Gluon
  `dense_ws_kernel`'s persistent loop, 3 files; without the label it
  reads `trips max(⌈(⌈param_17/128⌉·⌈param_18/256⌉ − %ctaid.x)/148⌉,
  0)` under `--grid 148,1,1`).
- **Counters the induction variable search refuses** (a header φ every back
  edge feeds back plus one unguarded step in the loop's own blocks, or
  a directly nested loop's exit after a constant trip count,
  `src/analysis/affine/induction_variables.rs`): a register stepped by a guarded
  write (`micro/guarded_step.ptx`, where ptxas 13.4 stores once at
  param_0) is no counter; its guard the same for every thread, it reads
  `[4·uniform(%r1@29) + param_0]`, and the loop's unique bytes are
  unknown (`micro-guarded-step.test`). Nor is a stage counter that
  wraps by `selp`, `add r1, r, 1; setp.gt.s32 p, r1, S−1; selp.b32 r,
  0, r1, p`, the buffer index of TMA and `tcgen05` pipelines: b200
  `blk_mx_128x256_s4w4.mx_block_scaled`'s `selp.b32 %r841, 0, %r146,
  %p72` (line 488) gives its stores `[16384·uniform(%r841@488) +
  global_smem + 131072]`; in the Blackwell corpus 4,369 such counters
  in 1,252 files, 17,133 address rows. Nor one stepped through `cvt`
  (`cvt.s64.s32`, then `cvt.u32.u64` and the `add`): census
  `00057.ptx`'s `$L__BB0_2` (line 104) reads "is defined in an
  enclosing loop: depends on %r18, carried around an enclosing loop",
  6 census loops.
- **A join's φ whose paths write one value through a loop counter is
  refused**, though both definitions and the join lie in one trip of
  the loop: `micro/join_same_value.ptx`'s `micro_join_counter`,
  128·i + %tid.x on both paths (words 0 to 255 on an RTX 4090), reads
  "depends on special register %tid.x", as census kernel `00376.ptx`'s
  four `st.global.v4.b32` of its source line 74 do.
- **Instruction families.** 67 of the 232 rows in
  `docs/ptx-instruction-coverage.md` are `Unknown`: integer, sparse
  and block-scaled `mma`; `wgmma`; `tcgen05` but its allocation,
  waits, fences and commit; bulk copies and the tensor copies but
  `cp.async.bulk.tensor`; textures and surfaces; `multimem`; video
  instructions. `tcgen05` is in 12 of the B200 fixtures
  (`tests/fixtures/b200`).
- **A tensor copy's bytes are unknown.** `cp.async.bulk.tensor` moves
  the bounding box its tensor map holds (PTX ISA §5.5.3.1, §5.5.8):
  the bytes of both its sides are unknown in every total,
  `bytes ? (not counted)` on its line, its global side reads `addr ?
  (the tensor map holds the address)`, its shared side has no
  wavefronts and no reason, and a loop holding one in its own blocks
  has no footprint and says nothing of it (`micro/tensor_copies.ptx`,
  `micro-tensor-copies.test`; in the Blackwell corpus, `trap` entry,
  78 loops with a numeric trip count in 69 files). A load's shared
  bytes are in the PTX, as the transaction count of the
  `mbarrier.arrive.expect_tx` on its barrier: a phase completes when
  the copies' complete-tx, each "equal to amount of data copied in
  bytes", meet it (§9.7.9.26.5.2, §9.7.14.16.6). b200
  `blk_mx_128x256_s4w4.mx_block_scaled` line 202 expects 50,688 B on
  `[%r14]`, which the four copies of lines 215–238 complete. Every one
  of the corpus's 4,841 parsed `expect_tx` has an immediate count, and
  20,000 of its 20,009 tensor loads follow a same-address `expect_tx`
  that dominates them; a store (`.bulk_group`, 950) has none.
- **Special registers PTX ISA §10 does not describe** — `%pm*`
  ("behavior is currently undefined"), `%envreg*` (driver-defined),
  `%globaltimer*` (target-specific), `%reserved_smem_offset_*` — read
  as divergent, "is not modelled"
  (`special_registers_by_the_threads_that_share_them`).
- **Instruction keywords after PTX ISA 9.3.** The parser knows the
  pinned manual's Table 2 and `fabric`; a keyword a later ISA adds
  leaves its statement unparsed (`mnemonics_outside_the_isa_are_unparsed`;
  nvcc 13.4 already emits `.version 9.4`).
- **Dynamic shared memory is not read from a record.** `--ncu` takes a
  record's grid, block shape and integer parameters, not its `shared`
  bytes, and no flag gives them, so a kernel declaring `.extern
  .shared`, as every Triton kernel does, reads `shared memory 0 B/CTA
  + dynamic`: 26 of the 34 B200 records hold a nonzero `shared`, from
  128 B to `blk_mx_128x256_s4w4.mx_block_scaled`'s 202,800 B.
- **Calls.** Module-scope `.func` declarations and definitions are
  discarded (`module_scope_functions_are_skipped`), so no callee's
  instructions are counted, and a kernel calling one does not
  round-trip (`micro/call_returns.ptx`: "Call to 'zero' requires call
  prototype"). A callee's work leaves the totals exact, with no `>=`
  (`micro-call-returns.test`), and a `.func` body's lines carry no
  note: in the Blackwell corpus 49,926 statements in 13 files
  (sol-study's `deepdive/warpid/out/WS_NOINLINE.ptx` line 199, a
  `tcgen05.mma`, prints nothing). A call's return value, `ld.param
  [retval0]`, is one per thread, so a branch on it is an unknown
  selector: 7,086 thread sets in 10 Triton concurrency-sanitizer builds
  of sol-study's Gluon kernels are bounds so.
- **In a kernel with a cycle of more than one entry**, the uniformity
  analysis takes every value defined on a cycle and read outside it as
  divergent, whatever the cycle's exits compare
  (`a_value_from_a_cycle_with_two_entries_is_divergent_after_it`), and
  a branch as selecting the value at every φ its successors reach, not
  only where their paths meet
  (`with_a_cycle_of_two_entries_every_block_reached_is_a_join`).
- **A φ at a loop header that every latch feeds the same value** is
  still taken as selected by the branch between the latches, as LLVM
  marks every φ of a join block (`taintAndPushPhiNodes`,
  llvm/include/llvm/ADT/GenericUniformityImpl.h:949): the step counter
  of `two_latches_select_at_the_header` is divergent.
- **`redux.sync`'s result is divergent.** PTX ISA §9.7.14.13 gives each
  thread of the membermask the reduction, one value for a warp whose
  mask is full, but the uniformity analysis has no arm for it, and
  affine analysis reads it as "value defined by unsupported instruction
  `redux`". The Blackwell corpus (`trap` entry) holds 6,676 in 500
  files, 1,254 of them sm_100a's `.f32` forms; reading the result as
  one value per warp turns 313 `addr ?` into addresses and removes 537
  `<= warp` bounds, in 98 files.
- **An `ld.param` at a register address is divergent.** PTX ISA
  §5.1.6.1 lets a kernel parameter's address move into a register and
  be read by `ld.param`, as FlagGems selects one of two parameters
  (`mov.b32` of each, `selp`, `ld.param.b64 %rd, [%r]`), but the
  uniformity analysis reads a parameter only at its name: census
  `00996.ptx` line 71's `ld.global.b16` reads `addr ? (depends on a
  value loaded from memory)`. 25 such loads in 8 census kernels;
  reading them as one value for the CTA gives 23 rows an address.
- **A value reading a register no statement writes is divergent, and
  a φ a path leaves unwritten is refused.** LLVM leaves registers
  undefined (`undef`, and `// implicit-def:` for one first written in a
  loop's first trip); PTX ISA §5.1.1 gives a register no initial value.
  The uniformity analysis takes a value with no definition as
  divergent, the half `mov.b64 {%r, _}` takes of a pack whose other
  half is undefined too, so a division by it is "by a non-constant
  divisor" (60 such rows in 5 sol-study files, among them
  `codegen_cmp/attn_extend/gluon_kernels/paged_g4`). Affine analysis
  refuses a header φ whose
  value on entry is undefined and names a thread-varying input instead
  of the unwritten path: the 348 `ld.global.b32` of sol-study's 18
  `rca_gemm` `blockfp8_tma.ptx` read `addr ? (depends on special
  register %tid.x)`, each of the address's two terms read on its own.
  338 files of the Blackwell corpus read a register no statement
  writes.
- **`setp`s not read as thread selectors.** A floating-point `setp`
  compares floats, not the traced integer forms, and `.f16x2` and
  `.bf16x2` each half on its own (PTX ISA §9.7.6.2, §9.7.7.2); an
  `lt`, `le`, `gt` or `ge` is not read where an operand's range, for
  `%tid` below the block shape, else below 1024 (§10.2), and `%ctaid`
  below the grid, crosses an edge of [−2^(width−1), 2^(width−1))
  shifted by a multiple of 2^width and, unsigned, of [0, 2^width) so
  shifted, and its value past its multiples of 2^width is no form
  (`extend`); nor, unsigned and read as signed, where the operand the
  other must stay below is not known nonnegative, with every `%ctaid`
  of any grid (§10.7) and every symbol nonnegative (gluon
  attn_fwd_ws's `setp.lt.u32 %p60, %r1880, -190` without its record's
  launch); an `eq` or `ne` where their difference's range leaves
  (−2^width, 2^width). What it guards is then a bound
  (`micro/setp_float.ptx`, `micro/setp_f16x2.ptx`,
  `micro/setp_wraps.ptx`'s first kernel, a signed comparison of
  2^28·%tid.x: `<= warp * × lane *`, `global store <= 256 B`).
- **A guard comparing the thread index with a CTA index needs the
  grid, and one with an unbound parameter or a loop counter is an
  unknown selector.** A comparison over `%tid` and `%ctaid` with
  constant coefficients, its operands under `--bind`
  (`micro/guard_bound_param.ptx`: `%tid.x < n` runs on 48 threads with
  `--bind 1:n=48`), is read only with `--grid` or a record's grid
  (`src/analysis/thread_participation.rs`, `across_ctas`): decided
  where the grid decides it, its `%ctaid` terms dropped where they take
  one value (b200 `set_mla_kv_buffer_kernel_norope`'s
  `8·(%tid.x mod 128) + 1024·%ctaid.y < 576` under 8192×1×1 CTAs: 72
  threads per CTA, as the record measures, `b200-mixed-index-guard.test`,
  `the_grid_decides_a_guard_on_the_cta_index`),
  and otherwise per CTA a bound, the lanes in at some CTA
  (`micro/cta_mask.ptx` under 16 CTAs, where the last stores on 40
  threads: `sectors/req <= 5 (ideal <= 4)`; `micro/cta_guards.ptx`).
  The ncu comparison sums such a set's count over the CTAs
  (`Pred::launch`; scan_tma's line 184,
  `scan_tma_s_line_184_counts_the_ctas_that_run_it`), but not past 2^24
  CTAs on the axes its comparisons read or 2^26 lane evaluations
  (`a_launch_sums_each_cta_s_threads_and_warps`). Without a grid the masks
  Triton builds from `pid·BLOCK + arange` leave what they guard a bound
  (`micro/cta_mask.ptx`, `store <= 256 B`). A guard on a loaded value
  is an unknown selector too, one for a warp's lanes where the load's
  address is (`micro/guards.ptx`: the store total stays a bound, a
  request's sectors are exact). A comparison with a loop counter is counted trip by trip
  (`micro/counter_guard.ptx`: 384 barriers and 300 stores per CTA, as
  measured; `fp8_gemm_kernel.lm_head_dx`'s `<block 4>`, `runs when
  iv[$L__BB0_3] < 510`, confirmed at 12,533,760, `gluon-counter-guard.test`,
  `lm_head_dx_s_counter_guard_confirms_its_block`) only where the
  block lies in that loop, the loop's trip count is a constant and the
  comparison holds that one counter with the thread index; with a
  parameter not bound (`micro/counter_guard.ptx`'s mask `64·k +
  %tid.x < n` without `--bind`), the CTA index, or a second counter, it
  is an unknown selector. The census counts by
  cause are in the message of commit `ccf8220`. Sectors per warp under
  such a comparison count every lane it holds at some CTA:
  `micro/cp_async_ignore_src.ptx` line 55 reads its source on 80
  threads over the launch, exactly, with sectors `<= 51`, where 40 are
  measured.
- **A test both arms of an unknown branch repeat leaves a bound.** Read
  in three-valued logic, `(? and %tid.x < 16) or (? and %tid.x < 16)`
  is not `%tid.x < 16`: the block both arms reach counts `<= 16`
  threads where 16 run it
  (`a_test_both_arms_of_an_unknown_branch_repeat_is_a_bound`).
- **A loop footprint's distinct and sector bytes count the threads a
  guard on its counter admits at any trip on every trip**; its
  requested bytes count each trip's: `micro/counter_guard.ptx` with
  `--bind 1:n=300` requests 1200 B, `unique <= 256 B, in sectors <=
  2048–2560 B` (`micro-counter-guard.test`); census kernel `02847.ptx`
  with `--bind 1:n=700 --grid 1,1,1` requests 6896 B, `unique <= 8192
  B`.
- **A block past 32 steps of control dependence is an unknown
  selector.** A ladder whose rungs each branch both ways to the next
  doubles the steps per rung; past 32 the block's set is a bound
  (`a_ladder_past_the_reach_budget_is_an_unknown_selector`). The census
  and fixtures take at most 17; sol-study's concurrency-sanitizer
  builds take more (1,232 thread sets in 10 files), each of which also
  reads a call's return value (**Calls.**).
- **A `cp.async` whose ignore-src is not read counts cp-size on every
  thread that runs it, as a bound.**
  - Census (13,489 in 300 kernels, sm_100a, no bindings): 479 read
    nothing (a constant true ignore-src), 285 all, 9 on a thread-index
    condition, 58 on a loop counter trip by trip, 901 on one the same
    for every thread, 213 partly (212 in loops), 11,544 not at all
    (4,175 in loops).
  - B200 fixtures (319 in 16 kernels): without their records none is
    read exactly (74 the same for every thread, 245 not read). With
    them: 84 read all, 48 on a thread-index condition, 34 trip by trip,
    40 partly, 113 not at all (49 in loops).


  A src-size register is a bound on which threads read and how much
  (`micro/cp_async_src_size.ptx`: `sectors/req <= 17 (ideal <= 16)`,
  `global load <= 1,024 B`, where 16 lanes read 8 sectors;
  `micro-cp-async-src-size.test`,
  `a_register_source_size_bounds_the_sectors_measured`).

### Presentation

- **Floors and mods of one variable are not merged back**, so a form
  can print longer than its value: k5's second `As` store reads
  `16·⌊%tid.x/8⌋ + 2·(%tid.x mod 8)`, which is `2·%tid.x`, and
  the B200 fused_rmsnorm's `4·⌊(⌊%tid.x/8⌋ mod 128)/4⌋` is
  `4·⌊%tid.x/32⌋` below 1024 threads.
- **A refused φ names the obstacle on one of its source chains**: in
  b200 `tma_best` 34 access rows say only "depends on special register
  %ctaid.x" (8 in `bmm_kernel_persistent`), not that a join of two
  values stopped the trace. A value an `ldmatrix` loads, used as an
  index, is named as "one of `ldmatrix…`'s vector destinations", not as
  loaded (census `00428.ptx`; 248 rows in 4 kernels), and a trip count
  refused for a value its own top-level loop defines says "is defined
  in an enclosing loop" (census `00143.ptx` line 40; 63 loops in 39
  files of the Blackwell corpus, `trap` entry).
- **A total left open by an unclassified statement does not say which,
  and every pipe and space is open.** One unclassified statement marks
  every pipe's FLOPs and the global and shared bytes
  (`micro-totals-annotate.test`). In the Blackwell corpus 1,258 `.entry`
  lines read `atomic flops ?`, 1,245 of them in files with no `atom` or
  `red`, 337 `sfu flops ?` with no SFU instruction, and 626 files with
  no tensor copy, whose unclassified statements move no global bytes,
  mark the global bytes.
- **`(trips of <header>)` names one loop**, the first of a total's
  unknown trip counts: census `01802.ptx` reads `global load ? B (trips
  of $L__BB0_5)`, though its sibling loop `$L__BB0_2`, trips also
  unknown, holds 15 global loads; about 1,460 of the Blackwell corpus's
  3,901 such items, in 539 files.
- **An unknown count of a known precision prints as two unknowns**,
  `flops f32 >= N + ?`, and as none counted where it is the pipe's
  only work: b200
  `attn_lightllm__llama3-8b__decode._fwd_kernel_flash_decode_stage2`
  under its record reads `sfu flops f32 >= 262,144 + ?; cuda-core flops
  ?` with no unclassified statement.
- **A statement's own threads print only where their text differs
  from its block's.** A guard narrowing a block whose set is a bound
  prints nothing, both reading `<= warp … × lane …`: 2,098 statements
  in 44 files of the Blackwell corpus, 1,806 of them elect-guarded. A
  `cp.async`'s source
  threads, those its ignore-src admits (PTX ISA §9.7.9.26.3.1), print
  nowhere, though its sectors, bytes and footprint count them
  (`micro-cp-async-ignore-src.test`): 35,587 copies in 959 files read
  a source set other than their own, 34,765 of them a bound.
- **An untargeted label after a branch is a block of no instructions
  whose line carries a block's notes**: LLVM's DWARF `$L__tmpN`
  (b200 `tma_best_Q64N64w8s3._stage1_tma` line 517, `$L__tmp2: // no
  thread`), 1,888 lines in 643 files of the Blackwell corpus, 663 of
  them with threads, conditions and counts; 9 are `ncu_corpus`'s
  `unchecked … 0 of 0 with SASS` blocks.
- **A `.local` access prints its address and no number or reason**
  (133 rows in 15 files of the Blackwell corpus).
- Per-thread totals of a warp-specialized kernel are `<=` the sum of
  its partitions, since a thread runs one of them: each block's
  annotation says which threads run it (`warp 0-3 × lane *`), and the
  per-CTA totals count each block on its own threads, but there is no per-role view of the per-thread numbers.
- Parameter names are positional (`param_2`); the PTX carries no
  source names.
- Floor division prints as `/`.
- **Pointers are taken as 32 B aligned.** Sectors per warp request
  and per loop take the memory a pointer parameter points to as
  aligned to its `.ptr .align N` or to `--ptr-align`, 32 by default,
  where that is more, and the `.entry` line says so
  (`tests/lit/ptr-align.test`): nvcc declares no `.ptr .align`, the
  Gluon records `.align 1`, the B200 Triton records 16. A pointer below
  it, into an allocation `cudaMalloc` aligned to 256 B, gets too few
  sectors; `--ptr-align 1` gives the range over every alignment under
  which the accesses are aligned (PTX ISA §6.4.1). Every record line
  the default closed, 1,127, measured its count
  (`ncu_corpus` snapshot). A base on a module-scope array or a loaded
  pointer is still a range: the array's `.align` is not read
  (`micro-uniform-bases-global.test`). A 16-byte access on a pointer
  does not narrow the range of a 2-byte access on the same pointer.
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

- **`tests/ncu_corpus.rs` fails only on a contradicted block.** A
  block that loses its verdict, a sector line that starts to differ or
  a launch's sectors that stop agreeing change its snapshot, which
  `cargo insta accept` takes like any other change.
- **Nsight Compute miscounts the L2 sectors of an `LDGSTS` that reads
  `P0`**, as its guard or unnegated as its source predicate: its L2
  Theoretical Sectors Global follow neither its predicated-on threads
  nor the L1TEX hardware counters. `micro/ldgsts_p0.ptx`, on an RTX 4090
  under Nsight Compute 2026.3.1 and 2025.3.1:
  - its `@!P0` copy reads 64 sectors where its 16 threads request 8;
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is 64, the
    8 + 24 + 8 + 24 its threads request.

  `measure.py` and `import_reports.py` keep no sectors for such a line,
  so it is not compared:
  - lm_head_dx's lines 979 and 985 read 6,268,416 each where the
    prediction is 6,266,880. The kernel's L1TEX load sectors,
    62,916,096, equal the sum of the predictions.
  - attn_bwd's eight `@!P0` copies.
  - 14 B200 copies that read `P0` as their source predicate, which
    measure fewer than predicted. Their reports' L1TEX load sectors
    exceed the sum over the lines by exactly that shortfall: 31,152 for
    `scaled_mm_kernel`, 6,896 for `ssd_pass_tlrange`.

  A copy reading `!P0` counts as requested
  (`micro/cp_async_ignore_src.ptx`).
- **A launch count past the lane budget is rejected late.**
  `Pred::weighted_launch` counts a group of CTAs' threads when it
  first meets the group, so a set whose CTAs form more groups than 2^26
  lane evaluations allow tallies every group's lanes before it returns
  none: the `spread` case of `a_launch_sums_each_cta_s_threads_and_warps`,
  65,537 groups of 1,024 lanes, evaluates about 134 million predicates.

- **The frontend is not panic-free for arbitrary UTF-8.** An input file
  containing only `é` causes `ptxroof analyze` to exit with a panic:
  `end byte index 1 is not a char boundary`. The byte-oriented lexer
  slices an unexpected multibyte character after advancing one byte
  (`src/ptx/parse/lexer.rs`, `make` and `next_token`).
- **Opaque kernel parameters are a parse error.** `.texref`,
  `.samplerref` and `.surfref` parameters (PTX ISA §11.2.1) are not
  fundamental types (`opaque_params_are_a_parse_error`).
- **Cluster launches are a parse error, and the model has no
  cluster.** `.explicitcluster` and `.reqnctapercluster` (PTX ISA
  §11.7.1–2; `.maxclusterrank` and `.blocksareclusters` take the same
  path) between a kernel's signature and its body stop the parse,
  "expected '{' starting kernel body", and the file prints nothing:
  158 files of the Blackwell corpus (`trap` entry), every Gluon 2-CTA
  kernel of sol-study (`mla_decode_2cta_kernel`, the 2-CTA GEMMs),
  holding every `cta_group::2` `tcgen05` and TMA form, `barrier.cluster`,
  `mapa`, `ld.shared::cluster`, `clusterlaunchcontrol` and
  `ld.param::entry` of the corpus. With the two lines deleted they
  analyze, but no cluster shape is kept: `%cluster_ctarank` is a symbol
  per CTA, so a block under `runs when %cluster_ctarank = 0` (5,903
  notes in 143 of them) counts as a bound per launch, where a cluster
  of n CTAs gives each rank to grid/n of them (§2.2.2, §10.16), and a
  successful `clusterlaunchcontrol.try_cancel` (§9.7.14.18) leaves
  CTAs of the grid unlaunched (sol-study's `gemm_g2clc.ptx`). No record
  holds a cluster launch: `tools/ncu/launch.c` sets no cluster
  dimension and `record_launches.py` keeps Triton's program grid, while
  sol-study's reports hold 23 cluster launches of Gluon kernels with
  PTX.
- **Thread sets grow with the checks along a dominator chain, and a
  file prints nothing until every kernel is done.** A block's set is
  its immediate dominator's and its own condition, which every
  consumer evaluates thread by thread, so a chain of checks whose
  failing side does not rejoin (Triton's concurrency sanitizer: `@%p
  bra` to `__assertfail; trap;`) costs time and memory quadratic in
  their number: 29 sanitizer builds of sol-study's Gluon kernels
  (20,036–44,815 statements; 24 behind the cluster parse error) take
  up to a minute and 1.5 GB each, most of it evaluating thread sets
  (the slowest run alone: 61 s), where no other Blackwell corpus file
  takes 3 s. Bindings that differ
  from `.reqntid` build a second set of thread sets for dead code, 40–50%
  more memory on those kernels. The output is written once, after every
  kernel, so a run stopped by a time or memory limit prints nothing.

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
  identical to the original's for 153 of the 163 fixtures
  (`tests/lit.sh -v tests/lit/roundtrip`, ptxas 13.4). k14's and
  `micro/address_operands.ptx`'s are rejected because the in-kernel
  `.local` declaration is discarded (`Unknown symbol
  '__local_depot0'`, `'lbuf'`). k1's and k2's assemble to different
  SASS because `.pragma "nounroll";` is discarded, so ptxas unrolls
  the loop it annotated. `.pragma "used_bytes_mask"` (PTX ISA §12.2;
  138 in 43 files of the Blackwell corpus) is discarded too: of 369
  corpus files reassembled, 356 give identical SASS, and census
  `03035.ptx`'s masked loads come back wider (916 `LDS.128` and 35
  `LDS` from the original, 940 and 3 from the dump). `micro/texture_outputs.ptx`,
  `call_returns.ptx`, `unparsed_writes.ptx` and `vector_register.ptx`
  lose what the parser drops (above). These are allowlisted in
  `tests/roundtrip-allowlist.txt`. One lit test per fixture
  (`tests/lit/roundtrip`), unsupported without the toolkit or when
  ptxas rejects the fixture's `.version` as newer than its own.
- Generated loops (`tests/gen_loops.py`, `tests/lit/gen-loops.test`): single counted
  loops drawn from a grammar of steps, initial values, bounds,
  comparisons, operand orders, branch polarities and read positions,
  some behind LLVM's loop guard, each simulated in Python and bound
  with `--bind`; the tool's count must equal the simulation at every
  binding, 0 where the guard skips the loop, a refused shape only
  counts, and a crash, other than one loop or a count left symbolic
  fails. The lit test pins seed 1's 200 kernels: 486 bindings compared,
  338 refused, 174 a loop that continues while two values are equal
  and 164 one whose bound moves away from its exit condition.
- Hardware cross-check on an RTX 4090 with Nsight Compute: k5's
  counts, and the c_fc fp8 GEMM's launch (M=4096, N=3072, K=768):
  `sm__ops_path_tensor_src_fp8.sum` = 19,327,352,832 = 2·M·N·K, the
  tool's 25,165,824 fp8 tensor flops per CTA at K=768 times 768 CTAs
  (`gluon-fp8-gemm-c-fc-warps.test`); `sm__inst_executed_pipe_tensor`
  = 2,359,296, that at 8,192 per `mma`; l1tex global-load
  bytes (151,191,552) = every `cp.async` byte (150,994,944: two
  prologue stages plus ten of the twelve iterations, the predicated
  prefetches off at the end) plus 256 B per CTA for the two scale
  loads at sector granularity, where the tool's 151,781,376 B counts
  the scale loads' bytes as the threads request them, 1,024 B per CTA
  (`gluon-fp8-gemm-c-fc.test`). The attention launches at B=2, T=2048,
  H=6, D=128: `sm__inst_executed_pipe_tensor` = 3,244,032 for the
  forward, whose loops' trips the tool does not know (above); 8,110,080
  for the backward, the tool's 33,218,887,680 bf16 tensor flops at
  4,096 per `mma` (`gluon-attn-bwd.test`); the pre and post
  kernels' cuda-core flops (2·ffma + fadd + fmul) 6,684,672 and
  3,145,728 are the tool's per-CTA 17,408 and 8,192 times 384 CTAs.
  Sectors per warp request, k5 and k1 at M=N=K=256 (`tests/lit/
  k5-footprint.test`, `k1-footprint.test`): l1tex global load
  requests and sectors 18,432 / 81,920 (k5) and 1,050,624 / 1,576,960
  (k1), stores 2,048 / 32,768 and 2,048 / 4,096, each the rows'
  aligned counts times their execution counts; per CTA, k5's K loop
  moves 98,304 B in sectors and the
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
  are marked `timing_dependent` (18 in each Gluon attention forward
  record, 238 in 12 B200 records). The comparison does not read the
  mark; such a line goes unchecked where its count is not known, and
  `tut_ll_BM128_BN128_w4_s2_wsTrue.attn_fwd`'s `mbarrier.try_wait`
  lines 1494, 2724 and 3365, predicted and measured 0, confirm their
  blocks. A `tcgen05` statement or `mbarrier.init` that one elected
  thread runs is uniform-datapath SASS (`UTCHMMA`, `UTCQMMA`, `UTCCP`,
  `SYNCS.EXCH.64` after `VOTEU.ANY`), which Nsight Compute's
  predicated-on thread count takes as the warp's 32 threads: 139 lines
  of 9 B200 records measure 25–32 times their count
  (`blk_mx_128x256_s4w4.mx_block_scaled` line 167: 9,216 predicted,
  294,912 measured), and the records keep no warp-level count for them.
- Kernels that ship only SASS have no PTX to read: 18 of the 149
  launches in sol-study's B200 reports (cuBLAS, cuDNN, CUTLASS and CuTe
  DSL, flashinfer's trtllm-gen attention, PyTorch's elementwise).
- `--ncu` compares no counts in a kernel with irreducible control
  flow, whose blocks' executions the tool does not know
  (`tests/ncu_comparisons.rs`,
  `an_irreducible_cycle_leaves_every_block_unchecked`), nor on a PTX
  line holding more than one instruction, which a record counts
  together (`two_instructions_on_one_line_are_not_compared`).
- No machine model. AI is printed without a ceiling; the PTX `.target`
  names a family, not a part.

## Missing features

- A value affine analysis cannot read but the lanes of a warp share as
  an affine variable, `uniform(v)`, keyed by its register SSA value:
  addresses read it, and loop footprints where every thread of the CTA
  shares it (`micro/uniform_address.ptx`,
  `micro/uniform_counter.ptx`, `micro/uniform_bases.ptx`), and trip
  counts where every thread of the CTA shares it and no enclosing loop
  defines it (`micro/trip_div_param.ptx`: `uniform(%r3@30)`), else
  their reason names it (`micro-trip-uniform.test`); thread selectors
  do not. Register SSA
  itself
  (`src/ssa.rs`) is what affine analysis, the uniformity analysis
  and the predicate reader read.

One line each, with the trigger that would start it.

- clang fixtures with a regen script. Trigger: the first clang-built
  kernel.
- `diff` between two builds of one kernel, PTX and SASS spill counts.
  Trigger: the first spill regression hunt.
- Local memory per thread from the `.local` depot declaration (k14:
  `__local_depot0[512]`), where spills go; keeping the declaration
  also lets k14's dump reassemble. Trigger: the same hunt.
- Array element operands, `a[i]` (PTX ISA §6.4.2), deliberately not
  parsed: none in the corpus (`tests/parse-allowlist.txt`) or in the
  missed-opt census's 13,209 PTX files, and the manual's own examples
  are their only source seen. A statement with one is unparsed,
  uncounted, and taken to write every register it names
  (`micro/unparsed_writes.ptx`). ptxas scales a constant index by the
  element size (there, `lbuf[1]` of a `.u32` array reads byte 4), while
  the manual's §4.3 example scales its register index itself. Trigger:
  the first fixture from a producer that emits them.
- Access patterns: done 2026-09-15 (the `accesses` rows: every
  fixture's global and shared operands except those listed above,
  including 2D tiles' `⌊%tid.x/8⌋` and `(%tid.x mod 8)` and counters
  stepped by `4·N`; sectors per warp request by enumerating
  the lanes under every alignment of the uniform part its pointer's
  alignment allows, so a coalesced 4-byte access reads `4 sectors`;
  needs `--launch` or `.reqntid` and
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
  instructions executed, and on a line with a global memory
  instruction Nsight Compute's L2 theoretical sectors, their ideal,
  each such SASS's access size and their predicated-on thread
  instructions executed, on a line with a shared memory instruction
  each such SASS's opcode, access size, warp-level instructions
  executed, L1 shared wavefronts and their ideal (the Profiling
  Guide's "assuming each not predicated-off thread performed the
  operation", not a count without bank conflicts), and for the launch the L1TEX hardware counts
  of the sectors its global loads and stores request
  (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `…_op_st.sum`)
  (`tests/fixtures/k5/k5.sm_89.256.ncu.json`;
  every Gluon kernel's first launch, `tests/fixtures/gluon/*.ncu.json`).
  The B200 records (`tests/fixtures/b200/*.ncu.json`) come from
  sol-study's reports instead: `import_reports.py` attributes their
  per-SASS counts and sectors to PTX lines, leaving out a TMA
  instruction, whose access size the report omits and whose sectors it
  counts as 0, and keeps the report's two L1TEX counts but where it
  reads the L1TEX global load requests as nan (the gated rmsnorm
  report, whose L1TEX, L2 and DRAM counters read 0 or nan although its
  loads ran), through the same `.loc` recompile under
  Triton's ptxas command, and takes each launch's parameters from a
  replay of the report's own command as the B200, matched by SASS and
  grid, with `i64:`, `i1:` and `tensormap:128` for the parameters of a
  tensor descriptor, which launch.c does not take; the PTX
  must still assemble to the recorded SASS (`b200-sass.test`, under
  ptxas 13.4 only). `regen.sh` no longer regenerates them: against
  sol-study `e1d124c` with uncommitted changes its replay of
  `attn_lightllm/deepseek-v3/decode` fails in Triton's binder
  (`dynamic_func() takes 6 positional arguments but 30 were given`),
  and the kernel corpus's `MANIFEST.jsonl` no longer lists the extend
  attention kernel's source; the shared memory entries were imported
  with the launches the records hold and the PTX headers kept. `ci.sh` does not re-measure them with
  `measure.py --check`, which would launch on zeroed inputs.
  `analyze --ncu RECORD` takes the record's grid, block shape and
  `u32` and `i32` parameters for `--grid`, `--launch` and `--bind`, and
  refuses a record whose
  lines are not the PTX's (`tests/lit/k5-ncu-stale.test`), and
  gives each block a verdict (`report::build::compare`, its
  `Comparison`'s blocks): a line whose
  `CountConstraint` is not `Free` and whose SASS is one opcode must
  equal an exact prediction or stay within a bound. A prediction is
  per-CTA threads times trips times CTAs, or, where the threads differ
  by CTA, their sum over the grid times trips. Each block also
  counts its instructions, those with SASS and those that agree, and
  lists every line whose measured count is not the predicted number,
  with its SASS opcodes (`differing`). A line whose global SASS all have the width of its
  one PTX global access also compares its L2 sectors and their ideal
  over the launch, the sectors per warp request times the warps with a
  lane that runs it, a range over the alignments its pointer allows
  (a block's `sectors`, `tests/ncu_comparisons.rs`,
  `a_line_at_its_own_width_compares_its_l2_sectors`; in the records
  only the three copies above differ, in
  `micro/shared_wavefronts.cp_async.ncu.json`,
  `gluon_records_agree_on_counts_sectors_and_launch_totals`), once
  the line's global SASS executed the predicted count, or within its
  bound. A line whose shared SASS that request wavefronts are one opcode
  at the width each lane of its one PTX shared access covers compares
  the predicted wavefronts and ideal per warp request with the
  record's over those SASS's warp-level executions
  (a block's `wavefronts`, `tests/ncu_comparisons.rs`,
  `every_counted_shared_line_s_wavefronts_are_measured`): all 1,018
  such lines of the records agree, 544 of them in 25 B200 Triton
  kernels and 414 in nine Gluon kernels and k5 on the RTX 4090
  (`tests/ncu_corpus.rs`). ptxas may run
  a load less often than its statement: `tma_best_Q64N64w8s3._stage1_tma`'s
  line 544 executes 655,360 times where its loop iteration runs
  2,424,832 times, so it and one line of
  `_fwd_kernel_flash_decode_stage1_padding` are listed as not compared
  (`outcome: executions_differ`). Not compared at all: a generic access,
  a line whose count is unknown, and a line with an `LDGSTS` reading `P0`,
  whose record keeps no sectors (Known limitations, Producers and
  validation). Over the launch, the predicted global load and store
  sectors, each line's prediction summed once over the record's lines
  with global SASS, are compared with the record's L1TEX counts when
  every such line but an atomic's is predicted at its recorded width
  and executions (the `Comparison`'s `launch`;
  `an_ldgsts_on_p0_keeps_no_sectors_but_counts_in_the_launch_total`,
  `lm_head_dx_s_counter_guard_confirms_its_block`: the sum includes the
  lines the `P0` rule leaves without sectors). In 39
  records both agree. 25 have a line not predicted so (ce_chunk's
  `ld.global.b64` that ptxas loads as 32-bit `LDG.E`), and the gated
  rmsnorm record has no counts. `tests/ncu_corpus.rs` keeps
  every record's block verdicts, their counts, its sector lines by
  outcome and its launch totals in one snapshot: 145 blocks confirmed,
  276 consistent, 491 unchecked; 1,442 sector lines agree, 3 differ
  (the copies above), 2 not compared; 40 launches' totals agree, the
  39 above and `tests/lit/Inputs`' one. `ci.sh` measures every record again on the GPU it
  names (`measure.py --check`) and fails if it changed, but for the
  driver version past its first two components; a record of
  another GPU is skipped, and fails the run under
  `PTXROOF_REQUIRE_GPU=1`; each prints whether it was measured.
- Per-line HTML view. Trigger: wanting it.
- `capabilities` verb that generates the audit doc's tables.
- sm_90 and sm_100 families (`wgmma`, TMA, `tcgen05`): need a
  per-instruction issue scope, which the model lacks; sm_100 fixtures
  exist (`tests/fixtures/b200`), sm_90 ones do not.
  `blk_mx_128x256_s4w4.mx_block_scaled` counts 224 `tcgen05.mma` per
  CTA and no FLOPs for them, though an MMA's M and N are in its idesc
  (PTX ISA §9.7.17.4.2 Table 45; K by `.kind`, Table 42), a `mov.b32`
  of an immediate before each of the Blackwell corpus's 49,606 parsed
  `tcgen05.mma` (56 through a copy of one). A TMA store's bytes and a
  copy's global address need its descriptor's shape, strides, box and
  swizzle (a load's shared bytes are in its barrier's `expect_tx`,
  **A tensor copy's bytes are unknown.**): the B200 records keep
  `tensormap:128` only (`record_launches.py`), and the reports give TMA
  instructions no access size and 0 sectors.
- Values loaded from memory as inputs: a loop bounded by one has no
  count (`attn_lightllm__llama3-8b__decode._fwd_kernel_flash_decode_stage2`,
  the sequence length its `ld.global` at line 64 reads); a count in a
  symbol `--bind` sets needs the loaded value one for the CTA, which
  PTX ISA §8.5 does not give. Trigger: wanting such a kernel's count
  for given lengths.
- A coverage summary per metric over a corpus: how many trip counts,
  FLOPs, byte totals, addresses and sector lines are exact, bounds or
  unknown. Trigger: comparing coverage between commits, so far done by
  scripts outside the repository.
- Totals per warp role. A warp-specialized kernel selects its blocks by
  warp index (`gluon/attn_fwd_ws_kernel.sm_89.ptx`, 8 warps: 34 blocks
  on `⌊%tid.x/32⌋ < 4`, 38 on `⌊%tid.x/32⌋ >= 4`), and every total sums
  all roles; grouping blocks by the warp-index condition their thread
  sets name would give each role's FLOPs, bytes and instructions.
  Trigger: wanting to know which role's work bounds such a kernel.

## Acceptance scenarios

Each id names a lit test, `tests/lit/s*.test`, in its first comment:

| id | question | fixture |
|---|---|---|
| S1.1, S1.2 | Is k5's design point what I computed on paper, at sm_80 and sm_89? | k5 |
| S6 | Where does the work go? (main loop and remainder trips) | k2 |
| S7.1, S7.2 | Did tiling pay off? (f32 FLOPs per global byte a trip, 8/16 vs 1024/32) | k1, k5 |
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

- Commits are planned at about 50 changed non-test lines and may reach
  80. A change that does not fit in 80 is split only at a natural
  semantic boundary; otherwise its code goes to the owner for review
  before it is committed (a subagent returns it to the main agent).
- No comments except externally imposed invariants, with a link. Cite
  the PTX ISA from `refs/` (`tools/fetch-manuals.sh`), not memory.
- This file and `docs/ptx-instruction-coverage.md` change in the same
  commit as what they describe.
