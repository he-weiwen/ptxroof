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
comparisons, with the grid given comparisons with the CTA index as
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
`warp: 4–5 sectors`); and per loop
with a numeric trip count, the global bytes one CTA
requests over the loop's own blocks, the bytes of the sectors those
requests touch, and the distinct bytes it touches. An instruction that
runs but whose results nothing with an effect reads (LLVM's ADCE over
the register SSA, `src/analysis/dead_code.rs`) counts in no total and
no loop footprint, and is listed as unused: ptxas emits no SASS for
any of the 954 such instructions in kernels with an Nsight Compute
record (`tests/dead_code_corpus.rs`). Where these and the instructions
under a guard no thread passes are left out, an instruction total
reads `live of` the count with them (rmsnorm: `1041 live of 1146`).
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

- **A `cp.async` that may read fewer bytes than cp-size counts
  cp-size.** With a register src-size or an ignore-src predicate (PTX
  ISA §9.7.9.26.3.1) a lane may read fewer bytes and zero-fill the
  rest, but the access row's sectors per warp and the global bytes count
  cp-size on every lane (`micro/cp_async_src_size.ptx`: 16 of 64 lanes
  read, the report says `16–17 sectors` per warp and `load 1024 B` per
  CTA where Nsight Compute measures 8 sectors over the launch,
  `tests/lit/micro-cp-async-src-size.test`). Triton writes its masked
  loads this way: 319 such `cp.async` in 16 B200 fixtures, 13,489 in the
  census.
- **Register effects are incomplete.** The affine tracer records the
  registers of the operands the register-effects decoder says an
  instruction writes, a register, a `p|q` pair or a `{…}` vector, as
  destinations (`micro/vector_def.ptx`: a guard on a register a vector
  load overwrote is a bound, `<= 9 sectors`), refuses an element of a
  multi-element `ld.param` vector rather than reading the parameter's
  bits, and does not model guarded definitions: a branch
  or guard comparing a value written under a guard is an unknown
  selector (`micro/guarded_def.ptx`: `<= 5 sectors`, not 64 threads
  exactly), even where every write agrees
  (`micro/guarded_selector.ptx`: in its second kernel 32 threads store,
  the report says `store <= 256 B`), and an address through one is
  read as the last write (`micro/unpack.ptx`'s `micro_unpack_guarded`:
  lanes 16–63 store at `param_0 + 16·%tid.x`, the report reads
  `[param_0]`, 1 sector). A form the
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
  The CFG treats `trap` as fallthrough. The `cp.async` classifier reports a
  fixed source-read size even with a runtime ignore-source predicate.
  Source locations, reproducible parser/classifier probes, and their outputs
  are in the [register-effects audit](docs/ptx-register-effects-audit.md#findings-in-the-current-implementation).
- **A predicated `ret` or `exit` is read as falling through.** The CFG
  gives its block one successor, the next block, so what follows counts
  on every thread, exactly (`micro/predicated_exit.ptx`: the store
  after `@%p1 exit` counts 32 per CTA, where on an RTX 4090 at n = 20
  it runs on none).
- **Vector registers are not modelled** (PTX ISA §6.4.3): the parser
  does not read `.reg .v2 .b32 %v;`, and an element `%v.x` is a register
  of its own, apart from `%v`. In `micro/vector_register.ptx`, after
  `ld.global.v2.b32 %v, [%rd1]` overwrites `%v`, `%v.x` is still the
  thread index an earlier `mov.b32 %v.x, %r1` wrote: the address is
  `[4 * %tid.x + param_0]`, where ptxas takes it from the load.
- **Integer arithmetic is taken not to wrap where its range is
  unbounded**: a 16- or 32-bit `add`, `sub`, `mul.lo`, `mad.lo`, `shl`
  or `bfe` result, and an integer `cvt`'s source and destination, or a
  `min` or `max`'s operands, are
  refused when the value's range, over constants, the block shape
  `.reqntid` or `--launch` gives, the grid `--grid` gives, and constant
  trip counts, leaves the type (`micro/mul_wrap.ptx`: `mul.lo.s32` of
  2^30 and 4, the same for every thread, so its address reads
  `uniform(%r2@24)`; `micro/cvt_reads.ptx`: `cvt.u32.u64` of `%tid.x` +
  2^32; `micro/grid_wrap.ptx`: 2^24·`%ctaid.x` under `--grid 257`;
  `micro/trip_max.ptx`: `min.u32` of n − 4 under `--bind 1:n=2`, where
  the loop runs 8 trips, unbound `max(min(param_1 - 4, 8), 1)`); one
  through a parameter, `%ctaid` with no grid given or a symbolic trip
  count keeps the tracer's documented assumption
  (`src/analysis/scalar/trace.rs`, module doc). A conversion to or from
  floating point, or a pack, is refused (`micro/cvt_reads.ptx`).
- **An and-mask that clears the sign bit keeps a value of unknown range
  as itself.** Up to the bit below the sign bit (nvcc's
  `& 0x7fffffc0`) the and is read as the value itself where its range
  lies in [0, 2^(width−1)), and refused where the range reaches outside
  it (`micro/and_sign_bit.ptx`: `(%tid.x − 256) & 0x7fffff00`, "clears
  bit 31 of a value that may set it"). A value through a parameter or
  `%ctaid` with no grid given has no range and keeps the tracer's
  documented assumption: `micro_and_sign_param`'s `(n − 256) &
  0x7fffff00` reads `[param_0 + 4 * param_1 - 4 * ((param_1 - 256) mod
  256) - 1024]`, where for n below 256 the and is 0x7fffff00 and the
  store 2^33 bytes higher; with `--bind 1:n=100` it is refused, and the
  address a uniform variable. A 16-bit and, and a 32-bit mask of more
  runs, take the remainder mod 2^(width−1) instead
  (`micro/and_sign_remainder.ptx`'s `n & 0x7fff`: `param_1 mod 32768`;
  `micro/and_runs.ptx`'s `micro_and_sign_runs`); a 64-bit mask of more
  runs is refused (`micro_and_sign_wide_runs`: a uniform variable).
- **`shfl.sync.idx` is read as if warps were rows of 32 `%tid.x`**,
  whatever the block shape: a warp is 32 consecutive
  `%tid.x + %tid.y·%ntid.x + %tid.z·%ntid.x·%ntid.y` (PTX ISA §3.1,
  §9.7.16.1), so in a block of more than one row whose `%ntid.x` is no
  multiple of 32 the lane read is another thread's. In
  `micro/shfl_2d_block.ptx` (`.reqntid 48, 2, 1`) every lane of a warp
  stores at lane 0's `%tid.x` (warp 2's lane 0 is thread (16, 1),
  param_0 + 64; ptxas 13.4); the store reads
  `[128 * ⌊%tid.x/32⌋ + param_0]`, `warp: 1–2 sectors`. `%laneid` is
  read only in a block of one row.
- **A parameter load at an offset reads the whole parameter** (PTX
  ISA §5.1.6.1): `ld.param.u32 %r3, [p+4]` reads as `param_1`, like
  the load at offset 0, so comparing two fields of a by-value
  aggregate is `0 == 0` (`micro/param_fields.ptx`: the store `on 64
  threads in 2 warps`, `store 256 B` per CTA).
- **Integer literals are read in decimal and hexadecimal below 2^63
  only**; PTX ISA §4.5.1 also writes them in octal and binary, with the
  suffix `U`, and up to 2^64 − 1 as `.u64`. In `micro/int_literals.ptx`,
  where ptxas folds each offset into the store, octal `020` reads as
  twenty (`[param_0 + 20]`, ptxas `[R2.64+0x10]`), `0b10000` and `16U`
  as symbols (`[param_0 + 0b10000]`), and `0xfffffffffffffff0` as no
  integer, the address a uniform variable (ptxas `[R2.64+-0x10]`). In
  the census the forms are 7,019 `prmt` selectors with a `U`
  (`0x3340U`, 119 kernels) and 19 masked loads' default
  `0xffffffffffffffff` (`mov.u64`, 3 kernels; `00101.ptx`).
- **A `!=` latch with a symbolic distance is taken to meet its
  bound**: the trip solver divides the distance by the step, rounding
  down, as `src/analysis/scalar/trip_counts.rs`'s doc assumes the
  division exact (k1's and k2's `K − K mod 4`, the fp8 GEMMs,
  `micro/two_register_counter.ptx`). A constant distance that is no
  positive multiple of the step is refused: `micro/ne_latch_unreached.ptx`,
  stepping by 2 from 0 while `!= 5`, reads `trips = unknown`, where the
  loop does not exit.
- **A loop's count drops its max with 0 or 1 on nonnegative
  parameters.** A loop is counted max(T, 0) per execution of a guard
  that skips it where its latch fails at k = 0 (`micro/loop_guard.ptx`),
  and max(T, 1) per entry otherwise (`micro/latch_exits_first.ptx`);
  the max is dropped where T is never below it with every parameter
  taken as nonnegative (`src/analysis/scalar/symexpr.rs`'s doc):
  `micro/single_loop.ptx` reads its `ld.param.u32` n as `param_1`,
  with exact totals, where an n of 2^31 or more, below 1 as its
  `setp.lt.s32` compares it, runs no trips.
- **Addresses the tracer cannot read are unknown rows**, with the
  reason, unless every thread of the CTA computes the part it cannot
  read alike: that part is then a variable, `uniform(%r54@90)`
  (the grouped tile schedule's
  `pid / num_pid_n` in the Gluon fp8 GEMM's global copies, k1's
  remainder loop reading the main loop's final k;
  `micro/uniform_address.ptx`). Not read: xor swizzles of shared
  addresses (every ldmatrix and cp.async destination in the Gluon GEMM
  and attention kernels, "behind `xor.b32`"); divisions of the thread
  index by a runtime value (`micro/uniform_address.ptx`'s
  `micro_uniform_div_lanes`); an and-mask's run of bits whose remainder
  the affine form does not hold, of a form whose offset is no multiple
  of its modulus (`micro/and_runs.ptx`'s `micro_and_offset`,
  `(%tid.x − 4) & 0x46`: "in a mod") or a negative coefficient, but for
  a value of 0 or −1 under a mask below the sign bit
  (`micro/and_bit_select.ptx`; refused with the sign bit,
  `micro_and_select_sign`, and for −(%tid.x & 3),
  `micro_and_select_two_bits`), or a 64-bit mask's run that starts at
  bit 63 (`micro/and_runs.ptx`'s `micro_and_bit63`); the bit select by a
  signed `bfe` of one bit (`micro/and_bit_select.ptx`'s
  `micro_and_select_bfe`: "bfe at 2 of 1 bits"; 1,227 census rows stop
  at a one-bit signed `bfe`); values loaded
  from memory (the cross-entropy's target gather: "depends on a value
  loaded from memory, behind `selp.b64`"); an `or` whose sides may
  both set a bit, by the bits the affine forms bound (`Affine::may_set`):
  a bitwise or that is no add (`micro/or_disjoint.ptx`'s
  `128·⌊%tid.x/32⌋ | 16·%tid.x`: "or whose sides may both set bit 7"),
  or one side's bits unknown, a parameter not bound
  (`micro_or_param`) or a value that may be negative
  (`micro/or_negative.ptx`), or bounded by more bits than it sets: a
  term c·v with v of 0 or 1 sets c's bits, `may_set` all from c's lowest
  to its highest (`micro/and_bit_select.ptx`'s `micro_and_select_or`,
  `2080·(%tid.x mod 2) | 128·⌊(%tid.x mod 16)/8⌋`: "or whose sides may
  both set bit 7", where ptxas adds them); a division by a constant that is no power
  of two, which LLVM writes as `mul.hi` by a magic number and a shift
  (`micro/magic_div.ptx`: "`mul.hi.u32` keeps a product's upper half";
  sglang `bmm_kernel_persistent`'s `/ 148` leaves its trip count
  unknown); a floor or mod of a value whose base is no multiple of the
  divisor, such as a 0-based counter's `i % 2` read before its
  increment, `(k − 1) mod 2`, which is the same for every thread and so
  a uniform variable the loop defines (`micro/counter_mod_offset.ptx`:
  `requested 16 B, unique unknown` where ptxas stores 8 distinct bytes;
  k12's double buffer `tile % 2`, `uniform(%r10@365)`); an element a
  `mov.b64 {…}` unpacks from a register no single `mov` packed, after a
  join of packs (`micro/unpack.ptx`'s `micro_unpack_join`; b200
  `bmm_kernel_persistent`'s `cp.async` sources, whose `%rd175` three
  `mov`s pack, lines 372, 540 and 711), with another element
  count (`micro_unpack_halves`), from 64-bit arithmetic
  (`micro_unpack_wide`) or from an unpacked element
  (`micro_unpack_twice`), or packed as an immediate wider than its
  element (`micro_unpack_imm`).
- **Offsets past 2^62 bytes are not counted**: an access whose lane
  offsets over the block, or whose loop's shifts over its trips, reach
  2^62 bytes has unknown sectors per warp or unknown unique loop bytes
  (`micro/offsets_wide.ptx`: `warp: ? (lane offsets reach 2^62
  bytes)`; `requested 32 B, unique unknown` for 8 distinct stores).
- **A loop footprint takes an address holding a uniform variable the
  loop computes as unread**: requested bytes only, unique and sector
  bytes unknown (`micro/uniform_address.ptx`, `micro_uniform_in_loop`:
  requested 2048 B, unique unknown).
- **Unique bytes per loop assume different pointer parameters are
  disjoint** (the cross-entropy's second pass, `tests/lit/
  gluon-ce-chunk.test`: the load and the store count 131,072 B
  unique for the 131,072 B requested; aliasing would halve it), and
  an offset from a pointer to stay in its allocation; bases on one
  pointer, or naming no single pointer, may overlap and give a range
  (`micro/one_pointer_two_bases.ptx`: a[k] and a[k + n] count
  16–32 B, 16 B with `--bind 1:n=0`; `micro/uniform_bases.ptx`), count
  the loop's own blocks only, and need a numeric trip count, the block
  shape, constant lane and counter coefficients, and every access's
  address (else `unique unknown`, `micro/footprint_unknown.ptx`), and
  are bounds where which threads run the loop is not known. k5's K loop at
  256³ (`k5-footprint.test`): requested 65,536 B, unique 65,536 B;
  k1's (`k1-footprint.test`): 1,048,576 B, 32,768 B, a bound until M and
  the grid are given, which its guard `x < M && y < N` reads.

### Reported as unknowns

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
- **Trip shapes.** Recognised: nvcc's in-place counter (`add r, r, c`
  then `setp` in the latch), countdown, derived-register latch, nvcc's
  unroll main+remainder pair, LLVM's two-register counter and its
  predicate-phi two-trip loop. Not recognised, reported as
  `trips = unknown`: grid-stride loops without `--grid`, whose step is
  `%nctaid.x` ("induction step is not a constant"), data-dependent
  bounds, multi-exit loops, inner loops bounded by an
  enclosing loop's counter (`micro/triangular.ptx`), a latch `setp`
  that combines its comparison with a predicate
  (`a_comparison_combined_with_a_predicate_is_not_a_trip_count`) or
  compares floats (`micro/latch_float.ptx`: "latch condition compares
  floating-point values"), a bound through a `selp` or shift of the
  CTA index, the mbarrier spin-waits
  (`mbarrier.test_wait` defines the predicate), and the persistent
  tile loop, which has a latch and an exit per partition. Bounds the
  tracer does not read, by the loop's source line: a division by a
  parameter not bound (`micro/trip_div_param.ptx` reads with `--bind`),
  a float immediate and an `and.pred` latch predicate (sol-study census
  `00511.ptx` `:121`, `00908.ptx` `:155`), and an `atom` result
  (`01542.ptx` `:223`).
  Over the census's 3,027 sm_100a loops (`missed-opt-census/art/sm100v3`,
  inferred arguments), 953 have a trip count: 1,445 are mbarrier waits
  and 411 read loaded values, 40 depend on `%ctaid` (15 through a
  `selp`, 7 a shift), 93 are grid-stride loops, 1 unpack, 20 atomics,
  20 divisions, 17 multi-exit.
- **Counters the recurrence search refuses** (a header φ every back
  edge feeds back plus one unguarded step in the loop's own blocks, or
  a directly nested loop's exit after a constant trip count,
  `src/analysis/scalar/recurrences.rs`): a register stepped by a guarded
  write (`micro/guarded_step.ptx`, where ptxas 13.4 stores once at
  param_0) is no counter; its guard the same for every thread, it reads
  `[4 * uniform(%r1@29) + param_0]`, varying with the loop.
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
  instructions are counted, and a kernel calling one does not
  round-trip (`micro/call_returns.ptx`: "Call to 'zero' requires call
  prototype").
- **Sibling loops on one source line** beyond the unroll pair are
  reported as variants and excluded from totals.
- **In a kernel with a cycle of more than one entry**, the uniformity
  analysis takes every value defined on a cycle and read outside it as
  divergent, whatever the cycle's exits compare
  (`a_value_from_a_cycle_with_two_entries_is_divergent_after_it`).
- **`setp`s not read as thread selectors.** A floating-point `setp`
  compares floats, not the traced integer forms, and `.f16x2` and
  `.bf16x2` each half on its own (PTX ISA §9.7.6.2, §9.7.7.2); an
  `lt`, `le`, `gt` or `ge` is read only when both operands lie in the
  range of its type, [0, 2^width) or [−2^(width−1), 2^(width−1)), for
  every `%tid` below 1024 (§10.2) and every `%ctaid` below the grid,
  so not on a CTA index when no grid is given (b200
  `tut_sg_prefixdense`'s `128·%ctaid.x + 64 < 128·%ctaid.x + 128`); an
  `eq` or `ne` only when their difference lies in (−2^width, 2^width).
  Otherwise what it guards is a bound (`micro/setp_float.ptx`,
  `micro/setp_f16x2.ptx`, `micro/setp_unsigned.ptx`,
  `micro/setp_wraps.ptx`: `(predicated)`, `store <= 256 B`).
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
  threads per CTA, as the record measures, `b200-mixed-index-guard.test`),
  and otherwise per CTA a bound, the lanes in at some CTA
  (`micro/cta_mask.ptx`: under 16 CTAs `<= 64 threads`, where the last
  stores on 40; `micro/cta_guards.ptx`). The ncu comparison sums such a
  set's count over the CTAs (`ThreadSet::launch`; scan_tma's line 184,
  `b200-launch-counts.test`), but not past 2^24 CTAs on the axes its
  comparisons read or 2^26 lane evaluations
  (`a_launch_sums_each_cta_s_threads_and_warps`). Without a grid the masks
  Triton builds from `pid·BLOCK + arange` leave what they guard a bound
  (`micro/cta_mask.ptx`, `store <= 256 B`). A guard on a loaded value
  is an unknown selector too (`micro/guards.ptx`: the store total stays
  a bound). A comparison with the counter of a loop holding the block,
  its trip count a constant, is read per trip, and the totals per CTA
  and per launch count each trip's threads (`micro/counter_guard.ptx`:
  384 barriers and 300 stores per CTA, as measured), but the ncu
  predictions count it over all of the loop's trips as a bound
  (`fp8_gemm_kernel.lm_head_dx`'s `<block 4>`, `(k < 511)`, runs on 510
  of its loop's 512 trips, 12,533,760 times as its record measures, and
  reads `<= 12582912`, `gluon-counter-guard.test`). The census counts by
  cause are in the message of commit `ccf8220`.

### Presentation

- **Floors and mods of one variable are not merged back**, so a form
  can print longer than its value: k5's second `As` store reads
  `16 * ⌊%tid.x/8⌋ + 2 * (%tid.x mod 8)`, which is `2 * %tid.x`, and
  the B200 fused_rmsnorm's `4 * ⌊(⌊%tid.x/8⌋ mod 128)/4⌋` is
  `4 * ⌊%tid.x/32⌋` below 1024 threads.
- **A refused φ names the obstacle on one of its source chains**: in
  each Gluon `attn_fwd_ws` kernel 76 access rows say only "depends on
  special register %ctaid.x" (34 in `tma_best`, 24 in
  `bmm_kernel_persistent`), not that a join of two values stopped the
  trace.
- Per-thread totals of a warp-specialized kernel are `<=` the sum of
  its partitions, since a thread runs one of them: the `threads` column
  of the block table says which threads run each block (`128
  (⌊%tid.x/32⌋ < 4)`), and the per-CTA totals count each block on its
  own threads, but there is no per-role view of the per-thread numbers.
- A kernel's `unknowns:` line lists what leaves a count incomplete
  (unparsed statements, unclassified instructions, unknown trip
  counts, calls), not unknown addresses, which are only on their access
  rows: `micro/footprint_unknown.ptx` reads `unknowns: none` beside a
  store whose address is `unknown`
  (`tests/lit/micro-footprint-unknown.test`).
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
  identical to the original's for 87 of the 96 fixtures
  (`tests/lit.sh -v tests/lit/roundtrip`, ptxas 13.4). k14's and
  `micro/address_operands.ptx`'s are rejected because the in-kernel
  `.local` declaration is discarded (`Unknown symbol
  '__local_depot0'`, `'lbuf'`). k1's and k2's assemble to different
  SASS because `.pragma "nounroll";` is discarded, so ptxas unrolls
  the loop it annotated. `micro/texture_outputs.ptx`,
  `call_returns.ptx`, `unparsed_writes.ptx` and `vector_register.ptx`
  lose what the parser drops (above). These are allowlisted in
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
  3072 warp `mma` per CTA at K=768 times 768 CTAs
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

- A value the tracer cannot read but every thread of the CTA shares as
  an affine variable, `uniform(v)`, keyed by its register SSA value:
  addresses and loop footprints read it (`micro/uniform_address.ptx`,
  `micro/uniform_counter.ptx`, `micro/uniform_bases.ptx`); trip counts
  and thread selectors do not. Register SSA
  itself
  (`src/ssa.rs`) is what the tracer, the uniformity analysis, trip
  counting and the predicate reader read.

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
  instructions executed, and on a line with a global memory
  instruction Nsight Compute's L2 theoretical sectors, their ideal and
  each such SASS's access size (`tests/fixtures/k5/k5.sm_89.256.ncu.json`;
  every Gluon kernel's first launch, `tests/fixtures/gluon/*.ncu.json`).
  The B200 records (`tests/fixtures/b200/*.ncu.json`), which have no
  sectors, come from
  sol-study's reports instead: `import_reports.py` attributes their
  per-SASS counts to PTX lines through the same `.loc` recompile under
  Triton's ptxas command, and takes each launch's parameters from a
  replay of the report's own command as the B200, matched by SASS and
  grid, with `i64:`, `i1:` and `tensormap:128` for the parameters of a
  tensor descriptor, which launch.c does not take; the PTX
  must still assemble to the recorded SASS (`b200-sass.test`, under
  ptxas 13.4 only). `ci.sh` does not re-measure them with
  `measure.py --check`, which would launch on zeroed inputs.
  `analyze --ncu RECORD` takes the record's grid, block shape and
  `u32` and `i32` parameters for `--grid`, `--launch` and `--bind`, and
  refuses a record whose
  lines are not the PTX's (`tests/lit/k5-ncu-stale.test`), and
  gives each block a verdict (`kernels[].measured`): a line whose
  `CountConstraint` is not `Free` and whose SASS is one opcode must
  equal an exact prediction or stay within a bound. A prediction is
  per-CTA threads times trips times CTAs, or, where the threads differ
  by CTA, their sum over the grid times trips. Each block also
  counts its instructions, those with SASS and those that agree, and
  lists every line whose measured count is not the predicted number,
  with its SASS opcodes (`differing`); the text report prints them
  under `measured`. A line whose global SASS all have the width of its
  one PTX global access also compares its L2 sectors and their ideal
  over the launch, the sectors per warp request times the warps with a
  lane that runs it, a range over the address's alignment
  (`kernels[].measured[].sectors`, `tests/lit/ncu-sectors.test`; the
  text report counts the lines that agree and prints those that differ,
  none in the Gluon records, `tests/lit/gluon-ncu.test`); not
  compared: a generic access, a line whose count is unknown, and the
  B200 records. `ci.sh` measures every record again on the GPU it
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
