# PTX ISA instruction coverage audit

How every instruction in the PTX ISA manual's Instructions chapter
([§9.7](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instructions))
is handled by `src/analysis/instruction_counts/classify.rs`; where handling is absent or wrong, a
recommendation; and an explicit list of instructions that do not fit
the tool's current model. Tables follow the manual's own order and
hierarchy, one row per instruction (sections that define several
instructions get one row each).

Pinned to: **PTX ISA 9.3** (the docs page as fetched 2026-07-16) and
`classify.rs` as of the last commit that touched this file
(`git log -1 -- docs/ptx-instruction-coverage.md`).

The instruction inventory is pinned to that ISA version; implementation
coverage is maintained with changes to the classifier, as required by
`CLAUDE.md`. The `capabilities` verb remains a missing feature in
`PLAN.md`; these tables are maintained manually until it exists.

For register uses/definitions, execution effects, and SSA readiness, see the
[register-effects audit](ptx-register-effects-audit.md). Counting classification
is not a complete instruction-effects contract. That audit independently checks
186 non-excluded manual sections and documents current parser/affine-analysis gaps,
including corrections to stale parsing claims in the tables below.

## The tool's model (what "fits" means)

`instruction_counts/measurement.rs` owns the accounting axes (`Precision`,
`Pipe`, `Direction`) and contribution records, with `Space`
from `ptx/modifier.rs`; `instruction_counts/classify.rs` maps
instructions to those records.
Scalar instruction interpretation lives in `affine/affine_analysis/transfer.rs`, with
instruction types and forms (`Form`) in `ptx/modifier.rs` and special-register sharing in
`ptx/special_registers.rs`. It leaves the value of an `add`, `sub`, `mul`, `mad`, `div`
or `rem` unknown when its type is a packed integer, which "computes per packed lane"
(`micro/packed_add.ptx`), or a float, which "is floating-point arithmetic"
(`micro/float_mul.ptx`).

The parser accepts instruction-shaped statements as
`mnemonic + modifiers + operands` (`src/ptx/parse/parser.rs`). The
mnemonic must be an instruction keyword, `Opcode` in `src/ptx/ir.rs`:
§4.3.2 Table 2, plus `fabric`, which Table 2 omits and §9.7.10.5
documents. Any other word leaves the statement unparsed, `nop`
included (ptxas 13.4: "Not a name of any known instruction";
`mnemonics_outside_the_isa_are_unparsed`). Modifiers and operands carry
no per-instruction knowledge. A dotted opcode like `cp.async.bulk.tensor`
becomes mnemonic `cp` with modifiers `async`, `bulk`, `tensor` — so
one `match` arm in `classify.rs` covers every dotted variant of a base
mnemonic; the "Today" column names the arm where routing matters.

An instruction that fails parsing inside a kernel becomes `Stmt::Unparsed`
and never reaches the classifier. Over the committed corpus that is
policed by CI (`tests/parse-allowlist.txt`: two micro fixtures, an array element operand and a vector register's declaration); at
`analyze` time each one's line says `not parsed (not counted)`
(`unparsed_statements_say_they_are_not_counted`), as an unclassified
instruction's says `not classified`. (The ISA's `|`-separated destination
operands — `setp ... p|q`, `match.all.sync d|p`, `elect.sync d|p` —
were the verified unparsed instance until the parser learned them.)

Past the parser, each instruction has zero or more
`Contribution { kind, count, memory_operand }` records, none for non-flop
arithmetic, sync, warp collectives, control and hints.

| `MeasureKind` | Instructions | `count` |
|---|---|---|
| `Flops { pipe, precision }` | CUDA-core, tensor-core and SFU arithmetic, FP atomics | FLOPs |
| `Bytes { space, direction }` | Loads, stores, copies, atomics and reductions | Bytes |
| `UnquantifiedBytes { space, direction }` | The same, of a size PTX does not give | 1 |
| `UnknownOps` | Unsupported instructions | 1 |

Collection attaches the statement `provenance` to each
contribution to produce `Measurement` records. `cp.async` emits separate
read and write measurements; FP `atom.add` emits read bytes, write bytes,
and atomic FLOPs. Every instruction is counted once, including ignored
instructions, from the block's statement indices rather than the
measurement records. Memory contributions identify
which memory operand they reference, so an atomic's two sides share one
address while a copy retains its source and destination.

Every count is per thread per execution: a warp-collective instruction
contributes its warp total divided by the 32 lanes that issue it (PTX ISA
§4.5.1). Loop-trip and launch multiplication happen at report time.
Copies of one opcode with different immediate sizes each count their own bytes.
Contribution annotations are per execution with a true
guard, a `cp.async`'s global read on a thread that reads its source, marked
`<=` with a src-size register.

Atomic/reduction FP add/min/max contribute one FLOP per scalar element,
including packed and vector multiplicity, on the `atomic` accounting pipe.
Integer, bitwise, exchange and CAS operations contribute no FP FLOPs.
Atomic FLOPs are summed apart from CUDA-core FLOPs.
The byte convention is retained: `atom` read+write, `red` write-only.
These are requested bytes, not estimates of DRAM traffic. Tests in
`tests/instruction_contributions.rs` cover scaling, predication, address
attribution, unknown work, and the distinct-instruction identity.

An instruction can now combine arithmetic, memory and other contributions.
Work owned by larger groups and quantities unavailable from PTX still
require the decisions below. The tables use compact descriptions such as
`Bytes { space, Load, bytes }` for a contribution and its count, and
none for an instruction with no contribution.

Coverage checks account for classified and unknown instructions and require
every unknown corpus mnemonic to appear in `tests/classify-allowlist.txt`.
That allowlist contains `wgmma`, used by `micro/unknown_in_loop.ptx` to
pin lower-bound reporting. The corpus has expanded beyond the original
nvcc ladder; see PLAN.md for its producers and fixtures.

Separately from its contributions, each instruction has a `CountConstraint`
(`src/ptx/count_constraint.rs`): how far the ISA
lets ptxas change how often it executes. Unit tests in that file pin
each clause.

| Constraint | Instructions | ISA |
|---|---|---|
| `Preserved` | `.volatile`, `.mmio` | §8.4.2: "The number of volatile instructions (not operations) executed by a program is preserved"; §8.4.1: mmio reads and writes are "always performed" and "never combined" |
| `Constrained` | `.relaxed`/`.acquire`/`.release`/`.acq_rel`; `atom`, `red`; `bar`, `barrier`; `fence`, `membar` except `.proxy`; `mbarrier.arrive`, `arrive_drop`, `test_wait`, `try_wait` | §8.4 strong and synchronizing operations; `atom`/`red` default to `.relaxed` (§9.7.14.5, §9.7.14.6), `fence` to `.acq_rel`, `membar` is `fence.sc` (§9.7.14.4); `mbarrier.arrive`/`arrive_drop` default to `.release`, `test_wait`/`try_wait` to `.acquire` (§9.7.14.16.16, §9.7.14.16.17, §9.7.14.16.19) |
| `Constrained` | `add`, `sub`, `mul` with `.rn`/`.rz`/`.rm`/`.rp`; `fma`, `mad` with one | §9.7.3.3–§9.7.5.2: an explicit rounding modifier "is treated conservatively by the code optimizer"; §9.7.3.6, §9.7.3.7: one rounding, by `.rnd` |
| `Free` | everything else, including the default `.weak` `ld`/`st` (§9.7.9.8) and unrounded `add`/`mul`, which "may be optimized to use fused-multiply-add instructions" (§9.7.3.3) | — |

Row verdict vocabulary: **OK** (correct as-is), **OK (deferred)**
(deliberately `UnknownOps` pending a future family, with its tier),
**Gap** (should have an arm today; recommendation given), **Wart**
(handled, but with a defect), **⚠ model misfit** (needs a model
decision, not just an arm).

---

## §9.7.1 Integer Arithmetic Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.1.1 | `add` | none (shared arm with FP `add`; FP type modifier absent ⇒ integer) | OK — address math is not flops. |
| 9.7.1.2 | `sub` | none (same mechanism) | OK. |
| 9.7.1.3 | `mul` | none (incl. `.hi/.lo/.wide`) | OK. |
| 9.7.1.4 | `mad` | none | OK — deliberate divergence from v1, which gave integer `mad` 2 flops of precision "Other"; documented and regression-tested. |
| 9.7.1.5 | `clmad` | none | OK — carry-less multiply-add. |
| 9.7.1.6 | `mul24` | none | OK. |
| 9.7.1.7 | `mad24` | none | OK. |
| 9.7.1.8 | `sad` | none | OK. |
| 9.7.1.9 | `div` | none (the `div` arm: no FP type modifier ⇒ integer, exactly as `rem`) | OK — the former collateral-Unknown wart is fixed. |
| 9.7.1.10 | `rem` | none | OK — and the standing counterexample to the `div` wart. |
| 9.7.1.11 | `abs` | none (via the FP-check arm) | OK. |
| 9.7.1.12 | `neg` | none (same) | OK. |
| 9.7.1.13 | `min` | none (same) | OK. |
| 9.7.1.14 | `max` | none (same) | OK. |
| 9.7.1.15 | `popc` | none | OK. |
| 9.7.1.16 | `clz` | none | OK. |
| 9.7.1.17 | `bfind` | none | OK. |
| 9.7.1.18 | `fns` | none | OK. |
| 9.7.1.19 | `brev` | none | OK. |
| 9.7.1.20 | `bfe` | none | OK. |
| 9.7.1.21 | `bfi` | none | OK. |
| 9.7.1.22 | `szext` | none | OK. |
| 9.7.1.23 | `bmsk` | none | OK. |
| 9.7.1.24 | `dp4a` | none | ⚠ model misfit (mild) — a 4-way int8 dot product is 8 integer multiply-adds of real arithmetic throughput, but the roofline counts FP flops only. No contribution is right under the current model; int8 inference workloads would need an integer-ops axis (see misfits §D). |
| 9.7.1.25 | `dp2a` | none | Same as `dp4a`. |

## §9.7.2 Extended-Precision Integer Arithmetic Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.2.1 | `add.cc` | none (mnemonic `add` + `cc` modifier — incidental but correct routing) | OK. |
| 9.7.2.2 | `addc` | none | OK — both halves of a carry chain now classify; the former wart. |
| 9.7.2.3 | `sub.cc` | none (via `sub`) | OK. |
| 9.7.2.4 | `subc` | none | OK. |
| 9.7.2.5 | `mad.cc` | none (via `mad`) | OK. |
| 9.7.2.6 | `madc` | none | OK. |

## §9.7.3 Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.3.1 | `testp` | none | OK. |
| 9.7.3.2 | `copysign` | `Flops` ×1 (verified) | OK — sign-bit transfer occupies the FP pipe; 1 flop per the Williams convention. |
| 9.7.3.3 | `add` | `Flops` ×1 × lanes | OK — incl. the packed `f32x2` form (sm_100+) via the lane multiplier. |
| 9.7.3.4 | `sub` | `Flops` ×1 × lanes | OK. |
| 9.7.3.5 | `mul` | `Flops` ×1 × lanes | OK. |
| 9.7.3.6 | `fma` | `Flops` ×2 × lanes | OK — the standard 2-flops-per-FMA convention. |
| 9.7.3.7 | `mad` | `Flops` ×2 | OK — FP `mad` is fused on all modern targets; identical to `fma` for counting. |
| 9.7.3.8 | `div` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy (misfit §D, decided): one flop per result on the `sfu` pipe, whatever the `.approx`/`.full`/`.rnd` form expands to in SASS. |
| 9.7.3.9 | `abs` | `Flops` ×1 | OK — with the cross-check caveat: the planned NCU measured-flops formula (`2·ffma + fmul + fadd`) excludes it, so abs-heavy kernels will show a static-vs-measured gap. Static side is defensible; document when NCU import lands. |
| 9.7.3.10 | `neg` | `Flops` ×1 | OK — same caveat as `abs`. |
| 9.7.3.11 | `min` | `Flops` ×1 | OK — same caveat. The three-input `.f32` form (sm_100) performs two two-input minimums and counts 1, the per-instruction rule; ptxas emits one `FMNMX3` (535 statements in 7 files of the Blackwell corpus, PLAN.md `trap` entry). |
| 9.7.3.12 | `max` | `Flops` ×1 | OK — same caveat. The three-input `.f32` form (sm_100) performs two two-input maximums and counts 1, the per-instruction rule; ptxas emits one `FMNMX3` (46,676 statements in 1,147 files of the Blackwell corpus, `quant…_per_token_group_quant_8bit` among the b200 fixtures). |
| 9.7.3.13 | `rcp` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.14 | `rcp.approx.ftz.f64` | `Flops { Sfu, f64, 1 }` (same `rcp` arm) | OK. |
| 9.7.3.15 | `sqrt` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.16 | `rsqrt` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.17 | `rsqrt.approx.ftz.f64` | `Flops { Sfu, f64, 1 }` (same `rsqrt` arm) | OK. |
| 9.7.3.18 | `sin` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.19 | `cos` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.20 | `lg2` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.21 | `ex2` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.22 | `tanh` | `Flops { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |

## §9.7.4 Half Precision Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.4.1 | `add` | `Flops` ×1 × lanes (`f16x2`/`bf16x2` ⇒ ×2) | OK. |
| 9.7.4.2 | `sub` | `Flops` ×1 × lanes | OK. |
| 9.7.4.3 | `mul` | `Flops` ×1 × lanes | OK. |
| 9.7.4.4 | `fma` | `Flops` ×2 × lanes (verified: `fma.rn.f16x2` = 4 flops) | OK. |
| 9.7.4.5 | `neg` | `Flops` ×1 × lanes | OK (NCU-formula caveat as §9.7.3.10). |
| 9.7.4.6 | `abs` | `Flops` ×1 × lanes | OK (same caveat). |
| 9.7.4.7 | `min` | `Flops` ×1 × lanes | OK (same caveat). |
| 9.7.4.8 | `max` | `Flops` ×1 × lanes | OK (same caveat). |
| 9.7.4.9 | `tanh` | `Flops { Sfu, f16/bf16, lanes }` | OK — `.f16x2`/`.bf16x2` count two results. |
| 9.7.4.10 | `ex2` | `Flops { Sfu, f16/bf16, lanes }` | OK — the fast-softmax core in attention kernels, two results per packed instruction. |

## §9.7.5 Mixed Precision Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.5.1 | `add` (e.g. `add.rn.f32.bf16`) | `Flops` ×1, bucketed **f32** (first FP modifier = destination type; verified) | OK — mixed ops land in the accumulator's precision bucket, which is what roofline tables want. Recommend a comment in `fp_precision_and_lanes` pinning the "PTX writes dst type first" assumption it relies on. |
| 9.7.5.2 | `sub` | Same | OK — same recommendation. |
| 9.7.5.3 | `fma` | `Flops` ×2, bucketed by dst type | OK — same recommendation. |

## §9.7.6 Comparison and Selection Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.6.1 | `set` | none | OK — produces a 0/1 value in a data register, but comparisons are not flops under the convention. |
| 9.7.6.2 | `setp` | none | OK, the dual-destination form `setp ... p|q, a, b` included: it parses as one `p|q` operand (`multiple_destination_registers`) and is counted (`micro/guards.ptx`'s `setp.lt.u32 %p2|%p8`). Its unsigned spellings `lo`, `ls`, `hi`, `hs` read as `lt`, `le`, `gt`, `ge` of the `.u` type ("may be used instead", §9.7.6.2 Integer Notes; `micro/latch_forms.ptx`); `lo` to `hs` of a `.s` type, `lt` to `ge` of a `.b` type and `.ftz` of an integer type are no integer comparison (`comparisons_read_their_predicate_width_and_bool_op`). |
| 9.7.6.3 | `selp` | none | OK. |
| 9.7.6.4 | `slct` | none | OK. |

## §9.7.7 Half Precision Comparison Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.7.1 | `set` (f16 variants) | none (same arm) | OK. |
| 9.7.7.2 | `setp` (f16 variants) | none (same arm) | OK, `p|q` included (`micro/setp_f16x2.ptx`'s `setp.lt.f16x2 %p1|%p2`). |

## §9.7.8 Logic and Shift Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.8.1 | `and` | none | OK — the `.pred` split is correct and tested. |
| 9.7.8.2 | `or` | Same | OK. |
| 9.7.8.3 | `xor` | Same | OK. |
| 9.7.8.4 | `not` | Same | OK. |
| 9.7.8.5 | `cnot` | none | OK. |
| 9.7.8.6 | `lop3` | none | OK. |
| 9.7.8.7 | `shf` | none | OK — funnel shift, nvcc's 64-bit shift lowering. |
| 9.7.8.8 | `shl` | none | OK. |
| 9.7.8.9 | `shr` | none | OK. |

## §9.7.9 Data Movement and Conversion Instructions

§9.7.9.1 (Cache Operators) and §9.7.9.2 (Cache Eviction Priority
Hints) define modifiers, not instructions; they are transparent to
requested-bytes accounting by construction (verified:
`ld.global.nc.L2::128B.b128` counts 16 bytes), which is correct — a
cache hint changes where bytes are served from, not how many are
requested.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.9.3 | `mov` | none | OK — includes special-register reads (`mov.u32 %r, %tid.x`) and a variable's address plus an offset, `avar+imm` (`Operand::Offset`, `micro/address_operands.ptx`), which `cvta`, `mapa` and `getctarank` also take; an immediate address `[immAddr]` parses too. |
| 9.7.9.4 | `mov` (vector pack/unpack form) | none (same arm) | OK. |
| 9.7.9.5 | `shfl` (deprecated) | none | OK — same arm as `shfl.sync`; zero DRAM bytes is the correct accounting for intra-warp exchange. |
| 9.7.9.6 | `shfl.sync` | none | OK. |
| 9.7.9.7 | `prmt` | none | OK. |
| 9.7.9.8 | `ld` | `Bytes { space, Load, width × v2/v4/v8 }` | OK — space from modifiers with `Generic` as its own honest bucket (`cvta`-provenance refinement is documented anti-scope until a fixture emits generic loads); `.shared::cta` folds into `Shared`, `.shared::cluster` distinct, `.param::entry` and `.param::func` fold into `Param`, and an `ld.param::entry` reads its kernel parameter as `ld.param` does (`micro/param_entry.ptx`); `volatile`/cache-op/eviction modifiers transparent. The parser keeps an address's `.unified` (`micro/unified_load.ptx`). The uniformity analysis takes the loaded value from the address alone, not from the cache_policy `.L2::cache_hint` adds, which "is treated as a performance hint only" (§9.7.9.8; `a_load_does_not_read_its_cache_policy`). |
| 9.7.9.9 | `ld.global.nc` | `Bytes { Global, Load, bytes }` (via the `ld` arm; verified) | OK. Its cache_policy is no input of the loaded value either (§9.7.9.9: "a performance hint only"). |
| 9.7.9.10 | `ldu` | `Bytes { Global, Load, bytes }` | OK. |
| 9.7.9.11 | `st` | `Bytes { space, Store, bytes }` | OK. A kernel's `st.param`, and its `ld.param` of a `.param` variable that is not one of its parameters, pass a call's arguments and return value (§5.1.6.4) and contribute nothing (`micro/call_returns.ptx`). |
| 9.7.9.12 | `st.async` | `Bytes` store via the `st` arm (verified: `st.async.shared::cluster.b32` ⇒ `{SharedCluster, Store, 4}`) | OK for byte accounting. ⚠ model misfit (mild, §B) — the bytes land in a *remote* CTA's shared memory with mbarrier completion; per-thread attribution is still right, but a future cluster-traffic view would need the remote/local distinction the modifier already carries. No action until a cluster fixture exists. |
| 9.7.9.13 | `multimem.st.async` | `UnknownOps` | OK (deferred, Tier 4) — ⚠ misfit §F: one store fans out to N GPUs' memory. |
| 9.7.9.14 | `st.bulk` | `UnquantifiedBytes { Shared, Store }` (verified) | OK — the size is an operand, not a type modifier, so `UnquantifiedBytes` is the honest result. Recommend quantifying the immediate-operand case when the bulk-copy family lands (Tier 2). |
| 9.7.9.15 | `multimem.ld_reduce` | `UnknownOps` | OK (deferred, Tier 4) — ⚠ misfit §A+§F: a load *and* a reduction across N GPUs' replicas in one instruction. |
| 9.7.9.15 | `multimem.st` | `UnknownOps` | OK (deferred, Tier 4) — misfit §F. |
| 9.7.9.15 | `multimem.red` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.9.16 | `prefetch` | none | OK — a prefetch duplicates a later architectural load; counting it would double-count requested bytes. (Genuinely ambiguous under a "DRAM traffic" reading — a prefetch of never-loaded data is real traffic — but the tool's contract is requested bytes, and there no contribution is right.) |
| 9.7.9.16 | `prefetchu` | none | OK — same. |
| 9.7.9.17 | `applypriority` | none | OK — cache-state hint. |
| 9.7.9.18 | `discard` | none | OK — invalidates lines without writeback; no data movement. |
| 9.7.9.19 | `createpolicy` | none | OK — builds a cache-policy register, touches no memory. |
| 9.7.9.20 | `isspacep` | none | OK. |
| 9.7.9.21 | `cvta` | none | OK — address-space cast, not a value conversion; tested. |
| 9.7.9.22 | `cvt` | none | OK — and future-proof: every format including FP8/FP6/FP4 (`e4m3`, `e5m2`, `e2m1`, …) lands here with no per-format code. An integer `cvt.sat`, which "limits the result to MININT..MAXINT for the size of the operation", gives no value to the tracer (`micro/cvt_reads.ptx`). |
| 9.7.9.23 | `cvt.pack` | none (mnemonic `cvt` + `pack` modifier) | OK. |
| 9.7.9.24 | `mapa` | none | OK — an address computation into a peer CTA's shared memory; the access it feeds is counted where it happens. |
| 9.7.9.25 | `getctarank` | none | OK. |

### §9.7.9.26 Asynchronous copy

Rows below route through the bare `cp` (or `multimem`) mnemonic.
Ordinary `cp.async` and its synchronization forms are handled, and
`cp.async.bulk.tensor`; the other bulk, tensor and multimem variants
remain visible, counted `UnknownOps` work.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.9.26.3.1 | `cp.async` | `Bytes` with Global read = src-size or cp-size and Shared write = cp-size. cp-size is the third operand. A fourth, without `.L2::cache_hint`, is src-size or ignore-src. Under `.L2::cache_hint` the last operand is the cache policy, so a fourth operand alone is the policy and a fifth follows a src-size or ignore-src. The global read happens on the threads whose ignore-src is false (`ThreadSets::sources`), read as a guard is: its sectors per warp, loop footprint, global load totals and sector comparison count them. A src-size register reads at most cp-size (`at_most`: `sectors/req <= 17 (ideal <= 16)`) on threads not known (PLAN.md, Reported as unknowns) | OK — the `cp` arm, pinned by k12/k14 (nvcc writes `..., 16, 16`) and the Gluon GEMMs (Triton writes `..., 0x10, 0x10`); the positions, an immediate cache policy no size, by `cp_async_is_a_global_read_and_a_shared_write`; the sectors per request, totals and sector comparison by `micro/cp_async_ignore_src.ptx` and its RTX 4090 record (`micro-cp-async-ignore-src.test`, `tests/ncu_comparisons.rs`'s `copies_that_ignore_their_source_read_their_predicted_sectors`) and `micro/cp_async_src_size.ptx`. `collect` records one memory instruction with two byte measurements: a global load and a shared store. Immediate source size zero has zero global bytes and sectors per request and per loop, while retaining shared writes and issued copies (`tests/instruction_contributions.rs::zero_source_copy_has_no_global_sectors_but_keeps_shared_writes`). |
| 9.7.9.26.3.2 | `cp.async.commit_group` | none | OK — moves no bytes. |
| 9.7.9.26.3.3 | `cp.async.wait_group` | none | OK. |
| 9.7.9.26.3.3 | `cp.async.wait_all` | none | OK. |
| 9.7.9.26.4.1 | `cp.async.bulk` | `UnknownOps` | OK (deferred, Tier 2) — ⚠ misfit §B+§C: issued by one thread (or one elected thread) for a CTA-scale transfer — per-thread count multiplication would overcount by orders of magnitude; size may be a runtime register (→ `UnquantifiedBytes`). |
| 9.7.9.26.4.2 | `cp.reduce.async.bulk` | `UnknownOps` | OK (deferred, Tier 2) — ⚠ misfit §A+§B: bulk copy *and* reduction arithmetic in one instruction. |
| 9.7.9.26.4.3 | `cp.async.bulk.prefetch` | `UnknownOps` | OK (deferred, Tier 2) — target: none, consistent with the `prefetch` policy. |
| 9.7.9.26.4.4 | `multimem.cp.async.bulk` | `UnknownOps` | OK (deferred, Tier 4) — misfit §F. |
| 9.7.9.26.4.5 | `multimem.cp.reduce.async.bulk` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.9.26.5.2 | `cp.async.bulk.tensor` | An `UnquantifiedBytes` load from `.src` and store to `.dst`, global to shared::cta or to shared::cluster (`.multicast::cluster` included), or shared::cta to global; `.shared`, which ptxas 13.4 rejects here, reads as shared::cta; another pair of state spaces stays `UnknownOps` | OK — "an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space" (§9.7.9.26.5.2). `.dst` is the first state space named: the manual's own example `cp.async.bulk.tensor.3d.im2col.shared::cluster.global` and Triton put `.load_mode` before it. The bytes are the bounding box of the tensor map, "a 128-byte opaque object" that "describes the tensor properties and the access properties" (§5.5.8, §5.5.3.1), **permanently unknowable from the PTX text**: `UnquantifiedBytes` is the end state (misfit §C), unless a `--bind`-style flag supplies the map. The shared side's address is traced; the global side's reads `addr ? (the tensor map holds the address)`, and a loop holding one in its own blocks has no footprint. Pinned by `a_tensor_copy_moves_the_bytes_its_tensor_map_gives` and `micro/tensor_copies.ptx`: every `.dim`, `.load_mode`, completion mechanism, `.multicast::cluster`, `.cta_group` and `.L2::cache_hint` form, which ptxas 13.4 assembles to its UTMALDG or UTMASTG, and its `--dump-ast` to the same SASS (`tests/lit/roundtrip`); its annotated lines by `micro-tensor-copies.test`. Single-thread-issued, CTA-scale (§B): one op per issuing thread. |
| 9.7.9.26.5.3 | `cp.reduce.async.bulk.tensor` | `UnknownOps` | OK (deferred, Tier 2) — same, plus misfit §A (reduction). |
| 9.7.9.26.5.4 | `cp.async.bulk.prefetch.tensor` | `UnknownOps` | OK (deferred, Tier 2) — target: none per the prefetch policy. |
| 9.7.9.26.6.1 | `cp.async.bulk.commit_group` | none | OK — moves no bytes. |
| 9.7.9.26.6.2 | `cp.async.bulk.wait_group` | none | OK — moves no bytes. |
| 9.7.9.27 | `tensormap.replace` | `UnknownOps` | Gap (harmless) — edits a tensor-map descriptor in memory; negligible traffic. Recommend none, or a small fixed `Bytes` write — decide when TMA lands (Tier 3). |

## §9.7.10 Fabric Instructions

New in recent ISA revisions (multi-node NVLink fabric). All are
`UnknownOps` today via the `fabric` mnemonic — correct: cross-GPU
transfers are a different roofline (misfit §F). All Tier 4.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.10.5.1 | `fabric.try_get` | `UnknownOps` | OK (deferred, Tier 4) — remote read over the fabric. |
| 9.7.10.5.2 | `fabric.try_put` | `UnknownOps` | OK (deferred, Tier 4) — remote write. |
| 9.7.10.5.3 | `fabric.try_red` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A+§F (remote reduction). |
| 9.7.10.5.4 | `fabric.try_pullred` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.10.5.5 | `fabric.submit` | `UnknownOps` | OK (deferred, Tier 4) — target: none if ever handled. |
| 9.7.10.5.6 | `fabric.wait` | `UnknownOps` | OK (deferred, Tier 4) — target: none if ever handled. |

## §9.7.11 Texture Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.11.3 | `tex` | `UnknownOps` (verified) | OK (deferred, Tier 4) — an *improvement* over v1, which ignored it: a texture fetch moves real bytes, so silently zeroing corrupted AI. ⚠ misfit §F: footprint is coordinate-dependent and the fetch includes fixed-function filtering arithmetic — neither plain bytes nor flops. `UnknownOps` is the honest state for an out-of-audience family. |
| 9.7.11.4 | `tld4` | `UnknownOps` | OK (deferred, Tier 4) — same reasoning. |
| 9.7.11.5 | `txq` | `UnknownOps` | OK (deferred, Tier 4) — a metadata query; target: none if the family is ever handled. |
| 9.7.11.6 | `istypep` | none | OK — a predicate query that lives in the texture chapter. |

## §9.7.12 Surface Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.12.1 | `suld` | `UnknownOps` | OK (deferred, Tier 4) — moves bytes (v1 wrongly ignored it); descriptor-held geometry (misfit §C). |
| 9.7.12.2 | `sust` | `UnknownOps` | OK (deferred, Tier 4) — same. |
| 9.7.12.3 | `sured` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A (reduction on memory) + §C. |
| 9.7.12.4 | `suq` | `UnknownOps` | OK (deferred, Tier 4) — query; target: none. |

## §9.7.13 Control Flow Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.13.1 | `{}` | Parser-level block structure; never reaches the classifier | OK — the right layer. |
| 9.7.13.2 | `@` | Parser-level; the instruction counts on the threads whose guard holds (`ThreadSets::guard`), a `≤` (at-most) bound where that set is not known | OK — this is the model's designed answer to divergence: bounds, not probabilities. |
| 9.7.13.3 | `bra` | none | OK. |
| 9.7.13.4 | `brx.idx` | none (mnemonic `brx`) | OK. Thread sets: an unguarded one sends all or none of the threads its index is one value for to a target (`a_jump_table_on_a_warp_s_index_selects_whole_warps`). |
| 9.7.13.5 | `call` | none | OK at the classifier layer; the parser reads its `(…)` return and argument lists (`Operand::ParamList`, `micro/call_returns.ptx`), and skips `.calltargets` and `.callprototype`. ⚠ misfit §E: flops/bytes inside the callee are not attributed to the call site. Fine while the corpus is fully-inlined nvcc output; recommend a report-level named unknown when a kernel calls a function whose body the module doesn't contain (extern), so the gap is visible the day it matters. |
| 9.7.13.6 | `ret` | none | OK at the classifier layer. ⚠ The CFG reads a predicated `ret` as falling through, so what follows counts on every thread (PLAN.md; `micro/predicated_exit.ptx`). |
| 9.7.13.7 | `exit` | none | OK at the classifier layer. ⚠ The same for a predicated `exit` (`micro/predicated_exit.ptx`: the store after `@%p1 exit` counts 32 per CTA, none run at n = 20 on an RTX 4090). |

## §9.7.14 Parallel Synchronization and Communication Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.14.1 | `bar` | none | OK. Note `bar.red` also computes a predicate reduction across the CTA — a mild §A dual-nature; no contribution is the right roofline treatment (the reduction is predicate work, never flops). |
| 9.7.14.1 | `barrier` | none | OK — same family, same note for `barrier.red`. |
| 9.7.14.2 | `bar.warp.sync` | none (via `bar`) | OK. |
| 9.7.14.3 | `barrier.cluster` | none (via `barrier`) | OK. |
| 9.7.14.4 | `membar` | none | OK. |
| 9.7.14.4 | `fence` | none | OK — all variants (`fence.proxy.*`, `fence.mbarrier_init`, …) via modifier transparency. |
| 9.7.14.5 | `atom` | Read and write `Bytes`; FP add/min/max also emit `Flops { Atomic, precision, elements }` | OK — §A supported by independent contributions. One memory instruction, shared address provenance, two or three measurements. Contention/serialization remains outside the model. |
| 9.7.14.6 | `red` | Store `Bytes`; FP add/min/max also emit atomic FLOPs | OK — write-only requested-byte policy retained; one FP operation per scalar element, including packed/vector forms. |
| 9.7.14.7 | `red.async` | Store `Bytes` through the `red` arm, and FP arithmetic when applicable | Byte and arithmetic contributions follow `red`; async completion/mbarrier effects are not modeled. |
| 9.7.14.8 | `multimem.red.async` | `UnknownOps` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.14.9 | `vote` (deprecated) | none | OK. |
| 9.7.14.10 | `vote.sync` | none | OK. |
| 9.7.14.11 | `match.sync` | none | OK for the plain form (verified). **Wart**: `match.all.sync` with the optional `d|p` destination hits the parser gap (verified `Unparsed`) — fix with the shared `d|p` operand form (Tier 2 rider). |
| 9.7.14.12 | `activemask` | none | OK. |
| 9.7.14.13 | `redux.sync` | none | OK, with a flag: ⚠ misfit §A (mild) — it also performs the reduction arithmetic (integer; `.f32` min/max, sm_100a). Counted as communication, not as flops: an integer reduction has no roofline home, and an `.f32` one is 31 maximums per 32 values, no whole number per thread (b200 `quant…_per_token_group_quant_8bit` line 97, `redux.sync.max.f32`; 1,254 `.f32` statements in 294 files of the Blackwell corpus). |
| 9.7.14.14 | `griddepcontrol` | none | OK — it orders dependent grids and moves no counted work. |
| 9.7.14.15 | `elect.sync` | none (the `d|p` destination parses since the `|` operand change) | OK — warp-collective leader election. |
| 9.7.14.16.12 | `mbarrier.init` | none (via `mbarrier`) | OK — technically writes 8 bytes of shared memory to initialize the object; negligible by construction, no contribution is right. |
| 9.7.14.16.13 | `mbarrier.inval` | none | OK. |
| 9.7.14.16.14 | `mbarrier.expect_tx` | none | OK. |
| 9.7.14.16.15 | `mbarrier.complete_tx` | none | OK. |
| 9.7.14.16.16 | `mbarrier.arrive` | none | OK. |
| 9.7.14.16.17 | `mbarrier.arrive_drop` | none | OK. |
| 9.7.14.16.18 | `cp.async.mbarrier.arrive` | none (the `cp` arm's `mbarrier` modifier) | OK. |
| 9.7.14.16.19 | `mbarrier.test_wait` | none | OK. |
| 9.7.14.16.19 | `mbarrier.try_wait` | none | OK. |
| 9.7.14.16.20 | `mbarrier.pending_count` | none | OK. |
| 9.7.14.16.21 | `mbarrier.check_layout` | none (via `mbarrier` — future variants land free) | OK. |
| 9.7.14.17 | `tensormap.cp_fenceproxy` | `UnknownOps` | Gap (harmless) — target: none (Tier 3). |
| 9.7.14.18 | `clusterlaunchcontrol.try_cancel` | `UnknownOps` | OK (deferred, Tier 3) — persistent-kernel launch control; target: none. Note: kernels built around it re-derive their work assignment in a loop the trip-count model will honestly fail on — an audience question before a classifier one. |
| 9.7.14.19 | `clusterlaunchcontrol.query_cancel` | `UnknownOps` | OK (deferred, Tier 3) — target: none (unpacks a result handle). |

## §9.7.15 Warp Level Matrix Multiply-Accumulate Instructions

§9.7.15.1–.3 (shapes, data types, block scaling) are context sections.
All entries are warp-collective: their per-thread count is the warp
total over 32 (`WARP_LANES` in `classify.rs`), which divides exactly
for every shape × element type the ISA lists.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.15.4.3 | `wmma.load` | `Bytes { space, Load, bytes }` — matrix `.a` = M×K, `.b` = K×N, `.c` = M×N elements × element bits / 8 / 32; space from the `.global`/`.shared` modifier, `Generic` without one | OK — role split on the first modifier (`wmma` arm). v1's worst bug — bare `wmma` matched as an MMA, 8192 phantom flops per load — is pinned by the `wmma_loads_and_stores_are_fragment_bytes` unit test: a load is `Bytes`, never `Flops`. |
| 9.7.15.4.4 | `wmma.store` | `Bytes { space, Store, bytes }`, `.d` = M×N elements | OK. |
| 9.7.15.4.5 | `wmma.mma` | `Flops { Tensor, atype, 2·M·N·K / 32 }`; atype is `.f16` when only `.dtype.ctype` are given (§9.7.15.4.5) | OK for the FP kinds (f16, bf16, tf32, f64). Integer (`s8`/`u8`/`s4`/`u4`) and single-bit (`b1`) kinds stay `UnknownOps` deliberately: misfit §D, not flops. |
| 9.7.15.5.14 | `mma` | `Flops { Tensor, atype, ops·2·M·N·K / 32 }` — atype is the second type modifier (`.dtype.atype.btype.ctype`); ops = 4 for `.m8n8k4` with `.f16` (§9.7.15.5.1), else 1 | OK for the FP kinds (fp8 = `e4m3`/`e5m2`, f16, bf16, tf32, f64) — the `mma` arm, pinned by the mma_demo, k14 and Gluon fp8 GEMM fixtures; the fp8 count equals Nsight Compute's `sm__ops_path_tensor_src_fp8` on the c_fc launch (PLAN.md, hardware cross-check). Integer (`s8`/`u8`/`s4`/`u4`), single-bit (`b1`), fp6/fp4 (`e2m1`…) and `.block_scale` kinds stay `UnknownOps` deliberately (misfit §D: not FP flops, or scale-factor arithmetic beyond 2·M·N·K). |
| 9.7.15.5.15 | `ldmatrix` | `Bytes { Shared, Load, num × rows × cols × bits / 8 / 32 }` — shape from `.m8n8`/`.m16n16`/`.m8n16`, count from `.x1/.x2/.x4`, width from `.b16`/`.b8`; the padded `.b6x16_p32`/`.b4x16_p64` source formats are unquantified bytes | OK — the `matrix_fragments` arm, pinned by k14. Shared always: the ISA defines generic addressing here as pointing into `.shared`. |
| 9.7.15.5.16 | `stmatrix` | `Bytes { Shared, Store, … }`, same sizing | OK — same arm. |
| 9.7.15.5.17 | `movmatrix` | `UnknownOps` | OK (deferred, Tier 2) — register-file transpose, zero bytes; target: none (a warp-collective register exchange). |
| 9.7.15.6.3 | `mma.sp` / `mma.sp::ordered_metadata` | `UnknownOps` (the `mma` arm returns early on `.sp*`) | OK (deferred, Tier 2) — ⚠ misfit §D: 2:4-sparse flop counting is genuinely ambiguous (dense 2·M·N·K, matching NCU's convention, vs effective half). Decide with a fixture and NCU cross-check, and label the choice in the report. |

## §9.7.16 Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions

All `UnknownOps` today via the `wgmma` mnemonic. Tier 2 (Hopper: Triton
and CUTLASS emit these on H100).

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.16.5.2 | `wgmma.mma_async` | `UnknownOps` | OK (deferred, Tier 2) — ⚠⚠ misfit §B: executed by a *warpgroup* (4 warps); `Scope` has no such variant — the enum must grow before this lands. Also misfit §E: the A/B operands can stream directly from shared memory via descriptors, so its smem traffic has no `ld` instructions to count — decide whether the arm emits implicit smem bytes. Flops = 2·M·N·K per warpgroup. |
| 9.7.16.6.3 | `wgmma.mma_async.sp` | `UnknownOps` | OK (deferred, Tier 2) — same, plus the sparse convention (misfit §D). |
| 9.7.16.7.1 | `wgmma.fence` | `UnknownOps` | OK (deferred, Tier 2) — target: none. |
| 9.7.16.7.2 | `wgmma.commit_group` | `UnknownOps` | OK (deferred, Tier 2) — target: none. |
| 9.7.16.7.3 | `wgmma.wait_group` | `UnknownOps` | OK (deferred, Tier 2) — target: none. |

## §9.7.17 TensorCore 5th Generation Family Instructions

Allocation, waits, fences and `commit` contribute nothing; the rest is
`UnknownOps` via the `tcgen05` mnemonic. Tier 2. Two family-wide misfits:
**tensor memory** is a fifth memory space the `Space` enum doesn't
have (§B/§C), and most operations are issued by a *single thread* on
behalf of a CTA or CTA pair (§B) — per-thread multiplication would be
wildly wrong.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.17.7.1 | `tcgen05.alloc` | none | OK — allocates tensor-memory columns: a per-CTA resource like `.shared`, so the eventual home is the `.entry` line beside `shared memory`, not an op class. Its 4-byte write of the address to `[dst]` is not counted. |
| 9.7.17.7.1 | `tcgen05.dealloc` | none | OK — moves no bytes. |
| 9.7.17.7.1 | `tcgen05.relinquish_alloc_permit` | none | OK — moves no bytes. |
| 9.7.17.8.3 | `tcgen05.ld` | `UnknownOps` | OK (deferred, Tier 2) — ⚠ misfit §B: tensor-memory→register load, warp-collective; needs the new space and PerWarp scope. |
| 9.7.17.8.4 | `tcgen05.st` | `UnknownOps` | OK (deferred, Tier 2) — same, store side. |
| 9.7.17.8.5 | `tcgen05.wait` | none | OK — moves no bytes. |
| 9.7.17.9.2 | `tcgen05.cp` | `UnknownOps` | OK (deferred, Tier 2) — ⚠ misfit §B+§C: single-thread-issued shared→tensor-memory copy, and the optional decompression modes (4-bit/6-bit → 8-bit) make bytes-read ≠ bytes-written; pick a side per space and document. |
| 9.7.17.9.3 | `tcgen05.shift` | `UnknownOps` | OK (deferred, Tier 2) — tensor-memory row shift; data movement within the new space. |
| 9.7.17.10.9.1 | `tcgen05.mma` | `UnknownOps` | OK (deferred, Tier 2) — ⚠⚠ misfit §B at its most extreme: a single thread issues an MMA executed for a CTA (or CTA *pair* with `cta_group::2`); no current scope expresses "per-CTA-pair, issued once". Also §E (descriptor-sourced smem/tmem operands) and §D (block-scaled `mxf4`/`mxf8f6f4` kinds). The flop math itself is still 2·M·N·K — the model question is *whose* flops. |
| 9.7.17.10.9.2 | `tcgen05.mma.sp` | `UnknownOps` | OK (deferred, Tier 2) — plus the sparse convention (§D). |
| 9.7.17.10.9.3 | `tcgen05.mma.ws` | `UnknownOps` | OK (deferred, Tier 2) — weight-stationary variant; same misfits. |
| 9.7.17.10.9.4 | `tcgen05.mma.ws.sp` | `UnknownOps` | OK (deferred, Tier 2) — same. |
| 9.7.17.11.1 | `tcgen05.fence` | none | OK — moves no bytes. |
| 9.7.17.12.1 | `tcgen05.commit` | none | OK — makes an mbarrier track prior tcgen05 operations; moves no bytes. |

## §9.7.18 Stack Manipulation Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.18.1 | `stacksave` | none | OK — register bookkeeping. |
| 9.7.18.2 | `stackrestore` | none | OK. |
| 9.7.18.3 | `alloca` | none | OK — the local-memory traffic it enables is counted at the `ld.local`/`st.local` that use it. |

## §9.7.19 Video Instructions

All `UnknownOps` today (distinct mnemonics, listed in the unsupported families' arm). Legacy fixed-point
video ops; modern compilers essentially never emit them (`dp4a`/`dp2a`
superseded the common uses). All Tier 4; if one ever appears, each
joins the non-flop arithmetic arm, with no contribution. The parser keeps an operand's byte or half
selector (`%r1.h0`) apart from the register (`Operand::Selected`,
`micro/video_selector.ptx`).

### §9.7.19.1 Scalar Video Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.19.1.1 | `vadd` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.1 | `vsub` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.1 | `vabsdiff` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.1 | `vmin` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.1 | `vmax` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.2 | `vshl` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.2 | `vshr` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.3 | `vmad` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.1.4 | `vset` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |

### §9.7.19.2 SIMD Video Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.19.2.1 | `vadd2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.1 | `vsub2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.1 | `vavrg2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.1 | `vabsdiff2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.1 | `vmin2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.1 | `vmax2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.2 | `vset2` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vadd4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vsub4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vavrg4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vabsdiff4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vmin4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.3 | `vmax4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |
| 9.7.19.2.4 | `vset4` | `UnknownOps` | OK (deferred, Tier 4) — target: none. |

## §9.7.20 Miscellaneous Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.20.1 | `brkpt` | none | OK. |
| 9.7.20.2 | `nanosleep` | none | OK — affects when, never how much (misfit §E). |
| 9.7.20.3 | `pmevent` | none | OK — profiler trigger. |
| 9.7.20.4 | `trap` | none | OK. |
| 9.7.20.5 | `setmaxnreg` | none | OK — register-file rebalancing hint, zero flop/byte. |

## Not in the ISA at all

| Mnemonic | Today | Verdict & recommendation |
|---|---|---|
| `ldg` | (no arm) | Was a dead arm inherited from v1 — `ldg` is not a PTX instruction (the ISA spells it `ld.global.nc`; zero occurrences of `ldg` in the manual). Deleted. |

---

## Instructions that do not fit the model

The audit identified six recurring model questions. §A is now supported
structurally and implemented for atomics; the remaining sections retain
their separate scope and counting decisions. Rows above cite these by letter.

**§A — one instruction, multiple contributions (arithmetic ⊗ memory): supported.**
`classify` returns an instruction's contributions; collection records the
instruction once. `atom` and `red` retain the byte convention while
FP add/min/max also contribute atomic FLOPs. Each contribution is summed
without duplicating the instruction's count.
`tests/instruction_contributions.rs` pins this through collection and the
annotated PTX.

Bulk reduction copies, surface and multimem/fabric reductions can use this
representation but still require their §B/§C/§F decisions. Integer reduction
throughput and collective reduction FLOP conventions (`redux.sync`,
`bar.red`) remain §D questions; multiple contributions alone do not quantify
them. Async completion effects remain outside this work-counting model.

**§B — work not owned by the issuing thread.** Counts are per
thread, and a warp-collective's per-thread share (warp total / 32)
divides exactly for every warp-level shape; the ISA also has per-warpgroup
(`wgmma.*`, 4 warps), single-thread-issues-CTA-scale
(`cp.async.bulk*`, `st.bulk`, `tcgen05.cp`), per-CTA-pair
(`tcgen05.mma` with `cta_group::2`), and writes-to-a-*remote*-CTA
(`st.async.shared::cluster`, `mapa`-derived accesses). Multiplying a
per-thread count by threads-per-CTA overcounts a TMA copy by two to
three orders of magnitude, and a single thread's share of a
CTA-scale copy is not a whole number of bytes — so **a scope axis on
`Measurement` is the structural prerequisite for every Tier 2
family**, and the reason those arms should not be added piecemeal
ahead of it. `cp.async.bulk.tensor` landed without it: its bytes are
unquantified, an op per issuing thread, so there is no byte count to
overcount.

**§C — bytes that are not in the instruction text.**
`cp.async.bulk.tensor` (+`.reduce`/`.prefetch` tensor forms): the byte
count lives in a host-side tensor-map descriptor — *permanently*
unknowable from PTX; `UnquantifiedBytes` is the correct end state
unless a `--bind`-style flag supplies the map, and is what
`cp.async.bulk.tensor` classifies as. Register-sized
`cp.async.bulk`/`st.bulk`: classic `UnquantifiedBytes`. `tex`/`tld4`/
`suld`/`sust`: coordinate- and descriptor-dependent footprints.
Related asymmetries where bytes-in ≠ bytes-out of a single
instruction: `cp.async` reads src-size, or cp-size on the threads whose
ignore-src is false (at most cp-size with a src-size register), from
global and writes cp-size to shared on the threads that execute it;
`tcgen05.cp` with decompression — pick one side per space and
document.

**§D — counting conventions with no single right answer.**
The SFU family (`div`, `rcp`, `sqrt`, `rsqrt`, `sin`, `cos`, `lg2`,
`ex2`, `tanh`): 1 flop per invocation? the SASS multi-instruction
expansion? NCU counts these in a separate pipe entirely. Sparse MMA
(`mma.sp`, `wgmma.mma_async.sp`, `tcgen05.mma.sp`): dense 2·M·N·K vs
effective half. Block-scaled MMA kinds (`mxf4`, `mxf8f6f4`, …):
scale-factor arithmetic on top of 2·M·N·K. Integer dot products and
integer MMA (`dp4a`, `dp2a`, `mma` with s8/u8/b1, `redux.sync`): real
arithmetic with no home in an FP-flops roofline — needs an
integer-ops axis if that audience ever materializes. Already-shipped
precedent: `min`/`max`/`abs`/`neg` = 1 flop (Williams) even though
an FMA/add/multiply-only NCU comparison will not see them. None of these
have a wrong answer; all of them have an *undocumented* answer as the
only failure mode. The existing pattern — a documented policy plus
the existing `pipe` axis — covers all of them.

**§E — the cost is somewhere else.** `call`: the callee's flops and
bytes are not attributed to the call site (recommend a report-level
named unknown for calls to bodies the module doesn't contain).
`wgmma.mma_async`/`tcgen05.mma` descriptor-sourced operands: matrix
data streams from shared/tensor memory with **no load instructions to
count** — instruction-level accounting is structurally blind to it;
the MMA arms must decide whether to emit implicit operand-traffic
bytes. `prefetch`/`applypriority`/`discard`/`griddepcontrol`/
`nanosleep`: affect *when/where*, never *how much* — no contribution is
correct and these fit fine; listed only because the reasoning differs
from "does nothing".

**§F — a different machine model entirely.** `multimem.*` (one
operation touches every GPU in a multicast team), `fabric.*`
(NVLink-fabric transfers to other GPUs' memory), texture filtering
(fixed-function interpolation arithmetic that is neither flops nor
bytes). These aren't missing arms — the roofline the tool computes
has no axis for them, which is why they are Tier 4 / anti-scope, and
why `UnknownOps` (loud) beats no contribution (silent) if one ever appears in
a fixture.

## Summary of warts (things to actually fix)

The audit found no wrong numbers — nothing is counted with incorrect
flops or bytes. In descending order of importance:

1. ~~**`Unparsed` statements are invisible at analyze time.**~~ Fixed:
   each one's line says `not parsed (not counted)`, and the `|`
   destination form that was the verified instance now parses.
2. ~~**Collateral `UnknownOps` from coarse arms**~~ Fixed: integer `div` and `addc`/`subc`/`madc` have arms.
   Originally: **Collateral `UnknownOps` from coarse arms**: integer `div` (caught
   by the SFU reservation); `addc`/`subc`/`madc` (miss the arm their
   `.cc` partners hit). One line each.
3. ~~**Missing one-line arms for mnemonics real producers emit**~~ Fixed but for `tensormap.*` and `clusterlaunchcontrol.*`, still `UnknownOps` in `classify` (406 and 6 statements in the Blackwell corpus).
   Originally: **Missing one-line arms for mnemonics real producers emit**:
   `elect.sync` (also needs the `d|p` parser form), `shf`,
   `setmaxnreg` — all three arrive with any Hopper corpus — plus the
   harmless tail (`cnot`, `fns`, `clmad`, `createpolicy`,
   `nanosleep`, `pmevent`, `mapa`, `getctarank`, `istypep`, stack
   ops).
4. ~~**One dead arm**: `ldg`.~~ Deleted.

The items above are resolved and retained as audit history; the remaining
unhandled families are listed below.

## Priority tiers for unhandled instructions

Priority of *inclusion as classification arms*, not a schedule — the
backlog stays demand-driven, and each tier names its trigger. Ranking
criteria: (a) does it do flops or move bytes (misclassification would
corrupt AI, not just clutter the unknown list); (b) does the target
audience — GEMM / conv / stencil / attention from nvcc, clang,
Triton — actually emit it; (c) how soon.

### Tier 1 — core-audience families — landed (PRs 17–29); historical priorities

Any tensor-core or reduction kernel on sm_80/sm_89 — including
everything Triton emits for these targets — hits these. All carry
flops or bytes, so they dominate the ranked-unknown histogram the
moment such a kernel is analyzed. Fixtures (k11, k12, k14) and
verified grammars already live in the backlog item.

- `mma` (dense) — flops = 2·M·N·K per warp; the single highest-value gap
- `wmma.load` / `wmma.store` / `wmma.mma` — with the role split and the v1 phantom-FLOP regression test
- `ldmatrix`, `stmatrix` — per-warp shared-memory fragment traffic
- `cp.async` — bytes from the explicit size operand; plus `cp.async.commit_group` / `wait_group` / `wait_all` and `cp.async.mbarrier.arrive` with no contribution
- `atom`, `red` — read+write / write-only requested bytes plus FP atomic FLOPs, with multiple contributions (§A)
- SFU family with a documented flop policy (misfit §D): `div`, `rcp`, `sqrt`, `rsqrt`, `sin`, `cos`, `lg2`, `ex2`, `tanh` — softmax/normalization paths in attention kernels

### Tier 2 — the Hopper/Blackwell data path (trigger: first sm_90+ kernel in the corpus)

Same roofline materiality, gated on newer targets — **and on the
scope-axis extension (misfit §B), which should land once, first, not
per-family**.

- `wgmma.mma_async` (+ `.sp`), with `wgmma.fence` / `commit_group` / `wait_group` contributing nothing
- TMA: `cp.async.bulk`, `cp.reduce.async.bulk` (+ `.tensor`), `cp.async.bulk.prefetch` (+ `.tensor`) (`cp.async.bulk.commit_group` / `wait_group` have landed, with no contribution); tensor variants land as honest unquantified bytes (misfit §C), as `cp.async.bulk.tensor` has
- Hopper warp-specialization boilerplate is already classified: `elect.sync` (with the `d|p` operand form in the parser) and `setmaxnreg`, with no contribution
- `red.async`, `movmatrix`, `mma.sp` (sparse policy, misfit §D), `st.async` remote-attribution review
- `tcgen05.*` (the 8 entries but allocation, wait, fence and commit) — tensor memory as a new `Space`, single-thread-issue scope

### Tier 3 — cheap correctness sweep — landed; historical priorities

These arms and the dead-arm cleanup have landed, but for
`tensormap.replace`, `tensormap.cp_fenceproxy` and
`clusterlaunchcontrol.*`, which `classify` still reads as `UnknownOps`.
The list records the original sweep.

- To no contribution: integer `div`, `addc`, `subc`, `madc`, `shf` (likeliest to appear first — nvcc funnel shifts), `cnot`, `fns`, `clmad`, `getctarank`, `istypep`, `mapa`, `stacksave`, `stackrestore`, `alloca`, `createpolicy`, `nanosleep`, `pmevent`, `tensormap.replace` (or a small fixed `Bytes` write — decide when TMA lands), `tensormap.cp_fenceproxy`, `clusterlaunchcontrol.try_cancel` / `.query_cancel`
- Housekeeping: delete the dead `ldg` arm

### Tier 4 — out of audience (trigger: an explicit audience change, i.e. an edit to PLAN.md's scope)

Families whose absence is a scope statement, not a gap. They stay
`UnknownOps` deliberately: several move real bytes (`tex`, `suld`,
`sust`, `multimem`, `fabric`), so no contribution would be wrong, and a real
arm would demand modeling the tool has declared anti-scope (misfit
§F). `UnknownOps` keeps them loud if one ever appears.

- Texture / surface: `tex`, `tld4`, `txq`, `suld`, `sust`, `sured`, `suq`
- Multi-GPU: `multimem.ld_reduce` / `.st` / `.red` / `.st.async` / `.red.async` / `.cp.async.bulk` / `.cp.reduce.async.bulk`, `fabric.try_get` / `.try_put` / `.try_red` / `.try_pullred` / `.submit` / `.wait`
- Legacy video: `vadd` / `vsub` / `vabsdiff` / `vmin` / `vmax` / `vshl` / `vshr` / `vmad` / `vset` and the SIMD `*2` / `*4` variants
