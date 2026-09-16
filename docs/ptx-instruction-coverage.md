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

## The tool's model (what "fits" means)

The parser accepts instruction-shaped statements as
`mnemonic + modifiers + operands` with no per-mnemonic knowledge
(`src/ptx/parse/parser.rs`). A dotted opcode like `cp.async.bulk.tensor`
becomes mnemonic `cp` with modifiers `async`, `bulk`, `tensor` — so
one `match` arm in `classify.rs` covers every dotted variant of a base
mnemonic; the "Today" column names the arm where routing matters.

An instruction that fails parsing inside a kernel becomes `Stmt::Unparsed`
and never reaches the classifier. Over the committed corpus that is
policed by CI (`tests/parse-allowlist.txt`, empty today); at
`analyze` time each one is counted in `instruction_classes.unparsed`
and listed in the report's unknowns, the same treatment an
unclassified instruction gets. (The ISA's `|`-separated destination
operands — `setp ... p|q`, `match.all.sync d|p`, `elect.sync d|p` —
were the verified unparsed instance until the parser learned them.)

Past the parser, the model is deliberately simple. Each instruction
maps to exactly **one** class:

| Class | Meaning | Roofline contribution |
|---|---|---|
| `Flop { pipe, precision, flops }` | CUDA-core, tensor-core, or SFU work under the documented counting convention | flops, bucketed by pipe and precision |
| `NonFlopArith { Conversion / Integer / Predicate / Move }` | `cvt`; integer/bit ops; compares & selects; `mov`/`cvta` | none (instruction-mix reporting only) |
| `Memory { space, direction, bytes }` | `bytes = None` feeds the unquantified counter, never zero | bytes, bucketed by state space |
| `Copy { from, to, read_bytes, written_bytes }` | one instruction that reads one space and writes another (`cp.async`) — one memory instruction, two byte measurements | bytes on both sides |
| `Sync` | barriers, fences, waits | none |
| `Communication` | shuffles, votes, `match`, `redux`, `elect` — lane-to-lane register exchange | none (counted per iteration: for a warp reduction it *is* the work) |
| `Control` | branches, returns, calls, traps | none |
| `Ignore` | provably zero flop/byte (hints, `nop`) | none, by policy |
| `Unknown` | no arm matched | **counted and named in the report, never dropped** |

Collection emits zero, one, or two
`Measurement { kind, count, predicated, provenance }` records per instruction:
`Ignore` emits none, `Copy` emits separate read and write measurements, and
other classes emit one. See `src/analysis/instruction_counts/collect.rs`
and `measurement.rs`. Every measurement count is per thread per execution: a
warp-collective instruction contributes its warp total divided by the
32 lanes that issue it (PTX ISA §4.5.1: `WARP_SZ` is 32 on every
target to date). Loop-trip and launch multiplication happen at report
time.

So an instruction *fits* the model when it is one thing (arithmetic
**or** memory **or** sync), its work belongs to one thread (or one
warp), and its byte count is either in the instruction text or
honestly unquantifiable. Instructions that break one of those
assumptions are flagged **⚠ model misfit** in the tables and
collected in [Instructions that do not fit the model](#instructions-that-do-not-fit-the-model).

Coverage checks account for classified and unknown instructions and require
every unknown corpus mnemonic to appear in `tests/classify-allowlist.txt`.
That allowlist contains `wgmma`, used by `micro/unknown_in_loop.ptx` to
pin lower-bound reporting. The corpus has expanded beyond the original
nvcc ladder; see PLAN.md for its producers and fixtures.

Row verdict vocabulary: **OK** (correct as-is), **OK (deferred)**
(deliberately `Unknown` pending a future family, with its tier),
**Gap** (should have an arm today; recommendation given), **Wart**
(handled, but with a defect), **⚠ model misfit** (needs a model
decision, not just an arm).

---

## §9.7.1 Integer Arithmetic Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.1.1 | `add` | `NonFlopArith::Integer` (shared arm with FP `add`; FP type modifier absent ⇒ integer) | OK — address math is not flops. |
| 9.7.1.2 | `sub` | `Integer` (same mechanism) | OK. |
| 9.7.1.3 | `mul` | `Integer` (incl. `.hi/.lo/.wide`) | OK. |
| 9.7.1.4 | `mad` | `Integer` | OK — deliberate divergence from v1, which gave integer `mad` 2 flops of precision "Other"; documented and regression-tested. |
| 9.7.1.5 | `clmad` | `Integer` | OK — carry-less multiply-add. |
| 9.7.1.6 | `mul24` | `Integer` | OK. |
| 9.7.1.7 | `mad24` | `Integer` | OK. |
| 9.7.1.8 | `sad` | `Integer` | OK. |
| 9.7.1.9 | `div` | `Integer` (the `div` arm: no FP type modifier ⇒ integer, exactly as `rem`) | OK — the former collateral-Unknown wart is fixed. |
| 9.7.1.10 | `rem` | `Integer` | OK — and the standing counterexample to the `div` wart. |
| 9.7.1.11 | `abs` | `Integer` (via the FP-check arm) | OK. |
| 9.7.1.12 | `neg` | `Integer` (same) | OK. |
| 9.7.1.13 | `min` | `Integer` (same) | OK. |
| 9.7.1.14 | `max` | `Integer` (same) | OK. |
| 9.7.1.15 | `popc` | `Integer` | OK. |
| 9.7.1.16 | `clz` | `Integer` | OK. |
| 9.7.1.17 | `bfind` | `Integer` | OK. |
| 9.7.1.18 | `fns` | `Integer` | OK. |
| 9.7.1.19 | `brev` | `Integer` | OK. |
| 9.7.1.20 | `bfe` | `Integer` | OK. |
| 9.7.1.21 | `bfi` | `Integer` | OK. |
| 9.7.1.22 | `szext` | `Integer` | OK. |
| 9.7.1.23 | `bmsk` | `Integer` | OK. |
| 9.7.1.24 | `dp4a` | `Integer` | ⚠ model misfit (mild) — a 4-way int8 dot product is 8 integer multiply-adds of real arithmetic throughput, but the roofline counts FP flops only. `Integer` is right under the current model; int8 inference workloads would need an integer-ops axis (see misfits §D). |
| 9.7.1.25 | `dp2a` | `Integer` | Same as `dp4a`. |

## §9.7.2 Extended-Precision Integer Arithmetic Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.2.1 | `add.cc` | `Integer` (mnemonic `add` + `cc` modifier — incidental but correct routing) | OK. |
| 9.7.2.2 | `addc` | `Integer` | OK — both halves of a carry chain now classify; the former wart. |
| 9.7.2.3 | `sub.cc` | `Integer` (via `sub`) | OK. |
| 9.7.2.4 | `subc` | `Integer` | OK. |
| 9.7.2.5 | `mad.cc` | `Integer` (via `mad`) | OK. |
| 9.7.2.6 | `madc` | `Integer` | OK. |

## §9.7.3 Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.3.1 | `testp` | `NonFlopArith::Predicate` | OK. |
| 9.7.3.2 | `copysign` | `Flop` ×1 (verified) | OK — sign-bit transfer occupies the FP pipe; 1 flop per the Williams convention. |
| 9.7.3.3 | `add` | `Flop` ×1 × lanes | OK — incl. the packed `f32x2` form (sm_100+) via the lane multiplier. |
| 9.7.3.4 | `sub` | `Flop` ×1 × lanes | OK. |
| 9.7.3.5 | `mul` | `Flop` ×1 × lanes | OK. |
| 9.7.3.6 | `fma` | `Flop` ×2 × lanes | OK — the standard 2-flops-per-FMA convention. |
| 9.7.3.7 | `mad` | `Flop` ×2 | OK — FP `mad` is fused on all modern targets; identical to `fma` for counting. |
| 9.7.3.8 | `div` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy (misfit §D, decided): one flop per result on the `sfu` pipe, whatever the `.approx`/`.full`/`.rnd` form expands to in SASS. |
| 9.7.3.9 | `abs` | `Flop` ×1 | OK — with the cross-check caveat: the planned NCU measured-flops formula (`2·ffma + fmul + fadd`) excludes it, so abs-heavy kernels will show a static-vs-measured gap. Static side is defensible; document when NCU import lands. |
| 9.7.3.10 | `neg` | `Flop` ×1 | OK — same caveat as `abs`. |
| 9.7.3.11 | `min` | `Flop` ×1 | OK — same caveat. |
| 9.7.3.12 | `max` | `Flop` ×1 | OK — same caveat. |
| 9.7.3.13 | `rcp` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.14 | `rcp.approx.ftz.f64` | `Flop { Sfu, f64, 1 }` (same `rcp` arm) | OK. |
| 9.7.3.15 | `sqrt` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.16 | `rsqrt` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.17 | `rsqrt.approx.ftz.f64` | `Flop { Sfu, f64, 1 }` (same `rsqrt` arm) | OK. |
| 9.7.3.18 | `sin` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.19 | `cos` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.20 | `lg2` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.21 | `ex2` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |
| 9.7.3.22 | `tanh` | `Flop { Sfu, type, 1 × lanes }` | OK — SFU policy as `div` (§9.7.3.8). |

## §9.7.4 Half Precision Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.4.1 | `add` | `Flop` ×1 × lanes (`f16x2`/`bf16x2` ⇒ ×2) | OK. |
| 9.7.4.2 | `sub` | `Flop` ×1 × lanes | OK. |
| 9.7.4.3 | `mul` | `Flop` ×1 × lanes | OK. |
| 9.7.4.4 | `fma` | `Flop` ×2 × lanes (verified: `fma.rn.f16x2` = 4 flops) | OK. |
| 9.7.4.5 | `neg` | `Flop` ×1 × lanes | OK (NCU-formula caveat as §9.7.3.10). |
| 9.7.4.6 | `abs` | `Flop` ×1 × lanes | OK (same caveat). |
| 9.7.4.7 | `min` | `Flop` ×1 × lanes | OK (same caveat). |
| 9.7.4.8 | `max` | `Flop` ×1 × lanes | OK (same caveat). |
| 9.7.4.9 | `tanh` | `Flop { Sfu, f16/bf16, lanes }` | OK — `.f16x2`/`.bf16x2` count two results. |
| 9.7.4.10 | `ex2` | `Flop { Sfu, f16/bf16, lanes }` | OK — the fast-softmax core in attention kernels, two results per packed instruction. |

## §9.7.5 Mixed Precision Floating-Point Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.5.1 | `add` (e.g. `add.rn.f32.bf16`) | `Flop` ×1, bucketed **f32** (first FP modifier = destination type; verified) | OK — mixed ops land in the accumulator's precision bucket, which is what roofline tables want. Recommend a comment in `fp_precision_and_lanes` pinning the "PTX writes dst type first" assumption it relies on. |
| 9.7.5.2 | `sub` | Same | OK — same recommendation. |
| 9.7.5.3 | `fma` | `Flop` ×2, bucketed by dst type | OK — same recommendation. |

## §9.7.6 Comparison and Selection Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.6.1 | `set` | `Predicate` | OK — produces a 0/1 value in a data register, but comparisons are not flops under the convention. |
| 9.7.6.2 | `setp` | `Predicate` | OK for the common form. **Wart**: the optional dual-destination form `setp ... p|q, a, b` hits the parser-level `d|p` gap (verified `Unparsed`) and is silently skipped at analyze time. nvcc rarely emits it; fix with the shared `d|p` operand form (Tier 2 rider). |
| 9.7.6.3 | `selp` | `Predicate` | OK. |
| 9.7.6.4 | `slct` | `Predicate` | OK. |

## §9.7.7 Half Precision Comparison Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.7.1 | `set` (f16 variants) | `Predicate` (same arm) | OK. |
| 9.7.7.2 | `setp` (f16 variants) | `Predicate` (same arm) | OK (same `d|p` caveat). |

## §9.7.8 Logic and Shift Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.8.1 | `and` | `Integer`, or `Predicate` when `.pred` | OK — the `.pred` split is correct and tested. |
| 9.7.8.2 | `or` | Same | OK. |
| 9.7.8.3 | `xor` | Same | OK. |
| 9.7.8.4 | `not` | Same | OK. |
| 9.7.8.5 | `cnot` | `Integer` | OK. |
| 9.7.8.6 | `lop3` | `Integer` | OK. |
| 9.7.8.7 | `shf` | `Integer` | OK — funnel shift, nvcc's 64-bit shift lowering. |
| 9.7.8.8 | `shl` | `Integer` | OK. |
| 9.7.8.9 | `shr` | `Integer` | OK. |

## §9.7.9 Data Movement and Conversion Instructions

§9.7.9.1 (Cache Operators) and §9.7.9.2 (Cache Eviction Priority
Hints) define modifiers, not instructions; they are transparent to
requested-bytes accounting by construction (verified:
`ld.global.nc.L2::128B.b128` counts 16 bytes), which is correct — a
cache hint changes where bytes are served from, not how many are
requested.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.9.3 | `mov` | `NonFlopArith::Move` | OK — includes special-register reads (`mov.u32 %r, %tid.x`). |
| 9.7.9.4 | `mov` (vector pack/unpack form) | `Move` (same arm) | OK. |
| 9.7.9.5 | `shfl` (deprecated) | `Communication` | OK — same arm as `shfl.sync`; zero DRAM bytes is the correct accounting for intra-warp exchange. |
| 9.7.9.6 | `shfl.sync` | `Communication` | OK. |
| 9.7.9.7 | `prmt` | `Integer` | OK. |
| 9.7.9.8 | `ld` | `Memory { space, Load, width × v2/v4/v8 }` | OK — space from modifiers with `Generic` as its own honest bucket (`cvta`-provenance refinement is documented anti-scope until a fixture emits generic loads); `.shared::cta` folds into `Shared`, `.shared::cluster` distinct; `volatile`/cache-op/eviction modifiers transparent. |
| 9.7.9.9 | `ld.global.nc` | `Memory { Global, Load, bytes }` (via the `ld` arm; verified) | OK. |
| 9.7.9.10 | `ldu` | `Memory { Global, Load, bytes }` | OK. |
| 9.7.9.11 | `st` | `Memory { space, Store, bytes }` | OK. |
| 9.7.9.12 | `st.async` | `Memory` store via the `st` arm (verified: `st.async.shared::cluster.b32` ⇒ `{SharedCluster, Store, 4}`) | OK for byte accounting. ⚠ model misfit (mild, §B) — the bytes land in a *remote* CTA's shared memory with mbarrier completion; per-thread attribution is still right, but a future cluster-traffic view would need the remote/local distinction the modifier already carries. No action until a cluster fixture exists. |
| 9.7.9.13 | `multimem.st.async` | `Unknown` | OK (deferred, Tier 4) — ⚠ misfit §F: one store fans out to N GPUs' memory. |
| 9.7.9.14 | `st.bulk` | `Memory { Shared, Store, bytes: None }` → unquantified counter (verified) | OK — the size is an operand, not a type modifier, so `UnquantifiedBytes` is the honest result. Recommend quantifying the immediate-operand case when the bulk-copy family lands (Tier 2). |
| 9.7.9.15 | `multimem.ld_reduce` | `Unknown` | OK (deferred, Tier 4) — ⚠ misfit §A+§F: a load *and* a reduction across N GPUs' replicas in one instruction. |
| 9.7.9.15 | `multimem.st` | `Unknown` | OK (deferred, Tier 4) — misfit §F. |
| 9.7.9.15 | `multimem.red` | `Unknown` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.9.16 | `prefetch` | `Ignore` | OK — a prefetch duplicates a later architectural load; counting it would double-count requested bytes. (Genuinely ambiguous under a "DRAM traffic" reading — a prefetch of never-loaded data is real traffic — but the tool's contract is requested bytes, and there `Ignore` is right.) |
| 9.7.9.16 | `prefetchu` | `Ignore` | OK — same. |
| 9.7.9.17 | `applypriority` | `Ignore` | OK — cache-state hint. |
| 9.7.9.18 | `discard` | `Ignore` | OK — invalidates lines without writeback; no data movement. |
| 9.7.9.19 | `createpolicy` | `Ignore` | OK — builds a cache-policy register, touches no memory. |
| 9.7.9.20 | `isspacep` | `Predicate` | OK. |
| 9.7.9.21 | `cvta` | `Move` | OK — address-space cast, not a value conversion; tested. |
| 9.7.9.22 | `cvt` | `Conversion` | OK — and future-proof: every format including FP8/FP6/FP4 (`e4m3`, `e5m2`, `e2m1`, …) lands here with no per-format code. Conversions as their own kind is load-bearing for the S8 precision audit (8 `cvt` per k2 main-loop iteration). |
| 9.7.9.23 | `cvt.pack` | `Conversion` (mnemonic `cvt` + `pack` modifier) | OK. |
| 9.7.9.24 | `mapa` | `Move` | OK — an address computation into a peer CTA's shared memory; the access it feeds is counted where it happens. |
| 9.7.9.25 | `getctarank` | `Integer` | OK. |

### §9.7.9.26 Asynchronous copy

All rows below route through the bare `cp` (or `multimem`) mnemonic
and classify `Unknown` today — deliberately: the async-copy family is
a missing feature in PLAN.md, and `Unknown` is visible and counted. The rows
give the target class each should land as.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.9.26.3.1 | `cp.async` | `Copy { Global → Shared, read = src-size or cp-size, written = cp-size }` — the immediates after the two addresses; a non-immediate `src-size` (a register) falls back to `cp-size`, an upper bound on the read | OK — the `cp` arm, pinned by k12/k14 (nvcc writes `..., 16, 16`) and the Gluon GEMMs (Triton writes `..., 0x10, 0x10`). `collect` records one memory instruction with two byte measurements: a global load and a shared store. |
| 9.7.9.26.3.2 | `cp.async.commit_group` | `Sync` | OK — moves no bytes. |
| 9.7.9.26.3.3 | `cp.async.wait_group` | `Sync` | OK. |
| 9.7.9.26.3.3 | `cp.async.wait_all` | `Sync` | OK. |
| 9.7.9.26.4.1 | `cp.async.bulk` | `Unknown` | OK (deferred, Tier 2) — ⚠ misfit §B+§C: issued by one thread (or one elected thread) for a CTA-scale transfer — per-thread count multiplication would overcount by orders of magnitude; size may be a runtime register (→ `UnquantifiedBytes`). |
| 9.7.9.26.4.2 | `cp.reduce.async.bulk` | `Unknown` | OK (deferred, Tier 2) — ⚠ misfit §A+§B: bulk copy *and* reduction arithmetic in one instruction. |
| 9.7.9.26.4.3 | `cp.async.bulk.prefetch` | `Unknown` | OK (deferred, Tier 2) — target `Ignore`, consistent with the `prefetch` policy. |
| 9.7.9.26.4.4 | `multimem.cp.async.bulk` | `Unknown` | OK (deferred, Tier 4) — misfit §F. |
| 9.7.9.26.4.5 | `multimem.cp.reduce.async.bulk` | `Unknown` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.9.26.5.2 | `cp.async.bulk.tensor` | `Unknown` | OK (deferred, Tier 2) — ⚠⚠ misfit §B+§C: TMA. Bytes live in a host-side tensor-map descriptor, **permanently unknowable from the PTX text**; `UnquantifiedBytes` is the correct end state, not a stopgap (unless a `--bind`-style flag supplies the map). Single-thread-issued, CTA-scale. |
| 9.7.9.26.5.3 | `cp.reduce.async.bulk.tensor` | `Unknown` | OK (deferred, Tier 2) — same, plus misfit §A (reduction). |
| 9.7.9.26.5.4 | `cp.async.bulk.prefetch.tensor` | `Unknown` | OK (deferred, Tier 2) — target `Ignore` per the prefetch policy. |
| 9.7.9.26.6.1 | `cp.async.bulk.commit_group` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.9.26.6.2 | `cp.async.bulk.wait_group` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.9.27 | `tensormap.replace` | `Unknown` | Gap (harmless) — edits a tensor-map descriptor in memory; negligible traffic. Recommend `Ignore` with a comment, or a small fixed `Memory` write — decide when TMA lands (Tier 3). |

## §9.7.10 Fabric Instructions

New in recent ISA revisions (multi-node NVLink fabric). All are
`Unknown` today via the `fabric` mnemonic — correct: cross-GPU
transfers are a different roofline (misfit §F). All Tier 4.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.10.5.1 | `fabric.try_get` | `Unknown` | OK (deferred, Tier 4) — remote read over the fabric. |
| 9.7.10.5.2 | `fabric.try_put` | `Unknown` | OK (deferred, Tier 4) — remote write. |
| 9.7.10.5.3 | `fabric.try_red` | `Unknown` | OK (deferred, Tier 4) — misfit §A+§F (remote reduction). |
| 9.7.10.5.4 | `fabric.try_pullred` | `Unknown` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.10.5.5 | `fabric.submit` | `Unknown` | OK (deferred, Tier 4) — target `Sync` if ever handled. |
| 9.7.10.5.6 | `fabric.wait` | `Unknown` | OK (deferred, Tier 4) — target `Sync` if ever handled. |

## §9.7.11 Texture Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.11.3 | `tex` | `Unknown` (verified) | OK (deferred, Tier 4) — an *improvement* over v1, which `Ignore`d it: a texture fetch moves real bytes, so silently zeroing corrupted AI. ⚠ misfit §F: footprint is coordinate-dependent and the fetch includes fixed-function filtering arithmetic — neither plain bytes nor flops. `Unknown` is the honest state for an out-of-audience family. |
| 9.7.11.4 | `tld4` | `Unknown` | OK (deferred, Tier 4) — same reasoning. |
| 9.7.11.5 | `txq` | `Unknown` | OK (deferred, Tier 4) — a metadata query; target `Ignore` if the family is ever handled. |
| 9.7.11.6 | `istypep` | `Predicate` | OK — a predicate query that lives in the texture chapter. |

## §9.7.12 Surface Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.12.1 | `suld` | `Unknown` | OK (deferred, Tier 4) — moves bytes (v1 wrongly `Ignore`d it); descriptor-held geometry (misfit §C). |
| 9.7.12.2 | `sust` | `Unknown` | OK (deferred, Tier 4) — same. |
| 9.7.12.3 | `sured` | `Unknown` | OK (deferred, Tier 4) — misfit §A (reduction on memory) + §C. |
| 9.7.12.4 | `suq` | `Unknown` | OK (deferred, Tier 4) — query; target `Ignore`. |

## §9.7.13 Control Flow Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.13.1 | `{}` | Parser-level block structure; never reaches the classifier | OK — the right layer. |
| 9.7.13.2 | `@` | Parser-level; sets the `predicated` bit on the guarded instruction's `Measurement`, which propagates as `≤` (at-most) bounds | OK — this is the model's designed answer to divergence: bounds, not probabilities. |
| 9.7.13.3 | `bra` | `Control` | OK. |
| 9.7.13.4 | `brx.idx` | `Control` (mnemonic `brx`) | OK — and CFG-side, unresolved `.branchtargets` surface as a report unknown (commit `d58b305`), so indirect branches degrade honestly. |
| 9.7.13.5 | `call` | `Control` | OK at the classifier layer. ⚠ misfit §E: flops/bytes inside the callee are not attributed to the call site. Fine while the corpus is fully-inlined nvcc output; recommend a report-level named unknown when a kernel calls a function whose body the module doesn't contain (extern), so the gap is visible the day it matters. |
| 9.7.13.6 | `ret` | `Control` | OK. |
| 9.7.13.7 | `exit` | `Control` | OK. |

## §9.7.14 Parallel Synchronization and Communication Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.14.1 | `bar` | `Sync` | OK. Note `bar.red` also computes a predicate reduction across the CTA — a mild §A dual-nature; `Sync` is the right roofline treatment (the reduction is predicate work, never flops). |
| 9.7.14.1 | `barrier` | `Sync` | OK — same family, same note for `barrier.red`. |
| 9.7.14.2 | `bar.warp.sync` | `Sync` (via `bar`) | OK. |
| 9.7.14.3 | `barrier.cluster` | `Sync` (via `barrier`) | OK. |
| 9.7.14.4 | `membar` | `Sync` | OK. |
| 9.7.14.4 | `fence` | `Sync` | OK — all variants (`fence.proxy.*`, `fence.mbarrier_init`, …) via modifier transparency. |
| 9.7.14.5 | `atom` | `Copy { space → space, width, width }` — one read and one write of the operand width in the instruction's state space, `Generic` without one | OK — misfit §A resolved by policy (2): bytes both ways, no flops; the `Copy` class the async-copy work introduced makes the two sides explicit. Contention/serialization stays NCU's side (anti-scope). |
| 9.7.14.6 | `red` | `Memory { space, Store, width }` | OK — write side only; the read is implicit (v1's policy, kept). |
| 9.7.14.7 | `red.async` | `Unknown` (via `red`) | OK (deferred, Tier 2) — misfit §A + async completion; cluster reductions. |
| 9.7.14.8 | `multimem.red.async` | `Unknown` | OK (deferred, Tier 4) — misfit §A+§F. |
| 9.7.14.9 | `vote` (deprecated) | `Communication` | OK. |
| 9.7.14.10 | `vote.sync` | `Communication` | OK. |
| 9.7.14.11 | `match.sync` | `Communication` | OK for the plain form (verified). **Wart**: `match.all.sync` with the optional `d|p` destination hits the parser gap (verified `Unparsed`) — fix with the shared `d|p` operand form (Tier 2 rider). |
| 9.7.14.12 | `activemask` | `Communication` | OK. |
| 9.7.14.13 | `redux.sync` | `Communication` | OK, with a flag: ⚠ misfit §A (mild) — it also performs the reduction arithmetic (integer; `f32` min/max/add on the newest targets). Counted as communication, not as flops: an integer reduction has no roofline home and the f32 form has no fixture. |
| 9.7.14.14 | `griddepcontrol` | `Ignore` | OK — defensible; arguably `Sync` (it orders dependent grids), but nothing downstream distinguishes them. Cosmetic; keep. |
| 9.7.14.15 | `elect.sync` | `Communication` (the `d|p` destination parses since the `|` operand change) | OK — warp-collective leader election. |
| 9.7.14.16.12 | `mbarrier.init` | `Sync` (via `mbarrier`) | OK — technically writes 8 bytes of shared memory to initialize the object; negligible by construction, `Sync` is right. |
| 9.7.14.16.13 | `mbarrier.inval` | `Sync` | OK. |
| 9.7.14.16.14 | `mbarrier.expect_tx` | `Sync` | OK. |
| 9.7.14.16.15 | `mbarrier.complete_tx` | `Sync` | OK. |
| 9.7.14.16.16 | `mbarrier.arrive` | `Sync` | OK. |
| 9.7.14.16.17 | `mbarrier.arrive_drop` | `Sync` | OK. |
| 9.7.14.16.18 | `cp.async.mbarrier.arrive` | `Sync` (the `cp` arm's `mbarrier` modifier) | OK. |
| 9.7.14.16.19 | `mbarrier.test_wait` | `Sync` | OK. |
| 9.7.14.16.19 | `mbarrier.try_wait` | `Sync` | OK. |
| 9.7.14.16.20 | `mbarrier.pending_count` | `Sync` | OK. |
| 9.7.14.16.21 | `mbarrier.check_layout` | `Sync` (via `mbarrier` — future variants land free) | OK. |
| 9.7.14.17 | `tensormap.cp_fenceproxy` | `Unknown` | Gap (harmless) — target `Sync` (Tier 3). |
| 9.7.14.18 | `clusterlaunchcontrol.try_cancel` | `Unknown` | OK (deferred, Tier 3) — persistent-kernel launch control; target `Sync`. Note: kernels built around it re-derive their work assignment in a loop the trip-count model will honestly fail on — an audience question before a classifier one. |
| 9.7.14.19 | `clusterlaunchcontrol.query_cancel` | `Unknown` | OK (deferred, Tier 3) — target `Integer`/`Move` (unpacks a result handle). |

## §9.7.15 Warp Level Matrix Multiply-Accumulate Instructions

§9.7.15.1–.3 (shapes, data types, block scaling) are context sections.
All entries are warp-collective: their per-thread count is the warp
total over 32 (`WARP_LANES` in `classify.rs`), which divides exactly
for every shape × element type the ISA lists.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.15.4.3 | `wmma.load` | `Memory { space, Load, bytes }` — matrix `.a` = M×K, `.b` = K×N, `.c` = M×N elements × element bits / 8 / 32; space from the `.global`/`.shared` modifier, `Generic` without one | OK — role split on the first modifier (`wmma` arm). v1's worst bug — bare `wmma` matched as an MMA, 8192 phantom flops per load — is pinned by the `wmma_loads_and_stores_are_fragment_bytes` unit test: a load is `Memory`, never `Flop`. |
| 9.7.15.4.4 | `wmma.store` | `Memory { space, Store, bytes }`, `.d` = M×N elements | OK. |
| 9.7.15.4.5 | `wmma.mma` | `Flop { Tensor, atype, 2·M·N·K / 32 }`; atype is `.f16` when only `.dtype.ctype` are given (§9.7.15.4.5) | OK for the FP kinds (f16, bf16, tf32, f64). Integer (`s8`/`u8`/`s4`/`u4`) and single-bit (`b1`) kinds stay `Unknown` deliberately: misfit §D, not flops. |
| 9.7.15.5.14 | `mma` | `Flop { Tensor, atype, ops·2·M·N·K / 32 }` — atype is the second type modifier (`.dtype.atype.btype.ctype`); ops = 4 for `.m8n8k4` with `.f16` (§9.7.15.5.1), else 1 | OK for the FP kinds (fp8 = `e4m3`/`e5m2`, f16, bf16, tf32, f64) — the `mma` arm, pinned by the mma_demo, k14 and Gluon fp8 GEMM fixtures; the fp8 count equals Nsight Compute's `sm__ops_path_tensor_src_fp8` on the c_fc launch (PLAN.md, hardware cross-check). Integer (`s8`/`u8`/`s4`/`u4`), single-bit (`b1`), fp6/fp4 (`e2m1`…) and `.block_scale` kinds stay `Unknown` deliberately (misfit §D: not FP flops, or scale-factor arithmetic beyond 2·M·N·K). |
| 9.7.15.5.15 | `ldmatrix` | `Memory { Shared, Load, num × rows × cols × bits / 8 / 32 }` — shape from `.m8n8`/`.m16n16`/`.m8n16`, count from `.x1/.x2/.x4`, width from `.b16`/`.b8`; the padded `.b6x16_p32`/`.b4x16_p64` source formats are unquantified bytes | OK — the `matrix_fragments` arm, pinned by k14. Shared always: the ISA defines generic addressing here as pointing into `.shared`. |
| 9.7.15.5.16 | `stmatrix` | `Memory { Shared, Store, … }`, same sizing | OK — same arm. |
| 9.7.15.5.17 | `movmatrix` | `Unknown` | OK (deferred, Tier 2) — register-file transpose, zero bytes; target `Sync` (warp-collective register exchange) or a PerWarp `Move`; either is defensible, document the choice. |
| 9.7.15.6.3 | `mma.sp` / `mma.sp::ordered_metadata` | `Unknown` (the `mma` arm returns early on `.sp*`) | OK (deferred, Tier 2) — ⚠ misfit §D: 2:4-sparse flop counting is genuinely ambiguous (dense 2·M·N·K, matching NCU's convention, vs effective half). Decide with a fixture and NCU cross-check, and label the choice in the report. |

## §9.7.16 Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions

All `Unknown` today via the `wgmma` mnemonic. Tier 2 (Hopper: Triton
and CUTLASS emit these on H100).

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.16.5.2 | `wgmma.mma_async` | `Unknown` | OK (deferred, Tier 2) — ⚠⚠ misfit §B: executed by a *warpgroup* (4 warps); `Scope` has no such variant — the enum must grow before this lands. Also misfit §E: the A/B operands can stream directly from shared memory via descriptors, so its smem traffic has no `ld` instructions to count — decide whether the arm emits implicit smem bytes. Flops = 2·M·N·K per warpgroup. |
| 9.7.16.6.3 | `wgmma.mma_async.sp` | `Unknown` | OK (deferred, Tier 2) — same, plus the sparse convention (misfit §D). |
| 9.7.16.7.1 | `wgmma.fence` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.16.7.2 | `wgmma.commit_group` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.16.7.3 | `wgmma.wait_group` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |

## §9.7.17 TensorCore 5th Generation Family Instructions

All `Unknown` today via the `tcgen05` mnemonic. Tier 2. Two
family-wide misfits:
**tensor memory** is a fifth memory space the `Space` enum doesn't
have (§B/§C), and most operations are issued by a *single thread* on
behalf of a CTA or CTA pair (§B) — per-thread multiplication would be
wildly wrong.

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.17.7.1 | `tcgen05.alloc` | `Unknown` | OK (deferred, Tier 2) — allocates tensor-memory columns: a per-CTA resource like `.shared`, so the eventual home is the resource report (alongside the `shared memory [static]` line), not an op class. Target `Sync` for the instruction itself. |
| 9.7.17.7.1 | `tcgen05.dealloc` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.17.7.1 | `tcgen05.relinquish_alloc_permit` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.17.8.3 | `tcgen05.ld` | `Unknown` | OK (deferred, Tier 2) — ⚠ misfit §B: tensor-memory→register load, warp-collective; needs the new space and PerWarp scope. |
| 9.7.17.8.4 | `tcgen05.st` | `Unknown` | OK (deferred, Tier 2) — same, store side. |
| 9.7.17.8.5 | `tcgen05.wait` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.17.9.2 | `tcgen05.cp` | `Unknown` | OK (deferred, Tier 2) — ⚠ misfit §B+§C: single-thread-issued shared→tensor-memory copy, and the optional decompression modes (4-bit/6-bit → 8-bit) make bytes-read ≠ bytes-written; pick a side per space and document. |
| 9.7.17.9.3 | `tcgen05.shift` | `Unknown` | OK (deferred, Tier 2) — tensor-memory row shift; data movement within the new space. |
| 9.7.17.10.9.1 | `tcgen05.mma` | `Unknown` | OK (deferred, Tier 2) — ⚠⚠ misfit §B at its most extreme: a single thread issues an MMA executed for a CTA (or CTA *pair* with `cta_group::2`); no current scope expresses "per-CTA-pair, issued once". Also §E (descriptor-sourced smem/tmem operands) and §D (block-scaled `mxf4`/`mxf8f6f4` kinds). The flop math itself is still 2·M·N·K — the model question is *whose* flops. |
| 9.7.17.10.9.2 | `tcgen05.mma.sp` | `Unknown` | OK (deferred, Tier 2) — plus the sparse convention (§D). |
| 9.7.17.10.9.3 | `tcgen05.mma.ws` | `Unknown` | OK (deferred, Tier 2) — weight-stationary variant; same misfits. |
| 9.7.17.10.9.4 | `tcgen05.mma.ws.sp` | `Unknown` | OK (deferred, Tier 2) — same. |
| 9.7.17.11.1 | `tcgen05.fence` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |
| 9.7.17.12.1 | `tcgen05.commit` | `Unknown` | OK (deferred, Tier 2) — target `Sync`. |

## §9.7.18 Stack Manipulation Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.18.1 | `stacksave` | `Move` | OK — register bookkeeping. |
| 9.7.18.2 | `stackrestore` | `Move` | OK. |
| 9.7.18.3 | `alloca` | `Move` | OK — the local-memory traffic it enables is counted at the `ld.local`/`st.local` that use it. |

## §9.7.19 Video Instructions

All `Unknown` today (distinct mnemonics, no arms). Legacy fixed-point
video ops; modern compilers essentially never emit them (`dp4a`/`dp2a`
superseded the common uses). All Tier 4; if one ever appears, each is
a one-line `Integer` arm (`vset*` → `Predicate`-flavored `Integer`,
they produce 0/1 results).

### §9.7.19.1 Scalar Video Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.19.1.1 | `vadd` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.1 | `vsub` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.1 | `vabsdiff` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.1 | `vmin` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.1 | `vmax` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.2 | `vshl` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.2 | `vshr` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.3 | `vmad` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.1.4 | `vset` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |

### §9.7.19.2 SIMD Video Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.19.2.1 | `vadd2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.1 | `vsub2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.1 | `vavrg2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.1 | `vabsdiff2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.1 | `vmin2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.1 | `vmax2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.2 | `vset2` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vadd4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vsub4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vavrg4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vabsdiff4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vmin4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.3 | `vmax4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |
| 9.7.19.2.4 | `vset4` | `Unknown` | OK (deferred, Tier 4) — target `Integer`. |

## §9.7.20 Miscellaneous Instructions

| § | Instruction | Today | Verdict & recommendation |
|---|---|---|---|
| 9.7.20.1 | `brkpt` | `Control` | OK. |
| 9.7.20.2 | `nanosleep` | `Ignore` | OK — affects when, never how much (misfit §E). |
| 9.7.20.3 | `pmevent` | `Ignore` | OK — profiler trigger. |
| 9.7.20.4 | `trap` | `Control` | OK. |
| 9.7.20.5 | `setmaxnreg` | `Ignore` | OK — register-file rebalancing hint, zero flop/byte. |

## Not in the ISA at all

| Mnemonic | Today | Verdict & recommendation |
|---|---|---|
| `ldg` | (no arm) | Was a dead arm inherited from v1 — `ldg` is not a PTX instruction (the ISA spells it `ld.global.nc`; zero occurrences of `ldg` in the manual). Deleted. |

---

## Instructions that do not fit the model

The classes above assume an instruction is one thing, owned by one
thread (or warp), with statically visible bytes. Six recurring ways
the ISA breaks those assumptions — each is a *decision to make*, not
just an arm to add. Rows above cite these by letter.

**§A — one instruction, two natures (arithmetic ⊗ memory).**
`atom`, `red`, `red.async`, `cp.reduce.async.bulk(.tensor)`, `sured`,
`multimem.ld_reduce`/`.red`/`.red.async`, `fabric.try_red`/
`.try_pullred`; mildly `redux.sync` and `bar.red`. A read-modify-write
with arithmetic combines memory traffic and computation. The current
`Copy` class already emits read and write measurements sharing provenance:
`atom` counts both byte directions, while `red` counts store bytes. Neither
counts arithmetic FLOPs. Counting both arithmetic and memory work would need
an expanded class or collection policy; multiple measurements per instruction
are already supported.

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
ahead of it.

**§C — bytes that are not in the instruction text.**
`cp.async.bulk.tensor` (+`.reduce`/`.prefetch` tensor forms): the byte
count lives in a host-side tensor-map descriptor — *permanently*
unknowable from PTX; `UnquantifiedBytes` is the correct end state
unless a `--bind`-style flag supplies the map. Register-sized
`cp.async.bulk`/`st.bulk`: classic `UnquantifiedBytes`. `tex`/`tld4`/
`suld`/`sust`: coordinate- and descriptor-dependent footprints.
Related asymmetries where bytes-in ≠ bytes-out of a single
instruction: `cp.async` with `src-size < cp-size` (zero-fill) and
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
`nanosleep`: affect *when/where*, never *how much* — `Ignore` is
correct and these fit fine; listed only because the reasoning differs
from "does nothing".

**§F — a different machine model entirely.** `multimem.*` (one
operation touches every GPU in a multicast team), `fabric.*`
(NVLink-fabric transfers to other GPUs' memory), texture filtering
(fixed-function interpolation arithmetic that is neither flops nor
bytes). These aren't missing arms — the roofline the tool computes
has no axis for them, which is why they are Tier 4 / anti-scope, and
why `Unknown` (loud) beats `Ignore` (silent) if one ever appears in
a fixture.

## Summary of warts (things to actually fix)

The audit found no wrong numbers — nothing is counted with incorrect
flops or bytes. In descending order of importance:

1. ~~**`Unparsed` statements are invisible at analyze time.**~~ Fixed:
   the report counts them (`instruction_classes.unparsed`) and lists
   them under unknowns, and the `|` destination form that was the
   verified instance now parses.
2. ~~**Collateral `Unknown`s from coarse arms**~~ Fixed: integer `div` and `addc`/`subc`/`madc` have arms.
   Originally: **Collateral `Unknown`s from coarse arms**: integer `div` (caught
   by the SFU reservation); `addc`/`subc`/`madc` (miss the arm their
   `.cc` partners hit). One line each.
3. ~~**Missing one-line arms for mnemonics real producers emit**~~ Fixed (all of the Tier 3 list below has arms).
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
- `cp.async` — bytes from the explicit size operand; plus `cp.async.commit_group` / `wait_group` / `wait_all` and `cp.async.mbarrier.arrive` as `Sync`
- `atom`, `red` — v1 byte policy (read+write / write-only), documented (misfit §A)
- SFU family with a documented flop policy (misfit §D): `div`, `rcp`, `sqrt`, `rsqrt`, `sin`, `cos`, `lg2`, `ex2`, `tanh` — softmax/normalization paths in attention kernels

### Tier 2 — the Hopper/Blackwell data path (trigger: first sm_90+ kernel in the corpus)

Same roofline materiality, gated on newer targets — **and on the
scope-axis extension (misfit §B), which should land once, first, not
per-family**.

- `wgmma.mma_async` (+ `.sp`), with `wgmma.fence` / `commit_group` / `wait_group` as `Sync`
- TMA: `cp.async.bulk`, `cp.async.bulk.tensor`, `cp.reduce.async.bulk` (+ `.tensor`), `cp.async.bulk.prefetch` (+ `.tensor`), with `cp.async.bulk.commit_group` / `wait_group` as `Sync`; tensor variants land as honest unquantified bytes (misfit §C)
- Hopper warp-specialization boilerplate is already classified: `elect.sync` (`Communication`, with the `d|p` operand form in the parser), `setmaxnreg` (`Ignore`)
- `red.async`, `movmatrix`, `mma.sp` (sparse policy, misfit §D), `st.async` remote-attribution review
- `tcgen05.*` (all 14 entries) — tensor memory as a new `Space`, single-thread-issue scope

### Tier 3 — cheap correctness sweep — landed; historical priorities

These arms and the dead-arm cleanup have landed. The list records the
original sweep rather than outstanding work.

- To `Integer`: integer `div`, `addc`, `subc`, `madc`, `shf` (likeliest to appear first — nvcc funnel shifts), `cnot`, `fns`, `clmad`, `getctarank`
- To `Predicate`: `istypep`
- To `Move`: `mapa`, `stacksave`, `stackrestore`, `alloca`
- To `Ignore`: `createpolicy`, `nanosleep`, `pmevent`, `tensormap.replace` (or a small fixed `Memory` write — decide when TMA lands)
- To `Sync`: `tensormap.cp_fenceproxy`, `clusterlaunchcontrol.try_cancel` / `.query_cancel`
- Housekeeping: delete the dead `ldg` arm

### Tier 4 — out of audience (trigger: an explicit audience change, i.e. an edit to PLAN.md's scope)

Families whose absence is a scope statement, not a gap. They stay
`Unknown` deliberately: several move real bytes (`tex`, `suld`,
`sust`, `multimem`, `fabric`), so `Ignore` would be wrong, and a real
arm would demand modeling the tool has declared anti-scope (misfit
§F). `Unknown` keeps them loud if one ever appears.

- Texture / surface: `tex`, `tld4`, `txq`, `suld`, `sust`, `sured`, `suq`
- Multi-GPU: `multimem.ld_reduce` / `.st` / `.red` / `.st.async` / `.red.async` / `.cp.async.bulk` / `.cp.reduce.async.bulk`, `fabric.try_get` / `.try_put` / `.try_red` / `.try_pullred` / `.submit` / `.wait`
- Legacy video: `vadd` / `vsub` / `vabsdiff` / `vmin` / `vmax` / `vshl` / `vshr` / `vmad` / `vset` and the SIMD `*2` / `*4` variants
