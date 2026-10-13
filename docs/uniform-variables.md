# Uniform variables: plan (temporary, for review)

Temporary design note, to be deleted when its steps land or it is
dropped. It settles what `docs/reaching-values-refactor.md`'s last
section left open. Revised after codex's reviews of `5edd553`,
`ba152c8` and `5afd97e`.

## Goal

A register read the tracer cannot interpret, whose value every lane of
the CTA executing the read sees alike, becomes an affine variable
`uniform(v)` keyed by its register SSA value `v` (`src/ssa.rs`),
instead of an unknown address. A warp's sectors depend only on an
address's lane-dependent terms and on the residue of the rest modulo
the access's alignment (`src/analysis/memory_footprint.rs`, module
doc), so `u + f(%tid)` with `u` uniform has the sectors of
`param_0 + f(%tid)`.

LLVM's ScalarEvolution has the same object: a `SCEVUnknown` wraps an IR
value it cannot analyze, and is variant in the loops holding its
definition (`llvm/lib/Analysis/ScalarEvolution.cpp:14466–14471`,
`computeLoopDisposition`, llvm-project main @ e32776a1).

## Measurement

A prototype, not committed, on `7db279e`, with the rules of "Design"
below, run over the 55 kernels outside `tests/fixtures/micro/`:

- 4,795 traced address operands: 1,332 read, 3,463 unknown. The
  prototype reads 316 more, 246 global and 70 shared; none of the 1,332
  changes (the forms compared with array names normalized).
- Their reasons before: "div by a non-constant divisor" 222 (Triton's
  grouped tile schedule, `pid / num_pid_in_group` and
  `pid % group_size_m`, e.g. the B200 fp8 GEMM's lines 104–136), "mod of
  an induction variable" 40, `selp` 30, the depth limit 8, "more than
  one reaching definition" 6 (k1's and k2's remainder loops, whose
  counter starts at the main loop's final counter, PLAN "Addresses the
  tracer cannot read"), 10 others. The 6 need the fallback in a
  counter's start; without it the prototype reads 304.
- Of the 246 global: sectors per warp for 49; 197 have a lane
  coefficient that is a parameter and need `--bind`, as other addresses
  do today.
- Still unknown, 3,147: xor swizzles of `%tid` 1,556 (lane-dependent),
  the depth limit 398, loaded from memory 249 (a load is a divergent
  source in `uniformity.rs`), a special register 249, `or` 323, others.
- A uniform variable defined in a loop holding the access: 78 of the
  316. 70 shared (k12's and k14's double-buffer index, mamba2's and
  ssd's `selp`), 8 global (the B200 `bmm_kernel_persistent`'s tile
  indices, recomputed each persistent iteration, 6; a `min` of a
  counter in ssd and mamba2, 2).
- codex's two counterexamples, a uniform value copied in a loop and read
  after an exit that depends on `%tid.x`, and a uniform value copied
  under a `%tid.x` guard, stay unknown; neither shape occurs in the 55.
  Its third, the copy minus lane 0's `shfl.sync.idx` of it, cancels
  before any check; the same cancellation passed a counter
  (`micro/escaped_shfl.ptx`), fixed by step U2.

## Design

### The variable

`Var::Uniform(Value)`; `Value` and `Register` derive `Ord` for the
`BTreeMap` of terms. It is not a lane variable (`depends_on_lane` is
false), `Affine::range` leaves it unbounded, `--bind` does not bind it.
Two reads of one `Value` in one execution of the statements between
them are one number; two different `Value`s are different variables,
with no relation known (`u − u` cancels, `u − w` does not).

The report renders a `Result` as `uniform(%r83@106)`, 106 the PTX line
of the defining statement (`Instr::offset` through the report's
`LineIndex`), and an `Argument` as `uniform(%r34@$L__BB0_5)`, its
block's `cfg.block_name`. One statement writes one value of a
register, so the register and the line name it unless two statements
share the line, which the parser accepts; the name then adds the
statement's index among them, `@106.2`.

### Where it is read

Only in a memory operand's address, and only when the address cannot
be read otherwise: the tracer's `trace_address` traces it as today, and
when that fails, traces it again strictly (`strictly`: a guarded write
is refused, not read as its new value) with the fallback on. An address
read today keeps its form, and a uniform variable never passes a
guarded write between its read and the address.

The fallback sits in `trace_reg`, at a read: the innermost read whose
trace fails and whose value `Uniformity::is_uniform(reg, pos)` holds at
that read becomes the variable, so `4 * uniform(%r70) + %tid.x` keeps
its lane term. `is_uniform` answers for the read: a guarded write is
uniform only when its guard and old value are, a join only when its
selecting branches are, a value defined in a loop and read after it
only when the loop's exits are (`uniformity.rs`, `def_reads`,
`read_sites`), and `Undef` never.

Between the read and the address the value flows through statements
the tracer reads (unguarded, in the strict pass) and φs it reads, which
take one immediate or are recurrences; what remains is leaving a loop
or a cycle. Within one lane, a term that cancels or is multiplied by
zero is gone from that lane's address, so checking the address is
enough, except where the tracer reads another lane's value as this
lane's: `shfl.idx` keeps every term but `%tid.x`. So a term is checked
where it would be uniform across lanes, through `Div` and `Mod`:

- a counter of a loop not holding the point: the loop's exits are
  uniform (`per_lane` at the address, and, since step U2, the
  `shfl.idx` rule at the shuffle);
- `uniform(v)`, at the same two points:
  `Uniformity::value_uniform_at(v, pos)`, `read_sites` over `v`'s
  defining statements (its own, or its φ's sources), every dependency
  it returns uniform: the exits of every loop holding one but not the
  point, γ branches among several sources, and the refusal of a read
  after a cycle with more than one entry (`uniformity.rs`).

Otherwise the address is unknown, "… differs between lanes after its
loop, whose exit …".

The fallback is off in `loop_trips` (trip counts, the nested steps and
exit values built on them, and the loop-keyed `constant_trips` cache)
and in a counter's register steps (a step must be free of variables);
it is on in a counter's start, the value entering the loop, defined
outside it: k1's remainder counter starts at `%r34 + …` and reads
`uniform(%r34@$L__BB0_5)` there. A counter whose start is itself the
value that fails is no counter; the read of its header φ then falls
back, a variable defined in the loop. Thread selectors (`thread_participation.rs`) and
`trip_counts.rs` keep their own tracer and their reasons: a uniform
bound gives neither a count nor a lane set.

### A variable that varies with a loop

`uniform(v)` with `v`'s definition inside loop `L` may differ between
`L`'s iterations. For `L`, and for every loop enclosing `L`'s blocks
that hold the definition, the access's stride note says "varies", as for
a floor or mod of the counter (step 24 of the refactor), through `Div`
and `Mod`; `at_trip` does not read it. For a loop not holding the
definition the variable is invariant.

### Loop footprints

A loop footprint counts every address holding a uniform variable as an
unread address: requested bytes only, unique and sector bytes unknown
(step 23 of the refactor). Taking such addresses into the per-base
intervals needs a rule for bases that may overlap; see "Open".

### Construction order

`Uniformity::new` reads the tracer's write table, SSA, CFG, loops and
reachability, and traces nothing. The tracer gets
`uniformity(&self) -> &Uniformity`, a `OnceCell::get_or_init(||
Uniformity::new(self))`; thread sets, accesses and `per_lane` read it
there instead of building their own.

### Not in this plan

Trip counts and thread selectors in uniform variables; a load from a
uniform address as uniform; lane-dependent swizzles.

## Open: overlapping bases (the owner's decision)

A loop footprint unions intervals per base, the address without its
lane and counter terms, and sums different bases. PLAN states that as
"different pointer parameters are disjoint"; the code also sums one
pointer at different symbolic offsets, `param_5 − 8·param_1` and
`param_5 − 6·param_1` in k1's main loop. A uniform variable can make
two bases coincide (`p + u` and `p + 64·%ctaid.x` when
`u = 64·%ctaid.x`), and a 64-bit one can hide a pointer (`selp.b64`
between two). Measured: with uniform variables defined outside the
loop, 8 loops would read every address (6 with each base on its own
pointer parameter, 2 with one pointer at several bases); most have
symbolic trip counts, so they need `--bind` to report bytes. Options:

1. Keep "Loop footprints" above (no U3).
2. Sum bases on different pointer parameters only when every uniform
   variable in them is at most 32 bits; any two bases on one pointer
   that differ in a uniform term make unique bytes unknown.
3. Treat uniform variables as the code treats parameters today, and
   list it in PLAN.

## Steps, one commit each

Each step diffs every fixture's text and JSON output against the
previous commit and explains each change.

U1. `AffineValueTracer::uniformity`, the lazily built answer; its users
    read it from the tracer. No output changes.
U2. `shfl.idx` keeps a counter of a loop not holding the shuffle only
    when the loop's exits are uniform. Done: only
    `micro/escaped_shfl.ptx`'s output changed, to unknown.
U3. `Var::Uniform` in addresses: `trace_address` and its strict second
    pass, the fallback and where it is off, `value_uniform_at` at the
    address and at `shfl.idx`, the
    rendering, the "varies" note, and loop footprints counting such
    addresses as unread. Expected: the 316 accesses read, no footprint
    changed (their loops' unique bytes were unknown already), and k1's
    remainder loop out of PLAN's "Addresses the tracer cannot read".
    Done: in the 55 kernels the 316 accesses read and nothing else
    changed but three `warp sectors: ?` lines (k1's and k2's remainder
    loops, whose block shape is not bound); `micro/mul_wrap.ptx`,
    `micro/mul_hi.ptx` and `micro/guarded_step.ptx`, unknown before,
    read their constant or guarded values as uniform variables.
    Fixtures `micro/uniform_address.ptx` and `micro/uniform_counter.ptx`:
    - `%ctaid.x / param_1` times 64 plus `%tid.x`: read, sectors per
      warp as for a parameter base;
    - the same `div` of `%tid.x`: unknown, its reason unchanged;
    - a uniform `div` of the counter in a loop: stride "varies", the
      loop's unique bytes unknown; and its floor, `shr` by 3, the same;
    - that value read after a loop whose exit depends on `%tid.x`, and
      copied in the loop and read after it: both unknown;
    - a uniform value copied under a `%tid.x` guard: unknown;
    - the copy minus lane 0's `shfl.sync.idx` of it after that exit:
      unknown;
    - a uniform value copied in a cycle with two entries and read after
      an exit that depends on `%tid.x`: unknown;
    - a counter starting at a uniform `div`: its header φ read as
      `uniform(%r4@$L__BB…)`, stride "varies";
    - a counter starting at `mov start, u` with `u` a uniform `div`:
      `uniform(u)` plus the counter's stride; the same with the loop
      inside an enclosing loop that computes `u` (stride "varies" for
      the enclosing loop); and read after the loop with a constant trip
      count (exit value) and with an exit on `%tid.x` (unknown);
    - a nested loop whose bound is a uniform `div`: its trip count stays
      unknown, so an outer counter it steps is no counter;
    - `u − u` of one uniform value: the address without it; `u − w` of
      two: both terms;
    - two uniform values written on one PTX line: `@106.1`, `@106.2`.

PLAN's "uniform(v)" missing-feature entry points here while the steps
land.
