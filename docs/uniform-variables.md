# Uniform variables: plan (temporary, for review)

Temporary design note, to be deleted when its steps land or it is
dropped. It settles what `docs/reaching-values-refactor.md`'s last
section left open.

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

A prototype, not committed, on `7db279e`: `Var::Uniform`, and in
`trace_reg`, when a read's trace fails and
`Uniformity::is_uniform(reg, pos)` holds, the variable of the value read;
on only while a memory operand's address is traced. Run over the 55
kernels outside `tests/fixtures/micro/`:

- 4,795 traced address operands: 1,332 read, 3,463 unknown. The
  prototype reads 316 more, 246 global and 70 shared; none of the 1,332
  changes (the forms compared with array names normalized).
- Their reasons before: "div by a non-constant divisor" 222 (Triton's
  grouped tile schedule, `pid / num_pid_in_group` and
  `pid % group_size_m`, e.g. the B200 fp8 GEMM's lines 104–136), "mod of
  an induction variable" 40, `selp` 30, the depth limit 8, "more than
  one reaching definition" 6 (k1's and k2's remainder loops reading the
  main loop's final counter, PLAN "Addresses the tracer cannot read"),
  10 others.
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
- One base with different uniform variables in one loop: once, rope_kv
  deepseek-v3's loop 0, `uniform(%r9)` and `uniform(%r1)` on
  `param_0 + 54·param_5 − 2·param_4`.
- Loop footprints: of 58 loops with global accesses, 15 read every
  address today; with uniform variables defined outside the loop, 8
  more do (mamba2 ×3, the sm_89 fp8 GEMM ×2, k1's two remainder loops
  and k2's).

## Design

### The variable

`Var::Uniform(Value)`; `Value` and `Register` derive `Ord` for the
`BTreeMap` of terms. It is not a lane variable (`depends_on_lane` is
false), `Affine::range` leaves it unbounded, `--bind` does not bind it,
and `per_lane` ignores it. The report renders a `Result` as
`uniform(%r83@106)`, 106 the defining statement's line in the PTX file,
and an `Argument` as `uniform(%r34@$L__BB0_5)`, its block.

Two reads of one `Value` are one variable: in one execution of the
statements between them they are the same number. Two different
`Value`s are different variables, with no relation known.

### Where it is read

Only in a memory operand's address. The tracer gains
`trace_address(op, pos, scope)`, which turns the fallback on for the
trace, as `strictly` turns on the refusal of guarded writes;
`loop_trips` turns it off, so a trip count, and with it a nested step
or an exit value, never holds a uniform variable. Thread selectors
(`thread_participation.rs`) and `trip_counts.rs` keep their tracer and
their reasons: a uniform bound gives neither a count nor a lane set.

The fallback sits in `trace_reg`, at a read: the innermost read whose
trace fails and whose value `is_uniform` at that read becomes the
variable, so `4 * uniform(%r70) + %tid.x` keeps its lane term.
`is_uniform` already answers for the read, not the definition: a value
defined in a loop and read after it is uniform only when the loop's
exits are (`uniformity.rs`, `read_sites`), a guarded write only when its
guard and old value are, and `Undef` never.

### A variable that varies with a loop

`uniform(v)` with `v`'s statement (or, for an `Argument`, its block)
inside loop `L` may differ between `L`'s iterations. So for `L`:

- the access's stride note for `L` says "varies", as for a floor or
  mod of `L`'s counter (step 24 of the refactor);
- `L`'s footprint counts the access as an unread address: requested
  bytes only, unique and sector bytes unknown (step 23).

For a loop not holding `v`'s definition the variable is invariant.

### Overlapping bases

A loop footprint unions intervals per base, the address without its
lane and counter terms, and takes different bases to be disjoint
(PLAN, "Unique bytes per loop assume different pointer parameters are
disjoint"). Two bases equal but for their uniform terms may overlap or
coincide, so the loop's unique bytes are unknown then.

### Construction order

`report/build.rs` builds the tracer, then `Uniformity::new(&tracer)`,
which reads the tracer's write table, SSA, CFG and loops but traces
nothing. The tracer holds a `OnceCell<Uniformity>`, set right after:
`tracer.set_uniformity(Uniformity::new(&tracer))`; thread sets,
accesses and `per_lane` read it from the tracer. Moving the facts both
read (module, kernel, CFG, loops, writes, SSA, reachability) into a
struct of their own would avoid the cell; it touches every user of
those fields, so it is left out.

### Not in this plan

Trip counts and thread selectors in uniform variables; a load from a
uniform address as uniform; lane-dependent swizzles.

## Steps, one commit each

Each step diffs every fixture's text and JSON output against the
previous commit and explains each change.

U1. The tracer holds the uniformity answer (`OnceCell`); no output
    changes.
U2. `Var::Uniform` in addresses, as above: `trace_address`, the
    fallback, `loop_trips` turning it off, the rendering, the "varies"
    note, and a loop footprint counting any address with a uniform
    variable as unread. Expected: the 316 accesses read, no footprint
    changed (their loops' unique bytes were unknown already). Fixture
    `micro/uniform_address.ptx`, ptxas's SASS for each kernel:
    - a CTA-uniform `div` by a parameter, `%ctaid.x / param_1`, times
      a stride plus `%tid.x`: read, sectors per warp as ptxas;
    - the same `div` of `%tid.x`: unknown, its reason unchanged;
    - a uniform value defined in a loop and used in it: stride
      "varies", unique unknown;
    - the same value read after a loop whose exit depends on `%tid.x`:
      unknown.
U3. A loop footprint reads a uniform variable defined outside the loop
    as part of the base, and its unique bytes are unknown when two
    bases differ only in uniform terms. Expected: the loops above that
    have numeric trip counts and bound shapes gain footprints; rope_kv
    deepseek-v3's loop 0 stays unknown. Fixture
    `micro/uniform_bases.ptx`: two accesses on `param_0 + 64·uniform(a)`
    and `param_0 + 64·uniform(b)` in one loop.

PLAN's "uniform(v)" missing-feature entry points here while the steps
land, and goes with U3; U2 takes k1's remainder loop out of "Addresses
the tracer cannot read".
