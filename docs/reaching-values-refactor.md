# Reaching values: plan (temporary, for review)

Temporary design note, to be deleted when the refactor lands or is
dropped. Rewritten 2026-10-13 after the register-effects decoder
landed, revised after codex's review of the rewrite, then given MLIR's
value names and one register type for `.reg` registers and CC, then
re-scoped from step 15 to recognize loop counters on SSA values, after
the wrong outputs recorded in `0a40cef` to `db51742`; the first version
and codex's review of it are in git (`b5cdcbf`, `4edba75`).

## Goal

One answer to "which value does a register read see", computed in one
place and shaped like an SSA value, so that

- the tracer, the uniformity analysis, trip counting and the predicate
  reader stop re-deriving it each in their own way;
- the next feature (a value the tracer cannot read but every thread of
  the CTA shares becomes an affine variable, `uniform(%r58@93)`, instead
  of an unknown address) can key that variable by value;
- the SSA addition of `docs/architecture.md` ("migrate the tracer's
  definition lookup first") starts here rather than replacing it.

## Why not a wrapper over `reach_def`

`AffineValueTracer::reach_sites` answers `Def`, `Carried(loop)`,
`Merged(defs)` or `None` by walking the dominator chain and searching
for paths that avoid the dominating definition. codex's review of the
first version found that this cannot be an identity:

- `Merged` keeps definitions a later one kills, and omits a path on
  which the register is not written, so `Merged([d])` reads like a
  plain definition;
- `Carried` names the innermost loop holding the read that also holds
  an interfering definition, not where the values meet;
- a guarded definition is taken to kill the previous value.

Its first finding, that a value identity needs complete write
discovery, is now met: the decoder gives every statement's written
registers and CC.CF, and an unparsed statement writes every register
it names. So the plan builds register SSA instead.

## Registers (`src/ptx/ir.rs`)

What an instruction writes and a later one reads, in one type:

```rust
/// A `.reg` register, or the condition code register CC, whose one bit
/// is CC.CF (PTX ISA §9.7.2) and which no operand names.
pub enum Register { Named(Symbol), Cc }
```

- CC is an implicit operand, as a flags register is in LLVM's
  `MCInstrDesc::implicit_defs` / `implicit_uses`. Which statements
  write and read it is the decoder's `carry` (`register_effects.rs`),
  unchanged:

  | Statement | Writes CC | Reads CC |
  |---|---|---|
  | `add`, `sub`, `mad` | with `.cc` | no |
  | `addc`, `subc`, `madc` | with `.cc` | yes |
  | `call` | yes | no |
  | a form the decoder does not read | yes | yes |
  | an unparsed statement, whatever it names | yes | not asked |
  | anything else | no | no |
- Special registers are not `Register`s. They are read-only (PTX ISA
  §10, "predefined, read-only variables"), so no write ever reaches a
  read of one, and two reads of `%clock` differ where two reads of
  `%tid.x` agree; LLVM and MLIR read both kinds through operations
  (`llvm.nvvm.read.ptx.sreg.*`, `nvvm.read.ptx.sreg.*`), pure ones
  for `%tid`, not for `%clock`. They stay `Operand::Special`.
- A guard stays `Predicate { reg: Symbol }`: it always names a `.pred`
  register.

The decoder's `stmt_written_registers` and `carry` become
`writes(module, stmt) -> Vec<Register>`, whose `Named` entries are
today's list in its order, and `reads_cc(module, instr) -> bool`. The
tracer's `defs` and `carry_defs` become one table,
`writes: HashMap<Register, Vec<usize>>`; a lookup by register symbol
asks for `Named(reg)`, and the induction chains take `Named` writes
only. The uniformity analysis keeps its nodes and the order it queues
them, which decides the reason a divergent node keeps (the first to
reach it, `Uniformity::new`): named-register writes sorted and
deduplicated as `Node::Def`, then CC writes in statement order as
`Node::Carry`. Its `read` and `read_carry` become one `read`, which
takes the node kind and the name in its reason (`%r1`, `CC.CF`) from
the register; a guarded write still reads the old value of what its
node is, its registers for `Node::Def`, CC for `Node::Carry`
(`uniformity.rs`, a guarded `add.cc` keeps `%r4` uniform and its carry
divergent).

## Values (`src/ssa.rs`)

Minimal SSA as in Cytron et al., "Efficiently Computing Static Single
Assignment Form and the Control Dependence Graph" (TOPLAS 1991): a φ
for a register at every block in the iterated dominance frontier of
the blocks writing it. Every register is `Undef` before the entry
block.

```rust
/// A value of a register: what an instruction wrote, what a block
/// merged, or nothing (MLIR's `OpResult` and `BlockArgument`).
pub enum Value {
    /// Statement `op`'s write of `reg`. Where a guarded write's guard is
    /// false, `reg` keeps the value `op` reads from it: a ψ
    /// (Stoutchinin and de Ferrière, MICRO 2001).
    Result { op: usize, reg: Register },
    /// A φ of `reg` at the start of `block`.
    Argument { block: BlockId, reg: Register },
    /// No write of `reg` on some path from the kernel's start.
    Undef(Register),
}

/// Register SSA of one kernel, over its unchanged statements and CFG,
/// as LLVM's `MemorySSA` is over memory.
pub struct RegisterSsa<'a> { /* cfg, doms, writes, φ blocks */ }

impl RegisterSsa<'_> {
    fn new(cfg, doms, writes: HashMap<Register, Vec<usize>>) -> Self;
    /// The value a read of `reg` just before statement `stmt` sees
    /// (MLIR's `getProgramPointBefore`); `stmt` is any statement of a
    /// block, a label included. `None` in an unreachable block.
    fn value_before(&self, stmt: usize, reg: Register) -> Option<Value>;
    /// What `arg` takes on each edge into its block; `None` is the
    /// kernel's start, an edge into the entry block.
    fn incoming(&self, arg: Value) -> Vec<(Option<BlockId>, Value)>;
    /// The `Result`s `v` may be, through `Argument`s, in statement
    /// order, then `Undef` if some path writes nothing; each once.
    fn sources(&self, v: Value) -> Vec<Value>;
}
```

- `value_before(stmt, reg)`: the last write of `reg` in `stmt`'s block
  before `stmt`; else the φ at that block, if one is placed; else the value at
  the end of the immediate dominator; at the entry block, `Undef`. In
  minimal SSA a block without a φ for a register starts with what its
  immediate dominator ends with.
- `incoming`: the same walk from after each reachable predecessor's
  last statement. The entry block has predecessors when a branch
  targets a label at the kernel's start; its φ also takes
  `(None, Undef)`.
- Unreachable blocks: the write table holds their writes, and the
  uniformity analysis and thread participation read in them. They are
  in no frontier and no φ takes a value from them, so they cannot
  reach a value in reachable code; `value_before` there is `None`.
- `sources` walks `incoming` with a visited set. A guarded `Result` is
  not walked through: its old value is `value_before(op, reg)`, which
  the user reads, as `uniformity::def_reads` does today.
- The dominance frontiers come from the loop forest's dominator tree
  (Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm",
  §4). Frontiers and φ placement need no loop structure; users keep
  their own refusal of cycles with two entries.

`RegisterSsa` takes the writes, not the module, so it is independent
of the analyses that consume it (`docs/architecture.md`); it depends on
`ptx` for `Register` and on the dominator tree. The tracer builds it in
`AffineValueTracer::new` from its write table.

## Recurrences (replacing `induction_vars` and `iv_init`)

Today a loop counter is a register with one static write in the loop's
blocks, nested loops' included, that dominates the latches, chained by
hand to the first register written before the loop; its entry value is
traced with the loop's own writes left out, and a read of it, or of any
register on its chain, is replaced by `init + k·step`. PLAN's wrong
outputs for `micro/carried_counter.ptx`, `nested_counter_step.ptx`,
`chain_offset.ptx`, `chain_start.ptx` and `guarded_step.ptx` come from
those rules, not from the representation, which stays: one
`k` per loop, restarting at 1 on each entry, in affine forms whose
start may hold an enclosing loop's `k`.

A counter becomes a header `Argument` that every back edge feeds back
plus a step, as ScalarEvolution recognizes an add-recurrence on a
header PHI (`llvm/lib/Analysis/ScalarEvolution.cpp`,
`createAddRecFromPHI`). Recognition is structural, on SSA values; the
start stays a value until a read interprets it. For loop `l` and an
`Argument` `a` at its header:

- Each back-edge value, `a`'s `incoming` from a predecessor in `l`,
  walks back to `a + s` through `mov`, integer `add` and `sub` in
  `l`'s own blocks, not a nested loop's: unguarded, not `.sat`, not
  floating-point or packed, 32- or 64-bit. `add.cc`'s named
  destination is a sum; `addc` and `subc` read the carry and end the
  walk. `sub` adds `−s` only as `x − s`; `s − x` ends it. A
  non-header `Argument` on the walk ends it. Every back edge gives the
  same `s`, and `s` is not 0; a loop with several latches qualifies,
  though the trip matcher still refuses its count.
- `s` is an immediate, or the value of a register read at the
  instruction that adds it, whose trace has no variable, passes through
  no guarded `Result`, and does not read `l`'s own header `Argument`s:
  a step computed before the loop or recomputed in it from invariants,
  as today.
- The start is the value `a`'s `incoming` from outside `l` agree on,
  the entry edges and `None` for the kernel's start; if they differ,
  no recurrence.

A value is then read as:

- a recurrence `Argument` of `l`: `start + (k_l − 1)·s`, the start
  traced where it is read; a start whose trace passes through a
  guarded `Result`, or holds `Undef`, refuses the read. The start may
  hold an enclosing loop's `k`; the step may not, as the affine form
  has no product of two `k`s;
- a `Result`: its statement's arithmetic, its operands read at it. The
  shortcut that reads a counter at or after its increment as
  `init + k·step` goes: `a + s` is that value;
- any other `Argument`: one immediate shared by its `sources`, none of
  them `Undef`, or refused;
- `Undef`: refused.

The integers are taken not to wrap, as the tracer's module doc says,
until step 27 checks widths.

With `k = 1, 2, …` counted at the latch (`affine.rs`), `start +
(k − 1)·s` and its update `start + k·s` are today's pre- and
post-increment rules and the trip matcher's `D(k)` is unchanged. The
step's lit tests expect:

| Fixture | Trips | Address |
|---|---|---|
| `micro/chain_offset.ptx` | 4 | `param_0 + 4k + 4` |
| `micro/chain_start.ptx` | 4, was −96 | `param_0 + 4k` |
| `micro/guarded_step.ptx` | 4 | unknown, its step guarded |
| `micro/nested_counter_step.ptx` | 8 and 4 | unknown, its step in a nested loop |
| `micro/carried_counter.ptx` | 8 and 4 | unknown, its start the outer header `Argument`, no recurrence until exit values |
| `micro/two_register_counter.ptx` | `param_1` | as now |

Exit values come after. At its loop's exit, after `n` trips, a
recurrence `a` is `start + (n − 1)·s` and `a + s` is `start + n·s`,
under the trip matcher's conditions: one latch, one exit edge, at the
latch. A value is replaced so only on that exit edge, a join that
also takes a path skipping the loop reading its own `incoming`; only
when `n` is a constant of at least 1, so the deferred negative trip
count never enters; and only when the loop's exits are uniform, as
`per_lane` (`report/build.rs`) checks escaped counters today, with
`micro-exit-pred.test` and `micro-exit-pred-stride.test` kept. An
outer counter stepped by an inner loop needs the inner exit relative
to the inner start: the outer back-edge walk meets `start_in + n·s`,
continues to the inner start, the outer header `Argument`, and finds
a recurrence with step `n·s`. Recognition therefore keeps starts as
values, so the inner recurrence is known before the outer one's start
is (`micro/nested_counter_step.ptx` `16·k`,
`micro/carried_counter.ptx` `16·(k_out − 1) + 4·(k_in − 1)`). It adds
no footprint over nested executions: a loop's row counts its own
blocks.

## Answers to `docs/architecture.md`'s five questions

1. Predicated writes: a guarded `Result` is a ψ; its old value is the
   value the statement reads from the register.
2. Loop merges: an `Argument` at the header, whose back-edge values are
   the values at the end of the latches.
3. Undefined inputs: `Undef`, kept in φs and refused where a value is
   interpreted. Unsupported operations: the decoder's fallback writes,
   opaque `Result`s.
4. The CFG is unchanged; values name PTX statements and blocks.
5. Registers and CC only; no memory, no special registers.

## Users

| User | Today | With `RegisterSsa` |
|---|---|---|
| `trace_reg` | `Def` interpreted, a counter at or after its increment `init + k·step`; `Carried(l)`: the counter of `l`, the innermost loop holding the read and an interfering definition, `init + (k−1)·step`; `Merged`: one shared immediate | step 15: `Result` as `Def`, a header `Argument` whose register `ivs` holds as `Carried`, another `Argument` as `Merged` with `Undef` refused; step 17: "Recurrences" above |
| `iv_init` | `trace_reg` at the header, the loop's own definitions excluded: a dominating definition, or a merge of one immediate | keeps `reach_def` through step 16; goes at step 17: a recurrence's start is its outside `incoming` |
| `induction_vars` | the loop's own definitions of a register, chained by hand; the chain's φ register is the first with a definition reaching the header from outside | goes at step 17, for "Recurrences" |
| two-trip predicate loop (`trip_counts.rs`) | the branch predicate one copy in the loop; the copied register one definition in the loop, in the latch; its value on entry one definition; both `mov` of an immediate | same restrictions, the one write in the loop and in the latch still counted in the write table; the copy is `value_before` the branch being a `Result`, the value on entry the header `Argument`'s outside `incoming` |
| `uniformity::read_sites` | `reach_sites`; `Carried` expanded to every definition reaching the read; a missing definition on one path not seen | `sources(v)`'s `Result`s; an `Undef` divergent; gating predicates as today |
| predicate reader | `reach_def` must be `Def`, unguarded, with uniform gates; else `Pred::Uniform` if the uniformity analysis finds it so | `value_before` must be an unguarded `Result`; the rest unchanged |
| `obstacle` | `Def` followed; `Merged`, `Carried`: `obstacle_of_any_def` | `Result` followed; `Argument`: `obstacle_of_any_def` |
| `obstacle_of_any_def` | every definition of the register | unchanged |
| `find_setp`, `predicate_reason` | the last write in the latch | unchanged |

The tracer reads a guarded write as its new value (PLAN, "an address
through one is read as the last write"); that stays, fixed in its own
commit, except in recurrences' starts, steps and their dependencies,
which refuse a guarded write. The
uniformity analysis already reads the old value and the guard.

## Steps, one commit each

Each step diffs every fixture's text and JSON output against the
previous commit and explains each change. A switch that changes an
answer (a killed definition dropped, `Undef` refused, a different
loop's counter) is its own commit with a test pinning the new answer.

1. Dominance frontiers. Done (`33792a0`).
2. Iterated dominance frontiers. Done (`0f19585`).
3. `Values::new`, `at` and `at_end`, under the names `Location` and
   `Site`; tests: straight line, diamond, one-sided write, loop
   counter, nested loops, two-entry cycle, entry with predecessors,
   unreachable block, CC.CF across a call. Done (`f1cec24`).
4. `Location` becomes `Register` in `src/ptx/ir.rs`. Done (`c4542c0`).
5. `Value` and `Site` become one `Value::{Result, Argument, Undef}`.
   Done (`8e25519`).
6. `Values` becomes `RegisterSsa` and `at` `value_before`; `at_end`
   goes, as nothing outside the tests reads it before `incoming`.
   Done (`361e32a`).
7. The decoder's `writes` and `reads_cc`, beside `stmt_written_registers`
   and `carry`. Done (`a375173`).
8. The tracer's one write table, read through `Named` and `Cc` as
   above. Done (`5435374`).
9. The uniformity analysis's one `read`. Done (`9b82e6e`).
10. `stmt_written_registers` goes and `carry` becomes private, the rule
    `writes` and `reads_cc` share; the induction chains, the SSA's test
    helper and the corpus test use `writes`, the SSA's test helper
    gaining the CC write of an unparsed statement it omitted. Done.
11. `incoming` and `sources`; tests through nested φs and a cycle. Done.
12. A temporary corpus snapshot, built from the tracer's write table,
    of every explicit and implicit (CC) read and every header query
    excluding its loop where today's `ReachingDefinition` and the new
    `Value` and `sources` disagree, by class
    (`src/analysis/scalar/snapshots/`). Done. Of 99,578 reads, 2,342
    `Carried` reads are a φ at a join inside the loop, none of them a
    counter of that loop, and 1,128 `Merged` reads keep a definition a
    later write kills; no read's `Merged` misses a path writing
    nothing. Of 29,911 header queries, 8,568 `None`s are a φ of an
    enclosing loop that may be `Undef`; 159 single definitions and 10
    `Merged`s miss a write an enclosing loop carries back in, 6 of
    them counters (PLAN, "A loop counter's value on entry");
    the rest name a φ nearer than `Carried`'s loop or drop a killed
    definition.
13. The tracer holds `RegisterSsa`, and `uniformity::read_sites` reads
    `sources`. Done; no fixture output changed.
14. `uniformity::read_sites` refuses `Undef`. Done; no fixture output
    changed.
15. `trace_reg` reads values: `value_before`, then a `Result` as
    today's `Def`, a header `Argument` whose register `ivs` holds as
    today's `Carried`, another `Argument` as today's `Merged` with
    `Undef` refused, `Undef` as today's `None`. A query excluding a
    loop, `iv_init`'s, keeps `reach_def` until step 17. Killed
    definitions drop out of the shared-immediate test. Done; no fixture
    output changed (`trip_counts.rs` tests pin both).
16. "Recurrences" recognition, a public function of its own module with
    unit tests, not yet called. Done (`src/analysis/scalar/recurrences.rs`).
17. The tracer reads recurrences: `induction_vars`, `ivs`, `iv_init`,
    `exclude` and the post-increment shortcut go, the snapshot's
    counter column reads recurrences, and the lit tests in the table
    above flip. Done. A recurrence whose start or step cannot be read
    reports that read's reason, as before; one whose register step
    varies is no counter, "carried around an enclosing loop". Besides
    the table's five fixtures, one output changed: the B200 fp8 GEMM's
    two scale loads, unknown before and after, now say "div by a
    non-constant divisor" where they said "depends on special register".
18. The two-trip predicate loop reads `RegisterSsa`. Done; no fixture
    output changed.
19. The predicate reader reads `RegisterSsa`. Done; no fixture output
    changed.
20. `obstacle` reads `RegisterSsa`. Done with step 21: the snapshot
    was `reach_def`'s last reader. No fixture output changed.
21. `reach_def` and `reach_sites` go, and the snapshot of step 12 with
    them; the focused tests stay. `path_avoiding` stays: the predicate
    reader asks it which branch selects a block. Done.
22. A `!=` latch whose `D(k) = A1·k + A0` has constant `A1` and `A0`
    has a trip count only when `−A0 / A1` is a positive integer
    (`micro/ne_latch_unreached.ptx`: unknown). A symbolic one keeps
    today's assumption that it divides (k1's and k2's `K − K mod 4`,
    the fp8 GEMMs, `micro/two_register_counter.ptx`), a PLAN entry.
    The lower bound `k ≥ 1` of the other comparisons waits for the
    negative trip count's design (`micro/latch_exits_first.ptx`, on
    the setp branch). Done; only `micro/ne_latch_unreached.ptx`'s output
    changed.
23. A loop footprint's requested bytes count every access whose size
    is known, its address read or not; its unique bytes and sectors are
    unknown when an address is not read
    (`micro/footprint_unknown.ptx`: requested 64 B, unique unknown).
    Done. A loop whose global accesses all have unknown addresses now
    has a row (Gluon's `ce_chunk_kernel.v8192`: requested 32768 B), and
    those of `micro/guarded_step.ptx`, `nested_counter_step.ptx` and
    `carried_counter.ptx` come back with their requested bytes.
24. A loop footprint reads `⌊k/d⌋` and `k mod m` of its own `k` per
    trip: unique-byte intervals, sector patterns and the reuse note
    (`micro/counter_floor_mod.ptx`: 12 B and 8 B). Done; that fixture's
    output alone changed, its reuse note now "varies".
25. Exit values, as in "Recurrences" (`micro/nested_counter_step.ptx`,
    `micro/carried_counter.ptx` addresses exact). Done: a nested
    loop's constant trip count of at least 1 comes from `loop_trips`,
    which the tracer now calls; only those two fixtures' outputs
    changed.
26. `mul.hi` and `mad.hi` are refused (`micro/mul_hi.ptx`: both stores
    unknown). Done; only that fixture's output changed.
27. Register widths: a value read through a 32-bit operation, a
    conversion or a recurrence's update is kept only when its range
    fits the type (`micro/mul_wrap.ptx`: unknown), as the setp
    branch's comparisons check theirs (`ae32551`). Planned in
    "Register widths" below, revised after codex's review; the
    owner decides the launch-bound rule before code.

Steps 4–10 changed no output.

## Register widths (step 27)

An affine form is an integer of no width; a PTX register is 16, 32
or 64 bits and its arithmetic wraps (PTX ISA §9.7.1). The tracer's
module doc assumes values nonnegative and non-overflowing, and nothing
checks it: `micro/mul_wrap.ptx` reads `mul.lo.s32` of 2^30 and 4 as
2^32 where the register holds 0, and a narrowing `cvt` passes its
source through (PLAN, "`cvt` is read as its last operand"). LLVM's
ScalarEvolution keeps every expression's type and proves no-wrap from
the IR's `nsw`/`nuw` flags; PTX carries no such flags, so a range has
to come from the program.

Rule: a 16- or 32-bit integer value whose form has a bounded range
that leaves the range its type reads is refused. A form whose range is
not bounded keeps the documented assumption, a PLAN limitation, as step
22 kept symbolic `!=` distances. 64-bit results are not checked; a
64-bit value read from a narrower one is.

Codex's review of the first version (bce1465) found that architectural
bounds refuse correct outputs: `%ctaid.x` up to 2^31 − 2 puts k2's
`32 * %ctaid.x`, k5's and rope_norm's `64 * %ctaid.x` and the B200 MLA
kernel's `576 * %ctaid.x` past 32 bits, though no launch they take
does. So a variable is bounded only by what the launch is known to be:

- `%tid.x`, `%tid.y`, `%tid.z` below the block shape, from `.reqntid`
  or `--bind`; `%ctaid.*` below `%nctaid.*` only when the grid is
  bound; a counter `k` in [1, n] for a constant trip count `n` of at
  least 1, with both ends, as `k` starts at 1; `⌊v/d⌋` and `v mod m`
  from `v`'s, each term on its own (a correlated `Div` and `Mod` of one
  variable bounded loosely). A symbolic coefficient or base, or an
  unbound variable, leaves the form unbounded. Range arithmetic that
  overflows `i64` refuses the value, not the check.
- The range a type reads: `u` in [0, 2^w), `s` in [−2^(w−1), 2^(w−1));
  a `b` value is a bit pattern, so its range is checked where it is
  read as a number: by a `cvt`'s source type, by `mul.wide`'s and
  `setp`'s type, or by an address's 64-bit add of a 32-bit index.
- Checked operations: `add`, `sub`, `mul.lo`, `mad.lo`, `shl`, `bfe`,
  and every integer `cvt`, both its source and its destination type,
  `.u8` and `.s8` included; a float or packing `cvt` is refused. A
  `mad.lo`'s product is checked before its addend, refusing a wrap the
  addend would cancel. `shl` by a count of at least the width is 0
  (PTX ISA §9.7.8.6). `mul.wide` doubles the width.
- A counter's update: the recognizer collapses its chain into a step,
  so each back edge's chain is checked again with the counter at its
  range's ends. A loop's own trip count is solved without that check
  first; a trip count found while its counter is being checked is not
  used to bound it.

Steps, one commit each:

27a. `range` over affine forms in `affine.rs`, with launch-known bounds
     and the rules above, and unit tests; not yet called. Independent of
     the setp branch, whose comparison check later reads it.
27b. The tracer checks each 16- and 32-bit `add`, `sub`, `mul.lo`,
     `mad.lo`, `shl` and `bfe` result where it interprets the
     instruction (`micro/mul_wrap.ptx`: unknown); every changed fixture
     output explained, none expected without a launch binding.
27c. Integer `cvt` checks both types, a float or packing `cvt` is
     refused (`micro/cvt_reads.ptx`: three unknowns).
27d. Counter updates checked as above.
27e. The setp branch merges: `git merge-tree` finds conflicts in
     PLAN.md and the register-effects corpus snapshot only; its
     comparison range check moves onto 27a's `range`.

## The uniform variable: what its plan must settle

- A value defined in a loop can be uniform in each iteration and
  differ between iterations (`uniformity.rs`,
  `a_value_from_a_loop_is_divergent_after_it_when_its_exit_is`). An
  address over `uniform(v)` is invariant only outside the loops
  holding `v`'s definition, as ScalarEvolution takes a `SCEVUnknown`
  defined inside a loop to vary in it; the loop footprint
  (`report/build.rs`) must not read a missing `Iter` term as stride 0.
- The tracer finds induction variables in `new`; `Uniformity` is
  built from a tracer afterwards. A tracer that falls back to
  `uniform(v)` needs the uniformity answer at that point.
