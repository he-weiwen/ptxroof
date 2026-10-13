# Reaching values: plan (temporary, for review)

Temporary design note, to be deleted when the refactor lands or is
dropped. Rewritten 2026-10-13 after the register-effects decoder
landed, revised after codex's review of the rewrite, then given MLIR's
value names and one register type for `.reg` registers and CC; the
first version and codex's review of it are in git (`b5cdcbf`,
`4edba75`).

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
| `trace_reg` | `Def` interpreted, a counter at or after its increment `init + k·step`; `Carried(l)`: the counter of `l`, the innermost loop holding the read and an interfering definition, `init + (k−1)·step`; `Merged`: one shared immediate | `Result`: same; `Argument` at the header of loop `l`: `l`'s counter rule; other `Argument`: one shared immediate, refused when a source is `Undef` |
| `iv_init` | `trace_reg` at the header, the loop's own definitions excluded: a dominating definition, or a merge of one immediate | the header `Argument`'s `incoming` from outside the loop, traced as one φ; with no `Argument` at the header, the value at its immediate dominator's end |
| `induction_vars` | the loop's own definitions of a register, chained by hand; the chain's φ register is the first with a definition reaching the header from outside | same chain and restrictions; the φ register is the first whose header `Argument` has an outside `incoming` with a `Result` among its `sources` |
| two-trip predicate loop (`trip_counts.rs`) | the branch predicate one copy in the loop; the copied register one definition in the loop, in the latch; its value on entry one definition; both `mov` of an immediate | same restrictions, the one write in the loop and in the latch still counted in the write table; the copy is `value_before` the branch being a `Result`, the value on entry the header `Argument`'s outside `incoming` |
| `uniformity::read_sites` | `reach_sites`; `Carried` expanded to every definition reaching the read; a missing definition on one path not seen | `sources(v)`'s `Result`s; an `Undef` divergent; gating predicates as today |
| predicate reader | `reach_def` must be `Def`, unguarded, with uniform gates; else `Pred::Uniform` if the uniformity analysis finds it so | `value_before` must be an unguarded `Result`; the rest unchanged |
| `obstacle` | `Def` followed; `Merged`, `Carried`: `obstacle_of_any_def` | `Result` followed; `Argument`: `obstacle_of_any_def` |
| `obstacle_of_any_def` | every definition of the register | unchanged |
| `find_setp`, `predicate_reason` | the last write in the latch | unchanged |

The tracer reads a guarded write as its new value (PLAN, "an address
through one is read as the last write"); that stays, fixed in its own
commit. The uniformity analysis already reads the old value and the
guard.

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
    (`src/analysis/scalar/snapshots/`). Done. Of 99,520 reads, 2,342
    `Carried` reads are a φ at a join inside the loop, none of them a
    counter of that loop, and 1,128 `Merged` reads keep a definition a
    later write kills; no read's `Merged` misses a path writing
    nothing. Of 29,887 header queries, 8,568 `None`s are a φ of an
    enclosing loop that may be `Undef`; 159 single definitions and 10
    `Merged`s miss a write an enclosing loop carries back in, 6 of
    them counters (PLAN, "A loop counter's value on entry");
    the rest name a φ nearer than `Carried`'s loop or drop a killed
    definition.
13. The tracer holds `RegisterSsa`, and `uniformity::read_sites` reads
    `sources`.
14. `uniformity::read_sites` refuses `Undef`.
15. `trace_reg`'s `Merged` branch reads `Argument`s, `Undef` refused.
16. `trace_reg`'s `Carried` branch reads header `Argument`s.
17. `iv_init` reads the header `Argument`.
18. `induction_vars` picks its φ register by the header `Argument`.
19. The two-trip predicate loop reads `RegisterSsa`.
20. The predicate reader reads `RegisterSsa`.
21. `obstacle` reads `RegisterSsa`.
22. `reach_def`, `reach_sites` and `path_avoiding` go, and the snapshot
    of step 12 with them; the focused tests stay.

Steps 4–10 changed no output.

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
