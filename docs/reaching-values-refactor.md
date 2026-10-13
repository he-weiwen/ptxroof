# Reaching values: plan (temporary, for review)

Temporary design note, to be deleted when the refactor lands or is
dropped. Rewritten 2026-10-13 after the register-effects decoder
landed, then revised after codex's review of the rewrite; the first
version and codex's review of it are in git (`b5cdcbf`, `4edba75`).

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

## Values (`src/ssa.rs`)

Minimal SSA as in Cytron et al., "Efficiently Computing Static Single
Assignment Form and the Control Dependence Graph" (TOPLAS 1991): a φ
for a location at every block in the iterated dominance frontier of
the blocks writing it. Every location holds `Entry` before the entry
block.

```rust
/// A register, or the carry flag CC.CF.
pub enum Location { Reg(Symbol), Carry }

/// A value of one location; `Copy + Eq + Hash`, an identity.
pub struct Value { pub loc: Location, pub site: Site }

pub enum Site {
    /// The statement writes the location. A guarded write keeps, where
    /// its guard is false, the value a read at the statement sees: a ψ
    /// (Stoutchinin and de Ferrière, MICRO 2001).
    Write(usize),
    /// A φ at the start of the block.
    Phi(BlockId),
    /// The location as the kernel starts: no write on some path.
    Entry,
}

/// The writes a value may hold, through φs, and whether some path
/// reaches it with no write.
pub struct Reaching { pub writes: Vec<usize>, pub entry: bool }

impl Values<'_> {
    fn new(cfg, doms, writes: HashMap<Location, Vec<usize>>) -> Self;
    fn at(&self, loc: Location, pos: usize) -> Option<Value>; // a read at `pos`
    fn at_end(&self, loc: Location, b: BlockId) -> Option<Value>; // after `b`'s last statement
    fn phi_operands(&self, v: Value) -> Vec<(Option<BlockId>, Value)>; // `None`: the kernel's start
    fn reaching(&self, v: Value) -> Reaching;
}
```

- `at(loc, pos)`: the last write of `loc` in `pos`'s block before
  `pos`; else the φ at that block, if one is placed; else the value at
  the end of the immediate dominator; at the entry block, `Entry`. In
  minimal SSA a block without a φ for a location starts with what its
  immediate dominator ends with.
- `at_end(loc, b)`: the same walk from after `b`'s last statement.
- `phi_operands`: `at_end` of each reachable predecessor. The entry
  block has predecessors when a branch targets a label at the kernel's
  start; its φ also has `(None, Entry)`, for the kernel's start.
- Unreachable blocks: the definition table holds their writes, and the
  uniformity analysis and thread participation read in them. They are
  in no frontier and no φ takes an operand from them, so they cannot
  reach a value in reachable code; `at` and `at_end` there are `None`.
- `reaching` walks φ operands with a visited set and returns writes in
  statement order. A guarded `Write` is not walked through: its old
  value is `at(loc, stmt)`, which the user reads, as
  `uniformity::def_reads` does today.
- The dominance frontiers come from the loop forest's dominator tree
  (Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm",
  §4). Frontiers and φ placement need no loop structure; users keep
  their own refusal of cycles with two entries.

`Values` takes the writes, not the module, so it is independent of the
analyses (`docs/architecture.md`). The tracer builds it in
`AffineValueTracer::new` from its definition table and `carry_defs`.

## Answers to `docs/architecture.md`'s five questions

1. Predicated writes: a guarded `Write` is a ψ; its old value is the
   value a read at the statement sees.
2. Loop merges: a φ at the header, whose back-edge operands are the
   values at the end of the latches.
3. Undefined inputs: `Entry`, kept in φs and refused where a value is
   interpreted. Unsupported operations: the decoder's fallback writes,
   opaque `Write`s.
4. The CFG is unchanged; values name PTX statements and blocks.
5. Registers and CC.CF only; no memory.

## Users

| User | Today | With `Values` |
|---|---|---|
| `trace_reg` | `Def` interpreted, a counter at or after its increment `init + k·step`; `Carried(l)`: the counter of `l`, the innermost loop holding the read and an interfering definition, `init + (k−1)·step`; `Merged`: one shared immediate | `Write`: same; φ at the header of loop `l`: `l`'s counter rule; other φ: one shared immediate, refused when `entry` |
| `iv_init` | `trace_reg` at the header, the loop's own definitions excluded: a dominating definition, or a merge of one immediate | the header φ's operands from outside the loop, traced as one φ; with no φ at the header, the value at its immediate dominator's end |
| `induction_vars` | the loop's own definitions of a register, chained by hand; the chain's φ register is the first with a definition reaching the header from outside | same chain and restrictions; the φ register is the first whose header φ has an outside operand with a write behind it |
| two-trip predicate loop (`trip_counts.rs`) | the branch predicate one copy in the loop; the copied register one definition in the loop, in the latch; its value on entry one definition; both `mov` of an immediate | same restrictions; "one definition" is `at` being a `Write`, the value on entry the header φ's outside operand |
| `uniformity::read_sites` | `reach_sites`; `Carried` expanded to every definition reaching the read; a missing definition on one path not seen | `reaching(v).writes`; `entry` divergent; gating predicates as today |
| predicate reader | `reach_def` must be `Def`, unguarded, with uniform gates; else `Pred::Uniform` if the uniformity analysis finds it so | `at` must be an unguarded `Write`; the rest unchanged |
| `obstacle` | `Def` followed; `Merged`, `Carried`: `obstacle_of_any_def` | `Write` followed; φ: `obstacle_of_any_def` |
| `obstacle_of_any_def` | every definition of the register | unchanged |
| `find_setp`, `predicate_reason` | the last write in the latch | unchanged |

The tracer reads a guarded write as its new value (PLAN, "an address
through one is read as the last write"); that stays, fixed in its own
commit. The uniformity analysis already reads the old value and the
guard.

## Steps, one commit each

Each step diffs every fixture's text and JSON output against the
previous commit and explains each change. A switch that changes an
answer (a killed definition dropped, `Entry` refused, a different
loop's counter) is its own commit with a test pinning the new answer.

1. Dominance frontiers. Done (`33792a0`).
2. `Location`, `Value`, `Site`, φ placement; tests of which blocks get
   which φs.
3. `at` and `at_end`; tests: straight line, diamond, one-sided write,
   loop counter, nested loops, two-entry cycle, entry with
   predecessors, unreachable block, CC.CF across a call.
4. `phi_operands` and `reaching`; tests through nested φs and a cycle.
5. The tracer builds `Values`. A temporary corpus snapshot of, for
   every explicit and implicit (CC.CF) read and every header query
   excluding its loop: today's `ReachingDefinition`, the new `Value`,
   its φ block, `reaching`. Each class of difference explained.
6. `uniformity::read_sites` reads `reaching`.
7. `uniformity::read_sites` refuses `entry`.
8. `trace_reg`'s `Merged` branch reads φs, `Entry` refused.
9. `trace_reg`'s `Carried` branch reads header φs.
10. `iv_init` reads the header φ.
11. `induction_vars` picks its φ register by the header φ.
12. The two-trip predicate loop reads `Values`.
13. The predicate reader reads `Values`.
14. `obstacle` reads `Values`.
15. `reach_def`, `reach_sites` and `path_avoiding` go, and the snapshot
    of step 5 with them; the focused tests stay.

## The uniform variable: what its plan must settle

- A value defined in a loop can be uniform in each iteration and
  differ between iterations (`uniformity.rs`,
  `a_value_from_a_loop_is_divergent_after_it_when_its_exit_is`). An
  address over `uniform(v)` is invariant only outside the loops
  holding `v`'s definition; the loop footprint
  (`report/build.rs`) must not read a missing `Iter` term as stride 0.
- The tracer finds induction variables in `new`; `Uniformity` is
  built from a tracer afterwards. A tracer that falls back to
  `uniform(v)` needs the uniformity answer at that point.
