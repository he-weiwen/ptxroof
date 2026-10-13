# Reaching values: plan (temporary, for review)

Temporary design note, to be deleted when the refactor lands or is
dropped. Rewritten 2026-10-13 after the register-effects decoder
landed; the first version and codex's review of it are in git
(`b5cdcbf`, `4edba75`).

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

## Values

Minimal SSA as in Cytron et al., "Efficiently Computing Static Single
Assignment Form and the Control Dependence Graph" (TOPLAS 1991): a φ
for a location at every block in the iterated dominance frontier of
the blocks writing it, the entry block counted as writing every
location.

```rust
/// A register, or the carry flag CC.CF.
pub(crate) enum Location { Reg(Symbol), Carry }

/// A value of one location; `Copy + Eq + Hash`, an identity.
pub(crate) struct Value { pub loc: Location, pub site: Site }

pub(crate) enum Site {
    /// Statement `stmt` writes the location. When `stmt` is guarded the
    /// value is a ψ (Stoutchinin and de Ferrière, "Efficient Static
    /// Single Assignment Form for Predication", MICRO 2001): the new
    /// value where the guard holds, `before(v)` elsewhere.
    Write(usize),
    /// A φ at the entry of block `b`, one operand per predecessor.
    Phi(BlockId),
    /// The location as the kernel starts: never written on the path.
    Entry,
}

impl Values {
    fn at(&self, loc: Location, pos: usize) -> Value; // the value a read at `pos` sees
    fn phi_operands(&self, v: Value) -> Vec<(BlockId, Value)>; // per predecessor
    fn before(&self, v: Value) -> Value; // a guarded write's old value
    fn writes(&self, v: Value) -> Vec<usize>; // the writes behind `v`, through φs and ψs
}
```

- `at(loc, pos)`: the last write of `loc` in `pos`'s block before
  `pos`; else the φ at that block, if one is placed; else the entry
  block's `Entry`; else the value at the end of the immediate
  dominator. In minimal SSA a block without a φ for a location sees
  what its immediate dominator ends with.
- `phi_operands`: the value at the end of each predecessor, the same
  walk from the predecessor's last statement. The entry block has
  predecessors when a branch targets a label at the kernel's start;
  its φ has `Entry` as one more operand, for the kernel's start.
- The dominance frontiers come from the dominator tree the loop forest
  already holds (Cooper, Harvey and Kennedy, "A Simple, Fast Dominance
  Algorithm", §4). Frontiers and φ placement need no loop structure, so
  a cycle with two entries needs nothing special.
- Unreachable blocks have no immediate dominator and are outside the
  analysis, as for `DominanceInfo`.

`Values` is built in `AffineValueTracer::new` from its definition
table and `carry_defs`; bindings do not change it. `at` and
`phi_operands` are walks up the dominator tree, not cached.

## Answers to `docs/architecture.md`'s five questions

1. Predicated writes: a guarded `Write` is a ψ; `before` gives the old
   value.
2. Loop merges: a φ at the header, whose back-edge operands are the
   values at the end of the latches.
3. Undefined inputs: `Entry`. Unsupported operations: the decoder's
   fallback writes, opaque `Write`s.
4. The CFG is unchanged; values name PTX statements and blocks.
5. Registers and CC.CF only; no memory.

## Users

| User | Today | With `Values` |
|---|---|---|
| `trace_reg` | `Def` interpreted; `Carried(l)`: `l`'s counter, `init + (k−1)·step`; `Merged`: one shared immediate | `Write`: same; φ at the header of a loop holding the read: the counter rule; other φ: one shared immediate |
| `iv_init` | `reach_def` excluding the loop's own definitions | the header φ's operands from outside the loop, if they agree |
| `obstacle`, `obstacle_of_any_def` | every definition of the register | `writes(v)` |
| `induction_vars` | the loop's own definitions of a register, chained by hand; the chain's φ is a register with a definition reaching the header from outside | the same chain, ending at a header φ |
| two-trip predicate loop (`trip_counts.rs`) | a hand scan for the φ register's one definition in the loop and `reach_def` excluding the loop | the header φ's operands |
| `uniformity::read_sites` | `reach_sites`; for `Carried`, every definition that can reach the read | `writes(v)`; gating predicates as today |
| predicate reader | `reach_def` must be `Def`, and unguarded | `at` must be an unguarded `Write` |
| `find_setp`, `predicate_reason` | the definition table, latch block only | unchanged; they ask which statement writes last in the latch, not which value |

Guarded writes stay as the users treat them today (PLAN, "an address
through one is read as the last write"); `before` is there for the
fix, which is its own commit.

## Steps, one commit each

Every step diffs every fixture's text and JSON output against the
previous commit and explains each change.

1. Dominance frontiers, with unit tests against hand-computed ones
   (diamond, loop, nested loops, a cycle with two entries).
2. `Values`, with unit tests: straight line, diamond, one-sided write,
   loop counter, nested loops, guarded write, a cycle with two entries,
   CC.CF across a call.
3. A corpus snapshot of every register read in the fixtures where
   `writes(at)` and `reach_def`'s definitions differ; each class of
   difference explained before any user switches.
4. `uniformity::read_sites` reads `Values`.
5. `trace_reg` and `iv_init` read `Values`.
6. `induction_vars` and the two-trip predicate loop read `Values`.
7. The predicate reader and the obstacle reasons read `Values`.
8. `reach_def` and `reach_sites` go, with the snapshot of step 3.

The uniform variable is a separate plan on top of step 8.

## Open questions for review

1. Should an `Entry` operand of a φ be dropped for the tracer (PTX does
   not define a read of an unwritten register), or kept, as for the
   uniformity analysis?
2. Does any user rely on today's `Merged` keeping killed definitions,
   or on `Carried` naming the innermost interfering loop?
3. `iv_init` for a header with several predecessors outside the loop:
   take the value when they agree, or refuse as today's matcher does?
