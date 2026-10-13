# Reaching values: refactor plan (temporary, for review)

Temporary design note, to be deleted when the refactor lands or is
dropped. Line references are to `main` at `1c5d988`.

## Goal

One answer to "which value does a register read see", computed in one
place and shaped like an SSA value, so that

- the tracer, the uniformity analysis, trip counting and the predicate
  reader stop re-deriving it each in their own way;
- the next feature (a uniform value the tracer cannot read becomes an
  affine variable, `uniform(%r58@93)`, instead of an unknown address)
  can key that variable by value rather than by register name;
- moving to SSA later (`docs/architecture.md`: "migrate the tracer's
  definition lookup first") replaces one implementation, not its users.

A value identity needs complete write discovery (review finding 1
below): the register-effects decoder (`src/analysis/register_effects.rs`)
now provides it.

## What exists

`AffineValueTracer::reach_def(reg, pos, exclude) -> ReachingDefinition`
(`src/analysis/scalar/trace.rs:370`):

```rust
enum ReachingDefinition {
    Def(usize),          // one definition, and no other can reach pos
    Carried(LoopId),     // redefined in a loop holding pos: innermost such loop
    Merged(Vec<usize>),  // the dominating definition plus those reaching pos around it
    None,                // no definition: a special register, or undefined
}
```

- Same block before `pos`, else the dominator chain; a definition that
  can reach `pos` on a path avoiding the dominating one makes the
  answer `Carried` (if it lies in a loop holding `pos`) or `Merged`.
- `exclude: Some(l)` ignores definitions inside loop `l`: the value on
  entry to `l`.
- `Carried` names the loop, not the definitions. `Merged` names the
  definitions, not where they meet.
- A guarded definition is returned like any other.
- Nothing is cached; every call walks the dominator chain and runs
  `path_avoiding` searches.

### Its users

| User | Variants used | What it does with them |
|---|---|---|
| `trace_reg` (`trace.rs:492`) | all, with `exclude` | `Def`: interpret the instruction; an enclosing loop's counter increment is `init + k·step`. `Carried(l)`: `init + (k−1)·step` if `reg` is `l`'s counter, else refuse. `Merged`: only one shared immediate. `None`: special registers. Guards ignored (PLAN, Wrong output). |
| `obstacle` (`trace.rs:860`) | all | `Def`: keep walking operands. `Carried`/`Merged`: `obstacle_of_any_def`, which walks **every** definition of `reg` (`trace.rs:150`), not those reaching `pos`. |
| `induction_vars` (`trace.rs:264`) | `Def`/`Merged` vs others | With `exclude = l` at the header: the chain register with a definition from outside the loop is the counter's entry value. |
| Two-trip predicate loop (`trip_counts.rs:265–288`) | `Def` only | `mov p, phi` reaching the branch; then a hand scan of `defs[phi]` for its one in-loop definition in the latch; then `reach_def(phi, header, Some(l))` for the entry definition; both immediates. This reconstructs a loop-header φ(entry, latch) by hand. |
| `find_setp` (`trip_counts.rs:171`, `trace.rs:304`) | none | Scans the latch block backwards for the `setp` writing the predicate: a second definition lookup. |
| `PredicateReader::read_definition` (`thread_participation.rs:501`) | `Def` only | Unguarded definition, branches between it and the read uniform (`gates_uniform`); memo keyed by definition. |
| `uniformity::read` (`uniformity.rs:151`) | all | `Def`/`Merged`: the definitions plus gating branches. `Carried`: re-derives the definitions from `defs[reg]` by reachability. Exits of loops holding a definition but not the read (η), branches selecting among definitions (γ). |
| `per_lane` (`report/build.rs:218`) | none directly | A counter `k` of a loop not holding the access is one value per warp only if the loop's exits are uniform (`exits_uniform`). |

Separately, which operands an instruction reads or writes is decided in
three places: the tracer's definition table and `defines_dest`
(`trace.rs:67`, `trace.rs:947`), `uniformity::registers`
(`uniformity.rs:246`), and the predicate reader's per-mnemonic decoding
(`thread_participation.rs:528–600`).

## What the users need

1. The definitions a read can see (uniformity, `obstacle`, the planned
   uniform variable). `Carried` does not give them.
2. Whether they meet at a loop header or at a join (the tracer's counter
   rule; uniformity's η versus γ).
3. The value on entry to a loop (counter initial values, the two-trip
   predicate loop).

These are a φ's operands, its block (a loop header — a μ in Gated SSA —
or a join — a γ), and a header φ's entry operand.

## Proposed form

```rust
/// The value a register read sees: one definition, or the merge of
/// several (an SSA φ).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct Value {
    pub reg: Symbol,
    pub defs: Vec<usize>,          // sorted; one element: a plain definition
    pub header_of: Option<LoopId>, // the merge is at this loop's header (today's Carried)
}

pub(crate) struct ReachingDefinitions<'a> {
    module, kernel, cfg, forest,
    defs: HashMap<Symbol, Vec<usize>>,   // moved from the tracer, same heuristic
    reach: Vec<Vec<bool>>,               // moved from the tracer
    cache: RefCell<HashMap<(Symbol, usize, Option<LoopId>), Option<Value>>>,
}

impl ReachingDefinitions<'_> {
    fn value_at(&self, reg: Symbol, pos: usize) -> Option<Value>;      // None: no definition
    fn value_on_entry(&self, reg: Symbol, l: LoopId) -> Option<Value>; // today's exclude
    fn block_of, in_loop, path_avoiding, last_instr                    // moved, unchanged
}
```

`value_at` is today's `reach_def` with `Carried` filled in with its
definitions (the reachability filter `uniformity::read` does now). The
tracer holds `&ReachingDefinitions`; `Uniformity::new` takes
`&ReachingDefinitions`, not the tracer (it uses only `defs`, `reach`,
`reach_def`, `block_of`, `in_loop`, `last_instr`, `cfg`, `forest`, and
no bindings, so one instance serves both tracers of a kernel).

Construction order per kernel (`report/build.rs:767–828`):
`ReachingDefinitions` → `Uniformity` → trip-count tracer and report
tracer → `thread_sets` → accesses.

## Migration, one commit each

| Step | Change | Output |
|---|---|---|
| 1 | Introduce `ReachingDefinitions` and `Value`; `reach_def` becomes a thin wrapper over `value_at`; tracer and uniformity take it | unchanged |
| 2 | `uniformity::read` uses `Value.defs`/`header_of`; drop its re-derivation | unchanged |
| 3 | `trace_reg` matches on `Value` (`defs.len()`, `header_of`) | unchanged |
| 4 | Two-trip predicate loop reads `value_at(phi, header)` = `{entry, latch}`, drop the hand scan | expected unchanged; checked by fixture diff |
| 5 | `induction_vars` filters the shared table by loop instead of rescanning blocks | unchanged |
| 6 | `obstacle` walks `Value.defs`, not every definition | reasons may change |
| 7 | `find_setp` becomes `value_at(pred, branch)` | can find a `setp` in a dominating block: trip counts may change |
| 8 | Uniform variable `Var::Uniform(Value)` in the tracer (separate plan) | new |

Each step diffs every fixture's text output against the previous
commit; steps 1–5 must show none.

## SSA correspondence

| Now | With SSA |
|---|---|
| `value_at` via `reach_def` | the SSA value a use refers to |
| `Value { defs, header_of }` | a definition, or a φ at a join / loop header |
| `value_on_entry` | a header φ's preheader operand |
| uniformity's γ via the `reach` matrix | Gated SSA γ predicates |
| uniformity's η via loop exits | LCSSA exit φs |
| destination heuristic | the register-effects decoder (audit) |

## Open questions for review

1. Is `Merged`'s list every definition that can reach `pos`, or can a
   definition reaching only through another merge be missed? `Value`
   would use it as the identity of a φ.
2. `Carried` reports the innermost loop holding `pos` with an
   interfering definition. For a register redefined in an outer loop
   and read in an inner one, is `header_of` the right loop?
3. Guarded definitions: should `value_at` return the guarded definition
   together with the value before it (a select of old and new), fixing
   part of PLAN's guarded-definition entry, or keep returning the
   guarded definition alone and leave each user to check the guard as
   today? The first changes tracer output.
4. Cache key `(reg, pos, exclude)`: any user that would see a stale
   answer (all inputs are immutable after construction)?
5. Should the destination heuristic move now (step 1) or wait for the
   decoder?

## Review findings (codex, 2026-10-11)

Not yet folded into the plan above.

1. `(reg, defs)` is a value identity only with complete write
   discovery: a write the definition table misses lets two reads of
   one register in one thread see different values under one key.
   The table now records `{…}` destinations, but not registers
   declared without `%`, guarded writes as merges, or the outputs the
   audit lists (PLAN, Wrong output).
2. `Value` loses distinctions users rely on. `Merged([d])` (only one
   branch defines the register) would become a plain definition, so
   `trace_reg`, the predicate reader and the trip matchers could
   accept what they refuse today. `Carried` names the innermost loop
   holding `pos` with an interfering definition, not where values
   merge, so `header_of` claims more than the code knows.
3. Steps 1–5 are not output-preserving: the singleton-merge
   conflation (1, 3); filling `Carried` by reachability keeps
   overwritten entry definitions the two-trip matcher excludes (4);
   induction discovery reads only a register first operand while the
   table reads `p|q` and `{…}` (5; `micro/counter_vector_write.ptx`).
4. Corrections to "What exists": `Merged` need not hold a dominating
   definition, several sites, or sorted ones, and can keep killed
   definitions; `None` means no candidate found; `reach_def` returns
   early for a missing register or a same-block definition. Missed
   users: `predicate_reason` (`trip_counts.rs`), which scans the latch
   with its own first-operand rule, and `Uniformity::new`, which
   enumerates every definition site. `find_setp` skips intervening
   non-`setp` writes, so step 7 can change trip counts within a latch.
5. Open questions: (1) `Merged` over-approximates and omits undefined
   incoming paths; it is not a φ. (2) An outer-only interfering write
   selects the outer loop; a dominating write can return `Def`.
   (3) A guarded result needs guard, new value and old value; keep
   today's behaviour in the refactor and fix guards separately.
   (4) No staleness within one kernel's immutable CFG and forest.
   (5) Move the destination rule now only with its exact behaviour;
   do not unify induction discovery's rule in a step claimed to
   preserve output.
6. Keep the interface explicitly approximate until operand roles are
   decoded (audit, "A small implementation plan"); prefer an opaque
   value identity with queries over exposing site sets.

Also: each kernel builds two tracers (`trip_counts.rs` and
`report/build.rs`), each scanning for definitions. Which operands an
instruction reads or writes is decided in six places: the definition
table, induction discovery, `find_setp`, `predicate_reason`,
`uniformity::registers`, and the predicate reader's per-mnemonic
decoding.
