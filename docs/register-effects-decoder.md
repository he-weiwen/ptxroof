# Register-effects decoder: plan (temporary, for review)

Temporary design note, to be deleted when the decoder lands or is
dropped. Line references are to `main` at `2f12ab8`. It replaces step 2
of the audit's "A small implementation plan"
(`docs/ptx-register-effects-audit.md`).

## Goal

One answer, per statement, to "which registers does it write, and
which does it read", computed once per kernel. Everything else an
instruction means stays where it is: what `cvt` or `setp` computes
(the tracer, the predicate reader, trip counting), control flow (the
CFG), memory and counting (the classifier), and whether a result
depends only on the lane's own inputs (`lane_wise`).

## Where operand roles are decided today

| Site | Rule |
|---|---|
| definition table, `trace.rs:70`, with `defines_dest`, `trace.rs:1045` | blacklist of 14 keywords; else the first operand if a register, `p\|q` or `{…}` |
| `induction_vars`, `trace.rs:199` | first operand, only if a register |
| `find_setp`, `trace.rs:311` | a `setp` whose first operand is the register |
| `predicate_reason`, `trip_counts.rs:215` | first operand, a register or a symbol |
| `obstacle_of_any_def`, `trace.rs:156` | the definition table |
| merged immediate, `trace.rs:135` | `[dst, src]` with `dst` a register |
| `mov {…}, %tid`, `trace.rs:592`; vector destinations, `trace.rs:604` | position in the first operand's list |
| `Uniformity::new`, `uniformity.rs:40` | `ops[1..]`; with a guard also `ops[..1]` and the guard |
| `operand_reads`, `uniformity.rs:253` | flattens `[base]`, `{…}`, `p\|q` |
| `lane_wise`, `uniformity.rs:220` | `addc`, `subc`, `madc` read CC.CF |
| predicate reader, `thread_participation.rs:527` | second register of `p\|q` |

The two tracers of a kernel (`trip_counts.rs:69`, `report/build.rs:827`)
each build the definition table.

## Facts the design rests on

1. **Destinations are a prefix of the operand list.** Every written
   operand in the audit's ledger (186 sections) is a leading one. Two
   forms write two operands:
   - `mbarrier.test_wait`/`try_wait` with `.phase_type::primary`:
     `waitComplete|reportPredicate {, reportValue}, [addr], …`, where
     `reportValue` is optional and `try_wait` has an optional trailing
     `timeHint` (§9.7.14.16.19, new in PTX ISA 9.3);
   - `tcgen05.ld.red`: `r, redval, [taddr]` (§9.7.17.8.3).
2. **Special registers are read-only** (§10). ptxas 13.4 rejects
   `mov.u32 %tid.x, 1`: "Read-only special register cannot be
   destination argument".
3. **CC.CF is the one register state no operand names** (§9.7.2).
   The audit: calls do not preserve it.
4. **`wgmma.mma_async` reads its destination** when it accumulates
   (§9.7.16.5.2).
5. **The input is PTX that ptxas accepts**, for the pinned ISA 9.3.
   Modifiers that do not change which operands are written are not
   validated.

## The 137 keywords

| Class | Decided by | Keywords | Count |
|---|---|---|---|
| A | keyword: one written operand, the first | `abs activemask alloca and atom bfe bfi bfind bmsk brev clmad clz cnot copysign cos createpolicy cvt cvta div dp2a dp4a elect ex2 fma fns getctarank isspacep ld ldmatrix ldu lg2 lop3 mad24 mapa match max min mov movmatrix mma mul mul24 neg not or popc prmt rcp redux rem rsqrt sad selp set setp shf shfl shl shr sin slct sqrt stacksave szext tanh testp vote xor` | 68 |
| A, carry | as A; `.cc` writes CC.CF; the `*c` forms read it | `add sub mad addc subc madc` | 6 |
| B | keyword: no written operand | `applypriority bra brkpt brx call cp discard exit fence griddepcontrol membar nanosleep pmevent prefetch prefetchu red ret setmaxnreg st stackrestore stmatrix tensormap trap`; `call` also writes CC.CF | 23 |
| C | sub-operation (first modifier) | below | 8 |
| D | not decoded: outside the audit | `fabric`, textures and surfaces (`istypep suld suq sured sust tex tld4 txq`), video (`vabsdiff* vadd* vavrg* vmad vmax* vmin* vset* vshl vshr vsub*`) | 32 |

Class C, written operands:

- `bar`, `barrier`: `.red` 1; otherwise 0.
- `mbarrier`: `init`, `inval`, `expect_tx`, `complete_tx` 0; `arrive`,
  `arrive_drop`, `pending_count`, `check_layout` 1; `test_wait`,
  `try_wait`: those before the `[addr]` operand, 1 or 2.
- `tcgen05`: `ld` 1, `ld` with `.red` 2; the other ten 0.
- `wgmma`: `mma_async` 1, also read; `fence`, `commit_group`,
  `wait_group` 0.
- `wmma`: `load`, `mma` 1; `store` 0.
- `multimem`: `ld_reduce` 1; otherwise 0.
- `clusterlaunchcontrol`: `query_cancel` 1; `try_cancel` 0.

An unknown sub-operation is not decoded. The corpus uses 53 keywords:
no class D keyword, no `.cc`, and of class C `bar`, `barrier`,
`mbarrier`, `tcgen05`, `wgmma` and `wmma`.

## Data structures

```rust
/// A register an instruction writes: one the kernel declares, or CC.CF
/// (PTX ISA §9.7.2), which no operand names.
pub enum Def { Reg(Symbol), Carry }

/// A register it reads. Special registers are only read.
pub enum Use { Reg(Symbol), Special(SpecialReg), Carry }

/// Every statement's definitions and uses, built once per kernel.
pub struct DefUse {
    stmts: Vec<StmtRegs>,                  // by statement index
    defs: Vec<Def>,                        // operand order: `{…}` elements in order, `p` before `q`, no `_`
    uses: Vec<Use>,                        // with `[base]` registers and the guard
    defs_of: HashMap<Symbol, Vec<usize>>,  // replaces the tracer's definition table
}
struct StmtRegs { defs: Range<u32>, uses: Range<u32>, decoded: bool }
```

- The decoder decides, per instruction, how many leading operands are
  written, adds `Carry` for the carry forms and `call`, and lists
  `wgmma.mma_async`'s destination among the uses. One helper flattens
  operands into registers for every instruction.
- A written operand that is not a register, `{…}` or `p|q` (a
  `Special`, an address, an immediate) makes the form not decoded.
- A statement that is not decoded, unparsed statements included, defines
  and uses every register it names, and `Carry`. `decoded` is kept so
  that a reason can say "may be written by `tex`".
- A guarded definition keeps the old value when the guard is false.
  That is a question of which definitions reach a read, left to the
  users; the guard is a use.
- Def order is operand order, so the position of a register in `defs`
  answers today's position questions (`p|q`, `ld.param.v2`,
  `mov.v4 {…}, %tid`).

## Prerequisite: one scope-aware name resolution

"Any variable declared within a scope is not available outside the
scope" (§9.7.13.1). Names are resolved in three places today:

- labels are renamed per scope (`enter_scope`, `parser.rs:438`);
- a `%` name that is a special register becomes `Operand::Special`
  unless a `.reg` declares it, checked against `kernel.reg_decls`
  (`parser.rs:320`, `parser.rs:883`), into which nested scopes'
  declarations are merged (`parser.rs:498`);
- a name declared without `%` stays `Operand::SymbolRef`.

Wrong outputs: `micro/bare_registers.ptx` and `micro/scoped_shadow.ptx`
(PLAN, "Registers declared without `%` are symbols"). Proposed: rename a
nested scope's `.reg` names as its labels are renamed, and resolve every
identifier against the enclosing scopes: a declared register, else a
special register (for a `%` name), else a label or symbol.

## Users that change

| Site | Change |
|---|---|
| definition table, `defines_dest` | `DefUse::defs_of`; `defines_dest` goes |
| `induction_vars` | reads `DefUse`; fixes `micro/counter_vector_write.ptx` |
| `find_setp`, `predicate_reason` | the last definition of the predicate in the latch, then whether it is a `setp`; can change trip counts where another write of the predicate follows the `setp` |
| `obstacle_of_any_def` | `defs_of` |
| merged immediate | the `mov` defines exactly one register |
| `mov {…}, %tid`, vector destinations | position in `defs` |
| `Uniformity::new`, `operand_reads`, branch reads | `DefUse` uses; `operand_reads` goes |
| `lane_wise` carry case | goes: CC.CF is a use like a register |
| predicate reader `p\|q` | position in `defs` |
| `trip_counts.rs:69`, `report/build.rs:827` | take one `DefUse` built in `report/build.rs` |
| parser, `Stmt::Unparsed` | keeps the names of the registers it mentions |

Unchanged: the classifier, the CFG, `count_constraint.rs`, the
tracer's interpretation of each opcode, the predicate reader's
`decode` and `condition`, trip counting's `setp` operand count, the
report's `cp` alignment lookup.

## Steps, one commit each

Each step diffs every fixture's output against the previous commit.

1. Name resolution. Fixes `bare_registers` and `scoped_shadow`; the
   round-trip allowlist loses `micro/scoped_shadow.ptx`.
2. Decoder and `DefUse`, replacing the definition table and
   `defines_dest`. Corpus tests: every (keyword, sub-operation, written
   count) seen, as a snapshot; every `Use::Reg` has a definition in its
   kernel. Expected difference from today: `tcgen05.dealloc`'s `taddr`
   is no longer defined (38 statements in 24 `tests/fixtures/b200` files).
3. `induction_vars`.
4. `find_setp`, `predicate_reason`.
5. Uniformity, the predicate reader, the tracer's position checks.
6. Unparsed statements keep their registers.

`docs/reaching-values-refactor.md` then builds on `DefUse`.

## Open questions for review

1. Is every written operand a leading one in every syntax form of the
   pinned manual, including those outside the corpus?
2. Does a modifier change which operands are written for any class A
   or B keyword?
3. Is CC.CF as a register sound under guards (a guarded `add.cc`
   writes it only when the guard holds) and across calls?
4. Is "defines and uses every register it names, and CC.CF" sound for
   an undecoded statement: does any instruction write a register it
   does not name?
5. Does any user need the operand a register came from, beyond its
   position in `defs`?
6. Name resolution: anything beyond `.reg` names and labels that a
   nested scope declares?
