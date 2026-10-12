# Register-effects decoder: plan (temporary)

Temporary design note, to be deleted when the decoder lands or is
dropped. Line references are to `main` at `2f12ab8`. It replaces step 2
of the audit's "A small implementation plan"
(`docs/ptx-register-effects-audit.md`). Reviewed by codex on
2026-10-12; its corrections are folded in below.

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
| `induction_vars`, `trace.rs:199` | first operand, only if a register, and `defines_dest` |
| `find_setp`, `trace.rs:311` | a `setp` whose first operand is the register |
| `predicate_reason`, `trip_counts.rs:215` | first operand, a register or a symbol |
| `obstacle_of_any_def`, `trace.rs:156`; `trace.rs:872`; `trace.rs:928` | the definition table; every operand after the first is an input |
| merged immediate, `trace.rs:135` | `[dst, src]` with `dst` a register |
| `mov {…}, %tid`, `trace.rs:592` | position in the first operand's list |
| vector destinations, `trace.rs:600` | the first operand's shape and element count |
| `Uniformity::new`, `uniformity.rs:40` | `ops[1..]`; with a guard also `ops[..1]` and the guard |
| `operand_reads`, `uniformity.rs:253` | flattens `[base]`, `{…}`, `p\|q` |
| `lane_wise`, `uniformity.rs:220` | `addc`, `subc`, `madc` read CC.CF |
| predicate reader, `thread_participation.rs:527` | second register of `p\|q`, read as the first's negation for `setp` |

The two tracers of a kernel (`trip_counts.rs:69`, `report/build.rs:827`)
each build the definition table. Induction-chain matching
(`trace.rs:227`), predicate-copy matching (`trip_counts.rs:258`) and
the report's memory-operand association (`report/build.rs:310`) read
operands by position too; they keep doing so.

## Facts the design rests on

1. **Written operands are a prefix of the operand list**, in every
   Syntax block of PTX ISA 9.3 (codex checked all 208, textures,
   surfaces and video included). Forms with two written operands:
   - `mbarrier.test_wait`/`try_wait` with `.phase_type::primary`:
     `waitComplete|reportPredicate {, reportValue}, [addr], …`
     (§9.7.14.16.19);
   - `tcgen05.ld.red`: `r, redval, [taddr]` (§9.7.17.8.3);
   - `call (ret-param), func, (param-list)`, whose return list can
     hold registers (§9.7.13.5, `call (d), h, (a, b)`).
2. **Special registers are read-only** (§10). ptxas 13.4 rejects
   `mov.u32 %tid.x, 1`: "Read-only special register cannot be
   destination argument".
3. **CC.CF is the only register state no operand names** (§9.7.2:
   "No other instructions access the condition code"; calls do not
   preserve it). Other unnamed state is not a PTX register: the stack
   pointer (`stacksave`, `stackrestore`, `alloca`), a call's return
   address, `pmevent`'s counters, `setmaxnreg`'s register ownership.
4. **`wgmma.mma_async` reads its destination**; unconditionally
   is an over-approximation when `scale-d` is false (§9.7.16.5.2).
5. **The sink `_`** may take a written operand's place: either output
   of `setp`, `match.all`, `elect`, `lop3`'s `d` with `.BoolOp`,
   elements of a vector `mov` or `ld`, and `mbarrier.arrive`'s state.
   It occupies its position and defines nothing.
6. **The input is PTX that ptxas accepts**, for the pinned ISA 9.3.
   Modifiers that do not change which operands are written are not
   validated.

## The 137 keywords

| Class | Decided by | Keywords | Count |
|---|---|---|---|
| A | keyword: one written operand, the first | `abs activemask alloca and atom bfe bfi bfind bmsk brev clmad clz cnot copysign cos createpolicy cvt cvta div dp2a dp4a elect ex2 fma fns getctarank isspacep ld ldmatrix ldu lg2 lop3 mad24 mapa match max min mov movmatrix mma mul mul24 neg not or popc prmt rcp redux rem rsqrt sad selp set setp shf shfl shl shr sin slct sqrt stacksave szext tanh testp vote xor` | 68 |
| A, carry | as A; `.cc` writes CC.CF; the `*c` forms read it | `add sub mad addc subc madc` | 6 |
| B | keyword: no written operand | `applypriority bra brkpt brx cp discard exit fence griddepcontrol membar nanosleep pmevent prefetch prefetchu red ret setmaxnreg st stackrestore stmatrix tensormap trap` | 22 |
| C | a sub-operation modifier | below | 8 |
| `call` | operand shape: the leading `(…)` list; not decoded until the parser keeps it | `call` | 1 |
| D | not decoded: outside the audit | `fabric`, textures and surfaces (`istypep suld suq sured sust tex tld4 txq`), video (`vabsdiff* vadd* vavrg* vmad vmax* vmin* vset* vshl vshr vsub*`) | 32 |

Class C, written operands. Each list is exhaustive; any other
sub-operation is not decoded.

- `bar`, `barrier`, by the modifier after an optional `.cta`
  (`barrier.cta.red`): `red` 1; `sync`, `arrive`, `warp`, `cluster`
  0.
- `mbarrier`: `init`, `inval`, `expect_tx`, `complete_tx` 0; `arrive`,
  `arrive_drop` (state, or `_`), `pending_count`, `check_layout` 1;
  `test_wait`, `try_wait`: the operands before the first `[addr]`, 1 or
  2.
- `tcgen05`: `ld` 1, `ld` with `.red` 2; `alloc`, `dealloc`,
  `relinquish_alloc_permit`, `st`, `wait::ld`, `wait::st`, `cp`,
  `shift`, `mma`, `fence::before_thread_sync`,
  `fence::after_thread_sync`, `commit` 0.
- `wgmma`: `mma_async` 1, also read; `fence`, `commit_group`,
  `wait_group` 0.
- `wmma`: `load`, `mma` 1; `store` 0.
- `multimem`: `ld_reduce` 1; `st`, `red`, `cp` 0.
- `clusterlaunchcontrol`: `query_cancel` 1; `try_cancel` 0.

## Data structures

The decoder answers per instruction:

```rust
/// How many of `instr`'s leading operands it writes; None: not decoded.
pub fn written_operands(module: &Module, instr: &Instr) -> Option<usize>;
```

and, when the uses side lands, whether it writes or reads CC.CF and
whether it reads its written operands (`wgmma.mma_async`). A written
register is identified by its statement, operand index and position
within that operand (`p|q`, `{…}`), so `_` keeps its place and
`setp _|q`'s `q` stays the second output.

The per-kernel table built from it:

```rust
pub enum Def { Reg(Symbol), Carry }
pub enum Use { Reg(Symbol), Special(SpecialReg), Carry }

pub struct DefUse {
    stmts: Vec<StmtRegs>,              // by statement index
    defs: Vec<(Def, u8, u8)>,          // with operand index and position
    uses: Vec<Use>,                    // with `[base]` registers and the guard
    defs_of: HashMap<Def, Vec<usize>>, // replaces the tracer's definition table
}
struct StmtRegs { defs: Range<u32>, uses: Range<u32>, decoded: bool }
```

- A statement that is not decoded, unparsed statements included,
  defines and uses every register it names, and `Carry`; `decoded`
  lets a reason say "may be written by `tex`".
- A guarded definition keeps the old value when the guard is false.
  The uniformity analysis adds the old value of every guarded
  definition, `Carry` included, as a dependency, as it does for
  registers today (`uniformity.rs:53`).
- An executed `call` defines `Carry` as unknown (Fact 3).

## Order, one commit per case

1. **Decoder cases.** `written_operands` with one case per commit; the
   definition table uses it, and today's rule (`defines_dest`, first
   operand) for a form not yet decoded. `tests/register_effects_corpus.rs`
   snapshots the answer for every form in the fixtures and checks that
   each written operand is a register, `{…}`, `p|q` or `_`. Each commit
   diffs every fixture's text and JSON output against the previous one.
2. **Induction discovery through the decoder**, then `defines_dest`
   goes and a form not decoded defines every register it names.
   Changes `micro/counter_vector_write.ptx` (`trips = 10` today).
3. **Uses and CC.CF**: `DefUse`; `Uniformity::new`, `operand_reads`,
   `lane_wise`'s carry case and the three `skip(1)` sites move to it,
   keeping the guarded old-value dependencies.
4. **Output identity, then `find_setp` and `predicate_reason`**: the
   predicate reader takes `setp`'s second output as the negated
   comparison for scalar types only (`micro/setp_f16x2.ptx`), and the
   latch's last write of the predicate decides, not the last `setp`.
5. **Parser recovery**: scoped names for every declaration kind,
   guards and bracket bases included (`micro/bare_registers.ptx`,
   `micro/scoped_shadow.ptx`); coordinates after a bracket's base;
   `!` operands and `call` lists; `Unparsed` keeps its registers. Then
   the corpus check that every `Use::Reg` has a definition.

`docs/reaching-values-refactor.md` then builds on `DefUse`.

## Coverage by case

Corpus: statements in `tests/fixtures` whose form the case decodes;
"differs" is a change from today's rule.

| Case | Corpus | Differs from today | Test |
|---|---|---|---|
| A and A-carry: the first operand (74 keywords) | 36 keywords, 184 forms, 51,527 statements | no | the snapshot; the other 38 keywords by one manual example each (`register_effects.rs`); no lit test, as no output can differ |
| B: nothing written (22 keywords) | 11 keywords, 43 forms, 3,211 statements | for a register operand of `nanosleep` or `stackrestore`; the corpus has neither (its 12 `nanosleep`s take an immediate) | `micro/nanosleep_reads.ptx`, before: "value defined by unsupported instruction `nanosleep`"; the other 11 keywords by one manual example each; no lit test for `stackrestore`, whose operand nothing reads after it in a kernel that uses it as the manual does |
| `bar`, `barrier`: `.red` 1, else 0 | `bar.sync` 448, `bar.warp.sync` 77, `barrier.sync` 71; no `.red` | for `.red`, which today defines nothing | `micro/barrier_red.ptx`, before: `[param_0]`, the value before the reduction; the other forms by one manual example each |
| `mbarrier`: by sub-operation; `test_wait`, `try_wait` the operands before the address | `arrive` 79, `init` 121, `inval` 109, `test_wait` 12, `try_wait` 123 | for `reportValue`, which the corpus does not have | one manual form per sub-operation; no lit test: ptxas 13.4 rejects `reportValue` ("Arguments mismatch for instruction 'mbarrier.test_wait'"), so no fixture of it round-trips |
| `tcgen05`: `ld` 1, `ld.red` 2, 12 others 0; the tracer's reasons skip the written operands, not the first | `alloc` 12, `commit` 61, `cp` 6, `dealloc` 19, `ld` 56, `mma` 320, `relinquish_alloc_permit` 12, `st` 49, `wait::ld` 35, `wait::st` 28; no `ld.red` | for `dealloc`, whose `taddr` no corpus kernel reads after it, and `ld.red` | `micro/tcgen05_outputs.ptx`, before: `[4 * %tid.x + param_0]` through `redval`, and "value defined by unsupported instruction `tcgen05`" after `dealloc`; `shift`, `fence::*` by one manual example each |
| `wgmma`: `mma_async` 1; `fence`, `commit_group`, `wait_group` 0; `mma_async`'s read of its accumulator comes with the uses | `fence` 1 | no | one manual form per sub-operation; no lit test, as no output can differ |
| `wmma`: `load`, `mma` 1; `store` 0 | `load` 74, `mma` 33, `store` 65 | no | the snapshot; no lit test, as no output can differ |
