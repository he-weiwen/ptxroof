# Launch grid: plan (temporary, for review)

Temporary design note, to be deleted when its steps land or it is
dropped. It takes up the guards PLAN lists under "A guard comparing the
thread index with a CTA index, an unbound parameter or a loop counter
is an unknown selector", for the case where the launch's grid is known.

## Goal

Given the grid (`--grid`, or the record `--ncu` names), a comparison
over `%tid` and `%ctaid` is evaluated for every CTA of the launch, so
that:

- per CTA, the threads column and the per-CTA totals bound each block
  by the lanes that hold at some CTA, and are exact where every CTA
  agrees;
- per launch, the sum over CTAs is exact, for the ncu comparison and a
  new "totals per launch".

## Measurement

A prototype, not committed, on `00a6135`, of steps G1–G3 without
G0 and without folding or substitution (G2), run with `--ncu` over the
46 kernels that have a record:

- No block's verdict became `contradicted`. tut_ll's confirmed blocks
  went from 11 to 15. Single-opcode lines that became exact and equal
  the measurement: norope's store (line 93, 589,824, today
  `<= 1048576`), tut_ll's lines 398–401 (245,760), prefixdense's
  line 4349 (0).
- Per-CTA totals that became exact: gated rmsnorm's global bytes
  (load 2304 B, store 1040 B, and AI), norope's (load 2176 B, store
  1152 B).
- Two lines predict 0 against a nonzero measurement, both of several
  SASS opcodes, so untested: tut_ll's line 1470, `bra.uni` in a block
  that `128·%ctaid.x ≠ 0x7FFFFF80` never reaches under a grid of
  16×128, where ptxas attributes `UMOV`×2 and `BRA.U` to the line;
  prefixdense's line 4358, whose `UTMALDG` does not run while 12 other
  SASS on the line do.
- One regression, which step G0 removes: in scaled_mm, 8 `cp.async`
  rows lose their address, "`cvt.u64.u32` leaves 32 bits". The address
  holds `1048576·%ctaid.x − 469762048·⌊%ctaid.x/448⌋`, which is
  `1048576·(%ctaid.x mod 448)`, below 2^29; `Affine::range` bounds each
  term on its own, ±3·10^10 with `%ctaid.x < 28672`. Under the records'
  bindings a variable beside its own floor occurs in 16 of the 46
  kernels.
- Comparisons of `%ctaid` alone that hold at every CTA print as
  selections (`-%ctaid.x >= -31` in attn_bwd, grid 32×12×1); G2 folds
  them.
- Time: tut_ll 0.21 s to 1.71 s, re-splitting each form on every lane
  evaluation; rope_cache (grid 262,144) 0.00 s to 3.37 s, enumerating
  the grid for every instruction, 0.06 s with a count per predicate.
  The design computes both once.

From the census of `ccf8220`, under the records' bindings, unknown
guards whose comparisons are over `%tid` and `%ctaid` only: about 46 of
262 on memory instructions, and 52 of 373 on branches. The rest are
loaded values, refused traces, loop counters and uniform values.

## Design

### Inputs

`analyze --grid x,y,z`, conflicting with `--ncu` as `--launch` does;
`--ncu` takes its record's `grid`. `analyze` binds `%nctaid.x`, `.y`
and `.z` in the kernel's bindings, beside `%ntid`. The tracer reads
them already: `%nctaid` is a symbol bound by name, and `bound` gives
`%ctaid` the range [0, n − 1] (`src/analysis/scalar/trace.rs:972`), so
the wrap check of 16- and 32-bit arithmetic covers `%ctaid` (PLAN,
"Integer arithmetic is taken not to wrap": "`%ctaid` (no grid is
bound)"). The compile-time pass of the dead-code analysis keeps
`.reqntid` alone. The report shows the grid beside the block size;
`LaunchInfo` gains `grid`.

### G0: a variable beside its floor

`Affine::range` takes a variable `x` together with its terms
`⌊x/d⌋` and `x mod d` for one divisor `d`: with `x = d·q + r`,
`c₁·x + c₂·⌊x/d⌋ + c₃·(x mod d) = (c₁·d + c₂)·q + (c₁ + c₃)·r`, `q`
over `⌊lo/d⌋..⌊hi/d⌋` and `r` over `0..d−1` (`lo mod d..hi mod d`
when `q` is one value). The group's range is the intersection of that
and the per-term range, both bounds. Divisors that differ, and nested
floors, stay per term.

### G2: comparisons with `%ctaid`

`condition` reads a `setp` whose `a − b`, under the bindings, has
constant coefficients and base over `%tid`, `%ctaid` and their floors
and remainders, when the grid is bound. The type check uses
`%ctaid`'s range over the grid beside `any_block`. With `U` the
`%ctaid` terms and their range over the grid:

- `U` one value: it is substituted, and the comparison is over `%tid`
  alone as today (norope, `%ctaid.y` under a grid of 8192×1×1);
- no `%tid` term, and the range decides the comparison: `True` or
  `False`, as a comparison of constants is today;
- otherwise `Pred::Cmp`, the `Constraint` holding `U`'s range.

A lane's value is then the interval `L(tid) + U`; the comparison is
`Yes` when it holds at both ends, `No` at neither, `Maybe` otherwise
(`eq`/`ne` by whether 0 lies in it), so `tally` counts the lanes in at
every CTA as certain and those in at some CTA as possible. A
`Constraint` whose `U` varies is not `exact`; it is `lane_exact` when
it has no `%tid` term, as `Uniform` is.

### G3: counts per launch

`ThreadSet::launch(shape, grid)` gives the threads and warps of the
whole launch, when every leaf of the predicate is decided at a given
CTA (no `Unknown` or `Uniform`), else none. It enumerates the CTAs
over the axes the predicate's `U`s use and multiplies by the others.
For each comparison with `U` it precomputes `L`'s least and greatest
value over the block; at a CTA, the comparison either holds on every
lane, on none, or depends on the lane, and the CTAs are grouped by
that and, in the last case, by `U`'s value. Each group's lanes are
counted once at one of its CTAs. More than 2^24 CTAs on the used axes
gives none.

The builder computes it once per block set and per guarded
instruction. The ncu comparison predicts trips × launch threads where
it exists, exact unless the chain or `.maxntid` makes it a bound.

### G4: totals per launch

With a block shape and a grid, `totals_per_launch` aggregates as per
CTA, each block and guarded instruction at its launch threads and
warps where `launch` gives them, and otherwise at its per-CTA count
times the CTAs, a bound where that is one. The text prints it after
the per-CTA totals.

### Not in this plan

Comparisons with a loop counter or a uniform value; a per-CTA count as
a range over CTAs instead of `<=`; grids past the budget; counts
without a grid.

## Steps, one commit each

Each step diffs every fixture's text and JSON output, with and without
the records, against the previous commit and explains each change.

G0. Joint range of a variable, its floor and its remainder by one
    divisor. Unit tests.
G1. `--grid`, the record's grid, `%nctaid` bound, `LaunchInfo.grid`.
    Fixture: a 32-bit product of `%ctaid.x` that wraps under the grid
    given and is read without it.
G2. Comparisons with `%ctaid`. Fixture: `%tid.x + 64·%ctaid.x < n`
    under a grid whose last CTA is partial, and a `%ctaid`-only branch;
    `b200-mixed-index-guard.test` becomes exact per CTA.
G3. Counts per launch in the ncu comparison. The norope store's
    prediction becomes 589,824.
G4. Totals per launch.
