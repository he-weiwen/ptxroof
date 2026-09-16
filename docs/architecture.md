# Source layout and SSA migration plan

The current refactor changes source ownership and module paths only. Function
bodies and data structures retain their behavior; no SSA representation or new
analysis framework is introduced.

## Current ownership

| Module | Responsibility | Previous location |
| --- | --- | --- |
| `support/index` | Typed indices, index vectors, and pooled ranges | `core/arena` |
| `support/intern` | String interning | `core/intern` |
| `support/paths` | Display path basename helper | `cfg/naming` |
| `ptx/ir` | Parsed, flattened PTX representation | `core/ir` |
| `ptx/literal` | Shared integer literal interpretation | helper in `parse/parser` |
| `ptx/parse` | Lexing and parsing | `parse/lexer`, `parse/parser` |
| `ptx/print` | Canonical PTX dump | `parse/ast` |
| `analysis/control_flow` | PTX CFG construction, dominators, loop discovery | `cfg`, except naming |
| `analysis/loop_names` | Source-derived loop identities and display names | loop portion of `cfg/naming` |
| `analysis/scalar/symexpr` | Symbolic count expressions | `core/symexpr` |
| `analysis/scalar/affine` | Affine values over thread, CTA, and loop variables | `affine` |
| `analysis/scalar/lane_eval` | Lane-dependent evaluation of affine values | helpers in `footprint` |
| `analysis/scalar/trace` | Reaching-definition lookup and affine interpretation | `tracer` |
| `analysis/scalar/trip_counts` | Loop trip matching and unroll pairing | `trips` |
| `analysis/thread_participation` | Threads that execute a block | `threads` |
| `analysis/memory_footprint` | Sectors and lines touched by a warp request | remaining `footprint` |
| `analysis/instruction_counts` | Classification, measurements, collection, and queries | `classify`, `core/measurement`, `report/collect`, `report/stats` |
| `report/build` | Analysis orchestration and report-specific aggregation | unchanged |
| `report/schema` | Owned, serializable output structures | `report/tree` |
| `report/names` | Kernel demangling for display | demangling portion of `cfg/naming` |
| `report/text` | Text rendering of the report schema | unchanged |

One crate remains sufficient. The CLI stays in `main.rs` and the library entry
points remain available through `lib.rs`.

## Dependency boundaries

Production code imports canonical modules directly. The old `core`, `cfg`,
and `parse` compatibility modules have been removed; integration tests now use
`ptx::ir`, `ptx::parse`, `ptx::print`, `analysis::control_flow`, and
`analysis::loop_names` instead. Callers using the removed paths must migrate.
Root analysis aliases and `report::{collect, stats, tree}` remain for compatibility.
Those aliases are not architectural layers: resolve them to their implementations
when assessing dependencies.

- Support utilities do not depend on PTX, analyses, or reports.
- PTX representation and parsing do not depend on analyses or reports.
- Analyses consume PTX IR and shared literal interpretation, not the parser.
  Unit tests may parse PTX to construct fixtures.
- Analyses do not depend on report construction or rendering.
- Report construction can depend on all analyses and the parser.
- The report schema currently embeds the memory-footprint `Range` type; that
  dependency is preserved.

Within analysis, the main dependencies are:

```text
trip_counts -> trace -> affine -> symexpr
     |           |        |
     |           |        +-> control_flow::LoopId
     |           +----------> control_flow
     +-> loop_names --------> control_flow

thread_participation -> trace
thread_participation -> lane_eval -> affine
memory_footprint -----> lane_eval
memory_footprint -----------------> affine

instruction_counts::stats -> collect -> measurement -> classify
                               +-> control_flow
```

`loop_names` is more than presentation: trip analysis uses source file/line
identity to pair unrolled main and remainder loops. It stays below reporting.
`report/build` retains its current mixed orchestration and aggregation role;
renaming it a driver would not separate those responsibilities.

## Planned SSA addition

Add SSA as a second representation alongside `ptx`, when its semantics are
implemented. Do not create empty modules or placeholder interfaces now.

```text
src/ssa/ir.rs       values, instructions, block/merge representation
src/ssa/build.rs    construction from parsed PTX and its control flow
src/ssa/verify.rs   representation invariants
```

The intended flow is:

```text
PTX text -> parsed PTX IR -> CFG -> SSA construction -> SSA-based analyses
                             +----------------------> existing analyses
```

Keep SSA representation definitions independent of the analyses that consume
them. SSA construction can use existing dominance information. Initially favor
preserving PTX block identities where the chosen SSA semantics permit it; decide
whether SSA needs its own CFG before committing to a shared graph interface.
If both representations need dominance and loop discovery, separate reusable
graph algorithms from PTX-specific CFG construction at that time.

Migrate the tracer's definition lookup first, then its affine interpretation,
and subsequently trip counts, thread participation, and address reasoning as
appropriate. `Affine` and `SymExpr` remain useful analysis value domains; SSA
does not by itself recognize induction recurrences or derive trip counts.

Preserve origin mappings from SSA operations to PTX instructions and source
locations. Instruction/FLOP/byte accounting remains tied to original PTX
operations; synthetic merge operations must not contribute to those counts.
Reports should continue attributing derived facts to those original operations.

Before implementing SSA, specify:

1. Predicated writes: a false predicate preserves the previous register value.
2. Loop merges and cyclic value dependencies.
3. Undefined inputs and unsupported operations, with conservative unknowns.
4. Whether construction changes the CFG, and how identities map between graphs.
5. Register SSA scope: memory dependencies require additional reasoning and are
   not implied by SSA register def-use links.

These future changes require functional work and are outside this layout-only
refactor. No generic dataflow framework, new IR, or pass manager is added here.

## Validation

Run `cargo fmt --all -- --check`, `cargo test --offline`, and build the CLI before
running `python3 tests/run.py --bin target/debug/ptxroof`. Keep existing snapshots,
acceptance expectations, and coverage thresholds unchanged. CUDA round trips
run only when the required toolkit executables are available.
