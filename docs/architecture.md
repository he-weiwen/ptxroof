# Source layout and execution

The production Rust files form a dependency DAG, which `tests/architecture.rs`
enforces. Unit-test modules are excluded: they may parse fixtures and call
higher-level entry points.

## Ownership

Paths below are relative to `src/`, with `.rs` omitted. Directory `mod.rs` files
only declare modules and selected public entry points.

| File | Intent and scope |
| --- | --- |
| `support/index` | Typed indices, index vectors, pooled ranges |
| `support/intern` | String interning |
| `ptx/ir` | Parsed PTX representation, including register-family membership |
| `ptx/literal` | Integer literal interpretation |
| `ptx/parse/lexer` | Tokenize PTX text |
| `ptx/parse/parser` | Construct IR from tokens |
| `ptx/print` | Canonical PTX dump |
| `ptx/cfg` | PTX basic blocks, edges and CFG construction |
| `analysis/control_flow/dominators` | Dominance, post-dominance and frontiers |
| `analysis/control_flow/loops` | Natural loop forest and irreducible edges |
| `analysis/control_flow/sync_dependence` | Where the paths from each branch's successors meet again (LLVM's sync dependence), and which branches decide when a thread leaves a loop |
| `ssa` | Register SSA values and def/use queries over the PTX CFG |
| `analysis/register_effects` | Instruction register reads and writes, CC included |
| `analysis/kernel_analyses` | One kernel's register writes, SSA, parameter indices, reachability, branch joins and loop exit branches, with control-flow queries |
| `analysis/special_registers` | Sharing and stability of PTX special-register values |
| `analysis/uniformity_info` | Propagate divergence through definitions and controlling branches |
| `analysis/affine/sym_expr` | Symbolic expressions and simplification |
| `analysis/affine/affine_expr` | Affine expressions over thread, CTA, loop and uniform variables |
| `analysis/affine/int_type` | Integer types and representable ranges |
| `analysis/affine/lane_eval` | Evaluate affine forms for lanes and warp layouts |
| `analysis/affine/induction_variables` | Recognize loop counter induction variables on SSA values |
| `analysis/affine/exit_limit` | Loop-exit inequalities solved for a trip count |
| `analysis/affine/affine_properties` | What the affine analysis keeps of an SSA value: its expression, or why it has none |
| `analysis/affine/unknown_reason` | Why a value has no affine expression, and its sentence where it is read |
| `analysis/affine/affine_analysis` | Register values as affine forms and one loop's trip count; their mutual recursion and the cache of both |
| `analysis/affine/affine_analysis/loop_trip_count` | A loop's trip count from its latch, guard and predicate φ, as methods of the analysis |
| `analysis/affine/affine_analysis/transfer` | The value each instruction writes, from the values it reads, as methods of the analysis |
| `analysis/affine/trip_counts` | Every loop's trip count, and unrolled main/remainder pairs |
| `analysis/loop_names` | Source-derived loop identities used by unroll pairing and display |
| `analysis/thread_participation` | Predicates and sets of threads executing blocks and accesses |
| `analysis/dead_code` | Instructions with no live effect through register SSA |
| `analysis/memory_footprint` | Warp sectors and loop requested, distinct and sector bytes |
| `analysis/instruction_counts/measurement` | Accounting dimensions and contribution/measurement records |
| `analysis/instruction_counts/classify` | Decode instructions into categories and contributions |
| `analysis/instruction_counts/count_constraint` | Constraints on comparing PTX and measured instruction counts |
| `analysis/instruction_counts/collect` | Attach block/loop execution context to instruction contributions |
| `analysis/execution_counts` | The threads, warps and trips that run each block and statement, per thread, per CTA or over the launch, exact or a bound; the FLOPs, bytes and conversions they sum to |
| `report/bindings` | Caller parameter bindings and their text syntax |
| `report/measured` | Import and validate NCU records; compare measured and predicted quantities, and the comparisons' results |
| `report/schema` | Owned serializable report structures |
| `report/build` | Orchestrate kernel analyses and aggregate their results into reports |
| `report/names` | Kernel demangling for display |
| `report/text` | Render report structures as text |
| `report/annotate` | The input PTX with each line's analysis results appended as a comment |
| `main` | CLI arguments, file I/O, output selection and error handling |
| `lib` | Library modules and version entry point |

One crate. `report::analyze`, `report::AnalyzeOptions`, `report::AnalyzeError`,
`report::BindingSpec`, `report::parse_bind` and `report::Report` are the public
report entry points; internal callers use canonical modules directly.

## Dependency boundaries

`tests/architecture.rs` rejects a dependency cycle between production files and
checks these boundaries:

- Support utilities depend only on support utilities.
- PTX representation, parsing and CFG construction depend only on PTX and support.
- Analyses and SSA do not depend on the parser or on reporting.
- Report construction may call analyses and the parser. The schema embeds the
  memory-footprint `CountRange`, `LoopBytes` and `ByteRange` types.

`register_member` belongs to the IR because register declarations define family
membership; both `Kernel::register_type` and the parser use it. `measurement`
owns `Precision`, `Pipe`, `Space`, `Direction` and `ArithKind` alongside
`Contribution` and `MeasureKind`; `classify` produces those records. `measured`
reads bindings from `report/bindings`, not from the report builder.

`KernelAnalyses` holds one kernel's structure and no scalar interpretation.
`UniformityInfo` borrows it and reads instruction semantics; it does not use
`AffineAnalysis`. An `AffineAnalysis` borrows a `KernelAnalyses` and its
`UniformityInfo`, and holds the recursion between value and trip-count queries with
a cache of both that keeps no result read from one still being computed
outside it; it counts every loop when it is built, inner loops first, and
traces a value when a query reads it. `trip_counts` reads its per-loop results; nothing in
`affine_analysis` calls back into it.

## Per-kernel flow

`main::run` calls `report::annotate::annotate`, which reads `each_kernel`;
`each_kernel` parses the module, checks an NCU
record against it, and for each kernel resolves the bindings and launch shape
and builds the CFG, the loop forest and one `KernelAnalyses`. `KernelReportBuilder`
then:

1. builds the kernel's `UniformityInfo` and one `AffineAnalysis` under those
   bindings;
2. names loops, collects every loop's trip count from that instance, and pairs
   unrolled loops;
3. collects instruction contributions per block;
4. computes thread participation from the same instance;
5. takes compile-time thread sets: the same sets when the bindings are exactly
   the `.reqntid` block shape, otherwise from a second `AffineAnalysis` over the
   same `KernelAnalyses` and `UniformityInfo` under that shape, so runtime parameter bindings never make
   a conditionally executed instruction look removable;
6. marks dead code;
7. builds `ExecutionCounts`, which takes the trip counts, thread participation
   and dead code (the builder reads them through it), and refines dominance and
   count qualifiers with the participation's dead edges;
8. traces addresses and computes memory footprints.

`report/annotate` prints the results on the PTX's lines; `report::build::compare`
compares them with an NCU record for the tests. `--dump-ast` reads, parses,
prints and returns.

## Register SSA semantics

SSA is a register representation alongside PTX. `src/ssa.rs` places merges over
the PTX CFG and identifies the value a read sees, using writes supplied by its
caller. It has no instructions of its own. Dominance and loop information retain
PTX block identities.

- Predicated writes: a guarded `Result` keeps the old value where its predicate
  is false.
- Loop merges: an `Argument` at the header has incoming values from the latches.
- Undefined inputs: `Undef` remains in merges and is refused when interpreted.
- Unsupported operations: conservative register writes and opaque `Result`s.
- Registers and CC only: memory dependence is not implied by register def/use.

Origin mappings stay tied to PTX statements and source locations. Synthetic
merges never contribute instruction, FLOP or byte counts.

## Validation

`cargo test --offline --test architecture` parses Rust syntax with `syn`. It
follows `use` items, including a module a `use` binds to a name, reexports and
qualified paths; rejects a dependency cycle with its path; and checks the
boundaries above. `mod` declarations and visibility paths are not use edges.
`#[cfg(test)] mod` contents are excluded; other conditional source is checked
conservatively. It does not expand macros or resolve inferred types: a method
call on a value whose type a file never names is not an edge. The same test
writes the graph it checked, each use's line and each file's items to
`target/tmp/module-graph.json`.

Run `cargo fmt --all -- --check`,
`CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`,
`cargo test --offline --locked`, and build the CLI before running
`tests/lit.sh -sv tests/lit`. Preserve existing snapshots and lit expectations.
CUDA round trips run when the toolkit executables are available.
