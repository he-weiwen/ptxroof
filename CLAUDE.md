# Project instructions

## Keep PLAN.md and the coverage audit current on every commit

`PLAN.md` is the list of known limitations, missing features and
scope, each with its evidence. `docs/ptx-instruction-coverage.md` is
the per-instruction PTX ISA coverage audit. Before finishing any
commit, fold into it whatever the commit changes in either:

- A commit that fixes a limitation deletes its entry; one that finds
  or introduces a gap adds it, with the evidence (fixture, command,
  output). A commit that starts, re-scopes or retires a missing
  feature says so in its line. Do not add design essays, predictions
  or "verified" notes: a claim either has a test or fixture that pins
  it, or it is listed as a limitation.
- Any change to instruction handling (`src/analysis/instruction_counts/classify.rs`, the parser's
  instruction surface, the model axes in `src/analysis/instruction_counts/measurement.rs`)
  updates the audit's affected rows and its assessment sections. If
  the pinned PTX ISA version changes, re-derive the instruction
  inventory from the manual before editing rows.

PLAN.md's output tallies table is recomputed by
`tests/output_tallies.rs`, which fails when a commit moves it; rewrite
it in that commit with `PTXROOF_WRITE_PLAN=1 cargo test --test
output_tallies`.

The same-commit rule is what keeps both files trustworthy.

## Propose a fix in one shape

When proposing how to solve problems (limitations, refusals, wrong
numbers), give each item, in this order:

- The problem and its size: census rows, record lines or fixtures it
  covers, and how they were counted.
- Where: file, function and line, and the types or enum variants
  involved.
- The change, concretely: which branch or condition, and which
  existing function it reuses.
- Whether a type, a function signature or a module dependency
  changes; if one cannot be reused because `tests/architecture.rs`
  would see a cycle, say so. When a type or an interface changes,
  write out in full, as they will read after the change, every new or
  changed type definition (all fields and variants) and every new or
  changed function signature, and name what is removed.
- The tests, fixtures and snapshots whose expectations move, and the
  PLAN.md entries it touches.
- Its rough size in lines, and what it leaves unsolved.

Then close with: which items, if any, change types or interfaces;
items withdrawn since an earlier proposal, with the number that
withdrew them; the commit split and order under the commit-size rule;
and what to remeasure after each commit.

## Keep the module graph page current

`tests/architecture.rs` fails `cargo test` on a dependency cycle between
production files, and writes the graph it checked to
`target/tmp/module-graph.json`. The page
https://claude.ai/artifact/UdgsHDL74hz5rWrqa7o2Lr draws that file and is
not in the repository. After a commit that adds, removes or moves a
module, or changes what a module imports, read the artifact, replace its
`graph-data` block with the fresh `target/tmp/module-graph.json`, fix
what its banner lists as stale in its `annotations` block, and republish
it to the same URL.
