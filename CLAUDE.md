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

The same-commit rule is what keeps both files trustworthy.

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
