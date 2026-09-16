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
