//! Reviewable AST snapshots for two micro fixtures.
//! Snapshots go through the symbol-resolving dumper, so they stay
//! readable — no raw indices. Review with `cargo insta review` (or
//! inspect the .snap diff directly; snapshots are committed).

use ptxroof::ptx::parse::parser::parse;
use ptxroof::ptx::print::dump;
use std::fs;
use std::path::PathBuf;

fn dump_fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let src = fs::read_to_string(&path).expect("fixture readable");
    dump(&parse(&src).expect("fixture parses"))
}

#[test]
fn single_loop_ast() {
    insta::assert_snapshot!(dump_fixture("micro/single_loop.ptx"));
}

#[test]
fn branchy_ast() {
    insta::assert_snapshot!(dump_fixture("micro/branchy.ptx"));
}
