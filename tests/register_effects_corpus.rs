//! How many leading operands the register-effects decoder takes every
//! instruction form in the fixtures to write, with the form's count of
//! statements, in one reviewable snapshot (`-`: not decoded). A changed
//! row is a change in which registers the tool believes an instruction
//! writes. Each written operand is a register, a selected register
//! (`%r1.h0`), a `{…}`, a `p|q`, a call's `(…)` or `_`.

use ptxroof::analysis::register_effects::{registers, split_operands, writes, written_operands};
use ptxroof::ptx::ir::{Operand, Register, Stmt};
use ptxroof::ptx::parse::parser::parse;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_ptx(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("fixture dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_ptx(&path, out);
        } else if path.extension().is_some_and(|e| e == "ptx") {
            out.push(path);
        }
    }
}

#[test]
fn written_operands_across_the_corpus() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    let mut forms: BTreeMap<(String, Option<usize>), usize> = BTreeMap::new();
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            for stmt in &k.stmts {
                let Stmt::Instr(i) = stmt else { continue };
                let written = written_operands(&m, i);
                let ops = m.operand_ids(i.operands);
                for &op in ops.iter().take(written.unwrap_or(0)) {
                    assert!(
                        match m.operand(op) {
                            Operand::Register(_)
                            | Operand::Selected { .. }
                            | Operand::VectorList { .. }
                            | Operand::ParamList { .. }
                            | Operand::MultipleDestinations { .. } => true,
                            Operand::SymbolRef(s) => m.interner.resolve(*s) == "_",
                            _ => false,
                        },
                        "{}: `{}` writes {:?}",
                        path.display(),
                        m.opcode(i),
                        m.operand(op)
                    );
                }
                *forms.entry((m.opcode(i), written)).or_default() += 1;
            }
        }
    }
    let mut out = String::new();
    for ((opcode, written), n) in &forms {
        let written = written.map_or("-".to_owned(), |w| w.to_string());
        let _ = writeln!(out, "{opcode}\t{written}\t{n}");
    }
    insta::assert_snapshot!(out);
}

/// Every register an instruction or its guard reads is written somewhere
/// in its kernel: a read of a name nothing writes is a name the parser
/// resolved to the wrong register, or a read before any write.
#[test]
fn every_register_read_is_written_in_its_kernel() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    let mut unwritten = BTreeSet::new();
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        let rel = path.strip_prefix(&root).expect("under the repo").display();
        for k in &m.kernels {
            let written: HashSet<_> = (k.stmts.iter()).flat_map(|s| writes(&m, s)).collect();
            for stmt in &k.stmts {
                let Stmt::Instr(i) = stmt else { continue };
                let guard = i.predicate.map(|p| p.reg);
                let reads = (split_operands(&m, i).1.iter()).flat_map(|&op| registers(&m, op));
                for r in reads
                    .chain(guard)
                    .filter(|&r| !written.contains(&Register::Named(r)))
                {
                    unwritten.insert(format!("{rel}: {}", m.interner.resolve(r)));
                }
            }
        }
    }
    let unwritten: Vec<_> = unwritten.into_iter().collect();
    assert!(
        unwritten.is_empty(),
        "read but never written:\n{}",
        unwritten.join("\n")
    );
}
