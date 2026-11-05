//! Corpus-wide classification coverage: every
//! instruction in every fixture classifies non-Unknown, or its
//! mnemonic is listed in tests/classify-allowlist.txt. This is the
//! check that makes "useful in the majority of cases" a tested number:
//! when a new toolchain or kernel family introduces an idiom we don't
//! classify, this names it before any user files a bug.

use ptxroof::analysis::instruction_counts::classify::{InstructionCategory, classify};
use ptxroof::analysis::instruction_counts::measurement::{MeasureKind, Pipe};
use ptxroof::ptx::ir::Stmt;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::ptx::register_effects::is_pure;
use std::collections::BTreeMap;
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
fn every_corpus_instruction_classifies_or_is_allowlisted() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let allow: Vec<String> = fs::read_to_string(root.join("tests/classify-allowlist.txt"))
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();

    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    assert!(files.len() >= 8, "fixture corpus missing?");

    // mnemonic -> count of Unknown classifications, across the corpus
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0usize;
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            for stmt in &k.stmts {
                if let Stmt::Instr(i) = stmt {
                    total += 1;
                    if classify(&m, k, i).category == InstructionCategory::Unknown {
                        let name = i.mnemonic.name().to_owned();
                        *unknown.entry(name).or_default() += 1;
                    }
                }
            }
        }
    }
    assert!(
        total > 1000,
        "corpus suspiciously small: {total} instructions"
    );

    let violations: Vec<String> = unknown
        .iter()
        .filter(|(name, _)| !allow.contains(name))
        .map(|(name, count)| format!("{name} (x{count})"))
        .collect();
    assert!(
        violations.is_empty(),
        "unclassified mnemonics not in classify-allowlist.txt: {}",
        violations.join(", ")
    );
}

/// An instruction `is_pure` takes as pure moves no memory and is
/// classified: it contributes only flops on the CUDA cores or the SFU.
#[test]
fn a_pure_instruction_contributes_only_cuda_core_or_sfu_flops() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    let mut other: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            for stmt in &k.stmts {
                let Stmt::Instr(i) = stmt else { continue };
                let flops = (classify(&m, k, i).contributions.iter()).all(|c| {
                    matches!(
                        c.kind,
                        MeasureKind::Flops {
                            pipe: Pipe::CudaCore | Pipe::Sfu,
                            ..
                        }
                    )
                });
                if is_pure(i.mnemonic) && !flops {
                    *other.entry(m.opcode(i)).or_default() += 1;
                }
            }
        }
    }
    assert!(
        other.is_empty(),
        "pure instructions with other contributions: {other:?}"
    );
}
