//! Corpus-wide parse checks.
//!
//! 1. Every fixture parses with zero `Stmt::Unparsed` statements, or is
//!    listed in tests/parse-allowlist.txt — additions there are named
//!    holes in frontend coverage and require review.
//! 2. dump → reparse → dump is idempotent on every fixture: the
//!    dump is a fixed point for this frontend. Assembler acceptance is
//!    checked separately by tests/run.py with a round-trip allowlist.
//! 3. The k2/k5 param tables match the signatures nvcc emits — pinned
//!    here because `--bind idx:name=value` (PR 12) addresses params by
//!    these positions.

use ptxroof::ptx::ir::{Module, Stmt};
use ptxroof::ptx::parse::parser::parse;
use ptxroof::ptx::print::dump;
use std::fs;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

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

fn corpus() -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    collect_ptx(&fixtures_dir(), &mut files);
    files.sort();
    assert!(files.len() >= 8, "fixture corpus missing?");
    files
        .into_iter()
        .map(|p| {
            let src = fs::read_to_string(&p).expect("fixture readable");
            (p, src)
        })
        .collect()
}

fn unparsed_count(module: &Module) -> usize {
    module
        .kernels
        .iter()
        .flat_map(|k| &k.stmts)
        .filter(|s| matches!(s, Stmt::Unparsed { .. }))
        .count()
}

fn allowlist() -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parse-allowlist.txt");
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

#[test]
fn every_fixture_parses_with_zero_unparsed_statements() {
    let allow = allowlist();
    for (path, src) in corpus() {
        let rel = path.strip_prefix(fixtures_dir()).expect("under fixtures");
        let module = parse(&src).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let unparsed = unparsed_count(&module);
        if unparsed > 0 && !allow.iter().any(|a| Path::new(a) == rel) {
            panic!(
                "{}: {unparsed} unparsed statement(s) and not in parse-allowlist.txt",
                rel.display()
            );
        }
    }
}

#[test]
fn dump_reparse_dump_is_idempotent_on_every_fixture() {
    let allow = allowlist();
    for (path, src) in corpus() {
        let rel = path.strip_prefix(fixtures_dir()).expect("under fixtures");
        if allow.iter().any(|a| Path::new(a) == rel) {
            continue;
        }
        let d1 = dump(&parse(&src).unwrap_or_else(|e| panic!("{}: {e}", path.display())));
        let m2 = parse(&d1)
            .unwrap_or_else(|e| panic!("{}: canonical dump does not reparse: {e}", path.display()));
        let d2 = dump(&m2);
        assert_eq!(d1, d2, "{}: dump is not a fixed point", path.display());
    }
}

#[test]
fn k2_and_k5_param_tables_match_nvcc_signatures() {
    let check = |fixture: &str, kernel_prefix: &str, expected: &[(&str, usize)]| {
        let src = fs::read_to_string(fixtures_dir().join(fixture)).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        let k = m
            .kernels
            .iter()
            .find(|k| m.interner.resolve(k.name).starts_with(kernel_prefix))
            .expect("kernel present");
        let got: Vec<(&str, usize)> = k
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| (m.interner.resolve(p.ty), i))
            .collect();
        let want: Vec<(&str, usize)> = expected.to_vec();
        assert_eq!(got, want, "{fixture} param table");
    };
    // int M, int N, int K, float alpha, const half* A, const half* B,
    // float beta, half* C  — both ladder kernels share the signature.
    let sig = [
        ("u32", 0),
        ("u32", 1),
        ("u32", 2),
        ("f32", 3),
        ("u64", 4),
        ("u64", 5),
        ("f32", 6),
        ("u64", 7),
    ];
    check("k2/k2.sm_80.ptx", "_Z15hgemm_coalesced", &sig);
    check("k5/k5.sm_80.ptx", "_Z20hgemm_2d_blocktiling", &sig);
}
