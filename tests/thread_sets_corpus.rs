//! The thread set of every block of every fixture, as the report
//! renders it, in one reviewable snapshot: a change here is a change
//! in which threads the tool believes execute a block.

use ptxroof::report::build::{AnalyzeOptions, analyze};
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
fn block_thread_sets_across_the_corpus() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    let mut out = String::new();
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let rel = path
            .strip_prefix(&root)
            .expect("under the repo")
            .display()
            .to_string();
        let report = analyze(&src, &rel, &AnalyzeOptions::default()).expect("fixture analyzes");
        for kernel in &report.kernels {
            for block in &kernel.blocks {
                if let Some(threads) = &block.threads {
                    let _ = writeln!(out, "{rel}\t{}\t{}\t{threads}", kernel.name, block.name);
                }
            }
        }
    }
    insta::assert_snapshot!(out);
}
