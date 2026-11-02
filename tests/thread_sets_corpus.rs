//! Which threads the tool believes execute each block and each guarded
//! instruction of every fixture, as the annotated PTX says: the threads,
//! the CTAs and `runs when`, in one reviewable snapshot; a change here is
//! a change in which threads the tool believes execute an instruction.

use ptxroof::report::annotate::annotate;
use ptxroof::report::build::AnalyzeOptions;
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

/// Part `p` of an annotation says which threads run its line: it is none
/// of a loop's, a count, a link, a memory operand or what is not counted.
fn says_which_threads(p: &str) -> bool {
    const OTHER: [&str; 15] = [
        "loop: ",
        "exec ",
        "nested in ",
        "under line ",
        "unused",
        "not classified",
        "not parsed",
        "bytes ?",
        "call (",
        "global ",
        "shared ",
        "shared::cluster ",
        "local ",
        "param ",
        "const ",
    ];
    !OTHER.iter().any(|o| p.starts_with(o))
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
        let rel = path.strip_prefix(&root).expect("under the repo").display();
        let annotated = annotate(&src, &AnalyzeOptions::default()).expect("fixture analyzes");
        for ((line, ptx), text) in (1..).zip(src.lines()).zip(annotated.lines()) {
            let Some(notes) = text.strip_prefix(ptx).and_then(|t| t.strip_prefix(" // ")) else {
                continue;
            };
            if ptx.contains(".entry ") {
                continue;
            }
            for note in notes.split(" ; ") {
                let parts: Vec<&str> = (note.split(" | "))
                    .map(|p| {
                        p.split_once("] ")
                            .filter(|_| p.starts_with("[<"))
                            .map_or(p, |x| x.1)
                    })
                    .filter(|p| says_which_threads(p))
                    .collect();
                if !parts.is_empty() {
                    let _ = writeln!(out, "{rel}:{line}\t{}", parts.join(" | "));
                }
            }
        }
    }
    insta::assert_snapshot!(out);
}
