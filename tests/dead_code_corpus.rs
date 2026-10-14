//! The instructions the dead-code analysis finds unused, across the
//! fixtures, in one reviewable snapshot; and against every Nsight Compute
//! record, that ptxas emitted no SASS for any of them, the evidence that
//! leaving them out of every total matches what runs.

use ptxroof::report::build::{AnalyzeOptions, analyze};
use ptxroof::report::measured::NcuRecord;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("fixture dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, ext, out);
        } else if path.to_string_lossy().ends_with(ext) {
            out.push(path);
        }
    }
}

#[test]
fn unused_instructions_across_the_corpus() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect(&root.join("tests/fixtures"), ".ptx", &mut files);
    files.sort();
    let mut by_opcode: BTreeMap<String, u64> = BTreeMap::new();
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
            for u in &kernel.unused {
                *by_opcode.entry(u.opcode.clone()).or_default() += 1;
                let _ = writeln!(out, "{rel}:{}\t{}\t{}", u.line, kernel.name, u.opcode);
            }
        }
    }
    let mut head = String::new();
    for (opcode, n) in &by_opcode {
        let _ = writeln!(head, "{n}\t{opcode}");
    }
    insta::assert_snapshot!(format!("{head}\n{out}"));
}

#[test]
fn no_unused_instruction_has_sass_in_any_record() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut records = Vec::new();
    collect(&root.join("tests/fixtures"), ".ncu.json", &mut records);
    records.sort();
    let (mut checked, mut kept) = (0, Vec::new());
    for rec in &records {
        let name = rec.to_string_lossy();
        let mut ptx = PathBuf::from(name.replace(".ncu.json", ".ptx"));
        if !ptx.exists() {
            let stem = name.trim_end_matches(".ncu.json");
            ptx = PathBuf::from(format!(
                "{}.ptx",
                stem.rsplit_once('.').expect("a launch tag").0
            ));
        }
        let src = fs::read_to_string(&ptx).expect("the record's PTX");
        let record: NcuRecord =
            serde_json::from_str(&fs::read_to_string(rec).expect("record readable"))
                .expect("record parses");
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
        };
        let report = analyze(&src, &ptx.display().to_string(), &opts).expect("fixture analyzes");
        let record = opts.ncu.as_ref().expect("set above");
        for kernel in report.kernels.iter().filter(|k| k.name == record.kernel) {
            for u in &kernel.unused {
                checked += 1;
                if let Some(line) = record.lines.get(&u.line) {
                    kept.push(format!(
                        "{}:{} {} {:?}",
                        ptx.display(),
                        u.line,
                        line.ptx,
                        line.sass
                    ));
                }
            }
        }
    }
    assert!(checked > 0, "no unused instruction in any record's kernel");
    assert!(
        kept.is_empty(),
        "ptxas kept unused instructions:\n{}",
        kept.join("\n")
    );
}
