//! Every Nsight Compute record's comparison in one reviewable snapshot:
//! each block's verdict and counts, and each record's L2 sector lines,
//! those that differ by line. A block that loses its verdict or a line
//! that stops agreeing changes it as a contradiction would.

use ptxroof::report::build::{AnalyzeOptions, analyze};
use ptxroof::report::measured::NcuRecord;
use ptxroof::report::schema::{MeasuredSectors, SectorOutcome, Verdict};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("fixture dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.to_string_lossy().ends_with(".ncu.json") {
            out.push(path);
        }
    }
}

fn kind(s: &MeasuredSectors) -> &'static str {
    match (s.outcome, s.at_most, s.predicted.min == s.predicted.max) {
        (SectorOutcome::Differs, _, _) => "differ",
        (SectorOutcome::ExecutionsDiffer, _, _) => "not compared, its SASS ran otherwise",
        (SectorOutcome::Agrees, true, _) => "within a bound",
        (SectorOutcome::Agrees, false, true) => "exact",
        (SectorOutcome::Agrees, false, false) => "within a range",
    }
}

#[test]
fn measured_verdicts_across_the_records() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut records = Vec::new();
    collect(&root.join("tests/fixtures"), &mut records);
    collect(&root.join("tests/lit/Inputs"), &mut records);
    records.sort();
    let mut verdicts: BTreeMap<String, u64> = BTreeMap::new();
    let mut sectors: BTreeMap<&str, u64> = BTreeMap::new();
    let mut out = String::new();
    for path in &records {
        let text = fs::read_to_string(path).expect("record readable");
        let fields: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
        let record: NcuRecord = serde_json::from_str(&text).expect("record parses");
        let ptx = fields["ptx"].as_str().expect("record names its ptx");
        let src = fs::read_to_string(root.join(ptx)).expect("the record's PTX");
        let kernel = record.kernel.clone();
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
        };
        let report = analyze(&src, ptx, &opts).expect("record analyzes");
        let rel = path.strip_prefix(&root).expect("under the repo").display();
        let _ = writeln!(out, "{rel}");
        let blocks = (report.kernels.iter())
            .filter(|k| k.name == kernel)
            .flat_map(|k| k.measured.iter().flatten());
        let mut lines: BTreeMap<&str, u64> = BTreeMap::new();
        let mut differing = Vec::new();
        let mut uncompared = Vec::new();
        for b in blocks {
            let verdict = format!("{:?}", b.verdict).to_lowercase();
            *verdicts.entry(verdict.clone()).or_default() += 1;
            let _ = writeln!(
                out,
                "  {}\t{verdict}\t{} of {} with SASS, {} agreeing",
                b.name, b.with_sass, b.instructions, b.agreeing
            );
            for s in &b.sectors {
                *lines.entry(kind(s)).or_default() += 1;
                *sectors.entry(kind(s)).or_default() += 1;
                match s.outcome {
                    SectorOutcome::Differs => differing.push(s.line.to_string()),
                    SectorOutcome::ExecutionsDiffer => uncompared.push(s.line.to_string()),
                    SectorOutcome::Agrees => {}
                }
            }
        }
        if !lines.is_empty() {
            let _ = writeln!(out, "  sectors\t{lines:?}");
        }
        if !differing.is_empty() {
            let _ = writeln!(out, "  sectors differ\tlines {}", differing.join(", "));
        }
        if !uncompared.is_empty() {
            let _ = writeln!(
                out,
                "  sectors not compared\tlines {}",
                uncompared.join(", ")
            );
        }
    }
    assert_eq!(
        verdicts.get(&format!("{:?}", Verdict::Contradicted).to_lowercase()),
        None
    );
    insta::assert_snapshot!(format!(
        "blocks {verdicts:?}\nsector lines {sectors:?}\n\n{out}"
    ));
}
