//! How often each item of the annotated PTX is exact, a bound or not
//! known, over the kernels of the Nsight Compute records but `micro/`'s
//! hand-written ones, each under its record's launch: the table PLAN.md
//! carries, recomputed. `PTXROOF_WRITE_PLAN=1` writes it there.

use ptxroof::report::annotate::annotate;
use ptxroof::report::build::AnalyzeOptions;
use ptxroof::report::measured::NcuRecord;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

const BEGIN: &str = "<!-- tests/output_tallies.rs -->\n";
const END: &str = "<!-- end tests/output_tallies.rs -->\n";

const ROWS: [&str; 24] = [
    "`.entry`: shared memory per CTA",
    "`.entry`: FLOPs of a pipe it prints",
    "`.entry`: global load bytes",
    "`.entry`: global store bytes",
    "`.entry`: shared load bytes",
    "`.entry`: shared store bytes",
    "loop: trips",
    "loop: global bytes requested per CTA",
    "loop: global bytes unique per CTA",
    "loop: global sector bytes per CTA",
    "block: threads",
    "block: CTAs",
    "block: `runs when`",
    "block: exec",
    "instruction: threads, where a guard narrows its block's",
    "instruction: exec, where it differs from its block's",
    "instruction: `unused`",
    "instruction: `not classified`",
    "instruction: `bytes ?`",
    "operand: address",
    "operand: global sectors/req",
    "operand: global sectors/req, ideal",
    "operand: shared wavefronts/req",
    "operand: shared wavefronts/req, ideal",
];

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Exact,
    Bound,
    Unknown,
    NoNumber,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    exact: u64,
    bound: u64,
    unknown: u64,
    no_number: u64,
    mbarrier: u64,
}

struct Tallies([Tally; ROWS.len()]);

impl Tallies {
    fn add(&mut self, row: &str, class: Class) {
        let i = ROWS.iter().position(|r| *r == row).expect("a row");
        let t = &mut self.0[i];
        match class {
            Class::Exact => t.exact += 1,
            Class::Bound => t.bound += 1,
            Class::Unknown => t.unknown += 1,
            Class::NoNumber => t.no_number += 1,
        }
    }
}

/// A printed count: `?` first is not known; `<=`, `>=` or a range
/// `a–b` is a bound.
fn class(v: &str) -> Class {
    let v = v.trim();
    match () {
        _ if v.starts_with('?') => Class::Unknown,
        _ if v.starts_with("<=") || v.starts_with(">=") || v.contains('–') => Class::Bound,
        _ => Class::Exact,
    }
}

fn is_threads(p: &str) -> bool {
    p.starts_with("warp ") || p.starts_with("<= warp ") || p == "no thread"
}

/// The `.entry` line's shared memory, and each pipe's FLOPs and the
/// global and shared bytes its sums print.
fn entry(parts: &[&str], t: &mut Tallies) {
    for p in parts {
        if p.starts_with("shared memory ") {
            let c = if p.ends_with("+ dynamic") {
                Class::Bound
            } else {
                Class::Exact
            };
            t.add(ROWS[0], c);
        }
        let Some((_, sums)) = p.split_once(": ").filter(|_| p.starts_with("per ")) else {
            continue;
        };
        for item in sums.split("; ") {
            if let Some((pipes, flops)) = item.split_once("flops ") {
                let pipes = pipes.trim_end().trim_end_matches(',');
                if flops == "?" {
                    let n = if pipes.is_empty() {
                        4
                    } else {
                        pipes.split(", ").count()
                    };
                    (0..n).for_each(|_| t.add(ROWS[1], Class::Unknown));
                    continue;
                }
                let terms: Vec<Class> = (flops.split(" + "))
                    .map(|term| match term.split_once(' ') {
                        Some((_, v)) => class(v),
                        None => Class::Bound,
                    })
                    .collect();
                let c = match () {
                    _ if terms.contains(&Class::Unknown) => Class::Unknown,
                    _ if terms.contains(&Class::Bound) => Class::Bound,
                    _ => Class::Exact,
                };
                t.add(ROWS[1], c);
                continue;
            }
            let Some((space, rest)) = item.split_once(' ') else {
                continue;
            };
            for io in rest.split(", ") {
                let (dir, v) = io.split_once(' ').expect("load or store and bytes");
                let row = match (space, dir) {
                    ("global", "load") => ROWS[2],
                    ("global", "store") => ROWS[3],
                    ("shared", "load") => ROWS[4],
                    ("shared", "store") => ROWS[5],
                    _ => continue,
                };
                t.add(row, class(v));
            }
        }
    }
}

/// A loop header's trips and the global bytes per CTA it prints.
fn header(trips: &str, global: Option<&str>, t: &mut Tallies) {
    let c = class(trips);
    t.add(ROWS[6], c);
    if c == Class::Unknown && trips.contains("mbarrier") {
        t.0[6].mbarrier += 1;
    }
    let Some(g) = global.and_then(|g| g.strip_prefix("global per CTA: requested ")) else {
        return;
    };
    let (requested, rest) = g.split_once(" B, unique ").expect("unique bytes");
    t.add(ROWS[7], class(requested));
    let (unique, sectors) = match rest.split_once(", sectors ") {
        Some((u, s)) => (u, Some(s)),
        None => (rest, None),
    };
    t.add(ROWS[8], class(unique));
    t.add(ROWS[9], sectors.map_or(Class::NoNumber, class));
}

/// A statement's parts: its threads and exec where they differ from its
/// block's, what is not counted, and its memory operands.
fn statement(parts: &[&str], t: &mut Tallies) {
    for p in parts {
        let c = || {
            if p.contains("<=") {
                Class::Bound
            } else {
                Class::Exact
            }
        };
        match *p {
            "unused" => t.add(ROWS[16], Class::Exact),
            _ if is_threads(p) => t.add(ROWS[14], c()),
            _ if p.starts_with("CTAs ") => {}
            _ if p.starts_with("exec ") => t.add(ROWS[15], class(&p[5..])),
            _ if p.starts_with("not classified") => t.add(ROWS[17], Class::Exact),
            _ if p.starts_with("bytes ?") => t.add(ROWS[18], Class::Exact),
            _ if p.starts_with("call (") || p.starts_with("not parsed") => {}
            _ => operand(p, t),
        }
    }
}

/// `<space> <direction> [<address>] <unit>/req <n> (ideal <m>)`, or
/// `addr ?`, or `<unit>/req ? (why)`.
fn operand(p: &str, t: &mut Tallies) {
    let mut words = p.splitn(3, ' ');
    let (space, _, rest) = (words.next(), words.next(), words.next().unwrap_or(""));
    let known = !rest.starts_with("addr ?");
    t.add(ROWS[19], if known { Class::Exact } else { Class::Unknown });
    let (count, ideal) = match space {
        Some("global" | "generic") => (ROWS[20], ROWS[21]),
        Some("shared") => (ROWS[22], ROWS[23]),
        _ => return,
    };
    let Some((_, req)) = rest.split_once("/req ") else {
        t.add(count, Class::NoNumber);
        return;
    };
    match req.split_once(" (ideal ") {
        Some((n, m)) => {
            t.add(count, class(n));
            t.add(ideal, class(m.trim_end_matches(')')));
        }
        None => t.add(count, class(req)),
    }
}

/// A block's annotation: a header's trips and bytes, its threads, CTAs,
/// `runs when`, nesting and exec, then its first statement's parts.
fn block(parts: &[&str], t: &mut Tallies) {
    let mut i = 0;
    if let Some(trips) = parts[0].strip_prefix("loop: trips ") {
        let global = parts.get(1).filter(|g| g.starts_with("global per CTA"));
        header(trips, global.copied(), t);
        i = 1 + usize::from(global.is_some());
    }
    let Some(threads) = parts.get(i).filter(|p| is_threads(p)) else {
        return statement(&parts[i..], t);
    };
    t.add(
        ROWS[10],
        if threads.contains("<=") {
            Class::Bound
        } else {
            Class::Exact
        },
    );
    i += 1;
    if threads.contains(" CTAs, other ") {
        t.add(ROWS[11], Class::Bound);
    } else if parts.get(i).is_some_and(|p| p.starts_with("CTAs ")) {
        t.add(ROWS[11], Class::Exact);
        i += 1;
    }
    if let Some(w) = parts.get(i).filter(|p| p.starts_with("runs when ")) {
        let unknown = w
            .split(' ')
            .any(|word| word.trim_matches(['(', ')']) == "?");
        t.add(
            ROWS[12],
            if unknown {
                Class::Unknown
            } else {
                Class::Exact
            },
        );
        i += 1;
    }
    if parts
        .get(i)
        .is_some_and(|p| p.starts_with("nested in ") || p.starts_with("under line "))
    {
        i += 1;
    }
    if let Some(e) = parts.get(i).and_then(|p| p.strip_prefix("exec ")) {
        t.add(ROWS[13], class(e));
        i += 1;
    }
    statement(&parts[i..], t);
}

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

fn table(t: &Tallies, kernels: usize) -> String {
    let n = |x: u64| {
        let digits = x.to_string();
        let mut s = String::new();
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i).is_multiple_of(3) {
                s.push(',');
            }
            s.push(c);
        }
        s
    };
    let mut out = format!(
        "{kernels} kernels.\n\n| Output | Count | Exact | Bound | ? | No number |\n|---|---|---|---|---|---|\n"
    );
    for (row, t) in ROWS.iter().zip(&t.0) {
        let total = t.exact + t.bound + t.unknown + t.no_number;
        let cell = |x: u64| match x {
            0 => String::new(),
            _ => format!("{} ({}%)", n(x), (100 * x + total / 2) / total),
        };
        let unknown = match t.mbarrier {
            0 => cell(t.unknown),
            m => format!("{}, {} mbarrier waits", cell(t.unknown), n(m)),
        };
        let marker = row.contains('`') && row.starts_with("instruction: `");
        let _ = match marker {
            true => writeln!(out, "| {row} | {} | | | | |", n(total)),
            false => writeln!(
                out,
                "| {row} | {} | {} | {} | {unknown} | {} |",
                n(total),
                cell(t.exact),
                cell(t.bound),
                cell(t.no_number)
            ),
        };
    }
    out
}

#[test]
fn plan_md_carries_the_output_tallies() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut records = Vec::new();
    collect(&root.join("tests/fixtures"), &mut records);
    records.retain(|p| !p.starts_with(root.join("tests/fixtures/micro")));
    records.sort();
    let mut t = Tallies([Tally::default(); ROWS.len()]);
    for path in &records {
        let text = fs::read_to_string(path).expect("record readable");
        let fields: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
        let record: NcuRecord = serde_json::from_str(&text).expect("record parses");
        let src = fs::read_to_string(root.join(fields["ptx"].as_str().expect("its ptx")))
            .expect("the record's PTX");
        let kernel = format!(".entry {}", record.kernel);
        let opts = AnalyzeOptions {
            bindings: record.bindings(),
            launch: Some(record.block),
            grid: Some(record.grid),
            ncu: Some(record),
            ..Default::default()
        };
        let annotated = annotate(&src, &opts).expect("record analyzes");
        let mut inside = false;
        for (ptx, line) in src.lines().zip(annotated.lines()) {
            if ptx.contains(".entry ") {
                let name = ptx.split_once(&kernel).map(|(_, r)| r.trim_start());
                inside = name.is_some_and(|r| r.is_empty() || r.starts_with('('));
            }
            let Some(notes) = (line.strip_prefix(ptx))
                .and_then(|n| n.strip_prefix(" // "))
                .filter(|_| inside)
            else {
                continue;
            };
            let code = ptx.split("//").next().unwrap_or_default().trim();
            let label = code.ends_with(':') && !code.contains(char::is_whitespace);
            for (i, note) in notes.split(" ; ").enumerate() {
                let parts: Vec<&str> = note.split(" | ").collect();
                if ptx.contains(".entry ") && i == 0 {
                    entry(&parts, &mut t);
                } else if label || note.starts_with('[') {
                    let mut parts = parts;
                    if let Some((_, first)) =
                        parts[0].split_once("] ").filter(|_| note.starts_with('['))
                    {
                        parts[0] = first;
                    }
                    block(&parts, &mut t);
                } else {
                    statement(&parts, &mut t);
                }
            }
        }
    }
    let table = table(&t, records.len());
    let path = root.join("PLAN.md");
    let plan = fs::read_to_string(&path).expect("PLAN.md readable");
    let (head, rest) = plan.split_once(BEGIN).expect("PLAN.md's tallies begin");
    let (old, tail) = rest.split_once(END).expect("PLAN.md's tallies end");
    if std::env::var_os("PTXROOF_WRITE_PLAN").is_some() {
        fs::write(&path, format!("{head}{BEGIN}{table}{END}{tail}")).expect("PLAN.md written");
    } else {
        assert_eq!(
            old, table,
            "PLAN.md's tallies are stale: rerun with PTXROOF_WRITE_PLAN=1"
        );
    }
}
