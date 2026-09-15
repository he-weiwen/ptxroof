//! Text view of the result tree. Renders the same
//! structs `--json` serializes — the two views cannot disagree.
//!
//! Conventions: every static quantity is labeled `[static]` once per
//! section header (bet 3: a lone static number without its provenance
//! label is a half-truth); `at_most` counts render with a `<=` prefix,
//! `at_least` ones with a `+ unknown` suffix;
//! zero rows are skipped in flop/byte tables but unknowns are always
//! printed, even (especially) when present.

use super::tree::*;
use std::collections::BTreeMap;
use std::fmt::Write;

pub fn render(report: &Report) -> String {
    let mut out = String::new();
    let w = &mut out;
    let _ = writeln!(w, "ptxroof analyze [static] — {}", report.input);
    if !report.bindings.is_empty() {
        let binds: Vec<String> = report
            .bindings
            .iter()
            .map(|b| format!("param {} ({}) = {}", b.param, b.name, b.value))
            .collect();
        let _ = writeln!(w, "bindings: {}", binds.join(", "));
    }

    for k in &report.kernels {
        let _ = writeln!(w);
        let _ = writeln!(w, "kernel {}", k.demangled);
        if k.demangled != k.name {
            let _ = writeln!(w, "  mangled: {}", k.name);
        }
        let params: Vec<String> = k
            .params
            .iter()
            .map(|p| format!("{}:{}", p.index, p.ty))
            .collect();
        let _ = writeln!(w, "  params: {}", params.join(" "));
        render_blocks(w, &k.blocks);
        if let Some(most) = &k.most_instructions_loop {
            let _ = writeln!(w, "  loop with the most instructions (static): {most}");
        }
        if let Some(l) = &k.launch {
            let (bound, note) = if l.exact {
                ("", "")
            } else {
                ("<= ", " — a maximum, not the launch")
            };
            let _ = writeln!(
                w,
                "  block size: {bound}{} threads ({}x{}x{} from {}{note})",
                l.threads, l.block[0], l.block[1], l.block[2], l.source
            );
        }
        let sm = &k.shared_memory;
        if sm.static_bytes > 0 || sm.dynamic {
            let dyn_note = if sm.dynamic {
                " + dynamic (set at launch)"
            } else {
                ""
            };
            let _ = writeln!(
                w,
                "  shared memory [static]: {} B per CTA{dyn_note}",
                sm.static_bytes
            );
        }
        if k.ranking.len() > 1 {
            let _ = writeln!(w, "  loops by instructions executed (static):");
            for (i, r) in k.ranking.iter().enumerate() {
                let _ = writeln!(
                    w,
                    "    {}. {}  ({} instructions)",
                    i + 1,
                    r.loop_name,
                    r.instructions
                );
            }
        }

        for l in &k.loops {
            render_loop(w, l, 1);
        }
        render_accesses(w, "  ", "accesses outside loops", &k.accesses);

        let _ = writeln!(w, "  totals [static]:");
        render_aggregates(w, &k.totals, "    ");
        if let Some(per_cta) = &k.totals_per_cta {
            let _ = writeln!(w, "  totals per CTA [static]:");
            render_aggregates(w, per_cta, "    ");
        }

        if k.unknowns.is_empty() {
            let _ = writeln!(w, "  unknowns: none");
        } else {
            let _ = writeln!(w, "  unknowns:");
            for u in &k.unknowns {
                let count = u.count.map(|c| format!(" x{c}")).unwrap_or_default();
                let _ = writeln!(w, "    {}{count} — {}", u.what, u.reason);
            }
        }
    }

    let _ = writeln!(w);
    for (metric, f) in &report.coverage {
        let pct = if f.den == 0 {
            100.0
        } else {
            100.0 * f.num as f64 / f.den as f64
        };
        let _ = writeln!(w, "coverage: {metric} {pct:.1}% ({}/{})", f.num, f.den);
    }
    out
}

/// One margin string per block row drawing every non-fallthrough edge
/// as a vertical line between its two rows, `objdump
/// --visualize-jumps` style: `/` opens the line on its top row, `\`
/// closes it on the bottom row, `>` marks the target row, `<->` is a
/// self edge. Shorter edges take the columns nearest the text; a
/// horizontal overwrites any vertical it crosses.
fn margins(blocks: &[BlockInfo]) -> Vec<String> {
    let row: BTreeMap<&str, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.as_str(), i))
        .collect();
    let row = &row;
    let mut edges: Vec<(usize, usize)> = blocks
        .iter()
        .enumerate()
        .flat_map(|(src, b)| b.successors.iter().map(move |s| (src, row[s.as_str()])))
        .filter(|&(src, dst)| dst != src + 1)
        .collect();
    edges.sort_by_key(|&(src, dst)| src.abs_diff(dst));
    let mut columns: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut placed: Vec<(usize, usize, usize)> = Vec::new();
    for (src, dst) in edges {
        let (lo, hi) = (src.min(dst), src.max(dst));
        let free = |col: &Vec<(usize, usize)>| col.iter().all(|&(a, b)| hi < a || b < lo);
        let col = columns.iter().position(free).unwrap_or_else(|| {
            columns.push(Vec::new());
            columns.len() - 1
        });
        columns[col].push((lo, hi));
        placed.push((src, dst, col));
    }
    let width = columns.len() + 2;
    let rank = |c: char| match c {
        '/' | '\\' | '<' => 4,
        '>' => 3,
        '-' => 2,
        '|' => 1,
        _ => 0,
    };
    let mut rows = vec![vec![' '; width]; blocks.len()];
    let mut put = |r: usize, x: usize, c: char| {
        if rank(c) > rank(rows[r][x]) {
            rows[r][x] = c;
        }
    };
    for (src, dst, col) in placed {
        let (lo, hi) = (src.min(dst), src.max(dst));
        let x = columns.len() - 1 - col;
        for r in lo..=hi {
            let glyph = match (r == lo, r == hi) {
                (true, true) => '<',
                (true, false) => '/',
                (false, true) => '\\',
                (false, false) => '|',
            };
            put(r, x, glyph);
            if glyph == '|' {
                continue;
            }
            for i in x + 1..width {
                put(r, i, '-');
            }
            if r == dst {
                put(r, width - 1, '>');
            }
        }
    }
    rows.into_iter().map(String::from_iter).collect()
}

fn render_blocks(w: &mut String, blocks: &[BlockInfo]) {
    let _ = writeln!(
        w,
        "  blocks (program order; loops are named by their header block):"
    );
    let margins = margins(blocks);
    let blank = " ".repeat(margins.first().map_or(0, String::len));
    let with_threads = blocks.iter().any(|b| b.threads.is_some());
    let mut header: Vec<String> = ["block", "lines", "instrs", "successors", "loop"]
        .map(String::from)
        .to_vec();
    if with_threads {
        header.insert(3, "threads".to_owned());
    }
    let rows: Vec<Vec<String>> = blocks
        .iter()
        .map(|b| {
            let role = b.r#loop.as_ref().map_or(String::new(), |l| {
                let flags: Vec<&str> = [(l.header, "header"), (l.latch, "latch")]
                    .into_iter()
                    .filter_map(|(on, name)| on.then_some(name))
                    .collect();
                match flags.is_empty() {
                    true => l.name.clone(),
                    false => format!("{} ({})", l.name, flags.join(", ")),
                }
            });
            let mut row = vec![
                b.name.clone(),
                b.lines
                    .clone()
                    .unwrap_or_else(|| "(no line info)".to_owned()),
                b.instructions.to_string(),
                match b.successors.is_empty() {
                    true => "(end)".to_owned(),
                    false => b.successors.join(", "),
                },
                role,
            ];
            if with_threads {
                row.insert(3, b.threads.clone().unwrap_or_default());
            }
            row
        })
        .collect();
    let width = |col: usize| {
        std::iter::once(&header)
            .chain(&rows)
            .map(|r| r[col].len())
            .max()
            .unwrap_or(0)
    };
    let widths: Vec<usize> = (0..header.len() - 1).map(width).collect();
    let margin_of = std::iter::once(&blank).chain(&margins);
    for (m, r) in margin_of.zip(std::iter::once(&header).chain(&rows)) {
        let mut line = format!("    {m}");
        for (i, cell) in r.iter().enumerate() {
            if i + 1 == r.len() {
                line.push_str(&format!("  {cell}"));
            } else if i == 2 {
                line.push_str(&format!("  {cell:>w$}", w = widths[i]));
            } else {
                line.push_str(&format!("  {cell:<w$}", w = widths[i]));
            }
        }
        let _ = writeln!(w, "{}", line.trim_end());
    }
}

fn render_loop(w: &mut String, l: &LoopNode, depth: usize) {
    let pad = "  ".repeat(depth);
    let unroll = match &l.unroll {
        Some(u) => format!("  [unrolled x{}, remainder: {}]", u.factor, u.remainder),
        None => String::new(),
    };
    let _ = writeln!(w, "{pad}loop {} ({}){unroll}", l.name, l.label);
    match (&l.trips.expr, &l.trips.unknown) {
        (Some(e), _) => {
            let _ = writeln!(w, "{pad}  trips = {e}");
        }
        (None, Some(reason)) => {
            let _ = writeln!(w, "{pad}  trips = unknown: {reason}");
        }
        _ => {}
    }
    let _ = writeln!(w, "{pad}  per iteration:");
    render_aggregates(w, &l.per_iteration, &format!("{pad}    "));
    render_accesses(w, &format!("{pad}  "), "accesses", &l.accesses);
    if let Some(b) = &l.global_bytes_per_cta {
        let bound = if b.at_most { "<= " } else { "" };
        let _ = writeln!(
            w,
            "{pad}  global bytes per CTA over the loop's own blocks: requested {bound}{} B, unique {bound}{} B",
            b.requested, b.unique
        );
    }
    for child in &l.loops {
        render_loop(w, child, depth + 1);
    }
}

fn intensity(ai: &Intensity) -> String {
    let sign = match ai.bound {
        Bound::Exact => "=",
        Bound::AtLeast => ">=",
        Bound::AtMost => "<=",
    };
    format!("{sign} {}", ai.value)
}

/// One row per memory operand: site, opcode, what moves, and where it
/// points or why that is unknown.
fn render_accesses(w: &mut String, pad: &str, title: &str, rows: &[Access]) {
    if rows.is_empty() {
        return;
    }
    let _ = writeln!(w, "{pad}{title}:");
    let range = |r: &crate::footprint::Range| {
        if r.min == r.max {
            r.min.to_string()
        } else {
            format!("{}–{}", r.min, r.max)
        }
    };
    let cols: Vec<[String; 6]> = rows
        .iter()
        .map(|a| {
            let bytes = a.bytes.map(|b| format!(" {b} B")).unwrap_or_default();
            let pred = if a.predicated { " (predicated)" } else { "" };
            let via = if a.path == "shared memory" {
                String::new()
            } else {
                format!(" via {}", a.path)
            };
            let what = format!("{} {}{bytes}{via}{pred}", a.space, a.direction);
            let plural = |r: &crate::footprint::Range, one: &str, many: &str| {
                format!("{} {}", range(r), if r.max == 1 { one } else { many })
            };
            let warp = match (
                &a.sectors_per_request,
                &a.lines_per_request,
                &a.footprint_unknown,
            ) {
                (Some(s), Some(l), _) => format!(
                    "warp: {}, {}",
                    plural(s, "sector", "sectors"),
                    plural(l, "line", "lines")
                ),
                (_, _, Some(why)) => format!("warp: ? ({why})"),
                _ => String::new(),
            };
            let addr = match (&a.address, &a.unknown) {
                (Some(x), _) => format!("[{x}]"),
                (None, Some(why)) => format!("unknown: {why}"),
                (None, None) => String::new(),
            };
            let reuse = a
                .reuse
                .iter()
                .map(|r| match &r.stride {
                    Some(s) if s.starts_with('-') => format!("k[{}]: {s} B/iter", r.r#loop),
                    Some(s) if s.parse::<i64>().is_ok() => format!("k[{}]: +{s} B/iter", r.r#loop),
                    Some(s) => format!("k[{}]: {s} B/iter", r.r#loop),
                    None => format!("k[{}]: invariant", r.r#loop),
                })
                .collect::<Vec<_>>()
                .join(", ");
            [a.site.clone(), a.opcode.clone(), what, warp, addr, reuse]
        })
        .collect();
    let width = |i: usize| cols.iter().map(|c| c[i].len()).max().unwrap_or(0);
    let (w0, w1, w2, w3, w4) = (width(0), width(1), width(2), width(3), width(4));
    for c in &cols {
        let line = format!(
            "{pad}  {:<w0$}  {:<w1$}  {:<w2$}  {:<w3$}  {:<w4$}  {}",
            c[0], c[1], c[2], c[3], c[4], c[5]
        );
        let _ = writeln!(w, "{}", line.trim_end());
    }
}

fn describe(at_most: bool, at_least: bool) -> &'static str {
    match (at_most, at_least) {
        (false, false) => "exact",
        (true, false) => "an upper bound",
        (false, true) => "a lower bound",
        (true, true) => "bounded in neither direction",
    }
}

/// Why AI(global) is missing although flops and global bytes are both
/// constants: the directions the two sides are known in.
fn unbounded_note(a: &Aggregates) -> Option<String> {
    let flops: Vec<&Count> = [&a.flops, &a.tensor_flops, &a.sfu_flops]
        .iter()
        .map(|t| &t["total"])
        .collect();
    let g = &a.bytes["global"];
    let bytes = g.load.expr.parse::<i64>().ok()? + g.store.expr.parse::<i64>().ok()?;
    if bytes == 0 || flops.iter().any(|c| c.expr.parse::<i64>().is_err()) {
        return None;
    }
    let f = describe(
        flops.iter().any(|c| c.at_most),
        flops.iter().any(|c| c.at_least),
    );
    let b = describe(
        g.load.at_most || g.store.at_most,
        g.load.at_least || g.store.at_least,
    );
    Some(format!("flops are {f}, global bytes are {b}"))
}

fn count(c: &Count, unit: &str) -> String {
    let bound = if c.at_most { "<= " } else { "" };
    let unknown = if c.at_least { " + unknown" } else { "" };
    format!("{bound}{}{unit}{unknown}", c.expr)
}

fn render_flops(w: &mut String, pad: &str, label: &str, table: &BTreeMap<String, Count>) {
    let total = &table["total"];
    if total.expr == "0" && !total.at_least {
        return;
    }
    let by_precision: Vec<String> = table
        .iter()
        .filter(|(k, v)| k.as_str() != "total" && v.expr != "0")
        .map(|(k, v)| format!("{k} {}", count(v, "")))
        .collect();
    let detail = if by_precision.is_empty() {
        String::new()
    } else {
        format!("  ({})", by_precision.join(", "))
    };
    let _ = writeln!(w, "{pad}{label} = {}{detail}", count(total, ""));
}

/// Kinds in descending count, opcodes beneath each kind likewise:
/// counts that grow with a parameter first, by leading coefficient,
/// then constants by value.
fn render_instructions(w: &mut String, pad: &str, i: &InstructionCounts) {
    if i.total.expr == "0" {
        return;
    }
    let _ = writeln!(w, "{pad}instructions = {}", count(&i.total, ""));
    let by_count = |a: &Count, b: &Count| rank(&b.expr).cmp(&rank(&a.expr));
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut kinds: Vec<_> = i.by_kind.iter().collect();
    kinds.sort_by(|a, b| by_count(&a.1.total, &b.1.total));
    for (kind, k) in kinds {
        rows.push((kind.clone(), count(&k.total, "")));
        let mut opcodes: Vec<_> = k.opcodes.iter().collect();
        opcodes.sort_by(|a, b| by_count(a.1, b.1));
        rows.extend(
            opcodes
                .into_iter()
                .map(|(o, n)| (format!("  {o}"), count(n, ""))),
        );
    }
    let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    let count_width = rows.iter().map(|(_, c)| c.len()).max().unwrap_or(0);
    for (name, c) in rows {
        let _ = writeln!(w, "{pad}  {name:<name_width$}  {c:>count_width$}");
    }
}

/// (grows with a parameter, coefficient of the first term): the number
/// the expression starts with, or 1 when it starts with a symbol.
fn rank(expr: &str) -> (bool, i64) {
    let digits: String = expr.chars().take_while(char::is_ascii_digit).collect();
    (
        expr.contains(char::is_alphabetic),
        digits.parse().unwrap_or(1),
    )
}

fn render_aggregates(w: &mut String, a: &Aggregates, pad: &str) {
    render_instructions(w, pad, &a.instructions);
    render_flops(w, pad, "flops", &a.flops);
    render_flops(w, pad, "tensor flops", &a.tensor_flops);
    render_flops(w, pad, "sfu flops", &a.sfu_flops);
    for (space, d) in &a.bytes {
        let zero = |c: &Count| c.expr == "0" && !c.at_least;
        if zero(&d.load) && zero(&d.store) {
            continue;
        }
        let _ = writeln!(
            w,
            "{pad}{space} bytes: load {}, store {}",
            count(&d.load, " B"),
            count(&d.store, " B")
        );
    }
    if a.conversions.expr != "0" || a.conversions.at_least {
        let _ = writeln!(w, "{pad}conversions = {}", count(&a.conversions, ""));
    }
    match a.ai_global {
        Some(ai) => {
            let _ = writeln!(w, "{pad}AI(global) {} flop/B", intensity(&ai));
        }
        None => {
            if let Some(note) = unbounded_note(a) {
                let _ = writeln!(w, "{pad}AI(global): not bounded ({note})");
            }
        }
    }
    if !a.unrolled_source_lines.is_empty() {
        let lines: Vec<String> = a
            .unrolled_source_lines
            .iter()
            .map(|(l, n)| format!("{l} x{n}"))
            .collect();
        let _ = writeln!(w, "{pad}unrolled source lines: {}", lines.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_render_their_bounds() {
        let c = |at_most, at_least| Count {
            expr: "8".into(),
            at_most,
            at_least,
        };
        assert_eq!(count(&c(false, false), " B"), "8 B");
        assert_eq!(count(&c(true, false), ""), "<= 8");
        assert_eq!(count(&c(false, true), " B"), "8 B + unknown");
        assert_eq!(count(&c(true, true), ""), "<= 8 + unknown");
    }

    fn blocks(edges: &[&[usize]]) -> Vec<BlockInfo> {
        edges
            .iter()
            .enumerate()
            .map(|(i, succs)| BlockInfo {
                name: format!("b{i}"),
                lines: None,
                instructions: 1,
                threads: None,
                successors: succs.iter().map(|s| format!("b{s}")).collect(),
                r#loop: None,
            })
            .collect()
    }

    #[test]
    fn fallthrough_only_draws_nothing() {
        assert_eq!(margins(&blocks(&[&[1], &[2], &[]])), ["  ", "  ", "  "]);
    }

    #[test]
    fn forward_skip_opens_at_the_source_and_points_at_the_target() {
        let m = margins(&blocks(&[&[2, 1], &[2], &[]]));
        assert_eq!(m, ["/--", "|  ", "\\->"]);
    }

    #[test]
    fn back_edge_points_at_the_header_above_and_a_self_edge_is_one_row() {
        let m = margins(&blocks(&[&[1], &[1, 2], &[2, 3], &[]]));
        assert_eq!(m, ["   ", "<->", "<->", "   "]);
        let m = margins(&blocks(&[&[1], &[2], &[1, 3], &[]]));
        assert_eq!(m, ["   ", "/->", "\\--", "   "]);
    }

    #[test]
    fn shorter_edges_sit_nearer_the_text_and_horizontals_cross_verticals() {
        let m = margins(&blocks(&[&[4, 1], &[2], &[1, 3], &[5, 4], &[5], &[]]));
        assert_eq!(m, ["/---", "|/->", "|\\--", "|/--", "\\-->", " \\->"]);
    }
}
