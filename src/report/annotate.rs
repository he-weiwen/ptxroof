//! The input PTX with what the analysis says of each line appended to
//! it as a ` // ` comment.

use crate::analysis::affine::affine_expr::Var;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::thread_participation::Pred;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::Stmt;
use crate::report::build::{AnalyzeError, AnalyzeOptions, KernelReportBuilder, each_kernel};
use std::collections::BTreeMap;

/// `source` with ` // ` and the annotation after each line that has one,
/// the annotations of its statements joined by ` ; `.
pub fn annotate(source: &str, opts: &AnalyzeOptions) -> Result<String, AnalyzeError> {
    let mut notes: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    each_kernel(source, opts, |builder| builder.annotations(&mut notes))?;
    let mut out = String::with_capacity(source.len());
    for (line, text) in (1..).zip(source.split_inclusive('\n')) {
        let (text, end) = text.strip_suffix('\n').map_or((text, ""), |t| (t, "\n"));
        out.push_str(text);
        if let Some(n) = notes.get(&line) {
            out.push_str(" // ");
            out.push_str(&n.join(" ; "));
        }
        out.push_str(end);
    }
    Ok(out)
}

impl KernelReportBuilder<'_> {
    /// Each block's annotation, a loop header's trip count first, on its
    /// label's line, else on its first statement's after `[<block name>] `.
    fn annotations(&self, notes: &mut BTreeMap<u32, Vec<String>>) {
        for (b, block) in self.cfg.blocks.iter_enumerated() {
            let header = self.forest.block_loop[b.0 as usize]
                .filter(|&l| self.forest.get(l).header == b)
                .map(|l| format!("loop: trips {}", self.trips(l)));
            let parts: Vec<String> = header.into_iter().chain(self.threads(b)).collect();
            let name = format!("[{}] ", self.cfg.block_name(self.module, b));
            let (offset, prefix) = match &self.kernel.stmts[block.start] {
                Stmt::Label { offset, .. } => (*offset, String::new()),
                Stmt::Unparsed { offset, .. } => (*offset, name),
                Stmt::Instr(i) => (i.offset, name),
                Stmt::BranchTargets(_) => continue,
            };
            if !parts.is_empty() {
                let note = format!("{prefix}{}", parts.join(" | "));
                notes.entry(self.lines.line(offset)).or_default().push(note);
            }
        }
    }

    /// The threads that run block `b`, by warps and lanes; without a block
    /// shape, the condition that selects them, if one does. Where they
    /// read the CTA index and the grid is known: the lanes and the CTAs'
    /// condition, or where the lanes differ by CTA the most CTAs' lanes.
    fn threads(&self, b: BlockId) -> Option<String> {
        let set = &self.counts.sets.blocks[b.0 as usize];
        let shape = self.counts.shape;
        if shape.contains(&0) {
            return self.condition(set);
        }
        let exact = set.lane_exact();
        let classes = (self.counts.grid()).and_then(|g| Some((g, set.cta_classes(shape, g)?)));
        let Some((grid, classes)) = classes else {
            return Some(warp_lanes(&set.lane_masks(shape, None), shape, exact));
        };
        let total: u64 = grid.iter().map(|&n| u64::from(n)).product();
        let running: Vec<_> = (classes.iter())
            .filter(|(m, _)| m.iter().any(|w| w.0 != 0))
            .collect();
        let n: u64 = running.iter().map(|(_, n)| n).sum();
        let most = running
            .iter()
            .copied()
            .reduce(|a, b| if b.1 > a.1 { b } else { a });
        let Some((masks, most)) = most else {
            return Some("no thread".to_owned());
        };
        let lanes = warp_lanes(masks, shape, exact);
        Some(match running.len() {
            1 => {
                let parts: Vec<String> = (cta_conjuncts(set).into_iter())
                    .filter_map(|p| self.condition(p))
                    .collect();
                format!("{lanes} | CTAs {} ({n}/{total})", parts.join(" and "))
            }
            _ => format!(
                "{lanes} in {most}/{total} CTAs, other {} CTAs differ",
                n - most
            ),
        })
    }

    /// The condition on the thread and CTA indices that selects `set`, a
    /// loop's iteration k as `iv + 1`.
    fn condition(&self, set: &Pred) -> Option<String> {
        let name = |v: &Var| match v {
            Var::Iteration(l) => format!("({} + 1)", self.iv(*l)),
            v => v.to_string(),
        };
        set.render_named(&name).map(|t| t.replace(" * ", "·"))
    }

    /// Loop `l`'s iteration from 0, `iv[<header label>]`, LLVM's canonical
    /// induction variable (llvm/include/llvm/Analysis/LoopInfo.h).
    fn iv(&self, l: LoopId) -> String {
        let header = self.forest.get(l).header;
        format!("iv[{}]", self.cfg.block_name(self.module, header))
    }

    /// Loop `l`'s trip count, or `?` and why it is not known; a product
    /// prints as `a·b`.
    fn trips(&self, l: LoopId) -> String {
        match &self.counts.trip_info.trips[l.0 as usize] {
            Ok(e) => {
                (e.bind(self.bind_map).render(&|leaf| self.leaf_name(leaf))).replace(" * ", "·")
            }
            Err(reason) => format!("? ({reason})"),
        }
    }
}

/// `warp <W> × lane <L>` per group of warps with the same lanes, joined by
/// ` ∪ `, `*` for every warp or every lane of one, after `<= ` where the
/// lanes are a bound, unless they are `exact` whenever the warp runs;
/// `no thread` when none may run. A warp's lanes are its threads' linear
/// indices modulo 32 (PTX ISA §3.1).
fn warp_lanes(masks: &[(u32, u32)], shape: [u32; 3], exact: bool) -> String {
    let threads: u64 = shape.iter().map(|&n| u64::from(n)).product();
    let full = |w: usize| u32::MAX >> (32 - (threads - 32 * w as u64).min(32));
    let mut groups: Vec<((bool, String), Vec<i64>)> = Vec::new();
    for (w, &(may, must)) in masks.iter().enumerate().filter(|(_, m)| m.0 != 0) {
        let lanes = match may == full(w) {
            true => "*".to_owned(),
            false => runs(&(0..32).filter(|l| may >> l & 1 == 1).collect::<Vec<_>>()),
        };
        let key = (may != must && !exact, lanes);
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(g) => g.1.push(w as i64),
            None => groups.push((key, vec![w as i64])),
        }
    }
    let group = |((bound, lanes), warps): &((bool, String), Vec<i64>)| {
        let warps = match warps.len() == masks.len() {
            true => "*".to_owned(),
            false => runs(warps),
        };
        format!(
            "{}warp {warps} × lane {lanes}",
            if *bound { "<= " } else { "" }
        )
    };
    match groups.is_empty() {
        true => "no thread".to_owned(),
        false => groups.iter().map(group).collect::<Vec<_>>().join(" ∪ "),
    }
}

/// The parts of a conjunction that read the CTA index.
fn cta_conjuncts(set: &Pred) -> Vec<&Pred> {
    match set {
        Pred::And(a, b) => [cta_conjuncts(a), cta_conjuncts(b)].concat(),
        p if p.varies_by_cta() => vec![p],
        _ => Vec::new(),
    }
}

/// Increasing values as `a-b` for a run of consecutive ones, `a-b:s` for
/// three or more `s` apart, else one by one, joined by `,`.
fn runs(xs: &[i64]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < xs.len() {
        let step = xs.get(i + 1).map_or(1, |x| x - xs[i]);
        let end = (i + 1..xs.len())
            .find(|&j| xs[j] - xs[j - 1] != step)
            .unwrap_or(xs.len());
        let len = if end - i == 2 && step != 1 {
            1
        } else {
            end - i
        };
        let (first, last) = (xs[i], xs[i + len - 1]);
        out.push(match (len, step) {
            (1, _) => first.to_string(),
            (_, 1) => format!("{first}-{last}"),
            _ => format!("{first}-{last}:{step}"),
        });
        i += len;
    }
    out.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_of_values_print_as_ranges_strides_and_singles() {
        assert_eq!(runs(&[0, 1, 2, 5, 6, 7]), "0-2,5-7");
        assert_eq!(runs(&[0, 8, 16, 24]), "0-24:8");
        assert_eq!(runs(&[0, 5]), "0,5");
        assert_eq!(runs(&[0, 2, 3, 4]), "0,2-4");
        assert_eq!(runs(&[3]), "3");
    }

    #[test]
    fn warps_with_the_same_lanes_print_as_one_group() {
        let shape = [80, 1, 1];
        let all = [(u32::MAX, u32::MAX), (u32::MAX, u32::MAX), (0xffff, 0xffff)];
        assert_eq!(warp_lanes(&all, shape, true), "warp * × lane *");
        let lane_0 = [(1, 1), (1, 1), (1, 0)];
        assert_eq!(
            warp_lanes(&lane_0, shape, false),
            "warp 0-1 × lane 0 ∪ <= warp 2 × lane 0"
        );
        assert_eq!(warp_lanes(&lane_0, shape, true), "warp * × lane 0");
        assert_eq!(warp_lanes(&[(0, 0); 3], shape, true), "no thread");
    }

    /// Without a block shape, a block's annotation is the condition on the
    /// thread index that selects it, a loop's iteration k = iv + 1.
    #[test]
    fn without_a_block_shape_a_block_names_its_condition() {
        let src = "\
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 p)
{
	.reg .pred %p<4>;
	.reg .b32 %r<3>;
	.reg .b64 %rd<2>;
	ld.param.u64 %rd1, [p];
	mov.u32 %r1, %tid.x;
	mov.u32 %r2, 0;
$L__BB0_1:
	add.s32 %r2, %r2, 1;
	setp.ge.u32 %p2, %r1, %r2;
	@%p2 bra $L__BB0_3;
	st.global.u32 [%rd1], %r2;
$L__BB0_3:
	setp.lt.u32 %p3, %r2, 4;
	@%p3 bra $L__BB0_1;
	setp.gt.u32 %p1, %r1, 31;
	@%p1 bra $L__BB0_5;
	st.global.u32 [%rd1], %r1;
$L__BB0_5:
	ret;
}
";
        let out = annotate(src, &AnalyzeOptions::default()).expect("annotates");
        let notes: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split_once(" // "))
            .map(|(_, n)| n)
            .collect();
        assert_eq!(
            notes,
            [
                "loop: trips 4",
                "[<block 2>] %tid.x - (iv[$L__BB0_1] + 1) < 0",
                "[<block 5>] %tid.x <= 31",
            ]
        );
    }
}
