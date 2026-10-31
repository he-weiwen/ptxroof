//! The input PTX with what the analysis says of each line appended to
//! it as a ` // ` comment.

use crate::analysis::control_flow::loops::LoopId;
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
    /// Each loop header's trip count, on the line of the header's first
    /// statement.
    fn annotations(&self, notes: &mut BTreeMap<u32, Vec<String>>) {
        for (b, block) in self.cfg.blocks.iter_enumerated() {
            let Some(l) = self.forest.block_loop[b.0 as usize] else {
                continue;
            };
            let offset = match &self.kernel.stmts[block.start] {
                Stmt::Label { offset, .. } | Stmt::Unparsed { offset, .. } => *offset,
                Stmt::Instr(i) => i.offset,
                Stmt::BranchTargets(_) => continue,
            };
            if self.forest.get(l).header == b {
                let note = format!("loop: trips {}", self.trips(l));
                notes.entry(self.lines.line(offset)).or_default().push(note);
            }
        }
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
