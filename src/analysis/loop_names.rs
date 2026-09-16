//! Source-derived identities and display names for loops.
//!
//! A loop is named by the source line of its back-edge branch — for a
//! `for` loop nvcc attributes the increment/compare/branch to the
//! `for` statement's line, so the latch is the one place in the body
//! that reliably carries the loop's own line (first body lines vary
//! with hoisting, e.g. k5's outer loop starts with a sunk line-17
//! computation). Fallbacks, in order: any other non-line-0 location in
//! the latch block, the header block's first non-line-0 location, the
//! raw label. Line 0 is PTX for "no source line" and never names
//! anything.
//!
//! File paths are reduced to their basename: absolute fixture paths
//! are machine-specific, and the basename is what a human greps for.
//!

use crate::analysis::control_flow::graph::Cfg;
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::ptx::ir::{Kernel, Module, SourceLoc};
use crate::support::paths::basename;

/// Human-readable identity of one loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopName {
    /// Source file basename + line, when debug info names the loop.
    pub file: Option<String>,
    pub line: Option<u32>,
    /// The header block's label (always present in compiler output;
    /// synthesized from the block id otherwise).
    pub label: String,
    /// What reports print: `file:line`, else the label.
    pub display: String,
}

/// Name every loop in the forest; indexed by `LoopId`.
pub fn loop_names(
    module: &Module,
    kernel: &Kernel,
    cfg: &Cfg,
    forest: &LoopForest,
) -> Vec<LoopName> {
    (0..forest.loops.len() as u32)
        .map(|i| loop_name(module, kernel, cfg, forest, LoopId(i)))
        .collect()
}

fn loop_name(
    module: &Module,
    kernel: &Kernel,
    cfg: &Cfg,
    forest: &LoopForest,
    id: LoopId,
) -> LoopName {
    let l = forest.get(id);
    let label = cfg.block_name(module, l.header);

    // 1. The back-edge branch's location (last instruction of a latch).
    let latch_branch_loc = l
        .latches
        .iter()
        .filter_map(|&latch| cfg.instrs(kernel, latch).last())
        .filter_map(|i| i.loc)
        .find(|loc| loc.line != 0);
    // 2. Any other location in a latch block, scanning backwards.
    let latch_any_loc = || {
        l.latches.iter().find_map(|&latch| {
            let instrs: Vec<_> = cfg.instrs(kernel, latch).collect();
            instrs
                .iter()
                .rev()
                .filter_map(|i| i.loc)
                .find(|loc| loc.line != 0)
        })
    };
    // 3. The header block's first location.
    let header_loc = || {
        cfg.instrs(kernel, l.header)
            .filter_map(|i| i.loc)
            .find(|loc| loc.line != 0)
    };

    let loc: Option<SourceLoc> = latch_branch_loc.or_else(latch_any_loc).or_else(header_loc);
    match loc {
        Some(loc) => {
            let file = basename(module.file_path(loc.file).unwrap_or("<unknown file>"));
            let display = format!("{file}:{}", loc.line);
            LoopName {
                file: Some(file),
                line: Some(loc.line),
                label,
                display,
            }
        }
        None => LoopName {
            file: None,
            line: None,
            display: label.clone(),
            label,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::{build_cfg, loop_forest};
    use crate::ptx::parse::parser::parse;

    fn names_of(body_and_trailer: &str) -> Vec<LoopName> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{body_and_trailer}"
        );
        let module = parse(&src).expect("test source parses");
        let kernel = &module.kernels[0];
        let cfg = build_cfg(&module, kernel);
        let forest = loop_forest(&cfg);
        loop_names(&module, kernel, &cfg, &forest)
    }

    #[test]
    fn loop_named_by_latch_branch_line() {
        let names = names_of(
            "mov.u32 %r1, 0;\n$L__L:\n.loc 1 9 5\nadd.s32 %r1, %r1, 1;\n\
             .loc 1 7 3\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;\n}\n\
             .file 1 \"/abs/path/kern.cu\"\n",
        );
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].display, "kern.cu:7");
        assert_eq!(names[0].file.as_deref(), Some("kern.cu"));
        assert_eq!(names[0].label, "$L__L");
    }

    #[test]
    fn line_zero_is_skipped_in_favor_of_earlier_latch_line() {
        let names = names_of(
            "mov.u32 %r1, 0;\n$L__L:\n.loc 1 9 5\nadd.s32 %r1, %r1, 1;\n\
             .loc 1 0 3\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;\n}\n\
             .file 1 \"kern.cu\"\n",
        );
        assert_eq!(names[0].display, "kern.cu:9");
    }

    #[test]
    fn no_loc_falls_back_to_label() {
        let names = names_of(
            "mov.u32 %r1, 0;\n$L__L:\nadd.s32 %r1, %r1, 1;\n\
             setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra $L__L;\nret;\n}\n",
        );
        assert_eq!(names[0].display, "$L__L");
        assert_eq!(names[0].file, None);
    }
}
