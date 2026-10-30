//! Every loop's trip count (`super::affine_analysis`), and nvcc's unroll
//! main+remainder pairs — sibling loops on the same source line whose
//! trips match `(X − X mod c)/c` and `X mod c` — linked into one logical
//! loop with factor c.

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::loop_names::LoopName;
use crate::analysis::uniformity_info::UniformityInfo;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Kernel, Module};
use std::collections::HashMap;

/// Trip count of one loop: an expression, or a named reason there
/// isn't one.
pub type TripCount = Result<SymExpr, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnrollPair {
    pub main: LoopId,
    pub remainder: LoopId,
    pub factor: i64,
}

#[derive(Debug)]
pub struct TripCountResults {
    /// Indexed by `LoopId`.
    pub trips: Vec<TripCount>,
    /// Indexed by `LoopId`: the block each count is per execution of, the
    /// loop's guard branch or its header.
    pub entries: Vec<BlockId>,
    /// Indexed by `LoopId`: for a `!=` latch, the distance its counter
    /// covers and its step; the count holds where the step divides the
    /// distance into a positive count.
    pub distances: Vec<Option<(SymExpr, i64)>>,
    pub unroll_pairs: Vec<UnrollPair>,
}

/// Every loop's trip count, its latch traced under `bindings` (`--bind`,
/// the launch shape) where a constant is needed.
pub fn trip_counts(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    names: &[LoopName],
    bindings: &HashMap<SymbolicConstant, i64>,
) -> TripCountResults {
    let data = KernelAnalyses::new(module, kernel, cfg, forest);
    let uniformity = UniformityInfo::new(&data);
    trip_counts_from(
        &AffineAnalysis::new(&data, &uniformity).with_bindings(bindings),
        names,
    )
}

/// Every loop's trip count under `affine`'s bindings.
pub(crate) fn trip_counts_from(affine: &AffineAnalysis, names: &[LoopName]) -> TripCountResults {
    let forest = affine.data.forest;
    let (mut trips, mut entries, mut distances) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..forest.loops.len() as u32 {
        let (t, entry, distance) = match affine.loop_trip_count(LoopId(i)) {
            Ok(t) => (Ok(t.count), t.entry, t.distance),
            Err(e) => (Err(e), forest.get(LoopId(i)).header, None),
        };
        trips.push(t);
        entries.push(entry);
        distances.push(distance);
    }

    // -- unroll main+remainder linking ----------------------------------
    let mut unroll_pairs = Vec::new();
    for main in 0..forest.loops.len() {
        for rem in 0..forest.loops.len() {
            if main == rem
                || forest.loops[main].parent != forest.loops[rem].parent
                || names[main].line.is_none()
                || names[main].line != names[rem].line
                || names[main].file != names[rem].file
            {
                continue;
            }
            if let (Ok(m), Ok(r)) = (&trips[main], &trips[rem])
                && let Some(factor) = unroll_factor(m, r)
            {
                unroll_pairs.push(UnrollPair {
                    main: LoopId(main as u32),
                    remainder: LoopId(rem as u32),
                    factor,
                });
            }
        }
    }

    TripCountResults {
        trips,
        entries,
        distances,
        unroll_pairs,
    }
}

/// `main = (X − X mod c)/c`, or its max with 0, and `rem = X mod c` →
/// factor c.
fn unroll_factor(main: &SymExpr, rem: &SymExpr) -> Option<i64> {
    let SymExpr::Mod(rem_x, c) = rem else {
        return None;
    };
    let main = match main {
        SymExpr::Max(m, z) if **z == SymExpr::Const(0) => m,
        _ => main,
    };
    let SymExpr::FloorDiv(inner, mc) = main else {
        return None;
    };
    if mc != c {
        return None;
    }
    let SymExpr::Sum(terms) = inner.as_ref() else {
        return None;
    };
    if terms.len() != 2 {
        return None;
    }
    let has_x = terms.iter().any(|t| t == rem_x.as_ref());
    let has_neg_mod = terms.iter().any(|t| match t {
        SymExpr::Prod(fs) => {
            fs.len() == 2 && fs[0] == SymExpr::Const(-1) && fs[1] == SymExpr::Mod(rem_x.clone(), *c)
        }
        _ => false,
    });
    (has_x && has_neg_mod).then_some(*c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::loop_names::loop_names;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    #[test]
    fn unroll_pair_links_main_and_remainder() {
        // Two sibling loops on the same source line with the
        // (X − X mod 4)/4 and X mod 4 trip shapes.
        let src = String::from(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u32 k_param_0\n)\n{\n\
             ld.param.u32 %r1, [k_param_0];\n\
             and.b32 %r9, %r1, 3;\n\
             sub.s32 %r7, %r9, %r1;\n\
             mov.u32 %r2, 0;\n\
             .loc 1 14 9\n\
             $L__M:\n\
             add.s32 %r2, %r2, 4;\n\
             add.s32 %r3, %r7, %r2;\n\
             setp.ne.s32 %p1, %r3, 0;\n@%p1 bra $L__M;\n\
             mov.u32 %r4, %r9;\n\
             $L__R:\n\
             .loc 1 14 9\n\
             add.s32 %r4, %r4, -1;\n\
             setp.ne.s32 %p2, %r4, 0;\n@%p2 bra $L__R;\n\
             ret;\n}\n.file 1 \"kern.cu\"\n",
        );
        let pairs = |src: &str, bindings: &HashMap<SymbolicConstant, i64>| {
            let m = parse(src).expect("parses");
            let k = &m.kernels[0];
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            let names = loop_names(&m, k, &cfg, &f);
            let info = trip_counts(&m, k, &cfg, &f, &names, bindings);
            let main = info.unroll_pairs.first().map(|p| {
                let trips = info.trips[p.main.0 as usize].as_ref();
                (p.factor, trips.unwrap().to_string())
            });
            (info.unroll_pairs.len(), main)
        };
        let free = HashMap::new();
        let shape = |x: &str| (1, Some((4, format!("({x} - {x} mod 4) / 4"))));
        assert_eq!(pairs(&src, &free), shape("param_0"));
        // A min-bounded pair keeps its shape under the bindings.
        let src = src.replace(
            "ld.param.u32 %r1, [k_param_0];",
            "ld.param.u32 %r0, [k_param_0];\nmin.s32 %r1, %r0, 8;",
        );
        let bound = [(SymbolicConstant::KernelParam(0), 100)]
            .into_iter()
            .collect();
        assert_eq!(pairs(&src, &free), shape("min(param_0, 8)"));
        assert_eq!(pairs(&src, &bound), shape("min(param_0, 8)"));
        // A unit-step main loop below (X − X mod 4)/4, behind a guard that
        // tests its latch at k = 0.
        let src = String::from(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u32 k_param_0\n)\n{\n\
             ld.param.u32 %r1, [k_param_0];\n\
             and.b32 %r9, %r1, 3;\n\
             sub.s32 %r7, %r1, %r9;\n\
             shr.u32 %r8, %r7, 2;\n\
             setp.lt.s32 %p3, %r8, 1;\n@%p3 bra $L__G;\n\
             mov.u32 %r2, 0;\n\
             .loc 1 14 9\n\
             $L__M:\n\
             add.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r8;\n@%p1 bra $L__M;\n\
             $L__G:\n\
             mov.u32 %r4, %r9;\n\
             $L__R:\n\
             .loc 1 14 9\n\
             add.s32 %r4, %r4, -1;\n\
             setp.ne.s32 %p2, %r4, 0;\n@%p2 bra $L__R;\n\
             ret;\n}\n.file 1 \"kern.cu\"\n",
        );
        let clamped = (
            1,
            Some((4, "max((param_0 - param_0 mod 4) / 4, 0)".to_owned())),
        );
        assert_eq!(pairs(&src, &free), clamped);
    }
}
