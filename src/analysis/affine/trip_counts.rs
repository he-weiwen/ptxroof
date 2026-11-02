//! Every loop's trip count (`super::affine_analysis`).

use crate::analysis::affine::affine_analysis::AffineAnalysis;
use crate::analysis::affine::affine_expr::Var;
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::uniformity_info::UniformityInfo;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{Kernel, Module, Register};
use crate::ssa::Value;
use std::collections::HashMap;

/// Trip count of one loop: an expression, or a named reason there
/// isn't one.
pub type TripCount = Result<SymExpr, String>;

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
}

/// Every loop's trip count, its latch traced under `bindings` (`--bind`,
/// the launch shape) where a constant is needed.
pub fn trip_counts(
    module: &Module,
    kernel: &Kernel,
    cfg: &ControlFlowGraph,
    forest: &LoopForest,
    bindings: &HashMap<SymbolicConstant, i64>,
) -> TripCountResults {
    let data = KernelAnalyses::new(module, kernel, cfg, forest);
    let uniformity = UniformityInfo::new(&data);
    trip_counts_from(
        &AffineAnalysis::new(&data, &uniformity, Some(bindings)),
        &|v| unknown_by_register(module, v),
    )
}

/// Variable `v` as a reason prints it outside a report, its unknown
/// value by register: `uniform(%r3)`.
pub(crate) fn unknown_by_register(module: &Module, v: &Var) -> String {
    match v {
        Var::Unknown(
            Value::Result { reg, .. } | Value::Argument { reg, .. } | Value::Undef(reg),
        ) => match reg {
            Register::Named(s) => format!("uniform({})", module.interner.resolve(*s)),
            Register::Cc => "uniform(CC.CF)".to_owned(),
        },
        other => other.to_string(),
    }
}

/// Every loop's trip count under `affine`'s bindings, a reason's unknown
/// values as `name` prints them.
pub(crate) fn trip_counts_from(
    affine: &AffineAnalysis,
    name: &dyn Fn(&Var) -> String,
) -> TripCountResults {
    let forest = affine.data.forest;
    let (mut trips, mut entries, mut distances) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..forest.loops.len() as u32 {
        let (t, entry, distance) = match affine.loop_trip_count(LoopId(i)) {
            Ok(t) => (Ok(t.count), t.entry, t.distance),
            Err(e) => (
                Err(e.render(affine.data, Some(LoopId(i)), true, name)),
                forest.get(LoopId(i)).header,
                None,
            ),
        };
        trips.push(t);
        entries.push(entry);
        distances.push(distance);
    }

    TripCountResults {
        trips,
        entries,
        distances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    #[test]
    fn an_unrolled_loop_s_main_part_runs_the_quotient() {
        // nvcc's unroll by 4: a main loop of (X − X mod 4)/4 trips and a
        // remainder of X mod 4.
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
            let info = trip_counts(&m, k, &cfg, &f, bindings);
            info.trips[0].as_ref().map(ToString::to_string).ok()
        };
        let free = HashMap::new();
        let shape = |x: &str| Some(format!("({x} - {x} mod 4) / 4"));
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
        let clamped = Some("max((param_0 - param_0 mod 4) / 4, 0)".to_owned());
        assert_eq!(pairs(&src, &free), clamped);
    }
}
