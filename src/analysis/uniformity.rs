//! Which register reads the lanes executing a statement see alike: LLVM's
//! UniformityAnalysis over PTX registers. Divergence starts at per-thread
//! special registers and at instructions not computed lane by lane from
//! their operands (loads, warp collectives, tensor-core operations), and
//! spreads forward through the definitions and branch predicates that
//! read it, to a fixed point. A read with more than one reaching
//! definition, or of a definition inside a loop from outside it, is
//! divergent.

use crate::analysis::instruction_counts::classify::{InstructionCategory, Pipe, classify};
use crate::analysis::instruction_counts::measurement::MeasureKind;
use crate::analysis::scalar::trace::{
    AffineValueTracer, ReachingDefinition, SpecialRegister, special_register,
};
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Instr, Module, Operand, OperandId, Stmt};
use crate::support::intern::Symbol;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Def(usize),
    Branch(BlockId),
}

enum Read {
    Uniform,
    Divergent(String),
    On(Vec<Node>),
}

pub(crate) struct Uniformity {
    divergent: HashMap<Node, String>,
}

impl Uniformity {
    pub(crate) fn new(t: &AffineValueTracer) -> Self {
        let mut reads: Vec<(Node, Read)> = Vec::new();
        let defs: BTreeSet<usize> = t.defs.values().flatten().copied().collect();
        for d in defs {
            let Stmt::Instr(instr) = &t.kernel.stmts[d] else {
                continue;
            };
            let (node, ops) = (Node::Def(d), t.module.operand_ids(instr.operands));
            if !lane_wise(t.module, instr) {
                let why = format!("comes from `{}`", t.module.opcode(instr));
                reads.push((node, Read::Divergent(why)));
                continue;
            }
            let mut regs = registers(t.module, &ops[1..]);
            if let Some(p) = instr.predicate {
                regs.push(p.reg);
                regs.extend(registers(t.module, &ops[..1]));
            }
            reads.extend(regs.into_iter().map(|r| (node, read(t, r, d))));
        }
        for b in (0..t.cfg.blocks.len() as u32).map(BlockId) {
            if let Some((pos, instr)) = t.last_instr(b)
                && t.module.interner.resolve(instr.mnemonic) == "bra"
                && let Some(p) = instr.predicate
            {
                reads.push((Node::Branch(b), read(t, p.reg, pos)));
            }
        }
        let mut users: HashMap<Node, Vec<Node>> = HashMap::new();
        let mut divergent = HashMap::new();
        let mut queue = Vec::new();
        for (node, r) in reads {
            match r {
                Read::Uniform => {}
                Read::Divergent(why) => {
                    if divergent.insert(node, why).is_none() {
                        queue.push(node);
                    }
                }
                Read::On(sources) => {
                    for s in sources {
                        users.entry(s).or_default().push(node);
                    }
                }
            }
        }
        while let Some(n) = queue.pop() {
            let why = divergent[&n].clone();
            for &m in users.get(&n).into_iter().flatten() {
                if let Entry::Vacant(e) = divergent.entry(m) {
                    e.insert(why.clone());
                    queue.push(m);
                }
            }
        }
        Uniformity { divergent }
    }

    /// Whether every lane executing `pos` reads the same value of `reg`;
    /// the reason when not.
    pub(crate) fn is_uniform(
        &self,
        t: &AffineValueTracer,
        reg: Symbol,
        pos: usize,
    ) -> Result<(), String> {
        match read(t, reg, pos) {
            Read::Uniform => Ok(()),
            Read::Divergent(why) => Err(why),
            Read::On(sources) => sources
                .iter()
                .find_map(|s| self.divergent.get(s))
                .map_or(Ok(()), |why| Err(why.clone())),
        }
    }
}

fn read(t: &AffineValueTracer, reg: Symbol, pos: usize) -> Read {
    let name = t.module.interner.resolve(reg);
    let d = match t.reach_def(reg, pos, None) {
        ReachingDefinition::Def(d) => d,
        ReachingDefinition::None => {
            return match special_register(name) {
                Some(SpecialRegister::PerCta) => Read::Uniform,
                Some(SpecialRegister::PerThread) => Read::Divergent(format!("depends on {name}")),
                None => Read::Divergent(format!("reads {name}, which has no definition")),
            };
        }
        ReachingDefinition::Merged(_) | ReachingDefinition::Carried(_) => {
            return Read::Divergent(format!("{name} has more than one reaching definition"));
        }
    };
    let mut l = t.forest.block_loop[t.block_of(d).0 as usize];
    while let Some(id) = l {
        if !t.in_loop(id, pos) {
            return Read::Divergent(format!("{name} is read after the loop defining it"));
        }
        l = t.forest.get(id).parent;
    }
    Read::On(vec![Node::Def(d)])
}

/// Each lane's result is a function of its own operands.
fn lane_wise(module: &Module, instr: &Instr) -> bool {
    let c = classify(module, instr);
    match c.category {
        InstructionCategory::NonFlopArith { .. } => true,
        InstructionCategory::Flop => c.contributions.iter().all(|k| {
            matches!(
                k.kind,
                MeasureKind::Flops {
                    pipe: Pipe::CudaCore | Pipe::Sfu,
                    ..
                }
            )
        }),
        _ => {
            module.interner.resolve(instr.mnemonic) == "ld"
                && module
                    .modifiers(instr)
                    .iter()
                    .any(|&m| module.interner.resolve(m) == "param")
        }
    }
}

fn registers(module: &Module, ops: &[OperandId]) -> Vec<Symbol> {
    ops.iter()
        .flat_map(|&op| match module.operand(op) {
            Operand::Register(r) => vec![*r],
            Operand::Memory { base, .. } => registers(module, &[*base]),
            Operand::VectorList { children } | Operand::MultipleDestinations { children } => {
                registers(module, module.operand_ids(*children))
            }
            Operand::Immediate(_) | Operand::SymbolRef(_) => Vec::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::control_flow::loop_forest;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::parse::parser::parse;

    /// Whether `reg` is uniform where the first `st.global.u32` reads it;
    /// `%r0` holds a parameter and `%r1` the thread index.
    fn uniform(body: &str, reg: &str) -> Result<(), String> {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u32 k_param_0\n)\n{{\n\
             ld.param.u32 %r0, [k_param_0];\nmov.u32 %r1, %tid.x;\n{body}\n\
             st.global.u32 [%rd9], {reg};\nret;\n}}\n"
        );
        let m = parse(&src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let t = AffineValueTracer::new(&m, k, &cfg, &forest);
        let pos = k
            .stmts
            .iter()
            .position(|s| matches!(s, Stmt::Instr(i) if m.opcode(i) == "st.global.u32"))
            .expect("the store");
        Uniformity::new(&t).is_uniform(&t, m.interner.get(reg).expect("register"), pos)
    }

    fn tid() -> Result<(), String> {
        Err("depends on %tid.x".to_owned())
    }

    #[test]
    fn divergence_follows_data_from_per_thread_registers_and_loads() {
        assert_eq!(uniform("shl.b32 %r2, %r1, 2;", "%r2"), tid());
        assert_eq!(uniform("", "%r0"), Ok(()));
        let cta = "mov.u32 %r2, %ctaid.x;\nmad.lo.s32 %r3, %r2, %r0, 7;";
        assert_eq!(uniform(cta, "%r3"), Ok(()));
        assert_eq!(
            uniform("ld.global.u32 %r2, [%rd1];\nadd.s32 %r3, %r2, 1;", "%r3"),
            Err("comes from `ld.global.u32`".to_owned())
        );
        assert_eq!(
            uniform("shfl.sync.idx.b32 %r2, %r0, 0, 31, -1;", "%r2"),
            Err("comes from `shfl.sync.idx.b32`".to_owned())
        );
    }

    #[test]
    fn a_guarded_write_keeps_the_old_value_in_lanes_it_skips() {
        let on_tid = "setp.lt.u32 %p1, %r1, 16;\nmov.u32 %r2, 0;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(uniform(on_tid, "%r2"), tid());
        let on_param = "setp.lt.u32 %p1, %r0, 16;\nmov.u32 %r2, 0;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(uniform(on_param, "%r2"), Ok(()));
        let old_on_tid = "setp.lt.u32 %p1, %r0, 16;\nmov.u32 %r2, %r1;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(uniform(old_on_tid, "%r2"), tid());
    }

    #[test]
    fn a_value_from_several_definitions_is_divergent() {
        let body = "setp.lt.u32 %p1, %r0, 16;\n@%p1 bra $L__A;\nmov.u32 %r2, 64;\n\
                    bra.uni $L__J;\n$L__A:\nmov.u32 %r2, 0;\n$L__J:";
        assert_eq!(
            uniform(body, "%r2"),
            Err("%r2 has more than one reaching definition".to_owned())
        );
    }
}
