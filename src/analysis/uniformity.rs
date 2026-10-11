//! Which register reads the lanes executing a statement see alike: LLVM's
//! UniformityAnalysis over PTX registers. Divergence starts at per-thread
//! special registers and at instructions not computed lane by lane from
//! their operands (loads, warp collectives, tensor-core operations), and
//! spreads forward through the definitions and branch predicates that
//! read it, to a fixed point. Where definitions meet, the branches that
//! select between them gate the value, as a γ's predicate does in Gated
//! SSA; a definition inside a loop read outside it is gated by the
//! loop's exits, as by an η's.

use crate::analysis::control_flow::loops::LoopId;
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
                && matches!(t.module.interner.resolve(instr.mnemonic), "bra" | "brx")
            {
                let mut regs = registers(t.module, t.module.operand_ids(instr.operands));
                regs.extend(instr.predicate.map(|p| p.reg));
                reads.extend(regs.into_iter().map(|r| (Node::Branch(b), read(t, r, pos))));
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

    /// Whether every lane executing definition `d` writes the same value.
    pub(crate) fn is_uniform_def(&self, d: usize) -> bool {
        !self.divergent.contains_key(&Node::Def(d))
    }

    /// Whether the branches between `reg`'s definition and `pos` (the
    /// exits of loops holding the definition but not `pos`, and those
    /// selecting among definitions) send every lane the same way; the
    /// reason when not.
    pub(crate) fn gates_uniform(
        &self,
        t: &AffineValueTracer,
        reg: Symbol,
        pos: usize,
    ) -> Result<(), String> {
        match read(t, reg, pos) {
            Read::Uniform => Ok(()),
            Read::Divergent(why) => Err(why),
            Read::On(sources) => (sources.iter())
                .filter(|s| matches!(s, Node::Branch(_)))
                .find_map(|s| self.divergent.get(s))
                .map_or(Ok(()), |why| Err(why.clone())),
        }
    }

    /// Whether the lanes in loop `id` leave it at the same iteration: its
    /// exits are uniform; the reason when not.
    pub(crate) fn exits_uniform(&self, t: &AffineValueTracer, id: LoopId) -> Result<(), String> {
        exits(t, id)
            .find_map(|n| self.divergent.get(&n))
            .map_or(Ok(()), |why| Err(why.clone()))
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
    let here = t.block_of(pos);
    let reaches = |a: BlockId, b: BlockId| a == b || t.reach[a.0 as usize][b.0 as usize];
    let defs = match t.reach_def(reg, pos, None) {
        ReachingDefinition::Def(d) => vec![d],
        ReachingDefinition::Merged(defs) => defs,
        ReachingDefinition::Carried(_) => t.defs[&reg]
            .iter()
            .copied()
            .filter(|&d| {
                (d < pos && t.block_of(d) == here)
                    || t.reach[t.block_of(d).0 as usize][here.0 as usize]
            })
            .collect(),
        ReachingDefinition::None => {
            return match special_register(name) {
                Some(SpecialRegister::PerCta) => Read::Uniform,
                Some(SpecialRegister::PerThread) => Read::Divergent(format!("depends on {name}")),
                None => Read::Divergent(format!("reads {name}, which has no definition")),
            };
        }
    };
    let mut sources: Vec<Node> = defs.iter().map(|&d| Node::Def(d)).collect();
    // An η's gating predicates: the exits of every loop holding a
    // definition but not the read. A cycle with more than one entry is
    // no loop of the forest, and its exits are not known.
    for &d in &defs {
        let at = t.block_of(d);
        if !t.forest.irreducible_edges.is_empty()
            && t.reach[at.0 as usize][at.0 as usize]
            && !(reaches(here, at) && reaches(at, here))
        {
            return Read::Divergent(format!("{name} is read after a cycle defining it"));
        }
        let mut l = t.forest.block_loop[at.0 as usize];
        while let Some(id) = l {
            if !t.in_loop(id, pos) {
                sources.extend(exits(t, id));
            }
            l = t.forest.get(id).parent;
        }
    }
    // A γ's gating predicates: branches with two successors that reach
    // the read, one of them through a definition.
    if defs.len() > 1 {
        sources.extend((0..t.cfg.blocks.len() as u32).map(BlockId).filter_map(|b| {
            let succs = &t.cfg.block(b).succs;
            (succs.iter().filter(|&&s| reaches(s, here)).count() > 1
                && defs
                    .iter()
                    .any(|&d| succs.iter().any(|&s| reaches(s, t.block_of(d)))))
            .then_some(Node::Branch(b))
        }));
    }
    Read::On(sources)
}

/// The branches of loop `id`'s blocks that can leave it.
fn exits<'a>(t: &'a AffineValueTracer, id: LoopId) -> impl Iterator<Item = Node> + 'a {
    let blocks = &t.forest.get(id).blocks;
    (blocks.iter())
        .filter(|&&b| (t.cfg.block(b).succs.iter()).any(|s| blocks.binary_search(s).is_err()))
        .map(|&b| Node::Branch(b))
}

/// Each lane's result is a function of its own operands.
fn lane_wise(module: &Module, instr: &Instr) -> bool {
    let c = classify(module, instr);
    match c.category {
        // PTX ISA §9.7.2: addc, subc and madc also read the carry flag
        // CC.CF, which no register holds.
        InstructionCategory::NonFlopArith { .. } => !matches!(
            module.interner.resolve(instr.mnemonic),
            "addc" | "subc" | "madc"
        ),
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

    /// Whether `reg` is uniform where the first `st.global.u32` reads it,
    /// one after `body` if it has none; `%r0` holds a parameter and `%r1`
    /// the thread index.
    fn uniform(body: &str, reg: &str) -> Result<(), String> {
        let store = match body.contains("st.global.u32") {
            true => String::new(),
            false => format!("st.global.u32 [%rd9], {reg};"),
        };
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n.param .u32 k_param_0\n)\n{{\n\
             ld.param.u32 %r0, [k_param_0];\nmov.u32 %r1, %tid.x;\n{body}\n\
             {store}\nret;\n}}\n"
        );
        uniform_at(&src, reg, None, "st.global.u32")
    }

    /// Whether `reg` is uniform at the first `opcode` after `label`.
    fn uniform_at(src: &str, reg: &str, label: Option<&str>, opcode: &str) -> Result<(), String> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let t = AffineValueTracer::new(&m, k, &cfg, &forest);
        let start = label.map_or(0, |l| {
            k.stmts
                .iter()
                .position(|s| matches!(s, Stmt::Label(s) if m.interner.resolve(*s) == l))
                .expect("the label")
        });
        let pos = (start..k.stmts.len())
            .find(|&i| matches!(&k.stmts[i], Stmt::Instr(i) if m.opcode(i) == opcode))
            .expect("the read");
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
        // The carry out of a per-thread sum: 1 in lanes 16 and up.
        let carry = "add.cc.u32 %r2, %r1, 0xfffffff0;\naddc.u32 %r3, 0, 0;";
        assert_eq!(
            uniform(carry, "%r3"),
            Err("comes from `addc.u32`".to_owned())
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
    fn a_join_is_divergent_when_a_divergent_branch_selects_the_definition() {
        let join = |on: &str| {
            format!(
                "setp.lt.u32 %p1, {on}, 16;\n@%p1 bra $L__A;\nmov.u32 %r2, 64;\n\
                 bra.uni $L__J;\n$L__A:\nmov.u32 %r2, 0;\n$L__J:"
            )
        };
        assert_eq!(uniform(&join("%r1"), "%r2"), tid());
        assert_eq!(uniform(&join("%r0"), "%r2"), Ok(()));
        // A divergent branch whose other side never reaches the read
        // selects nothing (k1's bounds check around its K loop).
        let guarded = format!(
            "setp.lt.u32 %p3, %r1, 100;\n@%p3 bra $L__X;\n{}\n\
             st.global.u32 [%rd9], %r2;\n$L__X:",
            join("%r0")
        );
        assert_eq!(uniform(&guarded, "%r2"), Ok(()));
    }

    #[test]
    fn a_counter_is_uniform_in_its_loop_unless_a_divergent_branch_skips_its_step() {
        let looped = |skip: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__L:\nst.global.u32 [%rd9], %r2;\n\
                 setp.lt.u32 %p2, {skip}, 3;\n@%p2 bra $L__S;\nadd.s32 %r2, %r2, 1;\n\
                 $L__S:\nsetp.lt.u32 %p1, %r2, %r0;\n@%p1 bra $L__L;"
            )
        };
        assert_eq!(uniform(&looped("%r0"), "%r2"), Ok(()));
        assert_eq!(uniform(&looped("%r1"), "%r2"), tid());
    }

    #[test]
    fn a_value_from_a_loop_is_divergent_after_it_when_its_exit_is() {
        let looped = |exit: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                 setp.lt.u32 %p1, %r2, {exit};\n@%p1 bra $L__L;"
            )
        };
        assert_eq!(uniform(&looped("%r0"), "%r2"), Ok(()));
        assert_eq!(uniform(&looped("%r1"), "%r2"), tid());
        // The lanes still in the loop agree on it.
        let inside = looped("%r1").replace("add.s32", "st.global.u32 [%rd9], %r2;\nadd.s32");
        assert_eq!(uniform(&inside, "%r2"), Ok(()));
    }

    /// A cycle entered at A or B: no loop of the forest, so its exits do
    /// not gate the counter, which is divergent after it.
    #[test]
    fn a_value_from_a_cycle_with_two_entries_is_divergent_after_it() {
        let body = "mov.u32 %r2, 0;\nsetp.lt.u32 %p1, %r0, 5;\n@%p1 bra $L__B;\n\
                    $L__A:\nadd.s32 %r2, %r2, 1;\n$L__B:\nadd.s32 %r2, %r2, 1;\n\
                    setp.lt.u32 %p2, %r2, %r1;\n@%p2 bra $L__A;";
        assert_eq!(
            uniform(body, "%r2"),
            Err("%r2 is read after a cycle defining it".to_owned())
        );
    }

    /// k1's remainder loop starts from the main loop's last counter, or 0
    /// when K < 4 skips it; both choices compare parameters only, and the
    /// bounds check on the thread index skips both loops.
    #[test]
    fn k1s_counter_after_its_main_loop_is_uniform() {
        let k1 = include_str!("../../tests/fixtures/k1/k1.sm_89.ptx");
        assert_eq!(
            uniform_at(k1, "%r34", Some("$L__BB0_5"), "mad.lo.s32"),
            Ok(())
        );
    }
}
