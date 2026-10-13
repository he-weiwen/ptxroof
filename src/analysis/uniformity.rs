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
use crate::analysis::register_effects::{carry, split_operands};
use crate::analysis::scalar::trace::{
    AffineValueTracer, ReachingDefinition, SpecialRegister, special_register,
};
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Instr, Module, Opcode, Operand, OperandId, SpecialReg, Stmt};
use crate::support::intern::Symbol;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Def(usize),
    /// The carry flag CC.CF a statement writes.
    Carry(usize),
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
            reads.extend(
                def_reads(t, d, false)
                    .into_iter()
                    .map(|r| (Node::Def(d), r)),
            );
        }
        for &d in &t.carry_defs {
            reads.extend(
                def_reads(t, d, true)
                    .into_iter()
                    .map(|r| (Node::Carry(d), r)),
            );
        }
        for b in (0..t.cfg.blocks.len() as u32).map(BlockId) {
            if let Some((pos, instr)) = t.last_instr(b)
                && matches!(instr.mnemonic, Opcode::Bra | Opcode::Brx)
            {
                let mut rs = operand_reads(t, t.module.operand_ids(instr.operands), pos);
                rs.extend(instr.predicate.map(|p| read(t, p.reg, pos)));
                reads.extend(rs.into_iter().map(|r| (Node::Branch(b), r)));
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

/// What the registers statement `d` writes depend on, or with
/// `carry_out` the CC.CF it writes: its inputs and, when guarded, the
/// guard and the value it keeps where the guard is false.
fn def_reads(t: &AffineValueTracer, d: usize, carry_out: bool) -> Vec<Read> {
    let Stmt::Instr(instr) = &t.kernel.stmts[d] else {
        return vec![Read::Divergent(
            "comes from a statement not parsed".to_owned(),
        )];
    };
    if !lane_wise(t.module, instr) {
        return vec![Read::Divergent(format!(
            "comes from `{}`",
            t.module.opcode(instr)
        ))];
    }
    let (written, inputs) = split_operands(t.module, instr);
    let mut rs = operand_reads(t, inputs, d);
    if carry(t.module, instr).1 {
        rs.push(read_carry(t, d));
    }
    if let Some(p) = instr.predicate {
        rs.push(read(t, p.reg, d));
        match carry_out {
            true => rs.push(read_carry(t, d)),
            false => rs.extend(operand_reads(t, written, d)),
        }
    }
    rs
}

fn read(t: &AffineValueTracer, reg: Symbol, pos: usize) -> Read {
    let sites = t.defs.get(&reg).map_or(&[][..], Vec::as_slice);
    read_sites(t, t.module.interner.resolve(reg), sites, pos, Node::Def)
}

fn read_carry(t: &AffineValueTracer, pos: usize) -> Read {
    read_sites(t, "CC.CF", &t.carry_defs, pos, Node::Carry)
}

/// A read at `pos` of `name`, which the statements `sites` define, each
/// the source `node` names.
fn read_sites(
    t: &AffineValueTracer,
    name: &str,
    sites: &[usize],
    pos: usize,
    node: fn(usize) -> Node,
) -> Read {
    let here = t.block_of(pos);
    let reaches = |a: BlockId, b: BlockId| a == b || t.reach[a.0 as usize][b.0 as usize];
    let defs = match t.reach_sites(sites, pos, None) {
        ReachingDefinition::Def(d) => vec![d],
        ReachingDefinition::Merged(defs) => defs,
        ReachingDefinition::Carried(_) => sites
            .iter()
            .copied()
            .filter(|&d| {
                (d < pos && t.block_of(d) == here)
                    || t.reach[t.block_of(d).0 as usize][here.0 as usize]
            })
            .collect(),
        ReachingDefinition::None => {
            return Read::Divergent(format!("reads {name}, which has no definition"));
        }
    };
    let mut sources: Vec<Node> = defs.iter().map(|&d| node(d)).collect();
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
            instr.mnemonic == Opcode::Ld
                && module
                    .modifiers(instr)
                    .iter()
                    .any(|&m| module.interner.resolve(m) == "param")
        }
    }
}

fn read_special(reg: SpecialReg) -> Read {
    let name = reg.name();
    match special_register(reg) {
        Some(SpecialRegister::PerCta | SpecialRegister::Zero) => Read::Uniform,
        Some(SpecialRegister::PerThread) => Read::Divergent(format!("depends on {name}")),
        Some(SpecialRegister::Varying) => {
            Read::Divergent(format!("{name} can change between reads"))
        }
        None => Read::Divergent(format!("{name} is not modelled")),
    }
}

fn operand_reads(t: &AffineValueTracer, ops: &[OperandId], pos: usize) -> Vec<Read> {
    ops.iter()
        .flat_map(|&op| match t.module.operand(op) {
            Operand::Register(r) => vec![read(t, *r, pos)],
            Operand::Special(reg) => vec![read_special(*reg)],
            Operand::Memory { base, rest, .. } => {
                let mut rs = operand_reads(t, &[*base], pos);
                rs.extend(operand_reads(t, t.module.operand_ids(*rest), pos));
                rs
            }
            Operand::Selected { base, .. } | Operand::Negated { inner: base } => {
                operand_reads(t, &[*base], pos)
            }
            Operand::VectorList { children }
            | Operand::MultipleDestinations { children }
            | Operand::ParamList { children } => {
                operand_reads(t, t.module.operand_ids(*children), pos)
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
        assert_eq!(uniform(carry, "%r3"), tid());
        let carry = "add.cc.u32 %r2, %r0, 0xfffffff0;\naddc.u32 %r3, 0, 0;";
        assert_eq!(uniform(carry, "%r3"), Ok(()));
        // A guarded `add.cc` leaves the per-thread carry where it is skipped.
        let carry = "add.cc.u32 %r2, %r1, 0xfffffff0;\nsetp.eq.u32 %p1, %r0, 0;\n\
                     mov.u32 %r4, 0;\n@%p1 add.cc.u32 %r4, %r0, 1;\naddc.u32 %r3, 0, 0;";
        assert_eq!(uniform(carry, "%r3"), tid());
        // The carry it keeps is not part of its register result.
        assert_eq!(uniform(carry, "%r4"), Ok(()));
        // PTX ISA §9.7.2: calls do not preserve CC.CF, and neither does
        // a statement the parser cannot read.
        let carry = "add.cc.u32 %r2, %r0, 1;\ncall (%r5), f, (%r0);\naddc.u32 %r3, 0, 0;";
        assert_eq!(uniform(carry, "%r3"), Err("comes from `call`".to_owned()));
        let carry = "add.cc.u32 %r2, %r0, 1;\nfuture.op %r5;\naddc.u32 %r3, 0, 0;";
        assert_eq!(
            uniform(carry, "%r3"),
            Err("comes from a statement not parsed".to_owned())
        );
        // One after the last terminator is in no block, and never runs.
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\nadd.cc.u32 %r0, 0, 0;\nbra L;\nL:\n\
                   addc.u32 %r1, 0, 0;\nret;\ncall (%r2), f, ();\n}\n";
        assert_eq!(uniform_at(src, "%r1", Some("L"), "ret"), Ok(()));
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
