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
use crate::analysis::instruction_counts::classify::{
    Direction, InstructionCategory, Pipe, Space, classify,
};
use crate::analysis::instruction_counts::measurement::MeasureKind;
use crate::analysis::register_effects::{reads_cc, split_operands};
use crate::analysis::scalar::trace::{AffineValueTracer, SpecialRegister, special_register};
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Instr, Opcode, Operand, OperandId, Register, SpecialReg, Stmt};
use crate::ssa::Value;
use crate::support::intern::Symbol;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};

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
        let written = written(t);
        let defs: BTreeSet<usize> = (t.writes.iter())
            .filter(|(r, _)| matches!(r, Register::Named(_)))
            .flat_map(|(_, ds)| ds.iter().copied())
            .collect();
        for d in defs {
            reads.extend(
                def_reads(t, d, false, written.as_ref())
                    .into_iter()
                    .map(|r| (Node::Def(d), r)),
            );
        }
        for &d in t.writes.get(&Register::Cc).into_iter().flatten() {
            reads.extend(
                def_reads(t, d, true, written.as_ref())
                    .into_iter()
                    .map(|r| (Node::Carry(d), r)),
            );
        }
        for b in (0..t.cfg.blocks.len() as u32).map(BlockId) {
            if let Some((pos, instr)) = t.last_instr(b)
                && matches!(instr.mnemonic, Opcode::Bra | Opcode::Brx)
            {
                let mut rs = operand_reads(t, t.module.operand_ids(instr.operands), pos);
                rs.extend(
                    instr
                        .predicate
                        .map(|p| read(t, Register::Named(p.reg), pos)),
                );
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
        match read(t, Register::Named(reg), pos) {
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
        self.answer(read(t, Register::Named(reg), pos))
    }

    /// Whether every lane executing `pos` sees value `v` alike, wherever
    /// it was read: its definitions, and the loop exits and branches
    /// between them and `pos`.
    pub(crate) fn value_uniform_at(
        &self,
        t: &AffineValueTracer,
        v: Value,
        pos: usize,
    ) -> Result<(), String> {
        self.answer(read_value(t, v, pos))
    }

    fn answer(&self, r: Read) -> Result<(), String> {
        match r {
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
fn def_reads(
    t: &AffineValueTracer,
    d: usize,
    carry_out: bool,
    written: Option<&BTreeSet<usize>>,
) -> Vec<Read> {
    let Stmt::Instr(instr) = &t.kernel.stmts[d] else {
        return vec![Read::Divergent(
            "comes from a statement not parsed".to_owned(),
        )];
    };
    if !lane_wise(t, instr, written) {
        return vec![Read::Divergent(format!(
            "comes from `{}`",
            t.module.opcode(instr)
        ))];
    }
    let (written, inputs) = split_operands(t.module, instr);
    let mut rs = operand_reads(t, inputs, d);
    if reads_cc(t.module, instr) {
        rs.push(read(t, Register::Cc, d));
    }
    if let Some(p) = instr.predicate {
        rs.push(read(t, Register::Named(p.reg), d));
        match carry_out {
            true => rs.push(read(t, Register::Cc, d)),
            false => rs.extend(operand_reads(t, written, d)),
        }
    }
    rs
}

fn read(t: &AffineValueTracer, reg: Register, pos: usize) -> Read {
    match t.ssa.value_before(pos, reg) {
        Some(v) => read_value(t, v, pos),
        None => Read::Divergent(format!("reads {} in a block no path reaches", name(t, reg))),
    }
}

fn name<'a>(t: &'a AffineValueTracer, reg: Register) -> &'a str {
    match reg {
        Register::Named(r) => t.module.interner.resolve(r),
        Register::Cc => "CC.CF",
    }
}

/// A read at `pos` of value `v`.
fn read_value(t: &AffineValueTracer, v: Value, pos: usize) -> Read {
    let reg = match v {
        Value::Result { reg, .. } | Value::Argument { reg, .. } | Value::Undef(reg) => reg,
    };
    let node: fn(usize) -> Node = match reg {
        Register::Named(_) => Node::Def,
        Register::Cc => Node::Carry,
    };
    let name = name(t, reg);
    let sources = t.ssa.sources(v);
    let defs: Vec<usize> = (sources.iter())
        .filter_map(|s| match *s {
            Value::Result { op, .. } => Some(op),
            _ => None,
        })
        .collect();
    if defs.is_empty() {
        return Read::Divergent(format!("reads {name}, which has no definition"));
    }
    if sources.iter().any(|s| matches!(s, Value::Undef(_))) {
        return Read::Divergent(format!("reads {name}, which a path leaves unwritten"));
    }
    read_sites(t, name, &defs, pos, node)
}

/// A read at `pos` of `name`, whose value the statements `defs` may
/// have written, each the source `node` names.
fn read_sites(
    t: &AffineValueTracer,
    name: &str,
    defs: &[usize],
    pos: usize,
    node: fn(usize) -> Node,
) -> Read {
    let here = t.block_of(pos);
    let reaches = |a: BlockId, b: BlockId| a == b || t.reach[a.0 as usize][b.0 as usize];
    let mut sources: Vec<Node> = defs.iter().map(|&d| node(d)).collect();
    // An η's gating predicates: the exits of every loop holding a
    // definition but not the read. A cycle with more than one entry is
    // no loop of the forest, and its exits are not known.
    for &d in defs {
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
        let blocks = (0..t.cfg.blocks.len() as u32).map(BlockId);
        let reachable = blocks.filter(|&b| t.forest.doms.is_reachable(b));
        sources.extend(reachable.filter_map(|b| {
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

/// Each lane's result is a function of its own operands: a kernel
/// parameter is read-only and one per grid, a function's `.param`, a
/// call's return value among them, one per thread (PTX ISA §5.1.6,
/// Table 7). A weak global load reads one value at an address where
/// nothing writes it during the launch (PTX ISA §8, §9.7.9.8): its
/// address points into kernel parameters `written` does not hold.
fn lane_wise(t: &AffineValueTracer, instr: &Instr, written: Option<&BTreeSet<usize>>) -> bool {
    let module = t.module;
    let mods: Vec<&str> = (module.modifiers(instr).iter())
        .map(|&m| module.interner.resolve(m))
        .collect();
    if instr.mnemonic == Opcode::Ld
        && mods.contains(&"global")
        && !(mods.iter()).any(|m| matches!(*m, "volatile" | "relaxed" | "acquire" | "mmio"))
    {
        return pointees(t, instr).is_some_and(|p| written.is_some_and(|w| w.is_disjoint(&p)));
    }
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
        _ => kernel_param(t, instr).is_some(),
    }
}

/// The kernel parameter an `ld.param` reads.
fn kernel_param(t: &AffineValueTracer, instr: &Instr) -> Option<usize> {
    let module = t.module;
    let param = (module.modifiers(instr).iter()).any(|&m| module.interner.resolve(m) == "param");
    if instr.mnemonic != Opcode::Ld || !param {
        return None;
    }
    module.operand_ids(instr.operands).iter().find_map(|&o| {
        let Operand::Memory { base, .. } = module.operand(o) else {
            return None;
        };
        match module.operand(*base) {
            Operand::SymbolRef(s) => t.params.get(s).copied(),
            _ => None,
        }
    })
}

/// The kernel parameters the addresses of `instr` may point into, as
/// LLVM's `getUnderlyingObjects` (llvm/lib/Analysis/ValueTracking.cpp)
/// collects them; `None` where one is not traced to them.
fn pointees(t: &AffineValueTracer, instr: &Instr) -> Option<BTreeSet<usize>> {
    let (mut seen, mut out) = (HashSet::new(), BTreeSet::new());
    for &o in t.module.operand_ids(instr.operands) {
        if let Operand::Memory { base, rest, .. } = t.module.operand(o) {
            for &r in std::iter::once(base).chain(t.module.operand_ids(*rest)) {
                reach(t, r, &mut seen, &mut out)?;
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Through every write of a register, the operands of `mov`, `cvta`,
/// `add`, `sub`, `and`, `or`, `xor`, `min`, `max` and a 64-bit `cvt`,
/// `selp`'s values and a `mad`'s addend, to the kernel parameters
/// `ld.param` reads; `mul`, `shl`, `shr`, `div`, `rem`, `neg`, `not`,
/// `bfe` and a narrower `cvt` make offsets. `None` at anything else, a
/// symbol, a vector or a statement not parsed.
fn reach(
    t: &AffineValueTracer,
    op: OperandId,
    seen: &mut HashSet<Symbol>,
    out: &mut BTreeSet<usize>,
) -> Option<()> {
    let reg = match t.module.operand(op) {
        Operand::Register(r) => *r,
        Operand::SymbolRef(_) | Operand::VectorList { .. } => return None,
        _ => return Some(()),
    };
    if !seen.insert(reg) {
        return Some(());
    }
    for &d in t.writes.get(&Register::Named(reg)).into_iter().flatten() {
        let Stmt::Instr(instr) = &t.kernel.stmts[d] else {
            return None;
        };
        let ops = t.module.operand_ids(instr.operands);
        let wide = (t.module.modifiers(instr).last())
            .is_some_and(|&m| t.module.interner.resolve(m).ends_with("64"));
        let from = match instr.mnemonic {
            Opcode::Mov | Opcode::Cvta | Opcode::Add | Opcode::Sub => &ops[1..],
            Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Min | Opcode::Max => &ops[1..],
            Opcode::Cvt if wide => &ops[1..],
            Opcode::Selp => ops.get(1..3)?,
            Opcode::Mad => &ops[ops.len() - 1..],
            Opcode::Ld => {
                out.insert(kernel_param(t, instr)?);
                &[]
            }
            Opcode::Mul | Opcode::Shl | Opcode::Shr | Opcode::Div | Opcode::Rem => &[],
            Opcode::Neg | Opcode::Not | Opcode::Bfe | Opcode::Cvt => &[],
            _ => return None,
        };
        for &o in from {
            reach(t, o, seen, out)?;
        }
    }
    Some(())
}

/// The kernel parameters a store may write global memory through; `None`
/// where one may write anywhere: a store `pointees` does not trace, a
/// `call`, an instruction the classifier does not handle that names
/// `.global`, or a statement not parsed.
fn written(t: &AffineValueTracer) -> Option<BTreeSet<usize>> {
    let module = t.module;
    let mut out = BTreeSet::new();
    for (_, b) in t.cfg.blocks.iter_enumerated() {
        for stmt in &t.kernel.stmts[b.start..b.end] {
            let instr = match stmt {
                Stmt::Instr(instr) => instr,
                Stmt::Unparsed { .. } => return None,
                _ => continue,
            };
            let c = classify(module, instr);
            let global =
                (module.modifiers(instr).iter()).any(|&m| module.interner.resolve(m) == "global");
            if instr.mnemonic == Opcode::Call
                || c.category == InstructionCategory::Unknown && global
            {
                return None;
            }
            let stores = c.contributions.iter().any(|k| match k.kind {
                MeasureKind::Bytes { space, direction }
                | MeasureKind::UnquantifiedBytes { space, direction } => {
                    matches!(space, Space::Global | Space::Generic) && direction == Direction::Store
                }
                _ => false,
            });
            if stores {
                out.extend(pointees(t, instr)?);
            }
        }
    }
    Some(out)
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
            Operand::Register(r) => vec![read(t, Register::Named(*r), pos)],
            Operand::Special(reg) => vec![read_special(*reg)],
            Operand::Memory { base, rest, .. } => {
                let mut rs = operand_reads(t, &[*base], pos);
                rs.extend(operand_reads(t, t.module.operand_ids(*rest), pos));
                rs
            }
            Operand::Selected { base, .. }
            | Operand::Negated { inner: base }
            | Operand::Offset { base, .. } => operand_reads(t, &[*base], pos),
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

    /// A weak global load is one value for the CTA where its address is
    /// and no store of the kernel may write through its pointer
    /// parameter; `%rd1` is param_0 + 4·%ctaid.x.
    #[test]
    fn a_global_load_through_a_pointer_nothing_writes_is_lane_wise() {
        let load = |ld: &str, rest: &str| {
            let src = format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0,\n.param .u64 k_param_1\n)\n{{\n\
                 ld.param.u64 %rd0, [k_param_0];\nld.param.u64 %rd9, [k_param_1];\n\
                 mov.u32 %r1, %ctaid.x;\nmul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd1, %rd0, %rd2;\n\
                 {ld} %r2, [%rd1];\n{rest}\nst.shared.u32 [%r9], %r2;\nret;\n}}\n"
            );
            uniform_at(&src, "%r2", None, "st.shared.u32")
        };
        let from = |op: &str| Err(format!("comes from `{op}`"));
        assert_eq!(load("ld.global.u32", ""), Ok(()));
        assert_eq!(load("ld.global.nc.L1::evict_last.u32", ""), Ok(()));
        let elsewhere = "st.global.u32 [%rd9], 0;";
        assert_eq!(load("ld.global.u32", elsewhere), Ok(()));
        let written = "st.global.u32 [%rd1], 0;";
        assert_eq!(load("ld.global.u32", written), from("ld.global.u32"));
        let volatile = "ld.volatile.global.u32";
        assert_eq!(load(volatile, ""), from(volatile));
        assert_eq!(load("ld.u32", ""), from("ld.u32"));
        // A store through a pointer loaded from memory may write anywhere;
        // an index loaded from memory is an offset from its pointer.
        let scatter = "ld.global.u64 %rd5, [%rd9];\nst.global.u32 [%rd5], 0;";
        assert_eq!(load("ld.global.u32", scatter), from("ld.global.u32"));
        let gather = "ld.global.u32 %r5, [%rd9];\nmul.wide.u32 %rd6, %r5, 4;\n\
                      add.s64 %rd7, %rd9, %rd6;\nst.global.u32 [%rd7], 0;";
        assert_eq!(load("ld.global.u32", gather), Ok(()));
        let masked = "and.b64 %rd8, %rd0, -16;\nst.global.u32 [%rd8], 0;";
        assert_eq!(load("ld.global.u32", masked), from("ld.global.u32"));
        // A pointer through `cvt.u64.s64` keeps its parameter, and one
        // from an atomic may point anywhere.
        let selected = "cvt.u64.s64 %rd3, %rd0;\nselp.b64 %rd4, %rd9, %rd3, %p1;\n\
                        st.global.u32 [%rd4], 0;";
        assert_eq!(load("ld.global.u32", selected), from("ld.global.u32"));
        let swapped = "atom.global.exch.b64 %rd3, [%rd9], 0;\n\
                       selp.b64 %rd4, %rd9, %rd3, %p1;\nst.global.u32 [%rd4], 0;";
        assert_eq!(load("ld.global.u32", swapped), from("ld.global.u32"));
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
    fn a_branch_no_path_reaches_selects_nothing() {
        // The block after the first arm's `bra` is dead; its branch on
        // %tid.x would gate the merge if it counted.
        let body = "setp.lt.u32 %p1, %r0, 4;\n@%p1 bra $L__A;\nmov.u32 %r2, 1;\nbra $L__J;\n\
                    setp.eq.u32 %p3, %r1, 0;\n@%p3 bra $L__J;\n\
                    $L__A:\nmov.u32 %r2, 2;\n$L__J:";
        assert_eq!(uniform(body, "%r2"), Ok(()));
    }

    #[test]
    fn a_value_one_path_leaves_unwritten_is_divergent() {
        let body = "setp.lt.u32 %p1, %r0, 4;\n@%p1 bra $L__J;\nmov.u32 %r2, 7;\n$L__J:";
        let why = "reads %r2, which a path leaves unwritten".to_owned();
        assert_eq!(uniform(body, "%r2"), Err(why));
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
