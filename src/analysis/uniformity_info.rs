//! Which register reads the lanes executing a statement see alike, in
//! the CTA or in a warp: LLVM's UniformityAnalysis over PTX registers.
//! Divergence starts at per-thread special registers and at instructions
//! not computed lane by lane from their operands (loads, warp collectives,
//! tensor-core operations); a shuffle from one lane and a load at an
//! address a warp reads alike give that warp one value instead. Both
//! spread forward through the definitions and branch predicates that read
//! them, to a fixed point. Where definitions meet, the branches that
//! select between them, those whose successors' paths meet again there
//! (`control_flow::sync_dependence`), gate the value, as a γ's predicate
//! does in Gated SSA; a definition inside a loop read outside it is
//! gated by the loop's exits, as by an η's.

use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{Instr, Opcode, Operand, OperandId, Register, SpecialReg, Stmt};
use crate::ptx::literal::parse_int;
use crate::ptx::modifier::{Form, Memory, ParamAccess, Sem, ShflMode, Space, param_access};
use crate::ptx::register_effects::{is_pure, reads_cc, split_operands};
use crate::ptx::special_registers::{SpecialRegister, special_register};
use crate::ptx::thread_hierarchy::whole_warp_shuffle;
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

/// The threads a uniform read gives one value: every thread of the CTA,
/// or the lanes of a warp that execute it together, "all active threads
/// in a warp that are currently executing this instruction" (PTX ISA
/// §9.7.13.3). One value for the CTA is one for each of its warps, so
/// `Cta < Warp`, and the larger of two is what both give.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Uniform {
    Cta,
    Warp,
}

enum Read {
    Uniform(Uniform),
    Divergent(String),
    On(Vec<Node>),
}

pub(crate) struct UniformityInfo<'a> {
    data: &'a KernelAnalyses<'a>,
    divergent: HashMap<Node, String>,
    warp_uniform: HashSet<Node>,
}

impl<'a> UniformityInfo<'a> {
    pub(crate) fn new(t: &'a KernelAnalyses<'a>) -> Self {
        let mut reads: Vec<(Node, Read)> = Vec::new();
        let defs: BTreeSet<usize> = (t.writes.iter())
            .filter(|(r, _)| matches!(r, Register::Named(_)))
            .flat_map(|(_, ds)| ds.iter().copied())
            .collect();
        for d in defs {
            reads.extend(
                def_reads(t, d, false)
                    .into_iter()
                    .map(|r| (Node::Def(d), r)),
            );
        }
        for &d in t.writes.get(&Register::Cc).into_iter().flatten() {
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
        let mut warp_uniform = HashSet::new();
        let mut queue = Vec::new();
        for (node, r) in reads {
            match r {
                Read::Uniform(Uniform::Cta) => {}
                Read::Uniform(Uniform::Warp) => {
                    warp_uniform.insert(node);
                }
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
        let mut queue: Vec<Node> = warp_uniform.iter().copied().collect();
        while let Some(n) = queue.pop() {
            for &m in users.get(&n).into_iter().flatten() {
                if warp_uniform.insert(m) {
                    queue.push(m);
                }
            }
        }
        UniformityInfo {
            data: t,
            divergent,
            warp_uniform,
        }
    }

    /// What every lane executing definition `d` writes.
    pub(crate) fn def_uniformity(&self, d: usize) -> Result<Uniform, String> {
        self.of(&[Node::Def(d)])
    }

    /// How alike the branches between `reg`'s definition and `pos` (the
    /// exits of loops holding the definition but not `pos`, and those
    /// selecting among definitions) send the lanes; the reason when each
    /// lane may go its own way.
    pub(crate) fn gate_uniformity(&self, reg: Symbol, pos: usize) -> Result<Uniform, String> {
        match read(self.data, Register::Named(reg), pos) {
            Read::Uniform(_) => Ok(Uniform::Cta),
            Read::Divergent(why) => Err(why),
            Read::On(sources) => self.of(&(sources.into_iter())
                .filter(|s| matches!(s, Node::Branch(_)))
                .collect::<Vec<_>>()),
        }
    }

    /// How alike the lanes in loop `id` leave it: at the same iteration
    /// where its exits are uniform; the reason when not.
    pub(crate) fn exit_uniformity(&self, id: LoopId) -> Result<Uniform, String> {
        self.of(&exits(self.data, id).collect::<Vec<_>>())
    }

    /// How alike the lanes executing `pos` read `reg`; the reason when
    /// each may read its own value.
    pub(crate) fn register_uniformity(&self, reg: Symbol, pos: usize) -> Result<Uniform, String> {
        self.answer(read(self.data, Register::Named(reg), pos))
    }

    /// How alike the lanes executing `pos` see value `v`, wherever it
    /// was read: its definitions, and the loop exits and branches
    /// between them and `pos`.
    pub(crate) fn value_uniformity(&self, v: Value, pos: usize) -> Result<Uniform, String> {
        self.answer(read_value(self.data, v, pos))
    }

    fn answer(&self, r: Read) -> Result<Uniform, String> {
        match r {
            Read::Uniform(u) => Ok(u),
            Read::Divergent(why) => Err(why),
            Read::On(sources) => self.of(&sources),
        }
    }

    /// What reading every one of `nodes` gives.
    fn of(&self, nodes: &[Node]) -> Result<Uniform, String> {
        if let Some(why) = nodes.iter().find_map(|n| self.divergent.get(n)) {
            return Err(why.clone());
        }
        Ok(match nodes.iter().any(|n| self.warp_uniform.contains(n)) {
            true => Uniform::Warp,
            false => Uniform::Cta,
        })
    }
}

/// What the registers statement `d` writes depend on, or with
/// `carry_out` the CC.CF it writes: its inputs and, when guarded, the
/// guard and the value it keeps where the guard is false.
fn def_reads(t: &KernelAnalyses, d: usize, carry_out: bool) -> Vec<Read> {
    let Stmt::Instr(instr) = &t.kernel.stmts[d] else {
        return vec![Read::Divergent(
            "comes from a statement not parsed".to_owned(),
        )];
    };
    let lane_wise = lane_wise(t, instr);
    let (written, inputs) = split_operands(t.module, instr);
    let mut rs = match lane_wise {
        true => operand_reads(t, data_inputs(t, instr, inputs), d),
        false => match warp_reads(t, instr, inputs, d) {
            Some(rs) => rs,
            None => {
                return vec![Read::Divergent(format!(
                    "comes from `{}`",
                    t.module.opcode(instr)
                ))];
            }
        },
    };
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

/// What a result one for the lanes of a warp executing it together
/// reads, where `instr` gives one: `shfl.sync.idx` from a lane they read
/// alike in one segment clamped at 31, "j = minLane | (bval[4:0] &
/// ~segmask[4:0])" (PTX ISA §9.7.9.6); and a weak load from global,
/// shared or constant memory at an address they read alike, which the
/// hardware executes once for the warp, as LLVM's NVPTX and AMDGPU
/// targets take it (`isSourceOfDivergence`).
fn warp_reads(
    t: &KernelAnalyses,
    instr: &Instr,
    inputs: &[OperandId],
    d: usize,
) -> Option<Vec<Read>> {
    let module = t.module;
    let reads = match (instr.mnemonic, Form::of(module, instr), inputs) {
        (Opcode::Shfl, Form::Shuffle(ShflMode::Idx), [_, lane, c, _]) => {
            let Operand::Immediate(c) = module.operand(*c) else {
                return None;
            };
            let c = parse_int(module.interner.resolve(*c))?;
            if !whole_warp_shuffle(c) {
                return None;
            }
            operand_reads(t, &[*lane], d)
        }
        (
            Opcode::Ld,
            Form::Memory(Memory {
                space: Some(Space::Global | Space::Shared | Space::Const),
                sem: Sem::Weak,
                ..
            }),
            _,
        ) => operand_reads(t, data_inputs(t, instr, inputs), d),
        _ => return None,
    };
    Some(
        reads
            .into_iter()
            .chain([Read::Uniform(Uniform::Warp)])
            .collect(),
    )
}

/// The `inputs` of `instr` its result is computed from: an `ld`'s but
/// the cache_policy `.L2::cache_hint` adds last, "a performance hint
/// only" (PTX ISA §9.7.9.8, §9.7.9.9).
fn data_inputs<'m>(t: &KernelAnalyses, instr: &Instr, inputs: &'m [OperandId]) -> &'m [OperandId] {
    let hint = || {
        matches!(
            Form::of(t.module, instr),
            Form::Memory(Memory {
                cache_hint: true,
                ..
            })
        )
    };
    match inputs {
        [address @ .., _] if instr.mnemonic == Opcode::Ld && hint() => address,
        _ => inputs,
    }
}

fn read(t: &KernelAnalyses, reg: Register, pos: usize) -> Read {
    match t.ssa.value_before(pos, reg) {
        Some(v) => read_value(t, v, pos),
        None => Read::Divergent(format!("reads {} in a block no path reaches", name(t, reg))),
    }
}

fn name<'a>(t: &'a KernelAnalyses, reg: Register) -> &'a str {
    match reg {
        Register::Named(r) => t.module.interner.resolve(r),
        Register::Cc => "CC.CF",
    }
}

/// A read at `pos` of value `v`.
fn read_value(t: &KernelAnalyses, v: Value, pos: usize) -> Read {
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
    let (merges, left) = merges(t, v);
    read_sites(t, name, &defs, pos, node, &merges, &left)
}

/// The blocks of the φs a read of `v` passes through, and the loops
/// whose values they take from outside them.
fn merges(t: &KernelAnalyses, v: Value) -> (Vec<BlockId>, Vec<LoopId>) {
    let (mut seen, mut work, mut out, mut left) = (HashSet::new(), vec![v], Vec::new(), Vec::new());
    while let Some(x) = work.pop() {
        if let Value::Argument { block, .. } = x
            && seen.insert(x)
        {
            out.push(block);
            for (_, w) in t.ssa.incoming(x) {
                let from = match w {
                    Value::Result { op, .. } => Some(t.block_of(op)),
                    Value::Argument { block, .. } => Some(block),
                    Value::Undef(_) => None,
                };
                let mut l = from.and_then(|b| t.forest.block_loop[b.0 as usize]);
                while let Some(id) = l {
                    if !t.forest.get(id).blocks.contains(&block) {
                        left.push(id);
                    }
                    l = t.forest.get(id).parent;
                }
                work.push(w);
            }
        }
    }
    (out, left)
}

/// A read at `pos` of `name`, whose value the statements `defs` may
/// have written, each the source `node` names, through φs in blocks
/// `merges`.
fn read_sites(
    t: &KernelAnalyses,
    name: &str,
    defs: &[usize],
    pos: usize,
    node: fn(usize) -> Node,
    merges: &[BlockId],
    left: &[LoopId],
) -> Read {
    let here = t.block_of(pos);
    let reaches = |a: BlockId, b: BlockId| a == b || t.reach[a.0 as usize][b.0 as usize];
    let mut sources: Vec<Node> = defs.iter().map(|&d| node(d)).collect();
    // An η's gating predicates: the exits of every loop holding a
    // definition but not the read, or a value a φ outside it takes. A
    // cycle with more than one entry is no loop of the forest, and its
    // exits are not known.
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
    for &id in left {
        sources.extend(exits(t, id));
    }
    // A γ's gating predicates: branches whose successors' paths meet
    // again at one of the φs.
    if defs.len() > 1 {
        let blocks = (0..t.cfg.blocks.len() as u32).map(BlockId);
        let reachable = blocks.filter(|&b| t.forest.doms.is_reachable(b));
        sources.extend(
            reachable
                .filter(|b| (t.joins[b.0 as usize].iter()).any(|j| merges.contains(j)))
                .map(Node::Branch),
        );
    }
    Read::On(sources)
}

/// The branches that decide at which trip, or by which exit, a thread
/// leaves loop `id`.
fn exits<'a>(t: &'a KernelAnalyses, id: LoopId) -> impl Iterator<Item = Node> + 'a {
    (t.exit_branches[id.0 as usize].iter()).map(|&b| Node::Branch(b))
}

/// Each lane's result is a function of its own operands: a kernel
/// parameter is read-only and one per grid, a function's `.param`, a
/// call's return value among them, one per thread (PTX ISA §5.1.6,
/// Table 7).
fn lane_wise(t: &KernelAnalyses, instr: &Instr) -> bool {
    is_pure(instr.mnemonic) || kernel_param(t, instr).is_some()
}

/// The kernel parameter an `ld.param` reads.
fn kernel_param(t: &KernelAnalyses, instr: &Instr) -> Option<usize> {
    match param_access(t.module, t.kernel, instr)? {
        ParamAccess::KernelParam { index, .. } => Some(index),
        _ => None,
    }
}

fn read_special(reg: SpecialReg) -> Read {
    let name = reg.name();
    match special_register(reg) {
        Some(SpecialRegister::PerCta | SpecialRegister::Zero) => Read::Uniform(Uniform::Cta),
        Some(SpecialRegister::PerThread) => Read::Divergent(format!("depends on {name}")),
        Some(SpecialRegister::Varying) => {
            Read::Divergent(format!("{name} can change between reads"))
        }
        None => Read::Divergent(format!("{name} is not modelled")),
    }
}

fn operand_reads(t: &KernelAnalyses, ops: &[OperandId], pos: usize) -> Vec<Read> {
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

    /// `Ok` where the CTA reads one value; the reason where it does not.
    fn cta_uniform(r: Result<Uniform, String>) -> Result<(), String> {
        match r? {
            Uniform::Cta => Ok(()),
            Uniform::Warp => Err("is one value per warp only".to_owned()),
        }
    }

    /// How alike the lanes read `reg` where the first `st.global.u32`
    /// reads it, one after `body` if it has none; `%r0` holds a parameter
    /// and `%r1` the thread index.
    fn uniformity_in(body: &str, reg: &str) -> Result<Uniform, String> {
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
        uniformity_at(&src, reg, None, "st.global.u32")
    }

    fn cta_uniform_in(body: &str, reg: &str) -> Result<(), String> {
        cta_uniform(uniformity_in(body, reg))
    }

    /// How alike the lanes read `reg` at the first `opcode` after `label`.
    fn uniformity_at(
        src: &str,
        reg: &str,
        label: Option<&str>,
        opcode: &str,
    ) -> Result<Uniform, String> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let forest = loop_forest(&cfg);
        let t = KernelAnalyses::new(&m, k, &cfg, &forest);
        let start = label.map_or(0, |l| {
            k.stmts
                .iter()
                .position(
                    |s| matches!(s, Stmt::Label { name, .. } if m.interner.resolve(*name) == l),
                )
                .expect("the label")
        });
        let pos = (start..k.stmts.len())
            .find(|&i| matches!(&k.stmts[i], Stmt::Instr(i) if m.opcode(i) == opcode))
            .expect("the read");
        UniformityInfo::new(&t).register_uniformity(m.interner.get(reg).expect("register"), pos)
    }

    fn cta_uniform_at(
        src: &str,
        reg: &str,
        label: Option<&str>,
        opcode: &str,
    ) -> Result<(), String> {
        cta_uniform(uniformity_at(src, reg, label, opcode))
    }

    fn tid() -> Result<(), String> {
        Err("depends on %tid.x".to_owned())
    }

    #[test]
    fn divergence_follows_data_from_per_thread_registers_and_loads() {
        assert_eq!(cta_uniform_in("shl.b32 %r2, %r1, 2;", "%r2"), tid());
        assert_eq!(cta_uniform_in("", "%r0"), Ok(()));
        let cta = "mov.u32 %r2, %ctaid.x;\nmad.lo.s32 %r3, %r2, %r0, 7;";
        assert_eq!(cta_uniform_in(cta, "%r3"), Ok(()));
        assert_eq!(
            cta_uniform_in("ld.local.u32 %r2, [%rd1];\nadd.s32 %r3, %r2, 1;", "%r3"),
            Err("comes from `ld.local.u32`".to_owned())
        );
        // The carry out of a per-thread sum: 1 in lanes 16 and up.
        let carry = "add.cc.u32 %r2, %r1, 0xfffffff0;\naddc.u32 %r3, 0, 0;";
        assert_eq!(cta_uniform_in(carry, "%r3"), tid());
        let carry = "add.cc.u32 %r2, %r0, 0xfffffff0;\naddc.u32 %r3, 0, 0;";
        assert_eq!(cta_uniform_in(carry, "%r3"), Ok(()));
        // A guarded `add.cc` leaves the per-thread carry where it is skipped.
        let carry = "add.cc.u32 %r2, %r1, 0xfffffff0;\nsetp.eq.u32 %p1, %r0, 0;\n\
                     mov.u32 %r4, 0;\n@%p1 add.cc.u32 %r4, %r0, 1;\naddc.u32 %r3, 0, 0;";
        assert_eq!(cta_uniform_in(carry, "%r3"), tid());
        // The carry it keeps is not part of its register result.
        assert_eq!(cta_uniform_in(carry, "%r4"), Ok(()));
        // PTX ISA §9.7.2: calls do not preserve CC.CF, and neither does
        // a statement the parser cannot read.
        let carry = "add.cc.u32 %r2, %r0, 1;\ncall (%r5), f, (%r0);\naddc.u32 %r3, 0, 0;";
        assert_eq!(
            cta_uniform_in(carry, "%r3"),
            Err("comes from `call`".to_owned())
        );
        let carry = "add.cc.u32 %r2, %r0, 1;\nfuture.op %r5;\naddc.u32 %r3, 0, 0;";
        assert_eq!(
            cta_uniform_in(carry, "%r3"),
            Err("comes from a statement not parsed".to_owned())
        );
        // One after the last terminator is in no block, and never runs.
        let src = ".version 8.7\n.target sm_80\n.address_size 64\n\
                   .visible .entry k()\n{\nadd.cc.u32 %r0, 0, 0;\nbra L;\nL:\n\
                   addc.u32 %r1, 0, 0;\nret;\ncall (%r2), f, ();\n}\n";
        assert_eq!(cta_uniform_at(src, "%r1", Some("L"), "ret"), Ok(()));
    }

    /// A weak global load is one value for a warp's lanes where its
    /// address is, whatever the kernel stores; `%rd1` is param_0 +
    /// 4·%ctaid.x.
    #[test]
    fn a_global_load_at_one_address_is_one_value_per_warp() {
        let load = |ld: &str, rest: &str| {
            let src = format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0,\n.param .u64 k_param_1\n)\n{{\n\
                 ld.param.u64 %rd0, [k_param_0];\nld.param.u64 %rd9, [k_param_1];\n\
                 mov.u32 %r1, %ctaid.x;\nmul.wide.u32 %rd2, %r1, 4;\nadd.s64 %rd1, %rd0, %rd2;\n\
                 {ld} %r2, [%rd1];\n{rest}\nst.shared.u32 [%r9], %r2;\nret;\n}}\n"
            );
            uniformity_at(&src, "%r2", None, "st.shared.u32")
        };
        let from = |op: &str| Err(format!("comes from `{op}`"));
        assert_eq!(load("ld.global.u32", ""), Ok(Uniform::Warp));
        assert_eq!(
            load("ld.global.nc.L1::evict_last.u32", ""),
            Ok(Uniform::Warp)
        );
        let written = "st.global.u32 [%rd1], 0;";
        assert_eq!(load("ld.global.u32", written), Ok(Uniform::Warp));
        let volatile = "ld.volatile.global.u32";
        assert_eq!(load(volatile, ""), from(volatile));
        assert_eq!(load("ld.u32", ""), from("ld.u32"));
    }

    /// `%r1` is %tid.x; `%r4` its warp's index, from lane 0.
    #[test]
    fn a_shuffle_from_one_lane_and_a_load_at_its_address_are_one_per_warp() {
        let warp = "shr.u32 %r5, %r1, 5;\nshfl.sync.idx.b32 %r4, %r5, 0, 31, -1;";
        let read = |body: &str| uniformity_in(&format!("{warp}\n{body}"), "%r3");
        let from = |op: &str| Err(format!("comes from `{op}`"));
        assert_eq!(read("mov.u32 %r3, %r4;"), Ok(Uniform::Warp));
        assert_eq!(
            read("shfl.sync.idx.b32 %r3, %r1, %r4, 31, -1;"),
            Ok(Uniform::Warp)
        );
        let by_lane = "shfl.sync.idx.b32 %r3, %r0, %r1, 31, -1;";
        assert_eq!(read(by_lane), Err("depends on %tid.x".to_owned()));
        let shfl = "shfl.sync.idx.b32";
        assert_eq!(
            read("shfl.sync.idx.b32 %r3, %r0, 0, 0x81f, -1;"),
            from(shfl)
        );
        assert_eq!(read("shfl.sync.idx.b32 %r3, %r0, 0, 30, -1;"), from(shfl));
        assert_eq!(read("ld.shared.u32 %r3, [%r4];"), Ok(Uniform::Warp));
        assert_eq!(
            read("ld.shared.u32 %r3, [%r1];"),
            Err("depends on %tid.x".to_owned())
        );
        assert_eq!(read("ld.local.u32 %r3, [%r4];"), from("ld.local.u32"));
        let strong = "ld.relaxed.cta.shared.u32";
        assert_eq!(read("ld.relaxed.cta.shared.u32 %r3, [%r4];"), from(strong));
    }

    /// The lightllm decode kernels' loads under a `createpolicy` policy,
    /// whose result is not lane-wise.
    #[test]
    fn a_load_does_not_read_its_cache_policy() {
        let load = |ld: &str, read: &str| {
            let src = format!(
                ".version 8.7\n.target sm_80\n.address_size 64\n\
                 .visible .entry k(\n.param .u64 k_param_0\n)\n{{\n\
                 ld.param.u64 %rd0, [k_param_0];\n\
                 createpolicy.fractional.L2::evict_last.b64 %rd1, 1.0;\n\
                 {ld} %r2, [%rd0], %rd1;\nadd.s64 %rd2, %rd1, 0;\n\
                 st.shared.u32 [%r9], {read};\nret;\n}}\n"
            );
            uniformity_at(&src, read, None, "st.shared.u32")
        };
        let cached = "ld.global.L1::evict_last.L2::cache_hint.u32";
        assert_eq!(load(cached, "%r2"), Ok(Uniform::Warp));
        let nc = "ld.global.nc.L2::cache_hint.u32";
        assert_eq!(load(nc, "%r2"), Ok(Uniform::Warp));
        let policy = "createpolicy.fractional.L2::evict_last.b64";
        assert_eq!(load(cached, "%rd2"), Err(format!("comes from `{policy}`")));
    }

    #[test]
    fn a_guarded_write_keeps_the_old_value_in_lanes_it_skips() {
        let on_tid = "setp.lt.u32 %p1, %r1, 16;\nmov.u32 %r2, 0;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(cta_uniform_in(on_tid, "%r2"), tid());
        let on_param = "setp.lt.u32 %p1, %r0, 16;\nmov.u32 %r2, 0;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(cta_uniform_in(on_param, "%r2"), Ok(()));
        let old_on_tid = "setp.lt.u32 %p1, %r0, 16;\nmov.u32 %r2, %r1;\n@%p1 mov.u32 %r2, 1;";
        assert_eq!(cta_uniform_in(old_on_tid, "%r2"), tid());
    }

    #[test]
    fn a_join_is_divergent_when_a_divergent_branch_selects_the_definition() {
        let join = |on: &str| {
            format!(
                "setp.lt.u32 %p1, {on}, 16;\n@%p1 bra $L__A;\nmov.u32 %r2, 64;\n\
                 bra.uni $L__J;\n$L__A:\nmov.u32 %r2, 0;\n$L__J:"
            )
        };
        assert_eq!(cta_uniform_in(&join("%r1"), "%r2"), tid());
        assert_eq!(cta_uniform_in(&join("%r0"), "%r2"), Ok(()));
        // A divergent branch whose other side never reaches the read
        // selects nothing (k1's bounds check around its K loop).
        let guarded = format!(
            "setp.lt.u32 %p3, %r1, 100;\n@%p3 bra $L__X;\n{}\n\
             st.global.u32 [%rd9], %r2;\n$L__X:",
            join("%r0")
        );
        assert_eq!(cta_uniform_in(&guarded, "%r2"), Ok(()));
    }

    #[test]
    fn a_counter_is_cta_uniform_in_its_loop_unless_a_divergent_branch_skips_its_step() {
        let looped = |skip: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__L:\nst.global.u32 [%rd9], %r2;\n\
                 setp.lt.u32 %p2, {skip}, 3;\n@%p2 bra $L__S;\nadd.s32 %r2, %r2, 1;\n\
                 $L__S:\nsetp.lt.u32 %p1, %r2, %r0;\n@%p1 bra $L__L;"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
    }

    /// A spin-wait in a loop sends every thread on to the same block, and
    /// selects none of the values the loop carries.
    #[test]
    fn a_spin_wait_selects_nothing() {
        let body = "mov.u32 %r2, %r0;\n$L__H:\nsetp.ge.u32 %p1, %r2, 4;\n@%p1 bra $L__X;\n\
                    $L__W:\nld.volatile.shared.u32 %r4, [%rd2];\nsetp.eq.u32 %p2, %r4, 0;\n\
                    @%p2 bra $L__W;\nst.global.u32 [%rd9], %r2;\nadd.s32 %r2, %r2, %r0;\n\
                    bra.uni $L__H;\n$L__X:";
        assert_eq!(cta_uniform_in(body, "%r2"), Ok(()));
    }

    /// Two latches meet at the header. Like LLVM, which marks every φ of
    /// a join block, the branch between them also selects a φ both
    /// latches carry the same value to.
    #[test]
    fn two_latches_select_at_the_header() {
        let looped = |on: &str| {
            format!(
                "mov.u32 %r2, 0;\nmov.u32 %r5, 0;\n$L__H:\nst.global.u32 [%rd9], %r2;\n\
                 setp.ge.u32 %p1, %r5, 2;\n@%p1 bra $L__X;\nadd.s32 %r5, %r5, 1;\n\
                 setp.lt.u32 %p2, {on}, 16;\n@%p2 bra $L__W;\nbra.uni $L__H;\n\
                 $L__W:\nxor.b32 %r2, %r2, 1;\nbra.uni $L__H;\n$L__X:"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r5"), tid());
    }

    /// The threads a latch's branch sends around again leave through the
    /// other exit, and meet those it lets out after the loop.
    #[test]
    fn going_around_again_selects_after_the_loop() {
        let looped = |on: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__H:\nsetp.eq.s32 %p1, %r2, 2;\n@%p1 bra $L__A;\n\
                 add.s32 %r2, %r2, 1;\nsetp.ge.u32 %p2, {on}, 16;\n@%p2 bra $L__H;\n\
                 mov.u32 %r3, 128;\nbra.uni $L__J;\n$L__A:\nmov.u32 %r3, 0;\n$L__J:"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r3"), tid());
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r3"), Ok(()));
    }

    /// A branch that skips an exit test lets the threads leave at
    /// different trips, though every exit compares the counter alike.
    #[test]
    fn a_branch_that_skips_an_exit_test_is_an_exit() {
        let looped = |on: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__H:\nadd.s32 %r2, %r2, 1;\nsetp.ge.u32 %p1, {on}, 16;\n\
                 @%p1 bra $L__L;\nsetp.eq.u32 %p2, %r2, 3;\n@%p2 bra $L__X;\n\
                 $L__L:\nsetp.lt.u32 %p3, %r2, 4;\n@%p3 bra $L__H;\n$L__X:"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
    }

    /// A header's branch that skips the only exit test for some threads:
    /// they leave at trip max(3, %tid.x + 1).
    #[test]
    fn a_header_branch_that_skips_the_exit_test_is_an_exit() {
        let looped = |on: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__H:\nsetp.le.s32 %p1, %r2, {on};\n@%p1 bra $L__L;\n\
                 setp.ge.s32 %p2, %r2, 3;\n@%p2 bra $L__X;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                 bra.uni $L__H;\n$L__X:"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
    }

    /// A value an inner loop leaves carried into its next run by the
    /// outer loop: lane 0 leaves the first run before the value is 1, the
    /// other lanes after.
    #[test]
    fn a_value_carried_out_of_an_inner_loop_and_back_is_gated_by_its_exits() {
        let looped = |on: &str| {
            format!(
                "mov.u32 %r2, 0;\nmov.u32 %r4, 0;\n$L__H2:\nmov.u32 %r3, 0;\n$L__H1:\n\
                 st.global.u32 [%rd9], %r2;\nsetp.eq.u32 %p1, {on}, 0;\n@%p1 bra $L__E1;\n\
                 add.s32 %r3, %r3, 1;\nsetp.ge.u32 %p2, %r3, 4;\n@%p2 bra $L__E1;\n\
                 mov.u32 %r2, 1;\nbra.uni $L__H1;\n$L__E1:\nadd.s32 %r4, %r4, 1;\n\
                 setp.lt.u32 %p3, %r4, 2;\n@%p3 bra $L__H2;"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
    }

    /// A value a uniform branch sets before a loop and the loop toggles:
    /// the lanes still in the loop agree on it, though they leave it at
    /// different trips.
    #[test]
    fn a_value_merged_before_a_loop_is_not_gated_by_its_exits() {
        let body = "setp.lt.u32 %p1, %r0, 4;\n@%p1 bra $L__A;\nmov.u32 %r2, 0;\nbra.uni $L__P;\n\
                    $L__A:\nmov.u32 %r2, 1;\n$L__P:\nmov.u32 %r3, 0;\n$L__H:\n\
                    st.global.u32 [%rd9], %r2;\nxor.b32 %r2, %r2, 1;\nadd.s32 %r3, %r3, 1;\n\
                    setp.lt.u32 %p2, %r3, %r1;\n@%p2 bra $L__H;";
        assert_eq!(cta_uniform_in(body, "%r2"), Ok(()));
    }

    #[test]
    fn a_branch_no_path_reaches_selects_nothing() {
        // The block after the first arm's `bra` is dead; its branch on
        // %tid.x would gate the merge if it counted.
        let body = "setp.lt.u32 %p1, %r0, 4;\n@%p1 bra $L__A;\nmov.u32 %r2, 1;\nbra $L__J;\n\
                    setp.eq.u32 %p3, %r1, 0;\n@%p3 bra $L__J;\n\
                    $L__A:\nmov.u32 %r2, 2;\n$L__J:";
        assert_eq!(cta_uniform_in(body, "%r2"), Ok(()));
    }

    #[test]
    fn a_value_one_path_leaves_unwritten_is_divergent() {
        let body = "setp.lt.u32 %p1, %r0, 4;\n@%p1 bra $L__J;\nmov.u32 %r2, 7;\n$L__J:";
        let why = "reads %r2, which a path leaves unwritten".to_owned();
        assert_eq!(cta_uniform_in(body, "%r2"), Err(why));
    }

    #[test]
    fn a_value_from_a_loop_is_divergent_after_it_when_its_exit_is() {
        let looped = |exit: &str| {
            format!(
                "mov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                 setp.lt.u32 %p1, %r2, {exit};\n@%p1 bra $L__L;"
            )
        };
        assert_eq!(cta_uniform_in(&looped("%r0"), "%r2"), Ok(()));
        assert_eq!(cta_uniform_in(&looped("%r1"), "%r2"), tid());
        // The lanes still in the loop agree on it.
        let inside = looped("%r1").replace("add.s32", "st.global.u32 [%rd9], %r2;\nadd.s32");
        assert_eq!(cta_uniform_in(&inside, "%r2"), Ok(()));
    }

    /// A cycle entered at A or B: no loop of the forest, so its exits do
    /// not gate the counter, which is divergent after it.
    #[test]
    fn a_value_from_a_cycle_with_two_entries_is_divergent_after_it() {
        let body = "mov.u32 %r2, 0;\nsetp.lt.u32 %p1, %r0, 5;\n@%p1 bra $L__B;\n\
                    $L__A:\nadd.s32 %r2, %r2, 1;\n$L__B:\nadd.s32 %r2, %r2, 1;\n\
                    setp.lt.u32 %p2, %r2, %r1;\n@%p2 bra $L__A;";
        assert_eq!(
            cta_uniform_in(body, "%r2"),
            Err("%r2 is read after a cycle defining it".to_owned())
        );
    }

    /// k1's remainder loop starts from the main loop's last counter, or 0
    /// when K < 4 skips it; both choices compare parameters only, and the
    /// bounds check on the thread index skips both loops.
    #[test]
    fn k1s_counter_after_its_main_loop_is_cta_uniform() {
        let k1 = include_str!("../../tests/fixtures/k1/k1.sm_89.ptx");
        assert_eq!(
            cta_uniform_at(k1, "%r34", Some("$L__BB0_5"), "mad.lo.s32"),
            Ok(())
        );
    }
}
