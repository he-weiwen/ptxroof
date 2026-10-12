//! The scalar affine tracer: where a register's value at a statement
//! comes from, and that value as an affine form over thread and CTA
//! indices, loop counters and the kernel parameters. The trip matcher
//! reads latch conditions through it; address and branch-condition
//! analyses read memory operands and predicates through the same code.
//!
//! Reaching definitions are computed on the dominator chain with a
//! check that no other definition lies on a path that avoids the
//! dominator: a value redefined inside a loop containing the read is
//! loop-carried (a counter read before its increment is the previous
//! iteration's value), and definitions meeting from several paths are
//! refused. The tracer follows supported arithmetic and moves
//! down to `ld.param`, constants and special registers; unsupported operations
//! is a named reason, preferring the fundamental obstacle (a special
//! register, a memory load, an atomic) behind arithmetic it does not
//! read. The domain is nonnegative and non-overflowing, as documented
//! in [`super::trip_counts`].

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::scalar::affine::{Affine, Axis, Var};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{
    Element, Instr, Kernel, Module, Opcode, Operand, OperandId, SpecialReg, SpecialScalar,
    SpecialVector, Stmt,
};
use crate::ptx::literal::parse_int;
use crate::support::intern::Symbol;
use std::collections::{HashMap, HashSet};

/// Where a register's value at a statement comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReachingDefinition {
    /// One definition, and no other can reach the statement.
    Def(usize),
    /// Defined again inside a loop containing the statement: the value
    /// is the header's merge of the entry value and the back edge.
    Carried(LoopId),
    /// Definitions from more than one path meet before the statement:
    /// these ones.
    Merged(Vec<usize>),
    None,
}

pub(crate) struct AffineValueTracer<'a> {
    pub(crate) module: &'a Module,
    pub(crate) kernel: &'a Kernel,
    pub(crate) cfg: &'a ControlFlowGraph,
    pub(crate) forest: &'a LoopForest,
    /// Definition sites (statement indices) per register symbol.
    pub(crate) defs: HashMap<Symbol, Vec<usize>>,
    /// param symbol -> positional index.
    pub(crate) params: HashMap<Symbol, usize>,
    /// `reach[a][b]`: a path of at least one edge from block a to b.
    pub(crate) reach: Vec<Vec<bool>>,
    /// Per loop: counter register ↦ (step, the chain register defined
    /// before the loop).
    pub(crate) ivs: Vec<HashMap<Symbol, (SymExpr, Symbol)>>,
    /// `--bind` values, applied where a trace needs a constant.
    pub(crate) bindings: Option<&'a HashMap<String, i64>>,
}

impl<'a> AffineValueTracer<'a> {
    pub(crate) fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        cfg: &'a ControlFlowGraph,
        forest: &'a LoopForest,
    ) -> Self {
        let mut defs: HashMap<Symbol, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            if let Stmt::Instr(instr) = stmt
                && let Some(&first) = module.operand_ids(instr.operands).first()
                && defines_dest(instr.mnemonic)
            {
                let written = match module.operand(first) {
                    Operand::Register(_) => std::slice::from_ref(&first),
                    Operand::MultipleDestinations { children }
                    | Operand::VectorList { children } => module.operand_ids(*children),
                    _ => &[],
                };
                for &id in written {
                    if let Operand::Register(reg) = module.operand(id) {
                        defs.entry(*reg).or_default().push(i);
                    }
                }
            }
        }
        let params = kernel
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name, i))
            .collect();
        let n = cfg.blocks.len();
        let mut reach = vec![vec![false; n]; n];
        for (a, row) in reach.iter_mut().enumerate() {
            let mut stack: Vec<BlockId> = cfg.block(BlockId(a as u32)).succs.clone();
            while let Some(b) = stack.pop() {
                if !row[b.0 as usize] {
                    row[b.0 as usize] = true;
                    stack.extend(cfg.block(b).succs.iter().copied());
                }
            }
        }
        let mut t = AffineValueTracer {
            module,
            kernel,
            cfg,
            forest,
            defs,
            params,
            reach,
            ivs: Vec::new(),
            bindings: None,
        };
        // Outer loops first: a step register may be an enclosing counter.
        t.ivs = vec![HashMap::new(); forest.loops.len()];
        let mut order: Vec<LoopId> = (0..forest.loops.len() as u32).map(LoopId).collect();
        order.sort_by_key(|&l| forest.get(l).depth);
        for l in order {
            let m = t.induction_vars(l);
            t.ivs[l.0 as usize] = m;
        }
        t
    }

    /// Every one of these definitions is `mov r, imm` with one immediate.
    fn same_immediate(&self, defs: &[usize]) -> Option<i64> {
        let mut value = None;
        for &d in defs {
            let Stmt::Instr(instr) = &self.kernel.stmts[d] else {
                return None;
            };
            let [dst, src] = self.module.operand_ids(instr.operands) else {
                return None;
            };
            let (Operand::Register(_), Operand::Immediate(text)) =
                (self.module.operand(*dst), self.module.operand(*src))
            else {
                return None;
            };
            if instr.mnemonic != Opcode::Mov {
                return None;
            }
            let c = parse_int(self.module.interner.resolve(*text))?;
            if value.is_some_and(|v| v != c) {
                return None;
            }
            value = Some(c);
        }
        value
    }

    /// A merged value: the obstacle on any of its definitions' chains.
    pub(crate) fn obstacle_of_any_def(&self, reg: Symbol, id: Option<LoopId>) -> Option<String> {
        let mut seen = HashSet::new();
        seen.insert(reg);
        self.defs.get(&reg)?.iter().find_map(|&d| {
            let Stmt::Instr(instr) = &self.kernel.stmts[d] else {
                return None;
            };
            self.module
                .operand_ids(instr.operands)
                .iter()
                .skip(1)
                .find_map(|&op| self.operand_obstacle(op, d, id, &mut seen))
        })
    }

    /// In-loop registers whose ONLY in-loop definition adds a constant
    /// to themselves → reg ↦ (step, phi). The addition may pass through
    /// other single-definition registers (`t = i + 2; i = t - 1`,
    /// nvcc's shape when the body also reads `i + 2`; `t = i + 1` then
    /// `mov i, t` in the latch, LLVM's two-register counter); the step
    /// is the sum of the constants along that chain, and phi is the
    /// chain's register defined before the loop.
    /// The bound block shape has one row and one plane.
    fn one_dimensional(&self) -> bool {
        self.bindings.is_some_and(|b| {
            let [_, y, z] = SpecialVector::Ntid.components();
            b.get(&y.name()) == Some(&1) && b.get(&z.name()) == Some(&1)
        })
    }

    pub(crate) fn with_bindings(mut self, bindings: &'a HashMap<String, i64>) -> Self {
        self.bindings = Some(bindings);
        self
    }

    /// A constant, after the bindings if there are any.
    fn constant(&self, a: &Affine) -> Option<i64> {
        match self.bindings {
            Some(b) => a.bind(b).as_const(),
            None => a.as_const(),
        }
    }

    pub(crate) fn induction_vars(&self, id: LoopId) -> HashMap<Symbol, (SymExpr, Symbol)> {
        let l = self.forest.get(id);
        let header_start = self.cfg.block(l.header).start;
        let mut in_loop_defs: HashMap<Symbol, Vec<usize>> = HashMap::new();
        for &b in &l.blocks {
            let blk = self.cfg.block(b);
            for (i, stmt) in self.kernel.stmts[blk.start..blk.end].iter().enumerate() {
                if let Stmt::Instr(instr) = stmt
                    && let Some(&first) = self.module.operand_ids(instr.operands).first()
                    && let Operand::Register(reg) = self.module.operand(first)
                    && defines_dest(instr.mnemonic)
                {
                    in_loop_defs.entry(*reg).or_default().push(blk.start + i);
                }
            }
        }
        // A loop-invariant step register: its value at the increment is
        // a plain expression (4·N, say), whether it was computed before
        // the loop or recomputed from invariants inside it. This loop's
        // own counters are not known yet, so a read of one is refused.
        let invariant_step = |s: Symbol, site: usize| -> Option<SymExpr> {
            let v = self.trace_reg(s, site, Some(id), 0, None).ok()?;
            v.is_invariant().then_some(v.base)
        };
        let add_const = |site: usize| -> Option<(Symbol, SymExpr)> {
            let Stmt::Instr(instr) = &self.kernel.stmts[site] else {
                return None;
            };
            let ops = self.module.operand_ids(instr.operands);
            match (instr.mnemonic, ops) {
                (Opcode::Mov, [_, src]) => match self.module.operand(*src) {
                    Operand::Register(r) => Some((*r, SymExpr::Const(0))),
                    _ => None,
                },
                (Opcode::Add, [_, a, b]) => {
                    match (self.module.operand(*a), self.module.operand(*b)) {
                        (Operand::Register(r), Operand::Immediate(c))
                        | (Operand::Immediate(c), Operand::Register(r)) => Some((
                            *r,
                            SymExpr::Const(parse_int(self.module.interner.resolve(*c))?),
                        )),
                        (Operand::Register(r), Operand::Register(s)) => invariant_step(*s, site)
                            .map(|step| (*r, step))
                            .or_else(|| invariant_step(*r, site).map(|step| (*s, step))),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        let mut ivs = HashMap::new();
        for (reg, sites) in &in_loop_defs {
            let [mut site] = sites[..] else { continue };
            // The increment must run every iteration.
            let own = self.block_of(site);
            if !l
                .latches
                .iter()
                .all(|&lt| self.forest.doms.dominates(own, lt))
            {
                continue;
            }
            let mut step = SymExpr::Const(0);
            let mut chain = vec![*reg];
            for _ in 0..8 {
                let Some((src, c)) = add_const(site) else {
                    break;
                };
                step = SymExpr::add(step, c);
                if src == *reg {
                    let phi = chain.iter().copied().find(|&r| {
                        matches!(
                            self.reach_def(r, header_start, Some(id)),
                            ReachingDefinition::Def(_) | ReachingDefinition::Merged(_)
                        )
                    });
                    if let Some(phi) = phi
                        && step != SymExpr::Const(0)
                    {
                        ivs.insert(*reg, (step, phi));
                    }
                    break;
                }
                let Some([next]) = in_loop_defs.get(&src).map(|s| &s[..]) else {
                    break;
                };
                chain.push(src);
                site = *next;
            }
        }
        ivs
    }

    /// The counter's value on entry to loop `l`: its chain register's
    /// reaching definition at the header, ignoring the loop's own.
    pub(crate) fn iv_init(&self, reg: Symbol, l: LoopId, depth: u32) -> Result<Affine, String> {
        let hdr = self.cfg.block(self.forest.get(l).header);
        self.trace_reg(reg, hdr.start, Some(l), depth, Some(l))
    }

    pub(crate) fn last_instr(&self, b: BlockId) -> Option<(usize, &Instr)> {
        let blk = self.cfg.block(b);
        self.kernel.stmts[blk.start..blk.end]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, s)| match s {
                Stmt::Instr(instr) => Some((blk.start + i, instr)),
                _ => None,
            })
    }

    pub(crate) fn find_setp(
        &self,
        b: BlockId,
        before: usize,
        pred: Symbol,
    ) -> Option<(usize, &Instr)> {
        let blk = self.cfg.block(b);
        self.kernel.stmts[blk.start..before]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, s)| match s {
                Stmt::Instr(instr)
                    if instr.mnemonic == Opcode::Setp
                        && self
                            .module
                            .operand_ids(instr.operands)
                            .first()
                            .is_some_and(|&id| {
                                matches!(self.module.operand(id),
                                         Operand::Register(r) if *r == pred)
                            }) =>
                {
                    Some((blk.start + i, instr))
                }
                _ => None,
            })
    }

    /// Reaching definition of `reg` strictly before statement `pos`:
    /// the latest def in the same block, else the latest def walking up
    /// the dominator chain. (Defs on non-dominating paths are shadowed
    /// by construction in the nvcc shapes; anything that depends on a
    /// merge degrades to unknown through the def-form rules below.)
    pub(crate) fn block_of(&self, stmt: usize) -> BlockId {
        (0..self.cfg.blocks.len() as u32)
            .map(BlockId)
            .find(|&b| {
                let blk = self.cfg.block(b);
                blk.start <= stmt && stmt < blk.end
            })
            .expect("statement belongs to a block")
    }

    /// A path from `from` to `to` of at least one edge that avoids `avoid`.
    pub(crate) fn path_avoiding(&self, from: BlockId, to: BlockId, avoid: BlockId) -> bool {
        let mut seen = vec![false; self.cfg.blocks.len()];
        let mut stack: Vec<BlockId> = self.cfg.block(from).succs.clone();
        while let Some(b) = stack.pop() {
            if b == avoid || seen[b.0 as usize] {
                continue;
            }
            if b == to {
                return true;
            }
            seen[b.0 as usize] = true;
            stack.extend(self.cfg.block(b).succs.iter().copied());
        }
        false
    }

    /// The definition of `reg` whose value statement `pos` reads: the
    /// latest in its own block, else the latest on the dominator chain,
    /// provided no other definition lies on a path from that dominator
    /// to `pos` that avoids the dominator. With `exclude`, definitions
    /// inside that loop are ignored: the value on entry to the loop.
    pub(crate) fn reach_def(
        &self,
        reg: Symbol,
        pos: usize,
        exclude: Option<LoopId>,
    ) -> ReachingDefinition {
        let Some(all) = self.defs.get(&reg) else {
            return ReachingDefinition::None;
        };
        let sites: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&d| !exclude.is_some_and(|l| self.in_loop(l, d)))
            .collect();
        let here = self.block_of(pos);
        let blk = self.cfg.block(here);
        if let Some(&d) = sites
            .iter()
            .rev()
            .find(|&&d| d >= blk.start && d < pos.min(blk.end))
        {
            return ReachingDefinition::Def(d);
        }
        let mut cur = self.forest.doms.idom[here.0 as usize];
        while let Some(b) = cur {
            let dblk = self.cfg.block(b);
            if let Some(&d) = sites
                .iter()
                .rev()
                .find(|&&d| d >= dblk.start && d < dblk.end)
            {
                let interfering: Vec<usize> = sites
                    .iter()
                    .copied()
                    .filter(|&o| {
                        let ob = self.block_of(o);
                        o != d
                            && ob != b
                            && self.reach[b.0 as usize][ob.0 as usize]
                            && self.path_avoiding(ob, here, b)
                    })
                    .collect();
                if interfering.is_empty() {
                    return ReachingDefinition::Def(d);
                }
                let mut l = self.forest.block_loop[here.0 as usize];
                while let Some(id) = l {
                    if interfering.iter().any(|&o| self.in_loop(id, o)) {
                        return ReachingDefinition::Carried(id);
                    }
                    l = self.forest.get(id).parent;
                }
                let mut merged = vec![d];
                merged.extend(interfering);
                return ReachingDefinition::Merged(merged);
            }
            if b == ControlFlowGraph::ENTRY {
                break;
            }
            cur = self.forest.doms.idom[b.0 as usize];
        }
        // No dominating definition: whatever reaches comes from paths
        // that meet before the statement.
        let arriving: Vec<usize> = sites
            .iter()
            .copied()
            .filter(|&o| {
                let ob = self.block_of(o);
                self.reach[ob.0 as usize][here.0 as usize]
            })
            .collect();
        if arriving.is_empty() {
            return ReachingDefinition::None;
        }
        let mut l = self.forest.block_loop[here.0 as usize];
        while let Some(id) = l {
            if arriving.iter().any(|&o| self.in_loop(id, o)) {
                return ReachingDefinition::Carried(id);
            }
            l = self.forest.get(id).parent;
        }
        ReachingDefinition::Merged(arriving)
    }

    pub(crate) fn trace_operand(
        &self,
        op: OperandId,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<Affine, String> {
        match self.module.operand(op) {
            Operand::Register(reg) => self.trace_reg(*reg, pos, id, depth, None),
            Operand::Special(reg) => self.trace_special(*reg),
            Operand::Immediate(text) => {
                let text = self.module.interner.resolve(*text);
                parse_int(text)
                    .map(|c| Affine::invariant(SymExpr::Const(c)))
                    .ok_or_else(|| format!("non-integer immediate {text}"))
            }
            // A named location (a shared array, the dynamic shared base):
            // its address is an invariant symbol.
            Operand::SymbolRef(s) => Ok(Affine::invariant(SymExpr::sym(
                self.module.interner.resolve(*s),
            ))),
            other => Err(format!(
                "unsupported operand form in latch trace: {other:?}"
            )),
        }
    }

    fn trace_special(&self, reg: SpecialReg) -> Result<Affine, String> {
        if let Some(v) = Var::special(reg) {
            return Ok(Affine::var(v));
        }
        match reg {
            _ if special_register(reg) == Some(SpecialRegister::Zero) => {
                Ok(Affine::invariant(SymExpr::Const(0)))
            }
            // PTX ISA §10.3: %laneid is "the thread's lane within the
            // warp", and §3.1: a warp "contains threads of consecutive,
            // increasing thread IDs with the first warp containing
            // thread 0"; in a one-dimensional block that is %tid.x mod 32.
            SpecialReg::Scalar(SpecialScalar::Laneid) if self.one_dimensional() => {
                Ok(Affine::var(Var::Mod(Box::new(Var::Tid(Axis::X)), 32)))
            }
            // One value for the CTA (the launch shape, the cluster, the
            // shared memory sizes): a symbol, bound when known.
            _ if special_register(reg) == Some(SpecialRegister::PerCta) => {
                Ok(Affine::invariant(SymExpr::sym(reg.name())))
            }
            _ => Err(format!(
                "latch condition depends on special register {}",
                reg.name()
            )),
        }
    }

    pub(crate) fn trace_reg(
        &self,
        reg: Symbol,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
        exclude: Option<LoopId>,
    ) -> Result<Affine, String> {
        if depth > 32 {
            return Err("value trace exceeds depth limit".to_owned());
        }
        let name = self.module.interner.resolve(reg);
        let def = match self.reach_def(reg, pos, exclude) {
            ReachingDefinition::Def(d) => d,
            ReachingDefinition::Carried(l) => {
                // The counter of loop l read before its increment: the
                // previous iteration's value, init + (k − 1)·step.
                if let Some((step, phi)) = self.ivs[l.0 as usize].get(&reg) {
                    let init = self.iv_init(*phi, l, depth + 1)?;
                    let back = SymExpr::mul(SymExpr::Const(-1), step.clone());
                    return Ok(Affine::term(Var::Iter(l), step.clone())
                        + init
                        + Affine::invariant(back));
                }
                return Err(self.obstacle_of_any_def(reg, id).unwrap_or_else(|| {
                    format!("latch condition depends on {name}, carried around an enclosing loop")
                }));
            }
            ReachingDefinition::Merged(defs) => {
                // The same immediate on every path is that value.
                if let Some(c) = self.same_immediate(&defs) {
                    return Ok(Affine::invariant(SymExpr::Const(c)));
                }
                return Err(self
                    .obstacle_of_any_def(reg, id)
                    .unwrap_or_else(|| format!("{name} has more than one reaching definition")));
            }
            ReachingDefinition::None => return Err(format!("no definition found for {name}")),
        };

        // The counter of this loop or an enclosing one, read at or after
        // its increment: init + k·step.
        let mut l = id;
        while let Some(cur) = l {
            if let Some((step, phi)) = self.ivs[cur.0 as usize].get(&reg)
                && self.in_loop(cur, def)
                && let Stmt::Instr(instr) = &self.kernel.stmts[def]
                && matches!(instr.mnemonic, Opcode::Add | Opcode::Mov)
            {
                let init = self.iv_init(*phi, cur, depth + 1)?;
                return Ok(Affine::term(Var::Iter(cur), step.clone()) + init);
            }
            l = self.forest.get(cur).parent;
        }

        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return Err("definition is not an instruction".to_owned());
        };
        let mnemonic = instr.mnemonic;
        let mods: Vec<&str> = self
            .module
            .modifiers(instr)
            .iter()
            .map(|&m| self.module.interner.resolve(m))
            .collect();
        let ops = self.module.operand_ids(instr.operands).to_vec();
        let arg =
            |i: usize| -> Result<Affine, String> { self.trace_operand(ops[i], def, id, depth + 1) };
        // A brace-enclosed destination's registers are the value's elements in
        // order (PTX ISA 6.4.3, "Vectors as Operands"): each is a parameter's
        // whole value only when the load has one element, and the element of
        // the same place in a special register's vector.
        if mnemonic == Opcode::Mov
            && let Some(Operand::VectorList { children }) =
                ops.first().map(|&o| self.module.operand(o))
            && let Some(&Operand::Special(SpecialReg::Vector(vector, None))) =
                ops.get(1).map(|&o| self.module.operand(o))
            && let Some(place) =
                self.module.operand_ids(*children).iter().position(
                    |&c| matches!(self.module.operand(c), Operand::Register(r) if *r == reg),
                )
            && let Some(&element) = [Element::X, Element::Y, Element::Z, Element::W].get(place)
        {
            return self.trace_special(SpecialReg::Vector(vector, Some(element)));
        }
        if let Some(Operand::VectorList { children }) = ops.first().map(|&o| self.module.operand(o))
            && (mnemonic != Opcode::Ld || mods.contains(&"param") && children.len() > 1)
        {
            let opcode = self.module.opcode(instr);
            return Err(format!("{name} is one of `{opcode}`'s vector destinations"));
        }

        match mnemonic {
            Opcode::Mov => arg(1),
            Opcode::Cvt | Opcode::Cvta => {
                // Width changes are value-preserving in the nonneg domain.
                self.trace_operand(*ops.last().expect("cvt has operands"), def, id, depth + 1)
            }
            Opcode::Ld => {
                if mods.contains(&"param")
                    && let Some(Operand::Memory { base, .. }) =
                        ops.get(1).map(|&i| self.module.operand(i))
                    && let Operand::SymbolRef(pname) = self.module.operand(*base)
                    && let Some(&idx) = self.params.get(pname)
                {
                    return Ok(Affine::invariant(SymExpr::sym(format!("param_{idx}"))));
                }
                if id.is_some_and(|l| self.in_loop(l, def)) {
                    Err("latch condition depends on a value loaded inside the loop".to_owned())
                } else {
                    Err("latch condition depends on a value loaded from memory".to_owned())
                }
            }
            Opcode::Add if ops.len() == 3 => Ok(arg(1)? + arg(2)?),
            Opcode::Sub if ops.len() == 3 => Ok(arg(1)? - arg(2)?),
            Opcode::And if ops.len() == 3 => {
                // and r, a, mask — PTX's lowering of `a mod 2^n`.
                let (val, mask) = (arg(1)?, arg(2)?);
                let (affine, konst) = match (self.constant(&mask), self.constant(&val)) {
                    (Some(c), _) => (val, c),
                    (_, Some(c)) => (mask, c),
                    _ => return Err("and-mask with two non-constant operands".to_owned()),
                };
                // A contiguous run of ones from bit `low` up to bit `top`:
                // x mod 2^top − x mod 2^low, and x itself above the register
                // width in the nonneg domain.
                let width = if mods.iter().any(|m| m.ends_with("64")) {
                    64
                } else {
                    32
                };
                let bits = konst as u64;
                let low = bits.trailing_zeros();
                let run = (bits >> low).trailing_ones();
                if bits == 0 || (bits >> low) >> run != 0 {
                    return Err(format!("and-mask {konst:#x} is not one run of bits"));
                }
                let top = low + run;
                let modulo = |x: Affine, k: u32| -> Result<Affine, String> {
                    x.clone()
                        .modulo_const(1i64 << k)
                        .ok_or_else(|| refuse(&x, id, "mod"))
                };
                let hi = if top >= width - 1 {
                    affine.clone()
                } else {
                    modulo(affine.clone(), top)?
                };
                if low == 0 {
                    return Ok(hi);
                }
                hi.clone()
                    .align_down(1i64 << low)
                    .ok_or_else(|| refuse(&hi, id, "mod"))
            }
            Opcode::Or if ops.len() == 3 => {
                // LLVM writes an add as `or` when no bits can overlap: every
                // part of one side is a multiple of a power of two above
                // the other side's largest value.
                let (a, b) = (arg(1)?, arg(2)?);
                // The bounded side: a constant, or a form of moduli whose
                // largest value is known.
                let bound = |x: &Affine| self.constant(x).or_else(|| x.max_value());
                let below = |small: &Affine, big: &Affine| -> Result<(), String> {
                    let max = bound(small)
                        .ok_or_else(|| "or of two values neither of known range".to_owned())?;
                    if max < 0 {
                        return Err(format!("or with a negative value {max:#x}"));
                    }
                    let bits = 64 - (max as u64).leading_zeros();
                    if bits > 62 {
                        return Err(format!("or with {max:#x}, no power of two above it"));
                    }
                    let m = 1i64 << bits;
                    big.clone().div_exact(m).map(|_| ()).ok_or_else(|| {
                        format!(
                            "or with a value below {m:#x} on a value not known to have those bits clear"
                        )
                    })
                };
                match below(&b, &a) {
                    Ok(()) => Ok(a + b),
                    Err(why) => match below(&a, &b) {
                        Ok(()) => Ok(a + b),
                        Err(other) => Err(if bound(&b).is_some() { why } else { other }),
                    },
                }
            }
            Opcode::Bfe if ops.len() == 4 => {
                // bfe d, a, pos, len: bits [pos, pos + len) of a. Signed from
                // bit 0 is a width change, value-preserving in the nonneg
                // domain like cvt; unsigned is (a >> pos) mod 2^len.
                let a = arg(1)?;
                let (Some(pos), Some(len)) = (self.constant(&arg(2)?), self.constant(&arg(3)?))
                else {
                    return Err("bfe with a non-constant position or length".to_owned());
                };
                let signed = mods.iter().any(|m| m.starts_with('s'));
                if signed && pos == 0 {
                    return Ok(a);
                }
                if signed || !(0..63).contains(&pos) || !(1..63).contains(&len) {
                    return Err(format!("bfe at {pos} of {len} bits"));
                }
                let shifted = a
                    .clone()
                    .div_const(1i64 << pos)
                    .ok_or_else(|| refuse(&a, id, "bit field"))?;
                shifted
                    .clone()
                    .modulo_const(1i64 << len)
                    .ok_or_else(|| refuse(&shifted, id, "bit field"))
            }
            Opcode::Shl if ops.len() == 3 => {
                let v = arg(1)?;
                match self.constant(&arg(2)?) {
                    Some(c) if (0..63).contains(&c) => Ok(v.scale(SymExpr::Const(1i64 << c))),
                    _ => Err("shift by a non-constant amount".to_owned()),
                }
            }
            Opcode::Shr if ops.len() == 3 => {
                let v = arg(1)?;
                let Some(c) = self.constant(&arg(2)?) else {
                    return Err("shift by a non-constant amount".to_owned());
                };
                if !v.is_invariant() {
                    if !(0..63).contains(&c) {
                        return Err(format!("shift by {c} is out of range"));
                    }
                    return v
                        .clone()
                        .div_const(1i64 << c)
                        .ok_or_else(|| refuse(&v, id, "shift"));
                }
                let signed_width = mods.iter().find_map(|m| match *m {
                    "s16" => Some(16),
                    "s32" => Some(32),
                    "s64" => Some(64),
                    _ => None,
                });
                // PTX ISA §9.7.8.9 (shr): "Signed shifts fill with the sign
                // bit" — in this tracer's nonnegative domain that bit is 0,
                // so shifting it down is nvcc's `x / 2^n` sign fix-up.
                if signed_width == Some(c + 1) {
                    return Ok(Affine::invariant(SymExpr::Const(0)));
                }
                if (0..63).contains(&c) {
                    Ok(Affine::invariant(SymExpr::floor_div(v.base, 1i64 << c)))
                } else {
                    Err(format!("shift by {c} is out of range"))
                }
            }
            Opcode::Div | Opcode::Rem if ops.len() == 3 => {
                let a = arg(1)?;
                let Some(d) = self.constant(&arg(2)?).filter(|&d| d > 0) else {
                    return Err(format!("{} by a non-constant divisor", mnemonic.name()));
                };
                let result = if mnemonic == Opcode::Div {
                    a.clone().div_const(d)
                } else {
                    a.clone().modulo_const(d)
                };
                result.ok_or_else(|| refuse(&a, id, "division"))
            }
            Opcode::Shfl if mods.contains(&"idx") && ops.len() == 5 => {
                // shfl.sync.idx d, a, c, 31, mask: a's value in lane c of the
                // warp. With warps of 32 consecutive %tid.x, that is a with
                // %tid.x replaced by 32·⌊%tid.x/32⌋ + c; a warp-uniform a
                // (a function of ⌊%tid.x/32⌋) is unchanged.
                let a = arg(1)?;
                let Some(c) = self.constant(&arg(2)?).filter(|c| (0..32).contains(c)) else {
                    return Err("shfl.idx from a non-constant lane".to_owned());
                };
                let warp = |v: Var| Affine::var(Var::Div(Box::new(v), 32));
                let lane = |v: &Var| -> Option<Affine> {
                    match v {
                        Var::Tid(Axis::X) => Some(
                            warp(Var::Tid(Axis::X)).scale(SymExpr::Const(32))
                                + Affine::invariant(SymExpr::Const(c)),
                        ),
                        Var::Div(inner, d) if **inner == Var::Tid(Axis::X) => {
                            if d % 32 == 0 {
                                Some(Affine::var(v.clone()))
                            } else if 32 % d == 0 {
                                Some(
                                    warp(Var::Tid(Axis::X)).scale(SymExpr::Const(32 / d))
                                        + Affine::invariant(SymExpr::Const(c / d)),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Mod(inner, m) if **inner == Var::Tid(Axis::X) => {
                            if 32 % m == 0 {
                                Some(Affine::invariant(SymExpr::Const(c % m)))
                            } else if m % 32 == 0 {
                                let w = Var::Mod(
                                    Box::new(Var::Div(Box::new(Var::Tid(Axis::X)), 32)),
                                    m / 32,
                                );
                                Some(
                                    Affine::var(w).scale(SymExpr::Const(32))
                                        + Affine::invariant(SymExpr::Const(c)),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Tid(_) => None,
                        other => Some(Affine::var(other.clone())),
                    }
                };
                a.map_vars(lane)
                    .ok_or_else(|| "shfl.idx of a value not a function of %tid.x".to_owned())
            }
            Opcode::Min | Opcode::Max if ops.len() == 3 => {
                // A clamp of two constants (after bindings) is a constant.
                let (a, b) = (arg(1)?, arg(2)?);
                match (self.constant(&a), self.constant(&b)) {
                    (Some(x), Some(y)) => Ok(Affine::invariant(SymExpr::Const(
                        if mnemonic == Opcode::Min {
                            x.min(y)
                        } else {
                            x.max(y)
                        },
                    ))),
                    _ if a.is_invariant() && b.is_invariant() => {
                        Err(format!("{} of values not both constant", mnemonic.name()))
                    }
                    _ => Err(refuse(&(a + b), id, mnemonic.name())),
                }
            }
            Opcode::Mul if ops.len() == 3 => {
                let (a, b) = (arg(1)?, arg(2)?);
                if a.is_invariant() {
                    Ok(b.scale(a.base))
                } else if b.is_invariant() {
                    Ok(a.scale(b.base))
                } else {
                    Err(refuse(&(a + b), id, "product"))
                }
            }
            Opcode::Mad if ops.len() == 4 => {
                let (a, b, c) = (arg(1)?, arg(2)?, arg(3)?);
                let prod = if a.is_invariant() {
                    b.scale(a.base)
                } else if b.is_invariant() {
                    a.scale(b.base)
                } else {
                    return Err(refuse(&(a + b), id, "product"));
                };
                Ok(prod + c)
            }
            other => {
                let mut seen = HashSet::new();
                let fundamental = ops
                    .iter()
                    .skip(1)
                    .find_map(|&op| self.operand_obstacle(op, def, id, &mut seen));
                Err(match fundamental {
                    Some(reason) => format!("{reason}, behind `{}`", self.module.opcode(instr)),
                    None => format!(
                        "value defined by unsupported instruction `{}`",
                        other.name()
                    ),
                })
            }
        }
    }

    /// Why a value cannot be traced, looking through arithmetic the
    /// tracer does not read: a special register, a memory load or an
    /// atomic on some operand chain is the reason a reader can act on,
    /// not the `or` or `bfe` in between.
    pub(crate) fn obstacle(
        &self,
        reg: Symbol,
        pos: usize,
        id: Option<LoopId>,
        seen: &mut HashSet<Symbol>,
    ) -> Option<String> {
        if !seen.insert(reg) || seen.len() > 64 {
            return None;
        }
        let def = match self.reach_def(reg, pos, None) {
            ReachingDefinition::Def(d) => d,
            ReachingDefinition::None => return None,
            ReachingDefinition::Carried(_) | ReachingDefinition::Merged(_) => {
                return self.obstacle_of_any_def(reg, id);
            }
        };
        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return None;
        };
        let is_param_load = self
            .module
            .modifiers(instr)
            .iter()
            .any(|&m| self.module.interner.resolve(m) == "param");
        match instr.mnemonic {
            Opcode::Ld if !is_param_load => Some(
                if id.is_some_and(|l| self.in_loop(l, def)) {
                    "latch condition depends on a value loaded inside the loop"
                } else {
                    "latch condition depends on a value loaded from memory"
                }
                .to_owned(),
            ),
            Opcode::Atom => Some("latch condition depends on an atomic operation".to_owned()),
            _ => self
                .module
                .operand_ids(instr.operands)
                .iter()
                .skip(1)
                .find_map(|&op| self.operand_obstacle(op, def, id, seen)),
        }
    }

    fn operand_obstacle(
        &self,
        op: OperandId,
        pos: usize,
        id: Option<LoopId>,
        seen: &mut HashSet<Symbol>,
    ) -> Option<String> {
        match self.module.operand(op) {
            Operand::Register(r) => self.obstacle(*r, pos, id, seen),
            Operand::Special(reg) => special_obstacle(*reg),
            _ => None,
        }
    }

    pub(crate) fn in_loop(&self, id: LoopId, stmt: usize) -> bool {
        self.forest.get(id).blocks.iter().any(|&b| {
            let blk = self.cfg.block(b);
            blk.start <= stmt && stmt < blk.end
        })
    }
}

fn special_obstacle(reg: SpecialReg) -> Option<String> {
    (special_register(reg) != Some(SpecialRegister::Zero))
        .then(|| format!("latch condition depends on special register {}", reg.name()))
}

/// Why an arithmetic form is refused: a special register or an enclosing
/// loop's counter among its variables outranks the generic reason.
pub(crate) fn refuse(a: &Affine, id: Option<LoopId>, what: &str) -> String {
    for v in a.terms.keys() {
        match v {
            Var::Iter(l) if Some(*l) == id => {}
            Var::Iter(_) => {
                return "latch condition depends on an enclosing loop's counter".to_owned();
            }
            other => {
                return format!("latch condition depends on special register {other}, in a {what}");
            }
        }
    }
    format!("{what} of an induction variable")
}

/// Which values of a special register the threads of one CTA can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecialRegister {
    /// Each thread its own.
    PerThread,
    /// One for the whole CTA, fixed for the kernel.
    PerCta,
    /// One that can differ from one read to the next.
    Varying,
    /// A vector register's fourth element: "unused and always returns
    /// zero" (PTX ISA §10.1, §10.2, §10.6, §10.7, §10.12–§10.15).
    Zero,
}

/// `None` where PTX ISA §10 does not say.
pub(crate) fn special_register(reg: SpecialReg) -> Option<SpecialRegister> {
    use SpecialRegister::*;
    use SpecialScalar::*;
    use SpecialVector::*;
    Some(match reg {
        SpecialReg::Vector(_, Some(Element::W)) => Zero,
        // §10.1 "Every thread in the CTA has a unique %tid"; §10.3 "the
        // thread's lane within the warp"; §10.18–§10.22 a mask of "the
        // thread's lane number in the warp".
        SpecialReg::Vector(Tid, _)
        | SpecialReg::Scalar(
            Laneid | LanemaskEq | LanemaskLe | LanemaskLt | LanemaskGe | LanemaskGt,
        ) => PerThread,
        // §10.2, §10.6, §10.7, §10.12–§10.17: the CTA's and its cluster's
        // place in, and the shape of, the grid and the cluster.
        SpecialReg::Vector(
            Ntid | Ctaid | Nctaid | Clusterid | Nclusterid | ClusterCtaid | ClusterNctaid,
            _,
        )
        | SpecialReg::Scalar(ClusterCtarank | ClusterNctarank) => PerCta,
        // §10.10 "the per-grid temporal grid identifier"; §10.11 whether
        // "the cluster launch is explicitly specified"; §10.30–§10.32 shared
        // memory allocated "at launch time"; §10.33 "the CUDA device graph
        // being currently executed"; §10.5, §10.9 "the maximum number of"
        // warp and SM identifiers.
        SpecialReg::Scalar(
            Gridid | IsExplicitCluster | TotalSmemSize | AggrSmemSize | DynamicSmemSize
            | CurrentGraphExec | Nwarpid | Nsmid,
        ) => PerCta,
        // §10.4, §10.8 "its value may change during execution"; §10.23,
        // §10.24 "cycle counters".
        SpecialReg::Scalar(Warpid | Smid | Clock | ClockHi | Clock64) => Varying,
        // §10.25, §10.26 "Their behavior is currently undefined"; §10.27
        // "Precise semantics of these registers is defined in the driver
        // documentation"; §10.28 "The behavior is target-specific"; §10.29
        // "Refer to CUDA Programming Guide".
        SpecialReg::Pm(_)
        | SpecialReg::Pm64(_)
        | SpecialReg::Envreg(_)
        | SpecialReg::ReservedSmemOffset(_)
        | SpecialReg::Scalar(
            Globaltimer
            | GlobaltimerLo
            | GlobaltimerHi
            | ReservedSmemOffsetBegin
            | ReservedSmemOffsetEnd
            | ReservedSmemOffsetCap,
        ) => return None,
    })
}

/// Mnemonics whose first operand is a register destination (for the
/// def table). Memory stores, branches, and sync ops define nothing.
fn defines_dest(mnemonic: Opcode) -> bool {
    !matches!(
        mnemonic,
        Opcode::St
            | Opcode::Bra
            | Opcode::Brx
            | Opcode::Ret
            | Opcode::Exit
            | Opcode::Bar
            | Opcode::Barrier
            | Opcode::Membar
            | Opcode::Fence
            | Opcode::Red
            | Opcode::Call
            | Opcode::Trap
            | Opcode::Brkpt
            | Opcode::Prefetch
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_registers_by_the_threads_that_share_them() {
        use SpecialRegister::*;
        let of = |name| special_register(SpecialReg::from_name(name).expect(name));
        for (name, class) in [
            ("%tid.x", Some(PerThread)),
            ("%laneid", Some(PerThread)),
            ("%lanemask_lt", Some(PerThread)),
            ("%ctaid.y", Some(PerCta)),
            ("%ntid", Some(PerCta)),
            ("%cluster_ctarank", Some(PerCta)),
            ("%dynamic_smem_size", Some(PerCta)),
            ("%nsmid", Some(PerCta)),
            ("%warpid", Some(Varying)),
            ("%smid", Some(Varying)),
            ("%clock64", Some(Varying)),
            ("%ctaid.w", Some(Zero)),
            ("%pm3", None),
            ("%envreg0", None),
            ("%globaltimer", None),
            ("%reserved_smem_offset_cap", None),
        ] {
            assert_eq!(of(name), class, "{name}");
        }
    }
}
