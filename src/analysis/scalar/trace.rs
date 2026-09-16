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
use crate::ptx::ir::{Instr, Kernel, Module, Operand, Stmt};
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
                && let Operand::Register(reg) = module.operand(first)
                && defines_dest(module.interner.resolve(instr.mnemonic))
            {
                defs.entry(*reg).or_default().push(i);
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
            let [_, src] = self.module.operand_ids(instr.operands) else {
                return None;
            };
            let Operand::Immediate(text) = self.module.operand(*src) else {
                return None;
            };
            if self.module.interner.resolve(instr.mnemonic) != "mov" {
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
                .find_map(|&op| match self.module.operand(op) {
                    Operand::Register(r) => self.obstacle(*r, d, id, &mut seen),
                    _ => None,
                })
        })
    }

    /// In-loop registers whose ONLY in-loop definition adds a constant
    /// to themselves → reg ↦ (step, phi). The addition may pass through
    /// other single-definition registers (`t = i + 2; i = t - 1`,
    /// nvcc's shape when the body also reads `i + 2`; `t = i + 1` then
    /// `mov i, t` in the latch, LLVM's two-register counter); the step
    /// is the sum of the constants along that chain, and phi is the
    /// chain's register defined before the loop.
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
                    && defines_dest(self.module.interner.resolve(instr.mnemonic))
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
            match (self.module.interner.resolve(instr.mnemonic), ops) {
                ("mov", [_, src]) => match self.module.operand(*src) {
                    Operand::Register(r) => Some((*r, SymExpr::Const(0))),
                    _ => None,
                },
                ("add", [_, a, b]) => match (self.module.operand(*a), self.module.operand(*b)) {
                    (Operand::Register(r), Operand::Immediate(c))
                    | (Operand::Immediate(c), Operand::Register(r)) => Some((
                        *r,
                        SymExpr::Const(parse_int(self.module.interner.resolve(*c))?),
                    )),
                    (Operand::Register(r), Operand::Register(s)) => invariant_step(*s, site)
                        .map(|step| (*r, step))
                        .or_else(|| invariant_step(*r, site).map(|step| (*s, step))),
                    _ => None,
                },
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
                    if self.module.interner.resolve(instr.mnemonic) == "setp"
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
        op: crate::ptx::ir::OperandId,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<Affine, String> {
        match self.module.operand(op) {
            Operand::Register(reg) => self.trace_reg(*reg, pos, id, depth, None),
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
            ReachingDefinition::None => {
                return match Var::special(name) {
                    Some(v) => Ok(Affine::var(v)),
                    // The launch shape is uniform: a symbol, bound when known.
                    None if name.starts_with("%ntid") || name.starts_with("%nctaid") => {
                        Ok(Affine::invariant(SymExpr::sym(name)))
                    }
                    None if is_special_register(name) => Err(format!(
                        "latch condition depends on special register {name}"
                    )),
                    None => Err(format!("no definition found for {name}")),
                };
            }
        };

        // The counter of this loop or an enclosing one, read at or after
        // its increment: init + k·step.
        let mut l = id;
        while let Some(cur) = l {
            if let Some((step, phi)) = self.ivs[cur.0 as usize].get(&reg)
                && self.in_loop(cur, def)
                && let Stmt::Instr(instr) = &self.kernel.stmts[def]
                && matches!(self.module.interner.resolve(instr.mnemonic), "add" | "mov")
            {
                let init = self.iv_init(*phi, cur, depth + 1)?;
                return Ok(Affine::term(Var::Iter(cur), step.clone()) + init);
            }
            l = self.forest.get(cur).parent;
        }

        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return Err("definition is not an instruction".to_owned());
        };
        let mnemonic = self.module.interner.resolve(instr.mnemonic).to_owned();
        let mods: Vec<&str> = self
            .module
            .modifiers(instr)
            .iter()
            .map(|&m| self.module.interner.resolve(m))
            .collect();
        let ops = self.module.operand_ids(instr.operands).to_vec();
        let arg =
            |i: usize| -> Result<Affine, String> { self.trace_operand(ops[i], def, id, depth + 1) };

        match mnemonic.as_str() {
            "mov" => arg(1),
            "cvt" | "cvta" => {
                // Width changes are value-preserving in the nonneg domain.
                self.trace_operand(*ops.last().expect("cvt has operands"), def, id, depth + 1)
            }
            "ld" => {
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
            "add" if ops.len() == 3 => Ok(arg(1)? + arg(2)?),
            "sub" if ops.len() == 3 => Ok(arg(1)? - arg(2)?),
            "and" if ops.len() == 3 => {
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
            "or" if ops.len() == 3 => {
                // LLVM writes an add as `or` when no bits can overlap: every
                // part of the other side is a multiple of a power of two above
                // the constant.
                let (a, b) = (arg(1)?, arg(2)?);
                // The bounded side: a constant, or a form of moduli whose
                // largest value is known.
                let bound = |x: &Affine| self.constant(x).or_else(|| x.max_value());
                let (small, max, big) = match (bound(&a), bound(&b)) {
                    (_, Some(m)) => (b, m, a),
                    (Some(m), _) => (a, m, b),
                    _ => return Err("or of two values neither of known range".to_owned()),
                };
                if max < 0 {
                    return Err(format!("or with a negative value {max:#x}"));
                }
                let m = 1i64 << (64 - (max as u64).leading_zeros());
                if big.clone().div_exact(m).is_some() {
                    Ok(big + small)
                } else {
                    Err(format!(
                        "or with a value below {m:#x} on a value not known to have those bits clear"
                    ))
                }
            }
            "bfe" if ops.len() == 4 => {
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
            "shl" if ops.len() == 3 => {
                let v = arg(1)?;
                match self.constant(&arg(2)?) {
                    Some(c) if (0..63).contains(&c) => Ok(v.scale(SymExpr::Const(1i64 << c))),
                    _ => Err("shift by a non-constant amount".to_owned()),
                }
            }
            "shr" if ops.len() == 3 => {
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
            "div" | "rem" if ops.len() == 3 => {
                let a = arg(1)?;
                let Some(d) = self.constant(&arg(2)?).filter(|&d| d > 0) else {
                    return Err(format!("{mnemonic} by a non-constant divisor"));
                };
                let result = if mnemonic == "div" {
                    a.clone().div_const(d)
                } else {
                    a.clone().modulo_const(d)
                };
                result.ok_or_else(|| refuse(&a, id, "division"))
            }
            "shfl" if mods.contains(&"idx") && ops.len() == 5 => {
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
            "min" | "max" if ops.len() == 3 => {
                // A clamp of two constants (after bindings) is a constant.
                let (a, b) = (arg(1)?, arg(2)?);
                match (self.constant(&a), self.constant(&b)) {
                    (Some(x), Some(y)) => {
                        Ok(Affine::invariant(SymExpr::Const(if mnemonic == "min" {
                            x.min(y)
                        } else {
                            x.max(y)
                        })))
                    }
                    _ if a.is_invariant() && b.is_invariant() => {
                        Err(format!("{mnemonic} of values not both constant"))
                    }
                    _ => Err(refuse(&(a + b), id, &mnemonic)),
                }
            }
            "mul" if ops.len() == 3 => {
                let (a, b) = (arg(1)?, arg(2)?);
                if a.is_invariant() {
                    Ok(b.scale(a.base))
                } else if b.is_invariant() {
                    Ok(a.scale(b.base))
                } else {
                    Err(refuse(&(a + b), id, "product"))
                }
            }
            "mad" if ops.len() == 4 => {
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
                let fundamental =
                    ops.iter()
                        .skip(1)
                        .find_map(|&op| match self.module.operand(op) {
                            Operand::Register(r) => self.obstacle(*r, def, id, &mut seen),
                            _ => None,
                        });
                Err(match fundamental {
                    Some(reason) => format!("{reason}, behind `{}`", self.module.opcode(instr)),
                    None => format!("value defined by unsupported instruction `{other}`"),
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
            ReachingDefinition::None => {
                let name = self.module.interner.resolve(reg);
                return is_special_register(name)
                    .then(|| format!("latch condition depends on special register {name}"));
            }
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
        match self.module.interner.resolve(instr.mnemonic) {
            "ld" if !is_param_load => Some(
                if id.is_some_and(|l| self.in_loop(l, def)) {
                    "latch condition depends on a value loaded inside the loop"
                } else {
                    "latch condition depends on a value loaded from memory"
                }
                .to_owned(),
            ),
            "atom" => Some("latch condition depends on an atomic operation".to_owned()),
            _ => self
                .module
                .operand_ids(instr.operands)
                .iter()
                .skip(1)
                .find_map(|&op| match self.module.operand(op) {
                    Operand::Register(r) => self.obstacle(*r, def, id, seen),
                    _ => None,
                }),
        }
    }

    pub(crate) fn in_loop(&self, id: LoopId, stmt: usize) -> bool {
        self.forest.get(id).blocks.iter().any(|&b| {
            let blk = self.cfg.block(b);
            blk.start <= stmt && stmt < blk.end
        })
    }
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

pub(crate) fn is_special_register(name: &str) -> bool {
    ["%tid", "%ctaid", "%ntid", "%nctaid", "%laneid", "%warpid"]
        .iter()
        .any(|p| name.starts_with(p))
}

/// Mnemonics whose first operand is a register destination (for the
/// def table). Memory stores, branches, and sync ops define nothing.
fn defines_dest(mnemonic: &str) -> bool {
    !matches!(
        mnemonic,
        "st" | "bra"
            | "brx"
            | "ret"
            | "exit"
            | "bar"
            | "barrier"
            | "membar"
            | "fence"
            | "red"
            | "call"
            | "trap"
            | "brkpt"
            | "nop"
            | "prefetch"
    )
}
