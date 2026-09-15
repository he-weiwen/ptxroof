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
//! refused. The tracer walks `mov/add/sub/and-mask/shl/shr/mul/mad/cvt`
//! down to `ld.param`, constants and special registers; anything else
//! is a named reason, preferring the fundamental obstacle (a special
//! register, a memory load, an atomic) behind arithmetic it does not
//! read. The domain is nonnegative and non-overflowing, as documented
//! in `trips`.

use crate::affine::{Affine, Var};
use crate::cfg::loops::{LoopForest, LoopId};
use crate::cfg::{BlockId, Cfg};
use crate::core::symexpr::SymExpr;
use crate::core::{Instr, Kernel, Module, Operand, Stmt, Symbol};
use crate::parse::parser::parse_int;
use std::collections::{HashMap, HashSet};

/// Where a register's value at a statement comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reach {
    /// One definition, and no other can reach the statement.
    Def(usize),
    /// Defined again inside a loop containing the statement: the value
    /// is the header's merge of the entry value and the back edge.
    Carried(LoopId),
    /// Definitions from more than one path meet before the statement.
    Merged,
    None,
}

pub(crate) struct Tracer<'a> {
    pub(crate) module: &'a Module,
    pub(crate) kernel: &'a Kernel,
    pub(crate) cfg: &'a Cfg,
    pub(crate) forest: &'a LoopForest,
    /// Definition sites (statement indices) per register symbol.
    pub(crate) defs: HashMap<Symbol, Vec<usize>>,
    /// param symbol -> positional index.
    pub(crate) params: HashMap<Symbol, usize>,
    /// `reach[a][b]`: a path of at least one edge from block a to b.
    pub(crate) reach: Vec<Vec<bool>>,
    /// Per loop: counter register ↦ (step, the chain register defined
    /// before the loop).
    pub(crate) ivs: Vec<HashMap<Symbol, (i64, Symbol)>>,
}

impl<'a> Tracer<'a> {
    pub(crate) fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        cfg: &'a Cfg,
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
        let mut t = Tracer {
            module,
            kernel,
            cfg,
            forest,
            defs,
            params,
            reach,
            ivs: Vec::new(),
        };
        t.ivs = (0..forest.loops.len() as u32)
            .map(|i| t.induction_vars(LoopId(i)))
            .collect();
        t
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
    pub(crate) fn induction_vars(&self, id: LoopId) -> HashMap<Symbol, (i64, Symbol)> {
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
        let add_const = |site: usize| -> Option<(Symbol, i64)> {
            let Stmt::Instr(instr) = &self.kernel.stmts[site] else {
                return None;
            };
            let ops = self.module.operand_ids(instr.operands);
            match (self.module.interner.resolve(instr.mnemonic), ops) {
                ("mov", [_, src]) => match self.module.operand(*src) {
                    Operand::Register(r) => Some((*r, 0)),
                    _ => None,
                },
                ("add", [_, a, b]) => match (self.module.operand(*a), self.module.operand(*b)) {
                    (Operand::Register(r), Operand::Immediate(c))
                    | (Operand::Immediate(c), Operand::Register(r)) => {
                        Some((*r, parse_int(self.module.interner.resolve(*c))?))
                    }
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
            let mut step = 0;
            let mut chain = vec![*reg];
            for _ in 0..8 {
                let Some((src, c)) = add_const(site) else {
                    break;
                };
                step += c;
                if src == *reg {
                    let phi = chain.iter().copied().find(|&r| {
                        matches!(
                            self.reach_def(r, header_start, Some(id)),
                            Reach::Def(_) | Reach::Merged
                        )
                    });
                    if let Some(phi) = phi
                        && step != 0
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
    pub(crate) fn reach_def(&self, reg: Symbol, pos: usize, exclude: Option<LoopId>) -> Reach {
        let Some(all) = self.defs.get(&reg) else {
            return Reach::None;
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
            return Reach::Def(d);
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
                    return Reach::Def(d);
                }
                let mut l = self.forest.block_loop[here.0 as usize];
                while let Some(id) = l {
                    if interfering.iter().any(|&o| self.in_loop(id, o)) {
                        return Reach::Carried(id);
                    }
                    l = self.forest.get(id).parent;
                }
                return Reach::Merged;
            }
            if b == Cfg::ENTRY {
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
            return Reach::None;
        }
        let mut l = self.forest.block_loop[here.0 as usize];
        while let Some(id) = l {
            if arriving.iter().any(|&o| self.in_loop(id, o)) {
                return Reach::Carried(id);
            }
            l = self.forest.get(id).parent;
        }
        Reach::Merged
    }

    pub(crate) fn trace_operand(
        &self,
        op: crate::core::OperandId,
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
            Reach::Def(d) => d,
            Reach::Carried(l) => {
                // The counter of loop l read before its increment: the
                // previous iteration's value, init + (k − 1)·step.
                if let Some(&(step, phi)) = self.ivs[l.0 as usize].get(&reg) {
                    let init = self.iv_init(phi, l, depth + 1)?;
                    return Ok(Affine::term(Var::Iter(l), SymExpr::Const(step))
                        + init
                        + Affine::invariant(SymExpr::Const(-step)));
                }
                return Err(self.obstacle_of_any_def(reg, id).unwrap_or_else(|| {
                    format!("latch condition depends on {name}, carried around an enclosing loop")
                }));
            }
            Reach::Merged => {
                return Err(self
                    .obstacle_of_any_def(reg, id)
                    .unwrap_or_else(|| format!("{name} has more than one reaching definition")));
            }
            Reach::None => {
                return match Var::special(name) {
                    Some(v) => Ok(Affine::var(v)),
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
            if let Some(&(step, phi)) = self.ivs[cur.0 as usize].get(&reg)
                && self.in_loop(cur, def)
                && let Stmt::Instr(instr) = &self.kernel.stmts[def]
                && matches!(self.module.interner.resolve(instr.mnemonic), "add" | "mov")
            {
                let init = self.iv_init(phi, cur, depth + 1)?;
                return Ok(Affine::term(Var::Iter(cur), SymExpr::Const(step)) + init);
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
                let (affine, konst) = match (mask.as_const(), val.as_const()) {
                    (Some(c), _) => (val, c),
                    (_, Some(c)) => (mask, c),
                    _ => return Err("and-mask with two non-constant operands".to_owned()),
                };
                let modulus = konst
                    .checked_add(1)
                    .filter(|m| *m > 0 && (m & (m - 1)) == 0)
                    .ok_or_else(|| format!("and-mask {konst:#x} is not 2^n − 1"))?;
                if !affine.is_invariant() {
                    return Err(refuse(&affine, id, "mod applied to an induction variable"));
                }
                Ok(Affine::invariant(SymExpr::modulo(affine.base, modulus)))
            }
            "shl" if ops.len() == 3 => {
                let v = arg(1)?;
                match arg(2)?.as_const() {
                    Some(c) if (0..63).contains(&c) => Ok(v.scale(SymExpr::Const(1i64 << c))),
                    _ => Err("shift by a non-constant amount".to_owned()),
                }
            }
            "shr" if ops.len() == 3 => {
                let v = arg(1)?;
                let Some(c) = arg(2)?.as_const() else {
                    return Err("shift by a non-constant amount".to_owned());
                };
                if !v.is_invariant() {
                    return Err(refuse(&v, id, "shift applied to an induction variable"));
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
            "mul" if ops.len() == 3 => {
                let (a, b) = (arg(1)?, arg(2)?);
                if a.is_invariant() {
                    Ok(b.scale(a.base))
                } else if b.is_invariant() {
                    Ok(a.scale(b.base))
                } else {
                    Err(refuse(
                        &(a + b),
                        id,
                        "product involving an induction variable",
                    ))
                }
            }
            "mad" if ops.len() == 4 => {
                let (a, b, c) = (arg(1)?, arg(2)?, arg(3)?);
                let prod = if a.is_invariant() {
                    b.scale(a.base)
                } else if b.is_invariant() {
                    a.scale(b.base)
                } else {
                    return Err(refuse(
                        &(a + b),
                        id,
                        "product involving an induction variable",
                    ));
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
                Err(fundamental.unwrap_or_else(|| {
                    format!("value defined by unsupported instruction `{other}`")
                }))
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
            Reach::Def(d) => d,
            Reach::None => {
                let name = self.module.interner.resolve(reg);
                return is_special_register(name)
                    .then(|| format!("latch condition depends on special register {name}"));
            }
            Reach::Carried(_) | Reach::Merged => return self.obstacle_of_any_def(reg, id),
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
pub(crate) fn refuse(a: &Affine, id: Option<LoopId>, generic: &str) -> String {
    for v in a.terms.keys() {
        match v {
            Var::Iter(l) if Some(*l) == id => {}
            Var::Iter(_) => {
                return "latch condition depends on an enclosing loop's counter".to_owned();
            }
            other => return format!("latch condition depends on special register {other}"),
        }
    }
    generic.to_owned()
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
