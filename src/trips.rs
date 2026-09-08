//! Trip-count matcher for the nvcc loop shapes.
//!
//! Per loop: extract the latch condition (`setp` + `@%p bra`),
//! recognize the induction variable (the loop's single in-loop
//! `add r, r, const` definition), and normalize the latch condition as
//! an affine expression of the IV and loop invariants with a scalar
//! affine tracer — real nvcc latches compare *derived* registers, not
//! the IV (k2 exits on `setp.ne.s32 %p6, %r29, 0` where
//! `%r29 = %r7 + %r35`, `%r7 = (K&3) − K`). The tracer walks reaching
//! definitions through `mov/add/sub/and-mask/shl/mul/mad/cvt` down to
//! `ld.param`/constants, following the dominator chain for
//! loop-invariant values.
//!
//! With the latch normalized to `continue while A1·k + A0  cmp  0`
//! (k = iteration number, IV read after its increment — every corpus
//! latch increments before it compares, and the tracer checks), trips
//! solve to:  `ne` → −A0/A1 (exact division — compiler-generated
//! not-equal latches step in divisor units);  `lt` → ceildiv(−A0, A1);
//! `le` → floordiv(−A0, A1) + 1;  `gt`/`ge` → mirrored.
//!
//! Anything else — multiple exits, exits not at the latch, latch
//! values loaded in the loop, special-register dependence, two IVs in
//! one condition — degrades to an `Err(reason)`: a *named* unknown,
//! never a guess (honesty principle). The symbols are assumed
//! nonnegative, and unsigned/signed comparison width is deliberately
//! not modeled (documented domain assumption).
//!
//! The kernel-level pass links nvcc's unroll main+remainder pair —
//! sibling loops on the same source line whose trips match
//! `(X − X mod c)/c` and `X mod c` — into one logical loop with
//! factor c.

use crate::cfg::loops::{LoopForest, LoopId};
use crate::cfg::naming::LoopName;
use crate::cfg::{BlockId, Cfg};
use crate::core::symexpr::SymExpr;
use crate::core::{Instr, Kernel, Module, Operand, Stmt, Symbol};
use crate::parse::parser::parse_int;
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
pub struct TripInfo {
    /// Indexed by `LoopId`.
    pub trips: Vec<TripCount>,
    pub unroll_pairs: Vec<UnrollPair>,
}

pub fn trip_counts(
    module: &Module,
    kernel: &Kernel,
    cfg: &Cfg,
    forest: &LoopForest,
    names: &[LoopName],
) -> TripInfo {
    let tracer = Tracer::new(module, kernel, cfg, forest);
    let trips: Vec<TripCount> = (0..forest.loops.len() as u32)
        .map(|i| tracer.loop_trips(LoopId(i)))
        .collect();

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

    TripInfo {
        trips,
        unroll_pairs,
    }
}

/// `main = (X − X mod c)/c` and `rem = X mod c` → factor c.
fn unroll_factor(main: &SymExpr, rem: &SymExpr) -> Option<i64> {
    let SymExpr::Mod(rem_x, c) = rem else {
        return None;
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

// -------------------------------------------------------------------------
// The scalar affine tracer.

/// An affine value: `coeff·IV + base`, with at most one IV involved.
#[derive(Debug, Clone)]
struct Affine {
    /// (IV register, coefficient); `None` = loop-invariant.
    iv: Option<(Symbol, i64)>,
    base: SymExpr,
}

impl Affine {
    fn invariant(base: SymExpr) -> Affine {
        Affine { iv: None, base }
    }

    fn combine(self, other: Affine, sign: i64) -> Result<Affine, String> {
        let iv = match (self.iv, other.iv) {
            (a, None) => a,
            (None, Some((r, c))) => Some((r, sign * c)),
            (Some((ra, ca)), Some((rb, cb))) if ra == rb => {
                let c = ca + sign * cb;
                (c != 0).then_some((ra, c))
            }
            _ => {
                return Err("latch condition mixes two induction variables".to_owned());
            }
        };
        let base = if sign >= 0 {
            SymExpr::add(self.base, other.base)
        } else {
            SymExpr::sub(self.base, other.base)
        };
        Ok(Affine { iv, base })
    }

    fn scale(self, c: i64) -> Affine {
        Affine {
            iv: self.iv.map(|(r, k)| (r, k * c)),
            base: SymExpr::mul(SymExpr::Const(c), self.base),
        }
    }
}

struct Tracer<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: &'a Cfg,
    forest: &'a LoopForest,
    /// Definition sites (statement indices) per register symbol.
    defs: HashMap<Symbol, Vec<usize>>,
    /// param symbol -> positional index.
    params: HashMap<Symbol, usize>,
}

impl<'a> Tracer<'a> {
    fn new(module: &'a Module, kernel: &'a Kernel, cfg: &'a Cfg, forest: &'a LoopForest) -> Self {
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
        Tracer {
            module,
            kernel,
            cfg,
            forest,
            defs,
            params,
        }
    }

    fn loop_trips(&self, id: LoopId) -> TripCount {
        let l = self.forest.get(id);

        // Structural requirements: one latch, one exit edge, at the latch.
        if l.latches.len() != 1 {
            return Err("loop has multiple latches".to_owned());
        }
        let latch = l.latches[0];
        let in_loop = |b: BlockId| l.blocks.binary_search(&b).is_ok();
        let mut exit_edges = Vec::new();
        for &b in &l.blocks {
            for &s in &self.cfg.block(b).succs {
                if !in_loop(s) {
                    exit_edges.push((b, s));
                }
            }
        }
        if exit_edges.len() != 1 {
            return Err(format!("loop has {} exit edges", exit_edges.len()));
        }
        if exit_edges[0].0 != latch {
            return Err("loop exit is not at the latch".to_owned());
        }

        // The latch terminator: `@%p bra HEADER` (or inverted).
        let latch_block = self.cfg.block(latch);
        let branch = self
            .last_instr(latch)
            .ok_or_else(|| "latch block has no instructions".to_owned())?;
        let pred = branch
            .1
            .predicate
            .ok_or_else(|| "latch branch is unconditional".to_owned())?;
        // succs[0] is the taken target (graph contract). Taken = header
        // means continue-if-true; predicate negation flips once more.
        let mut continue_if_true = latch_block.succs.first() == Some(&l.header);
        if pred.negated {
            continue_if_true = !continue_if_true;
        }

        // The setp defining the branch predicate, inside the latch block.
        let Some((setp_idx, setp)) = self.find_setp(latch, branch.0, pred.reg) else {
            return self
                .predicate_phi_trips(id, latch, branch.0, pred.reg, continue_if_true)
                .unwrap_or_else(|| {
                    Err("latch predicate is not defined in the latch block".to_owned())
                });
        };
        let cmp = self
            .module
            .modifiers(setp)
            .first()
            .map(|&m| self.module.interner.resolve(m).to_owned())
            .ok_or_else(|| "setp without comparison modifier".to_owned())?;

        // Induction variables of this loop.
        let ivs = self.induction_vars(id);

        // Trace both operands at the setp.
        let ops = self.module.operand_ids(setp.operands);
        if ops.len() < 3 {
            return Err("setp with unexpected operand count".to_owned());
        }
        let a = self.trace_operand(ops[1], setp_idx, id, &ivs, 0)?;
        let b = self.trace_operand(ops[2], setp_idx, id, &ivs, 0)?;
        let d = a.combine(b, -1)?; // D = A − B; condition is `D cmp 0`

        let Some((iv_reg, coeff)) = d.iv else {
            return Err("latch condition does not involve an induction variable".to_owned());
        };
        let (step, phi) = ivs[&iv_reg];

        // The compare must read the post-increment value: the in-loop
        // definition precedes it in the latch block, or sits in a block
        // that dominates the latch.
        let increment_precedes_compare = self.defs[&iv_reg].iter().any(|&d| {
            self.in_loop(id, d)
                && if d >= latch_block.start && d < latch_block.end {
                    d < setp_idx
                } else {
                    self.forest.doms.dominates(self.block_of(d), latch)
                }
        });
        if !increment_precedes_compare {
            return Err("induction increment does not precede the latch compare".to_owned());
        }

        // IV value at the latch on iteration k (k = 1, 2, ...):
        // init + k·step, so D(k) = coeff·step·k + (coeff·init + base).
        let init = self.iv_init(phi, l.header, id)?;
        let a1 = coeff
            .checked_mul(step)
            .ok_or_else(|| "induction step overflows".to_owned())?;
        let a0 = SymExpr::add(SymExpr::mul(SymExpr::Const(coeff), init), d.base);

        solve(&cmp, continue_if_true, a1, a0)
    }

    /// LLVM's two-trip loop: the latch predicate is a copy of a register
    /// set true before the loop and false in the latch (`mov.pred %q,
    /// -1; header: mov.pred %p, %q; latch: mov.pred %q, 0; @%p bra
    /// header`). Trips: 2 if the initial value continues, else 1.
    fn predicate_phi_trips(
        &self,
        id: LoopId,
        latch: BlockId,
        branch_idx: usize,
        pred: Symbol,
        continue_if_true: bool,
    ) -> Option<TripCount> {
        let mov_src = |site: usize| -> Option<&Operand> {
            let Stmt::Instr(instr) = &self.kernel.stmts[site] else {
                return None;
            };
            if self.module.interner.resolve(instr.mnemonic) != "mov" {
                return None;
            }
            let [_, src] = self.module.operand_ids(instr.operands) else {
                return None;
            };
            Some(self.module.operand(*src))
        };
        let copy = self.reaching_def(pred, branch_idx)?;
        let Operand::Register(phi) = mov_src(copy)? else {
            return None;
        };
        if !self.in_loop(id, copy) {
            return None;
        }
        let latch_blk = self.cfg.block(latch);
        let mut in_loop = self.defs.get(phi)?.iter().filter(|&&d| self.in_loop(id, d));
        let (&latch_def, None) = (in_loop.next()?, in_loop.next()) else {
            return None;
        };
        if latch_def < latch_blk.start || latch_def >= latch_blk.end {
            return None;
        }
        let header = self.forest.get(id).header;
        let init_def = self.reaching_def(*phi, self.cfg.block(header).start)?;
        let (Operand::Immediate(latch_val), Operand::Immediate(init_val)) =
            (mov_src(latch_def)?, mov_src(init_def)?)
        else {
            return None;
        };
        let continues = |text: &Symbol| {
            parse_int(self.module.interner.resolve(*text)).map(|v| (v != 0) == continue_if_true)
        };
        if continues(latch_val)? {
            return None;
        }
        Some(Ok(SymExpr::Const(if continues(init_val)? { 2 } else { 1 })))
    }

    /// In-loop registers whose ONLY in-loop definition adds a constant
    /// to themselves → reg ↦ (step, phi). The addition may pass through
    /// other single-definition registers (`t = i + 2; i = t - 1`,
    /// nvcc's shape when the body also reads `i + 2`; `t = i + 1` then
    /// `mov i, t` in the latch, LLVM's two-register counter); the step
    /// is the sum of the constants along that chain, and phi is the
    /// chain's register defined before the loop.
    fn induction_vars(&self, id: LoopId) -> HashMap<Symbol, (i64, Symbol)> {
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
            let mut step = 0;
            let mut chain = vec![*reg];
            for _ in 0..8 {
                let Some((src, c)) = add_const(site) else {
                    break;
                };
                step += c;
                if src == *reg {
                    let phi = chain.iter().copied().find(|&r| {
                        self.reaching_def(r, header_start)
                            .is_some_and(|d| !self.in_loop(id, d))
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

    /// Initial IV value: its reaching definition at loop entry.
    fn iv_init(&self, reg: Symbol, header: BlockId, id: LoopId) -> Result<SymExpr, String> {
        let hdr = self.cfg.block(header);
        let init = self.trace_reg(reg, hdr.start, id, &HashMap::new(), 0)?;
        match init.iv {
            None => Ok(init.base),
            Some(_) => Err("induction initial value depends on an induction variable".to_owned()),
        }
    }

    fn last_instr(&self, b: BlockId) -> Option<(usize, &Instr)> {
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

    fn find_setp(&self, b: BlockId, before: usize, pred: Symbol) -> Option<(usize, &Instr)> {
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
    fn block_of(&self, stmt: usize) -> BlockId {
        (0..self.cfg.blocks.len() as u32)
            .map(BlockId)
            .find(|&b| {
                let blk = self.cfg.block(b);
                blk.start <= stmt && stmt < blk.end
            })
            .expect("statement belongs to a block")
    }

    fn reaching_def(&self, reg: Symbol, pos: usize) -> Option<usize> {
        let sites = self.defs.get(&reg)?;
        let here = self.block_of(pos);
        let blk = self.cfg.block(here);
        if let Some(&d) = sites
            .iter()
            .rev()
            .find(|&&d| d >= blk.start && d < pos.min(blk.end))
        {
            return Some(d);
        }
        let mut cur = self.forest.doms.idom[here.0 as usize];
        while let Some(b) = cur {
            let blk = self.cfg.block(b);
            if let Some(&d) = sites.iter().rev().find(|&&d| d >= blk.start && d < blk.end) {
                return Some(d);
            }
            if b == Cfg::ENTRY {
                break;
            }
            cur = self.forest.doms.idom[b.0 as usize];
        }
        None
    }

    fn trace_operand(
        &self,
        op: crate::core::OperandId,
        pos: usize,
        id: LoopId,
        ivs: &HashMap<Symbol, (i64, Symbol)>,
        depth: u32,
    ) -> Result<Affine, String> {
        match self.module.operand(op) {
            Operand::Register(reg) => self.trace_reg(*reg, pos, id, ivs, depth),
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

    fn trace_reg(
        &self,
        reg: Symbol,
        pos: usize,
        id: LoopId,
        ivs: &HashMap<Symbol, (i64, Symbol)>,
        depth: u32,
    ) -> Result<Affine, String> {
        if depth > 32 {
            return Err("value trace exceeds depth limit".to_owned());
        }
        let name = self.module.interner.resolve(reg);
        let Some(def) = self.reaching_def(reg, pos) else {
            return Err(
                if name.starts_with("%tid")
                    || name.starts_with("%ctaid")
                    || name.starts_with("%ntid")
                    || name.starts_with("%nctaid")
                    || name.starts_with("%laneid")
                    || name.starts_with("%warpid")
                {
                    format!("latch condition depends on special register {name}")
                } else {
                    format!("no definition found for {name}")
                },
            );
        };

        // The IV itself: its single in-loop def is the self-add (or the
        // copy closing a two-register chain); reading it at/after that
        // def yields the post-increment latch value.
        if ivs.contains_key(&reg)
            && let Stmt::Instr(instr) = &self.kernel.stmts[def]
            && self.in_loop(id, def)
            && matches!(self.module.interner.resolve(instr.mnemonic), "add" | "mov")
        {
            return Ok(Affine {
                iv: Some((reg, 1)),
                base: SymExpr::Const(0),
            });
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
        let arg = |i: usize| -> Result<Affine, String> {
            self.trace_operand(ops[i], def, id, ivs, depth + 1)
        };

        match mnemonic.as_str() {
            "mov" => arg(1),
            "cvt" | "cvta" => {
                // Width changes are value-preserving in the nonneg domain.
                self.trace_operand(
                    *ops.last().expect("cvt has operands"),
                    def,
                    id,
                    ivs,
                    depth + 1,
                )
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
                if self.in_loop(id, def) {
                    Err("latch condition depends on a value loaded inside the loop".to_owned())
                } else {
                    Err("latch condition depends on a value loaded from memory".to_owned())
                }
            }
            "add" if ops.len() == 3 => arg(1)?.combine(arg(2)?, 1),
            "sub" if ops.len() == 3 => arg(1)?.combine(arg(2)?, -1),
            "and" if ops.len() == 3 => {
                // and r, a, mask — PTX's lowering of `a mod 2^n`.
                let (val, mask) = (arg(1), arg(2));
                let (val, mask) = match (val, mask) {
                    (Ok(v), Ok(m)) => (v, m),
                    (Err(e), _) | (_, Err(e)) => return Err(e),
                };
                let (affine, konst) = match (mask.base.as_const(), val.base.as_const()) {
                    (Some(c), _) if mask.iv.is_none() => (val, c),
                    (_, Some(c)) if val.iv.is_none() => (mask, c),
                    _ => return Err("and-mask with two non-constant operands".to_owned()),
                };
                let modulus = konst
                    .checked_add(1)
                    .filter(|m| *m > 0 && (m & (m - 1)) == 0)
                    .ok_or_else(|| format!("and-mask {konst:#x} is not 2^n − 1"))?;
                if affine.iv.is_some() {
                    return Err("mod applied to an induction variable".to_owned());
                }
                Ok(Affine::invariant(SymExpr::modulo(affine.base, modulus)))
            }
            "shl" if ops.len() == 3 => {
                let v = arg(1)?;
                let sh = arg(2)?;
                match (sh.iv, sh.base.as_const()) {
                    (None, Some(c)) if (0..63).contains(&c) => Ok(v.scale(1i64 << c)),
                    _ => Err("shift by a non-constant amount".to_owned()),
                }
            }
            "shr" if ops.len() == 3 => {
                let v = arg(1)?;
                let sh = arg(2)?;
                let Some(c) = sh.iv.is_none().then(|| sh.base.as_const()).flatten() else {
                    return Err("shift by a non-constant amount".to_owned());
                };
                if v.iv.is_some() {
                    return Err("shift applied to an induction variable".to_owned());
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
                let a = arg(1)?;
                let b = arg(2)?;
                match (
                    a.iv.is_none().then(|| a.base.as_const()).flatten(),
                    b.iv.is_none().then(|| b.base.as_const()).flatten(),
                ) {
                    (Some(c), _) => Ok(b.scale(c)),
                    (_, Some(c)) => Ok(a.scale(c)),
                    _ if a.iv.is_none() && b.iv.is_none() => {
                        Ok(Affine::invariant(SymExpr::mul(a.base, b.base)))
                    }
                    _ => Err("product involving an induction variable".to_owned()),
                }
            }
            "mad" if ops.len() == 4 => {
                let a = arg(1)?;
                let b = arg(2)?;
                let c = arg(3)?;
                let prod = match (
                    a.iv.is_none().then(|| a.base.as_const()).flatten(),
                    b.iv.is_none().then(|| b.base.as_const()).flatten(),
                ) {
                    (Some(k), _) => b.scale(k),
                    (_, Some(k)) => a.scale(k),
                    _ if a.iv.is_none() && b.iv.is_none() => {
                        Affine::invariant(SymExpr::mul(a.base, b.base))
                    }
                    _ => return Err("product involving an induction variable".to_owned()),
                };
                prod.combine(c, 1)
            }
            other => Err(format!(
                "value defined by unsupported instruction `{other}`"
            )),
        }
    }

    fn in_loop(&self, id: LoopId, stmt: usize) -> bool {
        self.forest.get(id).blocks.iter().any(|&b| {
            let blk = self.cfg.block(b);
            blk.start <= stmt && stmt < blk.end
        })
    }
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

/// Solve `continue while A1·k + A0 cmp 0` (k = 1, 2, ...) for the
/// iteration count.
fn solve(cmp: &str, continue_if_true: bool, a1: i64, a0: SymExpr) -> TripCount {
    // Normalize to the continue-condition comparison.
    let cond = if continue_if_true {
        cmp.to_owned()
    } else {
        match cmp {
            "lt" => "ge",
            "le" => "gt",
            "gt" => "le",
            "ge" => "lt",
            "ne" => "eq",
            "eq" => "ne",
            other => other,
        }
        .to_owned()
    };
    let neg = |e: SymExpr| SymExpr::mul(SymExpr::Const(-1), e);

    match cond.as_str() {
        // D ≠ 0: exits exactly when A1·k = −A0.
        "ne" => {
            if a1 == 0 {
                return Err("latch condition does not involve an induction variable".to_owned());
            }
            let num = if a1 > 0 { neg(a0) } else { a0 };
            Ok(SymExpr::floor_div(num, a1.abs()))
        }
        // D < 0: false at the smallest k with A1·k ≥ −A0.
        "lt" => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::ceil_div(neg(a0), a1)),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D ≤ 0: false at the smallest k with A1·k > −A0.
        "le" => match a1 {
            a1 if a1 > 0 => Ok(SymExpr::add(
                SymExpr::floor_div(neg(a0), a1),
                SymExpr::Const(1),
            )),
            0 => Err("latch condition does not involve an induction variable".to_owned()),
            _ => Err("loop bound moves away from the exit condition".to_owned()),
        },
        // D > 0 ⇔ −D < 0;   D ≥ 0 ⇔ −D ≤ 0.
        "gt" => solve("lt", true, -a1, neg(a0)),
        "ge" => solve("le", true, -a1, neg(a0)),
        "eq" => Err("loop continues only while values are equal — shape not recognized".to_owned()),
        other => Err(format!("unsupported latch comparison `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{build_cfg, loop_forest, loop_names};
    use crate::parse::parser::parse;

    /// Trips of every loop in a one-kernel source, by display name.
    fn trips_of(src: &str) -> Vec<(String, TripCount)> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        names
            .into_iter()
            .zip(info.trips)
            .map(|(n, t)| (n.display, t))
            .collect()
    }

    fn kernel_with(params: &str, body: &str) -> String {
        format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k(\n{params}\n)\n{{\n{body}\n}}\n"
        )
    }

    const N_PARAM: &str = ".param .u32 k_param_0";

    #[test]
    fn up_counting_lt_stride_1() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn up_counting_ne_form() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ne.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn stride_gt_1_is_ceil_div() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 8;\n\
             setp.lt.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "⌈param_0/8⌉");
    }

    #[test]
    fn signed_division_by_power_of_two_bound() {
        // nvcc's lowering of `n / 16` for a signed `n`, verbatim from the
        // k14 fixture: sign bit, shifted down, added, arithmetic shift.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\n\
             shr.s32 %r2, %r1, 31;\nshr.u32 %r3, %r2, 28;\n\
             add.s32 %r4, %r1, %r3;\nshr.s32 %r5, %r4, 4;\n\
             mov.u32 %r6, 0;\n\
             $L__L:\nadd.s32 %r6, %r6, 1;\n\
             setp.lt.u32 %p1, %r6, %r5;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0 / 16");
    }

    #[test]
    fn increment_through_an_intermediate_register() {
        // k14's main loop: `t = i + 2` is also read by the body, so nvcc
        // increments as `i = t - 1`.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r3, %r2, 2;\nsetp.ge.u32 %p2, %r3, %r1;\n\
             add.s32 %r2, %r3, -1;\n\
             setp.lt.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn countdown_to_zero() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, %r1;\n\
             $L__L:\nadd.s32 %r2, %r2, -1;\n\
             setp.ne.s32 %p1, %r2, 0;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn sixty_four_bit_iv() {
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\nmov.u64 %rd2, 0;\n\
             $L__L:\nadd.s64 %rd2, %rd2, 1;\n\
             setp.lt.s64 %p1, %rd2, %rd1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn rotated_do_while_constant_bound() {
        // do { } while (++i != 8) — the k5 inner shape with constants.
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ne.s32 %p1, %r2, 8;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "8");
    }

    #[test]
    fn derived_register_latch_verbatim_from_k2() {
        // The verified k2 shape, pinned verbatim: the latch compares
        // %r29 = %r7 + %r35 against 0, where %r7 = (K&3) − K.
        let src = kernel_with(
            ".param .u32 k_param_0,\n.param .u32 k_param_1,\n.param .u32 k_param_2",
            "ld.param.u32 %r19, [k_param_2];\n\
             and.b32 %r37, %r19, 3;\n\
             sub.s32 %r7, %r37, %r19;\n\
             mov.u32 %r35, 0;\n\
             $L__BB0_4:\n\
             add.s32 %r35, %r35, 4;\n\
             add.s32 %r29, %r7, %r35;\n\
             setp.ne.s32 %p6, %r29, 0;\n@%p6 bra $L__BB0_4;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(
            trips[0].1.as_ref().unwrap().to_string(),
            "(param_2 - param_2 mod 4) / 4"
        );
    }

    #[test]
    fn negated_predicate_inverts_the_condition() {
        // @!%p bra header with setp.ge: continue while NOT(i >= N) = i < N.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.ge.s32 %p1, %r2, %r1;\n@!%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0");
    }

    #[test]
    fn multi_exit_is_a_named_unknown() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.gt.s32 %p2, %r2, 100;\n@%p2 bra $L__OUT;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\n\
             $L__OUT:\nret;",
        );
        let trips = trips_of(&src);
        let err = trips[0].1.as_ref().unwrap_err();
        assert!(err.contains("exit"), "{err}");
    }

    #[test]
    fn data_dependent_latch_is_a_named_unknown() {
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\n\
             $L__L:\nld.global.f32 %f1, [%rd1];\nadd.s64 %rd1, %rd1, 4;\n\
             setp.gt.f32 %p1, %f1, 0f00000000;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(
            trips[0].1.as_ref().unwrap_err(),
            "latch condition depends on a value loaded inside the loop"
        );
    }

    #[test]
    fn special_register_dependence_is_named() {
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, %tid.x;\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        let err = trips[0].1.as_ref().unwrap_err();
        assert!(err.contains("special register %tid.x"), "{err}");
    }

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
        let m = parse(&src).expect("parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        assert_eq!(info.unroll_pairs.len(), 1, "{:?}", info);
        assert_eq!(info.unroll_pairs[0].factor, 4);
        let main = info.unroll_pairs[0].main;
        assert_eq!(
            info.trips[main.0 as usize].as_ref().unwrap().to_string(),
            "(param_0 - param_0 mod 4) / 4"
        );
    }
}
