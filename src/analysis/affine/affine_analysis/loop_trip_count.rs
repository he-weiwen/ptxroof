//! A loop's trip count: its latch's comparison solved for the counter's
//! iteration, per execution of the guard that skips the loop or per
//! entry.

use super::*;

impl AffineAnalysis<'_> {
    /// The count loop `id`'s latch comparison gives.
    pub(super) fn counted_trips(&self, id: LoopId) -> Result<LoopTripCount, UnknownReason> {
        let (module, cfg) = (self.data.module, self.data.cfg);
        let l = self.data.forest.get(id);

        // Structural requirements: one latch, one exit edge, at the latch.
        if l.latches.len() != 1 {
            return Err("loop has multiple latches".into());
        }
        let latch = l.latches[0];
        let in_loop = |b: BlockId| l.blocks.binary_search(&b).is_ok();
        let mut exit_edges = Vec::new();
        for &b in &l.blocks {
            for &s in &cfg.block(b).succs {
                if !in_loop(s) {
                    exit_edges.push((b, s));
                }
            }
        }
        if exit_edges.len() != 1 {
            return Err(format!("loop has {} exit edges", exit_edges.len()).into());
        }
        if exit_edges[0].0 != latch {
            return Err("loop exit is not at the latch".into());
        }

        // The latch terminator: `@%p bra HEADER` (or inverted).
        let latch_block = cfg.block(latch);
        let branch = self
            .data
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

        // The setp defining the branch predicate: in the latch block, or in
        // a block of this loop, not of a nested one, that dominates it.
        let dominating = || {
            let Some(Value::Result { op, .. }) = self
                .data
                .ssa
                .value_before(branch.0, Register::Named(pred.reg))
            else {
                return None;
            };
            let b = self.data.block_of(op);
            let own = self.data.forest.block_loop[b.0 as usize] == Some(id);
            (own && self.data.forest.doms.dominates(b, latch))
                .then(|| self.data.find_setp(b, op + 1, pred.reg))?
        };
        let found = (self.data.find_setp(latch, branch.0, pred.reg)).or_else(dominating);
        let Some((setp_idx, setp)) = found else {
            return self
                .predicate_phi_trips(id, latch, branch.0, pred.reg, continue_if_true)
                .unwrap_or_else(|| Err(self.predicate_reason(latch, branch.0, pred.reg)))
                .map(|count| LoopTripCount {
                    count,
                    entry: l.header,
                    distance: None,
                });
        };
        if setp.predicate.is_some() {
            let name = module.interner.resolve(pred.reg);
            return Err(format!("{name} is written under a guard").into());
        }
        let compared = match Form::of(module, setp) {
            Form::IntCompare { pred, .. } => Some(pred),
            // PTX ISA §9.7.7.2: a packed `setp`'s first output compares the low halves.
            Form::Float { lanes: 2 } => {
                return Err("latch condition compares packed halves".into());
            }
            Form::Float { .. } => None,
            _ => return Err("setp without comparison modifier".into()),
        };

        // Trace both operands at the setp.
        let ops = module.operand_ids(setp.operands);
        match ops.len() {
            3 => {}
            // PTX ISA §9.7.6.2: `setp.CmpOp.BoolOp.type p, a, b, c` sets
            // `p = BoolOp(a CmpOp b, c)`.
            4 => return Err("latch condition combines a comparison with a predicate".into()),
            _ => return Err("setp with unexpected operand count".into()),
        }
        // PTX ISA §9.7.6.2: a `setp` of no integer type compares floats,
        // not the traced integers.
        let Some(pred) = compared else {
            return Err("latch condition compares floating-point values".into());
        };
        let a = self.operand_at(ops[1], setp_idx, 0)?;
        let b = self.operand_at(ops[2], setp_idx, 0)?;
        let d = per_cta(self.cta_uniform_symbols(id, setp_idx, a - b)?); // D = A − B; condition is `D cmp 0`

        // D(k) = A1·k + A0 in this loop's iteration number alone.
        let k = Var::Iteration(id);
        if let Some(v) = d.terms.keys().find(|v| **v != k) {
            return Err(match v {
                Var::Iteration(_) => {
                    "latch condition depends on an enclosing loop's counter".into()
                }
                other => format!("latch condition depends on special register {other}").into(),
            });
        }
        let Some(coeff) = d.terms.get(&k) else {
            return Err("latch condition does not involve an induction variable".into());
        };
        let a1 = (self.constant(&AffineExpr::from_base(coeff.clone())))
            .ok_or_else(|| "induction step is not a constant".to_owned())?;
        if a1 == i64::MIN {
            return Err("induction step is −2^63, which has no negation".into());
        }
        let cmp = pred.cmp();
        let cond = if continue_if_true { cmp } else { cmp.inverse() };
        let bound = self.bindings.and_then(|b| d.base.bind(b).as_const());
        if cond == Cmp::Ne && bound.is_some_and(|x| x.checked_rem(a1).is_some_and(|r| r != 0)) {
            return Err("latch condition never meets its `!=` bound".into());
        }
        let trips = exit_limit(cond, a1, d.base.clone())?;
        let distance =
            (cond == Cmp::Ne).then(|| (SymExpr::mul(SymExpr::Const(-1), d.base.clone()), a1));
        match (below_zero(cond, d.base), self.guard_test(id)) {
            (Some(at_zero), Some((g, test))) if test == at_zero => Ok(LoopTripCount {
                count: SymExpr::max(trips, SymExpr::Const(0)),
                entry: g,
                distance: None,
            }),
            (Some(_), _) => Ok(LoopTripCount {
                count: SymExpr::max(trips, SymExpr::Const(1)),
                entry: l.header,
                distance: None,
            }),
            _ => Ok(LoopTripCount {
                count: trips,
                entry: l.header,
                distance,
            }),
        }
    }

    /// LLVM's loop guard branch (`Loop::getLoopGuardBranch`,
    /// llvm/lib/Analysis/LoopInfo.cpp): above the header's one predecessor
    /// outside the loop, through blocks of one predecessor and one successor
    /// and loops left only through their one latch, none ending in a
    /// predicated `ret` or `exit`, the first block, which
    /// must lie in the loop's parent and end in a two-way branch whose other
    /// successor does not reach the header without passing it. That block,
    /// and its test of entering the loop, read from the `setp` that reaches
    /// its branch predicate, as `e < 0`; `!=` and `==` of a side never
    /// negative as `>` and `<=`.
    fn guard_test(&self, id: LoopId) -> Option<(BlockId, SymExpr)> {
        let (module, cfg, forest) = (self.data.module, self.data.cfg, self.data.forest);
        let l = forest.get(id);
        let mut outside =
            (cfg.block(l.header).preds.iter()).filter(|p| l.blocks.binary_search(p).is_err());
        let (&pre, None) = (outside.next()?, outside.next()) else {
            return None;
        };
        let (mut g, mut into) = (pre, l.header);
        loop {
            (g, into) = match (&cfg.block(g).succs[..], &cfg.block(g).preds[..]) {
                ([_], &[p]) if !self.exits_some(g) => (p, g),
                ([_, _], _) => match self.sibling_entry(g, into) {
                    Some(entry) => entry,
                    None => break,
                },
                _ => break,
            };
        }
        let &[taken, other] = &cfg.block(g).succs[..] else {
            return None;
        };
        let skip = if taken == into { other } else { taken };
        let in_loop = |b: BlockId| forest.block_loop[b.0 as usize];
        if taken == other || in_loop(g) != l.parent || self.data.path_avoiding(skip, l.header, g) {
            return None;
        }
        let (pos, bra) = self.data.last_instr(g)?;
        let pred = bra.predicate?;
        let Some(Value::Result { op, .. }) =
            self.data.ssa.value_before(pos, Register::Named(pred.reg))
        else {
            return None;
        };
        let (def, setp) = self
            .data
            .find_setp(self.data.block_of(op), op + 1, pred.reg)?;
        let Form::IntCompare { pred: compare, .. } = Form::of(module, setp) else {
            return None;
        };
        let &[_, a, b] = module.operand_ids(setp.operands) else {
            return None;
        };
        if setp.predicate.is_some() {
            return None;
        }
        let form = per_cta(
            self.cta_uniform_symbols(
                id,
                def,
                self.operand_at(a, def, 0).ok()? - self.operand_at(b, def, 0).ok()?,
            )
            .ok()?,
        );
        let cmp = compare.cmp();
        let cmp = if (taken == into) != pred.negated {
            cmp
        } else {
            cmp.inverse()
        };
        let e = form.is_symbolic_or_constant().then_some(form.base)?;
        let nonneg = |e: &SymExpr| e.least().is_some_and(|l| l >= 0);
        let (cmp, e) = match cmp {
            Cmp::Ne | Cmp::Eq => {
                let e = if nonneg(&e) {
                    e
                } else {
                    SymExpr::mul(SymExpr::Const(-1), e)
                };
                (
                    if cmp == Cmp::Ne { Cmp::Gt } else { Cmp::Le },
                    nonneg(&e).then_some(e)?,
                )
            }
            _ => (cmp, e),
        };
        Some((g, below_zero(cmp, e)?))
    }

    /// The one block entering loop S from outside, and S's header, where `g`
    /// is S's one latch and `g → into` its one exit edge.
    fn sibling_entry(&self, g: BlockId, into: BlockId) -> Option<(BlockId, BlockId)> {
        let (forest, cfg) = (self.data.forest, self.data.cfg);
        let s = forest.get(forest.block_loop[g.0 as usize]?);
        let inside = |b: &BlockId| s.blocks.binary_search(b).is_ok();
        let exits = (s.blocks.iter())
            .flat_map(|&b| cfg.block(b).succs.iter().map(move |&t| (b, t)))
            .filter(|(_, t)| !inside(t));
        if s.latches != [g]
            || !exits.eq([(g, into)])
            || s.blocks.iter().any(|&b| self.exits_some(b))
        {
            return None;
        }
        let mut outside = (cfg.block(s.header).preds.iter()).filter(|p| !inside(p));
        let (&q, None) = (outside.next()?, outside.next()) else {
            return None;
        };
        Some((q, s.header))
    }

    /// Block `b` ends in a predicated `ret` or `exit`, which the CFG reads as
    /// falling through.
    fn exits_some(&self, b: BlockId) -> bool {
        (self.data.last_instr(b))
            .is_some_and(|(_, i)| matches!(i.mnemonic, Opcode::Ret | Opcode::Exit))
    }

    /// `a`, read at `pos`, with each unknown value as a symbol, where it is
    /// one value at every trip of every entry of loop `id`: every thread
    /// of the CTA sees it alike and the loops holding `id` do not define
    /// it (`alike`).
    fn cta_uniform_symbols(
        &self,
        id: LoopId,
        pos: usize,
        a: AffineExpr,
    ) -> Result<AffineExpr, UnknownReason> {
        let top = (std::iter::successors(Some(id), |&l| self.data.forest.get(l).parent).last())
            .unwrap_or(id);
        self.alike(&a, pos, Uniform::Cta, Some(top))?;
        Ok(a.as_symbols(&|v| match v {
            Var::Unknown(u) => Some(SymExpr::Sym(SymbolicConstant::Unknown(*u))),
            _ => None,
        }))
    }

    /// No setp defines the latch predicate: say what does, if anything
    /// in the latch block does (`mbarrier.test_wait` in a spin-wait).
    fn predicate_reason(&self, latch: BlockId, before: usize, pred: Symbol) -> UnknownReason {
        let module = self.data.module;
        let definer = self
            .data
            .last_write_in(latch, before, pred)
            .map(|d| &self.data.kernel.stmts[d]);
        match definer {
            Some(Stmt::Instr(i)) if i.mnemonic == Opcode::Setp => format!(
                "latch predicate is not `{}`'s first output",
                module.opcode(i)
            )
            .into(),
            Some(Stmt::Instr(i)) => format!(
                "latch predicate is defined by `{}`, not a comparison",
                module.opcode(i)
            )
            .into(),
            _ => "latch predicate is not defined in the latch block".into(),
        }
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
    ) -> Option<Result<SymExpr, UnknownReason>> {
        let (module, kernel, cfg, ssa) = (
            self.data.module,
            self.data.kernel,
            self.data.cfg,
            &self.data.ssa,
        );
        let mov_src = |site: usize| -> Option<&Operand> {
            let Stmt::Instr(instr) = &kernel.stmts[site] else {
                return None;
            };
            if instr.mnemonic != Opcode::Mov {
                return None;
            }
            let [_, src] = module.operand_ids(instr.operands) else {
                return None;
            };
            Some(module.operand(*src))
        };
        let Some(Value::Result { op: copy, .. }) =
            ssa.value_before(branch_idx, Register::Named(pred))
        else {
            return None;
        };
        let Operand::Register(phi) = mov_src(copy)? else {
            return None;
        };
        if !self.data.in_loop(id, copy) {
            return None;
        }
        let latch_blk = cfg.block(latch);
        let mut in_loop = self
            .data
            .writes
            .get(&Register::Named(*phi))?
            .iter()
            .filter(|&&d| self.data.in_loop(id, d));
        let (&latch_def, None) = (in_loop.next()?, in_loop.next()) else {
            return None;
        };
        if latch_def < latch_blk.start || latch_def >= latch_blk.end {
            return None;
        }
        let lp = self.data.forest.get(id);
        let entry = self
            .data
            .ssa
            .value_before(cfg.block(lp.header).start, Register::Named(*phi))?;
        let outside: Vec<Value> = (ssa.incoming(entry).into_iter())
            .filter(|(p, _)| p.is_none_or(|p| lp.blocks.binary_search(&p).is_err()))
            .map(|(_, v)| v)
            .collect();
        let [Value::Result { op: init_def, .. }, ..] = outside[..] else {
            return None;
        };
        if outside.iter().any(|&v| v != outside[0]) {
            return None;
        }
        for (site, reg) in [(copy, pred), (latch_def, *phi), (init_def, *phi)] {
            if let Stmt::Instr(instr) = &kernel.stmts[site]
                && instr.predicate.is_some()
            {
                let name = module.interner.resolve(reg);
                return Some(Err(format!("{name} is written under a guard").into()));
            }
        }
        let (Operand::Immediate(latch_val), Operand::Immediate(init_val)) =
            (mov_src(latch_def)?, mov_src(init_def)?)
        else {
            return None;
        };
        let continues = |text: &Symbol| {
            parse_int(module.interner.resolve(*text)).map(|v| (v != 0) == continue_if_true)
        };
        if continues(latch_val)? {
            return None;
        }
        Some(Ok(SymExpr::Const(if continues(init_val)? { 2 } else { 1 })))
    }
}
