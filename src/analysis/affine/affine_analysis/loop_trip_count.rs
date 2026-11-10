//! A loop's trip count: its latch's comparison solved for the counter's
//! iteration, per execution of the guard that skips the loop or per
//! entry.

use super::*;

impl AffineAnalysis<'_> {
    /// The count loop `id`'s latch comparison gives.
    pub(super) fn counted_trips(&self, id: LoopId) -> Result<LoopTripCount, UnknownReason> {
        let l = self.data.forest.get(id);
        let per_entry = |count| LoopTripCount {
            count,
            entry: l.header,
            distance: None,
        };
        let (latch, _) = self.exiting_latch(id)?;
        let (pos, pred, continue_if_true) = self.branch_to(latch, l.header)?;

        // The setp defining the branch predicate, in a block of this loop,
        // not of a nested one, that dominates the latch.
        let found = (self.data.setp_at(pred, pos)).filter(|&(d, ..)| {
            let b = self.data.block_of(d);
            self.data.forest.block_loop[b.0 as usize] == Some(id)
                && self.data.forest.doms.dominates(b, latch)
        });
        let Some(setp) = found else {
            return self
                .predicate_phi_trips(id, latch, pos, pred, continue_if_true)
                .unwrap_or_else(|| Err(self.predicate_reason(latch, pos, pred)))
                .map(per_entry);
        };
        // D = A − B; the latch continues where `D cond 0`.
        let (d, cond) = self.predicate_cmp(id, setp, pred, continue_if_true)?;

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
        let a1 = coeff
            .as_const()
            .ok_or_else(|| "induction step is not a constant".to_owned())?;
        if a1 == i64::MIN {
            return Err("induction step is −2^63, which has no negation".into());
        }
        let trips = exit_limit(cond, a1, d.base.clone())?;
        // `!=` (an `==` latch has no `exit_limit`): the count holds where
        // the step divides the distance, which the reader checks.
        let Some(at_zero) = below_zero(cond, d.base.clone()) else {
            let distance = Some((SymExpr::mul(SymExpr::Const(-1), d.base), a1));
            return Ok(LoopTripCount {
                count: trips,
                entry: l.header,
                distance,
            });
        };
        // Under a guard entering the loop exactly where the latch's
        // condition holds of the counter's start, D(0), the count is per
        // guard and from 0; else per entry, and each entry runs ≥ 1 trip.
        Ok(match self.zero_trip_guard(id, &at_zero) {
            Some(g) => LoopTripCount {
                count: self.max(trips, SymExpr::Const(0)),
                entry: g,
                distance: None,
            },
            None => per_entry(self.max(trips, SymExpr::Const(1))),
        })
    }

    /// Loop `id`'s guard (`loop_guard`), where it enters the loop exactly
    /// when `enters < 0`: the test read from the `setp` that reaches its
    /// branch predicate, `!=` and `==` of a side never negative as `>` and
    /// `<=`.
    fn zero_trip_guard(&self, id: LoopId, enters: &SymExpr) -> Option<BlockId> {
        let (g, into) = self.loop_guard(id)?;
        let (pos, pred, enters_if_true) = self.branch_to(g, into).ok()?;
        let setp = self.data.setp_at(pred, pos)?;
        let (form, cmp) = self.predicate_cmp(id, setp, pred, enters_if_true).ok()?;
        let e = form.is_symbolic_or_constant().then_some(form.base)?;
        let nonneg = |e: &SymExpr| self.nonnegative(&AffineExpr::from_base(e.clone()));
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
        (below_zero(cmp, e)? == *enters).then_some(g)
    }

    /// LLVM's loop guard branch (`Loop::getLoopGuardBranch`,
    /// llvm/lib/Analysis/LoopInfo.cpp): above the header's one predecessor
    /// outside the loop, through blocks of one predecessor and one successor
    /// and loops left only through their one latch, none ending in a
    /// predicated `ret` or `exit`, the first block, which
    /// must lie in the loop's parent and end in a two-way branch whose other
    /// successor does not reach the header without passing it. That block,
    /// and its successor toward loop `id`.
    fn loop_guard(&self, id: LoopId) -> Option<(BlockId, BlockId)> {
        let (cfg, forest) = (self.data.cfg, self.data.forest);
        let l = forest.get(id);
        let (mut g, mut into) = (l.source_of_only_entering_edge(cfg)?, l.header);
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
        let blocked = taken == other || in_loop(g) != l.parent;
        (!blocked && !self.data.path_avoiding(skip, l.header, g)).then_some((g, into))
    }

    /// Block `b`'s closing `@p bra`: its position, p, and whether it goes
    /// to `to` where p holds. `succs[0]` is the taken target (graph
    /// contract); the predicate's negation flips that once more.
    fn branch_to(&self, b: BlockId, to: BlockId) -> Result<(usize, Symbol, bool), UnknownReason> {
        let (pos, bra) = (self.data.last_instr(b))
            .ok_or_else(|| "latch block has no instructions".to_owned())?;
        let pred = (bra.predicate).ok_or_else(|| "latch branch is unconditional".to_owned())?;
        let taken = self.data.cfg.block(b).succs.first() == Some(&to);
        Ok((pos, pred.reg, taken != pred.negated))
    }

    /// `form cmp 0` (`loop_cmp`), holding exactly where `pred`, written by
    /// `setp` (`setp_at`'s triple: its second output where `complement`),
    /// is `holds`; both operands read at the setp as `cta_uniform_symbols`
    /// reads them.
    fn predicate_cmp(
        &self,
        id: LoopId,
        (def, setp, complement): (usize, &Instr, bool),
        pred: Symbol,
        holds: bool,
    ) -> Result<(AffineExpr, Cmp), UnknownReason> {
        let module = self.data.module;
        if setp.predicate.is_some() {
            let name = module.interner.resolve(pred);
            return Err(format!("{name} is written under a guard").into());
        }
        let (compare, width, a, b) = self.predicate_and_operands_of_int_setp(def, setp, 0)?;
        let (a, b) = self.cta_uniform_symbols(id, def, a, b)?;
        let (d, cmp) = (self.loop_cmp(id, compare, width, &a, &b))
            .ok_or_else(|| "latch condition does not read as one integer comparison".to_owned())?;
        Ok((
            d,
            if holds != complement {
                cmp
            } else {
                cmp.inverse()
            },
        ))
    }

    /// Loop `id`'s one latch, and the block its one exit edge, from that
    /// latch, enters.
    fn exiting_latch(&self, id: LoopId) -> Result<(BlockId, BlockId), UnknownReason> {
        let (cfg, l) = (self.data.cfg, self.data.forest.get(id));
        let &[latch] = &l.latches[..] else {
            return Err("loop has multiple latches".into());
        };
        let exits: Vec<(BlockId, BlockId)> = l.exit_edges(cfg).collect();
        match exits[..] {
            [(b, s)] if b == latch => Ok((latch, s)),
            [_] => Err("loop exit is not at the latch".into()),
            _ => Err(format!("loop has {} exit edges", exits.len()).into()),
        }
    }

    /// The one block entering loop S from outside, and S's header, where `g`
    /// is S's one latch and `g → into` its one exit edge.
    fn sibling_entry(&self, g: BlockId, into: BlockId) -> Option<(BlockId, BlockId)> {
        let forest = self.data.forest;
        let id = forest.block_loop[g.0 as usize]?;
        let s = forest.get(id);
        if self.exiting_latch(id).ok() != Some((g, into))
            || s.blocks.iter().any(|&b| self.exits_some(b))
        {
            return None;
        }
        Some((s.source_of_only_entering_edge(self.data.cfg)?, s.header))
    }

    /// Block `b` ends in a predicated `ret` or `exit`, which the CFG reads as
    /// falling through.
    fn exits_some(&self, b: BlockId) -> bool {
        (self.data.last_instr(b))
            .is_some_and(|(_, i)| matches!(i.mnemonic, Opcode::Ret | Opcode::Exit))
    }

    /// `a` and `b`, read at `pos`, with each unknown value and `%ctaid` as a
    /// symbol, where `a − b` is one value at every trip of every entry of
    /// loop `id`: every thread of the CTA sees it alike and the loops
    /// holding `id` do not define it (`alike`).
    fn cta_uniform_symbols(
        &self,
        id: LoopId,
        pos: usize,
        a: AffineExpr,
        b: AffineExpr,
    ) -> Result<(AffineExpr, AffineExpr), UnknownReason> {
        let top = (self.data.forest.loop_and_enclosing_loops(id).last()).unwrap_or(id);
        self.alike(&(a.clone() - b.clone()), pos, Uniform::Cta, Some(top))?;
        let symbols = |x: AffineExpr| {
            self.per_cta(x.as_symbols(&|v| match v {
                Var::Unknown(u) => Some(self.symbol(SymbolicConstant::Unknown(*u))),
                _ => None,
            }))
        };
        Ok((symbols(a), symbols(b)))
    }

    /// The comparison `form cmp 0` a latch or guard `setp` `pred` of loop
    /// `id` makes of `width`-bit operands `a` and `b`
    /// (`setp_to_zero_cmps`), an operand taken as nonnegative where `least`
    /// proves it, or where it does not read the loop's counter and its
    /// range is unknown, as the input contract (PLAN.md) reads a trip
    /// count; `None` where it is not one comparison.
    fn loop_cmp(
        &self,
        id: LoopId,
        pred: IntPredicate,
        width: u32,
        a: &AffineExpr,
        b: &AffineExpr,
    ) -> Option<(AffineExpr, Cmp)> {
        let nonnegative = |x: &AffineExpr| {
            let counted = x.terms.keys().any(|v| v.counter() == Some(id));
            (!counted && self.range(x).is_none()) || self.nonnegative(x)
        };
        match &self.setp_to_zero_cmps(pred, width, a, b, &nonnegative)?[..] {
            [(form, cmp)] => Some((form.clone(), *cmp)),
            _ => None,
        }
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
            .filter(|(p, _)| p.is_none_or(|p| !lp.contains_block(p)))
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
