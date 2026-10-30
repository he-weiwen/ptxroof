//! Affine analysis: a register's value at a statement, from register
//! SSA (`crate::ssa`), as an affine form over thread and CTA indices,
//! loop counters and the kernel parameters; and one loop's trip count,
//! read from its latch through the same values. Address and
//! branch-condition analyses read memory operands and predicates here.
//!
//! A loop counter is an induction variable (`super::induction_variables`),
//! read as its start plus `k − 1` steps; a φ at a join is the one value
//! with no loop counter that every path writes, another φ one immediate
//! every path writes, or refused. Supported arithmetic is followed down to
//! `ld.param`, constants and special registers; an unsupported operation
//! is a named reason, preferring the fundamental obstacle (a special
//! register, a memory load, an atomic) behind arithmetic it does not
//! read.
//!
//! A loop's latch (`setp` + `@%p bra`) is normalized to
//! `continue while A1·k + A0 cmp 0` (k = iteration number; a counter
//! read after its increment is init + k·step, read before it
//! init + (k − 1)·step) and solved by [`super::exit_limit::exit_limit`].
//! An ordered latch's count T is max(T, 0) per execution of a guard
//! branch that skips the loop where the latch fails at k = 0, and
//! max(T, 1) per entry otherwise, as LLVM's ScalarEvolution counts a
//! rotated loop's backedges from max(RHS, Start) unless a guard covers
//! it (llvm/lib/Analysis/AffineAnalysis.cpp:13628–13643 at
//! e32776a129f5).
//!
//! The two queries recurse into each other: a value after a loop, a
//! counter stepped by an inner loop's exit and an integer range check
//! need a constant trip count, and a latch reads values.
//! Each value's trace and each trip count is cached, except one that
//! read a trace or count still being computed outside it, as Tarjan's
//! lowlink tells which results of a strongly connected component are
//! final. A count read while it is being counted is unknown; one that
//! read itself so is counted again with itself and kept where both
//! agree.
//!
//! Anything else — multiple exits, exits not at the latch, latch values
//! loaded in the loop, dependence on a special register that is not one
//! value for the CTA, two IVs in one condition — is an `Err(reason)`: a
//! named unknown, never a guess. Symbols are assumed nonnegative; a
//! value whose range is unknown (through a parameter, `%ctaid` with no
//! grid given, a symbolic trip count) is assumed not to wrap its type,
//! and unsigned/signed comparison width is not modeled.

use crate::analysis::affine::affine_expr::{AffineExpr, Axis, Cmp, Var};
use crate::analysis::affine::affine_properties::AffineProperties;
use crate::analysis::affine::exit_limit::{below_zero, exit_limit};
use crate::analysis::affine::induction_variables::{InductionVariable, Step, induction_variables};
use crate::analysis::affine::int_type::{IntKind, IntType};
use crate::analysis::affine::lane_eval::depends_on_lane;
use crate::analysis::affine::sym_expr::{SymExpr, SymbolicConstant};
use crate::analysis::affine::unknown_reason::UnknownReason;
use crate::analysis::control_flow::loops::LoopId;
use crate::analysis::kernel_analyses::KernelAnalyses;
use crate::analysis::register_effects::{registers, split_operands};
use crate::analysis::special_registers::{SpecialRegister, special_register};
use crate::analysis::uniformity_info::UniformityInfo;
use crate::ptx::cfg::BlockId;
use crate::ptx::ir::{
    Element, Instr, Opcode, Operand, OperandId, Register, SpecialReg, SpecialScalar, SpecialVector,
    Stmt,
};
use crate::ptx::literal::parse_int;
use crate::ssa::Value;
use crate::support::intern::Symbol;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

mod loop_trip_count;
mod transfer;

/// How many definitions deep a trace follows a value before giving up.
const DEPTH_LIMIT: u32 = 256;

type Traced = (AffineProperties, Option<UnknownReason>);
type Memo<K, T> = RefCell<HashMap<K, Slot<T>>>;

/// A cached result, the frame computing it, or the count a loop's
/// frame is checking its count with.
enum Slot<T> {
    InProgress(usize),
    Candidate(usize, T),
    Done(T),
}

pub(crate) struct AffineAnalysis<'a> {
    pub(crate) data: &'a KernelAnalyses<'a>,
    /// Each loop counter by its header φ, with its loop.
    pub(crate) induction_variables: HashMap<Value, (LoopId, InductionVariable)>,
    /// Each loop's trip count.
    trips: Memo<LoopId, Result<LoopTripCount, UnknownReason>>,
    /// Each value's trace for the loop whose latch reads it, with the
    /// first reason it read a uniform value as a variable.
    values: Memo<(Value, Option<LoopId>), Traced>,
    /// How many traces and counts are being computed.
    frames: Cell<usize>,
    /// The outermost of them the result being computed read, or 0 after a
    /// trace cut at `DEPTH_LIMIT`: no result is cached that read one
    /// outside its own computation.
    provisional: Cell<usize>,
    /// `--bind` values, applied where a trace needs a constant.
    pub(crate) bindings: Option<&'a HashMap<SymbolicConstant, i64>>,
    uniformity: &'a UniformityInfo<'a>,
    /// Why the trace in progress first read a uniform value it cannot
    /// interpret as a `Var::Unknown`.
    unread: RefCell<Option<UnknownReason>>,
}

impl<'a> AffineAnalysis<'a> {
    /// The kernel's analysis under `bindings`, with every loop counted
    /// first, inner loops first, so that a count is the outermost
    /// computation of any cycle through it.
    pub(crate) fn new(
        data: &'a KernelAnalyses<'a>,
        uniformity: &'a UniformityInfo<'a>,
        bindings: Option<&'a HashMap<SymbolicConstant, i64>>,
    ) -> Self {
        let (forest, ssa) = (data.forest, &data.ssa);
        // Inner loops first: an outer counter may step by an inner exit.
        let mut order: Vec<LoopId> = (0..forest.loops.len() as u32).map(LoopId).collect();
        order.sort_by_key(|&l| std::cmp::Reverse(forest.get(l).depth));
        let (mut recurrences_by_phi, mut by_update) = (HashMap::new(), HashMap::new());
        for &l in &order {
            let lp = forest.get(l);
            for r in induction_variables(
                data.module,
                data.kernel,
                data.cfg,
                forest,
                ssa,
                l,
                &by_update,
            ) {
                for (p, v) in ssa.incoming(r.phi) {
                    if p.is_some_and(|p| lp.blocks.binary_search(&p).is_ok()) {
                        by_update.insert(v, (l, r.clone()));
                    }
                }
                recurrences_by_phi.insert(r.phi, (l, r));
            }
        }
        let analysis = AffineAnalysis {
            data,
            induction_variables: recurrences_by_phi,
            trips: RefCell::new(HashMap::new()),
            values: RefCell::new(HashMap::new()),
            frames: Cell::new(0),
            provisional: Cell::new(usize::MAX),
            bindings,
            uniformity,
            unread: RefCell::new(None),
        };
        for &l in &order {
            let _ = analysis.loop_trip_count(l);
        }
        analysis
    }

    /// Every one of these definitions is `mov r, imm` with one immediate.
    fn same_immediate(&self, defs: &[usize]) -> Option<i64> {
        let module = self.data.module;
        let mut value = None;
        for &d in defs {
            let Stmt::Instr(instr) = &self.data.kernel.stmts[d] else {
                return None;
            };
            let [dst, src] = module.operand_ids(instr.operands) else {
                return None;
            };
            let (Operand::Register(_), Operand::Immediate(text)) =
                (module.operand(*dst), module.operand(*src))
            else {
                return None;
            };
            if instr.mnemonic != Opcode::Mov {
                return None;
            }
            let c = parse_int(module.interner.resolve(*text))?;
            if value.is_some_and(|v| v != c) {
                return None;
            }
            value = Some(c);
        }
        value
    }

    /// The one value every one of these definitions writes to `reg`, with
    /// no loop counter, which each definition reads in its own loops.
    fn same_value(
        &self,
        defs: &[usize],
        reg: Symbol,
        id: Option<LoopId>,
        depth: u32,
    ) -> Option<AffineExpr> {
        let outer = self.unread.take();
        let mut values = defs.iter().map(|&d| self.transfer(d, reg, id, depth).ok());
        let value = (values.next().flatten())
            .filter(|a| a.terms.keys().all(|v| v.counter().is_none()))
            .filter(|a| values.all(|b| b.as_ref() == Some(a)));
        let inner = self.unread.replace(outer);
        if value.is_some()
            && let Some(why) = inner
        {
            self.unread.borrow_mut().get_or_insert(why);
        }
        value
    }

    /// A merged value: the obstacle on any of its definitions' chains.
    pub(crate) fn obstacle_of_any_def(
        &self,
        reg: Symbol,
        id: Option<LoopId>,
    ) -> Option<UnknownReason> {
        self.defs_obstacle(reg, id, &mut HashSet::from([reg]))
    }

    /// The obstacle on any of `reg`'s definitions' chains, through
    /// registers not yet `seen`.
    fn defs_obstacle(
        &self,
        reg: Symbol,
        id: Option<LoopId>,
        seen: &mut HashSet<Symbol>,
    ) -> Option<UnknownReason> {
        self.data
            .writes
            .get(&Register::Named(reg))?
            .iter()
            .find_map(|&d| {
                let Stmt::Instr(instr) = &self.data.kernel.stmts[d] else {
                    return None;
                };
                (split_operands(self.data.module, instr).1.iter())
                    .find_map(|&op| self.operand_obstacle(op, d, id, seen))
            })
    }

    /// The bound block shape has one row and one plane.
    fn one_dimensional(&self) -> bool {
        self.bindings.is_some_and(|b| {
            let [_, y, z] = SpecialVector::Ntid.components();
            b.get(&SymbolicConstant::SpecialReg(y)) == Some(&1)
                && b.get(&SymbolicConstant::SpecialReg(z)) == Some(&1)
        })
    }

    /// `f` of `a`, or of `a` under the bindings where `f` needs a
    /// coefficient the bindings make a constant.
    fn reduced(
        &self,
        a: &AffineExpr,
        f: impl Fn(AffineExpr) -> Option<AffineExpr>,
    ) -> Option<AffineExpr> {
        f(a.clone()).or_else(|| f(a.bind(self.bindings?)))
    }

    /// A constant, after the bindings if there are any.
    pub(crate) fn constant(&self, a: &AffineExpr) -> Option<i64> {
        match self.bindings {
            Some(b) => a.bind(b).as_const(),
            None => a.as_const(),
        }
    }

    pub(crate) fn operand_at(
        &self,
        op: OperandId,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<AffineExpr, UnknownReason> {
        let module = self.data.module;
        match module.operand(op) {
            Operand::Register(reg) => self.at_use(*reg, pos, id, depth),
            Operand::Special(reg) => self.trace_special(*reg),
            Operand::Immediate(text) => {
                let text = module.interner.resolve(*text);
                parse_int(text)
                    .map(|c| AffineExpr::from_base(SymExpr::Const(c)))
                    .ok_or_else(|| format!("non-integer immediate {text}").into())
            }
            // A named location (a shared array, the dynamic shared base):
            // its address is a symbol.
            Operand::SymbolRef(s) => Ok(AffineExpr::from_base(SymExpr::sym(
                module.interner.resolve(*s),
            ))),
            Operand::Offset { base, offset } => Ok(self.operand_at(*base, pos, id, depth)?
                + AffineExpr::from_base(SymExpr::Const(*offset))),
            other => Err(format!("unsupported operand form in latch trace: {other:?}").into()),
        }
    }

    fn trace_special(&self, reg: SpecialReg) -> Result<AffineExpr, UnknownReason> {
        if let Some(v) = Var::special(reg) {
            return Ok(AffineExpr::var(v));
        }
        match reg {
            _ if special_register(reg) == Some(SpecialRegister::Zero) => {
                Ok(AffineExpr::from_base(SymExpr::Const(0)))
            }
            // PTX ISA §10.3: %laneid is "the thread's lane within the
            // warp", and §3.1: a warp "contains threads of consecutive,
            // increasing thread IDs with the first warp containing
            // thread 0"; in a one-dimensional block that is %tid.x mod 32.
            SpecialReg::Scalar(SpecialScalar::Laneid) if self.one_dimensional() => {
                Ok(AffineExpr::var(Var::Mod(Box::new(Var::Tid(Axis::X)), 32)))
            }
            // One value for the CTA (the launch shape, the cluster, the
            // shared memory sizes): a symbol, bound when known.
            _ if special_register(reg) == Some(SpecialRegister::PerCta) => Ok(
                AffineExpr::from_base(SymExpr::Sym(SymbolicConstant::SpecialReg(reg))),
            ),
            _ => Err(format!("latch condition depends on special register {}", reg.name()).into()),
        }
    }

    pub(crate) fn at_use(
        &self,
        reg: Symbol,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<AffineExpr, UnknownReason> {
        let Some(v) = self.data.ssa.value_before(pos, Register::Named(reg)) else {
            let name = self.data.module.interner.resolve(reg);
            return Err(format!("{name} is read in a block no path reaches").into());
        };
        match self.properties(v, id, depth) {
            AffineProperties::Expr(a) => Ok(self.after_exits(a, pos)),
            AffineProperties::Unknown {
                reason,
                min_trailing_zeros,
            } if !matches!(v, Value::Undef(_))
                && self.uniformity().is_uniform(reg, pos).is_ok() =>
            {
                self.unread.borrow_mut().get_or_insert(reason);
                let u = Var::Unknown(v);
                Ok(match min_trailing_zeros {
                    0 => AffineExpr::var(u),
                    k => {
                        AffineExpr::term(Var::FloorDiv(Box::new(u), 1 << k), SymExpr::Const(1 << k))
                    }
                })
            }
            AffineProperties::Unknown { reason, .. } => Err(reason),
        }
    }

    /// How many low bits every definition of `v` leaves 0: an unguarded
    /// `shl` by a constant fills as many with zeros as its count, the
    /// constant's unsigned 32-bit value (PTX ISA §9.7.8.8), all of them
    /// where it reaches the width, and an unguarded `and` with a constant
    /// clears those its mask does.
    fn min_trailing_zeros(&self, v: Value) -> u32 {
        let module = self.data.module;
        let zeros = |s: &Value| {
            let Value::Result { op, .. } = *s else {
                return 0;
            };
            let Stmt::Instr(i) = &self.data.kernel.stmts[op] else {
                return 0;
            };
            let ops = module.operand_ids(i.operands);
            let imm = |k: usize| match ops.get(k).map(|&o| module.operand(o)) {
                Some(Operand::Immediate(t)) => parse_int(module.interner.resolve(*t)),
                _ => None,
            };
            let k = match i.mnemonic {
                _ if i.predicate.is_some() => None,
                Opcode::Shl => imm(2).map(|c| i64::from(c as u32)),
                Opcode::And => imm(2)
                    .or_else(|| imm(1))
                    .map(|m| i64::from(m.trailing_zeros())),
                _ => None,
            };
            k.map_or(0, |k| k.min(62) as u32)
        };
        (self.data.ssa.sources(v).iter())
            .map(zeros)
            .min()
            .unwrap_or(0)
    }

    /// `a` read at `pos`: the counter of a loop `pos` is outside, whose
    /// one latch exit comes after a constant number of trips, is that
    /// number.
    fn after_exits(&self, a: AffineExpr, pos: usize) -> AffineExpr {
        a.as_symbols(&|v| match v {
            Var::Iteration(l) if !self.data.in_loop(*l, pos) => {
                self.constant_trips(*l).map(SymExpr::Const)
            }
            _ => None,
        })
    }

    /// A φ: a counter's `start + (k − 1)·step`; at a join, the one value
    /// every path writes; else one immediate every path writes.
    fn transfer_phi(
        &self,
        v: Value,
        block: BlockId,
        reg: Symbol,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<AffineExpr, UnknownReason> {
        let name = self.data.module.interner.resolve(reg);
        let header = (self.data.forest.loops.iter()).any(|l| l.header == block);
        let refused = || {
            self.obstacle_of_any_def(reg, id)
                .unwrap_or_else(|| match header {
                    true => format!(
                        "latch condition depends on {name}, carried around an enclosing loop"
                    )
                    .into(),
                    false => format!("{name} has more than one reaching definition").into(),
                })
        };
        if let Some((l, r)) = self.induction_variables.get(&v) {
            return (self.counter(*l, r, id, depth)).map_err(|why| why.unwrap_or_else(refused));
        }
        let mut defs = Vec::new();
        for s in self.data.ssa.sources(v) {
            let why = match s {
                Value::Result { op, .. } => match &self.data.kernel.stmts[op] {
                    Stmt::Instr(i) if i.predicate.is_some() => {
                        format!("{name} is written under a guard on some path")
                    }
                    _ => {
                        defs.push(op);
                        continue;
                    }
                },
                _ => format!("{name} is unwritten on some path"),
            };
            return Err(self
                .obstacle_of_any_def(reg, id)
                .unwrap_or_else(|| why.into()));
        }
        let common = match header {
            true => (self.same_immediate(&defs)).map(|c| AffineExpr::from_base(SymExpr::Const(c))),
            false => self.same_value(&defs, reg, id, depth),
        };
        common.ok_or_else(refused)
    }

    /// Counter `r` of loop `l` in its iteration `k`: its start plus
    /// `(k − 1)` steps, the steps free of every variable; `None` when a
    /// step varies, so `r` counts nothing.
    fn counter(
        &self,
        l: LoopId,
        r: &InductionVariable,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<AffineExpr, Option<UnknownReason>> {
        let module = self.data.module;
        let start = self.properties(r.start, id, depth + 1).expr()?;
        let step = self.step(l, &r.step, depth)?;
        // The values the back edges feed, start + k·step for k in [1, n]
        // when the loop runs a constant n trips, stay in their type.
        if self.constant_trips(l).is_some() {
            let fed = AffineExpr::term(Var::Iteration(l), step.clone()) + start.clone();
            let lp = self.data.forest.get(l);
            for (_, v) in (self.data.ssa.incoming(r.phi).into_iter())
                .filter(|(p, _)| p.is_some_and(|p| lp.blocks.binary_search(&p).is_ok()))
            {
                if let Value::Result { op, .. } = v
                    && let Stmt::Instr(instr) = &self.data.kernel.stmts[op]
                {
                    let mods: Vec<&str> = (module.modifiers(instr).iter())
                        .map(|&m| module.interner.resolve(m))
                        .collect();
                    self.fits(fed.clone(), instr, &mods).map_err(Some)?;
                }
            }
        }
        if step == SymExpr::Const(0) {
            return Ok(start);
        }
        let back = SymExpr::mul(SymExpr::Const(-1), step.clone());
        Ok(AffineExpr::term(Var::Iteration(l), step) + start + AffineExpr::from_base(back))
    }

    /// A counter's step in loop `l`: its constant, its registers, each
    /// free of every variable, and each nested loop's step times that
    /// loop's constant trip count of at least 1; `None` otherwise.
    fn step(&self, l: LoopId, step: &Step, depth: u32) -> Result<SymExpr, Option<UnknownReason>> {
        let mut sum = SymExpr::Const(step.constant);
        for &(reg, at, sign) in &step.registers {
            let s = self.first_reason(|| self.at_use(reg, at, Some(l), depth + 1));
            let s = s.map_err(|_| None)?;
            if !s.is_symbolic_or_constant() {
                return Err(None);
            }
            sum = SymExpr::add(sum, SymExpr::mul(SymExpr::Const(sign), s.base));
        }
        for (c, inner) in &step.exits {
            let n = self.constant_trips(*c).ok_or(None)?;
            let s = self.step(*c, inner, depth)?;
            sum = SymExpr::add(sum, SymExpr::mul(SymExpr::Const(n), s));
        }
        Ok(sum)
    }

    /// Which reads every lane executing them sees alike.
    pub(crate) fn uniformity(&self) -> &UniformityInfo<'a> {
        self.uniformity
    }

    /// Loop `id`'s trip count. Counting a loop reads its counter, which
    /// holds only while the values its back edges feed stay in their
    /// type over the trips, and checking that needs the count: a count
    /// that read itself so assumed they do. It is counted again with
    /// itself as a candidate, which readers see, and kept only where both
    /// counts agree. For `for (int k = 0; k < K; k += 8)` under
    /// K = 2^31 − 1 the first count is ⌈K/8⌉ = 2^28; counted again, the
    /// last trip's `add.s32` yields 2^28 · 8 = 2^31, past `int`, where
    /// the hardware wraps to −2^31 and the loop goes on, so the count is
    /// refused (`a_signed_counter_that_wraps_before_its_count_has_no_count`).
    pub(crate) fn loop_trip_count(&self, id: LoopId) -> Result<LoopTripCount, UnknownReason> {
        let unfinished = || Err("the trip count is being counted".into());
        self.memoized(&self.trips, id, unfinished, |frame| {
            let first = self.counted_trips(id);
            if self.provisional.get() != frame || first.is_err() {
                return first;
            }
            (self.trips.borrow_mut()).insert(id, Slot::Candidate(frame, first.clone()));
            match self.counted_trips(id) {
                again if again == first => again,
                Err(why) => Err(why),
                Ok(_) => Err("the trip count changes once it is known".into()),
            }
        })
    }

    /// The address `offset` bytes past `base` at `pos` under the bindings,
    /// one form for every lane unless it counts the iterations of a loop
    /// that does not hold `pos` and that lanes leave at different
    /// iterations, or holds a uniform value the lanes executing `pos` do
    /// not see alike: then each lane reads its own, and the reason prints
    /// that variable by `name`.
    pub(crate) fn address(
        &self,
        base: OperandId,
        offset: i64,
        pos: usize,
        id: Option<LoopId>,
        name: &dyn Fn(&Var) -> String,
    ) -> Result<AffineExpr, UnknownReason> {
        let a = self.first_reason(|| self.operand_at(base, pos, id, 0))?
            + AffineExpr::from_base(SymExpr::Const(offset));
        let a = self.bindings.map_or(a.clone(), |b| a.bind(b));
        for v in a.terms.keys() {
            if let Some(l) = v.counter().filter(|&l| !self.data.in_loop(l, pos))
                && let Err(why) = self.uniformity().exits_uniform(l)
            {
                let k = name(&Var::Iteration(l));
                return Err(
                    format!("{k} differs between lanes after its loop, whose exit {why}").into(),
                );
            }
            if let Some(u) = v.uniform()
                && let Err(why) = self.uniformity().value_uniform_at(u, pos)
            {
                let u = name(&Var::Unknown(u));
                return Err(format!("{u} differs between lanes: {why}").into());
            }
        }
        Ok(a)
    }

    /// `f`, failing for the first reason it read a uniform value as a
    /// variable, if it did.
    pub(crate) fn first_reason<T>(
        &self,
        f: impl FnOnce() -> Result<T, UnknownReason>,
    ) -> Result<T, UnknownReason> {
        let outer = self.unread.take();
        let t = f();
        let first = self.unread.replace(outer);
        t.map_err(|why| first.unwrap_or(why))
    }

    /// Loop `l`'s trip count when it is a constant of at least 1 under the
    /// bindings, which makes its latch, and so its exit, the same for
    /// every lane.
    fn constant_trips(&self, l: LoopId) -> Option<i64> {
        (self.loop_trip_count(l).ok().map(|t| t.count))
            .and_then(|t| self.constant(&AffineExpr::from_base(t)))
            .filter(|&n| n >= 1)
    }

    /// `f`'s result for `key`, from `memo` or cached there unless it read
    /// a frame still in progress outside its own; `unfinished` where `key`
    /// is itself in progress.
    fn memoized<K: Copy + Eq + std::hash::Hash, T: Clone>(
        &self,
        memo: &Memo<K, T>,
        key: K,
        unfinished: impl FnOnce() -> T,
        f: impl FnOnce(usize) -> T,
    ) -> T {
        match memo.borrow().get(&key) {
            Some(Slot::Done(t)) => return t.clone(),
            Some(&Slot::InProgress(frame)) => {
                self.provisional.set(self.provisional.get().min(frame));
                return unfinished();
            }
            Some(Slot::Candidate(frame, t)) => {
                self.provisional.set(self.provisional.get().min(*frame));
                return t.clone();
            }
            None => {}
        }
        let frame = self.frames.get() + 1;
        self.frames.set(frame);
        memo.borrow_mut().insert(key, Slot::InProgress(frame));
        let outer = self.provisional.replace(usize::MAX);
        let t = f(frame);
        let read = self.provisional.get();
        self.frames.set(frame - 1);
        if read >= frame {
            memo.borrow_mut().insert(key, Slot::Done(t.clone()));
        } else {
            memo.borrow_mut().remove(&key);
        }
        self.provisional
            .set(outer.min(if read < frame { read } else { usize::MAX }));
        t
    }

    /// What value `v` is at its definition, read for the latch of loop `id`.
    fn properties(&self, v: Value, id: Option<LoopId>, depth: u32) -> AffineProperties {
        let unknown = |reason| AffineProperties::Unknown {
            reason,
            min_trailing_zeros: self.min_trailing_zeros(v),
        };
        let unfinished = || (unknown("value depends on itself".into()), None);
        let (p, why) = self.memoized(&self.values, (v, id), unfinished, |_| {
            let outer = self.unread.take();
            let p = self
                .trace_uncached(v, id, depth)
                .map_or_else(unknown, AffineProperties::Expr);
            (p, self.unread.replace(outer))
        });
        if let Some(why) = why {
            self.unread.borrow_mut().get_or_insert(why);
        }
        p
    }

    fn trace_uncached(
        &self,
        v: Value,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<AffineExpr, UnknownReason> {
        let module = self.data.module;
        if depth > DEPTH_LIMIT {
            self.provisional.set(0);
            return Err("value trace exceeds depth limit".into());
        }
        let (def, reg) = match v {
            Value::Result {
                op,
                reg: Register::Named(reg),
            } => (op, reg),
            Value::Argument {
                block,
                reg: Register::Named(reg),
            } => return self.transfer_phi(v, block, reg, id, depth),
            Value::Undef(Register::Named(reg)) => {
                let name = module.interner.resolve(reg);
                return Err(format!("no definition found for {name}").into());
            }
            _ => return Err("value is the carry flag".into()),
        };
        let value = self.transfer(def, reg, id, depth)?;
        if let Stmt::Instr(instr) = &self.data.kernel.stmts[def]
            && instr.predicate.is_some()
        {
            let name = module.interner.resolve(reg);
            return Err(format!("{name} is written under a guard").into());
        }
        Ok(value)
    }

    /// A variable's least and greatest value when the launch or a
    /// constant trip count fixes it: `%tid` below the bound block shape,
    /// `%ctaid` below a bound grid, a counter in [1, n].
    pub(crate) fn var_bounds(&self, v: &Var) -> Option<(i64, i64)> {
        let shape = |vector: SpecialVector, axis: &Axis| {
            let i = match axis {
                Axis::X => 0,
                Axis::Y => 1,
                Axis::Z => 2,
            };
            let n = *self
                .bindings?
                .get(&SymbolicConstant::SpecialReg(vector.components()[i]))?;
            (n >= 1).then_some((0, n - 1))
        };
        match v {
            Var::Tid(axis) => shape(SpecialVector::Ntid, axis),
            Var::Ctaid(axis) => shape(SpecialVector::Nctaid, axis),
            Var::Iteration(l) => self.constant_trips(*l).map(|n| (1, n)),
            Var::FloorDiv(..) | Var::Mod(..) | Var::Unknown(_) | Var::Swizzle(_) => None,
        }
    }

    /// Why a value cannot be traced, looking through arithmetic it
    /// does not read: a special register, a memory load or an
    /// atomic on some operand chain is the reason a reader can act on,
    /// not the `or` or `bfe` in between.
    pub(crate) fn obstacle(
        &self,
        reg: Symbol,
        pos: usize,
        id: Option<LoopId>,
        seen: &mut HashSet<Symbol>,
    ) -> Option<UnknownReason> {
        let module = self.data.module;
        if !seen.insert(reg) || seen.len() > 64 {
            return None;
        }
        let def = match self.data.ssa.value_before(pos, Register::Named(reg)) {
            Some(Value::Result { op, .. }) => op,
            Some(Value::Argument { .. }) => return self.defs_obstacle(reg, id, seen),
            Some(Value::Undef(_)) | None => return None,
        };
        let Stmt::Instr(instr) = &self.data.kernel.stmts[def] else {
            return None;
        };
        let is_param_load = self
            .data
            .module
            .modifiers(instr)
            .iter()
            .any(|&m| module.interner.resolve(m) == "param");
        match instr.mnemonic {
            Opcode::Ld if !is_param_load => Some(
                if id.is_some_and(|l| self.data.in_loop(l, def)) {
                    "latch condition depends on a value loaded inside the loop"
                } else {
                    "latch condition depends on a value loaded from memory"
                }
                .into(),
            ),
            Opcode::Atom => Some("latch condition depends on an atomic operation".into()),
            _ => (split_operands(module, instr).1.iter())
                .find_map(|&op| self.operand_obstacle(op, def, id, seen)),
        }
    }

    fn operand_obstacle(
        &self,
        op: OperandId,
        pos: usize,
        id: Option<LoopId>,
        seen: &mut HashSet<Symbol>,
    ) -> Option<UnknownReason> {
        match self.data.module.operand(op) {
            Operand::Register(r) => self.obstacle(*r, pos, id, seen),
            Operand::Special(reg) => special_obstacle(*reg),
            _ => None,
        }
    }
}

fn special_obstacle(reg: SpecialReg) -> Option<UnknownReason> {
    (special_register(reg) != Some(SpecialRegister::Zero))
        .then(|| format!("latch condition depends on special register {}", reg.name()).into())
}

/// The count, the block it is per execution of, and a `!=` latch's
/// distance and step.
#[derive(Clone, PartialEq)]
pub(crate) struct LoopTripCount {
    pub(crate) count: SymExpr,
    pub(crate) entry: BlockId,
    pub(crate) distance: Option<(SymExpr, i64)>,
}

/// `a` with its terms in the CTA index as symbols of its base: `%ctaid`
/// is the CTA's identifier, one value per CTA, never negative (PTX ISA
/// §10.6).
fn per_cta(a: AffineExpr) -> AffineExpr {
    a.as_symbols(&|v| {
        let Var::Ctaid(axis) = v else { return None };
        let reg = SpecialVector::Ctaid.components()[*axis as usize];
        Some(SymExpr::Sym(SymbolicConstant::SpecialReg(reg)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::affine::trip_counts::{TripCount, trip_counts};
    use crate::analysis::control_flow::loop_forest;
    use crate::analysis::loop_names::loop_names;
    use crate::ptx::cfg::build_cfg;
    use crate::ptx::ir::Module;
    use crate::ptx::parse::parser::parse;

    /// Trips of every loop in a one-kernel source, by display name.
    fn trips_of(src: &str) -> Vec<(String, TripCount)> {
        trips_under(src, &HashMap::new())
    }

    fn trips_under(
        src: &str,
        bindings: &HashMap<SymbolicConstant, i64>,
    ) -> Vec<(String, TripCount)> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names, bindings);
        names
            .into_iter()
            .zip(info.trips)
            .map(|(n, t)| (n.display, t.map(|t| by_register(&m, t))))
            .collect()
    }

    /// `e` with each uniform value a symbol named by its register.
    fn by_register(m: &Module, e: SymExpr) -> SymExpr {
        let all = |es: Vec<SymExpr>, f: fn(SymExpr, SymExpr) -> SymExpr| {
            (es.into_iter().map(|x| by_register(m, x)))
                .reduce(f)
                .expect("a sum or product has terms")
        };
        match e {
            SymExpr::Sym(SymbolicConstant::Unknown(v)) => {
                SymExpr::sym(format!("uniform({})", register(m)(v)))
            }
            SymExpr::Sum(ts) => all(ts, SymExpr::add),
            SymExpr::Prod(fs) => all(fs, SymExpr::mul),
            SymExpr::CeilDiv(x, c) => SymExpr::ceil_div(by_register(m, *x), c),
            SymExpr::FloorDiv(x, c) => SymExpr::floor_div(by_register(m, *x), c),
            SymExpr::Mod(x, c) => SymExpr::modulo(by_register(m, *x), c),
            SymExpr::Min(a, b) => SymExpr::min(by_register(m, *a), by_register(m, *b)),
            SymExpr::Max(a, b) => SymExpr::max(by_register(m, *a), by_register(m, *b)),
            other => other,
        }
    }

    /// A value by its register's name.
    fn register(m: &Module) -> impl Fn(Value) -> String + '_ {
        |v| match v {
            Value::Result { reg, .. } | Value::Argument { reg, .. } | Value::Undef(reg) => {
                match reg {
                    Register::Named(r) => m.interner.resolve(r).to_owned(),
                    Register::Cc => "CC.CF".to_owned(),
                }
            }
        }
    }

    /// Every loop's name, count and the block the count is per execution of.
    fn entries_of(src: &str) -> Vec<(String, Result<String, String>, String)> {
        let m = parse(src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names, &HashMap::new());
        (names.into_iter().zip(info.trips).zip(info.entries))
            .map(|((n, t), e)| {
                let t = t.map(|t| by_register(&m, t).to_string());
                (n.display, t, cfg.block_name(&m, e))
            })
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
    fn a_counter_that_wraps_before_its_loop_exits_has_no_count_in_either_order() {
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, 4294967294;\n$L__L:\nadd.u32 %r1, %r1, 1;\n\
             setp.le.u32 %p1, %r1, 4294967295;\n@%p1 bra $L__L;\nret;",
        );
        let m = parse(&src).expect("test source parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let data = KernelAnalyses::new(&m, k, &cfg, &f);
        let uniformity = UniformityInfo::new(&data);
        let (pos, setp) = (k.stmts.iter().enumerate())
            .find_map(|(i, s)| match s {
                Stmt::Instr(instr) if instr.mnemonic == Opcode::Setp => Some((i, instr)),
                _ => None,
            })
            .expect("a setp");
        let counted = |read_first: bool| {
            let affine = AffineAnalysis::new(&data, &uniformity, None);
            if read_first {
                let _ = affine.operand_at(m.operand_ids(setp.operands)[1], pos, None, 0);
            }
            affine.loop_trip_count(LoopId(0)).err().map(|r| r.render())
        };
        let wraps = Some("`add.u32` leaves 32 bits".to_owned());
        assert_eq!(counted(false), wraps);
        assert_eq!(counted(true), wraps);
    }

    #[test]
    fn a_signed_counter_that_wraps_before_its_count_has_no_count() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 8;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let k = |n: i64| HashMap::from([(SymbolicConstant::KernelParam(0), n)]);
        let trips = |n| trips_under(&src, &k(n))[0].1.clone().map(|t| t.to_string());
        assert_eq!(trips(4096), Ok("max(⌈param_0/8⌉, 1)".to_owned()));
        let wraps = Err("`add.s32` leaves 32 bits".to_owned());
        assert_eq!(trips(i64::from(i32::MAX)), wraps);
    }

    #[test]
    fn a_count_its_counter_refuses_once_known_is_unknown_to_every_reader() {
        let src = include_str!("../../../tests/fixtures/k5/k5.sm_80.ptx");
        let m = parse(src).expect("k5 parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let data = KernelAnalyses::new(&m, k, &cfg, &f);
        let k_max = HashMap::from([(SymbolicConstant::KernelParam(2), i64::from(i32::MAX))]);
        let uniformity = UniformityInfo::new(&data);
        let affine = AffineAnalysis::new(&data, &uniformity, Some(&k_max));
        let mut loops = (0..f.loops.len() as u32).map(LoopId);
        let outer = loops
            .find(|&l| f.get(l).parent.is_none())
            .expect("an outer loop");
        assert_eq!(affine.constant_trips(outer), None);
        let count = affine.loop_trip_count(outer).err().map(|r| r.render());
        assert_eq!(count.as_deref(), Some("`add.s32` leaves 32 bits"));
    }

    #[test]
    fn up_counting_lt_stride_1() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "max(param_0, 1)");
    }

    #[test]
    fn a_not_equal_bound_the_counter_never_meets_is_no_trip_count() {
        let latch = |start: i64, step: i64, bound: i64| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "mov.u32 %r2, {start};\n$L__L:\nadd.s32 %r2, %r2, {step};\n\
                     setp.ne.s32 %p1, %r2, {bound};\n@%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        assert_eq!(latch(0, 2, 6), Ok("3".to_owned()));
        let never = Err("latch condition never meets its `!=` bound".to_owned());
        assert_eq!(latch(0, 2, 5), never);
        assert_eq!(latch(6, 2, 4), never);
        assert_eq!(latch(4, -2, 0), Ok("2".to_owned()));
        // A bound the bindings make a constant is checked at its value.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmax.s32 %r3, %r1, 1;\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 2;\nsetp.ne.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let at = |n: i64| {
            let bound = [(SymbolicConstant::KernelParam(0), n)]
                .into_iter()
                .collect();
            trips_under(&src, &bound)[0]
                .1
                .clone()
                .map(|t| t.bind(&bound).to_string())
        };
        assert_eq!(at(6), Ok("3".to_owned()));
        assert_eq!(at(5), never);
    }

    /// The two-trip shape with each of its writes guarded: the lanes the
    /// guard skips keep another value.
    #[test]
    fn a_two_trip_loop_with_a_guarded_write_is_refused() {
        let trips = |init: &str, copy: &str, latch: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nsetp.lt.u32 %p1, %r1, 4;\n\
                     setp.ne.u32 %p3, %r1, %r1;\n{init}mov.pred %p3, -1;\n\
                     $L__L:\n{copy}mov.pred %p2, %p3;\n{latch}mov.pred %p3, 0;\n\
                     @%p2 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        let (g, guarded) = ("@%p1 ", |p: &str| {
            Err(format!("{p} is written under a guard"))
        });
        assert_eq!(trips("", "", ""), Ok("2".to_owned()));
        assert_eq!(trips(g, "", ""), guarded("%p3"));
        assert_eq!(trips("", g, ""), guarded("%p2"));
        assert_eq!(trips("", "", g), guarded("%p3"));
    }

    /// A bound of 7 that two false-guarded writes of 0 leave alone: the
    /// header φ's sources are both guarded, so it is no shared 0.
    #[test]
    fn a_bound_merged_from_guarded_writes_is_refused() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nsetp.ne.u32 %p2, %r1, %r1;\n\
             mov.u32 %r3, 7;\n@%p2 mov.u32 %r3, 0;\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p1, %r2, %r3;\n\
             @%p2 mov.u32 %r3, 0;\n@%p1 bra $L__L;\nret;",
        );
        let why = "%r3 is written under a guard on some path";
        assert_eq!(trips_of(&src)[0].1.as_ref().unwrap_err(), why);
    }

    /// A 32-bit bound computed past its width: a shift by more than 32
    /// leaves 0 (PTX ISA §9.7.8.8), a product past 2^32 is not read and
    /// stays the value it computes, the same for every thread.
    #[test]
    fn a_bound_past_its_register_width_wraps_or_is_a_uniform_value() {
        let bound = |compute: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "mov.u32 %r3, 4;\n{compute}\nadd.s32 %r5, %r4, 5;\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p1, %r2, %r5;\n\
                     @%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        assert_eq!(bound("shl.b32 %r4, %r3, 40;"), Ok("5".to_owned()));
        assert_eq!(bound("mul.lo.u32 %r4, %r3, 2;"), Ok("13".to_owned()));
        assert_eq!(
            bound("mul.lo.u32 %r4, %r3, 1073741824;"),
            Ok("uniform(%r4) + 5".to_owned())
        );
    }

    /// A bound two paths set to 7, one after a 3 it overwrites: the 3
    /// does not reach the loop.
    #[test]
    fn a_bound_every_path_last_sets_alike_is_that_value() {
        let bound = |other_arm: &str| {
            kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nsetp.lt.u32 %p2, %r1, 4;\n\
                     @%p2 bra $L__A;\n{other_arm}bra $L__J;\n\
                     $L__A:\nmov.u32 %r3, 7;\n$L__J:\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;"
                ),
            )
        };
        let trips = trips_of(&bound("mov.u32 %r3, 3;\nmov.u32 %r3, 7;\n"));
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "7");
        let trips = trips_of(&bound(""));
        assert_eq!(
            trips[0].1.as_ref().unwrap_err(),
            "%r3 is unwritten on some path"
        );
    }

    /// A bound two paths compute alike, as a shift and as a product, is
    /// that value; two values apart, or one value through an enclosing
    /// loop's counter, are not read.
    #[test]
    fn a_bound_every_path_computes_alike_is_that_value() {
        let bound = |other_arm: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nsetp.lt.u32 %p2, %r1, 4;\n\
                     @%p2 bra $L__A;\n{other_arm}\nbra.uni $L__J;\n\
                     $L__A:\nshl.b32 %r3, %r1, 2;\n$L__J:\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        let ok = |s: &str| Ok(s.to_owned());
        assert_eq!(bound("mul.lo.s32 %r3, %r1, 4;"), ok("max(4 * param_0, 1)"));
        let aligned = ok("max(4 * (uniform(%r3) / 4), 1)");
        assert_eq!(bound("shl.b32 %r3, %r1, 3;"), aligned);
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nsetp.lt.u32 %p2, %r1, 4;\nmov.u32 %r5, 0;\n\
             $L__O:\nadd.s32 %r5, %r5, 1;\n@%p2 bra $L__A;\nadd.s32 %r3, %r5, 2;\n\
             bra.uni $L__J;\n$L__A:\nadd.s32 %r3, %r5, 2;\n$L__J:\nmov.u32 %r2, 0;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\n\
             setp.lt.s32 %p3, %r5, 4;\n@%p3 bra $L__O;\nret;",
        );
        let trips = trips_of(&src);
        let inner = trips
            .iter()
            .find(|(n, _)| n == "$L__L")
            .expect("inner loop");
        let why = "%r3 has more than one reaching definition";
        assert_eq!(inner.1.as_ref().unwrap_err(), why);
    }

    /// PTX ISA §9.7.1.13, §9.7.1.14: a min or max of symbolic or constant
    /// values is one too; a `.relu` one, or one of a value its type does
    /// not read, is not read and stays a uniform value.
    #[test]
    fn a_min_or_max_bound_is_symbolic() {
        let bound = |minmax: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\n{minmax}\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        let ok = |s: &str| Ok(s.to_owned());
        assert_eq!(bound("max.s32 %r3, %r1, 2;"), ok("max(param_0, 2)"));
        assert_eq!(bound("min.s32 %r3, 2, %r1;"), ok("max(min(2, param_0), 1)"));
        assert_eq!(bound("min.u32 %r3, %r1, 8;"), ok("max(min(param_0, 8), 1)"));
        assert_eq!(bound("mov.u32 %r4, 5;\nmin.s32 %r3, %r4, 8;"), ok("5"));
        assert_eq!(
            bound("mov.u32 %r4, -1;\nmin.u32 %r3, %r4, 8;"),
            ok("max(uniform(%r3), 1)")
        );
        assert_eq!(
            bound("min.relu.s32 %r3, %r1, 8;"),
            ok("max(uniform(%r3), 1)")
        );
    }

    /// Under the bindings a min or max is checked and stepped by its
    /// value: `bfe.s32` of 8 bits of max(n, 1) at n = 128 is −128 (PTX ISA
    /// §9.7.1.20), and a step of max(n, 1) at n = 4 is 4.
    #[test]
    fn a_bound_min_or_max_is_read_at_its_value() {
        let trips = |body: &str, n: Option<i64>| {
            let bound: HashMap<SymbolicConstant, i64> = n
                .map(|n| (SymbolicConstant::KernelParam(0), n))
                .into_iter()
                .collect();
            let t = trips_under(&kernel_with(N_PARAM, body), &bound)[0]
                .1
                .clone();
            t.map(|t| t.bind(&bound).to_string())
        };
        let bfe = "ld.param.u32 %r1, [k_param_0];\nmax.u32 %r2, %r1, 1;\n\
                   bfe.s32 %r3, %r2, 0, 8;\nmov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\n\
                   setp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;";
        assert_eq!(trips(bfe, None), Ok("max(param_0, 1)".to_owned()));
        assert_eq!(trips(bfe, Some(100)), Ok("100".to_owned()));
        let leaves = Ok("max(uniform(%r3), 1)".to_owned());
        assert_eq!(trips(bfe, Some(128)), leaves);
        let step = "ld.param.u32 %r1, [k_param_0];\nmax.s32 %r2, %r1, 1;\nmov.u32 %r4, 0;\n\
                    $L__L:\nadd.s32 %r4, %r4, %r2;\nsetp.lt.s32 %p1, %r4, 1024;\n\
                    @%p1 bra $L__L;\nret;";
        let varies = Err("induction step is not a constant".to_owned());
        assert_eq!(trips(step, None), varies);
        assert_eq!(trips(step, Some(4)), Ok("256".to_owned()));
        // An outer counter stepped once per trip of an inner loop of
        // max(n, 2) trips: 4 a trip at n = 4, so 100 / 4 trips.
        let nested = "ld.param.u32 %r1, [k_param_0];\nmax.s32 %r9, %r1, 2;\nmov.u32 %r2, 0;\n\
                      $L__O:\nmov.u32 %r3, 0;\n$L__I:\nadd.s32 %r3, %r3, 1;\nadd.s32 %r2, %r2, 1;\n\
                      setp.lt.s32 %p1, %r3, %r9;\n@%p1 bra $L__I;\n\
                      setp.lt.s32 %p2, %r2, 100;\n@%p2 bra $L__O;\nret;";
        let bound = [(SymbolicConstant::KernelParam(0), 4)]
            .into_iter()
            .collect();
        let outer = trips_under(&kernel_with(N_PARAM, nested), &bound);
        let outer = outer
            .iter()
            .find(|(n, _)| n == "$L__O")
            .expect("outer loop");
        assert_eq!(outer.1, Ok(SymExpr::Const(25)));
    }

    /// LLVM's loop guard: a branch before the loop that skips it when its
    /// latch condition fails at k = 0 makes the count max(T, 0), however
    /// it writes the test; a guard on another condition, or none, leaves
    /// the count per entry, max(T, 1).
    #[test]
    fn a_guard_testing_the_latch_at_zero_clamps_the_count_at_zero() {
        let trips = |guard: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\n{guard}\nmov.u32 %r2, 10;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        let clamped = Ok("max(param_0 - 10, 0)".to_owned());
        assert_eq!(trips("setp.lt.s32 %p1, %r1, 11;\n@%p1 bra $L__E;"), clamped);
        assert_eq!(trips("setp.le.s32 %p1, %r1, 10;\n@%p1 bra $L__E;"), clamped);
        assert_eq!(trips("setp.ge.s32 %p1, 10, %r1;\n@%p1 bra $L__E;"), clamped);
        assert_eq!(
            trips("setp.gt.s32 %p1, %r1, 10;\n@!%p1 bra $L__E;"),
            clamped
        );
        let unclamped = Ok("max(param_0 - 10, 1)".to_owned());
        assert_eq!(
            trips("setp.lt.s32 %p1, %r1, 6;\n@%p1 bra $L__E;"),
            unclamped
        );
        assert_eq!(
            trips("setp.lt.s32 %p1, %r1, 11;\n@!%p1 bra $L__E;"),
            unclamped
        );
        assert_eq!(
            trips("setp.lt.f32 %p1, %r1, 11;\n@%p1 bra $L__E;"),
            unclamped
        );
        assert_eq!(trips(""), unclamped);
    }

    /// The clamped count is per execution of the guard, so the guard is
    /// the loop's entry; a branch inside another loop runs per trip of
    /// that loop and is not taken as the guard.
    #[test]
    fn a_guard_in_the_parent_is_the_loops_entry() {
        let entries = |pre: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r3, 0;\n{pre}\n\
                     setp.lt.s32 %p1, %r1, 11;\n@%p1 bra $L__S;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            entries_of(&src)
        };
        let guard = entries("mov.u32 %r2, 10;\nbra.uni $L__G;\n$L__S:\nbra.uni $L__E;\n$L__G:");
        assert_eq!(guard[0].1, Ok("max(param_0 - 10, 0)".to_owned()));
        assert_eq!(guard[0].2, "$L__G");
        let sibling = entries(
            "$L__S:\nadd.s32 %r3, %r3, 1;\nsetp.eq.s32 %p3, %r3, 7;\n@%p3 bra $L__E;\n\
             mov.u32 %r2, 10;",
        );
        let l = sibling.iter().find(|(n, ..)| n == "$L__L").expect("loop L");
        assert_eq!(l.1, Ok("max(param_0 - 10, 1)".to_owned()));
        assert_eq!(l.2, "$L__L");
    }

    /// The guard's predicate may be set in an earlier block, but not on
    /// only some of the paths to the guard.
    #[test]
    fn a_guard_reads_the_setp_that_reaches_its_predicate() {
        let entry = |other_path: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nsetp.lt.s32 %p1, %r1, 11;\n\
                     setp.eq.s32 %p3, %r1, 3;\n@%p3 bra $L__G;\n{other_path}\n\
                     $L__G:\n@%p1 bra $L__E;\nmov.u32 %r2, 10;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            let loops = entries_of(&src);
            (loops[0].1.clone().unwrap_or_default(), loops[0].2.clone())
        };
        let at = |t: &str, b: &str| (t.to_owned(), b.to_owned());
        assert_eq!(
            entry("mov.u32 %r5, 1;"),
            at("max(param_0 - 10, 0)", "$L__G")
        );
        assert_eq!(
            entry("setp.lt.s32 %p1, %r1, 5;"),
            at("max(param_0 - 10, 1)", "$L__L")
        );
    }

    /// A latch whose test at k = 0 is a sum of constants past i64, left
    /// unfolded (`i = −(2^63 − 1); do i += 2^63 − 1 while i <= 1`), is
    /// compared with a guard without overflow.
    #[test]
    fn a_latch_test_past_i64_at_zero_is_compared_unfolded() {
        let src = kernel_with(
            N_PARAM,
            "mov.u64 %rd1, -9223372036854775807;\n\
             $L__L:\nadd.s64 %rd1, %rd1, 9223372036854775807;\n\
             setp.le.s64 %p1, %rd1, 1;\n@%p1 bra $L__L;\nret;",
        );
        assert!(trips_of(&src)[0].1.is_ok());
    }

    /// The guard may lie several blocks above the loop, each of one
    /// predecessor and one successor; a join between them hides it.
    #[test]
    fn a_guard_above_a_chain_of_blocks_is_the_loops_entry() {
        let entry = |join: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\n{join}\n$L__G:\n\
                     setp.lt.s32 %p1, %r1, 11;\n@%p1 bra $L__E;\n\
                     mov.u32 %r2, 10;\nbra.uni $L__B;\n$L__B:\nmov.u32 %r5, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            let loops = entries_of(&src);
            assert_eq!(loops[0].0, "$L__L");
            (loops[0].1.clone().unwrap_or_default(), loops[0].2.clone())
        };
        let at = |t: &str, b: &str| (t.to_owned(), b.to_owned());
        assert_eq!(entry(""), at("max(param_0 - 10, 0)", "<block 0>"));
        let join = "mov.u32 %r2, 10;\nsetp.eq.s32 %p3, %r1, 3;\n@%p3 bra $L__B;";
        assert_eq!(entry(join), at("max(param_0 - 10, 1)", "$L__L"));
    }

    /// A predicated `exit` between the guard and the loop, or in a loop
    /// crossed on the way, lets threads the guard passes skip the loop:
    /// the guard is not its entry.
    #[test]
    fn a_predicated_exit_after_the_guard_hides_it() {
        let entry = |between: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nsetp.lt.s32 %p1, %r1, 11;\n\
                     @%p1 bra $L__E;\n{between}\nmov.u32 %r2, 10;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            let loops = entries_of(&src);
            let l = loops.iter().find(|(n, ..)| n == "$L__L").expect("loop L");
            (l.1.clone().unwrap_or_default(), l.2.clone())
        };
        let at = |t: &str, b: &str| (t.to_owned(), b.to_owned());
        let (entered, unguarded) = (
            at("max(param_0 - 10, 0)", "<block 0>"),
            at("max(param_0 - 10, 1)", "$L__L"),
        );
        let exit = "setp.eq.s32 %p3, %r1, 20;\n@%p3 exit;";
        let sibling = |body: &str| {
            format!(
                "mov.u32 %r3, 0;\n$L__S:\nadd.s32 %r3, %r3, 1;\n{body}\n\
                 setp.lt.s32 %p4, %r3, 4;\n@%p4 bra $L__S;"
            )
        };
        assert_eq!(entry(exit), unguarded);
        assert_eq!(entry(&sibling("")), entered);
        assert_eq!(entry(&sibling(exit)), unguarded);
    }

    /// A latch's `min` or `max` of the CTA index is one value per CTA, a
    /// row range's `min(16·%ctaid.y + 16, n) - 16·%ctaid.y`; ranged over
    /// the grid, a `max.u32(%ctaid.x, 1)` shifted by 31, or times
    /// `%ctaid.x` and shifted, leaves 32 bits with 3 CTAs and is not read.
    #[test]
    fn a_latch_min_or_max_of_the_cta_index_is_one_per_cta() {
        let trips = |bound: &str, grid: Option<i64>| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, %ctaid.x;\nmov.u32 %r6, %ctaid.y;\n\
                     {bound}\nmov.u32 %r8, 0;\n$L__L:\nadd.s32 %r8, %r8, 1;\n\
                     setp.lt.u32 %p1, %r8, %r9;\n@%p1 bra $L__L;\nret;"
                ),
            );
            let grid = grid.into_iter().flat_map(|n| {
                let [x, y, _] = SpecialVector::Nctaid
                    .components()
                    .map(SymbolicConstant::SpecialReg);
                [(x, n), (y, 1)]
            });
            let bindings = grid.collect();
            trips_under(&src, &bindings)[0]
                .1
                .clone()
                .map(|t| t.to_string())
        };
        let rows = "shl.b32 %r3, %r6, 4;\nadd.s32 %r4, %r3, 16;\nmin.u32 %r5, %r4, %r1;\n\
                    sub.s32 %r9, %r5, %r3;";
        let read = "max(min(16 * %ctaid.y + 16, param_0) - 16 * %ctaid.y, 1)";
        assert_eq!(trips(rows, None), Ok(read.to_owned()));
        let shifted = "max.u32 %r3, %r2, 1;\nshl.b32 %r9, %r3, 31;";
        assert!(trips(shifted, None).is_ok());
        let whole = "max(2147483648 * (uniform(%r9) / 2147483648), 1)";
        assert_eq!(trips(shifted, Some(3)), Ok(whole.to_owned()));
        let times = "max.u32 %r3, %r2, 1;\nmul.lo.u32 %r4, %r3, %r2;\nshl.b32 %r9, %r4, 31;";
        assert!(trips(times, None).is_ok());
        assert_eq!(trips(times, Some(3)), Ok(whole.to_owned()));
    }

    /// A guard that sums its bound in the other order than the latch
    /// tests the same value.
    #[test]
    fn a_guard_summing_in_the_other_order_is_the_loops_guard() {
        let src = kernel_with(
            ".param .u32 k_param_0,\n.param .u32 k_param_1",
            "ld.param.u32 %r1, [k_param_0];\nld.param.u32 %r5, [k_param_1];\n\
             add.s32 %r6, %r1, %r5;\nadd.s32 %r7, %r5, %r1;\n\
             setp.lt.s32 %p1, %r6, 11;\n@%p1 bra $L__E;\nmov.u32 %r2, 10;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r7;\n\
             @%p2 bra $L__L;\n$L__E:\nret;",
        );
        let trips = trips_of(&src)[0].1.clone().map(|t| t.to_string());
        assert_eq!(trips, Ok("max(param_0 + param_1 - 10, 0)".to_owned()));
    }

    /// A guard skipping a loop from 0 below n where `n == 0` tests its
    /// latch at k = 0, `0 < n`, as n is never negative; one where `n != 0`
    /// or `n == 5` does not.
    #[test]
    fn a_guard_n_ne_0_tests_0_lt_n() {
        let entry = |guard: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\n{guard}\n$L__P:\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 8;\nsetp.lt.u32 %p2, %r2, %r1;\n\
                     @%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            let loops = entries_of(&src);
            (loops[0].1.clone().unwrap_or_default(), loops[0].2.clone())
        };
        let guard = |g: &str| format!("setp.{g};\n@%p1 bra $L__E;");
        let at = |t: &str, b: &str| (t.to_owned(), b.to_owned());
        let (entered, unguarded) = (
            at("⌈param_0/8⌉", "<block 0>"),
            at("max(⌈param_0/8⌉, 1)", "$L__L"),
        );
        assert_eq!(entry(&guard("eq.s32 %p1, %r1, 0")), entered);
        assert_eq!(entry("setp.ne.s32 %p1, 0, %r1;\n@!%p1 bra $L__E;"), entered);
        assert_eq!(entry(&guard("ne.s32 %p1, %r1, 0")), unguarded);
        assert_eq!(entry(&guard("eq.s32 %p1, %r1, 5")), unguarded);
    }

    /// A loop from 0 below ⌊n/16⌋ is entered where n ≥ 16, and one below
    /// ⌈n/8⌉ where n ≥ 1: guards on n alone test those latches at k = 0.
    #[test]
    fn a_guard_on_the_numerator_tests_a_floor_or_ceiling_bound() {
        let entry = |bound: &str, guard: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\n{bound}\n{guard}\n@%p1 bra $L__E;\n\
                     mov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.u32 %p2, %r2, %r3;\n@%p2 bra $L__L;\n$L__E:\nret;"
                ),
            );
            let loops = entries_of(&src);
            (loops[0].1.clone().unwrap_or_default(), loops[0].2.clone())
        };
        let floor = "shr.u32 %r3, %r1, 4;";
        let ceil = "add.s32 %r4, %r1, 7;\nshr.u32 %r3, %r4, 3;";
        let at = |t: &str, b: &str| (t.to_owned(), b.to_owned());
        let entered = at("param_0 / 16", "<block 0>");
        assert_eq!(entry(floor, "setp.lt.s32 %p1, %r1, 16;"), entered);
        assert_eq!(entry(floor, "setp.lt.s32 %p1, %r3, 1;"), entered);
        assert_eq!(
            entry(floor, "setp.lt.s32 %p1, %r1, 15;"),
            at("max(param_0 / 16, 1)", "$L__L")
        );
        assert_eq!(
            entry(ceil, "setp.lt.s32 %p1, %r1, 1;"),
            at("⌈param_0/8⌉", "<block 0>")
        );
        assert_eq!(
            entry(ceil, "setp.lt.s32 %p1, %r1, 8;"),
            at("max(⌈param_0/8⌉, 1)", "$L__L")
        );
    }

    /// PTX ISA §10.32: `%dynamic_smem_size` is fixed for the CTA at launch;
    /// §10.24: `%clock64` is a cycle counter.
    #[test]
    fn a_bound_fixed_for_the_cta_is_a_symbol() {
        let trips = |sreg: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "mov.u32 %r1, {sreg};\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 4;\n\
                     setp.lt.u32 %p1, %r2, %r1;\n@%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0]
                .1
                .as_ref()
                .map(ToString::to_string)
                .map_err(Clone::clone)
        };
        assert_eq!(
            trips("%dynamic_smem_size"),
            Ok("max(⌈%dynamic_smem_size/4⌉, 1)".to_owned())
        );
        assert_eq!(
            trips("%clock"),
            Err("latch condition depends on special register %clock".to_owned())
        );
    }

    #[test]
    fn a_comparison_combined_with_a_predicate_is_not_a_trip_count() {
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             setp.ne.s32 %p2, %r1, 7;\n\
             $L__L:\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.and.s32 %p1, %r2, 100, %p2;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(
            trips[0].1,
            Err("latch condition combines a comparison with a predicate".to_owned())
        );
    }

    /// PTX ISA §9.7.6.2: `setp`'s second output is the complemented
    /// comparison, which the latch matcher does not read.
    #[test]
    fn a_latch_on_setps_first_output_is_a_trip_count() {
        let trips = |dests: &str, pred: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
                     $L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.s32 {dests}, %r2, %r1;\n@{pred} bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0]
                .1
                .as_ref()
                .map(ToString::to_string)
                .map_err(Clone::clone)
        };
        let second = Err("latch predicate is not `setp.lt.s32`'s first output".to_owned());
        assert_eq!(trips("%p1|%p2", "%p1"), Ok("max(param_0, 1)".to_owned()));
        assert_eq!(trips("%p2|%p1", "%p1"), second);
        assert_eq!(trips("_|%p1", "%p1"), second);
    }

    /// Both low halves are 1.0 on the first iteration, so the loop runs
    /// once, not the six times an integer comparison would give.
    #[test]
    fn a_packed_comparison_is_not_a_trip_count() {
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, 0;\nmov.u32 %r2, 0x00013c00;\n\
             $L__L:\nadd.s32 %r1, %r1, 0x3c00;\n\
             setp.lt.f16x2 %p1|%p2, %r1, %r2;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1,
            Err("latch condition compares packed halves".to_owned())
        );
    }

    /// The bits of k and of 10 are subnormal, flushed to zero by `.ftz`:
    /// the loop runs once, not the ten times an integer comparison gives.
    #[test]
    fn a_floating_point_comparison_is_not_a_trip_count() {
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, 0;\nmov.b32 %f2, 10;\n\
             $L__L:\nadd.s32 %r1, %r1, 1;\nmov.b32 %f1, %r1;\n\
             setp.lt.ftz.f32 %p1, %f1, %f2;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1,
            Err("latch condition compares floating-point values".to_owned())
        );
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
        assert_eq!(
            trips[0].1.as_ref().unwrap().to_string(),
            "max(⌈param_0/8⌉, 1)"
        );
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
        assert_eq!(
            trips[0].1.as_ref().unwrap().to_string(),
            "max(param_0 / 16, 1)"
        );
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
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "max(param_0, 1)");
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
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "max(param_0, 1)");
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
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "max(param_0, 1)");
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
    fn the_reason_is_the_obstacle_behind_unread_arithmetic() {
        // LLVM writes m_start + 127 as `or` on a value with zero low bits;
        // the CTA index one value per CTA, the bound is read.
        let src = kernel_with(
            N_PARAM,
            "mov.u32 %r1, %ctaid.x;\nshl.b32 %r2, %r1, 7;\nor.b32 %r3, %r2, 127;\n\
             mov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\n\
             setp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src)[0].1.clone().map(|t| t.to_string());
        assert_eq!(trips, Ok("128 * %ctaid.x + 127".to_owned()));
        // A loaded bound behind `bfe` is a loaded bound, one that may
        // differ between threads where the kernel writes its pointer.
        let src = kernel_with(
            ".param .u64 k_param_0",
            "ld.param.u64 %rd1, [k_param_0];\nld.global.u32 %r1, [%rd1];\n\
             bfe.s32 %r3, %r1, 0, 26;\nmov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\n\
             st.global.u32 [%rd1], %r4;\nsetp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert_eq!(err, "latch condition depends on a value loaded from memory");
        // Nothing fundamental behind it, the same for every thread: the
        // value, unread.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nor.b32 %r3, %r1, 1;\nmov.u32 %r4, 0;\n\
             $L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src)[0].1.clone().map(|t| t.to_string());
        assert_eq!(trips, Ok("max(uniform(%r3), 1)".to_owned()));
    }

    #[test]
    fn a_spin_wait_names_what_defines_its_predicate() {
        let src = kernel_with(
            N_PARAM,
            "{ .reg .pred complete;\nwaitLoop:\n\
             mbarrier.test_wait.parity.shared::cta.b64 complete, [%r1], %r2;\n\
             @!complete bra.uni waitLoop; }\nret;",
        );
        let err = trips_of(&src)[0].1.clone().unwrap_err();
        assert_eq!(
            err,
            "latch predicate is defined by `mbarrier.test_wait.parity.shared::cta.b64`, not a comparison"
        );
    }

    #[test]
    fn a_read_before_the_increment_is_the_previous_iterations_value() {
        // t = i (before i += 1), compared with n: continue while k − 1 < n.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__L:\nmov.u32 %r5, %r2;\nadd.s32 %r2, %r2, 1;\n\
             setp.lt.s32 %p1, %r5, %r1;\n@%p1 bra $L__L;\nret;",
        );
        let trips = trips_of(&src);
        assert_eq!(trips[0].1.as_ref().unwrap().to_string(), "param_0 + 1");
    }

    #[test]
    fn a_value_carried_around_an_enclosing_loop_is_refused() {
        // Triangular: j < i, where i is the outer counter. The old walk
        // resolved i to its initial value and reported 0 trips.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r2, 0;\n\
             $L__O:\nmov.u32 %r3, 0;\n$L__I:\nadd.s32 %r3, %r3, 1;\n\
             setp.lt.s32 %p1, %r3, %r2;\n@%p1 bra $L__I;\n\
             add.s32 %r2, %r2, 1;\nsetp.lt.s32 %p2, %r2, %r1;\n@%p2 bra $L__O;\nret;",
        );
        let trips = trips_of(&src);
        let inner = trips
            .iter()
            .find(|(n, _)| n == "$L__I")
            .expect("inner loop");
        let err = inner.1.clone().unwrap_err();
        assert_eq!(
            err,
            "latch condition depends on an enclosing loop's counter"
        );
        let outer = trips
            .iter()
            .find(|(n, _)| n == "$L__O")
            .expect("outer loop");
        assert_eq!(outer.1.as_ref().unwrap().to_string(), "max(param_0, 1)");
    }

    /// A value two paths set apart is not read: one branch on a
    /// parameter selects it alike for every thread, one on the thread
    /// index does not.
    #[test]
    fn a_value_merged_from_two_paths_is_a_uniform_value() {
        let merged = |on: &str| {
            let src = kernel_with(
                N_PARAM,
                &format!(
                    "ld.param.u32 %r1, [k_param_0];\nmov.u32 %r6, %tid.x;\n\
                     setp.lt.s32 %p2, {on}, 8;\n@%p2 bra $L__A;\n\
                     mov.u32 %r3, 5;\nbra.uni $L__J;\n$L__A:\nmov.u32 %r3, 7;\n$L__J:\n\
                     mov.u32 %r2, 0;\n$L__L:\nadd.s32 %r2, %r2, 1;\n\
                     setp.lt.s32 %p1, %r2, %r3;\n@%p1 bra $L__L;\nret;"
                ),
            );
            trips_of(&src)[0].1.clone().map(|t| t.to_string())
        };
        assert_eq!(merged("%r1"), Ok("max(uniform(%r3), 1)".to_owned()));
        let why = "%r3 has more than one reaching definition";
        assert_eq!(merged("%r6"), Err(why.to_owned()));
    }

    #[test]
    fn arithmetic_forms_the_tracer_reads() {
        // or on a value with the low bits provably clear is an add.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nshl.b32 %r2, %r1, 7;\nor.b32 %r3, %r2, 127;\n\
             mov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r3;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1.as_ref().unwrap().to_string(),
            "128 * param_0 + 127"
        );
        // bfe.s32 from bit 0 is a width change; bfe.u32 a shift and a mask.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nbfe.s32 %r3, %r1, 0, 26;\nbfe.u32 %r6, %r3, 3, 5;\n\
             mov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r6;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1.as_ref().unwrap().to_string(),
            "max((param_0 / 8) mod 32, 1)"
        );
        // div and rem by a constant; and-masks as one run of bits.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\ndiv.u32 %r2, %r1, 8;\nrem.u32 %r3, %r1, 8;\n\
             and.b32 %r5, %r1, 0x38;\nadd.s32 %r6, %r2, %r3;\nadd.s32 %r6, %r6, %r5;\n\
             mov.u32 %r4, 0;\n$L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r6;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1.as_ref().unwrap().to_string(),
            "max(param_0 / 8 + param_0 mod 64, 1)"
        );
        // A step that is a loop-invariant register: the count is not a
        // constant multiple, so it is refused with that reason.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nshl.b32 %r2, %r1, 2;\nmov.u32 %r4, 0;\n\
             $L__L:\nadd.s32 %r4, %r4, %r2;\nsetp.lt.s32 %p1, %r4, 1024;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1.clone().unwrap_err(),
            "induction step is not a constant"
        );
        // Two merged initial values that are the same immediate.
        let src = kernel_with(
            N_PARAM,
            "ld.param.u32 %r1, [k_param_0];\nsetp.lt.s32 %p2, %r1, 8;\n@%p2 bra $L__A;\n\
             mov.u32 %r4, 0;\nbra.uni $L__J;\n$L__A:\nmov.u32 %r4, 0;\n$L__J:\n\
             $L__L:\nadd.s32 %r4, %r4, 1;\nsetp.lt.s32 %p1, %r4, %r1;\n@%p1 bra $L__L;\nret;",
        );
        assert_eq!(
            trips_of(&src)[0].1.as_ref().unwrap().to_string(),
            "max(param_0, 1)"
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
}
