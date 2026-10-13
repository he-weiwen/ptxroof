//! The scalar affine tracer: a register's value at a statement, from
//! register SSA (`crate::ssa`), as an affine form over thread and CTA
//! indices, loop counters and the kernel parameters. The trip matcher
//! reads latch conditions through it; address and branch-condition
//! analyses read memory operands and predicates through the same code.
//!
//! A loop counter is a recurrence (`super::recurrences`), read as its
//! start plus `k − 1` steps; another φ is one immediate every path
//! writes, or refused. The tracer follows supported arithmetic and moves
//! down to `ld.param`, constants and special registers; unsupported operations
//! is a named reason, preferring the fundamental obstacle (a special
//! register, a memory load, an atomic) behind arithmetic it does not
//! read. The domain is nonnegative and non-overflowing, as documented
//! in [`super::trip_counts`].

use crate::analysis::control_flow::loops::{LoopForest, LoopId};
use crate::analysis::register_effects::{self, registers, split_operands};
use crate::analysis::scalar::affine::{Affine, Axis, Var};
use crate::analysis::scalar::recurrences::{Recurrence, Step, recurrences};
use crate::analysis::scalar::symexpr::SymExpr;
use crate::analysis::scalar::trip_counts::loop_trips;
use crate::analysis::uniformity::Uniformity;
use crate::ptx::cfg::{BlockId, ControlFlowGraph};
use crate::ptx::ir::{
    Element, Instr, Kernel, Module, Opcode, Operand, OperandId, Register, SpecialReg,
    SpecialScalar, SpecialVector, Stmt,
};
use crate::ptx::literal::parse_int;
use crate::ssa::{RegisterSsa, Value};
use crate::support::intern::Symbol;
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{HashMap, HashSet};

pub(crate) struct AffineValueTracer<'a> {
    pub(crate) module: &'a Module,
    pub(crate) kernel: &'a Kernel,
    pub(crate) cfg: &'a ControlFlowGraph,
    pub(crate) forest: &'a LoopForest,
    /// The statements writing each register, CC included, in order.
    pub(crate) writes: HashMap<Register, Vec<usize>>,
    /// The value each read of a register sees.
    pub(crate) ssa: RegisterSsa<'a>,
    /// param symbol -> positional index.
    pub(crate) params: HashMap<Symbol, usize>,
    /// `reach[a][b]`: a path of at least one edge from block a to b.
    pub(crate) reach: Vec<Vec<bool>>,
    /// Each loop counter by its header φ, with its loop.
    pub(crate) recurrences: HashMap<Value, (LoopId, Recurrence)>,
    /// Inside a counter's start or register step: a guarded write there
    /// is refused, not read as its new value.
    strict: Cell<u32>,
    /// Nested loops' constant trip counts, for counters stepped by their
    /// exits; `None` while one is being solved, or when it is not one.
    trips: RefCell<HashMap<LoopId, Option<i64>>>,
    /// `--bind` values, applied where a trace needs a constant.
    pub(crate) bindings: Option<&'a HashMap<String, i64>>,
    uniformity: OnceCell<Uniformity>,
}

impl<'a> AffineValueTracer<'a> {
    pub(crate) fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        cfg: &'a ControlFlowGraph,
        forest: &'a LoopForest,
    ) -> Self {
        let mut writes: HashMap<Register, Vec<usize>> = HashMap::new();
        for (i, stmt) in kernel.stmts.iter().enumerate() {
            let in_cfg = (cfg.blocks.iter()).any(|b| (b.start..b.end).contains(&i));
            if !in_cfg {
                continue;
            }
            for reg in register_effects::writes(module, stmt) {
                writes.entry(reg).or_default().push(i);
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
        let ssa = RegisterSsa::new(cfg, &forest.doms, writes.clone());
        // Inner loops first: an outer counter may step by an inner exit.
        let mut order: Vec<LoopId> = (0..forest.loops.len() as u32).map(LoopId).collect();
        order.sort_by_key(|&l| std::cmp::Reverse(forest.get(l).depth));
        let (mut recurrences_by_phi, mut by_update) = (HashMap::new(), HashMap::new());
        for l in order {
            let lp = forest.get(l);
            for r in recurrences(module, kernel, cfg, forest, &ssa, l, &by_update) {
                for (p, v) in ssa.incoming(r.phi) {
                    if p.is_some_and(|p| lp.blocks.binary_search(&p).is_ok()) {
                        by_update.insert(v, (l, r.clone()));
                    }
                }
                recurrences_by_phi.insert(r.phi, (l, r));
            }
        }
        AffineValueTracer {
            module,
            kernel,
            cfg,
            forest,
            writes,
            ssa,
            params,
            reach,
            recurrences: recurrences_by_phi,
            strict: Cell::new(0),
            trips: RefCell::new(HashMap::new()),
            bindings: None,
            uniformity: OnceCell::new(),
        }
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
        self.writes
            .get(&Register::Named(reg))?
            .iter()
            .find_map(|&d| {
                let Stmt::Instr(instr) = &self.kernel.stmts[d] else {
                    return None;
                };
                (split_operands(self.module, instr).1.iter())
                    .find_map(|&op| self.operand_obstacle(op, d, id, &mut seen))
            })
    }

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

    /// The last write of `reg` in block `b` before statement `before`.
    pub(crate) fn last_write_in(&self, b: BlockId, before: usize, reg: Symbol) -> Option<usize> {
        let blk = self.cfg.block(b);
        let defs = self.writes.get(&Register::Named(reg))?;
        (defs.iter().rev())
            .copied()
            .find(|&d| blk.start <= d && d < before)
    }

    /// The last write of `pred` in block `b` before statement `before`,
    /// if it is a `setp` whose first output is `pred`.
    pub(crate) fn find_setp(
        &self,
        b: BlockId,
        before: usize,
        pred: Symbol,
    ) -> Option<(usize, &Instr)> {
        let d = self.last_write_in(b, before, pred)?;
        let Stmt::Instr(instr) = &self.kernel.stmts[d] else {
            return None;
        };
        let first = match self
            .module
            .operand(*self.module.operand_ids(instr.operands).first()?)
        {
            Operand::MultipleDestinations { children } => self
                .module
                .operand(*self.module.operand_ids(*children).first()?),
            op => op,
        };
        (instr.mnemonic == Opcode::Setp && *first == Operand::Register(pred)).then_some((d, instr))
    }

    /// The block holding statement `stmt`.
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

    pub(crate) fn trace_operand(
        &self,
        op: OperandId,
        pos: usize,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<Affine, String> {
        match self.module.operand(op) {
            Operand::Register(reg) => self.trace_reg(*reg, pos, id, depth),
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
            Operand::Offset { base, offset } => Ok(self.trace_operand(*base, pos, id, depth)?
                + Affine::invariant(SymExpr::Const(*offset))),
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
    ) -> Result<Affine, String> {
        let Some(v) = self.ssa.value_before(pos, Register::Named(reg)) else {
            let name = self.module.interner.resolve(reg);
            return Err(format!("{name} is read in a block no path reaches"));
        };
        Ok(self.after_exits(self.trace_value(v, id, depth)?, pos))
    }

    /// `a` read at `pos`: the counter of a loop `pos` is outside, whose
    /// one latch exit comes after a constant number of trips, is that
    /// number.
    fn after_exits(&self, mut a: Affine, pos: usize) -> Affine {
        fn at(v: &Var, n: i64) -> Option<(LoopId, i64)> {
            match v {
                Var::Iter(l) => Some((*l, n)),
                Var::Div(inner, d) => at(inner, n).map(|(l, x)| (l, x.div_euclid(*d))),
                Var::Mod(inner, m) => at(inner, n).map(|(l, x)| (l, x.rem_euclid(*m))),
                _ => None,
            }
        }
        let escaped: Vec<Var> = (a.terms.keys())
            .filter(|v| at(v, 0).is_some_and(|(l, _)| !self.in_loop(l, pos)))
            .cloned()
            .collect();
        for v in escaped {
            let Some((l, _)) = at(&v, 0) else { continue };
            let Some((_, x)) = self.constant_trips(l).and_then(|n| at(&v, n)) else {
                continue;
            };
            let c = a.terms.remove(&v).expect("an escaped term");
            a.base = SymExpr::add(a.base, SymExpr::mul(c, SymExpr::Const(x)));
        }
        a
    }

    /// A φ: a counter's `start + (k − 1)·step`, else one immediate every
    /// path writes.
    fn trace_phi(
        &self,
        v: Value,
        block: BlockId,
        reg: Symbol,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<Affine, String> {
        let name = self.module.interner.resolve(reg);
        let header = (self.forest.loops.iter()).any(|l| l.header == block);
        let refused = || {
            self.obstacle_of_any_def(reg, id)
                .unwrap_or_else(|| match header {
                    true => format!(
                        "latch condition depends on {name}, carried around an enclosing loop"
                    ),
                    false => format!("{name} has more than one reaching definition"),
                })
        };
        if let Some((l, r)) = self.recurrences.get(&v) {
            return (self.counter(*l, r, id, depth)).map_err(|why| why.unwrap_or_else(refused));
        }
        let mut defs = Vec::new();
        for s in self.ssa.sources(v) {
            let why = match s {
                Value::Result { op, .. } => match &self.kernel.stmts[op] {
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
            return Err(self.obstacle_of_any_def(reg, id).unwrap_or(why));
        }
        match self.same_immediate(&defs) {
            Some(c) => Ok(Affine::invariant(SymExpr::Const(c))),
            None => Err(refused()),
        }
    }

    /// Counter `r` of loop `l` in its iteration `k`: its start plus
    /// `(k − 1)` steps, start and register steps read with no guarded
    /// write and the steps free of every variable; `None` when a step
    /// varies, so `r` counts nothing.
    fn counter(
        &self,
        l: LoopId,
        r: &Recurrence,
        id: Option<LoopId>,
        depth: u32,
    ) -> Result<Affine, Option<String>> {
        let (start, step) = self.strictly(|| {
            let start = self.trace_value(r.start, id, depth + 1)?;
            Ok::<_, Option<String>>((start, self.step(l, &r.step, depth)?))
        })?;
        // The values the back edges feed, start + k·step for k in [1, n]
        // when the loop runs a constant n trips, stay in their type.
        if self.constant_trips(l).is_some() {
            let fed = Affine::term(Var::Iter(l), step.clone()) + start.clone();
            let lp = self.forest.get(l);
            for (_, v) in (self.ssa.incoming(r.phi).into_iter())
                .filter(|(p, _)| p.is_some_and(|p| lp.blocks.binary_search(&p).is_ok()))
            {
                if let Value::Result { op, .. } = v
                    && let Stmt::Instr(instr) = &self.kernel.stmts[op]
                {
                    let mods: Vec<&str> = (self.module.modifiers(instr).iter())
                        .map(|&m| self.module.interner.resolve(m))
                        .collect();
                    self.fits(fed.clone(), instr, &mods).map_err(Some)?;
                }
            }
        }
        if step == SymExpr::Const(0) {
            return Ok(start);
        }
        let back = SymExpr::mul(SymExpr::Const(-1), step.clone());
        Ok(Affine::term(Var::Iter(l), step) + start + Affine::invariant(back))
    }

    /// A counter's step in loop `l`: its constant, its registers, each
    /// free of every variable, and each nested loop's step times that
    /// loop's constant trip count of at least 1; `None` otherwise.
    fn step(&self, l: LoopId, step: &Step, depth: u32) -> Result<SymExpr, Option<String>> {
        let mut sum = SymExpr::Const(step.constant);
        for &(reg, at, sign) in &step.registers {
            let s = (self.trace_reg(reg, at, Some(l), depth + 1)).map_err(|_| None)?;
            if !s.is_invariant() {
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

    /// Which reads every lane executing them sees alike, built on first
    /// use.
    pub(crate) fn uniformity(&self) -> &Uniformity {
        self.uniformity.get_or_init(|| Uniformity::new(self))
    }

    /// `f`, refusing every guarded write it reads.
    pub(crate) fn strictly<T>(&self, f: impl FnOnce() -> T) -> T {
        self.strict.set(self.strict.get() + 1);
        let t = f();
        self.strict.set(self.strict.get() - 1);
        t
    }

    /// Loop `l`'s trip count when it is a constant of at least 1, which
    /// makes its latch, and so its exit, the same for every lane.
    fn constant_trips(&self, l: LoopId) -> Option<i64> {
        if let Some(&n) = self.trips.borrow().get(&l) {
            return n;
        }
        self.trips.borrow_mut().insert(l, None);
        let n = (loop_trips(self, l).ok())
            .and_then(|t| t.as_const())
            .filter(|&n| n >= 1);
        self.trips.borrow_mut().insert(l, n);
        n
    }

    fn trace_value(&self, v: Value, id: Option<LoopId>, depth: u32) -> Result<Affine, String> {
        if depth > 32 {
            return Err("value trace exceeds depth limit".to_owned());
        }
        let (def, reg) = match v {
            Value::Result {
                op,
                reg: Register::Named(reg),
            } => (op, reg),
            Value::Argument {
                block,
                reg: Register::Named(reg),
            } => return self.trace_phi(v, block, reg, id, depth),
            Value::Undef(Register::Named(reg)) => {
                let name = self.module.interner.resolve(reg);
                return Err(format!("no definition found for {name}"));
            }
            _ => return Err("value is the carry flag".to_owned()),
        };
        let name = self.module.interner.resolve(reg);
        if self.strict.get() > 0
            && let Stmt::Instr(instr) = &self.kernel.stmts[def]
            && instr.predicate.is_some()
        {
            return Err(format!("{name} is written under a guard"));
        }

        let Stmt::Instr(instr) = &self.kernel.stmts[def] else {
            return Err("value defined by a statement the parser could not read".to_owned());
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
            && registers(self.module, ops[0]).contains(&reg)
        {
            let opcode = self.module.opcode(instr);
            return Err(format!("{name} is one of `{opcode}`'s vector destinations"));
        }

        let value = match mnemonic {
            Opcode::Mov => arg(1),
            Opcode::Cvta => {
                self.trace_operand(*ops.last().expect("cvta has operands"), def, id, depth + 1)
            }
            // PTX ISA §9.7.9.22: an integer `cvt` keeps a value its source
            // type reads that its destination type holds; any other
            // conversion changes the value's bits.
            Opcode::Cvt => {
                let opcode = self.module.opcode(instr);
                let types: Vec<(u32, i128, i128)> =
                    mods.iter().filter_map(|m| int_type_range(m)).collect();
                let numeric = mods.iter().filter(|m| m.contains(char::is_numeric)).count();
                let [to, from] = types[..] else {
                    return Err(format!("`{opcode}` is not an integer conversion"));
                };
                if numeric != 2 {
                    return Err(format!("`{opcode}` is not an integer conversion"));
                }
                let a =
                    self.trace_operand(*ops.last().expect("cvt has operands"), def, id, depth + 1)?;
                let bound_a = match self.bindings {
                    Some(b) => a.bind(b),
                    None => a.clone(),
                };
                let range = bound_a.range(&|v| self.bound(v));
                for (width, lo, hi) in [from, to] {
                    if range.is_some_and(|(x, y)| x < lo || y >= hi) {
                        return Err(format!("`{opcode}` leaves {width} bits"));
                    }
                }
                Ok(a)
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
                    let half = 1i128 << (len.clamp(1, 64) - 1);
                    return match a.range(&|v| self.bound(v)) {
                        Some((lo, hi)) if lo < -half || hi >= half => {
                            Err(format!("bfe of {len} bits leaves them"))
                        }
                        _ => Ok(a),
                    };
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
                // PTX ISA §9.7.8.8: a count past the width is clamped to
                // it, which shifts every bit out.
                let width = mods
                    .iter()
                    .rev()
                    .find_map(|m| m.get(1..)?.parse::<i64>().ok());
                match self.constant(&arg(2)?) {
                    Some(c) if width.is_some_and(|w| c >= w) => {
                        Ok(Affine::invariant(SymExpr::Const(0)))
                    }
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
                // Lane c's counter is this lane's only where lanes leave its
                // loop together.
                for l in (a.terms.keys()).filter_map(Var::counter) {
                    if !self.in_loop(l, def)
                        && let Err(why) = self.uniformity().exits_uniform(self, l)
                    {
                        return Err(format!(
                            "shfl.idx of a counter that differs between lanes after its loop, whose exit {why}"
                        ));
                    }
                }
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
            // PTX ISA §9.7.1.3, §9.7.1.4: `.hi` keeps the product's upper
            // half, which the affine form does not hold.
            Opcode::Mul | Opcode::Mad if mods.contains(&"hi") => Err(format!(
                "`{}` keeps a product's upper half",
                self.module.opcode(instr)
            )),
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
                Ok(self.fits(prod, instr, &mods)? + c)
            }
            other => {
                let mut seen = HashSet::new();
                let fundamental = (split_operands(self.module, instr).1.iter())
                    .find_map(|&op| self.operand_obstacle(op, def, id, &mut seen));
                Err(match fundamental {
                    Some(reason) => format!("{reason}, behind `{}`", self.module.opcode(instr)),
                    None => format!(
                        "value defined by unsupported instruction `{}`",
                        other.name()
                    ),
                })
            }
        }?;
        self.fits(value, instr, &mods)
    }

    /// A 16- or 32-bit `add`, `sub`, `mul.lo`, `mad.lo`, `shl` or `bfe`
    /// result, which wraps (PTX ISA §9.7.1), whose range over the known
    /// launch and constant trip counts leaves its type is refused.
    fn fits(&self, value: Affine, instr: &Instr, mods: &[&str]) -> Result<Affine, String> {
        let checked = match instr.mnemonic {
            Opcode::Add | Opcode::Sub | Opcode::Shl | Opcode::Bfe => true,
            Opcode::Mul | Opcode::Mad => mods.contains(&"lo"),
            _ => false,
        };
        let Some((width, lo, hi)) = mods.iter().rev().find_map(|m| type_range(m)) else {
            return Ok(value);
        };
        let bound = |v: &Var| self.bound(v);
        let bound_value = match self.bindings {
            Some(b) => value.bind(b),
            None => value.clone(),
        };
        match bound_value.range(&bound) {
            Some((a, b)) if checked && (a < lo || b >= hi) => Err(format!(
                "`{}` leaves {width} bits",
                self.module.opcode(instr)
            )),
            _ => Ok(value),
        }
    }

    /// A variable's least and greatest value when the launch or a
    /// constant trip count fixes it: `%tid` below the bound block shape,
    /// `%ctaid` below a bound grid, a counter in [1, n].
    fn bound(&self, v: &Var) -> Option<(i64, i64)> {
        let shape = |vector: SpecialVector, axis: &Axis| {
            let i = match axis {
                Axis::X => 0,
                Axis::Y => 1,
                Axis::Z => 2,
            };
            let n = *self.bindings?.get(&vector.components()[i].name())?;
            (n >= 1).then_some((0, n - 1))
        };
        match v {
            Var::Tid(axis) => shape(SpecialVector::Ntid, axis),
            Var::Ctaid(axis) => shape(SpecialVector::Nctaid, axis),
            Var::Iter(l) => self.constant_trips(*l).map(|n| (1, n)),
            Var::Div(..) | Var::Mod(..) => None,
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
        let def = match self.ssa.value_before(pos, Register::Named(reg)) {
            Some(Value::Result { op, .. }) => op,
            Some(Value::Argument { .. }) => return self.obstacle_of_any_def(reg, id),
            Some(Value::Undef(_)) | None => return None,
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
            _ => (split_operands(self.module, instr).1.iter())
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

/// A 16- or 32-bit integer type's width and the values it reads, a `b`
/// bit pattern either way.
fn type_range(m: &str) -> Option<(u32, i128, i128)> {
    int_type_range(m).filter(|(w, _, _)| matches!(w, 16 | 32))
}

/// An integer type's width and the values it reads: `u`, `s`, and a `b`
/// bit pattern either way.
fn int_type_range(m: &str) -> Option<(u32, i128, i128)> {
    let (kind, width) = m.split_at_checked(1)?;
    let width: u32 = width
        .parse()
        .ok()
        .filter(|w| matches!(w, 8 | 16 | 32 | 64))?;
    let (lo, hi) = (-(1i128 << (width - 1)), 1i128 << width);
    match kind {
        "u" => Some((width, 0, hi)),
        "s" => Some((width, lo, 1i128 << (width - 1))),
        "b" => Some((width, lo, hi)),
        _ => None,
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
