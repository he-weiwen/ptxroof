//! The value an instruction writes to a register, from the values of the
//! operands it reads: the transfer function of each instruction the
//! analysis reads.

use super::*;

impl AffineAnalysis<'_> {
    /// The value statement `def` writes to `reg`, were its guard true.
    pub(super) fn transfer(
        &self,
        def: usize,
        reg: Symbol,
        depth: u32,
    ) -> Result<AffineExpr, UnknownReason> {
        let module = self.data.module;
        let name = module.interner.resolve(reg);
        let Stmt::Instr(instr) = &self.data.kernel.stmts[def] else {
            return Err("value defined by a statement the parser could not read".into());
        };
        let mnemonic = instr.mnemonic;
        let mods: Vec<&str> = self
            .data
            .module
            .modifiers(instr)
            .iter()
            .map(|&m| module.interner.resolve(m))
            .collect();
        let ops = module.operand_ids(instr.operands).to_vec();
        let arg = |i: usize| -> Result<AffineExpr, UnknownReason> {
            self.operand_at(ops[i], def, depth + 1)
        };
        // A brace-enclosed destination's registers are the value's elements in
        // order (PTX ISA 6.4.3, "Vectors as Operands"): each is a parameter's
        // whole value only when the load has one element, and the element of
        // the same place in a special register's vector.
        if mnemonic == Opcode::Mov
            && let Some(Operand::VectorList { children }) = ops.first().map(|&o| module.operand(o))
            && let Some(&Operand::Special(SpecialReg::Vector(vector, None))) =
                ops.get(1).map(|&o| module.operand(o))
            && let Some(place) = module
                .operand_ids(*children)
                .iter()
                .position(|&c| matches!(module.operand(c), Operand::Register(r) if *r == reg))
            && let Some(&element) = [Element::X, Element::Y, Element::Z, Element::W].get(place)
        {
            return self.trace_special(SpecialReg::Vector(vector, Some(element)));
        }
        if mnemonic == Opcode::Mov
            && let Some(Operand::VectorList { children }) = ops.first().map(|&o| module.operand(o))
            && let Some(&Operand::Register(src)) = ops.get(1).map(|&o| module.operand(o))
            && let elements = module.operand_ids(*children)
            && let Some(place) = (elements.iter())
                .position(|&c| matches!(module.operand(c), Operand::Register(r) if *r == reg))
            && let Some(value) = self.unpacked(src, def, place, elements.len(), depth)
        {
            return value;
        }
        if let Some(Operand::VectorList { children }) = ops.first().map(|&o| module.operand(o))
            && (mnemonic != Opcode::Ld || mods.contains(&"param") && children.len() > 1)
            && registers(module, ops[0]).contains(&reg)
        {
            let opcode = module.opcode(instr);
            return Err(format!("{name} is one of `{opcode}`'s vector destinations").into());
        }

        let value = match mnemonic {
            Opcode::Mov => arg(1),
            Opcode::Cvta => {
                self.operand_at(*ops.last().expect("cvta has operands"), def, depth + 1)
            }
            // PTX ISA §9.7.9.22: an integer `cvt` keeps a value its source
            // type reads that its destination type holds; any other
            // conversion changes the value's bits.
            Opcode::Cvt => {
                let opcode = module.opcode(instr);
                let types: Vec<IntType> =
                    mods.iter().filter_map(|m| IntType::from_name(m)).collect();
                let numeric = mods.iter().filter(|m| m.contains(char::is_numeric)).count();
                let (&[to, from], 2) = (&types[..], numeric) else {
                    return Err(format!("`{opcode}` is not an integer conversion").into());
                };
                let a = self.operand_at(*ops.last().expect("cvt has operands"), def, depth + 1)?;
                for ty in [from, to] {
                    if !self.within(&a, ty.values()) {
                        return Err(format!("`{opcode}` leaves {} bits", ty.width).into());
                    }
                }
                Ok(a)
            }
            Opcode::Ld => {
                if mods.contains(&"param")
                    && let Some(Operand::Memory { base, .. }) =
                        ops.get(1).map(|&i| module.operand(i))
                    && let Operand::SymbolRef(pname) = module.operand(*base)
                    && let Some(&idx) = self.data.params.get(pname)
                {
                    return Ok(AffineExpr::from_base(SymExpr::Sym(
                        SymbolicConstant::KernelParam(idx),
                    )));
                }
                Err(UnknownReason::Load { def })
            }
            Opcode::Add if ops.len() == 3 => Ok(arg(1)? + arg(2)?),
            Opcode::Sub if ops.len() == 3 => Ok(arg(1)? - arg(2)?),
            // PTX ISA §9.7.1.12: an integer `neg` is 0 − a; `.s8x4`
            // negates each byte on its own.
            Opcode::Neg
                if ops.len() == 2 && matches!(mods.last(), Some(&("s16" | "s32" | "s64"))) =>
            {
                Ok(AffineExpr::from_base(SymExpr::Const(0)) - arg(1)?)
            }
            Opcode::And if ops.len() == 3 => {
                // and r, a, mask — PTX's lowering of `a mod 2^n`.
                let (val, mask) = (arg(1)?, arg(2)?);
                let (affine, konst) = match (self.constant(&mask), self.constant(&val)) {
                    (Some(c), _) => (val, c),
                    (_, Some(c)) => (mask, c),
                    _ => return Err("and-mask with two non-constant operands".into()),
                };
                // Each run of ones from bit `low` up to bit `top`: x mod
                // 2^top − x mod 2^low, and x itself when the run reaches the
                // register's top bit; one that stops just below it keeps x
                // for x in [0, 2^(width−1)), and, when x's range is unknown,
                // in the nonneg domain for a 32- or 64-bit mask of that run
                // alone. The runs' bits are disjoint, so the and is their
                // sum; PTX ISA §4.5.1 converts the constant to the
                // instruction's size.
                let width = (mods.iter())
                    .find_map(|m| m.strip_prefix('b')?.parse::<u32>().ok())
                    .unwrap_or(32);
                let modulo = |x: AffineExpr, k: u32| -> Result<AffineExpr, UnknownReason> {
                    (self.reduced(&x, |x| x.modulo_const(1i64 << k, &|v| self.var_bounds(v))))
                        .ok_or_else(|| refuse(&x, "mod"))
                };
                let bound = self.bindings.map_or(affine.clone(), |m| affine.bind(m));
                let range = bound.range(&|v| self.var_bounds(v));
                let all = konst as u64 & u64::MAX >> (64 - width);
                let (mut bits, mut sum) = (all, AffineExpr::from_base(SymExpr::Const(0)));
                let held = range.map(|(lo, hi)| lo >= 0 && hi >> (width - 1) == 0);
                // A value of 0 or −1, all ones, keeps none or all of the
                // mask: −mask·v, alike signed and unsigned below the sign bit.
                if range.is_some_and(|(lo, hi)| lo >= -1 && hi <= 0) && all >> (width - 1) == 0 {
                    return Ok(affine.scale(SymExpr::Const(-(all as i64))));
                }
                while bits != 0 {
                    let low = bits.trailing_zeros();
                    let top = low + (bits >> low).trailing_ones();
                    let hi = if top == width
                        || top == width - 1 && held.unwrap_or(bits == all && width >= 32)
                    {
                        affine.clone()
                    } else if top == width - 1 && (held.is_some() || top == 63) {
                        let bit = width - 1;
                        return Err(format!(
                            "and-mask {konst:#x} clears bit {bit} of a value that may set it"
                        )
                        .into());
                    } else {
                        modulo(affine.clone(), top)?
                    };
                    sum = sum
                        + match low {
                            0 => hi,
                            63 => {
                                return Err(
                                    format!("and-mask {konst:#x} keeps bit 63 without 62").into()
                                );
                            }
                            _ => (self.reduced(&hi, |x| {
                                x.align_down(1i64 << low, &|v| self.var_bounds(v))
                            }))
                            .ok_or_else(|| refuse(&hi, "mod"))?,
                        };
                    bits &= u64::MAX.checked_shl(top).unwrap_or(0);
                }
                Ok(sum)
            }
            Opcode::Or if ops.len() == 3 => {
                // LLVM writes an add as `or` when no bit is set on both
                // sides (its `or disjoint`, which PTX does not keep).
                let (a, b) = (arg(1)?, arg(2)?);
                let bound = |x: &AffineExpr| self.bindings.map_or(x.clone(), |m| x.bind(m));
                let largest = |v: &Var| self.var_bounds(v).map(|(_, hi)| hi);
                match bound(&a).may_set(&largest) & bound(&b).may_set(&largest) {
                    0 => Ok(a + b),
                    both => Err(format!(
                        "or whose sides may both set bit {}",
                        both.trailing_zeros()
                    )
                    .into()),
                }
            }
            // PTX ISA §9.7.8.3: a bitwise exclusive-or, of two xors of
            // thread index fields their xor.
            Opcode::Xor if ops.len() == 3 => {
                let bits =
                    |x: AffineExpr| self.bindings.map_or(x.clone(), |m| x.bind(m)).xor_swizzle();
                match (arg(1).map(bits), arg(2).map(bits)) {
                    (Ok(Some(x)), Ok(Some(y))) => Ok(x.xor(&y).affine()),
                    _ => Err(self.unread(instr, def)),
                }
            }
            // PTX ISA §9.7.1.20: pos and len are b & 0xff and c & 0xff; a
            // field past the msb, or all of a signed one's high bits, is the
            // sign bit a[min(pos + len - 1, msb)], 0 when unsigned or len is 0.
            // A signed field from past the msb ptxas 13.4 folds otherwise
            // (micro/bfe_fields.ptx).
            Opcode::Bfe if ops.len() == 4 => {
                let (Some(pos), Some(len)) = (self.constant(&arg(2)?), self.constant(&arg(3)?))
                else {
                    return Err("bfe with a non-constant position or length".into());
                };
                let (pos, len) = (pos & 0xff, len & 0xff);
                let Some(ty) = IntType::of(&mods).filter(|t| t.kind != IntKind::Bits) else {
                    return Err("bfe of no integer type".into());
                };
                let (w, signed) = (i64::from(ty.width), ty.kind == IntKind::Signed);
                if len == 0 || !signed && pos >= w {
                    return Ok(AffineExpr::from_base(SymExpr::Const(0)));
                }
                if pos >= w {
                    return Err(format!("bfe at {pos} of {len} bits").into());
                }
                let a = arg(1)?;
                let field = |pos: i64, len: i64| {
                    let shifted = (self
                        .reduced(&a, |x| x.div_const(1i64 << pos, &|v| self.var_bounds(v))))
                    .ok_or_else(|| refuse(&a, "bit field"))?;
                    (self.reduced(&shifted, |x| {
                        x.modulo_const(1i64 << len, &|v| self.var_bounds(v))
                    }))
                    .ok_or_else(|| refuse(&shifted, "bit field"))
                };
                match (signed, pos, len.min(w - pos)) {
                    (true, p, _) if len == 1 && p < 63 => {
                        Ok(field(p, 1)?.scale(SymExpr::Const(-1)))
                    }
                    (true, 0, _) => {
                        let half = 1i128 << (len.min(w) - 1);
                        match self.within(&a, (-half, half)) {
                            false => Err(format!("bfe of {len} bits leaves them").into()),
                            true => Ok(a),
                        }
                    }
                    (false, p, n) if p < 63 && n < 63 => field(p, n),
                    _ => Err(format!("bfe at {pos} of {len} bits").into()),
                }
            }
            Opcode::Selp if ops.len() == 4 => {
                let selected =
                    IntType::of(&mods).and_then(|ty| self.selected(&ops, def, depth, ty));
                selected.ok_or_else(|| self.unread(instr, def))
            }
            Opcode::Shl if ops.len() == 3 => {
                let v = arg(1)?;
                // PTX ISA §9.7.8.8: the count is b's unsigned 32-bit value,
                // and one past the width is clamped to it, which shifts
                // every bit out.
                let width = mods
                    .iter()
                    .rev()
                    .find_map(|m| m.get(1..)?.parse::<i64>().ok());
                match self.constant(&arg(2)?).map(|c| i64::from(c as u32)) {
                    Some(c) if width.is_some_and(|w| c >= w) => {
                        Ok(AffineExpr::from_base(SymExpr::Const(0)))
                    }
                    Some(c) if (0..63).contains(&c) => Ok(v.scale(SymExpr::Const(1i64 << c))),
                    _ => Err("shift by a non-constant amount".into()),
                }
            }
            Opcode::Shr if ops.len() == 3 => {
                let v = arg(1)?;
                // PTX ISA §9.7.8.9: "Signed shifts fill with the sign bit,
                // unsigned and untyped shifts fill with 0", of the value as
                // the type reads it.
                if let Some(ty) = IntType::of(&mods)
                    && let read = match ty.kind {
                        IntKind::Signed => ty.values(),
                        IntKind::Unsigned | IntKind::Bits => ty.nonnegative(),
                    }
                    && !self.within(&v, read)
                {
                    let opcode = module.opcode(instr);
                    return Err(format!("`{opcode}` reads a value its type does not hold").into());
                }
                let Some(c) = self.constant(&arg(2)?) else {
                    return Err("shift by a non-constant amount".into());
                };
                if !v.is_symbolic_or_constant() {
                    if !(0..63).contains(&c) {
                        return Err(format!("shift by {c} is out of range").into());
                    }
                    return (self.reduced(&v, |x| x.div_const(1i64 << c, &|v| self.var_bounds(v))))
                        .ok_or_else(|| refuse(&v, "shift"));
                }
                let signed_width = mods.iter().find_map(|m| match *m {
                    "s16" => Some(16),
                    "s32" => Some(32),
                    "s64" => Some(64),
                    _ => None,
                });
                // PTX ISA §9.7.8.9 (shr): "Signed shifts fill with the sign
                // bit" — in this module's nonnegative domain that bit is 0
                // unless the value is known negative, so shifting it down is
                // nvcc's `x / 2^n` sign fix-up.
                if signed_width == Some(c + 1) && self.within(&v, (0, i128::MAX)) {
                    return Ok(AffineExpr::from_base(SymExpr::Const(0)));
                }
                if (0..63).contains(&c) {
                    Ok(AffineExpr::from_base(SymExpr::floor_div(v.base, 1i64 << c)))
                } else {
                    Err(format!("shift by {c} is out of range").into())
                }
            }
            Opcode::Div | Opcode::Rem if ops.len() == 3 => {
                let (a, b) = (arg(1)?, arg(2)?);
                // PTX ISA §9.7.1.10: a negative number's remainder, and so
                // the quotient's rounding, is machine-dependent.
                if let Some(ty) = IntType::of(&mods)
                    && !(self.within(&a, ty.nonnegative()) && self.within(&b, ty.nonnegative()))
                {
                    let opcode = module.opcode(instr);
                    return Err(format!("`{opcode}` of a value that may be negative").into());
                }
                let Some(d) = self.constant(&b).filter(|&d| d > 0) else {
                    return Err(format!("{} by a non-constant divisor", mnemonic.name()).into());
                };
                let result = if mnemonic == Opcode::Div {
                    self.reduced(&a, |x| x.div_const(d, &|v| self.var_bounds(v)))
                } else {
                    self.reduced(&a, |x| x.modulo_const(d, &|v| self.var_bounds(v)))
                };
                result.ok_or_else(|| refuse(&a, "division"))
            }
            Opcode::Shfl if mods.contains(&"idx") && ops.len() == 5 => {
                // shfl.sync.idx d, a, c, 31, mask: a's value in lane c of the
                // warp; a fourth operand other than 31 packs a mask "for
                // logically splitting warps into sub-segments and an upper
                // bound for clamping the source lane index" (PTX ISA
                // §9.7.9.6). With warps of 32 consecutive %tid.x, that is a
                // with %tid.x replaced by 32·⌊%tid.x/32⌋ + c; a warp-uniform
                // a (a function of ⌊%tid.x/32⌋) is unchanged.
                let a = arg(1)?;
                let Some(c) = self.constant(&arg(2)?).filter(|c| (0..32).contains(c)) else {
                    return Err("shfl.idx from a non-constant lane".into());
                };
                if self.constant(&arg(3)?) != Some(31) {
                    return Err("shfl.idx in segments of the warp or a clamped lane".into());
                }
                let warp = |v: Var| AffineExpr::var(Var::FloorDiv(Box::new(v), 32));
                let lane = |v: &Var| -> Option<AffineExpr> {
                    match v {
                        Var::Tid(Axis::X) => Some(
                            warp(Var::Tid(Axis::X)).scale(SymExpr::Const(32))
                                + AffineExpr::from_base(SymExpr::Const(c)),
                        ),
                        Var::FloorDiv(inner, d) if **inner == Var::Tid(Axis::X) => {
                            if d % 32 == 0 {
                                Some(AffineExpr::var(v.clone()))
                            } else if 32 % d == 0 {
                                Some(
                                    warp(Var::Tid(Axis::X)).scale(SymExpr::Const(32 / d))
                                        + AffineExpr::from_base(SymExpr::Const(c / d)),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Mod(inner, m) if **inner == Var::Tid(Axis::X) => {
                            if 32 % m == 0 {
                                Some(AffineExpr::from_base(SymExpr::Const(c % m)))
                            } else if m % 32 == 0 {
                                let w = Var::Mod(
                                    Box::new(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), 32)),
                                    m / 32,
                                );
                                Some(
                                    AffineExpr::var(w).scale(SymExpr::Const(32))
                                        + AffineExpr::from_base(SymExpr::Const(c)),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Tid(_) => None,
                        other if depends_on_lane(other) => None,
                        other => Some(AffineExpr::var(other.clone())),
                    }
                };
                // Lane c's counter and unknown values are this lane's only
                // where the warp's lanes share them.
                self.alike(&a, def, Uniform::Warp, None)?;
                a.map_vars(lane).ok_or_else(|| {
                    "shfl.idx of a value the tracer does not read in another lane".into()
                })
            }
            // PTX ISA §9.7.1.13, §9.7.1.14: one integer type, compared as
            // that type reads both operands; `.relu` and the packed types
            // are other operations.
            Opcode::Min | Opcode::Max
                if ops.len() == 3 && mods.len() == 1 && IntType::of(&mods).is_some() =>
            {
                let (a, b) = (per_cta(arg(1)?), per_cta(arg(2)?));
                if !(a.is_symbolic_or_constant() && b.is_symbolic_or_constant()) {
                    return Err(refuse(&(a + b), mnemonic.name()));
                }
                if let Some(ty) = IntType::of(&mods)
                    && !(self.within(&a, ty.values()) && self.within(&b, ty.values()))
                {
                    let opcode = module.opcode(instr);
                    return Err(format!("`{opcode}` of a value its type does not hold").into());
                }
                Ok(AffineExpr::from_base(match mnemonic {
                    Opcode::Min => SymExpr::min(a.base, b.base),
                    _ => SymExpr::max(a.base, b.base),
                }))
            }
            // PTX ISA §9.7.1.3, §9.7.1.4: `.hi` keeps the product's upper
            // half, which the affine form does not hold.
            Opcode::Mul | Opcode::Mad if mods.contains(&"hi") => {
                Err(format!("`{}` keeps a product's upper half", module.opcode(instr)).into())
            }
            Opcode::Mul if ops.len() == 3 => {
                let (a, b) = (arg(1)?, arg(2)?);
                self.wide_operands(&[&a, &b], instr, &mods)?;
                if a.is_symbolic_or_constant() {
                    Ok(b.scale(a.base))
                } else if b.is_symbolic_or_constant() {
                    Ok(a.scale(b.base))
                } else {
                    Err(refuse(&(a + b), "product"))
                }
            }
            Opcode::Mad if ops.len() == 4 => {
                let (a, b, c) = (arg(1)?, arg(2)?, arg(3)?);
                self.wide_operands(&[&a, &b], instr, &mods)?;
                let prod = if a.is_symbolic_or_constant() {
                    b.scale(a.base)
                } else if b.is_symbolic_or_constant() {
                    a.scale(b.base)
                } else {
                    return Err(refuse(&(a + b), "product"));
                };
                Ok(self.fits(prod, instr, &mods)? + c)
            }
            _ => Err(self.unread(instr, def)),
        }?;
        self.fits(value, instr, &mods)
    }

    /// `selp a, b, c` of constants `a` and `b` of `ty` by a `c` that an
    /// unguarded integer `setp.eq` or `setp.ne` writes, as its first output
    /// or the complement as its second (PTX ISA §9.7.6.2, §9.7.6.3), of two
    /// forms whose difference is k·v + d, v a bit of `%tid.x`: the value
    /// chosen at v = 0, plus v times the change at v = 1.
    fn selected(
        &self,
        ops: &[OperandId],
        def: usize,
        depth: u32,
        ty: IntType,
    ) -> Option<AffineExpr> {
        let module = self.data.module;
        let &Operand::Register(c) = module.operand(ops[3]) else {
            return None;
        };
        let Some(Value::Result { op, .. }) = self.data.ssa.value_before(def, Register::Named(c))
        else {
            return None;
        };
        let Stmt::Instr(setp) = &self.data.kernel.stmts[op] else {
            return None;
        };
        let mods: Vec<&str> = (module.modifiers(setp).iter())
            .map(|&m| module.interner.resolve(m))
            .collect();
        let cmp = Cmp::from_name(mods.first()?).filter(|c| matches!(c, Cmp::Eq | Cmp::Ne))?;
        let width = IntType::of(&mods).filter(|_| mods.len() == 2)?.width;
        let sops = module.operand_ids(setp.operands);
        if setp.mnemonic != Opcode::Setp || setp.predicate.is_some() || sops.len() != 3 {
            return None;
        }
        let writes = |id: OperandId| matches!(module.operand(id), Operand::Register(r) if *r == c);
        let complement = match module.operand(sops[0]) {
            Operand::MultipleDestinations { children } => match *module.operand_ids(*children) {
                [p, _] if writes(p) => false,
                [_, q] if writes(q) => true,
                _ => return None,
            },
            _ if writes(sops[0]) => false,
            _ => return None,
        };
        let read = |id, at| {
            let x = self.operand_at(id, at, depth + 1).ok()?;
            Some(self.bindings.map_or(x.clone(), |m| x.bind(m)))
        };
        let diff = read(sops[1], op)? - read(sops[2], op)?;
        let &[(v, k)] = diff.terms.iter().collect::<Vec<_>>().as_slice() else {
            return None;
        };
        let bit = AffineExpr::var(v.clone()).xor_swizzle()?;
        if bit.constant != 0 || bit.columns.iter().any(|&col| col > 1) {
            return None;
        }
        let (k, d) = (i128::from(k.as_const()?), i128::from(diff.base.as_const()?));
        let holds = |x: i128| (x.rem_euclid(1 << width) == 0) ^ (cmp == Cmp::Ne) ^ complement;
        let (lo, hi) = ty.values();
        let arm = |id| {
            self.constant(&read(id, def)?)
                .filter(|&x| (lo..hi).contains(&i128::from(x)))
        };
        let (a, b) = (arm(ops[1])?, arm(ops[2])?);
        let [r0, r1] = [d, k + d].map(|x| if holds(x) { a } else { b });
        let base = AffineExpr::from_base(SymExpr::Const(r0));
        match r1.checked_sub(r0)? {
            0 => Some(base),
            change => Some(base + AffineExpr::var(v.clone()).scale(SymExpr::Const(change))),
        }
    }

    /// Why statement `def`'s instruction gives no value: what its inputs
    /// depend on behind it (`depends_on`), else the instruction.
    fn unread(&self, instr: &Instr, def: usize) -> UnknownReason {
        let module = self.data.module;
        let (mut values, mut found) = (Vec::new(), Vec::new());
        self.inputs(instr, def, &mut values, &mut found);
        match self.depends_on(values, found) {
            Some(reason) => UnknownReason::Behind {
                reason: Box::new(reason),
                opcode: module.opcode(instr).to_string(),
            },
            None => format!(
                "value defined by unsupported instruction `{}`",
                instr.mnemonic.name()
            )
            .into(),
        }
    }

    /// Element `place` of the `count` a `mov` unpacks from `src` at `pos`,
    /// when a `mov` packed `src` from as many, through scalar copies:
    /// unpacking gives back the packed elements bit for bit (PTX ISA
    /// §9.7.9.4). `None` when `src` was not packed so, or the element is
    /// an immediate wider than its bits.
    fn unpacked(
        &self,
        src: Symbol,
        pos: usize,
        place: usize,
        count: usize,
        depth: u32,
    ) -> Option<Result<AffineExpr, UnknownReason>> {
        let module = self.data.module;
        let Some(v @ Value::Result { op, .. }) =
            self.data.ssa.value_before(pos, Register::Named(src))
        else {
            return None;
        };
        let Stmt::Instr(instr) = &self.data.kernel.stmts[op] else {
            return None;
        };
        if depth > DEPTH_LIMIT {
            self.cut_at(v);
        }
        if instr.mnemonic != Opcode::Mov || depth > DEPTH_LIMIT {
            return None;
        }
        let &[dest, packed] = module.operand_ids(instr.operands) else {
            return None;
        };
        let value = match module.operand(packed) {
            Operand::VectorList { children } => {
                let elements = module.operand_ids(*children);
                let element = *elements.get(place).filter(|_| elements.len() == count)?;
                let bits = (module.modifiers(instr).iter()).find_map(|&m| {
                    let width = module.interner.resolve(m).strip_prefix('b')?;
                    width.parse::<u32>().ok()
                })? / count as u32;
                if let Operand::Immediate(t) = module.operand(element)
                    && !parse_int(module.interner.resolve(*t)).is_some_and(|v| {
                        (-(1i128 << (bits - 1))..1i128 << bits).contains(&i128::from(v))
                    })
                {
                    return None;
                }
                Some(self.operand_at(element, op, depth + 1))
            }
            &Operand::Register(copied) if matches!(module.operand(dest), Operand::Register(_)) => {
                self.unpacked(copied, op, place, count, depth + 1)
            }
            _ => None,
        }?;
        if instr.predicate.is_some() && value.is_ok() {
            let name = module.interner.resolve(src);
            return Some(Err(format!("{name} is written under a guard").into()));
        }
        Some(value)
    }

    /// A `.wide` product's operands, which it reads as its type (PTX ISA
    /// §9.7.1.3, §9.7.1.4): one whose range over the known launch and
    /// constant trip counts the type does not hold is refused.
    fn wide_operands(
        &self,
        ops: &[&AffineExpr],
        instr: &Instr,
        mods: &[&str],
    ) -> Result<(), UnknownReason> {
        let Some(ty) = (mods.contains(&"wide"))
            .then(|| IntType::of(mods))
            .flatten()
        else {
            return Ok(());
        };
        for a in ops {
            if !self.within(a, ty.values()) {
                let opcode = self.data.module.opcode(instr);
                return Err(format!("`{opcode}` reads a value its type does not hold").into());
            }
        }
        Ok(())
    }

    /// `a`'s range over the known launch and constant trip counts lies in
    /// [lo, hi), or is unknown.
    fn within(&self, a: &AffineExpr, (lo, hi): (i128, i128)) -> bool {
        let bound = self.bindings.map_or(a.clone(), |m| a.bind(m));
        (bound.range(&|v| self.var_bounds(v))).is_none_or(|(x, y)| lo <= x && y < hi)
    }

    /// A 16- or 32-bit `add`, `sub`, `neg`, `mul.lo`, `mad.lo`, `shl` or `bfe`
    /// result, which wraps (PTX ISA §9.7.1), whose range over the known
    /// launch and constant trip counts leaves its type is refused.
    pub(super) fn fits(
        &self,
        value: AffineExpr,
        instr: &Instr,
        mods: &[&str],
    ) -> Result<AffineExpr, UnknownReason> {
        let checked = match instr.mnemonic {
            Opcode::Add | Opcode::Sub | Opcode::Neg | Opcode::Shl | Opcode::Bfe => true,
            Opcode::Mul | Opcode::Mad => mods.contains(&"lo"),
            _ => false,
        };
        let ty = (mods.iter().rev())
            .find_map(|m| IntType::from_name(m).filter(|ty| matches!(ty.width, 16 | 32)));
        match ty {
            Some(ty) if checked && !self.within(&value, ty.values()) => Err(format!(
                "`{}` leaves {} bits",
                self.data.module.opcode(instr),
                ty.width
            )
            .into()),
            _ => Ok(value),
        }
    }
}

/// Operation `what` of `a`, whose variables it does not take.
fn refuse(a: &AffineExpr, what: &'static str) -> UnknownReason {
    UnknownReason::NotAffine {
        what,
        vars: a.terms.keys().cloned().collect(),
    }
}
