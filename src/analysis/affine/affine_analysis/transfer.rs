//! The value an instruction writes to a register, from the values of the
//! operands it reads: the transfer function of each instruction the
//! analysis reads.

use super::*;
use crate::ptx::thread_hierarchy::whole_warp_shuffle;
use crate::support::integer::same_window;

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
        let form = Form::of(module, instr);
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
            && (mnemonic != Opcode::Ld
                || matches!(
                    form,
                    Form::Memory(Memory {
                        space: Some(Space::Param),
                        ..
                    })
                ) && children.len() > 1)
            && registers(module, ops[0]).contains(&reg)
        {
            let opcode = module.opcode(instr);
            return Err(format!("{name} is one of `{opcode}`'s vector destinations").into());
        }

        let value = match (mnemonic, form) {
            // An instruction of a packed integer type computes each lane
            // apart (PTX ISA §9.7.1.1, §5.2.5); an arithmetic one of a float
            // type reads its operands' bits as floats (§9.7.3 to §9.7.5),
            // not as the integers their forms hold.
            (_, Form::PackedInt) => {
                Err(format!("`{}` computes per packed lane", module.opcode(instr)).into())
            }
            (
                Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Mad | Opcode::Div | Opcode::Rem,
                Form::Float { .. },
            ) => Err(format!("`{}` is floating-point arithmetic", module.opcode(instr)).into()),
            (Opcode::Mov, _) => arg(1),
            (Opcode::Cvta, _) => {
                self.operand_at(*ops.last().expect("cvta has operands"), def, depth + 1)
            }
            // PTX ISA §9.7.9.22: an integer `cvt` converts "the lower n bits
            // corresponding to the instruction-type width" of its source, and
            // its result "is sign-extended to the destination register width
            // for signed integers"; any other conversion changes the bits.
            (Opcode::Cvt, Form::IntConvert { from, to, sat }) if !sat => {
                let a = self.operand_at(*ops.last().expect("cvt has operands"), def, depth + 1)?;
                let opcode = module.opcode(instr);
                let read = |a: &AffineExpr, ty: IntInstrType| {
                    (self.extend(a, ty))
                        .ok_or_else(|| format!("`{opcode}` leaves {} bits", ty.width).into())
                };
                read(&read(&a, from)?, to)
            }
            (Opcode::Cvt, Form::IntConvert { .. }) => {
                let opcode = module.opcode(instr);
                Err(format!("`{opcode}` saturates to its destination type").into())
            }
            (Opcode::Cvt, _) => {
                Err(format!("`{}` is not an integer conversion", module.opcode(instr)).into())
            }
            (Opcode::Ld, _) => {
                if let Some(ParamAccess::KernelParam { index, .. }) =
                    param_access(module, self.data.kernel, instr)
                {
                    return Ok(AffineExpr::from_base(
                        self.symbol(SymbolicConstant::KernelParam(index)),
                    ));
                }
                Err(UnknownReason::Load { def })
            }
            // PTX ISA §9.7.1.1, §9.7.1.2: `.sat` "limits result to
            // MININT..MAXINT (no overflow) for the size and signedness of
            // the operation", which leaves a sum its type holds.
            (Opcode::Add | Opcode::Sub, Form::Int { ty, sat: true, .. }) if ops.len() == 3 => {
                let opcode = module.opcode(instr);
                let read = |x: AffineExpr| {
                    (self.extend(&x, ty))
                        .ok_or_else(|| format!("`{opcode}` reads a value its type does not hold"))
                };
                let (a, b) = (read(arg(1)?)?, read(arg(2)?)?);
                let value = if mnemonic == Opcode::Add {
                    a + b
                } else {
                    a - b
                };
                (self.extend(&value, ty).as_ref() == Some(&value))
                    .then_some(value)
                    .ok_or_else(|| {
                        format!("`{opcode}` clamps a value its type does not hold").into()
                    })
            }
            (Opcode::Add, _) if ops.len() == 3 => Ok(arg(1)? + arg(2)?),
            (Opcode::Sub, _) if ops.len() == 3 => Ok(arg(1)? - arg(2)?),
            // PTX ISA §9.7.1.12: an integer `neg` is 0 − a; `.s8x4`
            // negates each byte on its own.
            (Opcode::Neg, Form::Int { ty, .. })
                if ops.len() == 2 && ty.kind == IntKind::Signed && ty.width >= 16 =>
            {
                Ok(AffineExpr::constant(0) - arg(1)?)
            }
            // PTX ISA §9.7.8.4: `not` inverts every bit, which in two's
            // complement is −1 − a.
            (Opcode::Not, Form::Int { ty, .. })
                if ops.len() == 2 && ty.kind == IntKind::Bits && ty.width >= 16 =>
            {
                Ok(AffineExpr::constant(-1) - arg(1)?)
            }
            (Opcode::And, _) if ops.len() == 3 => {
                // and r, a, mask — PTX's lowering of `a mod 2^n`.
                let (val, mask) = (arg(1)?, arg(2)?);
                let (affine, konst) = match (mask.as_const(), val.as_const()) {
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
                let width = form.int_type().map_or(32, |t| t.width);
                let modulo = |x: AffineExpr, k: u32| -> Result<AffineExpr, UnknownReason> {
                    (x.clone().modulo_const(1i64 << k, &|v| self.var_bounds(v)))
                        .ok_or_else(|| refuse(&x, "mod"))
                };
                let range = self.range(&affine);
                let all = konst as u64 & u64::MAX >> (64 - width);
                let (mut bits, mut sum) = (all, AffineExpr::constant(0));
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
                            _ => (hi.clone().align_down(1i64 << low, &|v| self.var_bounds(v)))
                                .ok_or_else(|| refuse(&hi, "mod"))?,
                        };
                    bits &= u64::MAX.checked_shl(top).unwrap_or(0);
                }
                Ok(sum)
            }
            (Opcode::Or, _) if ops.len() == 3 => {
                // LLVM writes an add as `or` when no bit is set on both
                // sides (its `or disjoint`, which PTX does not keep).
                let (a, b) = (arg(1)?, arg(2)?);
                let bound = |v: &Var| self.var_bounds(v);
                match a.may_set(&bound) & b.may_set(&bound) {
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
            (Opcode::Xor, _) if ops.len() == 3 => {
                let bits = |x: AffineExpr| x.xor_swizzle();
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
            (Opcode::Bfe, _) if ops.len() == 4 => {
                let (Some(pos), Some(len)) = (arg(2)?.as_const(), arg(3)?.as_const()) else {
                    return Err("bfe with a non-constant position or length".into());
                };
                let (pos, len) = (pos & 0xff, len & 0xff);
                let Some(ty) = form.int_type().filter(|t| t.kind != IntKind::Bits) else {
                    return Err("bfe of no integer type".into());
                };
                let (w, signed) = (i64::from(ty.width), ty.kind == IntKind::Signed);
                if len == 0 || !signed && pos >= w {
                    return Ok(AffineExpr::constant(0));
                }
                if pos >= w {
                    return Err(format!("bfe at {pos} of {len} bits").into());
                }
                let a = arg(1)?;
                let field = |pos: i64, len: i64| {
                    let shifted = (a.clone().div_const(1i64 << pos, &|v| self.var_bounds(v)))
                        .ok_or_else(|| refuse(&a, "bit field"))?;
                    (shifted
                        .clone()
                        .modulo_const(1i64 << len, &|v| self.var_bounds(v)))
                    .ok_or_else(|| refuse(&shifted, "bit field"))
                };
                match (signed, pos, len.min(w - pos)) {
                    (true, p, _) if len == 1 && p < 63 => {
                        Ok(field(p, 1)?.scale(SymExpr::Const(-1)))
                    }
                    (true, 0, _) => {
                        let field = IntInstrType {
                            kind: IntKind::Signed,
                            width: len.min(w) as u32,
                        };
                        (self.extend(&a, field))
                            .ok_or_else(|| format!("bfe of {len} bits leaves them").into())
                    }
                    (false, p, n) if p < 63 && n < 63 => field(p, n),
                    _ => Err(format!("bfe at {pos} of {len} bits").into()),
                }
            }
            (Opcode::Selp, Form::Int { ty, .. }) if ops.len() == 4 => {
                (self.selected(&ops, def, depth, ty)).ok_or_else(|| self.unread(instr, def))
            }
            (Opcode::Shl, _) if ops.len() == 3 => {
                let v = arg(1)?;
                let Some(b) = arg(2)?.as_const() else {
                    return Err("shift by a non-constant amount".into());
                };
                let width = form.int_type().map_or(64, |t| t.width);
                // A shift by 63 scales by 1 << 63, −2^63, which is 2^63
                // modulo the 64 bits it is read in.
                match shift_count(b, width) {
                    c if c == width => Ok(AffineExpr::constant(0)),
                    c => Ok(v.scale(SymExpr::Const(1i64 << c))),
                }
            }
            (Opcode::Shr, _) if ops.len() == 3 => {
                // PTX ISA §9.7.8.9: "Signed shifts fill with the sign bit,
                // unsigned and untyped shifts fill with 0", of the value as
                // the type reads it.
                let ty = form.int_type();
                let v = match ty {
                    Some(ty) => self.extend(&arg(1)?, ty).ok_or_else(|| {
                        let opcode = module.opcode(instr);
                        format!("`{opcode}` reads a value its type does not hold")
                    })?,
                    None => arg(1)?,
                };
                let Some(b) = arg(2)?.as_const() else {
                    return Err("shift by a non-constant amount".into());
                };
                let width = ty.map_or(64, |t| t.width);
                let signed = ty.is_some_and(|t| t.kind == IntKind::Signed);
                let c = match shift_count(b, width) {
                    c if signed => c.min(width - 1),
                    c if c == width => return Ok(AffineExpr::constant(0)),
                    c => c,
                };
                let c = i64::from(c);
                if !v.is_symbolic_or_constant() {
                    if !(0..63).contains(&c) {
                        return Err(format!("shift by {c} is out of range").into());
                    }
                    return (v.clone().div_const(1i64 << c, &|v| self.var_bounds(v)))
                        .ok_or_else(|| refuse(&v, "shift"));
                }
                let signed_width = (ty.filter(|t| t.kind == IntKind::Signed && t.width >= 16))
                    .map(|t| i64::from(t.width));
                // PTX ISA §9.7.8.9 (shr): "Signed shifts fill with the sign
                // bit", 0 where no bound shows the value negative: nvcc's
                // `x / 2^n` sign fix-up.
                if signed_width == Some(c + 1) && self.not_known_negative(&v) {
                    return Ok(AffineExpr::constant(0));
                }
                if (0..63).contains(&c) {
                    Ok(AffineExpr::from_base(SymExpr::floor_div(v.base, 1i64 << c)))
                } else {
                    Err(format!("shift by {c} is out of range").into())
                }
            }
            (Opcode::Div | Opcode::Rem, _) if ops.len() == 3 => {
                let (mut a, mut b) = (arg(1)?, arg(2)?);
                let ty = form.int_type();
                if let Some(ty) = ty {
                    b = self.extend(&b, ty).unwrap_or(b);
                }
                let Some(d) = b.as_const().filter(|&d| d > 0) else {
                    return Err(format!("{} by a non-constant divisor", mnemonic.name()).into());
                };
                // PTX ISA §9.7.1.10: a negative number's remainder, and so
                // the quotient's rounding, is machine-dependent.
                if let Some(ty) = ty {
                    let opcode = module.opcode(instr);
                    a = (self.extend(&a, ty))
                        .filter(|x| match ty.kind {
                            IntKind::Signed => self.not_known_negative(x),
                            IntKind::Unsigned | IntKind::Bits => {
                                self.within_or_unknown(x, (0, i128::MAX))
                            }
                        })
                        .ok_or_else(|| format!("`{opcode}` of a value that may be negative"))?;
                }
                let result = if mnemonic == Opcode::Div {
                    a.clone().div_const(d, &|v| self.var_bounds(v))
                } else {
                    a.clone().modulo_const(d, &|v| self.var_bounds(v))
                };
                result.ok_or_else(|| refuse(&a, "division"))
            }
            (Opcode::Shfl, Form::Shuffle(ShflMode::Idx)) if ops.len() == 5 => {
                // shfl.sync.idx d, a, c, 31, mask: a's value in lane c of the
                // warp; a fourth operand other than 31 packs a mask "for
                // logically splitting warps into sub-segments and an upper
                // bound for clamping the source lane index" (PTX ISA
                // §9.7.9.6). With warps of 32 consecutive %tid.x, that is a
                // with %tid.x replaced by 32·⌊%tid.x/32⌋ + c; a warp-uniform
                // a (a function of ⌊%tid.x/32⌋) is unchanged.
                let a = arg(1)?;
                let Some(c) = arg(2)?.as_const().filter(|c| (0..WARP_SZ).contains(c)) else {
                    return Err("shfl.idx from a non-constant lane".into());
                };
                if !arg(3)?.as_const().is_some_and(whole_warp_shuffle) {
                    return Err("shfl.idx in segments of the warp or a clamped lane".into());
                }
                let warp = |v: Var| AffineExpr::var(Var::FloorDiv(Box::new(v), WARP_SZ));
                let lane = |v: &Var| -> Option<AffineExpr> {
                    match v {
                        Var::Tid(Axis::X) => Some(
                            warp(Var::Tid(Axis::X)).scale(SymExpr::Const(WARP_SZ))
                                + AffineExpr::constant(c),
                        ),
                        Var::FloorDiv(inner, d) if **inner == Var::Tid(Axis::X) => {
                            if d % WARP_SZ == 0 {
                                Some(AffineExpr::var(v.clone()))
                            } else if WARP_SZ % d == 0 {
                                Some(
                                    warp(Var::Tid(Axis::X)).scale(SymExpr::Const(WARP_SZ / d))
                                        + AffineExpr::constant(c / d),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Mod(inner, m) if **inner == Var::Tid(Axis::X) => {
                            if WARP_SZ % m == 0 {
                                Some(AffineExpr::constant(c % m))
                            } else if m % WARP_SZ == 0 {
                                let w = Var::Mod(
                                    Box::new(Var::FloorDiv(Box::new(Var::Tid(Axis::X)), WARP_SZ)),
                                    m / WARP_SZ,
                                );
                                Some(
                                    AffineExpr::var(w).scale(SymExpr::Const(WARP_SZ))
                                        + AffineExpr::constant(c),
                                )
                            } else {
                                None
                            }
                        }
                        Var::Tid(_) => None,
                        other if other.depends_on_tid() => None,
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
            (Opcode::Min | Opcode::Max, Form::Int { ty, relu, .. }) if ops.len() == 3 && !relu => {
                let (a, b) = (self.per_cta(arg(1)?), self.per_cta(arg(2)?));
                if !(a.is_symbolic_or_constant() && b.is_symbolic_or_constant()) {
                    return Err(refuse(&(a + b), mnemonic.name()));
                }
                let opcode = module.opcode(instr);
                let read = |x: &AffineExpr| {
                    (self.extend(x, ty))
                        .ok_or_else(|| format!("`{opcode}` of a value its type does not hold"))
                };
                let (a, b) = (read(&a)?, read(&b)?);
                Ok(AffineExpr::from_base(match mnemonic {
                    Opcode::Min => SymExpr::min(a.base, b.base),
                    _ => self.max(a.base, b.base),
                }))
            }
            // PTX ISA §9.7.1.3, §9.7.1.4: `.hi` keeps the product's upper
            // half, which the affine form does not hold.
            (
                Opcode::Mul | Opcode::Mad,
                Form::Int {
                    mode: Some(MulMode::Hi),
                    ..
                },
            ) => Err(format!("`{}` keeps a product's upper half", module.opcode(instr)).into()),
            (Opcode::Mul, _) if ops.len() == 3 => {
                let [a, b] = self.wide_operands([arg(1)?, arg(2)?], instr, form)?;
                if a.is_symbolic_or_constant() {
                    Ok(b.scale(a.base))
                } else if b.is_symbolic_or_constant() {
                    Ok(a.scale(b.base))
                } else {
                    Err(refuse(&(a + b), "product"))
                }
            }
            (Opcode::Mad, _) if ops.len() == 4 => {
                let (a, b, c) = (arg(1)?, arg(2)?, arg(3)?);
                let [a, b] = self.wide_operands([a, b], instr, form)?;
                let prod = if a.is_symbolic_or_constant() {
                    b.scale(a.base)
                } else if b.is_symbolic_or_constant() {
                    a.scale(b.base)
                } else {
                    return Err(refuse(&(a + b), "product"));
                };
                Ok(prod + c)
            }
            _ => Err(self.unread(instr, def)),
        }?;
        Ok(value)
    }

    /// `selp a, b, c` of constants `a` and `b` of `ty`'s width by a `c` that an
    /// unguarded integer `setp.eq` or `setp.ne` writes, as its first output
    /// or the complement as its second (PTX ISA §9.7.6.2, §9.7.6.3), that
    /// `setp_to_zero_cmps` reads as `k·v + d cmp 0`, v a bit of `%tid.x`:
    /// the value chosen at v = 0, plus v times the change at v = 1.
    fn selected(
        &self,
        ops: &[OperandId],
        def: usize,
        depth: u32,
        ty: IntInstrType,
    ) -> Option<AffineExpr> {
        let module = self.data.module;
        let &Operand::Register(c) = module.operand(ops[3]) else {
            return None;
        };
        let (op, setp, complement) = self.data.setp_at(c, def)?;
        let Form::IntCompare {
            pred: pred @ (IntPredicate::Eq | IntPredicate::Ne),
            width,
            bool_op: None,
        } = Form::of(module, setp)
        else {
            return None;
        };
        let sops = module.operand_ids(setp.operands);
        if setp.predicate.is_some() || sops.len() != 3 {
            return None;
        }
        let read = |id, at| self.operand_at(id, at, depth + 1).ok();
        let (x, y) = (read(sops[1], op)?, read(sops[2], op)?);
        let [(diff, cmp)] = &self.setp_to_zero_cmps(pred, width, &x, &y, &|_| false)?[..] else {
            return None;
        };
        let &[(v, k)] = diff.terms.iter().collect::<Vec<_>>().as_slice() else {
            return None;
        };
        let bit = AffineExpr::var(v.clone()).xor_swizzle()?;
        if bit.constant != 0 || bit.columns.iter().any(|&col| col > 1) {
            return None;
        }
        let (k, d) = (i128::from(k.as_const()?), i128::from(diff.base.as_const()?));
        let holds = |x: i128| cmp.holds(x) != complement;
        let (lo, hi) = (-(1i128 << (ty.width - 1)), 1i128 << ty.width);
        let arm = |id| (read(id, def)?.as_const()).filter(|&x| (lo..hi).contains(&i128::from(x)));
        let (a, b) = (arm(ops[1])?, arm(ops[2])?);
        let [r0, r1] = [d, k + d].map(|x| if holds(x) { a } else { b });
        let base = AffineExpr::constant(r0);
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
                let ty =
                    (Form::of(module, instr).int_type()).filter(|t| t.kind == IntKind::Bits)?;
                let bits = ty.width / count as u32;
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

    /// A product's operands as it reads them: a `.wide` one's as its type
    /// (PTX ISA §9.7.1.3, §9.7.1.4), through `extend`, refused where one
    /// crosses the edge of the type's values.
    fn wide_operands(
        &self,
        ops: [AffineExpr; 2],
        instr: &Instr,
        form: Form,
    ) -> Result<[AffineExpr; 2], UnknownReason> {
        let Form::Int {
            ty,
            mode: Some(MulMode::Wide),
            ..
        } = form
        else {
            return Ok(ops);
        };
        let [a, b] = ops.map(|a| self.extend(&a, ty));
        a.zip(b).map(|(a, b)| [a, b]).ok_or_else(|| {
            let opcode = self.data.module.opcode(instr);
            format!("`{opcode}` reads a value its type does not hold").into()
        })
    }

    /// `a` is negative under no bound known of it: its range over the
    /// launch nor its `least`, every symbol nonnegative; one of no bound
    /// is read so by the input contract (PLAN.md).
    fn not_known_negative(&self, a: &AffineExpr) -> bool {
        self.within_or_unknown(a, (0, i128::MAX)) && self.least(a).is_none_or(|l| l >= 0)
    }

    /// `a`'s range over the known launch and constant trip counts lies in
    /// [lo, hi), or is unknown.
    fn within_or_unknown(&self, a: &AffineExpr, (lo, hi): (i128, i128)) -> bool {
        self.range(a).is_none_or(|(x, y)| lo <= x && y < hi)
    }

    /// The integer the low `ty.width` bits of `a` read as, sign-extended
    /// where `ty` is signed and zero-extended otherwise (PTX ISA §9.4:
    /// "integer operands are silently cast to the instruction type";
    /// §9.4.1: "If the corresponding instruction type is signed integer, the
    /// data is sign-extended; otherwise, the data is zero-extended"): `a`
    /// where its range over the known launch and constant trip counts
    /// lies in those values or is unknown, as the input contract
    /// (PLAN.md) reads such a value; `a − m·2^w` where it lies in them
    /// shifted by m·2^w; else, below 63 bits, least + (a − least) mod 2^w,
    /// the rest past its multiples of 2^w shifted into the values where
    /// its range lies in one block of 2^w, else as `divmod` reads it.
    pub(super) fn extend(&self, a: &AffineExpr, ty: IntInstrType) -> Option<AffineExpr> {
        let Some((lo, hi)) = self.range(a) else {
            return Some(a.clone());
        };
        let least = match ty.kind {
            IntKind::Signed => ty.values().0,
            IntKind::Unsigned | IntKind::Bits => 0,
        };
        let shifted = |x: AffineExpr, m: i128| -> Option<AffineExpr> {
            let shift = i64::try_from(m.checked_mul(1i128 << ty.width)?).ok()?;
            Some(x - AffineExpr::constant(shift))
        };
        match same_window(lo, hi, least, 1i128 << ty.width) {
            Some(0) => Some(a.clone()),
            Some(m) => shifted(a.clone(), m),
            None if ty.width < 63 => {
                let (d, least) = (1i64 << ty.width, i64::try_from(least).ok()?);
                let (_, r) = (a.clone() - AffineExpr::constant(least)).multiple_part(d)?;
                let rem = match self
                    .range(&r)
                    .and_then(|(lo, hi)| same_window(lo, hi, 0, d.into()))
                {
                    Some(0) => r,
                    Some(m) => shifted(r, m)?,
                    None => r.divmod(d, &|v| self.var_bounds(v))?.1,
                };
                Some(rem + AffineExpr::constant(least))
            }
            None => None,
        }
    }

    /// The comparisons `form cmp 0` whose conjunction an integer `setp` `pred`
    /// of `width`-bit operands `a` and `b` holds on, each operand the integer
    /// its bits read as (`extend`); Polly's `ScopBuilder::buildConditionSets`
    /// translates an icmp likewise. `None` where `extend` reads no form of
    /// one. An `eq` or `ne` is `a − b − m·2^width cmp 0` for the one
    /// multiple m·2^width in the range of `a − b`, so that it is zero exactly
    /// where `a − b` is zero modulo 2^width; `1 cmp 0` where the range holds
    /// none, and `a − b cmp 0` where it is unknown. An `lt`, `le`, `gt` or `ge` of a signed type is `A − B cmp 0`
    /// of the operands as signed. Of an unsigned type (PTX ISA §9.3.1.1: "The
    /// unsigned comparisons are eq, ne, lo (lower), ls (lower-or-same), hi
    /// (higher), and hs (higher-or-same)") it is `A − B cmp 0` of them as
    /// unsigned where each has a known range or is `nonnegative`, a value of
    /// unknown range below 2^(width−1) by the input contract (PLAN.md); else
    /// of them as signed, X the lower side and Y the higher, `lo` and `ls`
    /// comparing A with B and `hi` and `hs` B with A, where Y is
    /// `nonnegative`: the bits of X are below those of Y in [0, 2^(width−1))
    /// when 0 ≤ X and X < Y, at or below them when 0 ≤ X and X ≤ Y, `0 ≤ X`
    /// dropped where X is `nonnegative`. A thread condition takes as
    /// `nonnegative` only what `least` proves, the premise of LLVM's fold of
    /// `0 ≤ i && i < n` into `i <u n` (`InstCombinerImpl::simplifyRangeCheck`
    /// makes it only where n's known bits are nonnegative). A loop's latch
    /// or guard also takes so an operand of unknown range free of its
    /// counter, as the input contract reads a trip count, and as SCEV's
    /// `howManyLessThans` reads an unsigned bound as its unsigned value,
    /// asking only that the counter not wrap.
    pub(crate) fn setp_to_zero_cmps(
        &self,
        pred: IntPredicate,
        width: u32,
        a: &AffineExpr,
        b: &AffineExpr,
        nonnegative: &dyn Fn(&AffineExpr) -> bool,
    ) -> Option<Vec<(AffineExpr, Cmp)>> {
        let read = |x: &AffineExpr, kind| self.extend(x, IntInstrType { kind, width });
        let unsigned = |x: &AffineExpr| {
            read(x, IntKind::Unsigned).filter(|_| self.range(x).is_some() || nonnegative(x))
        };
        let w = 1i128 << width;
        match pred.reads() {
            IntKind::Bits => {
                let form = a.clone() - b.clone();
                let Some((lo, hi)) = self.range(&form) else {
                    return Some(vec![(form, pred.cmp())]);
                };
                let (first, last) = (lo.checked_add(w - 1)?.div_euclid(w), hi.div_euclid(w));
                return match last.checked_sub(first)? {
                    ..0 => Some(vec![(AffineExpr::constant(1), pred.cmp())]),
                    0 => {
                        let multiple = i64::try_from(first.checked_mul(w)?).ok()?;
                        Some(vec![(form - AffineExpr::constant(multiple), pred.cmp())])
                    }
                    _ => None,
                };
            }
            IntKind::Unsigned if let (Some(a), Some(b)) = (unsigned(a), unsigned(b)) => {
                return Some(vec![(a - b, pred.cmp())]);
            }
            IntKind::Signed | IntKind::Unsigned => {}
        }
        let (a, b) = (read(a, IntKind::Signed)?, read(b, IntKind::Signed)?);
        let (x, y) = match pred {
            IntPredicate::Ult | IntPredicate::Ule => (&a, &b),
            IntPredicate::Ugt | IntPredicate::Uge => (&b, &a),
            _ => return Some(vec![(a - b, pred.cmp())]),
        };
        if !nonnegative(y) {
            return None;
        }
        let sign = (!nonnegative(x)).then(|| (x.clone(), Cmp::Ge));
        Some(sign.into_iter().chain([(a - b, pred.cmp())]).collect())
    }
}

/// Operation `what` of `a`, whose variables it does not take.
fn refuse(a: &AffineExpr, what: &'static str) -> UnknownReason {
    UnknownReason::NotAffine {
        what,
        vars: a.terms.keys().cloned().collect(),
    }
}
