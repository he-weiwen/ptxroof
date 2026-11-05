//! What an instruction's modifiers name — instruction types, comparisons
//! and state spaces, read by the syntax of the instruction's family — and
//! what a `.param` address refers to.

use crate::ptx::ir::{FundamentalType, Instr, Kernel, Module, Opcode, Operand, named_enum};

/// An integer instruction type of 8 to 64 bits, unsigned `.u`, signed
/// `.s` or a `.b` bit pattern: how an instruction reads its operands
/// (PTX ISA §9.4); an operand of another, compatible type is "silently
/// cast to the instruction type" (§6.1). Not the type a declaration
/// gives a variable, `FundamentalType` (§5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntInstrType {
    pub(crate) kind: IntKind,
    pub(crate) width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntKind {
    Unsigned,
    Signed,
    Bits,
}

impl IntInstrType {
    /// The type a modifier names, `u32`.
    fn from_name(m: &str) -> Option<IntInstrType> {
        use FundamentalType::*;
        let ty = FundamentalType::from_name(m)?;
        let kind = match ty {
            S8 | S16 | S32 | S64 => IntKind::Signed,
            U8 | U16 | U32 | U64 => IntKind::Unsigned,
            B8 | B16 | B32 | B64 => IntKind::Bits,
            F16 | F16x2 | F32 | F64 | B128 | Pred => return None,
        };
        let width = 8 * ty.bytes()?;
        Some(IntInstrType { kind, width })
    }

    /// The values the type reads, [lo, hi): a bit pattern as either sign.
    pub(crate) fn values(self) -> (i128, i128) {
        let (half, full) = (1i128 << (self.width - 1), 1i128 << self.width);
        match self.kind {
            IntKind::Unsigned => (0, full),
            IntKind::Signed => (-half, half),
            IntKind::Bits => (-half, full),
        }
    }

    /// The values it reads that are not negative.
    pub(crate) fn nonnegative(self) -> (i128, i128) {
        (0, self.values().1)
    }
}

/// The type an instruction-type modifier names (PTX ISA §5.2): an integer or
/// a packed integer, a float, scalar or of 2 `lanes`, `.b128` or `.pred`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstrType {
    Int(IntInstrType),
    PackedInt,
    Float { lanes: u8 },
    B128,
    Pred,
}

impl InstrType {
    /// A Table 8 fundamental type (§5.2.1), or `bf16` (§5.2.2), `bf16x2`,
    /// `f32x2`, `u16x2`, `s16x2`, `u8x4` or `s8x4` (§5.2.5).
    fn from_name(m: &str) -> Option<InstrType> {
        use FundamentalType::*;
        match m {
            "bf16" => return Some(InstrType::Float { lanes: 1 }),
            "bf16x2" | "f32x2" => return Some(InstrType::Float { lanes: 2 }),
            "u16x2" | "s16x2" | "u8x4" | "s8x4" => return Some(InstrType::PackedInt),
            _ => {}
        }
        Some(match FundamentalType::from_name(m)? {
            S8 | S16 | S32 | S64 | U8 | U16 | U32 | U64 | B8 | B16 | B32 | B64 => {
                InstrType::Int(IntInstrType::from_name(m)?)
            }
            F16 | F32 | F64 => InstrType::Float { lanes: 1 },
            F16x2 => InstrType::Float { lanes: 2 },
            B128 => InstrType::B128,
            Pred => InstrType::Pred,
        })
    }

    /// The form of an instruction of this one type and no other modifier.
    fn form(self) -> Option<Form> {
        Some(match self {
            InstrType::Int(ty) => Form::Int {
                ty,
                mode: None,
                sat: false,
                relu: false,
            },
            InstrType::PackedInt => Form::PackedInt,
            InstrType::Float { lanes } => Form::Float { lanes },
            InstrType::Pred => Form::Pred,
            InstrType::B128 => return None,
        })
    }
}

/// A comparison of two integers, `a cmp b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    /// The comparison that holds exactly when this one does not (LLVM's
    /// `CmpInst::getInversePredicate`).
    pub fn inverse(self) -> Cmp {
        match self {
            Cmp::Eq => Cmp::Ne,
            Cmp::Ne => Cmp::Eq,
            Cmp::Lt => Cmp::Ge,
            Cmp::Ge => Cmp::Lt,
            Cmp::Le => Cmp::Gt,
            Cmp::Gt => Cmp::Le,
        }
    }

    /// The comparison with its operands swapped (LLVM's
    /// `CmpInst::getSwappedPredicate`).
    pub fn swapped(self) -> Cmp {
        match self {
            Cmp::Lt => Cmp::Gt,
            Cmp::Gt => Cmp::Lt,
            Cmp::Le => Cmp::Ge,
            Cmp::Ge => Cmp::Le,
            eq_or_ne => eq_or_ne,
        }
    }
}

/// An integer `setp`'s comparison of its operands' bits (PTX ISA §9.7.6.2),
/// LLVM's `ICmpInst::Predicate`: `lt` and `lo` of an unsigned type are both
/// `Ult`, and `eq` of any type is `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntPredicate {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

impl IntPredicate {
    /// §9.7.6.2: "The signed and unsigned comparison operators are eq, ne,
    /// lt, le, gt, ge. For unsigned values, the comparison operators lo, ls,
    /// hi, and hs for lower, lower-or-same, higher, and higher-or-same may be
    /// used instead of lt, le, gt, ge, respectively. The untyped, bit-size
    /// comparisons are eq and ne."
    fn of(op: CmpOp, kind: IntKind) -> Option<IntPredicate> {
        Some(match (op, kind) {
            (CmpOp::Eq, _) => Self::Eq,
            (CmpOp::Ne, _) => Self::Ne,
            (CmpOp::Lt, IntKind::Signed) => Self::Slt,
            (CmpOp::Le, IntKind::Signed) => Self::Sle,
            (CmpOp::Gt, IntKind::Signed) => Self::Sgt,
            (CmpOp::Ge, IntKind::Signed) => Self::Sge,
            (CmpOp::Lt | CmpOp::Lo, IntKind::Unsigned) => Self::Ult,
            (CmpOp::Le | CmpOp::Ls, IntKind::Unsigned) => Self::Ule,
            (CmpOp::Gt | CmpOp::Hi, IntKind::Unsigned) => Self::Ugt,
            (CmpOp::Ge | CmpOp::Hs, IntKind::Unsigned) => Self::Uge,
            _ => return None,
        })
    }

    /// How it reads its operands: `Bits` for `Eq` and `Ne`, which compare
    /// the bits.
    pub(crate) fn reads(self) -> IntKind {
        match self {
            Self::Eq | Self::Ne => IntKind::Bits,
            Self::Slt | Self::Sle | Self::Sgt | Self::Sge => IntKind::Signed,
            Self::Ult | Self::Ule | Self::Ugt | Self::Uge => IntKind::Unsigned,
        }
    }

    /// The comparison of the integers its operands read as.
    pub(crate) fn cmp(self) -> Cmp {
        match self {
            Self::Eq => Cmp::Eq,
            Self::Ne => Cmp::Ne,
            Self::Slt | Self::Ult => Cmp::Lt,
            Self::Sle | Self::Ule => Cmp::Le,
            Self::Sgt | Self::Ugt => Cmp::Gt,
            Self::Sge | Self::Uge => Cmp::Ge,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Space {
    Global,
    Shared,
    /// `.shared::cluster` — distinct from CTA-local shared (`::cta` is
    /// the ISA default and folds into `Shared`).
    SharedCluster,
    Local,
    Const,
    Param,
    /// `ld`/`st` with no state space: PTX generic addressing. Its own
    /// honest bucket — unclassifiable to a concrete space without
    /// provenance (anti-scope until a fixture emits one).
    Generic,
}

impl Space {
    /// The state space a modifier names (PTX ISA §5.1): `.shared` is
    /// `.shared::cta`, and `.param` takes either sub-qualifier (§5.1.6).
    fn from_modifier(modifier: &str) -> Option<Space> {
        Some(match modifier {
            "global" => Space::Global,
            "shared" | "shared::cta" => Space::Shared,
            "shared::cluster" => Space::SharedCluster,
            "local" => Space::Local,
            "const" => Space::Const,
            "param" | "param::entry" | "param::func" => Space::Param,
            _ => return None,
        })
    }

    pub fn key(self) -> &'static str {
        match self {
            Space::Global => "global",
            Space::Shared => "shared",
            Space::SharedCluster => "shared::cluster",
            Space::Local => "local",
            Space::Const => "const",
            Space::Param => "param",
            Space::Generic => "generic",
        }
    }
}

/// `instr`'s modifiers, in order.
fn names<'a>(module: &'a Module, instr: &Instr) -> impl Iterator<Item = &'a str> {
    (module.modifiers(instr).iter()).map(|&m| module.interner.resolve(m))
}

/// What an instruction's modifiers say, read by the syntax of its family
/// (PTX ISA §9.7), for the instructions the analyses read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    /// `add`, `sub`, `mul`, `mad`, `div`, `rem`, `neg`, `min`, `max`, `bfe`
    /// (§9.7.1, §9.7.2), `and`, `or`, `xor`, `not`, `shl`, `shr` (§9.7.8),
    /// `selp` (§9.7.6.3) or `mov` (§9.7.9.3, §9.7.9.4) of one scalar integer
    /// type; `mode` is a `mul`'s or `mad`'s `.lo`, `.hi` or `.wide`.
    Int {
        ty: IntInstrType,
        mode: Option<MulMode>,
        sat: bool,
        relu: bool,
    },
    /// One of those of a packed integer type, which computes each lane
    /// apart (§9.7.1.1, §5.2.5).
    PackedInt,
    /// One of those, or a `setp`, of a float type, scalar or of 2 `lanes`
    /// (§9.7.3 to §9.7.5, §9.7.6.2, §9.7.7.2).
    Float { lanes: u8 },
    /// `and`, `or`, `xor`, `not` or `mov` of `.pred`.
    Pred,
    /// `setp.CmpOp{.BoolOp}.type` of an integer type (§9.7.6.2).
    IntCompare {
        pred: IntPredicate,
        width: u32,
        bool_op: Option<BoolOp>,
    },
    /// `cvt{.sat}.dtype.atype` of two integer types (§9.7.9.22).
    IntConvert {
        from: IntInstrType,
        to: IntInstrType,
        sat: bool,
    },
    /// `shfl` (§9.7.9.5, §9.7.9.6).
    Shuffle(ShflMode),
    /// `ld`, `ldu` or `st` (§9.7.9.8, §9.7.9.10, §9.7.9.11).
    Memory(Memory),
    /// Any other instruction, or modifiers outside its family's syntax.
    Other,
}

named_enum! {
    /// The bits of a product a `mul` or `mad` keeps: its low or high
    /// half, or all of it (§9.7.1.3).
    pub(crate) enum MulMode { Lo = "lo", Hi = "hi", Wide = "wide" }
}

named_enum! {
    /// A `setp` comparison (§9.3.1).
    enum CmpOp {
        Eq = "eq", Ne = "ne", Lt = "lt", Le = "le", Gt = "gt", Ge = "ge",
        Lo = "lo", Ls = "ls", Hi = "hi", Hs = "hs",
        Equ = "equ", Neu = "neu", Ltu = "ltu", Leu = "leu", Gtu = "gtu", Geu = "geu",
        Num = "num", Nan = "nan",
    }
}

named_enum! {
    /// How a `setp` combines its comparison with predicate `c` (§9.7.6.2).
    pub(crate) enum BoolOp { And = "and", Or = "or", Xor = "xor" }
}

named_enum! {
    /// The lane a `shfl` reads from (§9.7.9.6).
    pub(crate) enum ShflMode { Up = "up", Down = "down", Bfly = "bfly", Idx = "idx" }
}

/// A load's or store's state space, its memory-consistency semantics and
/// whether it takes a cache policy (§9.7.9.8, §9.7.9.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Memory {
    pub(crate) space: Option<Space>,
    pub(crate) sem: Sem,
    pub(crate) cache_hint: bool,
}

/// A memory instruction's semantics (§8.4): `.weak`, the default,
/// `.volatile`, `.relaxed`, `.acquire`, `.release` or `.mmio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sem {
    Weak,
    Volatile,
    Relaxed,
    Acquire,
    Release,
    Mmio,
}

impl Form {
    /// `instr`'s form: its family by opcode, its modifiers by that
    /// family's syntax.
    pub(crate) fn of(module: &Module, instr: &Instr) -> Form {
        use Opcode::*;
        let mods: Vec<&str> = names(module, instr).collect();
        let one = || match mods[..] {
            [m] => InstrType::from_name(m),
            _ => None,
        };
        let form = match instr.mnemonic {
            Add | Sub | Mul | Mad | Div | Rem | Neg | Min | Max | Bfe => {
                arithmetic(instr.mnemonic, &mods)
            }
            And | Or | Xor | Not | Selp | Mov => one().and_then(InstrType::form),
            Shl | Shr => match one() {
                Some(t @ InstrType::Int(_)) => t.form(),
                _ => None,
            },
            Setp => comparison(&mods),
            Cvt => conversion(&mods),
            Shfl => mods
                .iter()
                .find_map(|m| ShflMode::from_name(m))
                .map(Form::Shuffle),
            Ld | Ldu | St => Some(Form::Memory(memory(module, instr, &mods))),
            _ => None,
        };
        form.unwrap_or(Form::Other)
    }

    /// The type of a scalar integer instruction, `Form::Int`.
    pub(crate) fn int_type(self) -> Option<IntInstrType> {
        match self {
            Form::Int { ty, .. } => Some(ty),
            _ => None,
        }
    }
}

/// The modifiers of `op`, an integer or float arithmetic instruction: the
/// first that names a type picks the family; a float's others are not
/// read, an integer's are each at most once `.sat`, `.relu`, `.cc` and,
/// of a `mul` or `mad`, `.lo`, `.hi` or `.wide`.
fn arithmetic(op: Opcode, mods: &[&str]) -> Option<Form> {
    let (at, ty) =
        (mods.iter().enumerate()).find_map(|(i, m)| Some((i, InstrType::from_name(m)?)))?;
    if let InstrType::Float { lanes } = ty {
        return Some(Form::Float { lanes });
    }
    let product = matches!(op, Opcode::Mul | Opcode::Mad);
    let (mut mode, mut sat, mut relu, mut cc) = (None, false, false, false);
    for (i, &m) in mods.iter().enumerate() {
        let flag = match (m, MulMode::from_name(m)) {
            _ if i == at => continue,
            ("sat", _) => &mut sat,
            ("relu", _) => &mut relu,
            ("cc", _) => &mut cc,
            (_, Some(named)) if product && mode.is_none() => {
                mode = Some(named);
                continue;
            }
            _ => return None,
        };
        if std::mem::replace(flag, true) {
            return None;
        }
    }
    match ty {
        InstrType::Int(ty) => Some(Form::Int {
            ty,
            mode,
            sat,
            relu,
        }),
        InstrType::PackedInt => Some(Form::PackedInt),
        _ => None,
    }
}

/// `setp`'s modifiers: its comparison first, its type last, and between
/// them at most one `BoolOp` and, of a float type, one `.ftz` (§9.7.6.2:
/// "Modifier .ftz applies only to .f32 comparisons"; §9.7.7.2).
fn comparison(mods: &[&str]) -> Option<Form> {
    let (first, rest) = mods.split_first()?;
    let (last, between) = rest.split_last()?;
    let (cmp, ty) = (CmpOp::from_name(first)?, InstrType::from_name(last)?);
    let (mut ftz, mut bool_op) = (false, None);
    for &m in between {
        match BoolOp::from_name(m) {
            Some(op) if bool_op.is_none() => bool_op = Some(op),
            None if m == "ftz" && !ftz => ftz = true,
            _ => return None,
        }
    }
    match ty {
        InstrType::Int(t) if !ftz => Some(Form::IntCompare {
            pred: IntPredicate::of(cmp, t.kind)?,
            width: t.width,
            bool_op,
        }),
        InstrType::Float { lanes } => Some(Form::Float { lanes }),
        _ => None,
    }
}

/// A `cvt`'s types, the two of its modifiers that contain a digit,
/// destination first, when both are integer types, and its `.sat`.
fn conversion(mods: &[&str]) -> Option<Form> {
    let mut sized = mods.iter().filter(|m| m.contains(char::is_numeric));
    let (to, from) = (sized.next()?, sized.next()?);
    let (InstrType::Int(to), InstrType::Int(from)) =
        (InstrType::from_name(to)?, InstrType::from_name(from)?)
    else {
        return None;
    };
    let sat = mods.contains(&"sat");
    sized
        .next()
        .is_none()
        .then_some(Form::IntConvert { from, to, sat })
}

/// A load's or store's state space, semantics and cache policy: `.mmio`
/// over the `.relaxed` it takes (§9.7.9.8), `.L2::cache_hint` for its
/// cache-policy operand.
fn memory(module: &Module, instr: &Instr, mods: &[&str]) -> Memory {
    let has = |m: &str| mods.contains(&m);
    let sem = [
        ("mmio", Sem::Mmio),
        ("volatile", Sem::Volatile),
        ("acquire", Sem::Acquire),
        ("release", Sem::Release),
        ("relaxed", Sem::Relaxed),
    ]
    .into_iter()
    .find_map(|(m, sem)| has(m).then_some(sem))
    .unwrap_or(Sem::Weak);
    Memory {
        space: space(module, instr),
        sem,
        cache_hint: has("L2::cache_hint"),
    }
}

/// The state spaces `instr`'s modifiers name, in order.
fn spaces(module: &Module, instr: &Instr) -> impl Iterator<Item = Space> {
    names(module, instr).filter_map(Space::from_modifier)
}

/// The state space `instr`'s modifiers name first; `None` when they name
/// none, as an `ld` by generic address does (PTX ISA §6.4.1.1).
pub fn space(module: &Module, instr: &Instr) -> Option<Space> {
    spaces(module, instr).next()
}

/// A copy's source and destination, the two state spaces its modifiers
/// name, destination first (`cp.async.ca.shared.global`, PTX ISA
/// §9.7.9.26.3.1).
pub fn copy_spaces(module: &Module, instr: &Instr) -> Option<(Space, Space)> {
    let mut named = spaces(module, instr);
    let (to, from) = (named.next()?, named.next()?);
    named.next().is_none().then_some((from, to))
}

/// What a `ld.param` or `st.param` address refers to (PTX ISA §5.1.6).
/// The `::entry` and `::func` sub-qualifiers, which few producers write,
/// agree with it in valid PTX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamAccess {
    /// `[var]` or `[var+immOff]`, `var` the kernel parameter `index`
    /// (§5.1.6.1, `::entry`); `offset` is `immOff`, 0 for `[var]`.
    KernelParam { index: usize, offset: i64 },
    /// A device function parameter (§5.1.6.4, `::func`): in a `.entry`
    /// body, a locally-scoped `.param` variable a call passes or
    /// returns. Kernel parameters are read-only (§5.1.6.1), and a
    /// `st.param` without sub-qualifier is `st.param::func` (§5.1.6).
    DeviceFunctionParam,
    /// `[reg]` or `[reg+immOff]`: an address register, such as a kernel
    /// parameter's address a `mov` put there (§5.1.6.1, §6.4.1).
    AddressRegister,
    /// `[immAddr]`: an immediate absolute byte address (§6.4.1).
    ImmediateAddress,
}

/// What `instr`, a `.entry` body's statement, accesses in the `.param`
/// state space; `None` when it is not a `ld.param` or `st.param`.
pub(crate) fn param_access(module: &Module, kernel: &Kernel, instr: &Instr) -> Option<ParamAccess> {
    if space(module, instr) != Some(Space::Param) {
        return None;
    }
    match instr.mnemonic {
        Opcode::St => return Some(ParamAccess::DeviceFunctionParam),
        Opcode::Ld => {}
        _ => return None,
    }
    let &address = module.operand_ids(instr.operands).get(1)?;
    let &Operand::Memory { base, offset, .. } = module.operand(address) else {
        return None;
    };
    Some(match module.operand(base) {
        Operand::SymbolRef(var) => match kernel.params.iter().position(|p| p.name == *var) {
            Some(index) => ParamAccess::KernelParam { index, offset },
            None => ParamAccess::DeviceFunctionParam,
        },
        Operand::Immediate(_) => ParamAccess::ImmediateAddress,
        _ => ParamAccess::AddressRegister,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    /// The form of `text`, one instruction.
    fn form_of(text: &str) -> Form {
        let src = format!(
            ".version 8.7\n.target sm_90\n.address_size 64\n\
             .visible .entry k()\n{{\n{text}\n}}\n"
        );
        let m = parse(&src).expect("snippet parses");
        let Some(Stmt::Instr(instr)) = m.kernels[0].stmts.first() else {
            panic!("`{text}` is not an instruction");
        };
        Form::of(&m, instr)
    }

    fn int(name: &str) -> IntInstrType {
        IntInstrType::from_name(name).expect("an integer type")
    }

    /// `Form::Int` of type `name`, `mode`, `.sat` and `.relu`.
    fn int_form(name: &str, mode: Option<MulMode>, sat: bool, relu: bool) -> Form {
        Form::Int {
            ty: int(name),
            mode,
            sat,
            relu,
        }
    }

    fn scalar(name: &str) -> Form {
        int_form(name, None, false, false)
    }

    /// §5.2.1 Table 8, §5.2.2 and §5.2.5: the instruction types.
    #[test]
    fn instruction_types() {
        let float = |lanes| InstrType::Float { lanes };
        for (name, ty) in [
            ("u8", InstrType::Int(int("u8"))),
            ("s64", InstrType::Int(int("s64"))),
            ("b16", InstrType::Int(int("b16"))),
            ("u16x2", InstrType::PackedInt),
            ("s16x2", InstrType::PackedInt),
            ("u8x4", InstrType::PackedInt),
            ("s8x4", InstrType::PackedInt),
            ("f16", float(1)),
            ("bf16", float(1)),
            ("f16x2", float(2)),
            ("bf16x2", float(2)),
            ("f32", float(1)),
            ("f32x2", float(2)),
            ("f64", float(1)),
            ("b128", InstrType::B128),
            ("pred", InstrType::Pred),
        ] {
            assert_eq!(InstrType::from_name(name), Some(ty), "{name}");
        }
        for name in ["tf32", "e4m3", "e4m3x2", "s4", "u16x4", "lo"] {
            assert_eq!(InstrType::from_name(name), None, "{name}");
        }
    }

    /// The type of a scalar integer instruction, and of nothing else.
    #[test]
    fn int_type_is_that_of_scalar_integer_instructions() {
        for (text, ty) in [
            ("add.cc.s32 %r1, %r2, %r3;", Some(int("s32"))),
            ("mul.wide.u16 %r1, %rs2, %rs3;", Some(int("u16"))),
            ("and.b64 %rd1, %rd2, %rd3;", Some(int("b64"))),
            ("shr.u32 %r1, %r2, 1;", Some(int("u32"))),
            ("selp.b32 %r1, %r2, %r3, %p1;", Some(int("b32"))),
            ("mov.u32 %r1, %tid.x;", Some(int("u32"))),
            ("add.s16x2 %r1, %r2, %r3;", None),
            ("add.f32 %f1, %f2, %f3;", None),
            ("or.pred %p1, %p2, %p3;", None),
            ("setp.lt.s32 %p1, %r1, %r2;", None),
            ("cvt.u64.u32 %rd1, %r1;", None),
        ] {
            assert_eq!(form_of(text).int_type(), ty, "{text}");
        }
    }

    /// PTX ISA §9.7.1, §9.7.2: `add{.sat}.s32`, `add{.cc}.u32`,
    /// `mul.mode.type`, `mad.mode{.sat}.type`, `min{.relu}.btype`, and the
    /// packed `add.s16x2` (§9.7.1.1).
    #[test]
    fn integer_arithmetic_reads_its_type_and_modifiers() {
        for (text, expected) in [
            ("add.s32 %r1, %r2, %r3;", scalar("s32")),
            (
                "add.sat.s32 %r1, %r2, %r3;",
                int_form("s32", None, true, false),
            ),
            ("sub.cc.u32 %r1, %r2, %r3;", scalar("u32")),
            (
                "min.relu.s32 %r1, %r2, %r3;",
                int_form("s32", None, false, true),
            ),
            (
                "mul.lo.s32 %r1, %r2, %r3;",
                int_form("s32", Some(MulMode::Lo), false, false),
            ),
            (
                "mul.wide.u16 %r1, %rs2, %rs3;",
                int_form("u16", Some(MulMode::Wide), false, false),
            ),
            (
                "mad.hi.sat.s32 %r1, %r2, %r3, %r4;",
                int_form("s32", Some(MulMode::Hi), true, false),
            ),
            (
                "mad.lo.cc.u64 %rd1, %rd2, %rd3, %rd4;",
                int_form("u64", Some(MulMode::Lo), false, false),
            ),
            ("neg.s16 %rs1, %rs2;", scalar("s16")),
            ("div.u64 %rd1, %rd2, %rd3;", scalar("u64")),
            ("rem.s32 %r1, %r2, %r3;", scalar("s32")),
            ("max.u32 %r1, %r2, %r3;", scalar("u32")),
            ("bfe.u32 %r1, %r2, 8, 4;", scalar("u32")),
            ("add.s16x2 %r1, %r2, %r3;", Form::PackedInt),
            ("min.relu.s16x2 %r1, %r2, %r3;", Form::PackedInt),
            ("max.u16x2 %r1, %r2, %r3;", Form::PackedInt),
        ] {
            assert_eq!(form_of(text), expected, "{text}");
        }
    }

    /// A float type picks the float family, whatever else the
    /// instruction says (§9.7.3 to §9.7.5), scalar or packed.
    #[test]
    fn float_arithmetic_is_one_form() {
        for (text, lanes) in [
            ("add.rn.ftz.f32 %f1, %f2, %f3;", 1),
            ("add.rn.f32.bf16 %f1, %f2, %rs3;", 1),
            ("add.f32x2 %rd1, %rd2, %rd3;", 2),
            ("sub.f64 %fd1, %fd2, %fd3;", 1),
            ("mul.rn.bf16x2 %r1, %r2, %r3;", 2),
            ("mad.rn.f32 %f1, %f2, %f3, %f4;", 1),
            ("div.approx.ftz.f32 %f1, %f2, %f3;", 1),
            ("neg.f16 %rs1, %rs2;", 1),
            ("max.NaN.xorsign.abs.f16x2 %r1, %r2, %r3;", 2),
        ] {
            assert_eq!(form_of(text), Form::Float { lanes }, "{text}");
        }
    }

    /// Modifiers outside the integer syntax: `.lo` of an `add`, two
    /// modes or a repeated flag, a rounding, or no type, `.b128` or
    /// `.pred` for one.
    #[test]
    fn arithmetic_outside_its_syntax_is_other() {
        for text in [
            "add.lo.s32 %r1, %r2, %r3;",
            "mul.lo.hi.s32 %r1, %r2, %r3;",
            "add.sat.sat.s32 %r1, %r2, %r3;",
            "add.rn.s32 %r1, %r2, %r3;",
            "div.full.s32 %r1, %r2, %r3;",
            "add %r1, %r2, %r3;",
            "add.b128 %rq1, %rq2, %rq3;",
            "neg.pred %p1, %p2;",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }

    /// §9.7.8.1 to §9.7.8.4: `and.type d, a, b`, `.type = { .pred, .b16,
    /// .b32, .b64 }`; §9.7.8.8, §9.7.8.9: `shl.type`, `shr.type`, of
    /// integer types.
    #[test]
    fn logic_and_shifts_take_one_type() {
        for (text, expected) in [
            ("and.b32 %r1, %r2, %r3;", scalar("b32")),
            ("or.pred %p1, %p2, %p3;", Form::Pred),
            ("not.b64 %rd1, %rd2;", scalar("b64")),
            ("shl.b32 %r1, %r2, 3;", scalar("b32")),
            ("shr.s64 %rd1, %rd2, 3;", scalar("s64")),
        ] {
            assert_eq!(form_of(text), expected, "{text}");
        }
        for text in [
            "xor %r1, %r2, %r3;",
            "and.sync.b32 %r1, %r2, %r3;",
            "shl.pred %p1, %p2, 1;",
            "shr.f32 %f1, %f2, 1;",
            "shr.wrap.u32 %r1, %r2, 1;",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }

    /// §9.7.6.2, §9.7.7.2: `setp.CmpOp{.ftz}.type p[|q], a, b` and
    /// `setp.CmpOp.BoolOp{.ftz}.type p[|q], a, b, {!}c`, `.ftz` of a float
    /// type only.
    #[test]
    fn comparisons_read_their_predicate_width_and_bool_op() {
        let compare = |pred, width, bool_op| Form::IntCompare {
            pred,
            width,
            bool_op,
        };
        for (text, expected) in [
            (
                "setp.lt.s32 %p1, %r1, %r2;",
                compare(IntPredicate::Slt, 32, None),
            ),
            (
                "setp.hs.u64 %p1, %rd1, %rd2;",
                compare(IntPredicate::Uge, 64, None),
            ),
            (
                "setp.ne.and.b32 %p1, %r1, %r2, %p2;",
                compare(IntPredicate::Ne, 32, Some(BoolOp::And)),
            ),
            (
                "setp.gtu.or.ftz.f32 %p1, %f1, %f2, !%p2;",
                Form::Float { lanes: 1 },
            ),
            ("setp.lt.f16x2 %p1|%p2, %r1, %r2;", Form::Float { lanes: 2 }),
            ("setp.nan.bf16 %p1, %rs1, %rs2;", Form::Float { lanes: 1 }),
        ] {
            assert_eq!(form_of(text), expected, "{text}");
        }
        for text in [
            "setp.s32.lt %p1, %r1, %r2;",
            "setp.lt %p1, %r1, %r2;",
            "setp.lt.ftz.ftz.f32 %p1, %f1, %f2;",
            "setp.lt.and.or.s32 %p1, %r1, %r2, %p2;",
            "setp.lt.sat.s32 %p1, %r1, %r2;",
            "setp.lt.s32.ftz %p1, %r1, %r2;",
            "setp.lt.ftz.s32 %p1, %r1, %r2;",
            "setp.lo.s32 %p1, %r1, %r2;",
            "setp.lt.b32 %p1, %r1, %r2;",
            "setp.equ.s32 %p1, %r1, %r2;",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }

    /// §9.7.6.2: `lt` and `lo` of an unsigned type are one predicate, and
    /// `eq` of every integer type one; each reads its operands as signed,
    /// unsigned or bits and compares the integers by its `Cmp`.
    #[test]
    fn spellings_of_one_comparison_are_one_predicate() {
        let pred = |text: &str| match form_of(&format!("setp.{text} %p1, %r1, %r2;")) {
            Form::IntCompare { pred, .. } => pred,
            other => panic!("{text}: {other:?}"),
        };
        for (texts, expected, reads, cmp) in [
            (
                &["lt.u32", "lo.u32"][..],
                IntPredicate::Ult,
                IntKind::Unsigned,
                Cmp::Lt,
            ),
            (
                &["le.u32", "ls.u32"],
                IntPredicate::Ule,
                IntKind::Unsigned,
                Cmp::Le,
            ),
            (
                &["gt.u64", "hi.u64"],
                IntPredicate::Ugt,
                IntKind::Unsigned,
                Cmp::Gt,
            ),
            (
                &["ge.u16", "hs.u16"],
                IntPredicate::Uge,
                IntKind::Unsigned,
                Cmp::Ge,
            ),
            (&["lt.s32"], IntPredicate::Slt, IntKind::Signed, Cmp::Lt),
            (&["le.s16"], IntPredicate::Sle, IntKind::Signed, Cmp::Le),
            (&["gt.s64"], IntPredicate::Sgt, IntKind::Signed, Cmp::Gt),
            (&["ge.s32"], IntPredicate::Sge, IntKind::Signed, Cmp::Ge),
            (
                &["eq.s32", "eq.u32", "eq.b32"],
                IntPredicate::Eq,
                IntKind::Bits,
                Cmp::Eq,
            ),
            (
                &["ne.s64", "ne.u64", "ne.b64"],
                IntPredicate::Ne,
                IntKind::Bits,
                Cmp::Ne,
            ),
        ] {
            for text in texts {
                assert_eq!(pred(text), expected, "{text}");
            }
            assert_eq!(
                (expected.reads(), expected.cmp()),
                (reads, cmp),
                "{expected:?}"
            );
        }
    }

    /// §9.7.6.3: `selp.type`; §9.7.9.3, §9.7.9.4: `mov.type`, `.pred`
    /// included; a `mov.b128` is no form the analyses read.
    #[test]
    fn selections_and_moves_take_one_type() {
        for (text, expected) in [
            ("selp.b32 %r1, %r2, %r3, %p1;", scalar("b32")),
            ("selp.f32 %f1, %f2, %f3, %p1;", Form::Float { lanes: 1 }),
            ("mov.u32 %r1, %tid.x;", scalar("u32")),
            ("mov.pred %p1, 0;", Form::Pred),
        ] {
            assert_eq!(form_of(text), expected, "{text}");
        }
        for text in [
            "mov %r1, %r2;",
            "mov.u32.u32 %r1, %r2;",
            "selp.sat.s32 %r1, %r2, %r3, %p1;",
            "mov.b128 %rq1, {%rd1, %rd2};",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }

    /// §9.7.9.22: `cvt{.irnd}{.ftz}{.sat}.dtype.atype`, the destination's
    /// type first, of two integer types; a float, a format that is no
    /// instruction type, or a third sized modifier is other.
    #[test]
    fn conversions_read_two_integer_types_and_sat() {
        for (text, from, to, sat) in [
            ("cvt.u64.u32 %rd1, %r1;", "u32", "u64", false),
            ("cvt.sat.s8.s32 %rs1, %r1;", "s32", "s8", true),
        ] {
            let (from, to) = (int(from), int(to));
            assert_eq!(form_of(text), Form::IntConvert { from, to, sat }, "{text}");
        }
        for text in [
            "cvt.rn.f32.s32 %f1, %r1;",
            "cvt.rzi.ftz.s32.f32 %r1, %f1;",
            "cvt.rn.relu.bf16x2.f32 %r1, %f1, %f2;",
            "cvt.rn.satfinite.e4m3x2.f32 %rs1, %f1, %f2;",
            "cvt.rna.tf32.f32 %r1, %f1;",
            "cvt.pack.sat.u8.s32.b32 %r1, %r2, %r3, %r4;",
            "cvt.u32 %r1, %r2;",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }

    /// §9.7.9.5, §9.7.9.6: `shfl{.sync}.mode.b32`.
    #[test]
    fn shuffles_read_their_mode() {
        for (text, mode) in [
            ("shfl.sync.idx.b32 %r1, %r2, 0, 31, -1;", ShflMode::Idx),
            (
                "shfl.sync.bfly.b32 %r1|%p1, %r2, 1, 31, -1;",
                ShflMode::Bfly,
            ),
            ("shfl.up.b32 %r1, %r2, 1, 0;", ShflMode::Up),
            ("shfl.sync.down.b32 %r1, %r2, 1, 31, -1;", ShflMode::Down),
        ] {
            assert_eq!(form_of(text), Form::Shuffle(mode), "{text}");
        }
        assert_eq!(form_of("shfl.sync.b32 %r1, %r2, 0, 31, -1;"), Form::Other);
    }

    /// §9.7.9.8, §9.7.9.10, §9.7.9.11: a memory instruction's space,
    /// `.weak` by default, its other semantics, and `.L2::cache_hint`.
    #[test]
    fn loads_and_stores_read_space_semantics_and_cache_policy() {
        let memory = |space, sem| {
            Form::Memory(Memory {
                space,
                sem,
                cache_hint: false,
            })
        };
        let global = Some(Space::Global);
        for (text, expected) in [
            ("ld.global.u32 %r1, [%rd1];", memory(global, Sem::Weak)),
            ("ld.weak.global.u32 %r1, [%rd1];", memory(global, Sem::Weak)),
            ("ld.u32 %r1, [%rd1];", memory(None, Sem::Weak)),
            ("ldu.global.f32 %f1, [%rd1];", memory(global, Sem::Weak)),
            (
                "ld.volatile.shared.u32 %r1, [%r2];",
                memory(Some(Space::Shared), Sem::Volatile),
            ),
            (
                "ld.relaxed.gpu.global.u32 %r1, [%rd1];",
                memory(global, Sem::Relaxed),
            ),
            (
                "ld.acquire.sys.global.u32 %r1, [%rd1];",
                memory(global, Sem::Acquire),
            ),
            (
                "st.release.cta.global.u32 [%rd1], %r1;",
                memory(global, Sem::Release),
            ),
            (
                "ld.mmio.relaxed.sys.global.u32 %r1, [%rd1];",
                memory(global, Sem::Mmio),
            ),
            (
                "st.param.b32 [p], %r1;",
                memory(Some(Space::Param), Sem::Weak),
            ),
        ] {
            assert_eq!(form_of(text), expected, "{text}");
        }
        assert_eq!(
            form_of("ld.global.L2::cache_hint.u32 %r1, [%rd1], %rd2;"),
            Form::Memory(Memory {
                space: global,
                sem: Sem::Weak,
                cache_hint: true,
            })
        );
    }

    #[test]
    fn other_instructions_are_other() {
        for text in [
            "fma.rn.f32 %f1, %f2, %f3, %f4;",
            "bra $L1;",
            "atom.global.add.u32 %r1, [%rd1], 1;",
        ] {
            assert_eq!(form_of(text), Form::Other, "{text}");
        }
    }
}
