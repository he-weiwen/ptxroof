//! The flat module IR (the ground rule, after Sampson's
//! "Flattening ASTs").
//!
//! No pointer structures anywhere: instructions live in per-kernel
//! `Vec<Stmt>`, operands in a module-level arena referenced by spans,
//! nested operands (memory bases, vector lists) via `OperandId` — never
//! `Box`. Parsing returns an owned module; analyses borrow it immutably by
//! convention, but its public fields permit mutation. Handles are local to
//! their owning module. The PTX printer resolves them for textual output.

use crate::support::index::{IdxRange, IndexVec, newtype_idx};
use crate::support::intern::{Interner, Symbol};
use std::path::Path;

newtype_idx! {
    /// Handle into the [`Module::operands`] arena.
    pub struct OperandId;
}

macro_rules! named_enum {
    ($(#[$m:meta])* $vis:vis enum $name:ident { $($variant:ident = $text:literal),* $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name { $($variant),* }

        impl $name {
            pub fn name(self) -> &'static str {
                match self { $(Self::$variant => $text),* }
            }

            pub fn from_name(text: &str) -> Option<Self> {
                match text { $($text => Some(Self::$variant),)* _ => None }
            }
        }
    };
}

named_enum! {
    /// PTX ISA §5.2.1, Table 8.
    pub enum FundamentalType {
        S8 = "s8", S16 = "s16", S32 = "s32", S64 = "s64",
        U8 = "u8", U16 = "u16", U32 = "u32", U64 = "u64",
        F16 = "f16", F16x2 = "f16x2", F32 = "f32", F64 = "f64",
        B8 = "b8", B16 = "b16", B32 = "b32", B64 = "b64", B128 = "b128",
        Pred = "pred",
    }
}

impl FundamentalType {
    /// `None` for `.pred`: "Predicate variables may only be declared in
    /// the register state space" (PTX ISA §5.4.1).
    pub fn bytes(self) -> Option<u32> {
        use FundamentalType::*;
        match self {
            S8 | U8 | B8 => Some(1),
            S16 | U16 | F16 | B16 => Some(2),
            S32 | U32 | F32 | F16x2 | B32 => Some(4),
            S64 | U64 | F64 | B64 => Some(8),
            B128 => Some(16),
            Pred => None,
        }
    }
}

/// A source location from a `.loc` directive. For the extended form
/// (`.loc f l c, function_name $sym, inlined_at f l c`) this is the
/// `inlined_at` location: attribution wants the user's line, not the
/// header the code was inlined from. `line` 0 is PTX for "no source
/// line" and is skipped at attribution time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceLoc {
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    pub reg: Symbol,
    pub negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    /// `%rd5`, `%tid.x` — register or special-register reference.
    Register(Symbol),
    /// Numeric immediate, original text preserved (`-7`, `0f3F800000`).
    Immediate(Symbol),
    /// Bare identifier: branch-target label, global symbol, `WARP_SZ`.
    SymbolRef(Symbol),
    /// `[base]` / `[base+offset]`.
    Memory { base: OperandId, offset: i64 },
    /// `{a, b, ...}` — vector / fragment list; children are a span into
    /// [`Module::operand_lists`].
    VectorList { children: IdxRange<OperandId> },
    /// `p|q` — an instruction writing several destination registers
    /// (PTX ISA §9.7, "Instructions": `setp.lt.s32 p|q, a, b`).
    MultipleDestinations { children: IdxRange<OperandId> },
}

/// One executable instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instr {
    pub mnemonic: Symbol,
    /// Span into [`Module::modifier_pool`].
    pub modifiers: IdxRange<Symbol>,
    /// Span into [`Module::operand_lists`].
    pub operands: IdxRange<OperandId>,
    pub predicate: Option<Predicate>,
    pub loc: Option<SourceLoc>,
    /// Byte offset of the instruction start: `@` for a predicated
    /// instruction, otherwise the mnemonic. Measurement provenance instead
    /// uses the index in `Kernel::stmts`.
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Instr(Instr),
    Label(Symbol),
    /// `LABEL: .branchtargets L0, L1, ...;` — the jump table for a
    /// preceding `brx.idx`.
    BranchTargets(Vec<Symbol>),
    /// A statement the parser could not understand. Counted by the
    /// corpus-wide parse check; never poisons the kernel.
    Unparsed {
        offset: u32,
    },
}

/// `.param .u32 name` in a kernel signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    pub ty: FundamentalType,
    pub name: Symbol,
}

/// `.reg .f32 %f<789>;` — a counted register family (count is the
/// declared `<N>`), or a singular declaration (`.reg .pred p;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegDecl {
    pub class: FundamentalType,
    pub prefix: Symbol,
    pub count: Option<u32>,
}

/// In-body `.shared .align A .b8 name[N];` (or `.extern` dynamic form
/// with empty brackets — `size` is `None` there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedDecl {
    pub name: Symbol,
    pub align: Option<u32>,
    pub ty: FundamentalType,
    pub size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub reg_decls: Vec<RegDecl>,
    pub shared_decls: Vec<SharedDecl>,
    /// `.maxntid x, y, z` / `.reqntid x, y, z` between signature and body.
    pub maxntid: Option<[u32; 3]>,
    pub reqntid: Option<[u32; 3]>,
    pub stmts: Vec<Stmt>,
}

/// `.file N "path"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileDirective {
    pub index: u32,
    pub path: Symbol,
}

#[derive(Debug)]
pub struct Module {
    pub interner: Interner,
    /// `.version major.minor`.
    pub version: (u32, u32),
    /// `.target` first argument (`sm_80`).
    pub target: Symbol,
    /// `.address_size`, `None` when omitted: the manual's default of 32
    /// (PTX ISA §11.1.3) is not what ptxas assembles, which is 64.
    pub address_size: Option<u32>,
    pub files: Vec<FileDirective>,
    pub kernels: Vec<Kernel>,
    /// Operand arena; `Operand::Memory::base` points here.
    pub operands: IndexVec<OperandId, Operand>,
    /// Pool of operand-id lists; `Instr::operands` and
    /// `Operand::VectorList::children` are spans into it.
    pub operand_lists: Vec<OperandId>,
    /// Pool of modifier symbols; `Instr::modifiers` is a span into it.
    pub modifier_pool: Vec<Symbol>,
    /// Module-scope `.shared` declarations, visible to every kernel:
    /// LLVM emits dynamic shared memory as
    /// `.extern .shared .align 16 .b8 global_smem[];` ahead of the entry.
    pub shared_decls: Vec<SharedDecl>,
}

impl Module {
    pub fn operand(&self, id: OperandId) -> &Operand {
        &self.operands[id]
    }

    pub fn operand_ids(&self, span: IdxRange<OperandId>) -> &[OperandId] {
        &self.operand_lists[span.range()]
    }

    pub fn modifiers(&self, instr: &Instr) -> &[Symbol] {
        &self.modifier_pool[instr.modifiers.range()]
    }

    /// The instruction as PTX spells it: mnemonic plus modifiers,
    /// e.g. `ld.shared.v4.b32`.
    pub fn opcode(&self, instr: &Instr) -> String {
        std::iter::once(instr.mnemonic)
            .chain(self.modifiers(instr).iter().copied())
            .map(|s| self.interner.resolve(s))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Resolve a `.file` index to its path, if declared.
    pub fn file_path(&self, index: u32) -> Option<&str> {
        self.files
            .iter()
            .find(|f| f.index == index)
            .map(|f| self.interner.resolve(f.path))
    }

    /// The last component of a `.file` index's path, if it has one.
    pub fn file_name(&self, index: u32) -> Option<&str> {
        Path::new(self.file_path(index)?).file_name()?.to_str()
    }
}
