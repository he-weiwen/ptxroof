//! Instruction → semantic record.
//!
//! Transcribed from v1's `lib/PTX/Classifier.cpp` (v1 lives only in
//! git history now, last at 690d81d): cuda-core, tensor-core and SFU
//! flops, non-flop arithmetic (conversions counted separately — they
//! are the precision-audit overhead S8 reports), memory, copies that
//! read one space and write another, sync, control, ignore, unknown.
//! What has no arm (Hopper bulk copies and warpgroup MMA, integer and
//! sparse MMA kinds, textures) classifies as `Unknown`: visible,
//! counted, and policed by the corpus coverage check — never silently
//! zero. `docs/ptx-instruction-coverage.md` walks the ISA chapter row
//! by row.
//!
//! Two deliberate divergences from the v1 reference, both documented
//! at the match arm:
//! - an arithmetic mnemonic is a flop only when it carries an FP type
//!   modifier; `mad.lo.s32`/`add.s64` address arithmetic is non-flop
//!   (v1 gave integer `mad` 2 flops of precision "Other");
//! - `div`/`sqrt`/`rcp`/... are special-function-unit work: one flop
//!   per invocation on the `Sfu` pipe, never mixed into the cuda-core
//!   table (v1 counted them as ordinary 1-flop arithmetic).
//!
//! FLOPs convention (Williams roofline accounting, as v1): fma/mad = 2
//! per invocation, add/sub/mul/min/max/abs/neg/copysign = 1, the SFU
//! family = 1 per result (the operation, not its SASS expansion), all
//! multiplied by packed lanes (`f16x2` = ×2).

use crate::ptx::ir::{Instr, Module, Operand};
use crate::ptx::literal::parse_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Precision {
    F16,
    BF16,
    /// Tensor-core only (PTX ISA §9.7.15.2): 32-bit storage, 10-bit
    /// mantissa.
    TF32,
    F32,
    F64,
    /// Tensor-core only: the 8-bit `.e4m3` and `.e5m2` multiplicands
    /// (PTX ISA §9.7.15.5.1, `mma` with `.kind::f8f6f4` shapes).
    FP8,
}

impl Precision {
    pub const ALL: [Precision; 6] = [
        Precision::FP8,
        Precision::F16,
        Precision::BF16,
        Precision::TF32,
        Precision::F32,
        Precision::F64,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
            Precision::TF32 => "tf32",
            Precision::F32 => "f32",
            Precision::F64 => "f64",
            Precision::FP8 => "fp8",
        }
    }
}

/// The execution unit a flop runs on; each gets its own flop table
/// in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pipe {
    CudaCore,
    Tensor,
    /// Special function unit: transcendentals, reciprocals, roots.
    Sfu,
}

impl Pipe {
    pub const ALL: [Pipe; 3] = [Pipe::CudaCore, Pipe::Tensor, Pipe::Sfu];

    pub fn key(self) -> &'static str {
        match self {
            Pipe::CudaCore => "cuda-core",
            Pipe::Tensor => "tensor",
            Pipe::Sfu => "sfu",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Load,
    Store,
}

/// Non-flop arithmetic, split because conversions are first-class in a
/// precision audit (S8: 8 `cvt` per k2 main-loop iteration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArithKind {
    /// `cvt` — precision/width conversion.
    Conversion,
    /// Integer/bit arithmetic: address math, masks, shifts.
    Integer,
    /// Predicate computation and selection: `setp`, `selp`, `set`...
    Predicate,
    /// Register/address bookkeeping: `mov`, `cvta`.
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpClass {
    /// Cuda-core floating-point work. `flops` already includes the
    /// base convention and packed lanes.
    Flop {
        pipe: Pipe,
        precision: Precision,
        flops: u32,
    },
    NonFlopArith {
        kind: ArithKind,
    },
    /// `bytes` is per-thread per-execution; `None` = statically
    /// unquantifiable (surfaces in the unquantified counter, never 0).
    Memory {
        space: Space,
        direction: Direction,
        bytes: Option<u32>,
    },
    /// One instruction that both reads and writes memory: a read of
    /// `read_bytes` from `from` and a write of `written_bytes` to `to`,
    /// each per thread per execution, each `None` when unquantifiable.
    Copy {
        from: Space,
        to: Space,
        read_bytes: Option<u32>,
        written_bytes: Option<u32>,
    },
    /// Barriers, fences, and waits (PTX ISA §9.7.14, "Synchronization").
    Sync,
    /// Lane-to-lane register exchange and warp votes (§9.7.14,
    /// "Communication"): shuffles, votes, `match`, `redux`, `elect`.
    Communication,
    /// Branches, returns, calls.
    Control,
    /// Correctly contributes nothing (`nop`, cache hints).
    Ignore,
    /// Not yet handled — counted and named, never dropped.
    Unknown,
}

pub fn classify(module: &Module, instr: &Instr) -> OpClass {
    let interner = &module.interner;
    let mnemonic = interner.resolve(instr.mnemonic);
    let mods: Vec<&str> = module
        .modifiers(instr)
        .iter()
        .map(|&m| interner.resolve(m))
        .collect();

    let fp = fp_precision_and_lanes(&mods);

    match mnemonic {
        // -- cuda-core flops (FP type modifier required) -----------------
        "fma" | "mad" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: p,
                flops: 2 * lanes,
            },
            // mad.lo.s32 and friends are address arithmetic, not flops
            // (divergence from v1, which counted 2 "Other" flops here).
            None => OpClass::NonFlopArith {
                kind: ArithKind::Integer,
            },
        },
        "add" | "sub" | "mul" | "min" | "max" | "abs" | "neg" | "copysign" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: p,
                flops: lanes,
            },
            None => OpClass::NonFlopArith {
                kind: ArithKind::Integer,
            },
        },

        // -- special function unit (PTX ISA §9.7.3.8, .13–.22; §9.7.4.9–10) --
        "rcp" | "sqrt" | "rsqrt" | "sin" | "cos" | "lg2" | "ex2" | "tanh" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                pipe: Pipe::Sfu,
                precision: p,
                flops: lanes,
            },
            None => OpClass::Unknown,
        },
        "div" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                pipe: Pipe::Sfu,
                precision: p,
                flops: lanes,
            },
            None => OpClass::NonFlopArith {
                kind: ArithKind::Integer,
            },
        },

        // -- non-flop arithmetic ------------------------------------------
        "cvt" => OpClass::NonFlopArith {
            kind: ArithKind::Conversion,
        },
        "mov" | "cvta" | "mapa" | "stacksave" | "stackrestore" | "alloca" => {
            OpClass::NonFlopArith {
                kind: ArithKind::Move,
            }
        }
        "setp" | "selp" | "set" | "slct" | "testp" | "isspacep" | "istypep" => {
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate,
            }
        }
        "and" | "or" | "xor" | "not" | "shl" | "shr" | "shf" | "lop3" | "bfe" | "bfi" | "brev"
        | "popc" | "clz" | "bfind" | "bmsk" | "szext" | "prmt" | "rem" | "sad" | "dp4a"
        | "dp2a" | "mul24" | "mad24" | "addc" | "subc" | "madc" | "cnot" | "fns" | "clmad"
        | "getctarank" => {
            if mods.contains(&"pred") {
                OpClass::NonFlopArith {
                    kind: ArithKind::Predicate,
                }
            } else {
                OpClass::NonFlopArith {
                    kind: ArithKind::Integer,
                }
            }
        }

        // -- memory -----------------------------------------------------------
        "ld" | "st" => {
            let direction = if mnemonic == "ld" {
                Direction::Load
            } else {
                Direction::Store
            };
            OpClass::Memory {
                space: space_of(&mods),
                direction,
                bytes: bytes_of(&mods),
            }
        }
        // Uniform global loads (PTX ISA §9.7.9.9).
        "ldu" => OpClass::Memory {
            space: Space::Global,
            direction: Direction::Load,
            bytes: bytes_of(&mods),
        },

        // -- tensor cores (warp-collective; counts are per lane) ----------
        "wmma" => wmma(&mods),
        "mma" => mma(&mods),
        "ldmatrix" => matrix_fragments(&mods, Direction::Load),
        "stmatrix" => matrix_fragments(&mods, Direction::Store),

        // -- asynchronous copies -----------------------------------------------
        "cp" => cp(module, instr, &mods),

        // -- atomics (PTX ISA §9.7.14.5–6): a read-modify-write of one
        // location; the arithmetic is not counted as flops -------------
        "atom" => OpClass::Copy {
            from: space_of(&mods),
            to: space_of(&mods),
            read_bytes: bytes_of(&mods),
            written_bytes: bytes_of(&mods),
        },
        "red" => OpClass::Memory {
            space: space_of(&mods),
            direction: Direction::Store,
            bytes: bytes_of(&mods),
        },

        // -- sync / warp collectives ---------------------------------------
        "bar" | "barrier" | "mbarrier" | "membar" | "fence" => OpClass::Sync,
        "shfl" | "vote" | "match" | "activemask" | "redux" | "elect" => OpClass::Communication,

        // -- control -----------------------------------------------------------
        "bra" | "brx" | "ret" | "exit" | "call" | "trap" | "brkpt" => OpClass::Control,

        // -- correctly ignored -------------------------------------------------
        "nop" | "prefetch" | "prefetchu" | "discard" | "applypriority" | "griddepcontrol"
        | "createpolicy" | "nanosleep" | "pmevent" | "setmaxnreg" => OpClass::Ignore,

        // -- everything else: unsupported instruction families ----------
        // (tensor/wmma/mma/ldmatrix, cp.async, atom/red, SFU
        // transcendentals incl. div/sqrt/rcp/sin/cos/ex2/lg2/tanh/rsqrt,
        // tex/surf). Counted and named by the coverage check.
        _ => OpClass::Unknown,
    }
}

// PTX ISA §4.5.1: "The predefined integer constant WARP_SZ specifies
// the number of threads per warp for the target platform; to date, all
// target architectures have a WARP_SZ value of 32."
const WARP_LANES: u32 = 32;

/// `m16n8k16` → (16, 8, 16).
fn matrix_shape(modifier: &str) -> Option<(u32, u32, u32)> {
    let (m, rest) = modifier.strip_prefix('m')?.split_once('n')?;
    let (n, k) = rest.split_once('k')?;
    Some((m.parse().ok()?, n.parse().ok()?, k.parse().ok()?))
}

/// `m8n8` → (8, 8).
fn matrix_rows_cols(modifier: &str) -> Option<(u32, u32)> {
    let (rows, cols) = modifier.strip_prefix('m')?.split_once('n')?;
    Some((rows.parse().ok()?, cols.parse().ok()?))
}

/// Bits per matrix element for the tensor-core element types
/// (PTX ISA §9.7.15.2, Matrix Data-types) and the `ldmatrix` /
/// `stmatrix` fragment types.
fn element_bits(ty: &str) -> Option<u32> {
    Some(match ty {
        "b1" => 1,
        "s4" | "u4" => 4,
        "s8" | "u8" | "b8" | "e4m3" | "e5m2" => 8,
        "f16" | "bf16" | "b16" => 16,
        "tf32" | "f32" | "s32" => 32,
        "f64" => 64,
        _ => return None,
    })
}

fn tensor_precision(ty: &str) -> Option<Precision> {
    Some(match ty {
        "f16" => Precision::F16,
        "bf16" => Precision::BF16,
        "tf32" => Precision::TF32,
        "f64" => Precision::F64,
        "e4m3" | "e5m2" => Precision::FP8,
        _ => return None,
    })
}

/// One lane's share of a warp-wide byte count, exact or nothing.
fn per_lane_bytes(bits: u32) -> Option<u32> {
    bits.is_multiple_of(8 * WARP_LANES)
        .then_some(bits / (8 * WARP_LANES))
}

/// `wmma.{load,store,mma}` (PTX ISA §9.7.15.4.3–5): the role is the
/// first modifier; the shape modifier fixes every fragment's size and
/// the flop count; the element types follow the shape.
fn wmma(mods: &[&str]) -> OpClass {
    let Some(shape_at) = mods.iter().position(|m| matrix_shape(m).is_some()) else {
        return OpClass::Unknown;
    };
    let (m, n, k) = matrix_shape(mods[shape_at]).expect("position found a shape");
    let types: Vec<&str> = mods[shape_at + 1..]
        .iter()
        .copied()
        .filter(|t| element_bits(t).is_some())
        .collect();
    match (mods.first(), mods.get(1)) {
        (Some(&role @ ("load" | "store")), Some(&matrix)) => {
            let elements = match matrix {
                "a" => m * k,
                "b" => k * n,
                "c" | "d" => m * n,
                _ => return OpClass::Unknown,
            };
            let Some(bits) = types.last().and_then(|t| element_bits(t)) else {
                return OpClass::Unknown;
            };
            OpClass::Memory {
                space: space_of(mods),
                direction: if role == "load" {
                    Direction::Load
                } else {
                    Direction::Store
                },
                bytes: per_lane_bytes(elements * bits),
            }
        }
        (Some(&"mma"), _) => {
            // §9.7.15.4.5: "For wmma.mma without explicit .atype and
            // .btype: .atype and .btype are implicitly set to .f16."
            let atype = if types.len() == 2 {
                "f16"
            } else {
                types.get(1).copied().unwrap_or("")
            };
            match tensor_precision(atype) {
                Some(precision) => OpClass::Flop {
                    pipe: Pipe::Tensor,
                    precision,
                    flops: 2 * m * n * k / WARP_LANES,
                },
                None => OpClass::Unknown,
            }
        }
        _ => OpClass::Unknown,
    }
}

/// `mma.sync` (PTX ISA §9.7.15.5.14): D = A·B + C once per warp for
/// the shape modifier; the types follow the shape as `.dtype.atype
/// .btype.ctype`. Sparse (`.sp`) and block-scaled forms are left
/// Unknown: their flop count is a convention still to be chosen.
fn mma(mods: &[&str]) -> OpClass {
    if mods
        .iter()
        .any(|m| m.starts_with("sp") || *m == "block_scale")
    {
        return OpClass::Unknown;
    }
    let Some(shape_at) = mods.iter().position(|m| matrix_shape(m).is_some()) else {
        return OpClass::Unknown;
    };
    let (m, n, k) = matrix_shape(mods[shape_at]).expect("position found a shape");
    let types: Vec<&str> = mods[shape_at + 1..]
        .iter()
        .copied()
        .filter(|t| element_bits(t).is_some())
        .collect();
    let Some(precision) = types.get(1).and_then(|t| tensor_precision(t)) else {
        return OpClass::Unknown;
    };
    // §9.7.15.5.1: "A warp executing mma.m8n8k4 with .f16 floating point
    // type will compute 4 MMA operations of shape .m8n8k4"; §9.7.15.5.2:
    // the .f64 form computes one.
    let operations = if (m, n, k) == (8, 8, 4) && precision == Precision::F16 {
        4
    } else {
        1
    };
    OpClass::Flop {
        pipe: Pipe::Tensor,
        precision,
        flops: operations * 2 * m * n * k / WARP_LANES,
    }
}

/// `ldmatrix` / `stmatrix` (PTX ISA §9.7.15.5.15–16): `.num` matrices
/// of the shape modifier, always in shared memory ("If no state space
/// is provided, generic addressing is used, such that the address in p
/// points into .shared space"). The padded 6-/4-bit source formats
/// have no element width here and stay unquantified.
fn matrix_fragments(mods: &[&str], direction: Direction) -> OpClass {
    let shape = mods.iter().find_map(|m| matrix_rows_cols(m));
    let count = mods.iter().find_map(|m| match *m {
        "x1" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        _ => None,
    });
    let (Some((rows, cols)), Some(count)) = (shape, count) else {
        return OpClass::Unknown;
    };
    let bits = mods.iter().rev().find_map(|m| element_bits(m));
    OpClass::Memory {
        space: Space::Shared,
        direction,
        bytes: bits.and_then(|b| per_lane_bytes(count * rows * cols * b)),
    }
}

/// `cp.async` (PTX ISA §9.7.9.26.3.1): "Operand src specifies a
/// location in the global state space and dst specifies a location
/// in the shared state space"; `cp-size` is an immediate, and the
/// optional immediate `src-size` is the number of bytes actually
/// read ("remaining bytes in destination dst are filled with
/// zeros"). The group bookkeeping instructions (§9.7.9.26.3.2–3) and
/// the mbarrier arrive are synchronization; the bulk and tensor
/// forms (§9.7.9.26.4–5) are not modeled.
fn cp(module: &Module, instr: &Instr, mods: &[&str]) -> OpClass {
    let has = |name: &str| mods.contains(&name);
    if !has("async") || has("bulk") || has("reduce") {
        return OpClass::Unknown;
    }
    if has("commit_group") || has("wait_group") || has("wait_all") || has("mbarrier") {
        return OpClass::Sync;
    }
    if !(has("shared") || has("shared::cta")) || !has("global") {
        return OpClass::Unknown;
    }
    let immediates: Vec<u32> = module
        .operand_ids(instr.operands)
        .iter()
        .skip(2)
        .filter_map(|&id| match module.operand(id) {
            Operand::Immediate(text) => {
                parse_int(module.interner.resolve(*text)).and_then(|v| u32::try_from(v).ok())
            }
            _ => None,
        })
        .collect();
    OpClass::Copy {
        from: Space::Global,
        to: Space::Shared,
        read_bytes: immediates.get(1).or(immediates.first()).copied(),
        written_bytes: immediates.first().copied(),
    }
}

/// FP precision + packed-lane multiplier from type modifiers.
/// `bf16` is checked before `f16` deliberately (v1 had to dodge the
/// substring trap; exact matching keeps the order only for clarity).
fn fp_precision_and_lanes(mods: &[&str]) -> Option<(Precision, u32)> {
    for &m in mods {
        let hit = match m {
            "bf16" => Some((Precision::BF16, 1)),
            "bf16x2" => Some((Precision::BF16, 2)),
            "f16" => Some((Precision::F16, 1)),
            "f16x2" => Some((Precision::F16, 2)),
            "f32" => Some((Precision::F32, 1)),
            "f32x2" => Some((Precision::F32, 2)),
            "f64" => Some((Precision::F64, 1)),
            _ => None,
        };
        if hit.is_some() {
            return hit;
        }
    }
    None
}

fn space_of(mods: &[&str]) -> Space {
    for &m in mods {
        match m {
            "global" => return Space::Global,
            "shared" | "shared::cta" => return Space::Shared,
            "shared::cluster" => return Space::SharedCluster,
            "local" => return Space::Local,
            "const" => return Space::Const,
            "param" => return Space::Param,
            _ => {}
        }
    }
    Space::Generic
}

/// Per-execution byte count: type width × vector multiplier.
fn bytes_of(mods: &[&str]) -> Option<u32> {
    let mut width: Option<u32> = None;
    let mut vec = 1u32;
    for &m in mods {
        match m {
            "b8" | "u8" | "s8" | "e4m3" | "e5m2" => width = Some(1),
            "b16" | "u16" | "s16" | "f16" | "bf16" => width = Some(2),
            "b32" | "u32" | "s32" | "f32" | "f16x2" | "bf16x2" => width = Some(4),
            "b64" | "u64" | "s64" | "f64" | "f32x2" => width = Some(8),
            "b128" => width = Some(16),
            "v2" => vec = 2,
            "v4" => vec = 4,
            "v8" => vec = 8,
            _ => {}
        }
    }
    width.map(|w| w * vec)
}

/// Byte width of a single PTX fundamental type token (`b8`, `f32`,
/// `b128`, …) — the per-element size used to size `.shared` array
/// declarations. Thin wrapper over [`bytes_of`]; array element types
/// never carry the vector multipliers `bytes_of` also handles.
pub(crate) fn type_width(ty: &str) -> Option<u32> {
    bytes_of(&[ty])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::ir::Stmt;
    use crate::ptx::parse::parser::parse;

    /// Classify the single instruction in `text`.
    fn class_of(text: &str) -> OpClass {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{text}\n}}\n"
        );
        let m = parse(&src).expect("snippet parses");
        let instr = m.kernels[0]
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Instr(i) => Some(*i),
                _ => None,
            })
            .expect("one instruction");
        classify(&m, &instr)
    }

    #[test]
    fn flops_follow_the_fma_2_convention_and_packed_lanes() {
        assert_eq!(
            class_of("fma.rn.f32 %f1, %f2, %f3, %f4;"),
            OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: Precision::F32,
                flops: 2
            }
        );
        assert_eq!(
            class_of("fma.rn.f16x2 %r1, %r2, %r3, %r4;"),
            OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: Precision::F16,
                flops: 4
            }
        );
        assert_eq!(
            class_of("add.f32 %f1, %f2, %f3;"),
            OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: Precision::F32,
                flops: 1
            }
        );
        assert_eq!(
            class_of("mul.bf16x2 %r1, %r2, %r3;"),
            OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: Precision::BF16,
                flops: 2
            }
        );
        assert_eq!(
            class_of("max.f64 %fd1, %fd2, %fd3;"),
            OpClass::Flop {
                pipe: Pipe::CudaCore,
                precision: Precision::F64,
                flops: 1
            }
        );
    }

    #[test]
    fn integer_arithmetic_is_not_a_flop() {
        // The divergence-from-v1 case: integer mad is address math.
        assert_eq!(
            class_of("mad.lo.s32 %r1, %r2, %r3, %r4;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
        assert_eq!(
            class_of("add.s64 %rd1, %rd2, 8;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
        assert_eq!(
            class_of("mul.wide.s32 %rd1, %r1, 2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
    }

    #[test]
    fn conversions_are_their_own_kind() {
        assert_eq!(
            class_of("cvt.f32.f16 %f1, %rs1;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Conversion
            }
        );
        assert_eq!(
            class_of("cvt.rn.f16.f32 %rs1, %f1;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Conversion
            }
        );
        // cvta is an address-space cast, not a value conversion.
        assert_eq!(
            class_of("cvta.to.global.u64 %rd1, %rd2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Move
            }
        );
    }

    #[test]
    fn predicate_ops_including_or_pred() {
        assert_eq!(
            class_of("setp.lt.s32 %p1, %r1, %r2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate
            }
        );
        assert_eq!(
            class_of("or.pred %p3, %p1, %p2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate
            }
        );
        assert_eq!(
            class_of("and.b32 %r1, %r2, 3;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
    }

    #[test]
    fn memory_width_space_and_direction() {
        assert_eq!(
            class_of("ld.global.u16 %rs1, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(2)
            }
        );
        assert_eq!(
            class_of("st.shared.u16 [%r1], %rs1;"),
            OpClass::Memory {
                space: Space::Shared,
                direction: Direction::Store,
                bytes: Some(2)
            }
        );
        assert_eq!(
            class_of("ld.param.u64 %rd1, [k_param_4];"),
            OpClass::Memory {
                space: Space::Param,
                direction: Direction::Load,
                bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("ld.global.v2.f32 {%f1, %f2}, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("ld.global.nc.L2::128B.b128 {%rd1, %rd2}, [%rd3];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(16)
            }
        );
    }

    #[test]
    fn hopper_and_extended_precision_bookkeeping_have_arms() {
        let integer = OpClass::NonFlopArith {
            kind: ArithKind::Integer,
        };
        for text in [
            "addc.cc.u32 %r1, %r2, %r3;",
            "subc.u32 %r1, %r2, %r3;",
            "madc.hi.cc.u32 %r1, %r2, %r3, %r4;",
            "shf.l.wrap.b32 %r1, %r2, %r3, %r4;",
            "cnot.b32 %r1, %r2;",
            "fns.b32 %r1, %r2, %r3, %r4;",
            "clmad.lo.u64 %rd1, %rd2, %rd3, %rd4;",
            "getctarank.shared::cluster.u32 %r1, %r2;",
        ] {
            assert_eq!(class_of(text), integer, "{text}");
        }
        assert_eq!(
            class_of("istypep.texref %p1, %rd1;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate
            }
        );
        for text in [
            "mapa.shared::cluster.u32 %r1, %r2, %r3;",
            "stacksave.u64 %rd1;",
            "alloca.u64 %rd1, %rd2, 16;",
        ] {
            assert_eq!(
                class_of(text),
                OpClass::NonFlopArith {
                    kind: ArithKind::Move
                },
                "{text}"
            );
        }
        assert_eq!(class_of("elect.sync %r1|%p1, %r2;"), OpClass::Communication);
        for text in [
            "createpolicy.fractional.L2::evict_last.b64 %rd1, 0.25;",
            "nanosleep.u32 100;",
            "pmevent 3;",
            "setmaxnreg.inc.sync.aligned.u32 232;",
        ] {
            assert_eq!(class_of(text), OpClass::Ignore, "{text}");
        }
    }

    #[test]
    fn generic_and_cluster_spaces_are_distinct_buckets() {
        assert_eq!(
            class_of("ld.f32 %f1, [%rd1];"),
            OpClass::Memory {
                space: Space::Generic,
                direction: Direction::Load,
                bytes: Some(4)
            }
        );
        assert_eq!(
            class_of("st.shared::cluster.b32 [%r1], %r2;"),
            OpClass::Memory {
                space: Space::SharedCluster,
                direction: Direction::Store,
                bytes: Some(4)
            }
        );
        assert_eq!(
            class_of("ld.shared::cta.b32 %r1, [%r2];"),
            OpClass::Memory {
                space: Space::Shared,
                direction: Direction::Load,
                bytes: Some(4)
            }
        );
    }

    #[test]
    fn sync_control_ignore() {
        assert_eq!(class_of("bar.sync 0;"), OpClass::Sync);
        assert_eq!(class_of("membar.gl;"), OpClass::Sync);
        assert_eq!(
            class_of("shfl.sync.bfly.b32 %r1, %r2, %r3, %r4, %r5;"),
            OpClass::Communication
        );
        assert_eq!(
            class_of("vote.sync.any.pred %p1, %p2, %r1;"),
            OpClass::Communication
        );
        assert_eq!(
            class_of("redux.sync.add.s32 %r1, %r2, %r3;"),
            OpClass::Communication
        );
        assert_eq!(class_of("@%p1 bra $L__X;"), OpClass::Control);
        assert_eq!(class_of("ret;"), OpClass::Control);
        assert_eq!(class_of("nop;"), OpClass::Ignore);
    }

    #[test]
    fn wmma_loads_and_stores_are_fragment_bytes() {
        // k11 (sm_80) forms. Per lane: an m16n16k16 f16 A fragment is
        // 16*16*2 B / 32 = 16 B; the f32 accumulator 16*16*4 / 32 = 32 B.
        let load = "wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%r9], %r10;";
        assert_eq!(
            class_of(load),
            OpClass::Memory {
                space: Space::Shared,
                direction: Direction::Load,
                bytes: Some(16)
            }
        );
        assert_eq!(
            class_of(
                "wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, [%rd1], %r1;"
            ),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(32)
            }
        );
        assert_eq!(
            class_of(
                "wmma.store.d.sync.aligned.row.m16n16k16.global.f16 [%rd1], {%r1, %r2, %r3, %r4}, %r5;"
            ),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Store,
                bytes: Some(16)
            }
        );
        // m8n32k16: B is 16x32 bf16 = 1024 B per warp = 32 B per lane.
        assert_eq!(
            class_of("wmma.load.b.sync.aligned.col.m8n32k16.global.bf16 {%r1}, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(32)
            }
        );
    }

    #[test]
    fn wmma_mma_is_tensor_flops_by_multiplicand_type() {
        // One m16n16k16 MMA per warp: 2*16*16*16 / 32 = 256 flops per lane;
        // `.f32.f32` alone means f16 multiplicands (§9.7.15.4.5).
        assert_eq!(
            class_of(
                "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, {%r9, %r10, %r11, %r12, %r13, %r14, %r15, %r16}, {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8};"
            ),
            OpClass::Flop {
                pipe: Pipe::Tensor,
                precision: Precision::F16,
                flops: 256
            }
        );
        assert_eq!(
            class_of(
                "wmma.mma.sync.aligned.row.col.m8n32k16.f32.bf16.bf16.f32 {%f1}, {%r1}, {%r2}, {%f2};"
            ),
            OpClass::Flop {
                pipe: Pipe::Tensor,
                precision: Precision::BF16,
                flops: 256
            }
        );
        assert_eq!(
            class_of(
                "wmma.mma.sync.aligned.row.col.m8n8k4.rn.f64.f64.f64.f64 {%fd1}, {%fd2}, {%fd3}, {%fd4};"
            ),
            OpClass::Flop {
                pipe: Pipe::Tensor,
                precision: Precision::F64,
                flops: 16
            }
        );
        // Integer MMA is not floating-point work: loud, not zero.
        assert_eq!(
            class_of(
                "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 {%r1}, {%r2}, {%r3}, {%r4};"
            ),
            OpClass::Unknown
        );
    }

    #[test]
    fn mma_sync_flops_by_shape_and_multiplicand_type() {
        let tensor = |precision, flops| OpClass::Flop {
            pipe: Pipe::Tensor,
            precision,
            flops,
        };
        // k14's form: 2*16*8*16 / 32 = 128 flops per lane.
        assert_eq!(
            class_of(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};"
            ),
            tensor(Precision::F16, 128)
        );
        assert_eq!(
            class_of(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f1}, {%r1}, {%r2}, {%f2};"
            ),
            tensor(Precision::TF32, 64)
        );
        assert_eq!(
            class_of(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1}, {%r1}, {%r2}, {%f2};"
            ),
            tensor(Precision::BF16, 128)
        );
        // m8n8k4: four MMAs per warp with f16, one with f64.
        assert_eq!(
            class_of("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%f1}, {%r1}, {%r2}, {%f2};"),
            tensor(Precision::F16, 64)
        );
        assert_eq!(
            class_of(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1}, {%fd2}, {%fd3}, {%fd4};"
            ),
            tensor(Precision::F64, 16)
        );
        // The Gluon fp8 GEMM's form: 2*16*8*32 / 32 = 256 flops per lane.
        assert_eq!(
            class_of(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%r1,%r2,%r3,%r4}, {%r5,%r6,%r7,%r8}, {%r9,%r10}, {%r1,%r2,%r3,%r4};"
            ),
            tensor(Precision::FP8, 256)
        );
        assert_eq!(
            class_of(
                "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32 {%f1}, {%r1}, {%r2}, {%f2};"
            ),
            tensor(Precision::FP8, 128)
        );
        for text in [
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%r1}, {%r2}, {%r3}, {%r4};",
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%f1}, {%r1}, {%r2}, {%f2}, %r3, 0;",
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 {%f1}, {%r1}, {%r2}, {%f2}, %r3, {0, 0}, %r4, {0, 0};",
        ] {
            assert_eq!(class_of(text), OpClass::Unknown, "{text}");
        }
    }

    #[test]
    fn ldmatrix_and_stmatrix_move_num_fragments_of_shared_memory() {
        let shared = |direction, bytes| OpClass::Memory {
            space: Space::Shared,
            direction,
            bytes,
        };
        // k14's forms: an 8x8 b16 matrix is 128 B per warp, 4 B per lane.
        assert_eq!(
            class_of("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];"),
            shared(Direction::Load, Some(16))
        );
        assert_eq!(
            class_of("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r1, %r2}, [%r3];"),
            shared(Direction::Load, Some(8))
        );
        assert_eq!(
            class_of("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%r1, %r2}, [%r3];"),
            shared(Direction::Load, Some(8))
        );
        assert_eq!(
            class_of("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r1], {%r2};"),
            shared(Direction::Store, Some(4))
        );
        // Padded 6-bit source data: width is not an element width.
        assert_eq!(
            class_of("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%r1}, [%r2];"),
            shared(Direction::Load, None)
        );
    }

    #[test]
    fn cp_async_is_a_global_read_and_a_shared_write() {
        // Triton writes the sizes in hex, with spaces inside the brackets.
        assert_eq!(
            class_of("cp.async.cg.shared.global [ %r24 + 0 ], [ %rd5 + 0 ], 0x10, 0x10;"),
            class_of("cp.async.cg.shared.global [%r20], [%rd11], 16, 16;")
        );
        let copy = |read_bytes, written_bytes| OpClass::Copy {
            from: Space::Global,
            to: Space::Shared,
            read_bytes: Some(read_bytes),
            written_bytes: Some(written_bytes),
        };
        // k12's form: nvcc writes the source size explicitly.
        assert_eq!(
            class_of("cp.async.cg.shared.global [%r20], [%rd11], 16, 16;"),
            copy(16, 16)
        );
        assert_eq!(
            class_of("cp.async.ca.shared::cta.global.L2::128B [%r1+8], [%rd1], 4;"),
            copy(4, 4)
        );
        // Zero-fill form: 8 bytes read, 16 written.
        assert_eq!(
            class_of("cp.async.cg.shared.global [%r1], [%rd1], 16, 8;"),
            copy(8, 16)
        );
        // ignore-src predicate and cache policy are not sizes.
        assert_eq!(
            class_of("cp.async.cg.shared.global.L2::cache_hint [%r1], [%rd1], 16, %p1, %rd2;"),
            copy(16, 16)
        );
        for text in [
            "cp.async.commit_group;",
            "cp.async.wait_group 1;",
            "cp.async.wait_all;",
            "cp.async.mbarrier.arrive.shared.b64 [%r1];",
        ] {
            assert_eq!(class_of(text), OpClass::Sync, "{text}");
        }
        for text in [
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1], 1024, [%r2];",
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%rd1], [%r1], 256;",
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3}], [%r4];",
        ] {
            assert_eq!(class_of(text), OpClass::Unknown, "{text}");
        }
    }

    #[test]
    fn sfu_family_is_one_flop_per_result_on_its_own_pipe() {
        let sfu = |precision, flops| OpClass::Flop {
            pipe: Pipe::Sfu,
            precision,
            flops,
        };
        assert_eq!(
            class_of("ex2.approx.ftz.f32 %f1, %f2;"),
            sfu(Precision::F32, 1)
        );
        assert_eq!(
            class_of("ex2.approx.f16x2 %r1, %r2;"),
            sfu(Precision::F16, 2)
        );
        assert_eq!(
            class_of("rsqrt.approx.f64 %fd1, %fd2;"),
            sfu(Precision::F64, 1)
        );
        assert_eq!(
            class_of("div.rn.f32 %f1, %f2, %f3;"),
            sfu(Precision::F32, 1)
        );
        assert_eq!(
            class_of("div.full.f32 %f1, %f2, %f3;"),
            sfu(Precision::F32, 1)
        );
        assert_eq!(class_of("sqrt.rn.f64 %fd1, %fd2;"), sfu(Precision::F64, 1));
        assert_eq!(
            class_of("tanh.approx.bf16x2 %r1, %r2;"),
            sfu(Precision::BF16, 2)
        );
        // Integer division (§9.7.1.9) is address arithmetic, like rem.
        assert_eq!(
            class_of("div.u32 %r1, %r2, %r3;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
    }

    #[test]
    fn atomics_are_bytes_both_ways_reductions_store_only() {
        assert_eq!(
            class_of("atom.global.add.u32 %r1, [%rd1], %r2;"),
            OpClass::Copy {
                from: Space::Global,
                to: Space::Global,
                read_bytes: Some(4),
                written_bytes: Some(4)
            }
        );
        assert_eq!(
            class_of("atom.shared.cas.b64 %rd1, [%r1], %rd2, %rd3;"),
            OpClass::Copy {
                from: Space::Shared,
                to: Space::Shared,
                read_bytes: Some(8),
                written_bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("red.relaxed.gpu.global.add.v2.f32 [%rd1], {%f1, %f2};"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Store,
                bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("red.add.f16x2 [%rd1], %r1;"),
            OpClass::Memory {
                space: Space::Generic,
                direction: Direction::Store,
                bytes: Some(4)
            }
        );
    }

    #[test]
    fn out_of_audience_families_are_unknown_not_zero() {
        let tex = "tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [t, {%f5, %f6}];";
        assert_eq!(class_of(tex), OpClass::Unknown);
    }
}
