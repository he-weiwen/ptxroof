//! Sharing and stability of PTX special-register values within one CTA,
//! and the ranges the ISA gives them.

use crate::ptx::ir::{Element, SpecialReg, SpecialScalar, SpecialVector};

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

/// The least and greatest value the ISA lets `reg` take on any target.
/// PTX ISA §10.1 "0 <= %tid.x < %ntid.x"; §10.2 "CTA dimensions are
/// non-zero", and its "Maximum values of %ntid.{x,y,z}" at most 1024,
/// 1024 and 64; §10.3 "The lane identifier ranges from zero to
/// WARP_SZ-1", and §4.5.1 "all target architectures have a WARP_SZ value
/// of 32"; §10.6 "0 <= %ctaid.x < %nctaid.x"; §10.7 "each element having
/// a value of at least 1", and its "Maximum values of %nctaid.{x,y,z}"
/// at most 2^31 − 1, 65535 and 65535. `None` for the rest.
pub(crate) fn special_register_range(reg: SpecialReg) -> Option<(i64, i64)> {
    use Element::*;
    use SpecialVector::*;
    Some(match reg {
        SpecialReg::Scalar(SpecialScalar::Laneid) => (0, 31),
        SpecialReg::Vector(Tid, Some(X | Y)) => (0, 1023),
        SpecialReg::Vector(Tid, Some(Z)) => (0, 63),
        SpecialReg::Vector(Ntid, Some(X | Y)) => (1, 1024),
        SpecialReg::Vector(Ntid, Some(Z)) => (1, 64),
        SpecialReg::Vector(Ctaid, Some(X)) => (0, (1 << 31) - 2),
        SpecialReg::Vector(Ctaid, Some(Y | Z)) => (0, 65534),
        SpecialReg::Vector(Nctaid, Some(X)) => (1, (1 << 31) - 1),
        SpecialReg::Vector(Nctaid, Some(Y | Z)) => (1, 65535),
        _ => return None,
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

    #[test]
    fn special_register_ranges_are_the_isa_s_on_any_target() {
        let of = |name| special_register_range(SpecialReg::from_name(name).expect(name));
        for (name, range) in [
            ("%tid.x", Some((0, 1023))),
            ("%tid.y", Some((0, 1023))),
            ("%tid.z", Some((0, 63))),
            ("%tid.w", None),
            ("%ntid.y", Some((1, 1024))),
            ("%ntid.z", Some((1, 64))),
            ("%laneid", Some((0, 31))),
            ("%ctaid.x", Some((0, (1 << 31) - 2))),
            ("%ctaid.z", Some((0, 65534))),
            ("%nctaid.x", Some((1, (1 << 31) - 1))),
            ("%nctaid.y", Some((1, 65535))),
            ("%ntid", None),
            ("%warpid", None),
        ] {
            assert_eq!(of(name), range, "{name}");
        }
    }
}
