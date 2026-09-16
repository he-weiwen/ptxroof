//! Display names for kernels.

/// Demangle an Itanium-mangled C++ name; plain names pass through.
pub fn demangle(name: &str) -> String {
    cpp_demangle::Symbol::new(name)
        .ok()
        .and_then(|s| s.demangle(&cpp_demangle::DemangleOptions::default()).ok())
        .unwrap_or_else(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demangle_ladder_kernel_names() {
        assert_eq!(
            demangle("_Z11hgemm_naiveiiifPK6__halfS1_fPS_"),
            "hgemm_naive(int, int, int, float, __half const*, __half const*, float, __half*)"
        );
        assert!(
            demangle("_Z20hgemm_2d_blocktilingILi64ELi64ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_")
                .starts_with("void hgemm_2d_blocktiling<64, 64, 8, 8, 8>(")
        );
    }

    #[test]
    fn plain_names_pass_through_demangling() {
        assert_eq!(demangle("micro_single_loop"), "micro_single_loop");
        assert_eq!(demangle("_not_mangled"), "_not_mangled");
    }
}
