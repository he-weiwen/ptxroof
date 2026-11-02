//! Kernels against their Nsight Compute records through
//! `report::build::compare`: block verdicts and counts, the lines whose
//! measured count is not the predicted one, L2 sector and shared
//! wavefront outcomes, and the launch's L1TEX sector totals.

use ptxroof::report::build::{AnalyzeOptions, compare};
use ptxroof::report::measured::{
    Comparison, MeasuredLine, MeasuredSectors, NcuRecord, SectorCount, SectorOutcome, Verdict,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// The comparison of a record, a path from the repository root, with the
/// PTX it names.
fn comparison(record: &str) -> Comparison {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = fs::read_to_string(root.join(record)).expect("record readable");
    let fields: serde_json::Value = serde_json::from_str(&text).expect("record is JSON");
    let ptx = fields["ptx"].as_str().expect("record names its PTX");
    let source = fs::read_to_string(root.join(ptx)).expect("the record's PTX");
    let record: NcuRecord = serde_json::from_str(&text).expect("record parses");
    let opts = AnalyzeOptions {
        bindings: record.bindings(),
        launch: Some(record.block),
        grid: Some(record.grid),
        ncu: Some(record),
    };
    (compare(&source, &opts).expect("record analyzes")).expect("the record's kernel")
}

/// Each block's name, verdict, instructions, those with SASS and those
/// whose count agrees.
fn blocks(c: &Comparison) -> Vec<(&str, Verdict, u64, u64, u64)> {
    (c.blocks.iter())
        .map(|b| {
            (
                b.name.as_str(),
                b.verdict,
                b.instructions,
                b.with_sass,
                b.agreeing,
            )
        })
        .collect()
}

fn differing(c: &Comparison, line: u32) -> Option<&MeasuredLine> {
    (c.blocks.iter().flat_map(|b| &b.differing)).find(|d| d.line == line)
}

fn sass(opcodes: &[(&str, u32)]) -> BTreeMap<String, u32> {
    (opcodes.iter())
        .map(|&(op, n)| (op.to_owned(), n))
        .collect()
}

/// k5 at M=N=K=256 against its record, block by block. The epilogue is
/// confirmed: each fma.rn.f32 line is one FFMA per execution and executes
/// 64 threads x 16 CTAs = 1024 times. At K = 256 the branch on K never
/// takes $L__BB0_5, the path for K < 1, so the K loop is always entered
/// and its blocks are confirmed too ($L__BB0_3's 14 differing lines are
/// address arithmetic ptxas hoists out of the loop and ld.shared.u16 it
/// merges into LDS.64); $L__BB0_5 runs on no thread, and its one line with
/// SASS executes 0 times.
#[test]
fn k5_at_256_confirms_its_loops_and_epilogue() {
    use Verdict::{Confirmed, Consistent};
    let c = comparison("tests/fixtures/k5/k5.sm_89.256.ncu.json");
    assert_eq!(
        blocks(&c),
        [
            ("<block 0>", Consistent, 10, 7, 6),
            ("<block 1>", Consistent, 74, 5, 1),
            ("$L__BB0_2", Confirmed, 155, 119, 117),
            ("$L__BB0_3", Confirmed, 111, 97, 83),
            ("<block 4>", Confirmed, 8, 7, 5),
            ("<block 5>", Consistent, 1, 1, 1),
            ("$L__BB0_5", Consistent, 64, 1, 1),
            ("$L__BB0_6", Confirmed, 412, 400, 366),
        ]
    );
}

/// k5 at M=N=K=256 runs otherwise than its PTX where ptxas rewrites it: in
/// the inner loop it runs four address lines once per K iteration (32768
/// of 262144) and merges the eight ld.shared.u16 lines into three loads;
/// each epilogue st.global.u16 gains a PRMT.
#[test]
fn k5_lines_ptxas_hoists_merges_or_widens_differ() {
    let c = comparison("tests/fixtures/k5/k5.sm_89.256.ncu.json");
    let epilogue = [("PRMT", 1), ("STG.E.U16", 1)];
    for (line, ptx, predicted, executions, opcodes) in [
        (
            60,
            "setp.eq.s32 %p1, %r8, 0;",
            1024,
            2048,
            &[("ISETP.NE.AND", 1), ("ULDC.64", 1)][..],
        ),
        (
            315,
            "shl.b32 %r105, %r14, 3;",
            262144,
            32768,
            &[("SHF.L.U32", 1)],
        ),
        (
            320,
            "ld.shared.u16 %rs17, [%r110];",
            262144,
            98304,
            &[("LDS.64", 2), ("LDS.U16", 1)],
        ),
        (
            722,
            "st.global.u16 [%rd52+10], %rs44;",
            1024,
            2048,
            &epilogue,
        ),
    ] {
        let expected = MeasuredLine {
            line,
            ptx: ptx.to_owned(),
            tested: false,
            predicted,
            at_most: false,
            executions,
            sass: sass(opcodes),
        };
        assert_eq!(differing(&c, line), Some(&expected));
    }
}

/// micro/irreducible at param 1 = 32 runs block A ten times per thread
/// (the record: 320 fma.rn.f32 over 32 threads), but A belongs to no
/// natural loop, so the tool has no count for it: no block is compared.
#[test]
fn an_irreducible_cycle_leaves_every_block_unchecked() {
    let c = comparison("tests/fixtures/micro/irreducible.32.ncu.json");
    let verdicts: Vec<(&str, Verdict)> = (c.blocks.iter())
        .map(|b| (b.name.as_str(), b.verdict))
        .collect();
    assert_eq!(
        verdicts,
        [
            ("<block 0>", Verdict::Unchecked),
            ("$L__BB0_A", Verdict::Unchecked),
            ("$L__BB0_B", Verdict::Unchecked),
            ("<block 3>", Verdict::Unchecked),
        ]
    );
}

/// Two fma.rn.f32 on one line: the record counts the line's 64 FFMA
/// executions together, so neither instruction is compared; the block
/// rests on its other lines.
#[test]
fn two_instructions_on_one_line_are_not_compared() {
    let c = comparison("tests/lit/Inputs/ncu-two-per-line.ncu.json");
    assert_eq!(blocks(&c), [("<block 0>", Verdict::Consistent, 7, 6, 3)]);
}

/// A line whose global access ptxas lowers at its own width compares its
/// L2 sectors over the launch with the record's
/// `l2_theoretical_sectors_global`: the predicted sectors per warp request
/// times the warps with a lane that runs it. Here one warp loads and
/// stores one float at one address, one sector each.
#[test]
fn a_line_at_its_own_width_compares_its_l2_sectors() {
    let c = comparison("tests/lit/Inputs/ncu-two-per-line.ncu.json");
    let one = SectorCount { min: 1, max: 1 };
    let sectors = |line| MeasuredSectors {
        line,
        predicted: one,
        ideal: one,
        at_most: false,
        measured: 1,
        measured_ideal: 1,
        executions: 32,
        outcome: SectorOutcome::Agrees,
    };
    assert_eq!(c.blocks.len(), 1);
    assert_eq!(c.blocks[0].sectors, [sectors(17), sectors(19)]);
}
