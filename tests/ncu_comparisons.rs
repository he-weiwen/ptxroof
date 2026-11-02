//! Kernels against their Nsight Compute records through
//! `report::build::compare`: block verdicts and counts, the lines whose
//! measured count is not the predicted one, L2 sector and shared
//! wavefront outcomes, and the launch's L1TEX sector totals.

use ptxroof::report::build::{AnalyzeOptions, compare};
use ptxroof::report::measured::{
    Comparison, MeasuredLine, MeasuredSectors, MeasuredTotal, NcuRecord, SectorCount,
    SectorOutcome, Verdict,
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

/// Every record in `dir`, a path from the repository root, with its
/// comparison.
fn records(dir: &str) -> Vec<(String, Comparison)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<String> = (fs::read_dir(root.join(dir)).expect("fixture dir readable"))
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with(".ncu.json"))
        .map(|name| format!("{dir}/{name}"))
        .collect();
    paths.sort();
    (paths.into_iter())
        .map(|path| {
            let c = comparison(&path);
            (path, c)
        })
        .collect()
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

/// The lines whose L2 sectors are compared, by outcome: those that agree
/// with an exact prediction, a range or a bound, those that differ, and
/// those whose global SASS ran other than predicted.
fn sectors(c: &Comparison) -> BTreeMap<&'static str, usize> {
    let mut out = BTreeMap::new();
    for s in c.blocks.iter().flat_map(|b| &b.sectors) {
        let kind = match (s.outcome, s.at_most, s.predicted.min == s.predicted.max) {
            (SectorOutcome::Differs, _, _) => "differ",
            (SectorOutcome::ExecutionsDiffer, _, _) => "not compared",
            (SectorOutcome::Agrees, true, _) => "within a bound",
            (SectorOutcome::Agrees, false, true) => "exact",
            (SectorOutcome::Agrees, false, false) => "within a range",
        };
        *out.entry(kind).or_default() += 1;
    }
    out
}

/// Each line whose L2 sectors are compared, its outcome and the measured
/// sectors.
fn sector_lines(c: &Comparison) -> Vec<(u32, SectorOutcome, u64)> {
    (c.blocks.iter().flat_map(|b| &b.sectors))
        .map(|s| (s.line, s.outcome, s.measured))
        .collect()
}

fn verdict(c: &Comparison, block: &str) -> Option<Verdict> {
    (c.blocks.iter())
        .find(|b| b.name == block)
        .map(|b| b.verdict)
}

fn total(min: u64, max: u64, measured: u64) -> Option<MeasuredTotal> {
    Some(MeasuredTotal {
        predicted: SectorCount { min, max },
        at_most: false,
        measured,
        agrees: true,
    })
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

/// Every B200 kernel against its Nsight Compute record
/// (`fixtures/b200/regen.sh`): no block's predicted count is contradicted
/// by the measured one.
#[test]
fn no_b200_block_is_contradicted() {
    let records = records("tests/fixtures/b200");
    assert_eq!(records.len(), 34);
    let contradicted: Vec<String> = (records.iter())
        .flat_map(|(path, c)| {
            (c.blocks.iter())
                .filter(|b| b.verdict == Verdict::Contradicted)
                .map(move |b| format!("{path}: {}", b.name))
        })
        .collect();
    assert_eq!(contradicted, Vec::<String>::new());
}

/// Every Gluon kernel against its Nsight Compute record (`regen.sh`): no
/// block's predicted count is contradicted by the measured one, each
/// compares the L2 sectors of some line and none differs from the
/// measured ones, and no launch's load or store sectors differ from its
/// L1TEX counts.
#[test]
fn gluon_records_agree_on_counts_sectors_and_launch_totals() {
    let records = records("tests/fixtures/gluon");
    assert_eq!(records.len(), 12);
    for (path, c) in &records {
        let verdicts: Vec<Verdict> = c.blocks.iter().map(|b| b.verdict).collect();
        assert!(!verdicts.contains(&Verdict::Contradicted), "{path}");
        let sectors = sectors(c);
        let agree: usize = ["exact", "within a range", "within a bound"]
            .map(|k| sectors.get(k).copied().unwrap_or(0))
            .iter()
            .sum();
        assert!(agree > 0, "{path}");
        assert_eq!(sectors.get("differ"), None, "{path}");
        let totals = c.launch.iter().flat_map(|t| [&t.loads, &t.stores]);
        assert!(totals.flatten().all(|t| t.agrees), "{path}");
    }
}

/// Counts over the launch, summed over the record's CTAs. In
/// _chunk_scan_fwd_tma (grid 2x32x80, 384 threads), the load on line 184
/// runs on the 256 threads of warps 0 to 7 in the CTAs whose %ctaid.y is
/// not 0, 31 of every 32: 2·31·80·256 = 1269760 times. ptxas attributes
/// to the line an IMAD.MOV.U32 that runs on those 256 threads of every
/// CTA, 1310720 times, and the record measures 1310720 + 1269760.
#[test]
fn scan_tma_s_line_184_counts_the_ctas_that_run_it() {
    let c = comparison("tests/fixtures/b200/scan_tma_ws_best._chunk_scan_fwd_tma.sm_100a.ncu.json");
    let expected = MeasuredLine {
        line: 184,
        ptx: "@%p17 ld.global.b32 { %r93 }, [ %rd62 + 0 ];".to_owned(),
        tested: false,
        predicted: 1269760,
        at_most: false,
        executions: 2580480,
        sass: sass(&[("IMAD.MOV.U32", 1), ("LDG.E", 1)]),
    };
    assert_eq!(differing(&c, 184), Some(&expected));
}

/// A guard comparing the thread index with a CTA index, read under the
/// record's grid. In set_mla_kv_buffer_kernel_norope
/// `8·(%tid.x mod 128) + 1024·%ctaid.y < 576` guards a 16 B load and
/// store. The grid is 8192x1x1, so %ctaid.y is 0: the store on line 93
/// runs on 72 threads of each CTA, 589824 times over the launch, as the
/// record measures. ptxas attributes two CS2R to the load's line 88, which
/// the record measures more often.
#[test]
fn the_grid_decides_a_guard_on_the_cta_index() {
    let c = comparison(
        "tests/fixtures/b200/rope_kv__deepseek-v3__prefill__sgl_mla_set_norope.set_mla_kv_buffer_kernel_norope.sm_100a.ncu.json",
    );
    let expected = MeasuredLine {
        line: 88,
        ptx: "@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd3 + 0 ];".to_owned(),
        tested: false,
        predicted: 589824,
        at_most: false,
        executions: 2686976,
        sass: sass(&[("CS2R", 2), ("LDG.E.128", 1)]),
    };
    assert_eq!(differing(&c, 88), Some(&expected));
    assert_eq!(differing(&c, 93), None);
}

/// fp8_gemm_kernel.lm_head_dx's K loop runs 512 trips under its record,
/// and its <block 4> only on those whose counter passes line 957's
/// `setp.ge.u64 %p3, %rd55, %rd2`, read per trip as `k < 511`: 510 trips
/// x 256 threads x 96 CTAs = 12,533,760, as Nsight Compute measures on its
/// barrier, line 973. Its two `cp.async` lines, 979 and 985, are `@!P0
/// LDGSTS`, whose L2 sectors Nsight Compute miscounts, so the record keeps
/// none for them (micro/ldgsts_p0.ptx, PLAN.md). The launch's L1TEX load
/// count, 62,916,096, is the least of the predictions summed with them.
#[test]
fn lm_head_dx_s_counter_guard_confirms_its_block() {
    let c = comparison("tests/fixtures/gluon/fp8_gemm_kernel.lm_head_dx.sm_89.ncu.json");
    let block = (blocks(&c).into_iter()).find(|b| b.0 == "<block 4>");
    assert_eq!(block, Some(("<block 4>", Verdict::Confirmed, 16, 10, 8)));
    assert_eq!(differing(&c, 973), None);
    let verdicts: Vec<Verdict> = c.blocks.iter().map(|b| b.verdict).collect();
    assert!(!verdicts.contains(&Verdict::Contradicted));
    let expected = BTreeMap::from([("exact", 10), ("within a range", 12)]);
    assert_eq!(sectors(&c), expected);
    let loads = c.launch.and_then(|t| t.loads);
    assert_eq!(loads, total(62916096, 69207552, 62916096));
}

/// micro/ldgsts_p0's copy A (`@!P0 LDGSTS`, 16 threads) keeps no sectors
/// in its record: Nsight Compute counts it as 64 where its threads request
/// 8. Copy B (`@!P1 LDGSTS`, 48 threads) and the two loads are compared
/// and agree. Over the launch, the four lines' predictions sum to 48–86
/// load sectors, and the L1TEX hardware count is 64, copy A's 8 included.
#[test]
fn an_ldgsts_on_p0_keeps_no_sectors_but_counts_in_the_launch_total() {
    let c = comparison("tests/fixtures/micro/ldgsts_p0.ncu.json");
    assert_eq!(c.blocks.len(), 1);
    use SectorOutcome::Agrees;
    assert_eq!(
        sector_lines(&c),
        [(39, Agrees, 24), (41, Agrees, 8), (42, Agrees, 24)]
    );
    let launch = c.launch.expect("the record's L1TEX counts");
    assert_eq!(launch.incomplete, None);
    assert_eq!(launch.loads, total(48, 86, 64));
    assert_eq!(launch.stores, total(0, 0, 0));
}

/// A do-while loop whose latch is false at its first evaluation runs once
/// (ptxas, and on an RTX 4090): the record at n = 5 agrees with
/// micro_latch_exits_first_param's count of 1 on both stores' L2 sectors.
#[test]
fn a_loop_whose_latch_fails_first_stores_its_predicted_sectors() {
    let c = comparison("tests/fixtures/micro/latch_exits_first.ncu.json");
    assert_eq!(sectors(&c), BTreeMap::from([("exact", 2)]));
}

/// LLVM's loop guard branch: micro_loop_guard skips its loop when n < 11,
/// the latch condition at k = 0, so the count is max(n − 10, 0) per
/// execution of the guard. On an RTX 4090 at n = 20 the guarded loop runs
/// 10 trips: its store requests 10 L2 sectors, as predicted.
#[test]
fn a_guarded_loop_s_store_requests_its_predicted_sectors() {
    let c = comparison("tests/fixtures/micro/loop_guard.ncu.json");
    assert_eq!(sectors(&c), BTreeMap::from([("exact", 1)]));
    assert_eq!(sector_lines(&c), [(34, SectorOutcome::Agrees, 10)]);
}

/// A guard skipping two consecutive loops is the second loop's entry when
/// the first leaves only through its latch, so micro_loop_guard_sibling
/// counts the second loop per execution of the guard. On an RTX 4090 at
/// n = 20 the second loop's store requests 10 L2 sectors, as predicted.
#[test]
fn the_second_of_two_guarded_loops_stores_its_predicted_sectors() {
    let c = comparison("tests/fixtures/micro/loop_guard_sibling.ncu.json");
    assert_eq!(sectors(&c), BTreeMap::from([("exact", 1)]));
    assert_eq!(sector_lines(&c), [(44, SectorOutcome::Agrees, 10)]);
}

/// A loop of n / d trips, its latch traced under the record's bindings:
/// at n = 20, d = 4 the quotient gives 5 trips, as on an RTX 4090, which
/// confirms the loop.
#[test]
fn a_loop_of_a_quotient_s_trips_is_confirmed() {
    let c = comparison("tests/fixtures/micro/trip_div_param.ncu.json");
    assert_eq!(verdict(&c, "$L__BB0_2"), Some(Verdict::Confirmed));
}

/// A loop of max(n, 2) trips: micro_trip_max at n = 1 runs 2, as on an
/// RTX 4090, which confirms the loop.
#[test]
fn a_loop_of_a_max_s_trips_is_confirmed() {
    let c = comparison("tests/fixtures/micro/trip_max.ncu.json");
    assert_eq!(verdict(&c, "$L__BB0_1"), Some(Verdict::Confirmed));
}

/// A causal loop: CTA c runs c trips. Over the launch each CTA's threads
/// count its trips, 0 + 1 + 2 + 3, so the barrier runs 192 times, as on an
/// RTX 4090 with 4 CTAs, which confirms the loop. Without the guard
/// skipping CTA 0, CTA 0 runs once, and the sum over the grid, 7, gives
/// the 224 barriers the RTX 4090 runs. In both the store's L2 sectors are
/// as predicted.
#[test]
fn a_loop_of_cta_index_trips_is_confirmed_over_the_launch() {
    let exact = BTreeMap::from([("exact", 1)]);
    let guarded = comparison("tests/fixtures/micro/cta_trips.ncu.json");
    assert_eq!(verdict(&guarded, "$L__BB0_2"), Some(Verdict::Confirmed));
    assert_eq!(sectors(&guarded), exact);
    let unguarded = comparison("tests/fixtures/micro/cta_trips.unguarded.ncu.json");
    assert_eq!(verdict(&unguarded, "$L__BB1_1"), Some(Verdict::Confirmed));
    assert_eq!(sectors(&unguarded), exact);
}

/// Guards on a loop counter, read per trip: the barrier's `k >= 3`
/// compares the counter alone, though the loop also holds a branch on the
/// thread index. Per CTA each trip counts its own threads: 6 trips of 64
/// barriers, as the RTX 4090 measures, which confirms the barrier's block.
#[test]
fn a_barrier_under_a_counter_guard_is_confirmed() {
    let c = comparison("tests/fixtures/micro/counter_guard.ncu.json");
    let block = (blocks(&c).into_iter()).find(|b| b.0 == "<block 2>");
    assert_eq!(block, Some(("<block 2>", Verdict::Confirmed, 1, 1, 1)));
}

/// A loop counter's range under the bindings: the store's `or` of 4·k and
/// %tid.x << 8 is an add only while 4·k < 256. With n bound to 32 by the
/// record the counter runs 1..32, the address reads, and on an RTX 4090
/// the store's 4,096 L2 sectors (32 per warp request) agree with the
/// prediction.
#[test]
fn a_counter_bound_by_the_record_reads_the_store_s_sectors() {
    let c = comparison("tests/fixtures/micro/counter_range_bound.ncu.json");
    assert_eq!(sectors(&c), BTreeMap::from([("exact", 1)]));
    assert_eq!(sector_lines(&c), [(37, SectorOutcome::Agrees, 4096)]);
}

/// A block two branches decide, one on its dominator chain and one below
/// it: the threads below 64 reach it where n != 0, or where n == 0 and m
/// != 0, so its set is a bound. On an RTX 4090 with n = m = 0 its barrier
/// runs on no thread, within the bound.
#[test]
fn a_barrier_two_branches_decide_runs_within_its_bound() {
    let c = comparison("tests/fixtures/micro/or_entry.ncu.json");
    assert_eq!(verdict(&c, "$L__BB0_3"), Some(Verdict::Consistent));
    let expected = MeasuredLine {
        line: 40,
        ptx: "bar.sync 1, 64;".to_owned(),
        tested: true,
        predicted: 64,
        at_most: true,
        executions: 0,
        sass: sass(&[("BAR.SYNC.DEFER_BLOCKING", 1)]),
    };
    assert_eq!(differing(&c, 40), Some(&expected));
}

/// Wrong output (PLAN.md): a predicated `exit` is read as falling through,
/// so the store after it counts on every thread, exactly; on an RTX 4090
/// at n = 20 it runs on none.
#[test]
fn a_store_after_a_predicated_exit_runs_on_no_thread() {
    let c = comparison("tests/fixtures/micro/predicated_exit.ncu.json");
    let expected = MeasuredLine {
        line: 28,
        ptx: "st.global.u32 [%rd1], %r1;".to_owned(),
        tested: false,
        predicted: 32,
        at_most: false,
        executions: 0,
        sass: sass(&[("MOV", 2), ("STG.E", 1), ("ULDC.64", 1)]),
    };
    assert_eq!(differing(&c, 28), Some(&expected));
}

/// cp.async with an ignore-src predicate: each copy's global read counts
/// the lanes whose ignore-src is false. Under the record, over the launch,
/// the cache-hinted copy, whose ignore-src compares the CTA index with a
/// parameter, reads on 64 lanes of CTA 0 and 16 of CTA 1, within its
/// bound; the four lines' 68 measured L2 sectors agree with the
/// predictions, three within a range and the cache-hinted copy's within a
/// bound.
#[test]
fn copies_that_ignore_their_source_read_their_predicted_sectors() {
    let c = comparison("tests/fixtures/micro/cp_async_ignore_src.ncu.json");
    let expected = BTreeMap::from([("within a range", 3), ("within a bound", 1)]);
    assert_eq!(sectors(&c), expected);
    use SectorOutcome::Agrees;
    assert_eq!(
        sector_lines(&c),
        [
            (36, Agrees, 16),
            (42, Agrees, 4),
            (47, Agrees, 8),
            (55, Agrees, 40)
        ]
    );
}

/// cp.async with a register src-size, 16 on lanes 0..15 and 0 on the
/// others: the register is not read, so which lanes read, and how much,
/// is a bound, within which Nsight Compute's 8 sectors over the launch
/// fall.
#[test]
fn a_register_source_size_bounds_the_sectors_measured() {
    let c = comparison("tests/fixtures/micro/cp_async_src_size.ncu.json");
    assert_eq!(sectors(&c), BTreeMap::from([("within a bound", 1)]));
    assert_eq!(sector_lines(&c), [(30, SectorOutcome::Agrees, 8)]);
}

/// Shared memory wavefronts per warp request, by how an RTX 4090 serves a
/// request's lanes (micro/shared_wavefronts.ptx): under the record every
/// counted line's wavefronts and ideal are Nsight Compute's over its
/// executions.
#[test]
fn every_counted_shared_line_s_wavefronts_are_measured() {
    let c = comparison("tests/fixtures/micro/shared_wavefronts.ncu.json");
    let lines: Vec<bool> = (c.blocks.iter().flat_map(|b| &b.wavefronts))
        .map(|w| w.agrees)
        .collect();
    assert_eq!(lines.len(), 59);
    assert!(lines.iter().all(|&agrees| agrees));
}
