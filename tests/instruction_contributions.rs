//! One instruction retains one category while contributing to several totals.
use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::{
    Direction, InstructionCategory, Pipe, Precision, Space,
};
use ptxroof::analysis::instruction_counts::collect::collect;
use ptxroof::analysis::instruction_counts::measurement::MeasureKind;
use ptxroof::analysis::memory_footprint::CountRange;
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::measured::{NcuLine, NcuRecord};
use ptxroof::report::schema::{ByteRange, ContributionDetails, Count, InstructionVariant};
use ptxroof::report::{AnalyzeOptions, analyze, text};

fn source(body: &str) -> String {
    format!(".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() {{\n{body}\n}}")
}

fn variants<'a>(
    agg: &'a ptxroof::report::schema::Aggregates,
    opcode: &str,
) -> &'a [InstructionVariant] {
    agg.instructions
        .by_kind
        .values()
        .find_map(|kind| kind.contribution_variants.get(opcode))
        .unwrap()
}

#[test]
fn atomic_contributions_scale_with_loops_and_launch_but_instruction_counts_do_not_duplicate() {
    let src = source(
        "mov.u32 %r1, 0;
L:
    atom.global.add.f32 %f1, [output], %f3;
    red.global.add.f32 [output], %f2;
    fma.rn.f32 %f3, %f1, %f2, %f3;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, 3;
    @%p1 bra L;
    ret;",
    );
    let report = analyze(
        &src,
        "contributions",
        &AnalyzeOptions {
            launch: Some([32, 1, 1]),
            ..Default::default()
        },
    )
    .unwrap();
    let kernel = &report.kernels[0];
    assert_eq!(kernel.instruction_classes.total, 8);
    assert_eq!(kernel.instruction_classes.memory, 2);
    assert_eq!(kernel.instruction_classes.flop, 1);
    let totals = &kernel.totals;
    assert_eq!(totals.instructions.total.expr, "20");
    assert_eq!(totals.flops["total"].expr, "6");
    assert_eq!(totals.atomic_flops["total"].expr, "6");
    assert_eq!(totals.bytes["global"].load.expr, "12");
    assert_eq!(totals.bytes["global"].store.expr, "24");
    assert_eq!(totals.ai_global.as_ref().unwrap().value, 1.0 / 3.0);
    let cta = kernel.totals_per_cta.as_ref().unwrap();
    assert_eq!(cta.instructions.total.expr, "640");
    assert_eq!(cta.atomic_flops["total"].expr, "192");
    assert_eq!(cta.bytes["global"].load.expr, "384");
    assert_eq!(cta.bytes["global"].store.expr, "768");
    let atom = &variants(cta, "atom.global.add.f32")[0];
    assert_eq!(atom.issued.expr, "96");
    assert_eq!(atom.contributions_per_execution.len(), 3);
    assert!(
        matches!(&atom.contributions_per_execution[2], ContributionDetails::Flops { pipe, precision, count: 1 } if pipe == "atomic" && precision == "f32")
    );
    assert_eq!(kernel.loops[0].accesses.len(), 2);
    assert_eq!(kernel.loops[0].accesses[0].direction, "load+store");
    assert_eq!(kernel.loops[0].accesses[0].bytes, Some(4));
    let rendered = text::render(&report);
    assert!(rendered.contains(
        "per thread per execution: global load 4 B; global store 4 B; atomic f32 FLOPs 1"
    ));
    assert!(rendered.contains("atomic flops = 6"));
    let json = serde_json::to_value(&report).unwrap();
    let details = &json["kernels"][0]["totals"]["instructions"]["by_kind"]["global atomic 4 B"]["contribution_variants"]
        ["atom.global.add.f32"][0];
    assert_eq!(details["issued"]["expr"], "3");
    assert_eq!(details["contributions_per_execution"][2]["kind"], "flops");
}

#[test]
fn predicated_atomic_has_one_issued_instruction_and_bounded_contributions() {
    let src = source("@%p1 atom.global.add.f32 %f1, [output], %f2;\nret;");
    let module = parse(&src).unwrap();
    let kernel = &module.kernels[0];
    let cfg = build_cfg(&module, kernel);
    let blocks = collect(&module, kernel, &cfg, &loop_forest(&cfg));
    let instruction = &blocks[0].instructions[0];
    assert_eq!(instruction.classified.category, InstructionCategory::Memory);
    assert!(instruction.predicated);
    let contributions: Vec<_> = blocks[0]
        .measurements
        .iter()
        .filter(|m| m.provenance == instruction.provenance)
        .collect();
    let mut kinds: Vec<_> = contributions.iter().map(|m| (m.kind, m.count)).collect();
    kinds.sort();
    assert_eq!(
        kinds,
        [
            (
                MeasureKind::Flops {
                    pipe: Pipe::Atomic,
                    precision: Precision::F32
                },
                1
            ),
            (
                MeasureKind::Bytes {
                    space: Space::Global,
                    direction: Direction::Load
                },
                4
            ),
            (
                MeasureKind::Bytes {
                    space: Space::Global,
                    direction: Direction::Store
                },
                4
            ),
        ]
    );
    assert!(contributions.iter().all(|m| m.predicated));
    let report = analyze(&src, "predicated", &Default::default()).unwrap();
    let totals = &report.kernels[0].totals;
    assert_eq!(totals.instructions.total.expr, "2");
    let atom = &variants(totals, "atom.global.add.f32")[0];
    assert_eq!(atom.issued.expr, "1");
    assert!(!atom.issued.at_most);
    assert!(totals.atomic_flops["total"].at_most);
    assert!(totals.bytes["global"].load.at_most);
    assert!(totals.bytes["global"].store.at_most);
    assert!(totals.ai_global.is_none());
}

#[test]
fn same_opcode_preserves_copy_size_variants_and_address_associations() {
    let src = source(
        "cp.async.cg.shared.global [destination], [source], 16, 0;
cp.async.cg.shared.global [destination], [source], 16, 16;
cp.async.cg.shared.global [destination], [source], 16, 16;
ret;",
    );
    let report = analyze(&src, "copies", &Default::default()).unwrap();
    let kernel = &report.kernels[0];
    let totals = &kernel.totals;
    assert_eq!(kernel.instruction_classes.memory, 3);
    assert_eq!(totals.bytes["global"].load.expr, "32");
    assert_eq!(totals.bytes["shared"].store.expr, "48");
    let copies = variants(totals, "cp.async.cg.shared.global");
    assert_eq!(copies.len(), 2);
    assert_eq!(copies[0].issued.expr, "1");
    assert_eq!(copies[1].issued.expr, "2");
    assert!(matches!(
        copies[0].contributions_per_execution[0],
        ContributionDetails::Bytes { count: 0, .. }
    ));
    assert!(matches!(
        copies[1].contributions_per_execution[0],
        ContributionDetails::Bytes { count: 16, .. }
    ));
    assert_eq!(kernel.accesses.len(), 6);
    for access in &kernel.accesses {
        match access.space.as_str() {
            "global" => {
                assert_eq!(access.direction, "load");
                assert_eq!(access.address.as_deref(), Some("source"));
            }
            "shared" => {
                assert_eq!(access.direction, "store");
                assert_eq!(access.address.as_deref(), Some("destination"));
            }
            _ => panic!("unexpected space"),
        }
    }
    let rendered = text::render(&report);
    assert!(rendered.contains("variant issued"));
    assert!(rendered.contains("global load 0 B; shared store 16 B"));
    assert!(rendered.contains("global load 16 B; shared store 16 B"));
}

#[test]
fn ignored_unknown_and_unquantified_contributions_stay_distinct() {
    let report = analyze(
        &source("prefetch.global.L2 [source];\nwgmma.fence.sync.aligned;\nld.global %r1, [source];\nst.global.u32 [source], %r1;\nret;"),
        "unknown",
        &Default::default(),
    )
    .unwrap();
    let kernel = &report.kernels[0];
    assert_eq!(kernel.instruction_classes.ignore, 1);
    assert_eq!(kernel.instruction_classes.unknown, 1);
    let totals = &kernel.totals;
    assert!(
        variants(totals, "prefetch.global.L2")[0]
            .contributions_per_execution
            .is_empty()
    );
    assert!(
        matches!(&variants(totals, "wgmma.fence.sync.aligned")[0].contributions_per_execution[0], ContributionDetails::UnknownOps { mnemonic } if mnemonic == "wgmma")
    );
    assert!(
        matches!(&variants(totals, "ld.global")[0].contributions_per_execution[0], ContributionDetails::UnquantifiedBytes { space, direction } if space == "global" && direction == "load")
    );
    assert!(totals.atomic_flops["total"].at_least);
    assert!(totals.bytes["global"].load.at_least);
    assert!(!kernel.unknowns.is_empty());
}

#[test]
fn zero_source_copy_has_no_global_sectors_but_keeps_shared_writes() {
    let src = ".version 8.7
.target sm_80
.address_size 64
.visible .entry zero_copy(.param .u64 input) {
    .reg .b64 %rd<3>;
    .reg .b32 %r<4>;
    .reg .pred %p;
    .shared .align 16 .b8 destination[512];
    ld.param.u64 %rd1, [input];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd2, %r1, 16;
    add.u64 %rd2, %rd1, %rd2;
    mul.lo.u32 %r1, %r1, 16;
    mov.u32 %r3, destination;
    add.u32 %r1, %r3, %r1;
    mov.u32 %r2, 0;
L:
    cp.async.cg.shared.global [%r1], [%rd2], 16, 0;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p, %r2, 2;
    @%p bra L;
    cp.async.wait_all;
    ret;
}";
    let zero = ptxroof::analysis::memory_footprint::CountRange { min: 0, max: 0 };
    for launch in [Some([32, 1, 1]), None] {
        let report = analyze(
            src,
            "zero-copy",
            &AnalyzeOptions {
                launch,
                ..Default::default()
            },
        )
        .unwrap();
        let kernel = &report.kernels[0];
        let loop_node = &kernel.loops[0];
        let global = loop_node
            .accesses
            .iter()
            .find(|a| a.space == "global")
            .unwrap();
        assert_eq!(global.bytes, Some(0));
        assert_eq!(global.sectors_per_request, Some(zero));
        assert_eq!(global.ideal_sectors_per_request, Some(zero));
        assert!(global.footprint_unknown.is_none());
        assert_eq!(kernel.totals.bytes["global"].load.expr, "0");
        assert_eq!(kernel.totals.bytes["shared"].store.expr, "32");
        assert_eq!(
            variants(&kernel.totals, "cp.async.cg.shared.global")[0]
                .issued
                .expr,
            "2"
        );
        if launch.is_some() {
            let bytes = loop_node.global_bytes_per_cta.as_ref().unwrap();
            assert_eq!(bytes.requested, 0);
            assert_eq!(bytes.unique, Some(ByteRange { min: 0, max: 0 }));
            let sectors = bytes.sector_bytes.as_ref().unwrap();
            assert_eq!((sectors.min, sectors.max), (0, 0));
            let cta = kernel.totals_per_cta.as_ref().unwrap();
            assert_eq!(cta.bytes["shared"].store.expr, "1024");
            assert_eq!(
                variants(cta, "cp.async.cg.shared.global")[0].issued.expr,
                "64"
            );
        }
    }
}

/// A `.reqntid 64` kernel that copies 16 B per thread with `copy`, after
/// `setup`; `%r1` holds the thread index and `%r4` the shared address.
fn ignore_src_kernel(name: &str, setup: &str, copy: &str) -> String {
    format!(
        ".version 8.7\n.target sm_89\n.address_size 64\n\
         .visible .entry {name}(.param .u64 k_param_0, .param .u32 k_param_1)\n\
         .reqntid 64, 1, 1\n{{\n\
         .reg .pred %p<4>;\n.reg .b32 %r<10>;\n.reg .b64 %rd<8>;\n\
         .shared .align 16 .b8 buf[1024];\n\
         ld.param.u64 %rd1, [k_param_0];\nld.param.u32 %r9, [k_param_1];\n\
         mov.u32 %r1, %tid.x;\nmov.u32 %r2, buf;\nshl.b32 %r3, %r1, 4;\n\
         add.s32 %r4, %r2, %r3;\n{setup}\n{copy}\ncp.async.wait_all;\nret;\n}}\n"
    )
}

/// A `cp.async` reads its source only where its ignore-src is false
/// (PTX ISA §9.7.9.26.3.1). Over 4 trips, a copy 16 of 64 lanes read and
/// one whose ignore-src compares a parameter, the same for every thread,
/// request at most 4 × (16 + 64) × 16 B; the shared row is not
/// predicated, the source rows are.
#[test]
fn a_loop_footprint_counts_the_lanes_that_may_read_a_copy_s_source() {
    let setup = "setp.lt.u32 %p1, %r1, 16;\nnot.pred %p2, %p1;\nsetp.lt.u32 %p3, %r9, 7;\n\
                 mov.u32 %r5, 0;";
    let copy = "$L__LOOP:\nshl.b32 %r6, %r5, 6;\nadd.s32 %r7, %r6, %r1;\n\
                mul.wide.u32 %rd2, %r7, 16;\nadd.s64 %rd3, %rd1, %rd2;\n\
                cp.async.cg.shared.global [%r4], [%rd3], 16, %p2;\n\
                cp.async.cg.shared.global [%r4], [%rd3+65536], 16, %p3;\n\
                add.s32 %r5, %r5, 1;\nsetp.lt.u32 %p1, %r5, 4;\n@%p1 bra $L__LOOP;";
    let src = ignore_src_kernel("loop_ignore_src", setup, copy);
    let report = analyze(&src, "loop", &AnalyzeOptions::default()).unwrap();
    let node = &report.kernels[0].loops[0];
    let bytes = node.global_bytes_per_cta.as_ref().unwrap();
    assert_eq!(bytes.requested, 5120);
    assert_eq!(
        bytes.unique,
        Some(ByteRange {
            min: 5120,
            max: 5120
        })
    );
    let sectors = bytes.sector_bytes.as_ref().unwrap();
    assert_eq!(
        (sectors.min, sectors.max, bytes.at_most),
        (5120, 5504, true)
    );
    let flags: Vec<(&str, bool)> = (node.accesses.iter())
        .map(|a| (a.space.as_str(), a.predicated))
        .collect();
    let expected = [("shared", false), ("global", true)];
    assert_eq!(flags, [expected, expected].concat());
}

/// Lanes 8 B apart read 16 B where ignore-src is false. The even lanes,
/// one residue modulo 16, touch 8–9 sectors per warp; where a loaded
/// value decides, the lanes that may read differ modulo 16 and the
/// footprint is unknown, not a bound.
#[test]
fn a_source_read_at_mixed_residues_by_unknown_lanes_has_no_footprint() {
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16, %p1;";
    let at = "mul.wide.u32 %rd2, %r1, 8;\nadd.s64 %rd3, %rd1, %rd2;\n";
    let source = |name: &str, setup: &str| {
        let src = ignore_src_kernel(name, &format!("{at}{setup}"), copy);
        let report = analyze(&src, name, &AnalyzeOptions::default()).unwrap();
        (report.kernels[0].accesses.iter())
            .find(|a| a.space == "global" && a.opcode.starts_with("cp.async"))
            .cloned()
            .unwrap()
    };
    let odd = source("odd", "and.b32 %r5, %r1, 1;\nsetp.eq.u32 %p1, %r5, 1;");
    let range = |min, max| Some(CountRange { min, max });
    assert_eq!(odd.sectors_per_request, range(8, 9));
    assert_eq!(odd.threads.and_then(|t| t.count), Some(32));
    let loaded = source(
        "loaded",
        "ld.global.u32 %r5, [%rd3];\nsetp.eq.u32 %p1, %r5, 1;",
    );
    assert!(loaded.predicated && loaded.sectors_per_request.is_none());
    assert_eq!(
        loaded.footprint_unknown.as_deref(),
        Some("lane addresses differ modulo the 16 B access size")
    );
}

/// A src-size register reads at most cp-size (PTX ISA §9.7.9.26.3.1):
/// beside a copy that reads its 16 B under the same opcode, its row and
/// its variant read `<= 16 B`, in text and JSON, and the two variants
/// stay apart.
#[test]
fn a_src_size_register_reads_at_most_cp_size() {
    let at = "mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;";
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16;\n\
                cp.async.cg.shared.global [%r4+1024], [%rd3+1024], 16, %r9;";
    let src = ignore_src_kernel("partial_src_size", at, copy);
    let report = analyze(&src, "partial", &AnalyzeOptions::default()).unwrap();
    let rendered = text::render(&report);
    assert!(rendered.contains("global load <= 16 B (predicated)"));
    assert!(rendered.contains("global load 16 B; shared store 16 B"));
    assert!(rendered.contains("global load <= 16 B; shared store 16 B"));
    let json = serde_json::to_value(&report).unwrap();
    let kernel = &json["kernels"][0];
    let rows: Vec<_> = (kernel["accesses"].as_array().unwrap().iter())
        .filter(|a| a["space"] == "global")
        .map(|a| (a.get("bytes_at_most").is_some(), a["predicated"] == true))
        .collect();
    assert_eq!(rows, [(false, false), (true, true)]);
    let copies = &kernel["totals_per_cta"]["instructions"]["by_kind"]["global -> shared copy 16 B"]
        ["contribution_variants"]["cp.async.cg.shared.global"];
    let read = |v: usize| copies[v]["contributions_per_execution"][0].clone();
    assert_eq!(read(0).get("at_most"), None);
    assert_eq!(read(1)["at_most"], true);
    assert_eq!(copies[1]["issued"]["expr"], "64");
    let load = &report.kernels[0].totals_per_cta.as_ref().unwrap().bytes["global"].load;
    assert_eq!((load.expr.as_str(), load.at_most), ("2048", true));
}

/// Each guard against each ignore-src, per CTA of 64 threads: a 16-byte
/// copy reads its source on the threads it runs on whose ignore-src is
/// false, a loaded mask a bound, and writes shared memory on all it runs
/// on. It issues on the block's threads; under a guard no thread passes
/// it is in no live count and predicts no execution.
#[test]
fn a_copy_reads_where_its_guard_runs_and_its_ignore_src_is_false() {
    let guards = [
        ("", ""),
        ("mov.pred %p3, 1;", "@%p3 "),
        ("mov.pred %p3, 0;", "@%p3 "),
        ("setp.lt.u32 %p3, %r1, 48;", "@%p3 "),
    ];
    let ignores = [
        "mov.pred %p2, 0;",
        "mov.pred %p2, -1;",
        "setp.ge.u32 %p2, %r1, 16;",
        "ld.global.u32 %r5, [%rd1];\nsetp.eq.u32 %p2, %r5, 0;",
    ];
    let reads = ["1024", "0", "256", "<= 1280"];
    let global = [reads, reads, ["0"; 4], ["768", "0", "256", "<= 1024"]];
    let (shared, predicted) = (["1024", "1024", "0", "768"], [64, 64, 0, 48]);
    let count = |c: &Count| format!("{}{}", if c.at_most { "<= " } else { "" }, c.expr);
    for (g, (def, guard)) in guards.into_iter().enumerate() {
        for (i, ignore) in ignores.into_iter().enumerate() {
            let setup =
                format!("mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;\n{ignore}\n{def}");
            let copy = format!("{guard}cp.async.cg.shared.global [%r4], [%rd3], 16, %p2;");
            let src = ignore_src_kernel("k", &setup, &copy);
            let line = src.lines().position(|l| l == copy).unwrap() as u32 + 1;
            let measured = NcuLine {
                ptx: copy,
                sass: [("LDGSTS.E.BYPASS.128".to_owned(), 1)].into(),
                executions: 1 << 20,
                l2_theoretical_sectors_global: None,
                l2_theoretical_sectors_global_ideal: None,
                global_access_sizes: Vec::new(),
            };
            let record = NcuRecord {
                kernel: "k".to_owned(),
                grid: [1, 1, 1],
                block: [64, 1, 1],
                params: vec!["ptr:1024".to_owned(), "u32:0".to_owned()],
                lines: [(line, measured)].into(),
            };
            let opts = AnalyzeOptions {
                grid: Some([1, 1, 1]),
                ncu: Some(record),
                ..Default::default()
            };
            let report = analyze(&src, "matrix", &opts).unwrap();
            let kernel = &report.kernels[0];
            let cta = kernel.totals_per_cta.as_ref().unwrap();
            let cell = (g, i);
            assert_eq!(count(&cta.bytes["global"].load), global[g][i], "{cell:?}");
            assert_eq!(count(&cta.bytes["shared"].store), shared[g], "{cell:?}");
            let copies = cta.instructions.by_kind.get("global -> shared copy 16 B");
            let live = copies.map(|c| c.total.expr.as_str());
            assert_eq!(live, (g != 2).then_some("64"), "{cell:?}");
            if g == 2 {
                let all = cta.instructions.all.as_ref().map(|c| c.expr.as_str());
                assert_eq!(all, Some(if i == 3 { "896" } else { "832" }), "{cell:?}");
            }
            let lines: Vec<u64> = (kernel.measured.as_ref().unwrap().iter())
                .flat_map(|b| b.differing.iter().map(|d| d.predicted))
                .collect();
            assert_eq!(lines, [predicted[g]], "{cell:?}");
        }
    }
}

/// An ignore-src that is always true reads on no thread: no sectors.
#[test]
fn a_source_every_thread_ignores_is_read_by_none() {
    let at = "mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;\nmov.pred %p1, -1;";
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16, %p1;";
    let src = ignore_src_kernel("never", at, copy);
    let report = analyze(&src, "never", &AnalyzeOptions::default()).unwrap();
    let global = (report.kernels[0].accesses.iter())
        .find(|a| a.space == "global")
        .unwrap();
    let none = Some(CountRange { min: 0, max: 0 });
    assert_eq!(global.sectors_per_request, none);
    let threads = global.threads.as_ref().unwrap();
    assert_eq!(
        (threads.condition.as_str(), threads.count),
        ("no thread", Some(0))
    );
}
