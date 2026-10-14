//! One instruction retains one category while contributing to several totals.
use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::{
    Direction, InstructionCategory, Pipe, Precision, Space,
};
use ptxroof::analysis::instruction_counts::collect::collect;
use ptxroof::analysis::instruction_counts::measurement::MeasureKind;
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::schema::{ByteRange, ContributionDetails, InstructionVariant};
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
