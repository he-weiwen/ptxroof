//! One instruction retains one category while contributing to several totals.
use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::classify::{Direction, InstructionCategory, Space};
use ptxroof::analysis::instruction_counts::collect::{CountQualifier, collect};
use ptxroof::analysis::instruction_counts::stats;
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::schema::{ContributionDetails, InstructionVariant};
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
    atom.global.add.f32 %f1, [output], %f2;
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
    let ids = stats::all_blocks(&blocks);
    let bytes = stats::bytes(&blocks, &ids, Some(Space::Global), None);
    assert_eq!(
        (bytes.value, bytes.ops, bytes.qualifier),
        (8, 1, CountQualifier::AtMost)
    );
    assert_eq!(
        stats::bytes(&blocks, &ids, None, Some(Direction::Load)).ops,
        1
    );
    assert_eq!(stats::flops(&blocks, &ids, None).ops, 1);
    let instruction = &blocks[0].instructions[0];
    assert_eq!(instruction.classified.category, InstructionCategory::Memory);
    assert!(instruction.predicated);
    let contributions: Vec<_> = blocks[0]
        .measurements
        .iter()
        .filter(|m| m.provenance == instruction.provenance)
        .collect();
    assert_eq!(contributions.len(), 3);
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
        &source("nop;\nfuture.op %r1;\nld.global %r1, [source];\nret;"),
        "unknown",
        &Default::default(),
    )
    .unwrap();
    let kernel = &report.kernels[0];
    assert_eq!(kernel.instruction_classes.ignore, 1);
    assert_eq!(kernel.instruction_classes.unknown, 1);
    let totals = &kernel.totals;
    assert!(
        variants(totals, "nop")[0]
            .contributions_per_execution
            .is_empty()
    );
    assert!(
        matches!(&variants(totals, "future.op")[0].contributions_per_execution[0], ContributionDetails::UnknownOps { mnemonic } if mnemonic == "future")
    );
    assert!(
        matches!(&variants(totals, "ld.global")[0].contributions_per_execution[0], ContributionDetails::UnquantifiedBytes { space, direction } if space == "global" && direction == "load")
    );
    assert!(totals.atomic_flops["total"].at_least);
    assert!(totals.bytes["global"].load.at_least);
    assert!(!kernel.unknowns.is_empty());
}
