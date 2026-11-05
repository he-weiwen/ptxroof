//! One instruction is one record while contributing to several
//! totals, checked on collection's records, the annotated PTX and
//! `report::build::compare`.
use ptxroof::analysis::control_flow::loop_forest;
use ptxroof::analysis::instruction_counts::collect::collect;
use ptxroof::analysis::instruction_counts::measurement::{Direction, MeasureKind, Pipe, Precision};
use ptxroof::ptx::cfg::build_cfg;
use ptxroof::ptx::modifier::Space;
use ptxroof::ptx::parse::parser::parse;
use ptxroof::report::AnalyzeOptions;
use ptxroof::report::annotate::annotate;
use ptxroof::report::build::compare;
use ptxroof::report::measured::{NcuLine, NcuRecord, SectorOutcome};

fn source(body: &str) -> String {
    format!(".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() {{\n{body}\n}}")
}

/// The annotations of the lines of `annotated` whose PTX starts with
/// `ptx`, in order; a line without one is left out.
fn notes<'a>(annotated: &'a str, ptx: &str) -> Vec<&'a str> {
    (annotated.lines())
        .filter_map(|l| l.split_once(" // "))
        .filter(|(line, _)| line.trim_start().starts_with(ptx))
        .map(|(_, note)| note)
        .collect()
}

/// A loop of 3 trips holds an fma, an atom, which loads and stores
/// `[output]`, and a red, which only stores it (PTX ISA §9.7.14.5,
/// §9.7.14.6). Run by a CTA of 32 threads, the atom has no count of its
/// own: it runs as its block does, 96 times per CTA. The CTA's sums are
/// 32 times a thread's: the fma's 6 f32 FLOPs, the atom's 6 atomic f32
/// FLOPs, 12 B loaded and 24 B stored.
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
    let opts = AnalyzeOptions {
        launch: Some([32, 1, 1]),
        ..Default::default()
    };
    let out = annotate(&src, &opts).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        [
            "block 32×1×1 | per CTA: flops f32 192; atomic flops f32 192; global load 384 B, store 768 B"
        ]
    );
    assert_eq!(
        notes(&out, "L:"),
        [
            "loop: trips 3 | global per CTA: requested 768 B, unique 4 B, sectors 192 B | warp * × lane * | exec 96/CTA"
        ]
    );
    assert_eq!(
        notes(&out, "atom"),
        ["global load+store [output] sectors/req 1 (ideal 1)"]
    );
    assert_eq!(
        notes(&out, "red"),
        ["global store [output] sectors/req 1 (ideal 1)"]
    );
}

/// A predicated atom is one memory instruction: collection records its
/// atomic FLOP and its 4 B loaded and stored, each predicated. Under a
/// guard whose threads are not known it runs at most once per thread, and
/// a thread's atomic FLOPs, global loads and stores are bounds.
#[test]
fn predicated_atomic_is_one_memory_instruction_with_bounded_contributions() {
    let src = source("@%p1 atom.global.add.f32 %f1, [output], %f2;\nret;");
    let module = parse(&src).unwrap();
    let kernel = &module.kernels[0];
    let cfg = build_cfg(&module, kernel);
    let blocks = collect(&module, kernel, &cfg, &loop_forest(&cfg));
    assert_eq!(blocks[0].instructions, [0, 1]);
    let contributions: Vec<_> = blocks[0]
        .measurements
        .iter()
        .filter(|m| m.provenance == 0)
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
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        ["per thread: atomic flops f32 <= 1; global load <= 4 B, store <= 4 B"]
    );
    assert_eq!(
        notes(&out, "@%p1 atom"),
        [
            "[<block 0>] exec <= 1/thread | global load+store [output] sectors/req ? (block shape unknown: no .reqntid; pass --launch)"
        ]
    );
}

/// Three copies of one opcode, the first of source size 0 (PTX ISA
/// §9.7.9.26.3.1): each loads at `[source]` and stores at
/// `[destination]`, and a thread loads 0 + 16 + 16 = 32 B and stores
/// 48 B.
#[test]
fn copies_of_one_opcode_count_their_own_source_sizes_and_addresses() {
    let src = source(
        "cp.async.cg.shared.global [destination], [source], 16, 0;
cp.async.cg.shared.global [destination], [source], 16, 16;
cp.async.cg.shared.global [destination], [source], 16, 16;
ret;",
    );
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        ["per thread: global load 32 B; shared store 48 B"]
    );
    assert_eq!(
        notes(&out, "cp.async"),
        [
            "[<block 0>] shared store [destination] wavefronts/req ? (wavefronts not modelled) | global load [source] sectors/req 0 (ideal 0)",
            "shared store [destination] wavefronts/req ? (wavefronts not modelled) | global load [source] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
            "shared store [destination] wavefronts/req ? (wavefronts not modelled) | global load [source] sectors/req ? (block shape unknown: no .reqntid; pass --launch)",
        ]
    );
}

/// An ignored prefetch has no annotation (PTX ISA §9.7.9.16); the
/// unclassified `wgmma.fence` leaves its FLOPs and bytes out of the
/// totals, and an `ld.global` of unknown size its bytes. A thread's FLOPs,
/// global loads and shared bytes are then unknown, its global stores at
/// least the `st.global.u32`'s 4 B.
#[test]
fn ignored_unknown_and_unquantified_contributions_stay_distinct() {
    let src = source(
        "prefetch.global.L2 [source];\nwgmma.fence.sync.aligned;\nld.global %r1, [source];\nst.global.u32 [source], %r1;\nret;",
    );
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        ["per thread: flops ?; global load ? B, store >= 4 B; shared load ? B, store ? B"]
    );
    assert!(notes(&out, "prefetch").is_empty());
    assert_eq!(
        notes(&out, "wgmma"),
        ["not classified (its FLOPs and bytes not counted)"]
    );
    assert_eq!(
        notes(&out, "ld.global"),
        ["bytes ? (not counted) | global load [source]"]
    );
}

/// A `cp.async` of source size 0 reads no global memory (PTX ISA
/// §9.7.9.26.3.1): in a loop of 2 trips its sectors per request are 0,
/// ideal 0, and no global load is summed, while it runs twice per thread
/// and stores 32 B to shared memory. Given a CTA of 32 threads, the count
/// and the sums are 32 times as large and the loop's global bytes per CTA
/// are 0.
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
    let cases = [
        (
            Some([32, 1, 1]),
            "block 32×1×1 | shared memory 512 B/CTA | per CTA: shared store 1,024 B; param load 256 B",
            "loop: trips 2 | global per CTA: requested 0 B, unique 0 B, sectors 0 B | warp * × lane * | exec 64/CTA",
        ),
        (
            None,
            "shared memory 512 B/CTA | per thread: shared store 32 B; param load 8 B",
            "loop: trips 2 | exec 2/thread",
        ),
    ];
    for (launch, entry, header) in cases {
        let opts = AnalyzeOptions {
            launch,
            ..Default::default()
        };
        let out = annotate(src, &opts).expect("annotates");
        assert_eq!(notes(&out, ".visible .entry"), [entry]);
        assert_eq!(notes(&out, "L:"), [header]);
        assert_eq!(
            notes(&out, "cp.async.cg"),
            [
                "shared store [16·%tid.x + destination] wavefronts/req ? (wavefronts not modelled) | \
              global load [16·%tid.x + param_0] sectors/req 0 (ideal 0)"
            ]
        );
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
/// (PTX ISA §9.7.9.26.3.1). Over 4 trips, a copy 16 of 64 lanes read, 8–9
/// sectors per warp request, and one whose ignore-src compares a
/// parameter, the same for every thread, request at most 4 × (16 + 64) ×
/// 16 B = 5,120 B per CTA, as many distinct bytes and 5,120–5,504 B of
/// sectors, each a bound; both store to shared memory on all 64 threads,
/// 8,192 B.
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
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        [
            "block 64×1×1 | pointer params assumed 32 B aligned | shared memory 1,024 B/CTA | per CTA: global load <= 5,120 B; shared store 8,192 B; param load 768 B"
        ]
    );
    assert_eq!(
        notes(&out, "$L__LOOP:"),
        [
            "loop: trips 4 | global per CTA: requested <= 5,120 B, unique <= 5,120 B, sectors <= 5,120 B | warp * × lane * | exec 256/CTA"
        ]
    );
    assert_eq!(
        notes(&out, "cp.async.cg"),
        [
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
             global load [16·%tid.x + 1024·iv[$L__LOOP] + param_0] sectors/req 8 (ideal 8)",
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
             global load [16·%tid.x + 1024·iv[$L__LOOP] + param_0 + 65536] sectors/req 16 (ideal 16)",
        ]
    );
}

/// Lanes 8 B apart read 16 B where ignore-src is false. The even lanes,
/// one residue modulo 16, touch 8–9 sectors per warp request, and the
/// CTA's 32 of them load 512 B; where a loaded value decides, the copy's
/// read is a bound, `<= 1,280 B` with the load's 256 B, and the lanes that
/// may read differ modulo 16: its sectors per request are unknown, not a
/// bound.
#[test]
fn a_source_read_at_mixed_residues_by_unknown_lanes_has_no_footprint() {
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16, %p1;";
    let at = "mul.wide.u32 %rd2, %r1, 8;\nadd.s64 %rd3, %rd1, %rd2;\n";
    let annotated = |name: &str, setup: &str| {
        let src = ignore_src_kernel(name, &format!("{at}{setup}"), copy);
        annotate(&src, &AnalyzeOptions::default()).expect("annotates")
    };
    let odd = annotated("odd", "and.b32 %r5, %r1, 1;\nsetp.eq.u32 %p1, %r5, 1;");
    assert_eq!(
        notes(&odd, ".visible .entry"),
        [
            "block 64×1×1 | pointer params assumed 32 B aligned | shared memory 1,024 B/CTA | per CTA: global load 512 B; shared store 1,024 B; param load 512 B"
        ]
    );
    assert_eq!(
        notes(&odd, "cp.async.cg"),
        [
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
          global load [8·%tid.x + param_0] sectors/req 8 (ideal 8)"
        ]
    );
    let loaded = annotated(
        "loaded",
        "ld.global.u32 %r5, [%rd3];\nsetp.eq.u32 %p1, %r5, 1;",
    );
    assert_eq!(
        notes(&loaded, ".visible .entry"),
        [
            "block 64×1×1 | pointer params assumed 32 B aligned | shared memory 1,024 B/CTA | per CTA: global load <= 1,280 B; shared store 1,024 B; param load 512 B"
        ]
    );
    assert_eq!(
        notes(&loaded, "cp.async.cg"),
        [
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
          global load [8·%tid.x + param_0] sectors/req ? (lane addresses differ modulo the 16 B access size)"
        ]
    );
}

/// A src-size register reads at most cp-size (PTX ISA §9.7.9.26.3.1):
/// beside a copy that reads its 16 B under the same opcode, 16 sectors
/// per warp request, its sectors per request are bounds, `<= 16 (ideal
/// <= 16)`, and the CTA's 64 threads load at most 2 × 64 × 16 B = 2,048 B.
/// Each copy runs as its block does, 64 times per CTA.
#[test]
fn a_src_size_register_reads_at_most_cp_size() {
    let at = "mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;";
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16;\n\
                cp.async.cg.shared.global [%r4+1024], [%rd3+1024], 16, %r9;";
    let src = ignore_src_kernel("partial_src_size", at, copy);
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        [
            "block 64×1×1 | pointer params assumed 32 B aligned | shared memory 1,024 B/CTA | per CTA: global load <= 2,048 B; shared store 2,048 B; param load 768 B"
        ]
    );
    assert_eq!(
        notes(&out, "ld.param.u64"),
        ["[<block 0>] warp * × lane * | exec 64/CTA"]
    );
    assert_eq!(
        notes(&out, "cp.async.cg"),
        [
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
             global load [16·%tid.x + param_0] sectors/req 16 (ideal 16)",
            "shared store [16·%tid.x + buf + 1024] wavefronts/req ? (wavefronts not modelled) | \
             global load [16·%tid.x + param_0 + 1024] sectors/req <= 16 (ideal <= 16)",
        ]
    );
}

/// Each guard against each ignore-src, over a launch of one CTA of 64
/// threads: a 16-byte copy reads its source on the threads it runs on
/// whose ignore-src is false, a loaded mask a bound, and writes shared
/// memory on all it runs on; the `.entry` line leaves a zero sum out.
/// Against a record, its line's predicted executions are the threads it
/// runs on, none under a guard no thread passes.
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
    let reads = [
        "global load 1,024 B",
        "",
        "global load 256 B",
        "global load <= 1,280 B",
    ];
    let guarded = [
        "global load 768 B",
        "",
        "global load 256 B",
        "global load <= 1,024 B",
    ];
    let global = [reads, reads, [""; 4], guarded];
    let shared = [
        "shared store 1,024 B",
        "shared store 1,024 B",
        "",
        "shared store 768 B",
    ];
    let predicted = [64, 64, 0, 48];
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
                global_executions: None,
                shared: Vec::new(),
            };
            let record = NcuRecord {
                kernel: "k".to_owned(),
                grid: [1, 1, 1],
                block: [64, 1, 1],
                params: vec!["ptr:1024".to_owned(), "u32:0".to_owned()],
                lines: [(line, measured)].into(),
                load_sectors: None,
                store_sectors: None,
            };
            let opts = AnalyzeOptions {
                grid: Some([1, 1, 1]),
                ncu: Some(record),
                ..Default::default()
            };
            let out = annotate(&src, &opts).expect("annotates");
            let entry = notes(&out, ".visible .entry");
            let sums: Vec<&str> = (entry[0].split_once("per launch: "))
                .map_or("", |(_, sums)| sums)
                .split("; ")
                .filter(|s| s.starts_with("global") || s.starts_with("shared"))
                .collect();
            let cell = (g, i);
            let expected: Vec<&str> = [global[g][i], shared[g]]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect();
            assert_eq!(sums, expected, "{cell:?}");
            let c = compare(&src, &opts)
                .expect("analyzes")
                .expect("the record's kernel");
            let lines: Vec<u64> = (c.blocks.iter())
                .flat_map(|b| b.differing.iter().map(|d| d.predicted))
                .collect();
            assert_eq!(lines, [predicted[g]], "{cell:?}");
        }
    }
}

/// An ignore-src the same for every thread but not read, a comparison of
/// an unbound parameter, may leave every warp unread: the copy's sectors
/// over the launch are a bound, `<= 32 (ideal <= 32)`, within which a
/// record's 0 falls.
#[test]
fn a_cta_uniform_ignore_src_predicts_a_bound_of_sectors() {
    let setup = "mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;\nsetp.lt.u32 %p1, %r9, 5;";
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16, %p1;";
    let src = ignore_src_kernel("k", setup, copy);
    let line = src.lines().position(|l| l == copy).unwrap() as u32 + 1;
    let measured = NcuLine {
        ptx: copy.to_owned(),
        sass: [("LDGSTS.E.BYPASS.128".to_owned(), 1)].into(),
        executions: 64,
        l2_theoretical_sectors_global: Some(0),
        l2_theoretical_sectors_global_ideal: Some(0),
        global_access_sizes: vec![128],
        global_executions: Some(64),
        shared: Vec::new(),
    };
    let record = NcuRecord {
        kernel: "k".to_owned(),
        grid: [1, 1, 1],
        block: [64, 1, 1],
        params: vec!["ptr:1024".to_owned(), "u32:7".to_owned()],
        lines: [(line, measured)].into(),
        load_sectors: None,
        store_sectors: None,
    };
    let opts = AnalyzeOptions {
        grid: Some([1, 1, 1]),
        ncu: Some(record),
        ..Default::default()
    };
    let c = compare(&src, &opts)
        .expect("analyzes")
        .expect("the record's kernel");
    let [sectors] = &c.blocks[0].sectors[..] else {
        panic!("one line compared");
    };
    let predicted = (sectors.predicted.max, sectors.ideal.max);
    assert_eq!(
        (predicted, sectors.at_most, sectors.outcome),
        ((32, 32), true, SectorOutcome::Agrees)
    );
}

/// An ignore-src that is always true reads on no thread: no sectors per
/// request, and the CTA loads no global bytes while it stores 1,024 B to
/// shared memory.
#[test]
fn a_source_every_thread_ignores_is_read_by_none() {
    let at = "mul.wide.u32 %rd2, %r1, 16;\nadd.s64 %rd3, %rd1, %rd2;\nmov.pred %p1, -1;";
    let copy = "cp.async.cg.shared.global [%r4], [%rd3], 16, %p1;";
    let src = ignore_src_kernel("never", at, copy);
    let out = annotate(&src, &AnalyzeOptions::default()).expect("annotates");
    assert_eq!(
        notes(&out, ".visible .entry"),
        [
            "block 64×1×1 | shared memory 1,024 B/CTA | per CTA: shared store 1,024 B; param load 512 B"
        ]
    );
    assert_eq!(
        notes(&out, "cp.async.cg"),
        [
            "shared store [16·%tid.x + buf] wavefronts/req ? (wavefronts not modelled) | \
          global load [16·%tid.x + param_0] sectors/req 0 (ideal 0)"
        ]
    );
}
