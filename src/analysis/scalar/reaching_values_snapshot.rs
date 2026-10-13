//! Temporary (docs/reaching-values-refactor.md, step 12, deleted with
//! `reach_def`): every register read in the fixtures, CC and guarded
//! old values included, and every header query that excludes its loop,
//! where the tracer's reaching definitions and the register SSA
//! disagree, counted by class with the first examples of each.

use crate::analysis::control_flow::loops::{LoopId, loop_forest};
use crate::analysis::register_effects::{reads_cc, registers, split_operands, writes};
use crate::analysis::scalar::trace::{AffineValueTracer, ReachingDefinition};
use crate::ptx::cfg::{BlockId, ControlFlowGraph, build_cfg};
use crate::ptx::ir::{Module, Register, Stmt};
use crate::ptx::parse::parser::parse;
use crate::ssa::{RegisterSsa, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_ptx(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("fixture dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_ptx(&path, out);
        } else if path.extension().is_some_and(|e| e == "ptx") {
            out.push(path);
        }
    }
}

/// One kernel's tracer and SSA, and how to name its statements.
struct Kernel<'a> {
    m: &'a Module,
    cfg: &'a ControlFlowGraph,
    t: &'a AffineValueTracer<'a>,
    newlines: &'a [usize],
}

impl Kernel<'_> {
    fn line(&self, stmt: usize) -> String {
        let offset = match &self.t.kernel.stmts[stmt] {
            Stmt::Instr(i) => i.offset as usize,
            Stmt::Unparsed { offset, .. } => *offset as usize,
            _ => return format!("stmt{stmt}"),
        };
        format!("L{}", self.newlines.partition_point(|&n| n < offset) + 1)
    }

    fn block(&self, b: BlockId) -> String {
        self.cfg.block_name(self.m, b)
    }

    fn header(&self, l: LoopId) -> BlockId {
        self.t.forest.get(l).header
    }

    /// Whether `reg` is a counter of loop `l`, or the register a
    /// counter's chain starts from.
    fn counter(&self, reg: Register, l: LoopId) -> bool {
        let Register::Named(r) = reg else {
            return false;
        };
        (self.t.ivs[l.0 as usize].iter()).any(|(&c, (_, phi))| c == r || *phi == r)
    }

    fn reg(&self, reg: Register) -> String {
        match reg {
            Register::Named(r) => self.m.interner.resolve(r).to_owned(),
            Register::Cc => "CC".to_owned(),
        }
    }

    fn show_old(&self, old: &ReachingDefinition) -> String {
        match old {
            ReachingDefinition::Def(d) => self.line(*d),
            ReachingDefinition::Merged(ds) => {
                let ds: Vec<String> = ds.iter().map(|&d| self.line(d)).collect();
                format!("merged({})", ds.join(" "))
            }
            ReachingDefinition::Carried(l) => format!("carried({})", self.block(self.header(*l))),
            ReachingDefinition::None => "none".to_owned(),
        }
    }

    fn show_new(&self, v: Option<Value>, sources: &[Value]) -> String {
        let one = |v: &Value| match *v {
            Value::Result { op, .. } => self.line(op),
            Value::Argument { block, .. } => format!("phi({})", self.block(block)),
            Value::Undef(_) => "undef".to_owned(),
        };
        let Some(v) = v else {
            return "none".to_owned();
        };
        let sources: Vec<String> = sources.iter().map(one).collect();
        format!("{} [{}]", one(&v), sources.join(" "))
    }

    /// Why the old answer and the new disagree, or `None`.
    fn class(
        &self,
        old: &ReachingDefinition,
        new: Option<Value>,
        sources: &[Value],
    ) -> Option<String> {
        let Some(new) = new else {
            return Some("read in an unreachable block".to_owned());
        };
        let results: BTreeSet<usize> = (sources.iter())
            .filter_map(|v| match *v {
                Value::Result { op, .. } => Some(op),
                _ => None,
            })
            .collect();
        let undef = sources.iter().any(|v| matches!(v, Value::Undef(_)));
        let kind = |v: Value| match v {
            Value::Result { .. } => "a write",
            Value::Argument { .. } => "a φ",
            Value::Undef(_) => "undef",
        };
        match (old, new) {
            (ReachingDefinition::Def(d), Value::Result { op, .. }) if *d == op => None,
            (ReachingDefinition::None, Value::Undef(_)) => None,
            (ReachingDefinition::Merged(defs), Value::Argument { .. }) => {
                let defs: BTreeSet<usize> = defs.iter().copied().collect();
                let mut why = Vec::new();
                if !defs.is_subset(&results) {
                    why.push("keeps a definition the φ does not");
                }
                if !results.is_subset(&defs) {
                    why.push("misses a definition the φ has");
                }
                if undef {
                    why.push("misses a path writing nothing");
                }
                (!why.is_empty()).then(|| format!("merged {}", why.join(", ")))
            }
            (ReachingDefinition::Carried(l), Value::Argument { block, reg }) => {
                let of = match self.counter(reg, *l) {
                    true => ", a counter of the carrying loop",
                    false => "",
                };
                if block == self.header(*l) {
                    None
                } else if (self.t.forest.loops.iter()).any(|x| x.header == block) {
                    Some(format!("carried names another loop than the φ's{of}"))
                } else {
                    Some(format!("carried where the φ is at a join{of}"))
                }
            }
            (ReachingDefinition::None, Value::Argument { .. }) => Some(match undef {
                true => "no definition, SSA a φ that may be undef".to_owned(),
                false => "no definition, SSA a φ always written".to_owned(),
            }),
            (ReachingDefinition::Def(_), v) => Some(format!("one definition, SSA {}", kind(v))),
            (ReachingDefinition::None, v) => Some(format!("no definition, SSA {}", kind(v))),
            (ReachingDefinition::Merged(_), v) => Some(format!("merged, SSA {}", kind(v))),
            (ReachingDefinition::Carried(_), v) => Some(format!("carried, SSA {}", kind(v))),
        }
    }

    fn old_at(&self, reg: Register, pos: usize, exclude: Option<LoopId>) -> ReachingDefinition {
        let sites = self.t.writes.get(&reg).map_or(&[][..], Vec::as_slice);
        self.t.reach_sites(sites, pos, exclude)
    }

    /// The registers statement `pos` reads: its read operands' and its
    /// guard's, CC when it reads it, and, when guarded, what it writes.
    fn reads(&self, pos: usize) -> BTreeSet<RegisterKey> {
        let Stmt::Instr(i) = &self.t.kernel.stmts[pos] else {
            return BTreeSet::new();
        };
        let mut regs: Vec<Register> = (split_operands(self.m, i).1.iter())
            .flat_map(|&op| registers(self.m, op))
            .map(Register::Named)
            .collect();
        if let Some(p) = i.predicate {
            regs.push(Register::Named(p.reg));
            regs.extend(writes(self.m, &self.t.kernel.stmts[pos]));
        }
        if reads_cc(self.m, i) {
            regs.push(Register::Cc);
        }
        regs.into_iter().map(RegisterKey).collect()
    }
}

/// A `Register` ordered by its symbol, CC last.
#[derive(PartialEq, Eq)]
struct RegisterKey(Register);

impl Ord for RegisterKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |r: &Register| match *r {
            Register::Named(s) => (0, Some(s)),
            Register::Cc => (1, None),
        };
        key(&self.0).cmp(&key(&other.0))
    }
}

impl PartialOrd for RegisterKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct Classes {
    agree: usize,
    by_class: BTreeMap<String, (usize, Vec<String>)>,
}

impl Classes {
    fn add(&mut self, class: Option<String>, example: impl FnOnce() -> String) {
        let Some(class) = class else {
            self.agree += 1;
            return;
        };
        let (n, examples) = self.by_class.entry(class).or_default();
        *n += 1;
        if examples.len() < 3 {
            examples.push(example());
        }
    }

    fn write(&self, out: &mut String, title: &str) {
        let total = self.agree + self.by_class.values().map(|(n, _)| n).sum::<usize>();
        let _ = writeln!(out, "## {title}: {total}, agreeing {}", self.agree);
        for (class, (n, examples)) in &self.by_class {
            let _ = writeln!(out, "{class}: {n}");
            for e in examples {
                let _ = writeln!(out, "    {e}");
            }
        }
    }
}

#[test]
fn reaching_definitions_against_register_ssa() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_ptx(&root.join("tests/fixtures"), &mut files);
    files.sort();
    let (mut reads, mut headers) = (Classes::default(), Classes::default());
    for path in &files {
        let src = fs::read_to_string(path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        let rel = path.strip_prefix(&root).expect("under the repo").display();
        let newlines: Vec<usize> = src.match_indices('\n').map(|(i, _)| i).collect();
        for kernel in &m.kernels {
            let cfg = build_cfg(&m, kernel);
            let forest = loop_forest(&cfg);
            let t = AffineValueTracer::new(&m, kernel, &cfg, &forest);
            let ssa = RegisterSsa::new(&cfg, &forest.doms, t.writes.clone());
            let k = Kernel {
                m: &m,
                cfg: &cfg,
                t: &t,
                newlines: &newlines,
            };
            for b in cfg.blocks.iter() {
                for pos in b.start..b.end {
                    for RegisterKey(reg) in k.reads(pos) {
                        let old = k.old_at(reg, pos, None);
                        let new = ssa.value_before(pos, reg);
                        let sources = new.map_or(Vec::new(), |v| ssa.sources(v));
                        reads.add(k.class(&old, new, &sources), || {
                            format!(
                                "{rel}:{} {}: {} / {}",
                                k.line(pos),
                                k.reg(reg),
                                k.show_old(&old),
                                k.show_new(new, &sources)
                            )
                        });
                    }
                }
            }
            for (id, l) in forest.loops.iter().enumerate() {
                let id = LoopId(id as u32);
                let start = cfg.block(l.header).start;
                let written: BTreeSet<RegisterKey> = (t.writes.iter())
                    .filter(|(r, ws)| {
                        matches!(r, Register::Named(_)) && ws.iter().any(|&w| t.in_loop(id, w))
                    })
                    .map(|(&r, _)| RegisterKey(r))
                    .collect();
                for RegisterKey(reg) in written {
                    let old = k.old_at(reg, start, Some(id));
                    let at_header = ssa.value_before(start, reg);
                    // The header φ's values from outside the loop: one
                    // value when they agree, else the φ itself, read
                    // through those values alone.
                    let (new, sources) = match at_header {
                        Some(v @ Value::Argument { block, .. }) if block == l.header => {
                            let outside: Vec<Value> = (ssa.incoming(v).into_iter())
                                .filter(|(p, _)| {
                                    p.is_none_or(|p| l.blocks.binary_search(&p).is_err())
                                })
                                .map(|(_, w)| w)
                                .collect();
                            let mut sources: Vec<Value> =
                                outside.iter().flat_map(|&w| ssa.sources(w)).collect();
                            sources.sort_by_key(|v| match *v {
                                Value::Result { op, .. } => op,
                                _ => usize::MAX,
                            });
                            sources.dedup();
                            match outside[..] {
                                [w, ref rest @ ..] if rest.iter().all(|&x| x == w) => {
                                    (Some(w), sources)
                                }
                                _ => (Some(v), sources),
                            }
                        }
                        v => (v, v.map_or(Vec::new(), |v| ssa.sources(v))),
                    };
                    let role = match k.counter(reg, id) {
                        true => "counter",
                        false => "other",
                    };
                    let class = k.class(&old, new, &sources).map(|c| format!("{role}: {c}"));
                    headers.add(class, || {
                        format!(
                            "{rel}: {} on entry to {}: {} / {}",
                            k.reg(reg),
                            k.block(l.header),
                            k.show_old(&old),
                            k.show_new(new, &sources)
                        )
                    });
                }
            }
        }
    }
    let mut out = String::new();
    reads.write(&mut out, "reads");
    headers.write(&mut out, "header queries excluding the loop");
    insta::assert_snapshot!(out);
}
