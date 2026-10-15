use crate::analysis::instruction_counts::count_constraint::CountConstraint;
use crate::ptx::ir::Module;
use crate::report::build::BindingSpec;
use crate::report::schema::{MeasuredBlock, MeasuredLine, MeasuredSectors, SectorCount, Verdict};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct NcuRecord {
    pub kernel: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub params: Vec<String>,
    pub lines: BTreeMap<u32, NcuLine>,
}

#[derive(Debug, Deserialize)]
pub struct NcuLine {
    pub ptx: String,
    pub sass: BTreeMap<String, u32>,
    pub executions: u64,
    pub l2_theoretical_sectors_global: Option<u64>,
    pub l2_theoretical_sectors_global_ideal: Option<u64>,
    #[serde(default)]
    pub global_access_sizes: Vec<u32>,
}

pub struct Prediction {
    pub line: u32,
    pub constraint: CountConstraint,
    pub executions: Option<u64>,
    pub at_most: bool,
    pub sectors: Option<Sectors>,
}

/// The sectors a line's one global access of `bits` requests over the
/// launch, and the ideal number.
pub struct Sectors {
    pub bits: u32,
    pub requested: SectorCount,
    pub ideal: SectorCount,
    pub at_most: bool,
}

impl NcuRecord {
    pub fn bindings(&self) -> Vec<BindingSpec> {
        let value = |param: &str| match param.split_once(':')? {
            ("u32", v) => v.parse::<u32>().ok().map(i64::from),
            ("i32", v) => v.parse::<i32>().ok().map(i64::from),
            _ => None,
        };
        (self.params.iter().enumerate())
            .filter_map(|(index, param)| {
                Some(BindingSpec {
                    index: Some(index),
                    name: format!("param_{index}"),
                    value: value(param)?,
                })
            })
            .collect()
    }

    pub fn check(&self, module: &Module, source: &str) -> Result<(), String> {
        if !(module.kernels.iter()).any(|k| module.interner.resolve(k.name) == self.kernel) {
            return Err(format!("no kernel `{}`", self.kernel));
        }
        let text: Vec<&str> = source.lines().collect();
        for (&line, measured) in &self.lines {
            let here = (line.checked_sub(1)).and_then(|i| text.get(i as usize));
            let here = squeeze(here.copied().unwrap_or_default());
            if here != squeeze(&measured.ptx) {
                return Err(format!(
                    "line {line}: the record measured `{}`, the PTX has `{here}`",
                    measured.ptx
                ));
            }
        }
        Ok(())
    }

    pub fn compare(&self, block: &str, predictions: &[Prediction]) -> MeasuredBlock {
        let mut out = MeasuredBlock {
            name: block.to_owned(),
            verdict: Verdict::Unchecked,
            instructions: predictions.len() as u64,
            with_sass: 0,
            agreeing: 0,
            differing: Vec::new(),
            sectors: Vec::new(),
        };
        let (mut confirmed, mut contradicted) = (false, false);
        for p in predictions {
            let Some(measured) = self.lines.get(&p.line) else {
                continue;
            };
            out.with_sass += 1;
            if let Some(s) = &p.sectors
                && let (Some(requested), Some(ideal)) = (
                    measured.l2_theoretical_sectors_global,
                    measured.l2_theoretical_sectors_global_ideal,
                )
                && !measured.global_access_sizes.is_empty()
                && measured.global_access_sizes.iter().all(|&b| b == s.bits)
            {
                let within = |n: u64, c: SectorCount| n <= c.max && (s.at_most || c.min <= n);
                out.sectors.push(MeasuredSectors {
                    line: p.line,
                    predicted: s.requested,
                    ideal: s.ideal,
                    at_most: s.at_most,
                    measured: requested,
                    measured_ideal: ideal,
                    agrees: within(requested, s.requested) && within(ideal, s.ideal),
                });
            }
            let Some(predicted) = p.executions else {
                continue;
            };
            let agrees = match p.at_most {
                true => measured.executions <= predicted,
                false => measured.executions == predicted,
            };
            let tested = p.constraint != CountConstraint::Free && measured.sass.len() == 1;
            out.agreeing += u64::from(agrees);
            confirmed |= tested && agrees && !p.at_most;
            contradicted |= tested && !agrees;
            if measured.executions != predicted {
                out.differing.push(MeasuredLine {
                    line: p.line,
                    ptx: measured.ptx.clone(),
                    tested,
                    predicted,
                    at_most: p.at_most,
                    executions: measured.executions,
                    sass: measured.sass.clone(),
                });
            }
        }
        out.verdict = match (contradicted, confirmed, out.agreeing > 0) {
            (true, _, _) => Verdict::Contradicted,
            (_, true, _) => Verdict::Confirmed,
            (_, _, true) => Verdict::Consistent,
            _ => Verdict::Unchecked,
        };
        out
    }
}

pub struct LineIndex(Vec<usize>);

impl LineIndex {
    pub fn new(source: &str) -> Self {
        LineIndex(source.match_indices('\n').map(|(i, _)| i).collect())
    }

    pub fn line(&self, offset: u32) -> u32 {
        self.0.partition_point(|&n| n < offset as usize) as u32 + 1
    }
}

fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::CountConstraint::{Constrained, Free};
    use super::*;

    fn record(lines: &[(u32, &[&str], u64)]) -> NcuRecord {
        let line = |sass: &[&str], executions| NcuLine {
            ptx: String::new(),
            sass: sass.iter().map(|s| (s.to_string(), 1)).collect(),
            executions,
            l2_theoretical_sectors_global: None,
            l2_theoretical_sectors_global_ideal: None,
            global_access_sizes: Vec::new(),
        };
        NcuRecord {
            kernel: "k".into(),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            params: Vec::new(),
            lines: (lines.iter())
                .map(|&(n, sass, executions)| (n, line(sass, executions)))
                .collect(),
        }
    }

    fn predict(
        line: u32,
        constraint: CountConstraint,
        executions: u64,
        at_most: bool,
    ) -> Prediction {
        Prediction {
            line,
            constraint,
            executions: Some(executions),
            at_most,
            sectors: None,
        }
    }

    fn verdict(record: &NcuRecord, predictions: &[Prediction]) -> Verdict {
        record.compare("b", predictions).verdict
    }

    #[test]
    fn u32_and_i32_parameters_bind_by_position() {
        let mut r = record(&[]);
        r.params = ["u32:4294967295", "i32:-1", "f32:1", "ptr:4", "u32:-1"]
            .map(String::from)
            .to_vec();
        let bound: Vec<(Option<usize>, i64)> =
            r.bindings().iter().map(|b| (b.index, b.value)).collect();
        assert_eq!(bound, [(Some(0), 4294967295), (Some(1), -1)]);
    }

    #[test]
    fn a_constrained_line_with_one_sass_opcode_decides_the_block() {
        let r = record(&[(1, &["FFMA"], 64), (2, &["IADD3"], 32)]);
        let free = predict(2, Free, 64, false);
        assert_eq!(
            verdict(&r, &[predict(1, Constrained, 64, false), free]),
            Verdict::Confirmed
        );
        assert_eq!(
            verdict(&r, &[predict(1, Constrained, 128, false)]),
            Verdict::Contradicted
        );
    }

    #[test]
    fn free_lines_and_lines_split_across_sass_opcodes_only_agree() {
        let r = record(&[(1, &["IADD3"], 64), (2, &["PRMT", "STG.E.U16"], 128)]);
        assert_eq!(
            verdict(&r, &[predict(1, Free, 64, false)]),
            Verdict::Consistent
        );
        assert_eq!(
            verdict(&r, &[predict(1, Free, 32, false)]),
            Verdict::Unchecked
        );
        assert_eq!(
            verdict(&r, &[predict(2, Constrained, 64, false)]),
            Verdict::Unchecked
        );
    }

    #[test]
    fn a_bound_agrees_when_not_exceeded_and_never_confirms() {
        let r = record(&[(1, &["FFMA"], 0), (2, &["FFMA"], 2048)]);
        assert_eq!(
            verdict(&r, &[predict(1, Constrained, 1024, true)]),
            Verdict::Consistent
        );
        assert_eq!(
            verdict(&r, &[predict(2, Constrained, 1024, true)]),
            Verdict::Contradicted
        );
    }

    #[test]
    fn every_line_whose_executions_are_not_the_predicted_number_is_listed() {
        let r = record(&[
            (1, &["FFMA"], 64),
            (2, &["PRMT", "STG.E.U16"], 128),
            (3, &["LDS.64"], 16),
        ]);
        let block = r.compare(
            "b",
            &[
                predict(1, Constrained, 64, false),
                predict(2, Free, 64, false),
                predict(3, Free, 64, true),
                predict(4, Free, 64, false),
            ],
        );
        let listed: Vec<(u32, u64, bool)> = (block.differing.iter())
            .map(|d| (d.line, d.executions, d.at_most))
            .collect();
        assert_eq!(listed, [(2, 128, false), (3, 16, true)]);
        assert_eq!(
            (block.instructions, block.with_sass, block.agreeing),
            (4, 3, 2)
        );
    }

    #[test]
    fn lines_without_sass_or_a_numeric_prediction_leave_a_block_unchecked() {
        let r = record(&[(1, &["FFMA"], 64)]);
        let symbolic = Prediction {
            line: 1,
            constraint: Constrained,
            executions: None,
            at_most: false,
            sectors: None,
        };
        assert_eq!(
            verdict(&r, &[predict(2, Constrained, 64, false)]),
            Verdict::Unchecked
        );
        assert_eq!(verdict(&r, &[symbolic]), Verdict::Unchecked);
    }

    #[test]
    fn a_line_lowered_at_its_own_width_checks_its_sectors() {
        let mut r = record(&[]);
        for (n, sizes, measured) in [
            (1, vec![128], Some((32, 16))),
            (2, vec![32, 32], Some((8, 16))),
            (3, vec![32], Some((4, 4))),
            (4, vec![], None),
            (5, vec![32], Some((65, 64))),
        ] {
            let mut line = record(&[(n, &["LDG.E"], 64)]).lines.remove(&n).unwrap();
            line.global_access_sizes = sizes;
            line.l2_theoretical_sectors_global = measured.map(|m| m.0);
            line.l2_theoretical_sectors_global_ideal = measured.map(|m| m.1);
            r.lines.insert(n, line);
        }
        let count = |min, max| SectorCount { min, max };
        let at = |line, bits, requested, ideal, at_most| Prediction {
            sectors: Some(Sectors {
                bits,
                requested,
                ideal,
                at_most,
            }),
            ..predict(line, Free, 64, false)
        };
        let block = r.compare(
            "b",
            &[
                at(1, 128, count(32, 40), count(16, 16), false),
                at(2, 32, count(8, 8), count(8, 8), false),
                at(3, 64, count(4, 4), count(4, 4), false),
                at(4, 32, count(4, 4), count(4, 4), false),
                at(5, 32, count(64, 64), count(64, 64), true),
            ],
        );
        let checked: Vec<(u32, bool)> =
            (block.sectors.iter()).map(|s| (s.line, s.agrees)).collect();
        assert_eq!(checked, [(1, true), (2, false), (5, false)]);
    }
}
