use crate::analysis::instruction_counts::count_constraint::CountConstraint;
use crate::ptx::ir::Module;
use crate::report::build::BindingSpec;
use crate::report::schema::{MeasuredBlock, Verdict};
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
}

pub struct Prediction {
    pub line: u32,
    pub constraint: CountConstraint,
    pub executions: Option<u64>,
    pub at_most: bool,
}

impl NcuRecord {
    pub fn bindings(&self) -> Vec<BindingSpec> {
        let value = |param: &str| {
            param
                .strip_prefix("u32:")?
                .parse::<u32>()
                .ok()
                .map(i64::from)
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
        let (mut agreeing, mut confirmed, mut contradicted) = (false, false, false);
        for p in predictions {
            let (Some(measured), Some(predicted)) = (self.lines.get(&p.line), p.executions) else {
                continue;
            };
            let agrees = match p.at_most {
                true => measured.executions <= predicted,
                false => measured.executions == predicted,
            };
            let tested = p.constraint != CountConstraint::Free && measured.sass.len() == 1;
            agreeing |= agrees;
            confirmed |= tested && agrees && !p.at_most;
            contradicted |= tested && !agrees;
        }
        let verdict = match (contradicted, confirmed, agreeing) {
            (true, _, _) => Verdict::Contradicted,
            (_, true, _) => Verdict::Confirmed,
            (_, _, true) => Verdict::Consistent,
            _ => Verdict::Unchecked,
        };
        MeasuredBlock {
            name: block.to_owned(),
            verdict,
        }
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
        }
    }

    fn verdict(record: &NcuRecord, predictions: &[Prediction]) -> Verdict {
        record.compare("b", predictions).verdict
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
    fn lines_without_sass_or_a_numeric_prediction_leave_a_block_unchecked() {
        let r = record(&[(1, &["FFMA"], 64)]);
        let symbolic = Prediction {
            line: 1,
            constraint: Constrained,
            executions: None,
            at_most: false,
        };
        assert_eq!(
            verdict(&r, &[predict(2, Constrained, 64, false)]),
            Verdict::Unchecked
        );
        assert_eq!(verdict(&r, &[symbolic]), Verdict::Unchecked);
    }
}
