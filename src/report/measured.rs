use crate::ptx::ir::Module;
use crate::report::build::BindingSpec;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct NcuRecord {
    pub kernel: String,
    pub block: [u32; 3],
    pub params: Vec<String>,
    pub lines: BTreeMap<u32, NcuLine>,
}

#[derive(Debug, Deserialize)]
pub struct NcuLine {
    pub ptx: String,
}

impl NcuRecord {
    pub fn bindings(&self) -> Vec<BindingSpec> {
        let value = |param: &str| param.strip_prefix("u32:")?.parse().ok();
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
}

fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
