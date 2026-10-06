use crate::report::build::BindingSpec;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NcuRecord {
    pub block: [u32; 3],
    pub params: Vec<String>,
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
}
