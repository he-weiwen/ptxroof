//! Caller-supplied parameter bindings and their textual syntax.
//! Shared by report construction, the CLI and measured-record import.

/// One `--bind` argument: `name=value` or `idx:name=value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingSpec {
    pub index: Option<usize>,
    pub name: String,
    pub value: i64,
}

pub fn parse_bind(text: &str) -> Result<BindingSpec, String> {
    let (lhs, value) = text
        .split_once('=')
        .ok_or_else(|| format!("`{text}`: expected name=value or idx:name=value"))?;
    let value: i64 = value
        .parse()
        .map_err(|_| format!("`{text}`: value `{value}` is not an integer"))?;
    let (index, name) = match lhs.split_once(':') {
        Some((idx, name)) => {
            let idx = idx
                .parse()
                .map_err(|_| format!("`{text}`: index `{idx}` is not a number"))?;
            (Some(idx), name)
        }
        None => (None, lhs),
    };
    if name.is_empty() {
        return Err(format!("`{text}`: empty parameter name"));
    }
    Ok(BindingSpec {
        index,
        name: name.to_owned(),
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_parsing_grammar() {
        assert_eq!(
            parse_bind("K=4096"),
            Ok(BindingSpec {
                index: None,
                name: "K".into(),
                value: 4096
            })
        );
        assert_eq!(
            parse_bind("2:K=4096"),
            Ok(BindingSpec {
                index: Some(2),
                name: "K".into(),
                value: 4096
            })
        );
        assert!(parse_bind("K").is_err());
        assert!(parse_bind("K=x").is_err());
        assert!(parse_bind("=4").is_err());
        assert!(parse_bind("a:K=4").is_err());
    }
}
