//! Path display helpers.

pub(crate) fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}
