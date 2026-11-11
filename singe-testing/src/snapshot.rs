use std::{env, path::PathBuf};

#[must_use]
pub fn snapshot_if_set(name: &str) -> Option<PathBuf> {
    env::var_os(format!("SINGE_SNAPSHOT_{name}")).map(PathBuf::from)
}
