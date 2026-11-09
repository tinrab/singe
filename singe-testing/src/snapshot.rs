use std::{env, path::PathBuf};

pub const SNAPSHOT_IGNORED: &str = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>";

#[must_use]
pub fn snapshot(name: &str) -> PathBuf {
    let variable = format!("SINGE_SNAPSHOT_{name}");
    env::var_os(&variable).map_or_else(
        || panic!("run the ignored tests with {variable} set"),
        PathBuf::from,
    )
}

#[must_use]
pub fn snapshot_if_set(name: &str) -> Option<PathBuf> {
    env::var_os(format!("SINGE_SNAPSHOT_{name}")).map(PathBuf::from)
}
