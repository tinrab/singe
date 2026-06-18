use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn workspace_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask manifest is missing workspace parent")
}
