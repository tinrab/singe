use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

/// Returns the parent directory of the xtask crate's manifest.
///
/// # Errors
///
/// Returns an error if the manifest directory has no parent.
pub fn workspace_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask manifest is missing workspace parent")
}

/// Writes `contents` and creates parent directories if the file differs or cannot be read.
///
/// # Errors
///
/// Returns an error if a parent directory cannot be created or the file cannot be written.
///
/// # Panics
///
/// Panics if `path` has no parent directory.
pub fn write_generated_file(path: &Path, contents: &str) -> Result<()> {
    if fs::read_to_string(path).is_ok_and(|current| current == contents) {
        return Ok(());
    }

    fs::create_dir_all(
        path.parent()
            .expect("output paths always have a parent directory"),
    )?;
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))?;

    println!("updated {}", path.display());

    Ok(())
}
