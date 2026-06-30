use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn workspace_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask manifest is missing workspace parent")
}

pub fn write_generated_file(path: &Path, contents: impl ToString) -> Result<()> {
    let contents = contents.to_string();
    if fs::read_to_string(path).is_ok_and(|current| current == contents) {
        return Ok(());
    }

    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))?;

    println!("updated {}", path.display());

    Ok(())
}
