// Where a case's snapshot is: `SINGE_SNAPSHOT_<NAME>` when it is set, or else the snapshot in the Hugging Face cache of the repository and revision `snapshots.toml` gives the name, as `support.py` finds it.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result};
use toml_edit::DocumentMut;

// The directory of snapshot `name`, or `None` when neither the variable nor the cache has it.
pub fn find(root: &Path, name: &str) -> Result<Option<PathBuf>> {
    if let Some(directory) = env::var_os(format!("SINGE_SNAPSHOT_{name}")) {
        return Ok(Some(PathBuf::from(directory)));
    }

    let path = root.join("snapshots.toml");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let document = text
        .parse::<DocumentMut>()
        .with_context(|| format!("parse {}", path.display()))?;
    // An entry is the repository; a table of its `repository`, the `revision` to read, such as the safetensors conversion bot's `refs/pr/N`, and the `files` to download, which only the download reads; or a pipeline table of a `pipeline` class, a `base` entry, and the `components` entries added to it.
    let entry = document
        .get("snapshots")
        .and_then(|snapshots| snapshots.get(name))
        .with_context(|| format!("{} names no snapshot {name}", path.display()))?;
    if let Some(pipeline) = entry.get("pipeline").and_then(|item| item.as_str()) {
        let base = entry
            .get("base")
            .and_then(|item| item.as_str())
            .with_context(|| format!("pipeline {name} names no base snapshot"))?;
        let added = entry
            .get("components")
            .and_then(|item| item.as_table_like())
            .map(|components| {
                components
                    .iter()
                    .map(|(component, part)| {
                        (
                            component.to_owned(),
                            part.as_str().unwrap_or_default().to_owned(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        return assemble(root, name, pipeline, base, &added);
    }
    let (repository, revision) = match entry.as_str() {
        Some(repository) => (repository, "main"),
        None => {
            let table = entry
                .as_table_like()
                .with_context(|| format!("snapshot {name} is neither a repository nor a table"))?;
            (
                table
                    .get("repository")
                    .and_then(|item| item.as_str())
                    .with_context(|| format!("snapshot {name} names no repository"))?,
                table
                    .get("revision")
                    .and_then(|item| item.as_str())
                    .unwrap_or("main"),
            )
        }
    };
    Ok(cached(repository, revision))
}

// The Hugging Face cache, as `huggingface_hub` finds it: `HF_HUB_CACHE`, or `hub` under `HF_HOME` or `~/.cache/huggingface`.
fn cache() -> Option<PathBuf> {
    env::var_os("HF_HUB_CACHE").map(PathBuf::from).or_else(|| {
        env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| Path::new(&home).join(".cache/huggingface")))
            .map(|home| home.join("hub"))
    })
}

// The cached snapshot of `repository` that `refs/<revision>` names, as `huggingface_hub` resolves a revision offline, such as `refs/refs/pr/3` for `refs/pr/3`.
fn cached(repository: &str, revision: &str) -> Option<PathBuf> {
    let directory = cache()?.join(format!("models--{}", repository.replace('/', "--")));
    let commit = fs::read_to_string(directory.join("refs").join(revision)).ok()?;
    let snapshot = directory.join("snapshots").join(commit.trim());
    snapshot.is_dir().then_some(snapshot)
}

// A `diffusers` pipeline assembled from snapshots, as `support.py` assembles it: `base`'s components and the `added` snapshots, linked under the tests' temporary directory beside a `model_index.json` naming `pipeline`; `None` when a part is missing.
fn assemble(
    root: &Path,
    name: &str,
    pipeline: &str,
    base: &str,
    added: &[(String, String)],
) -> Result<Option<PathBuf>> {
    let Some(base) = find(root, base)? else {
        return Ok(None);
    };
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("snapshots")
        .join(name);
    fs::create_dir_all(&directory).with_context(|| format!("create {}", directory.display()))?;
    let read = |path: &Path| -> Result<serde_json::Value> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    };
    let mut index = read(&base.join("model_index.json"))?;
    index["_class_name"] = pipeline.into();
    let mut parts = fs::read_dir(&base)
        .with_context(|| format!("read {}", base.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .filter_map(|path| Some((path.file_name()?.to_string_lossy().into_owned(), path)))
        .collect::<Vec<_>>();
    for (component, part) in added {
        let Some(part) = find(root, part)? else {
            return Ok(None);
        };
        let class = read(&part.join("config.json"))?["_class_name"].clone();
        index[component.as_str()] = serde_json::json!(["diffusers", class]);
        parts.push((component.clone(), part));
    }
    // Components the base names but its download left out, such as a safety checker, are dropped.
    if let Some(entries) = index.as_object_mut() {
        entries.retain(|key, value| {
            !value.is_array() || parts.iter().any(|(component, _)| component == key)
        });
    }
    for (component, part) in parts {
        let link = directory.join(&component);
        if link.symlink_metadata().is_ok() {
            fs::remove_file(&link).with_context(|| format!("remove {}", link.display()))?;
        }
        let target =
            fs::canonicalize(&part).with_context(|| format!("resolve {}", part.display()))?;
        std::os::unix::fs::symlink(&target, &link)
            .with_context(|| format!("link {}", link.display()))?;
    }
    let path = directory.join("model_index.json");
    fs::write(&path, format!("{index:#}\n"))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(Some(directory))
}
