// Singe's own format: `config.json` holds `{"singe": {"model": name, "head": head, "config": …, "stop_tokens": […]}}`, and each part's state is `{part}.safetensors` under Singe's names.
// Tokenizer and processor files carry over as the source stores them, since processors read them as they are.

use std::{fs, path::Path};

use serde_json::{Value, json};
use singe::nn::ModelState;

use crate::{
    error::{Error, Result},
    head::Head,
    source::Source,
};

// Files that hold another format's weights, or its model config at the top, which the native files replace.
fn replaced(name: &str, top: bool) -> bool {
    (top && name == "config.json")
        || [
            ".safetensors",
            ".safetensors.index.json",
            ".bin",
            ".pt",
            ".pth",
            ".h5",
            ".msgpack",
            ".onnx",
            ".gguf",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

pub(crate) fn save(
    source: &Source,
    directory: &Path,
    (model, head): (&str, Head),
    config: &Value,
    parts: &[(String, ModelState)],
) -> Result<()> {
    fs::create_dir_all(directory).map_err(Error::io(directory))?;
    // Copying into the source would copy each file onto itself or recurse into its own copy.
    let canonical = |path: &Path| fs::canonicalize(path).map_err(Error::io(path));
    if canonical(directory)?.starts_with(canonical(source.directory())?) {
        return Err(Error::unsupported(
            "saving a model inside its source directory",
        ));
    }

    copy_files(source.directory(), directory, true)?;
    // The stop tokens, which the source may name only in the config this one replaces.
    let file = json!({"singe": {
        "model": model,
        "head": head.id(),
        "config": config,
        "stop_tokens": source.stop_tokens()?,
    }});
    let text =
        serde_json::to_string_pretty(&file).map_err(Error::json("write the native config"))?;
    let path = directory.join("config.json");
    fs::write(&path, text).map_err(Error::io(&path))?;
    for (part, state) in parts {
        state.save_safetensors(directory.join(format!("{part}.safetensors")))?;
    }
    Ok(())
}

// Copies every file of `from` but those `replaced` names, keeping subdirectories, such as a sentence-transformers pooling config.
fn copy_files(from: &Path, to: &Path, top: bool) -> Result<()> {
    for entry in fs::read_dir(from).map_err(Error::io(from))? {
        let entry = entry.map_err(Error::io(from))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let path = entry.path();
        let target = to.join(name);
        if path.is_dir() {
            if !name.starts_with('.') {
                fs::create_dir_all(&target).map_err(Error::io(&target))?;
                copy_files(&path, &target, false)?;
            }
        } else if !replaced(name, top) {
            fs::copy(&path, &target).map_err(Error::io(&path))?;
        }
    }
    Ok(())
}
