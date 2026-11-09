use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use singe_onnx::Tensor;

use crate::onnx::error::{Error, Result};

pub fn read_external_data(
    tensor: &Tensor,
    name: &str,
    model_dir: Option<&Path>,
) -> Result<Vec<u8>> {
    let Some(model_dir) = model_dir else {
        return Err(Error::invalid_initializer(
            name,
            "external data requires loading from a model path",
        ));
    };
    let location = tensor
        .external_data
        .get("location")
        .ok_or_else(|| Error::invalid_initializer(name, "external data missing location"))?;
    let path = external_data_path(name, model_dir, location)?;
    let bytes = fs::read(&path).map_err(|error| Error::InvalidExternalData {
        name: name.into(),
        path: path.clone(),
        reason: error.to_string(),
    })?;
    let offset = tensor
        .external_data
        .get("offset")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| Error::InvalidExternalData {
            name: name.into(),
            path: path.clone(),
            reason: error.to_string(),
        })?
        .unwrap_or(0);
    let length = tensor
        .external_data
        .get("length")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| Error::InvalidExternalData {
            name: name.into(),
            path: path.clone(),
            reason: error.to_string(),
        })?
        .unwrap_or_else(|| bytes.len().saturating_sub(offset));
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::InvalidExternalData {
            name: name.into(),
            path: path.clone(),
            reason: "external data range overflow".into(),
        })?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::InvalidExternalData {
            name: name.into(),
            path,
            reason: "external data range out of bounds".into(),
        })?;
    Ok(slice.to_vec())
}

pub fn external_data_path(name: &str, model_dir: &Path, location: &str) -> Result<PathBuf> {
    let location_path = Path::new(location);
    if location_path.is_absolute()
        || location_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::Prefix(_) | Component::RootDir
            )
        })
    {
        return Err(Error::InvalidExternalData {
            name: name.into(),
            path: location_path.into(),
            reason: "external data location must stay within the model directory".into(),
        });
    }

    let model_dir = model_dir
        .canonicalize()
        .map_err(|error| Error::InvalidExternalData {
            name: name.into(),
            path: model_dir.into(),
            reason: error.to_string(),
        })?;
    let path = model_dir.join(location_path);
    let canonical_path = path
        .canonicalize()
        .map_err(|error| Error::InvalidExternalData {
            name: name.into(),
            path: path.clone(),
            reason: error.to_string(),
        })?;
    if !canonical_path.starts_with(&model_dir) {
        return Err(Error::InvalidExternalData {
            name: name.into(),
            path,
            reason: "external data location must stay within the model directory".into(),
        });
    }
    Ok(canonical_path)
}
