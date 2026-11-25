use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
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
    let failed = |source| Error::ExternalDataIo {
        name: name.into(),
        path: path.clone(),
        source,
    };
    // Only the tensor's range is read, since one file holds every tensor of a large model.
    let mut file = File::open(&path).map_err(failed)?;
    let size = usize::try_from(file.metadata().map_err(failed)?.len()).unwrap_or(usize::MAX);
    let entry = |key: &'static str| {
        tensor
            .external_data
            .get(key)
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|source| Error::ExternalDataEntry {
                name: name.into(),
                key,
                source,
            })
    };
    let offset = entry("offset")?.unwrap_or(0);
    let length = entry("length")?.unwrap_or_else(|| size.saturating_sub(offset));
    if offset.checked_add(length).is_none_or(|end| end > size) {
        return Err(Error::ExternalDataRange {
            name: name.into(),
            path,
            offset,
            length,
        });
    }

    let mut bytes = vec![0; length];
    file.seek(SeekFrom::Start(offset as u64)).map_err(failed)?;
    file.read_exact(&mut bytes).map_err(failed)?;
    Ok(bytes)
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
        return Err(Error::ExternalDataOutside {
            name: name.into(),
            path: location_path.into(),
        });
    }

    let model_dir = model_dir
        .canonicalize()
        .map_err(|source| Error::ExternalDataIo {
            name: name.into(),
            path: model_dir.into(),
            source,
        })?;
    let path = model_dir.join(location_path);
    let canonical_path = path
        .canonicalize()
        .map_err(|source| Error::ExternalDataIo {
            name: name.into(),
            path: path.clone(),
            source,
        })?;
    if !canonical_path.starts_with(&model_dir) {
        return Err(Error::ExternalDataOutside {
            name: name.into(),
            path,
        });
    }
    Ok(canonical_path)
}
