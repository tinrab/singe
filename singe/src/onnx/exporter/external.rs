// Weights in a side file beside the model, as ONNX's external data stores them.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

use prost::Message;
use singe_onnx::proto::onnx::{ModelProto, StringStringEntryProto, tensor_proto::DataLocation};

use crate::{
    graph::{GraphProgram, StateSnapshot},
    onnx::{
        error::{Error, Result},
        exporter::{model, proto::unsupported},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightPlacement {
    #[default]
    Automatic,
    Inline,
    Beside,
}

const LARGE_MODEL: usize = 3 << 29;

const SMALL_TENSOR: usize = 1 << 10;

const ALIGNED_TENSOR: usize = 1 << 20;
const ALIGNMENT: u64 = 1 << 16;

const MESSAGE_LIMIT: usize = i32::MAX as usize;

pub fn export_file(
    program: &GraphProgram,
    state: &StateSnapshot,
    path: &Path,
    weights: WeightPlacement,
) -> Result<()> {
    let mut model = model(program, state)?;
    let size = model
        .graph
        .iter()
        .flat_map(|graph| &graph.initializer)
        .map(|tensor| tensor.raw_data.len())
        .sum::<usize>();
    let beside = match weights {
        WeightPlacement::Automatic => size > LARGE_MODEL,
        WeightPlacement::Inline => false,
        WeightPlacement::Beside => true,
    };
    if beside {
        let name = path
            .file_name()
            .ok_or_else(|| unsupported("an export path without a file name"))?;
        let location = format!("{}_data", name.to_string_lossy());
        write_beside(&mut model, &path.with_file_name(&location), &location)?;
    }
    fs::write(path, encoded(&model)?).map_err(|source| Error::ExportWrite {
        path: path.into(),
        source,
    })
}

pub fn encoded(model: &ModelProto) -> Result<Vec<u8>> {
    if model.encoded_len() > MESSAGE_LIMIT {
        return Err(unsupported(
            "a model over protobuf's 2 GiB limit, which needs its weights in external data",
        ));
    }
    Ok(model.encode_to_vec())
}

fn write_beside(model: &mut ModelProto, path: &Path, location: &str) -> Result<()> {
    let failed = |source| Error::ExportWrite {
        path: path.into(),
        source,
    };
    let mut writer = BufWriter::new(File::create(path).map_err(failed)?);
    let mut offset = 0_u64;
    let initializers = model
        .graph
        .iter_mut()
        .flat_map(|graph| &mut graph.initializer)
        .filter(|tensor| tensor.raw_data.len() >= SMALL_TENSOR);
    for tensor in initializers {
        let length = tensor.raw_data.len();
        if length > ALIGNED_TENSOR {
            let aligned = offset.next_multiple_of(ALIGNMENT);
            let padding = usize::try_from(aligned - offset)
                .map_err(|_| unsupported("external data padding"))?;
            writer.write_all(&vec![0; padding]).map_err(failed)?;
            offset = aligned;
        }

        writer.write_all(&tensor.raw_data).map_err(failed)?;
        let entry = |key: &str, value: String| StringStringEntryProto {
            key: key.into(),
            value,
        };
        tensor.external_data = vec![
            entry("location", location.into()),
            entry("offset", offset.to_string()),
            entry("length", length.to_string()),
        ];
        tensor.data_location = DataLocation::External as i32;
        tensor.raw_data = Vec::new();
        offset += u64::try_from(length).map_err(|_| unsupported("external data length"))?;
    }
    writer.flush().map_err(failed)
}
