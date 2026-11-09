use std::{io::Read, path::Path};

use singe_onnx::{read as read_onnx, read_path as read_onnx_path};

use crate::onnx::{
    error::Result,
    importer::{ImportedModel, import_model},
    model::OnnxModel,
    options::OnnxOptions,
};

impl OnnxModel {
    /// Reads, imports, and validates an ONNX model from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the ONNX model cannot be parsed, import encounters unsupported ONNX semantics, or graph construction fails.
    pub fn create(path: impl AsRef<Path>, options: OnnxOptions) -> Result<Self> {
        let path = path.as_ref();
        let model = read_onnx_path(path)?;
        let model_dir = path.parent().map(Path::to_path_buf);
        Self::from_imported(import_model(model, options, model_dir)?)
    }

    /// Reads and imports an ONNX model from a reader.
    ///
    /// External tensor data cannot be resolved from this constructor.
    /// Use [`OnnxModel::create`] for models with external data.
    ///
    /// # Errors
    ///
    /// Returns an error if reading, parsing, import, or graph construction fails.
    pub fn from_reader(mut reader: impl Read, options: OnnxOptions) -> Result<Self> {
        let model = read_onnx(&mut reader)?;
        Self::from_imported(import_model(model, options, None)?)
    }

    fn from_imported(imported: ImportedModel) -> Result<Self> {
        let program = imported.graph.program()?;
        let initializers = program.bind_state(&imported.graph.initializers)?;
        let program = program.specialize_state(imported.specialized_initializers, &initializers)?;
        Ok(Self {
            program,
            initializers,
            device: imported.device,
            inputs: imported.inputs,
            outputs: imported.outputs.clone(),
            output_names: imported
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
            metadata: imported.metadata,
        })
    }
}

#[cfg(test)]
mod tests;
