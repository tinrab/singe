//! ONNX model loading and execution for Singe.

mod attributes;
mod error;
mod export_contract;
mod exporter;
mod importer;
mod model;
mod ops;
mod options;
mod shape;
mod tensor;
mod utility;

pub use crate::onnx::{
    error::{Error, Result},
    exporter::{
        compare::{Divergence, first_divergence},
        export,
        external::{WeightPlacement, export_file},
    },
    model::{OnnxModel, OnnxModelMetadata, OnnxValueInfo},
    options::OnnxOptions,
};
