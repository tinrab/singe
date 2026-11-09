//! Core tensor, graph, storage, and backend abstractions for Singe.
//!
//! This crate owns the high-level tensor interface plus CPU and CUDA backend integration.
//!
//! Scalars have rank zero, with shape `[]`.
//! Logical shapes and physical layouts are separate, and CPU allocations retain their scalar type and alignment.

/// Reverse-mode automatic differentiation.
pub mod autograd;

/// CPU and CUDA backends, operation schemas, and kernel registration.
pub mod backend;

/// Tensor element data types.
pub mod data_type;

/// Tensor storage and execution devices.
pub mod device;

pub mod dispatch;

/// Crate error and result types.
pub mod error;

/// Token generation over decoders that keep their own caches.
pub mod generation;

/// Captured tensor operation graphs.
pub mod graph;

/// Tensor indexing and slicing helpers.
pub mod index;

/// Tensor layout metadata.
pub mod layout;

/// Composable inference layers and named checkpoint state.
pub mod nn;

/// ONNX model loading and execution.
#[cfg(feature = "onnx")]
pub mod onnx;

/// Optimizers that update parameters from accumulated gradients.
pub mod optim;

pub mod parallel;

pub mod random;

/// Typed scalar arguments and conversion semantics.
pub mod scalar;

/// Logical tensor dimensions and element counts.
pub mod shape;

/// Immutable named state shared by layers and graph execution.
pub mod state;

pub mod signal;

/// Backend storage allocation abstractions.
pub mod storage;

/// High-level tensor API.
pub mod tensor;

/// Framework activity counters and CUDA test helpers.
#[cfg(feature = "testing")]
pub mod testing;

/// Storage-free logical value specifications.
pub mod value;

mod activity;
mod observation;
mod utility;

#[cfg(feature = "onnx")]
pub use self::onnx::{OnnxModel, OnnxModelMetadata, OnnxOptions, OnnxValueInfo};
pub use self::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel::{num_threads, set_num_threads},
    shape::Shape,
    tensor::{Tensor, TensorDataType, TensorSpec},
};
