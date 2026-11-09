mod graph;
#[cfg(feature = "onnx")]
mod onnx;

pub use self::graph::compile_graph;
#[cfg(feature = "onnx")]
pub use self::onnx::compile_onnx_model;
