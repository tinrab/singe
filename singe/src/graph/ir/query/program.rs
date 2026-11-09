#[cfg(feature = "onnx")]
use crate::graph::shape_program::ShapeProgram;
use crate::graph::{CaptureInfo, generators::GraphGenerator, ir::Graph};

impl Graph {
    /// Returns the public name of output `index`, if it has one.
    pub(crate) fn output_name(&self, index: usize) -> Option<&str> {
        self.output_names.get(index).and_then(Option::as_deref)
    }

    /// Returns the symbolic shape values, reshape expressions, and constraints that compilation resolves.
    #[cfg(feature = "onnx")]
    pub(crate) fn shape_program(&self) -> &ShapeProgram {
        &self.shape_program
    }

    /// Returns the inputs that the trailing outputs, in order, write back after each run.
    pub(crate) fn input_writes(&self) -> &[String] {
        &self.input_writes
    }

    pub(crate) fn counted_outputs(&self) -> &[usize] {
        &self.counted_outputs
    }

    /// Returns the generators the graph reads through inputs.
    pub(crate) fn generators(&self) -> &[GraphGenerator] {
        &self.generators
    }

    /// Returns the observation contract when this graph originated in eager capture.
    #[must_use]
    pub fn capture_info(&self) -> Option<CaptureInfo> {
        self.capture_info
    }
}
