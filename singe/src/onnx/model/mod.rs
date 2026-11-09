mod load;

use std::collections::{BTreeMap, HashMap};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error as SingeError, Result as SingeResult},
    graph::{
        CompiledGraph, GraphExecutionMode, GraphExecutor, GraphPrepareOptions, GraphProgram,
        ir::Graph,
    },
    onnx::error::{Error, Result},
    shape::Shape,
    state::StateSnapshot,
    tensor::Tensor,
};

/// ONNX model imported into a Singe graph.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    program: GraphProgram,
    initializers: StateSnapshot,
    device: Device,
    inputs: Vec<OnnxValueInfo>,
    outputs: Vec<OnnxValueInfo>,
    output_names: Vec<String>,
    metadata: OnnxModelMetadata,
}

impl OnnxModel {
    /// Prepares the imported program and initializer state for execution using `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if state binding, device placement, graph specialization, backend compilation, or preparation policy validation fails.
    pub fn prepare(&self, options: GraphPrepareOptions) -> SingeResult<GraphExecutor> {
        self.program.prepare(&self.initializers, options)
    }

    /// Returns the logical signature and operations independently of initializer storage.
    #[must_use]
    pub fn program(&self) -> &GraphProgram {
        &self.program
    }

    /// Returns the immutable initializer publication, stored on CPU after import.
    #[must_use]
    pub fn initializers(&self) -> &StateSnapshot {
        &self.initializers
    }

    /// Prepares the imported graph for repeated interpreted execution.
    /// Shape expressions are resolved for the import representatives; different dimensions are rejected.
    /// Use `prepare` for runtime shape specialization.
    ///
    /// # Errors
    ///
    /// Returns an error if initializer binding, device placement, graph normalization, shape resolution, or interpreted compilation fails.
    // The model's graph with its initializers bound, as exports take it.
    pub fn graph(&self) -> SingeResult<Graph> {
        self.program.bound_graph(&self.initializers)
    }

    pub fn compile(&self) -> SingeResult<CompiledGraph> {
        self.program
            .bound_graph(&self.initializers)?
            .placed_on(self.device)?
            .compile()
    }

    /// Projects the imported graph into explicit graph IR.
    ///
    /// # Errors
    ///
    /// Returns an error if the imported graph is internally inconsistent.
    pub fn to_ir(&self) -> Result<Graph> {
        Ok(self.program.bound_graph(&self.initializers)?.to_ir()?)
    }

    /// Returns imported graph input metadata.
    #[must_use]
    pub fn inputs(&self) -> &[OnnxValueInfo] {
        &self.inputs
    }

    /// Returns imported graph output metadata.
    #[must_use]
    pub fn outputs(&self) -> &[OnnxValueInfo] {
        &self.outputs
    }

    /// Returns imported model metadata.
    #[must_use]
    pub fn metadata(&self) -> &OnnxModelMetadata {
        &self.metadata
    }

    /// Runs the model with named Singe tensors and returns named outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if graph input validation or execution fails.
    pub fn run(&self, inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let outputs = self.run_ordered(inputs)?;
        if outputs.len() != self.output_names.len() {
            return Err(Error::Singe(SingeError::missing_output("model run")));
        }
        Ok(self.output_names.iter().cloned().zip(outputs).collect())
    }

    /// Runs the model and returns outputs in ONNX graph output order.
    ///
    /// # Errors
    ///
    /// Returns an error if graph input validation or execution fails.
    pub fn run_ordered(&self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let mut executor = self.prepare(
            GraphPrepareOptions::new(self.device).with_execution_mode(GraphExecutionMode::Eager),
        )?;
        Ok(executor.run(inputs)?)
    }
}

/// Imported ONNX model metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxModelMetadata {
    /// ONNX IR version.
    pub ir_version: i64,
    /// Model producer name.
    pub producer_name: String,
    /// Model producer version.
    pub producer_version: String,
    /// Model domain.
    pub domain: Option<String>,
    /// Model version.
    pub model_version: i64,
    /// Imported opset versions by domain.
    pub operator_sets: BTreeMap<Option<String>, i64>,
}

/// Tensor value information exposed by an imported ONNX model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxValueInfo {
    /// ONNX value name.
    pub name: String,
    /// Singe tensor data type.
    pub data_type: DataType,
    /// Singe tensor shape.
    pub shape: Shape,
    /// Whether this graph input has an initializer default.
    pub has_initializer: bool,
}
