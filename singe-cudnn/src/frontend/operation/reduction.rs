use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::Result,
    execution::reduction::{ReductionDescriptor, ReductionOperation as BackendReductionOperation},
    frontend::{
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::FrontendOperationTensors,
    },
    reduction::ReduceTensorOperator,
    tensor::TensorId,
};

/// Frontend reduction operation.
///
/// cuDNN reductions reduce an input tensor using a mode selected by
/// [`ReduceTensorOperator`]. The reduced dimensions are inferred from the input and
/// output tensor shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionOperation {
    Reduce {
        op: ReduceTensorOperator,
        input: TensorId,
        output: TensorId,
        compute_type: DataType,
        is_deterministic: bool,
    },
}

impl FrontendOperationTensors for ReductionOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Reduce { input, output, .. } => {
                tensors.extend([*input, *output]);
            }
        }
    }
}

impl ReductionOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::Reduce {
                op,
                input,
                output,
                compute_type,
                is_deterministic,
            } => {
                let descriptor =
                    ReductionDescriptor::create(*op, *compute_type, *is_deterministic)?;
                Ok(LoweredOperation::Reduction(
                    BackendReductionOperation::create(
                        &descriptor,
                        tensor_at(tensors, *input)?,
                        tensor_at(tensors, *output)?,
                    )?,
                ))
            }
        }
    }
}

/// Reduction axes selected by this Rust frontend helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReductionAxes {
    /// Reduce the last dimension.
    Last,
    /// Reduce all dimensions.
    All,
}

/// Attributes for building a frontend reduction operation.
///
/// This maps to the C++ frontend `Reduction_attributes` mode and compute data
/// type setters, with Rust-side axis and determinism options.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReductionConfig {
    op: ReduceTensorOperator,
    compute_type: DataType,
    axes: ReductionAxes,
    is_deterministic: bool,
}

impl ReductionConfig {
    pub fn new(op: ReduceTensorOperator, compute_type: DataType) -> Self {
        Self {
            op,
            compute_type,
            axes: ReductionAxes::Last,
            is_deterministic: false,
        }
    }

    pub fn with_axes(mut self, axes: ReductionAxes) -> Self {
        self.axes = axes;
        self
    }

    pub fn with_deterministic(mut self) -> Self {
        self.is_deterministic = true;
        self
    }

    pub fn op(&self) -> ReduceTensorOperator {
        self.op
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn axes(&self) -> ReductionAxes {
        self.axes
    }

    pub fn is_deterministic(&self) -> bool {
        self.is_deterministic
    }
}
