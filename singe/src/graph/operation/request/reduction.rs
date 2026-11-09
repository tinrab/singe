use crate::{
    backend::registry::{Operation, OperationAttributes, ReductionAttributes, TopKAttributes},
    graph::{
        ir::ValueId,
        operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    },
    scalar::Scalar,
};

impl GraphOperationRequest {
    /// Creates a single-input request with a signed dimension attribute.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn dimension(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        dim: i64,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::Dimension(dim),
                vec![input],
            ),
            output,
        }
    }

    /// Creates a single-input request with a scalar argument converted to the canonical scalar representation.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    pub fn scalar(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        scalar: impl Into<Scalar>,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::Scalar(scalar.into()),
                vec![input],
            ),
            output,
        }
    }

    /// Creates a reduction request with selected dimensions and the reduced-axis retention flag.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn reduction(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        dimensions: Vec<i64>,
        keepdim: bool,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::Reduction(ReductionAttributes {
                    dimensions,
                    keepdim,
                }),
                vec![input],
            ),
            output,
        }
    }

    /// Creates a top-k request with a dimension, result count, direction, and sorting flag.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn topk(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::TopK(TopKAttributes {
                    k,
                    dim,
                    largest,
                    sorted,
                }),
                vec![input],
            ),
            output,
        }
    }
}
