use crate::{
    backend::registry::{Conv2dAttributes, Operation, OperationAttributes, Pool2dAttributes},
    graph::{
        ir::ValueId,
        operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    },
};

impl GraphOperationRequest {
    /// Creates a convolution request with input, filter, and optional bias operands.
    /// Stride and dilation use height and width order; padding uses height start, height end, width start, and width end order.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn conv2d(
        name: Option<String>,
        input: ValueId,
        weight: ValueId,
        bias: Option<ValueId>,
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
        groups: i64,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                Operation::Conv2d,
                OperationAttributes::Conv2d(Conv2dAttributes {
                    has_bias: bias.is_some(),
                    stride,
                    padding,
                    dilation,
                    groups,
                }),
                [Some(input), Some(weight), bias]
                    .into_iter()
                    .flatten()
                    .collect(),
            ),
            output,
        }
    }

    /// Creates a pooling request with explicit kernel, stride, padding, dilation, and ceiling-mode attributes.
    /// The include-padding divisor flag is retained only for average pooling.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn pool2d(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        kernel_size: (i64, i64),
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
        ceil_mode: bool,
        count_include_pad: bool,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::Pool2d(Pool2dAttributes {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    count_include_pad: (operation == Operation::AvgPool2d)
                        .then_some(count_include_pad),
                }),
                vec![input],
            ),
            output,
        }
    }
}
