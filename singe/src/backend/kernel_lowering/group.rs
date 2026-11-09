//! Element-wise groups: the operations one kernel computes from its loaded inputs, and where it stores their values.

use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    layout::Layout,
    shape::Shape,
};

/// Where one operand of a fused step comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOperand {
    /// The kernel input in this slot.
    Input(usize),
    /// The value an earlier step computed.
    Step(usize),
}

/// What a fused kernel stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusedStores {
    /// The last step's value, into a new contiguous output of the kernel's shape.
    Last,
    /// The value of each of these steps, in order, into a new contiguous output of the kernel's shape each.
    Steps(Vec<usize>),
    /// Each value, in order, into consecutive slices of the kernel's shape along `axis` of a new contiguous output, as a concatenation of equal shapes lays them out.
    Concat {
        axis: usize,
        values: Vec<FusedOperand>,
    },
}

/// One element-wise operation of a fused kernel.
#[derive(Debug, Clone)]
pub struct FusedStep<'a> {
    pub operation: Operation,
    pub attributes: &'a OperationAttributes,
    pub operands: Vec<FusedOperand>,
}

/// Returns whether `operation` can be a step of a fused element-wise kernel.
///
/// Each step keeps its own data type, so comparisons, logical operations, `where`, and casts join chains like the arithmetic does.
#[must_use]
pub fn is_fusible(operation: Operation) -> bool {
    operation.is_binary_elementwise()
        || matches!(
            operation,
            Operation::Clamp | Operation::Lerp | Operation::Where | Operation::Cast
        )
        || operation.is_scalar_elementwise()
        || (operation.is_unary_elementwise()
            && !matches!(operation, Operation::Copy | Operation::Contiguous))
}

/// Returns whether a row lowering computes `operation` with `attributes`: a layer or RMS normalization, a softmax or log-softmax, a sum, mean, maximum, minimum, or product, or the position of a maximum or minimum.
#[must_use]
pub fn is_row_operation(operation: Operation, attributes: &OperationAttributes) -> bool {
    is_row_opcode(operation)
        && (attributes.layer_normalization().is_some()
            || attributes.reduction().is_some()
            || matches!(
                operation,
                Operation::Softmax
                    | Operation::LogSoftmax
                    | Operation::ArgMax
                    | Operation::ArgMin
                    | Operation::Cumsum
                    | Operation::Cumprod
            ))
}

/// Returns whether `operation` is one a row lowering can compute, given matching attributes.
#[must_use]
pub const fn is_row_opcode(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::LayerNorm
            | Operation::RmsNorm
            | Operation::Softmax
            | Operation::LogSoftmax
            | Operation::Sum
            | Operation::Mean
            | Operation::Max
            | Operation::Min
            | Operation::Product
            | Operation::LogSumExp
            | Operation::ArgMax
            | Operation::ArgMin
            | Operation::Cumsum
            | Operation::Cumprod
    )
}

/// The value a row lowering reads: an element-wise group computed in the kernel, or one input when the group has no steps.
#[derive(Debug, Clone)]
pub struct ElementGroup<'a> {
    /// The data type of the group's value.
    pub data_type: DataType,
    /// The shape of the group's value, to which every input broadcasts.
    pub shape: Shape,
    /// The group's inputs, which `steps` name by position.
    pub inputs: Vec<(DataType, &'a Layout)>,
    /// The group's operations; the last one's value is the group's.
    pub steps: &'a [FusedStep<'a>],
    /// Whether the kernel also stores the group's value, into a new contiguous output after its own, because other operations read it.
    pub stored: bool,
}

impl<'a> ElementGroup<'a> {
    /// Returns the group whose value is the input of `data_type` laid out by `layout`.
    #[must_use]
    pub fn input(data_type: DataType, layout: &'a Layout) -> Self {
        Self {
            data_type,
            shape: layout.shape().clone(),
            inputs: vec![(data_type, layout)],
            steps: &[],
            stored: false,
        }
    }
}
