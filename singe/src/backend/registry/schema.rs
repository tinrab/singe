use std::iter::repeat;

use crate::{data_type::DataType, device::Device, shape::Shape};

/// Storage-sharing contract between an operation result and its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AliasBehavior {
    /// The result uses storage independent of the inputs.
    None,
    /// The result describes a view of the first input.
    View,
    /// The result may share storage with the first input.
    MayAliasInput,
    /// The operation writes its first input and returns no results.
    WritesInput,
}

/// Dispatch group used to select an operation executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationExecutorFamily {
    /// Basic binary arithmetic and scalar arithmetic operations.
    Binary,
    /// Elementwise operations with one tensor input.
    Unary,
    /// Activation functions and softmax operations.
    Activation,
    /// Binary comparisons and logical operations.
    Comparison,
    /// Binary floating-point operations such as remainder and extrema.
    NumericBinary,
    /// Value reductions over tensor dimensions.
    Reduction,
    /// Conversions, quantization, selection, clamping, and interpolation.
    NumericOther,
    /// Tensor shape, layout, and indexing operations.
    Shape,
    /// Matrix multiplication and tensor contraction.
    TensorAlgebra,
    /// Convolution, pooling, normalization, embeddings, and attention.
    Neural,
    /// Index reductions and top-k selection.
    ArgReduction,
    /// Transfers between devices.
    CopyTo,
    /// Operations registered at run time.
    Custom,
}

/// Semantic role of a tensor operand in registry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// The primary input tensor, or the query tensor for attention.
    Input,
    /// The tensor an operation writes.
    Destination,
    /// The values written into the destination.
    Source,
    /// `F32` quantization scales.
    Scale,
    /// `I64` quantization zero points.
    ZeroPoint,
    /// The `I64` generator state `[seed, offset]`.
    Generator,
    /// The left operand, or values selected when a condition is true.
    Lhs,
    /// The right operand, or values selected when a condition is false.
    Rhs,
    /// The Boolean selection mask.
    Condition,
    /// Learned weights or an embedding table.
    Weight,
    /// The additive bias tensor.
    Bias,
    /// The stored batch-normalization mean.
    RunningMean,
    /// The stored batch-normalization variance.
    RunningVar,
    /// The tensor added to a scaled contraction result.
    Accumulator,
    /// Integer indices used to select tensor values.
    Indices,
    /// Cosine factors for rotary embeddings.
    Cos,
    /// Sine factors for rotary embeddings.
    Sin,
    /// The attention key tensor.
    Key,
    /// The attention value tensor.
    Value,
    /// Values at interpolation weight zero.
    InterpolationStart,
    /// Values at interpolation weight one.
    InterpolationEnd,
    /// Per-head attention sink logits.
    Sinks,
}

/// Allowed tensor operand counts and their ordered semantic roles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperandSchema {
    /// A fixed sequence of operand roles shared by operations.
    Fixed(&'static [OperandRole]),
    /// Required operands followed by one optional operand.
    TrailingOptional {
        /// Roles of operands that must be supplied, in order.
        required: &'static [OperandRole],
        /// Role of the optional final operand.
        optional: OperandRole,
    },
    /// Fixed leading operands followed by repeated operands with one role.
    Variadic {
        /// Roles of the operands before the repeated ones, in order.
        leading: &'static [OperandRole],
        /// Role assigned to every repeated operand.
        role: OperandRole,
        /// Minimum number of repeated operands accepted.
        min: usize,
    },
    /// An owned, fixed sequence of operand roles.
    Concrete(Box<[OperandRole]>),
}

impl OperandSchema {
    /// Returns whether `count` satisfies this schema's operand-count constraint.
    #[must_use]
    pub fn accepts_count(&self, count: usize) -> bool {
        match self {
            Self::Fixed(roles) => roles.len() == count,
            Self::TrailingOptional { required, .. } => {
                count == required.len() || count == required.len() + 1
            }
            Self::Variadic { leading, min, .. } => count >= leading.len() + min,
            Self::Concrete(roles) => roles.len() == count,
        }
    }

    /// Returns up to `count` declared roles, continuing with copies of a variadic role.
    ///
    /// This method does not validate the count.
    /// Fixed and optional schemas stop at the end of their role sequence.
    /// Use [`Self::accepts_count`] to check whether a count is valid.
    #[must_use]
    pub fn roles_for_count(&self, count: usize) -> Vec<OperandRole> {
        match self {
            Self::Fixed(roles) => roles.iter().copied().take(count).collect(),
            Self::TrailingOptional { required, optional } => required
                .iter()
                .copied()
                .chain([*optional])
                .take(count)
                .collect(),
            Self::Variadic { leading, role, .. } => leading
                .iter()
                .copied()
                .chain(repeat(*role))
                .take(count)
                .collect(),
            Self::Concrete(roles) => roles.iter().copied().take(count).collect(),
        }
    }
}

/// The primary input of a unary operation.
pub const UNARY_OPERANDS: &[OperandRole] = &[OperandRole::Input];

/// The written tensor and the values written into it, in that order.
pub const ASSIGN_OPERANDS: &[OperandRole] = &[OperandRole::Destination, OperandRole::Source];

/// The indexed tensor, which `index` operands follow with one `I64` index tensor per indexed dimension.
pub const INDEX_LEADING_OPERANDS: &[OperandRole] = &[OperandRole::Input];

/// The written tensor and the values written, which `index_put` operands follow with one `I64` index tensor per indexed dimension.
pub const INDEX_PUT_LEADING_OPERANDS: &[OperandRole] = &[OperandRole::Input, OperandRole::Source];

// The destination an in-place index write changes, and the values it writes.
pub const INDEX_WRITE_LEADING_OPERANDS: &[OperandRole] =
    &[OperandRole::Destination, OperandRole::Source];

/// A tensor with its quantization scale and zero point, in that order.
pub const QUANTIZE_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Scale,
    OperandRole::ZeroPoint,
];

/// Two quantized matrices followed by each one's scale and zero point.
pub const QUANTIZED_MATMUL_OPERANDS: &[OperandRole] = &[
    OperandRole::Lhs,
    OperandRole::Rhs,
    OperandRole::Scale,
    OperandRole::ZeroPoint,
    OperandRole::Scale,
    OperandRole::ZeroPoint,
];

/// The generator state of a random operation without tensor inputs.
pub const RANDOM_OPERANDS: &[OperandRole] = &[OperandRole::Generator];

/// Probabilities followed by the generator state.
pub const SAMPLING_OPERANDS: &[OperandRole] = &[OperandRole::Input, OperandRole::Generator];

/// A flat base and the values scattered into a copy of it, in that order.
pub const SCATTER_OPERANDS: &[OperandRole] = &[OperandRole::Input, OperandRole::Source];

/// The left and right inputs of a binary operation, in that order.
pub const BINARY_OPERANDS: &[OperandRole] = &[OperandRole::Lhs, OperandRole::Rhs];

/// Selection operands ordered as condition, true values, and false values.
pub const TERNARY_WHERE_OPERANDS: &[OperandRole] =
    &[OperandRole::Condition, OperandRole::Lhs, OperandRole::Rhs];

/// Matrix multiplication operands ordered as left and right matrices.
pub const MATMUL_OPERANDS: &[OperandRole] = &[OperandRole::Lhs, OperandRole::Rhs];

/// Required convolution operands ordered as input and filter weights.
pub const CONV2D_OPERANDS: &[OperandRole] = &[OperandRole::Input, OperandRole::Weight];

/// Contraction operands ordered as left input, right input, and accumulator.
pub const CONTRACTION_OPERANDS: &[OperandRole] =
    &[OperandRole::Lhs, OperandRole::Rhs, OperandRole::Accumulator];

/// Selection operands ordered as input values and integer indices.
pub const INDEX_SELECT_OPERANDS: &[OperandRole] = &[OperandRole::Input, OperandRole::Indices];

/// Embedding operands ordered as weight table and lookup indices.
pub const EMBEDDING_OPERANDS: &[OperandRole] = &[OperandRole::Weight, OperandRole::Indices];

/// Rotary embedding operands ordered as input, cosine factors, and sine factors.
pub const ROTARY_EMBEDDING_OPERANDS: &[OperandRole] =
    &[OperandRole::Input, OperandRole::Cos, OperandRole::Sin];

/// Attention operands ordered as query, key, and value.
pub const SCALED_DOT_PRODUCT_ATTENTION_OPERANDS: &[OperandRole] =
    &[OperandRole::Input, OperandRole::Key, OperandRole::Value];

/// `scaled_mm`'s FP8 operands and their `F32` scales.
pub const SCALED_MATMUL_OPERANDS: &[OperandRole] = &[
    OperandRole::Lhs,
    OperandRole::Rhs,
    OperandRole::Scale,
    OperandRole::Scale,
];

/// Paged attention's queries, key and value caches, query offsets, page offsets, pages, and last page lengths.
pub const PAGED_ATTENTION_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Key,
    OperandRole::Value,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
];

// A causal convolution's inputs and per-channel taps, its state, and the step's query offsets, real token counts, and positions.
pub const CAUSAL_CONV1D_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Weight,
    OperandRole::Input,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
];

// A causal convolution's inputs and state, and the step's query offsets, real token counts, and positions.
pub const CAUSAL_CONV1D_STATE_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Input,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
];

// The gated delta rule's queries, keys, values, decays, write strengths, state, and the step's query offsets, real token counts, and positions.
pub const GATED_DELTA_RULE_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Key,
    OperandRole::Value,
    OperandRole::Input,
    OperandRole::Input,
    OperandRole::Input,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
];

// A grouped matmul's rows, its groups' weights, and the group offsets.
pub const GROUPED_MATMUL_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Weight,
    OperandRole::Indices,
];

// Latent attention's queries without and with positions, its latent and positional caches, and the query offsets, page offsets, pages, and last page lengths.
pub const PAGED_MLA_ATTENTION_OPERANDS: &[OperandRole] = &[
    OperandRole::Input,
    OperandRole::Input,
    OperandRole::Key,
    OperandRole::Key,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
    OperandRole::Indices,
];

/// The required batch-normalization input, before any optional affine operands.
pub const BATCH_NORM_OPERANDS: &[OperandRole] = &[OperandRole::Input];

/// The required layer-normalization input, before any optional affine operands.
pub const LAYER_NORM_OPERANDS: &[OperandRole] = &[OperandRole::Input];

/// Interpolation operands ordered as start and end values.
pub const LERP_OPERANDS: &[OperandRole] = &[
    OperandRole::InterpolationStart,
    OperandRole::InterpolationEnd,
];

/// Semantic role that determines an operation output's data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputRole {
    /// A value tensor with the declared output data type.
    Output,
    /// An index tensor with data type [`DataType::I64`].
    Indices,
    /// A mask tensor with data type [`DataType::Bool`].
    Mask,
    // The `F32` log-sum-exp of each row an attention normalizes.
    LogSumExp,
}

impl OutputRole {
    /// Returns the declared type for values, `I64` for indices, or `Bool` for masks.
    #[must_use]
    pub fn data_type(self, declared_data_type: DataType) -> DataType {
        match self {
            Self::Output => declared_data_type,
            Self::Indices => DataType::I64,
            Self::Mask => DataType::Bool,
            Self::LogSumExp => DataType::F32,
        }
    }
}

/// Optional output data type, device, and shape supplied explicitly by an operation.
///
/// Each getter returns `None` when that part of the output metadata has no explicit declaration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeclaredOutputSchema {
    data_type: Option<DataType>,
    device: Option<Device>,
    shape: Option<Shape>,
}

impl DeclaredOutputSchema {
    pub(crate) fn with_data_type(data_type: DataType) -> Self {
        Self {
            data_type: Some(data_type),
            ..Self::default()
        }
    }

    pub(crate) fn with_device(device: Device) -> Self {
        Self {
            device: Some(device),
            ..Self::default()
        }
    }

    pub(crate) fn with_shape(shape: Shape) -> Self {
        Self {
            shape: Some(shape),
            ..Self::default()
        }
    }

    /// Returns the explicitly declared output data type, if any.
    #[must_use]
    pub fn data_type(&self) -> Option<DataType> {
        self.data_type
    }

    /// Returns the explicitly declared output device, if any.
    #[must_use]
    pub fn device(&self) -> Option<Device> {
        self.device
    }

    /// Returns the explicitly declared output shape, if any.
    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }
}

/// No results, for operations that only write an input.
pub const NO_OUTPUTS: &[OutputRole] = &[];

/// One value output with the operation's inferred data type.
pub const VALUE_OUTPUT: &[OutputRole] = &[OutputRole::Output];

/// One signed 64-bit index output.
pub const INDEX_OUTPUT: &[OutputRole] = &[OutputRole::Indices];

/// Top-k outputs ordered as selected values and signed 64-bit indices.
pub const TOPK_OUTPUTS: &[OutputRole] = &[OutputRole::Output, OutputRole::Indices];

// The attention output and the base-two log-sum-exp of each row's scaled scores.
pub const ATTENTION_LSE_OUTPUTS: &[OutputRole] = &[OutputRole::Output, OutputRole::LogSumExp];

// A log-sum-exp alone.
pub const LSE_OUTPUT: &[OutputRole] = &[OutputRole::LogSumExp];

/// One Boolean mask output.
pub const MASK_OUTPUT: &[OutputRole] = &[OutputRole::Mask];
