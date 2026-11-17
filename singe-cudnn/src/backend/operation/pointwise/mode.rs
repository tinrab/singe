use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::impl_enum_display;

use singe_cudnn_sys as sys;

singe_core::native_enum! {
    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
    )]
    /// Pointwise math operation for backend pointwise operation descriptors.
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PointwiseMode: sys::cudnnPointwiseMode_t {
        /// Element-wise floored modulo: `y = x - floor(x / b) * b`.
        FloorMod = CUDNN_POINTWISE_FLOOR_MOD,
        /// Pointwise addition between two tensors.
        Add = CUDNN_POINTWISE_ADD,
        /// Pointwise addition between the first tensor and the square of the second tensor.
        AddSquare = CUDNN_POINTWISE_ADD_SQUARE,
        /// Pointwise true division of the first tensor by the second tensor.
        Div = CUDNN_POINTWISE_DIV,
        /// Pointwise maximum between two tensors.
        Max = CUDNN_POINTWISE_MAX,
        /// Pointwise minimum between two tensors.
        Min = CUDNN_POINTWISE_MIN,
        /// Pointwise floating-point remainder of the first tensor's division by the second tensor.
        Mod = CUDNN_POINTWISE_MOD,
        /// Pointwise multiplication between two tensors.
        Mul = CUDNN_POINTWISE_MUL,
        /// Pointwise value of the first tensor raised to the power of the second tensor.
        Pow = CUDNN_POINTWISE_POW,
        /// Pointwise subtraction between two tensors.
        Sub = CUDNN_POINTWISE_SUB,
        /// Pointwise absolute value of the input tensor.
        Abs = CUDNN_POINTWISE_ABS,
        /// Pointwise ceiling of the input tensor.
        Ceil = CUDNN_POINTWISE_CEIL,
        /// Pointwise cosine of the input tensor.
        Cos = CUDNN_POINTWISE_COS,
        /// Pointwise exponential of the input tensor.
        Exp = CUDNN_POINTWISE_EXP,
        /// Pointwise floor of the input tensor.
        Floor = CUDNN_POINTWISE_FLOOR,
        /// Pointwise natural logarithm of the input tensor.
        Log = CUDNN_POINTWISE_LOG,
        /// Pointwise negation of the input tensor.
        Neg = CUDNN_POINTWISE_NEG,
        /// Pointwise reciprocal square root of the input tensor.
        Rsqrt = CUDNN_POINTWISE_RSQRT,
        /// Pointwise sine of the input tensor.
        Sin = CUDNN_POINTWISE_SIN,
        /// Pointwise square root of the input tensor.
        Sqrt = CUDNN_POINTWISE_SQRT,
        /// Pointwise tangent of the input tensor.
        Tan = CUDNN_POINTWISE_TAN,
        /// Pointwise error function.
        Erf = CUDNN_POINTWISE_ERF,
        /// No computation.
        /// As with other pointwise modes, this mode provides implicit conversions by specifying the data type of the input tensor as one type, and the data type of the output tensor as another.
        Identity = CUDNN_POINTWISE_IDENTITY,
        /// Pointwise reciprocal of the input tensor.
        /// In other words, for every element `x` in the input tensor, computes `1 / x`.
        Reciprocal = CUDNN_POINTWISE_RECIPROCAL,
        /// Two-argument arctangent.
        Atan2 = CUDNN_POINTWISE_ATAN2,
        /// Pointwise rectified linear activation function of the input tensor.
        ReluFwd = CUDNN_POINTWISE_RELU_FWD,
        /// Pointwise tanh activation function of the input tensor.
        TanhFwd = CUDNN_POINTWISE_TANH_FWD,
        /// Pointwise sigmoid activation function of the input tensor.
        SigmoidFwd = CUDNN_POINTWISE_SIGMOID_FWD,
        /// Pointwise exponential linear unit activation function of the input tensor.
        EluFwd = CUDNN_POINTWISE_ELU_FWD,
        /// Pointwise Gaussian error linear unit activation function of the input tensor.
        GeluFwd = CUDNN_POINTWISE_GELU_FWD,
        /// Pointwise softplus activation function of the input tensor.
        SoftplusFwd = CUDNN_POINTWISE_SOFTPLUS_FWD,
        /// Pointwise swish activation function of the input tensor.
        SwishFwd = CUDNN_POINTWISE_SWISH_FWD,
        /// Pointwise tanh approximation of the Gaussian error linear unit activation function.
        /// Computes the tanh GELU approximation as $0.5x\left( 1+\tanh\left( \sqrt{2/\pi}\left( x+0.044715x^{3} \right) \right) \right)$.
        /// See the [Gaussian Error Linear Unit paper](https://arxiv.org/pdf/1606.08415.pdf).
        GeluApproxTanhFwd = CUDNN_POINTWISE_GELU_APPROX_TANH_FWD,
        /// Pointwise first derivative of rectified linear activation for the input tensor.
        ReluBwd = CUDNN_POINTWISE_RELU_BWD,
        /// Pointwise first derivative of tanh activation for the input tensor.
        TanhBwd = CUDNN_POINTWISE_TANH_BWD,
        /// Pointwise first derivative of sigmoid activation for the input tensor.
        SigmoidBwd = CUDNN_POINTWISE_SIGMOID_BWD,
        /// Pointwise first derivative of exponential linear unit activation for the input tensor.
        EluBwd = CUDNN_POINTWISE_ELU_BWD,
        /// Pointwise first derivative of Gaussian error linear unit activation for the input tensor.
        GeluBwd = CUDNN_POINTWISE_GELU_BWD,
        /// Pointwise first derivative of softplus activation for the input tensor.
        SoftplusBwd = CUDNN_POINTWISE_SOFTPLUS_BWD,
        /// Pointwise first derivative of swish activation for the input tensor.
        SwishBwd = CUDNN_POINTWISE_SWISH_BWD,
        /// Pointwise first derivative of the tanh approximation of Gaussian error linear unit activation.
        /// Computes $0.5\left( 1+\tanh\left( b\left( x+cx^{3} \right) \right)+bxsech^{2}\left( b\left( cx^{3}+x \right) \right)\left( 3cx^{2}+1 \right)dy \right)$ where $b$ is $\sqrt{2/\pi}$ and $c$ is $0.044715$.
        GeluApproxTanhBwd = CUDNN_POINTWISE_GELU_APPROX_TANH_BWD,
        /// Pointwise equality comparison between the first and second tensors.
        CmpEq = CUDNN_POINTWISE_CMP_EQ,
        /// Pointwise inequality comparison between the first and second tensors.
        CmpNeq = CUDNN_POINTWISE_CMP_NEQ,
        /// Pointwise greater-than comparison between the first and second tensors.
        CmpGt = CUDNN_POINTWISE_CMP_GT,
        /// Pointwise greater-than-or-equal comparison between the first and second tensors.
        CmpGe = CUDNN_POINTWISE_CMP_GE,
        /// Pointwise less-than comparison between the first and second tensors.
        CmpLt = CUDNN_POINTWISE_CMP_LT,
        /// Pointwise less-than-or-equal comparison between the first and second tensors.
        CmpLe = CUDNN_POINTWISE_CMP_LE,
        /// Pointwise logical `AND` between the first and second tensors.
        LogicalAnd = CUDNN_POINTWISE_LOGICAL_AND,
        /// Pointwise logical `OR` between the first and second tensors.
        LogicalOr = CUDNN_POINTWISE_LOGICAL_OR,
        /// Pointwise logical `NOT` of the input tensor.
        LogicalNot = CUDNN_POINTWISE_LOGICAL_NOT,
        /// Pointwise index value generated along a given axis.
        GenIndex = CUDNN_POINTWISE_GEN_INDEX,
        /// Pointwise selection between two input tensors using a predicate tensor.
        BinarySelect = CUDNN_POINTWISE_BINARY_SELECT,
    }
}

/// Native tensor ports used by a pointwise operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointwiseOperandKind {
    /// One input `X` and one output `Y`.
    Unary,
    /// Inputs `X` and `B`, and output `Y`.
    Binary,
    /// Inputs `X`, `B` and predicate `T`, and output `Y`.
    Ternary,
    /// Inputs `DY` and `X`, and output `DX`.
    ActivationBackward,
}

impl PointwiseMode {
    /// Returns the native input/output role pattern required by this pointwise mode.
    #[must_use]
    pub const fn operand_kind(self) -> PointwiseOperandKind {
        match self {
            Self::BinarySelect => PointwiseOperandKind::Ternary,
            Self::ReluBwd
            | Self::TanhBwd
            | Self::SigmoidBwd
            | Self::EluBwd
            | Self::GeluBwd
            | Self::SoftplusBwd
            | Self::SwishBwd
            | Self::GeluApproxTanhBwd => PointwiseOperandKind::ActivationBackward,
            Self::FloorMod
            | Self::Add
            | Self::AddSquare
            | Self::Div
            | Self::Max
            | Self::Min
            | Self::Mod
            | Self::Mul
            | Self::Pow
            | Self::Sub
            | Self::Atan2
            | Self::CmpEq
            | Self::CmpNeq
            | Self::CmpGt
            | Self::CmpGe
            | Self::CmpLt
            | Self::CmpLe
            | Self::LogicalAnd
            | Self::LogicalOr => PointwiseOperandKind::Binary,
            Self::Abs
            | Self::Ceil
            | Self::Cos
            | Self::Exp
            | Self::Floor
            | Self::Log
            | Self::Neg
            | Self::Rsqrt
            | Self::Sin
            | Self::Sqrt
            | Self::Tan
            | Self::Erf
            | Self::Identity
            | Self::Reciprocal
            | Self::ReluFwd
            | Self::TanhFwd
            | Self::SigmoidFwd
            | Self::EluFwd
            | Self::GeluFwd
            | Self::SoftplusFwd
            | Self::SwishFwd
            | Self::GeluApproxTanhFwd
            | Self::LogicalNot
            | Self::GenIndex => PointwiseOperandKind::Unary,
        }
    }

    /// Returns whether this mode consumes upstream gradient `DY` and input `X` to produce `DX`.
    #[must_use]
    pub const fn is_activation_backward(self) -> bool {
        matches!(
            self.operand_kind(),
            PointwiseOperandKind::ActivationBackward
        )
    }

    /// Number of input tensors required by this native mode.
    #[must_use]
    pub const fn operand_count(self) -> usize {
        match self.operand_kind() {
            PointwiseOperandKind::Unary => 1,
            PointwiseOperandKind::Binary | PointwiseOperandKind::ActivationBackward => 2,
            PointwiseOperandKind::Ternary => 3,
        }
    }
}

impl_enum_display!(PointwiseMode, {
    Self::FloorMod => "CUDNN_POINTWISE_FLOOR_MOD",
    Self::Add => "CUDNN_POINTWISE_ADD",
    Self::AddSquare => "CUDNN_POINTWISE_ADD_SQUARE",
    Self::Div => "CUDNN_POINTWISE_DIV",
    Self::Max => "CUDNN_POINTWISE_MAX",
    Self::Min => "CUDNN_POINTWISE_MIN",
    Self::Mod => "CUDNN_POINTWISE_MOD",
    Self::Mul => "CUDNN_POINTWISE_MUL",
    Self::Pow => "CUDNN_POINTWISE_POW",
    Self::Sub => "CUDNN_POINTWISE_SUB",
    Self::Abs => "CUDNN_POINTWISE_ABS",
    Self::Ceil => "CUDNN_POINTWISE_CEIL",
    Self::Cos => "CUDNN_POINTWISE_COS",
    Self::Exp => "CUDNN_POINTWISE_EXP",
    Self::Floor => "CUDNN_POINTWISE_FLOOR",
    Self::Log => "CUDNN_POINTWISE_LOG",
    Self::Neg => "CUDNN_POINTWISE_NEG",
    Self::Rsqrt => "CUDNN_POINTWISE_RSQRT",
    Self::Sin => "CUDNN_POINTWISE_SIN",
    Self::Sqrt => "CUDNN_POINTWISE_SQRT",
    Self::Tan => "CUDNN_POINTWISE_TAN",
    Self::Erf => "CUDNN_POINTWISE_ERF",
    Self::Identity => "CUDNN_POINTWISE_IDENTITY",
    Self::Reciprocal => "CUDNN_POINTWISE_RECIPROCAL",
    Self::Atan2 => "CUDNN_POINTWISE_ATAN2",
    Self::ReluFwd => "CUDNN_POINTWISE_RELU_FWD",
    Self::TanhFwd => "CUDNN_POINTWISE_TANH_FWD",
    Self::SigmoidFwd => "CUDNN_POINTWISE_SIGMOID_FWD",
    Self::EluFwd => "CUDNN_POINTWISE_ELU_FWD",
    Self::GeluFwd => "CUDNN_POINTWISE_GELU_FWD",
    Self::SoftplusFwd => "CUDNN_POINTWISE_SOFTPLUS_FWD",
    Self::SwishFwd => "CUDNN_POINTWISE_SWISH_FWD",
    Self::GeluApproxTanhFwd => "CUDNN_POINTWISE_GELU_APPROX_TANH_FWD",
    Self::ReluBwd => "CUDNN_POINTWISE_RELU_BWD",
    Self::TanhBwd => "CUDNN_POINTWISE_TANH_BWD",
    Self::SigmoidBwd => "CUDNN_POINTWISE_SIGMOID_BWD",
    Self::EluBwd => "CUDNN_POINTWISE_ELU_BWD",
    Self::GeluBwd => "CUDNN_POINTWISE_GELU_BWD",
    Self::SoftplusBwd => "CUDNN_POINTWISE_SOFTPLUS_BWD",
    Self::SwishBwd => "CUDNN_POINTWISE_SWISH_BWD",
    Self::GeluApproxTanhBwd => "CUDNN_POINTWISE_GELU_APPROX_TANH_BWD",
    Self::CmpEq => "CUDNN_POINTWISE_CMP_EQ",
    Self::CmpNeq => "CUDNN_POINTWISE_CMP_NEQ",
    Self::CmpGt => "CUDNN_POINTWISE_CMP_GT",
    Self::CmpGe => "CUDNN_POINTWISE_CMP_GE",
    Self::CmpLt => "CUDNN_POINTWISE_CMP_LT",
    Self::CmpLe => "CUDNN_POINTWISE_CMP_LE",
    Self::LogicalAnd => "CUDNN_POINTWISE_LOGICAL_AND",
    Self::LogicalOr => "CUDNN_POINTWISE_LOGICAL_OR",
    Self::LogicalNot => "CUDNN_POINTWISE_LOGICAL_NOT",
    Self::GenIndex => "CUDNN_POINTWISE_GEN_INDEX",
    Self::BinarySelect => "CUDNN_POINTWISE_BINARY_SELECT",
});
