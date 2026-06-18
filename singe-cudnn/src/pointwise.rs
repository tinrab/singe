use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Pointwise math operation for backend pointwise operation descriptors.
#[repr(u32)]
#[non_exhaustive]
pub enum PointwiseMode {
    /// Pointwise addition between two tensors.
    Add = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD as _,
    /// Pointwise addition between the first tensor and the square of the second tensor.
    AddSquare = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD_SQUARE as _,
    /// Pointwise true division of the first tensor by the second tensor.
    Div = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_DIV as _,
    /// Pointwise maximum between two tensors.
    Max = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_MAX as _,
    /// Pointwise minimum between two tensors.
    Min = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_MIN as _,
    /// Pointwise floating-point remainder of the first tensor's division by the second tensor.
    Mod = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_MOD as _,
    /// Pointwise multiplication between two tensors.
    Mul = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL as _,
    /// Pointwise value of the first tensor raised to the power of the second tensor.
    Pow = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_POW as _,
    /// Pointwise subtraction between two tensors.
    Sub = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SUB as _,
    /// Pointwise absolute value of the input tensor.
    Abs = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ABS as _,
    /// Pointwise ceiling of the input tensor.
    Ceil = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CEIL as _,
    /// Pointwise cosine of the input tensor.
    Cos = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_COS as _,
    /// Pointwise exponential of the input tensor.
    Exp = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_EXP as _,
    /// Pointwise floor of the input tensor.
    Floor = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_FLOOR as _,
    /// Pointwise natural logarithm of the input tensor.
    Log = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_LOG as _,
    /// Pointwise negation of the input tensor.
    Neg = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_NEG as _,
    /// Pointwise reciprocal square root of the input tensor.
    Rsqrt = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_RSQRT as _,
    /// Pointwise sine of the input tensor.
    Sin = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SIN as _,
    /// Pointwise square root of the input tensor.
    Sqrt = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SQRT as _,
    /// Pointwise tangent of the input tensor.
    Tan = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_TAN as _,
    /// Pointwise error function.
    Erf = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ERF as _,
    /// No computation.
    /// As with other pointwise modes, this mode provides implicit conversions by specifying the data type of the input tensor as one type, and the data type of the output tensor as another.
    Identity = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_IDENTITY as _,
    /// Pointwise reciprocal of the input tensor.
    /// In other words, for every element `x` in the input tensor, computes `1 / x`.
    Reciprocal = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_RECIPROCAL as _,
    Atan2 = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ATAN2 as _,
    /// Pointwise rectified linear activation function of the input tensor.
    ReluFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD as _,
    /// Pointwise tanh activation function of the input tensor.
    TanhFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_TANH_FWD as _,
    /// Pointwise sigmoid activation function of the input tensor.
    SigmoidFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SIGMOID_FWD as _,
    /// Pointwise exponential linear unit activation function of the input tensor.
    EluFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ELU_FWD as _,
    /// Pointwise Gaussian error linear unit activation function of the input tensor.
    GeluFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_GELU_FWD as _,
    /// Pointwise softplus activation function of the input tensor.
    SoftplusFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SOFTPLUS_FWD as _,
    /// Pointwise swish activation function of the input tensor.
    SwishFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SWISH_FWD as _,
    /// Pointwise tanh approximation of the Gaussian error linear unit activation function.
    /// Computes the tanh GELU approximation as $0.5x\left( 1+\tanh\left( \sqrt{2/\pi}\left( x+0.044715x^{3} \right) \right) \right)$.
    /// See the [Gaussian Error Linear Unit paper](https://arxiv.org/pdf/1606.08415.pdf).
    GeluApproxTanhFwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_GELU_APPROX_TANH_FWD as _,
    /// Pointwise first derivative of rectified linear activation for the input tensor.
    ReluBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_BWD as _,
    /// Pointwise first derivative of tanh activation for the input tensor.
    TanhBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_TANH_BWD as _,
    /// Pointwise first derivative of sigmoid activation for the input tensor.
    SigmoidBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SIGMOID_BWD as _,
    /// Pointwise first derivative of exponential linear unit activation for the input tensor.
    EluBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_ELU_BWD as _,
    /// Pointwise first derivative of Gaussian error linear unit activation for the input tensor.
    GeluBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_GELU_BWD as _,
    /// Pointwise first derivative of softplus activation for the input tensor.
    SoftplusBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SOFTPLUS_BWD as _,
    /// Pointwise first derivative of swish activation for the input tensor.
    SwishBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_SWISH_BWD as _,
    /// Pointwise first derivative of the tanh approximation of Gaussian error linear unit activation.
    /// Computes $0.5\left( 1+\tanh\left( b\left( x+cx^{3} \right) \right)+bxsech^{2}\left( b\left( cx^{3}+x \right) \right)\left( 3cx^{2}+1 \right)dy \right)$ where $b$ is $\sqrt{2/\pi}$ and $c$ is $0.044715$.
    GeluApproxTanhBwd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_GELU_APPROX_TANH_BWD as _,
    /// Pointwise equality comparison between the first and second tensors.
    CmpEq = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_EQ as _,
    /// Pointwise inequality comparison between the first and second tensors.
    CmpNeq = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_NEQ as _,
    /// Pointwise greater-than comparison between the first and second tensors.
    CmpGt = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_GT as _,
    /// Pointwise greater-than-or-equal comparison between the first and second tensors.
    CmpGe = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_GE as _,
    /// Pointwise less-than comparison between the first and second tensors.
    CmpLt = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_LT as _,
    /// Pointwise less-than-or-equal comparison between the first and second tensors.
    CmpLe = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_CMP_LE as _,
    /// Pointwise logical `AND` between the first and second tensors.
    LogicalAnd = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_LOGICAL_AND as _,
    /// Pointwise logical `OR` between the first and second tensors.
    LogicalOr = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_LOGICAL_OR as _,
    /// Pointwise logical `NOT` of the input tensor.
    LogicalNot = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_LOGICAL_NOT as _,
    /// Pointwise index value generated along a given axis.
    GenIndex = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_GEN_INDEX as _,
    /// Pointwise selection between two input tensors using a predicate tensor.
    BinarySelect = sys::cudnnPointwiseMode_t::CUDNN_POINTWISE_BINARY_SELECT as _,
}

impl_enum_conversion!(sys::cudnnPointwiseMode_t, PointwiseMode);

impl_enum_display!(PointwiseMode, {
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
