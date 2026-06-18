use std::fmt::{self, Display, Formatter};

use singe_core::{impl_enum_conversion, impl_enum_display};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cutensor_sys as sys;

/// Algorithm used to perform a tensor operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum Algorithm {
    /// More time-consuming than CUTENSOR_DEFAULT, but typically provides a more accurate kernel selection.
    DefaultPatient = sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT_PATIENT as _,
    /// Choose the GETT algorithm (only applicable to contractions).
    Gett = sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT as _,
    /// Transpose (A or B) + GETT (only applicable to contractions).
    Tgett = sys::cutensorAlgo_t::CUTENSOR_ALGO_TGETT as _,
    /// Transpose-Transpose-GEMM-Transpose (requires additional memory) (only applicable to contractions).
    Ttgt = sys::cutensorAlgo_t::CUTENSOR_ALGO_TTGT as _,
    /// A performance model chooses the appropriate algorithm and kernel.
    Default = sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT as _,
}

impl_enum_conversion!(i32, sys::cutensorAlgo_t, Algorithm);

impl_enum_display!(Algorithm, {
    Algorithm::DefaultPatient => "CUTENSOR_ALGO_DEFAULT_PATIENT",
    Algorithm::Gett => "CUTENSOR_ALGO_GETT",
    Algorithm::Tgett => "CUTENSOR_ALGO_TGETT",
    Algorithm::Ttgt => "CUTENSOR_ALGO_TTGT",
    Algorithm::Default => "CUTENSOR_ALGO_DEFAULT",
});

/// Encodes cuTENSOR's legacy compute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeType {
    F16 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_16F as _,
    Bf16 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_16BF as _,
    Tf32 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_TF32 as _,
    Tf32x3 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_3XTF32 as _,
    F32 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_32F as _,
    F64 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_64F as _,
    U8 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_8U as _,
    I8 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_8I as _,
    U32 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_32U as _,
    I32 = sys::cutensorComputeType_t::CUTENSOR_COMPUTE_32I as _,
}

impl_enum_conversion!(sys::cutensorComputeType_t, ComputeType);

impl_enum_display!(ComputeType, {
    ComputeType::F16 => "CUTENSOR_COMPUTE_16F",
    ComputeType::Bf16 => "CUTENSOR_COMPUTE_16BF",
    ComputeType::Tf32 => "CUTENSOR_COMPUTE_TF32",
    ComputeType::Tf32x3 => "CUTENSOR_COMPUTE_3XTF32",
    ComputeType::F32 => "CUTENSOR_COMPUTE_32F",
    ComputeType::F64 => "CUTENSOR_COMPUTE_64F",
    ComputeType::U8 => "CUTENSOR_COMPUTE_8U",
    ComputeType::I8 => "CUTENSOR_COMPUTE_8I",
    ComputeType::U32 => "CUTENSOR_COMPUTE_32U",
    ComputeType::I32 => "CUTENSOR_COMPUTE_32I",
});

/// Unary and binary element-wise operations supported by cuTENSOR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Operator {
    /// Identity operator; elements are not changed.
    Identity = sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY as _,
    /// Square root.
    Sqrt = sys::cutensorOperator_t::CUTENSOR_OP_SQRT as _,
    /// Rectified linear unit.
    Relu = sys::cutensorOperator_t::CUTENSOR_OP_RELU as _,
    /// Complex conjugate.
    Conj = sys::cutensorOperator_t::CUTENSOR_OP_CONJ as _,
    /// Reciprocal.
    Rcp = sys::cutensorOperator_t::CUTENSOR_OP_RCP as _,
    /// y=1/(1+exp(-x)).
    Sigmoid = sys::cutensorOperator_t::CUTENSOR_OP_SIGMOID as _,
    /// y=tanh(x).
    Tanh = sys::cutensorOperator_t::CUTENSOR_OP_TANH as _,
    /// Exponentiation.
    Exp = sys::cutensorOperator_t::CUTENSOR_OP_EXP as _,
    /// Log (base e).
    Log = sys::cutensorOperator_t::CUTENSOR_OP_LOG as _,
    /// Absolute value.
    Abs = sys::cutensorOperator_t::CUTENSOR_OP_ABS as _,
    /// Negation.
    Neg = sys::cutensorOperator_t::CUTENSOR_OP_NEG as _,
    /// Sine.
    Sin = sys::cutensorOperator_t::CUTENSOR_OP_SIN as _,
    /// Cosine.
    Cos = sys::cutensorOperator_t::CUTENSOR_OP_COS as _,
    /// Tangent.
    Tan = sys::cutensorOperator_t::CUTENSOR_OP_TAN as _,
    /// Hyperbolic sine.
    Sinh = sys::cutensorOperator_t::CUTENSOR_OP_SINH as _,
    /// Hyperbolic cosine.
    Cosh = sys::cutensorOperator_t::CUTENSOR_OP_COSH as _,
    /// Inverse sine.
    Asin = sys::cutensorOperator_t::CUTENSOR_OP_ASIN as _,
    /// Inverse cosine.
    Acos = sys::cutensorOperator_t::CUTENSOR_OP_ACOS as _,
    /// Inverse tangent.
    Atan = sys::cutensorOperator_t::CUTENSOR_OP_ATAN as _,
    /// Inverse hyperbolic sine.
    Asinh = sys::cutensorOperator_t::CUTENSOR_OP_ASINH as _,
    /// Inverse hyperbolic cosine.
    Acosh = sys::cutensorOperator_t::CUTENSOR_OP_ACOSH as _,
    /// Inverse hyperbolic tangent.
    Atanh = sys::cutensorOperator_t::CUTENSOR_OP_ATANH as _,
    /// Ceiling.
    Ceil = sys::cutensorOperator_t::CUTENSOR_OP_CEIL as _,
    /// Floor.
    Floor = sys::cutensorOperator_t::CUTENSOR_OP_FLOOR as _,
    /// Mish y=x\*tanh(softplus(x)).
    Mish = sys::cutensorOperator_t::CUTENSOR_OP_MISH as _,
    /// Swish y=x\*sigmoid(x).
    Swish = sys::cutensorOperator_t::CUTENSOR_OP_SWISH as _,
    /// Softplus y=log(exp(x)+1).
    SoftPlus = sys::cutensorOperator_t::CUTENSOR_OP_SOFT_PLUS as _,
    /// Softsign y=x/(abs(x)+1).
    SoftSign = sys::cutensorOperator_t::CUTENSOR_OP_SOFT_SIGN as _,
    /// Addition of two elements.
    Add = sys::cutensorOperator_t::CUTENSOR_OP_ADD as _,
    /// Multiplication of two elements.
    Mul = sys::cutensorOperator_t::CUTENSOR_OP_MUL as _,
    /// Maximum of two elements.
    Max = sys::cutensorOperator_t::CUTENSOR_OP_MAX as _,
    /// Minimum of two elements.
    Min = sys::cutensorOperator_t::CUTENSOR_OP_MIN as _,
    /// Reserved for internal use only.
    Unknown = sys::cutensorOperator_t::CUTENSOR_OP_UNKNOWN as _,
}

impl_enum_conversion!(sys::cutensorOperator_t, Operator);

impl_enum_display!(Operator, {
    Operator::Identity => "CUTENSOR_OP_IDENTITY",
    Operator::Sqrt => "CUTENSOR_OP_SQRT",
    Operator::Relu => "CUTENSOR_OP_RELU",
    Operator::Conj => "CUTENSOR_OP_CONJ",
    Operator::Rcp => "CUTENSOR_OP_RCP",
    Operator::Sigmoid => "CUTENSOR_OP_SIGMOID",
    Operator::Tanh => "CUTENSOR_OP_TANH",
    Operator::Exp => "CUTENSOR_OP_EXP",
    Operator::Log => "CUTENSOR_OP_LOG",
    Operator::Abs => "CUTENSOR_OP_ABS",
    Operator::Neg => "CUTENSOR_OP_NEG",
    Operator::Sin => "CUTENSOR_OP_SIN",
    Operator::Cos => "CUTENSOR_OP_COS",
    Operator::Tan => "CUTENSOR_OP_TAN",
    Operator::Sinh => "CUTENSOR_OP_SINH",
    Operator::Cosh => "CUTENSOR_OP_COSH",
    Operator::Asin => "CUTENSOR_OP_ASIN",
    Operator::Acos => "CUTENSOR_OP_ACOS",
    Operator::Atan => "CUTENSOR_OP_ATAN",
    Operator::Asinh => "CUTENSOR_OP_ASINH",
    Operator::Acosh => "CUTENSOR_OP_ACOSH",
    Operator::Atanh => "CUTENSOR_OP_ATANH",
    Operator::Ceil => "CUTENSOR_OP_CEIL",
    Operator::Floor => "CUTENSOR_OP_FLOOR",
    Operator::Mish => "CUTENSOR_OP_MISH",
    Operator::Swish => "CUTENSOR_OP_SWISH",
    Operator::SoftPlus => "CUTENSOR_OP_SOFT_PLUS",
    Operator::SoftSign => "CUTENSOR_OP_SOFT_SIGN",
    Operator::Add => "CUTENSOR_OP_ADD",
    Operator::Mul => "CUTENSOR_OP_MUL",
    Operator::Max => "CUTENSOR_OP_MAX",
    Operator::Min => "CUTENSOR_OP_MIN",
    Operator::Unknown => "CUTENSOR_OP_UNKNOWN",
});

/// Controls the workspace amount suggested by [`Plan::estimate_workspace_size`](crate::plan::Plan::estimate_workspace_size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum WorkspacePreference {
    /// Least memory requirement; at least one algorithm is available.
    Min = sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_MIN as _,
    /// Aims to attain high performance while also reducing the workspace requirement.
    Default = sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT as _,
    /// Highest memory requirement; all algorithms are available.
    Max = sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_MAX as _,
}

impl_enum_conversion!(sys::cutensorWorksizePreference_t, WorkspacePreference);

impl_enum_display!(WorkspacePreference, {
    WorkspacePreference::Min => "CUTENSOR_WORKSPACE_MIN",
    WorkspacePreference::Default => "CUTENSOR_WORKSPACE_DEFAULT",
    WorkspacePreference::Max => "CUTENSOR_WORKSPACE_MAX",
});

/// Controls what is considered a cache hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CacheMode {
    /// Plans are not cached.
    None = sys::cutensorCacheMode_t::CUTENSOR_CACHE_MODE_NONE as _,
    /// All descriptor parameters must match the cached plan.
    Pedantic = sys::cutensorCacheMode_t::CUTENSOR_CACHE_MODE_PEDANTIC as _,
}

impl_enum_conversion!(sys::cutensorCacheMode_t, CacheMode);

impl_enum_display!(CacheMode, {
    CacheMode::None => "CUTENSOR_CACHE_MODE_NONE",
    CacheMode::Pedantic => "CUTENSOR_CACHE_MODE_PEDANTIC",
});

/// Attributes of a cuTENSOR operation descriptor that can be read or modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum OperationDescriptorAttribute {
    /// `i32` tag that distinguishes otherwise identical problems in the software-managed plan cache.
    /// (default value: 0).
    Tag = sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_TAG as _,
    /// [`DataType`](singe_cuda::data_type::DataType): data type of the scaling factors.
    ScalarType =
        sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE as _,
    /// `f32`: number of floating-point operations necessary to perform this operation, assuming all scalars are nonzero unless otherwise specified.
    Flops = sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_FLOPS as _,
    /// `f32`: minimal number of bytes transferred from or to global memory, assuming all scalars are nonzero unless otherwise specified.
    MovedBytes =
        sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES as _,
    /// `u32` array of length `desc_out.num_modes`; entry `i` is the number of padded values to the left of dimension `i`.
    PaddingLeft =
        sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT
            as _,
    /// `u32` array of length `desc_out.num_modes`; entry `i` is the number of padded values to the right of dimension `i`.
    PaddingRight =
        sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT
            as _,
    /// Host-side pointer to an element of the same type as the output tensor: constant padding value.
    PaddingValue =
        sys::cutensorOperationDescriptorAttribute_t::CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE
            as _,
}

impl_enum_conversion!(
    sys::cutensorOperationDescriptorAttribute_t,
    OperationDescriptorAttribute
);

impl_enum_display!(OperationDescriptorAttribute, {
    OperationDescriptorAttribute::Tag => "CUTENSOR_OPERATION_DESCRIPTOR_TAG",
    OperationDescriptorAttribute::ScalarType => "CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE",
    OperationDescriptorAttribute::Flops => "CUTENSOR_OPERATION_DESCRIPTOR_FLOPS",
    OperationDescriptorAttribute::MovedBytes => "CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES",
    OperationDescriptorAttribute::PaddingLeft => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT",
    OperationDescriptorAttribute::PaddingRight => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT",
    OperationDescriptorAttribute::PaddingValue => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE",
});

/// Attributes of a [`PlanPreference`](crate::plan::PlanPreference) that can be read or modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanPreferenceAttribute {
    /// [`AutotuneMode`]: determines whether recurrent executions of the plan autotune.
    AutotuneMode =
        sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE as _,
    /// [`CacheMode`]: determines whether the algorithm or kernel for this plan is cached and controls what is considered a cache hit.
    CacheMode = sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_CACHE_MODE as _,
    /// `i32`: only applicable when [`PlanPreferenceAttribute::CacheMode`] is set to [`AutotuneMode::Incremental`].
    IncrementalCount =
        sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT as _,
    /// [`Algorithm`]: fixes the algorithm.
    Algorithm = sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_ALGO as _,
    /// `i32`: fixes a kernel subvariant of an algorithm; for example, `kernel_rank == 1` with [`Algorithm::Tgett`] selects the second-best GETT kernel variant according to cuTENSOR's performance model, while `kernel_rank == 2` selects the third-best.
    KernelRank = sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK as _,
    /// [`JitMode`]: determines whether just-in-time compilation is enabled (default: [`JitMode::None`]).
    JitMode = sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_JIT as _,
    /// `i32`: plan for a specific GPU architecture instead of the architecture associated with the context.
    /// The value encodes the SM version as `10 * SM.major + SM.minor`.
    /// Currently only SM versions 80, 90 and 100 are supported.
    GpuArch = sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_GPU_ARCH as _,
}

impl_enum_conversion!(
    sys::cutensorPlanPreferenceAttribute_t,
    PlanPreferenceAttribute
);

impl_enum_display!(PlanPreferenceAttribute, {
    PlanPreferenceAttribute::AutotuneMode => "CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE",
    PlanPreferenceAttribute::CacheMode => "CUTENSOR_PLAN_PREFERENCE_CACHE_MODE",
    PlanPreferenceAttribute::IncrementalCount => "CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT",
    PlanPreferenceAttribute::Algorithm => "CUTENSOR_PLAN_PREFERENCE_ALGO",
    PlanPreferenceAttribute::KernelRank => "CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK",
    PlanPreferenceAttribute::JitMode => "CUTENSOR_PLAN_PREFERENCE_JIT",
    PlanPreferenceAttribute::GpuArch => "CUTENSOR_PLAN_PREFERENCE_GPU_ARCH",
});

/// Controls cuTENSOR autotuning behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AutotuneMode {
    /// Indicates no autotuning (default); the cache reduces plan-creation overhead.
    /// On a cache hit, the cached plan is reused; otherwise the plan cache is ignored.
    None = sys::cutensorAutotuneMode_t::CUTENSOR_AUTOTUNE_MODE_NONE as _,
    /// Indicates incremental autotuning.
    /// Each corresponding [`sys::cutensorCreatePlan`] invocation creates a plan based on a different algorithm or kernel.
    /// [`PlanPreferenceAttribute::IncrementalCount`] defines the maximum number of kernels tested.
    /// May choose different algorithms across plan creations, so it does not
    /// guarantee bitwise-identical results.
    Incremental = sys::cutensorAutotuneMode_t::CUTENSOR_AUTOTUNE_MODE_INCREMENTAL as _,
}

impl_enum_conversion!(sys::cutensorAutotuneMode_t, AutotuneMode);

impl_enum_display!(AutotuneMode, {
    AutotuneMode::None => "CUTENSOR_AUTOTUNE_MODE_NONE",
    AutotuneMode::Incremental => "CUTENSOR_AUTOTUNE_MODE_INCREMENTAL",
});

/// Controls cuTENSOR just-in-time compilation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum JitMode {
    /// Indicates that no kernel is just-in-time compiled.
    None = sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE as _,
    /// Indicates that the corresponding plan tries to compile a dedicated kernel for the given operation.
    /// Only supported for GPUs with compute capability &gt;= 8.0 (Ampere or newer).
    Default = sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_DEFAULT as _,
}

impl_enum_conversion!(sys::cutensorJitMode_t, JitMode);

impl_enum_display!(JitMode, {
    JitMode::None => "CUTENSOR_JIT_MODE_NONE",
    JitMode::Default => "CUTENSOR_JIT_MODE_DEFAULT",
});

/// Attributes of a cuTENSOR plan that can be retrieved via [`sys::cutensorPlanGetAttribute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanAttribute {
    /// Exact required workspace in bytes needed to execute the plan.
    RequiredWorkspace = sys::cutensorPlanAttribute_t::CUTENSOR_PLAN_REQUIRED_WORKSPACE as _,
}

impl_enum_conversion!(sys::cutensorPlanAttribute_t, PlanAttribute);

impl_enum_display!(PlanAttribute, {
    PlanAttribute::RequiredWorkspace => "CUTENSOR_PLAN_REQUIRED_WORKSPACE",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum LoggerLevel {
    Off = 0,
    Error = 1,
    PerformanceTrace = 2,
    PerformanceHints = 3,
    HeuristicsTrace = 4,
    ApiTrace = 5,
}

impl LoggerLevel {
    pub const fn as_raw(self) -> i32 {
        self as i32
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct LoggerMask: i32 {
        const OFF = 0;
        const ERROR = 1;
        const PERFORMANCE_TRACE = 2;
        const PERFORMANCE_HINTS = 4;
        const HEURISTICS_TRACE = 8;
        const API_TRACE = 16;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Mode(i32);

impl Mode {
    pub const fn new(raw: i32) -> Self {
        Self(raw)
    }

    pub const fn from_char(mode: char) -> Self {
        Self(mode as i32)
    }

    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl From<char> for Mode {
    fn from(value: char) -> Self {
        Self::from_char(value)
    }
}

impl From<i32> for Mode {
    fn from(value: i32) -> Self {
        Self::new(value)
    }
}

impl From<Mode> for i32 {
    fn from(value: Mode) -> Self {
        value.as_raw()
    }
}

impl Display for Mode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(ch) = char::from_u32(self.0 as u32)
            && !ch.is_control()
        {
            return write!(f, "{ch}");
        }

        write!(f, "{}", self.0)
    }
}
