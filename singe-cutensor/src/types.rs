use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cutensor_sys as sys;

singe_core::native_enum! {
    /// Algorithm used to perform a tensor operation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum Algorithm: sys::cutensorAlgo_t {
        /// More time-consuming than [`Algorithm::Default`], but typically provides a more accurate kernel selection.
        DefaultPatient = CUTENSOR_ALGO_DEFAULT_PATIENT,
        /// Choose the GETT algorithm (only applicable to contractions).
        Gett = CUTENSOR_ALGO_GETT,
        /// Transpose (A or B) + GETT (only applicable to contractions).
        Tgett = CUTENSOR_ALGO_TGETT,
        /// Transpose-Transpose-GEMM-Transpose (requires additional memory) (only applicable to contractions).
        Ttgt = CUTENSOR_ALGO_TTGT,
        /// A performance model chooses the appropriate algorithm and kernel.
        Default = CUTENSOR_ALGO_DEFAULT,
    }
}

impl_enum_display!(Algorithm, {
    Algorithm::DefaultPatient => "CUTENSOR_ALGO_DEFAULT_PATIENT",
    Algorithm::Gett => "CUTENSOR_ALGO_GETT",
    Algorithm::Tgett => "CUTENSOR_ALGO_TGETT",
    Algorithm::Ttgt => "CUTENSOR_ALGO_TTGT",
    Algorithm::Default => "CUTENSOR_ALGO_DEFAULT",
});

// Only cuTENSORMg takes the legacy compute type, and only its bindings declare it.
#[cfg(feature = "mg")]
singe_core::native_enum! {
    /// Encodes cuTENSOR's legacy compute type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ComputeType: sys::cutensorComputeType_t {
        ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
        F16 = CUTENSOR_COMPUTE_16F,
        ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
        Bf16 = CUTENSOR_COMPUTE_16BF,
        ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
        Tf32 = CUTENSOR_COMPUTE_TF32,
        ///< floating-point: More precise than TF32, but less precise than float
        Tf32x3 = CUTENSOR_COMPUTE_3XTF32,
        ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
        F32 = CUTENSOR_COMPUTE_32F,
        ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
        F64 = CUTENSOR_COMPUTE_64F,
        ///< 8-bit unsigned integer
        U8 = CUTENSOR_COMPUTE_8U,
        ///< 8-bit signed integer
        I8 = CUTENSOR_COMPUTE_8I,
        ///< 32-bit unsigned integer
        U32 = CUTENSOR_COMPUTE_32U,
        ///< 32-bit signed integer
        I32 = CUTENSOR_COMPUTE_32I,
    }
}

#[cfg(feature = "mg")]
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

singe_core::native_enum! {
    /// Unary and binary element-wise operations supported by cuTENSOR.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Operator: sys::cutensorOperator_t {
        /// Identity operator; elements are not changed.
        Identity = CUTENSOR_OP_IDENTITY,
        /// Square root.
        Sqrt = CUTENSOR_OP_SQRT,
        /// Rectified linear unit.
        Relu = CUTENSOR_OP_RELU,
        /// Complex conjugate.
        Conj = CUTENSOR_OP_CONJ,
        /// Reciprocal.
        Rcp = CUTENSOR_OP_RCP,
        /// y=1/(1+exp(-x)).
        Sigmoid = CUTENSOR_OP_SIGMOID,
        /// y=tanh(x).
        Tanh = CUTENSOR_OP_TANH,
        /// Exponentiation.
        Exp = CUTENSOR_OP_EXP,
        /// Log (base e).
        Log = CUTENSOR_OP_LOG,
        /// Absolute value.
        Abs = CUTENSOR_OP_ABS,
        /// Negation.
        Neg = CUTENSOR_OP_NEG,
        /// Sine.
        Sin = CUTENSOR_OP_SIN,
        /// Cosine.
        Cos = CUTENSOR_OP_COS,
        /// Tangent.
        Tan = CUTENSOR_OP_TAN,
        /// Hyperbolic sine.
        Sinh = CUTENSOR_OP_SINH,
        /// Hyperbolic cosine.
        Cosh = CUTENSOR_OP_COSH,
        /// Inverse sine.
        Asin = CUTENSOR_OP_ASIN,
        /// Inverse cosine.
        Acos = CUTENSOR_OP_ACOS,
        /// Inverse tangent.
        Atan = CUTENSOR_OP_ATAN,
        /// Inverse hyperbolic sine.
        Asinh = CUTENSOR_OP_ASINH,
        /// Inverse hyperbolic cosine.
        Acosh = CUTENSOR_OP_ACOSH,
        /// Inverse hyperbolic tangent.
        Atanh = CUTENSOR_OP_ATANH,
        /// Ceiling.
        Ceil = CUTENSOR_OP_CEIL,
        /// Floor.
        Floor = CUTENSOR_OP_FLOOR,
        /// Mish y=x\*tanh(softplus(x)).
        Mish = CUTENSOR_OP_MISH,
        /// Swish y=x\*sigmoid(x).
        Swish = CUTENSOR_OP_SWISH,
        /// Softplus y=log(exp(x)+1).
        SoftPlus = CUTENSOR_OP_SOFT_PLUS,
        /// Softsign y=x/(abs(x)+1).
        SoftSign = CUTENSOR_OP_SOFT_SIGN,
        /// Addition of two elements.
        Add = CUTENSOR_OP_ADD,
        /// Multiplication of two elements.
        Mul = CUTENSOR_OP_MUL,
        /// Maximum of two elements.
        Max = CUTENSOR_OP_MAX,
        /// Minimum of two elements.
        Min = CUTENSOR_OP_MIN,
        /// Reserved for internal use only.
        Unknown = CUTENSOR_OP_UNKNOWN,
    }
}

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

singe_core::native_enum! {
    /// Controls the workspace amount suggested by [`Plan::estimate_workspace_size`](crate::plan::Plan::estimate_workspace_size).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum WorkspacePreference: sys::cutensorWorksizePreference_t {
        /// Least memory requirement; at least one algorithm is available.
        Min = CUTENSOR_WORKSPACE_MIN,
        /// Aims to attain high performance while also reducing the workspace requirement.
        Default = CUTENSOR_WORKSPACE_DEFAULT,
        /// Highest memory requirement; all algorithms are available.
        Max = CUTENSOR_WORKSPACE_MAX,
    }
}

impl_enum_display!(WorkspacePreference, {
    WorkspacePreference::Min => "CUTENSOR_WORKSPACE_MIN",
    WorkspacePreference::Default => "CUTENSOR_WORKSPACE_DEFAULT",
    WorkspacePreference::Max => "CUTENSOR_WORKSPACE_MAX",
});

singe_core::native_enum! {
    /// Controls what is considered a cache hit.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CacheMode: sys::cutensorCacheMode_t {
        /// Plans are not cached.
        None = CUTENSOR_CACHE_MODE_NONE,
        /// All descriptor parameters must match the cached plan.
        Pedantic = CUTENSOR_CACHE_MODE_PEDANTIC,
    }
}

impl_enum_display!(CacheMode, {
    CacheMode::None => "CUTENSOR_CACHE_MODE_NONE",
    CacheMode::Pedantic => "CUTENSOR_CACHE_MODE_PEDANTIC",
});

singe_core::native_enum! {
    /// Attributes of a cuTENSOR operation descriptor that can be read or modified.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum OperationDescriptorAttribute: sys::cutensorOperationDescriptorAttribute_t {
        /// `i32` tag that distinguishes otherwise identical problems in the software-managed plan cache. (default value: 0).
        Tag = CUTENSOR_OPERATION_DESCRIPTOR_TAG,
        /// [`DataType`](singe_cuda::data_type::DataType): data type of the scaling factors.
        ScalarType = CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
        /// `f32`: number of floating-point operations necessary to perform this operation, assuming all scalars are nonzero unless otherwise specified.
        Flops = CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
        /// `f32`: minimal number of bytes transferred from or to global memory, assuming all scalars are nonzero unless otherwise specified.
        MovedBytes = CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
        /// `u32` array of length `desc_out.num_modes`; entry `i` is the number of padded values to the left of dimension `i`.
        PaddingLeft = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT,
        /// `u32` array of length `desc_out.num_modes`; entry `i` is the number of padded values to the right of dimension `i`.
        PaddingRight = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT,
        /// Host-side pointer to an element of the same type as the output tensor: constant padding value.
        PaddingValue = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE,
    }
}

impl_enum_display!(OperationDescriptorAttribute, {
    OperationDescriptorAttribute::Tag => "CUTENSOR_OPERATION_DESCRIPTOR_TAG",
    OperationDescriptorAttribute::ScalarType => "CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE",
    OperationDescriptorAttribute::Flops => "CUTENSOR_OPERATION_DESCRIPTOR_FLOPS",
    OperationDescriptorAttribute::MovedBytes => "CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES",
    OperationDescriptorAttribute::PaddingLeft => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT",
    OperationDescriptorAttribute::PaddingRight => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT",
    OperationDescriptorAttribute::PaddingValue => "CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE",
});

singe_core::native_enum! {
    /// Attributes of a [`PlanPreference`](crate::plan::PlanPreference) that can be read or modified.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PlanPreferenceAttribute: sys::cutensorPlanPreferenceAttribute_t {
        /// [`AutotuneMode`]: determines whether recurrent executions of the plan autotune.
        AutotuneMode = CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
        /// [`CacheMode`]: determines whether the algorithm or kernel for this plan is cached and controls what is considered a cache hit.
        CacheMode = CUTENSOR_PLAN_PREFERENCE_CACHE_MODE,
        /// `i32`: only applicable when [`PlanPreferenceAttribute::CacheMode`] is set to [`AutotuneMode::Incremental`].
        IncrementalCount = CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
        /// [`Algorithm`]: fixes the algorithm.
        Algorithm = CUTENSOR_PLAN_PREFERENCE_ALGO,
        /// `i32`: fixes a kernel subvariant of an algorithm; for example, `kernel_rank == 1` with [`Algorithm::Tgett`] selects the second-best GETT kernel variant according to cuTENSOR's performance model, while `kernel_rank == 2` selects the third-best.
        KernelRank = CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
        /// [`JitMode`]: determines whether just-in-time compilation is enabled (default: [`JitMode::None`]).
        JitMode = CUTENSOR_PLAN_PREFERENCE_JIT,
        /// `i32`: plan for a specific GPU architecture instead of the architecture associated with the context.
        /// The value encodes the SM version as `10 * SM.major + SM.minor`.
        /// Currently only SM versions 80, 90 and 100 are supported.
        GpuArch = CUTENSOR_PLAN_PREFERENCE_GPU_ARCH,
    }
}

impl_enum_display!(PlanPreferenceAttribute, {
    PlanPreferenceAttribute::AutotuneMode => "CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE",
    PlanPreferenceAttribute::CacheMode => "CUTENSOR_PLAN_PREFERENCE_CACHE_MODE",
    PlanPreferenceAttribute::IncrementalCount => "CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT",
    PlanPreferenceAttribute::Algorithm => "CUTENSOR_PLAN_PREFERENCE_ALGO",
    PlanPreferenceAttribute::KernelRank => "CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK",
    PlanPreferenceAttribute::JitMode => "CUTENSOR_PLAN_PREFERENCE_JIT",
    PlanPreferenceAttribute::GpuArch => "CUTENSOR_PLAN_PREFERENCE_GPU_ARCH",
});

singe_core::native_enum! {
    /// Controls cuTENSOR autotuning behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum AutotuneMode: sys::cutensorAutotuneMode_t {
        /// Indicates no autotuning (default); the cache reduces plan-creation overhead.
        /// On a cache hit, the cached plan is reused; otherwise the plan cache is ignored.
        None = CUTENSOR_AUTOTUNE_MODE_NONE,
        /// Indicates incremental autotuning.
        /// Each [`Plan::create`](crate::plan::Plan::create) call creates a plan based on a different algorithm or kernel.
        /// [`PlanPreferenceAttribute::IncrementalCount`] defines the maximum number of kernels tested.
        /// May choose different algorithms across plan creations, so it does not guarantee bitwise-identical results.
        Incremental = CUTENSOR_AUTOTUNE_MODE_INCREMENTAL,
    }
}

impl_enum_display!(AutotuneMode, {
    AutotuneMode::None => "CUTENSOR_AUTOTUNE_MODE_NONE",
    AutotuneMode::Incremental => "CUTENSOR_AUTOTUNE_MODE_INCREMENTAL",
});

singe_core::native_enum! {
    /// Controls cuTENSOR just-in-time compilation behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum JitMode: sys::cutensorJitMode_t {
        /// Indicates that no kernel is just-in-time compiled.
        None = CUTENSOR_JIT_MODE_NONE,
        /// Indicates that the corresponding plan tries to compile a dedicated kernel for the given operation.
        /// Only supported for GPUs with compute capability >= 8.0 (Ampere or newer).
        Default = CUTENSOR_JIT_MODE_DEFAULT,
    }
}

impl_enum_display!(JitMode, {
    JitMode::None => "CUTENSOR_JIT_MODE_NONE",
    JitMode::Default => "CUTENSOR_JIT_MODE_DEFAULT",
});

singe_core::native_enum! {
    /// Attributes that can be queried from a cuTENSOR [`Plan`](crate::plan::Plan).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PlanAttribute: sys::cutensorPlanAttribute_t {
        /// Exact required workspace in bytes needed to execute the plan.
        RequiredWorkspace = CUTENSOR_PLAN_REQUIRED_WORKSPACE,
    }
}

impl_enum_display!(PlanAttribute, {
    PlanAttribute::RequiredWorkspace => "CUTENSOR_PLAN_REQUIRED_WORKSPACE",
});

/// An integer mode label identifying the same logical index across tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Mode(i32);

impl Mode {
    /// Creates a symbolic tensor mode from its native integer label.
    #[must_use]
    pub const fn new(raw: i32) -> Self {
        Self(raw)
    }

    /// Creates a mode whose label is the Unicode scalar value of `mode`.
    #[must_use]
    pub const fn from_char(mode: char) -> Self {
        Self(mode as i32)
    }

    /// Returns the native integer mode label.
    #[must_use]
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
        if let Some(ch) = u32::try_from(self.0).ok().and_then(char::from_u32)
            && !ch.is_control()
        {
            return write!(f, "{ch}");
        }

        write!(f, "{}", self.0)
    }
}
