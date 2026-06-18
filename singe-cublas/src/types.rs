#![allow(deprecated)]

use std::fmt::{self, Display, Formatter};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cublas_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Indicates whether scalar values are read from host memory or device memory.
/// If an operation uses several scalar values, all of them must use the same pointer mode.
/// The pointer mode can be set and retrieved using
/// [`Context::set_scalar_pointer_mode`](crate::context::Context::set_scalar_pointer_mode) and
/// [`Context::scalar_pointer_mode`](crate::context::Context::scalar_pointer_mode), respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PointerMode {
    /// Scalars are read from host memory.
    Host = sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST as _,
    /// Scalars are read from device memory.
    Device = sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE as _,
}

impl_enum_conversion!(sys::cublasPointerMode_t, PointerMode);

/// Indicates whether cuBLAS operations that have an alternate implementation using atomics can use it.
/// The atomics mode can be set and queried using
/// [`Context::set_atomics_mode`](crate::context::Context::set_atomics_mode) and
/// [`Context::atomics_mode`](crate::context::Context::atomics_mode), respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AtomicsMode {
    /// The usage of atomics is not allowed.
    NotAllowed = sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED as _,
    /// The usage of atomics is allowed.
    Allowed = sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED as _,
}

impl_enum_conversion!(sys::cublasAtomicsMode_t, AtomicsMode);

bitflags::bitflags! {
    /// [`MathMode`] is used with [`Context::set_math_mode`](crate::context::Context::set_math_mode)
    /// to choose compute precision modes.
    ///
    /// The values can be combined with [`MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION`],
    /// except for deprecated [`MathMode::TENSOR_OP`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MathMode: u32 {
        /// Default and highest-performance mode, using compute and intermediate storage
        /// precisions with at least the requested mantissa and exponent bit counts.
        /// Tensor Cores are used whenever possible.
        const DEFAULT = sys::cublasMath_t::CUBLAS_DEFAULT_MATH as _;
        /// Deprecated mode.
        /// Allows the library to use Tensor Core operations whenever possible.
        /// For single precision GEMM operations, cuBLAS uses the [`ComputeType::F32FastF16`] compute type.
        #[deprecated]
        const TENSOR_OP = sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH as _;
        /// Uses the prescribed precision and standardized arithmetic for all calculation
        /// phases, primarily for numerical robustness studies, testing, and debugging.
        /// May be slower than the other modes.
        const PEDANTIC = sys::cublasMath_t::CUBLAS_PEDANTIC_MATH as _;
        /// Enable acceleration of single-precision operations using TF32 tensor cores.
        /// Input conversions round to nearest even.
        const TF32_TENSOR_OP = sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH as _;
        /// Enable acceleration of single-precision operations using the BF16x9 algorithm.
        /// See [`EmulationStrategy`] for floating-point emulation controls.
        /// For single precision GEMM operations, cuBLAS uses the [`ComputeType::F32EmulatedBf16x9`] compute type.
        const FP32_EMULATED_BF16X9 = sys::cublasMath_t::CUBLAS_FP32_EMULATED_BF16X9_MATH as _;
        /// Enable acceleration of double-precision operations using fixed-point emulation algorithms.
        /// See [`EmulationStrategy`] for floating-point emulation controls.
        const FP64_EMULATED_FIXED_POINT =
            sys::cublasMath_t::CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH as _;
        /// Forces reductions during matrix multiplications to use the accumulator
        /// type, not the output type, for mixed precision operations whose output
        /// precision is lower than the compute type precision.
        const DISALLOW_REDUCED_PRECISION_REDUCTION =
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION as _;
    }
}

impl MathMode {
    pub(crate) unsafe fn from_raw_bits(bits: u32) -> Self {
        Self::from_bits_retain(bits)
    }

    pub(crate) unsafe fn as_raw(self) -> sys::cublasMath_t {
        // SAFETY: cuBLAS documents cublasMath_t as accepting bitwise combinations
        // even though bindgen represents it as a Rust enum. The ABI is a u32 C
        // enum value, and this is used only at the FFI boundary.
        unsafe { std::mem::transmute::<u32, sys::cublasMath_t>(self.bits()) }
    }
}

/// Indicates whether the lower or upper part of the dense matrix is filled and used.
///
/// This corresponds to BLAS `L`/`l` (lower) and `U`/`u` (upper) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FillMode {
    /// The lower part of the matrix is filled.
    Lower = sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER as _,
    /// The upper part of the matrix is filled.
    Upper = sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER as _,
    /// The full matrix is filled.
    Full = sys::cublasFillMode_t::CUBLAS_FILL_MODE_FULL as _,
}

impl_enum_conversion!(sys::cublasFillMode_t, FillMode);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DiagonalType {
    NonUnit = sys::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT as _,
    Unit = sys::cublasDiagType_t::CUBLAS_DIAG_UNIT as _,
}

impl_enum_conversion!(sys::cublasDiagType_t, DiagonalType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SideMode {
    Left = sys::cublasSideMode_t::CUBLAS_SIDE_LEFT as _,
    Right = sys::cublasSideMode_t::CUBLAS_SIDE_RIGHT as _,
}

impl_enum_conversion!(sys::cublasSideMode_t, SideMode);

/// Selects the operation to perform with a dense matrix.
///
/// This corresponds to BLAS `N`/`n` (non-transpose), `T`/`t` (transpose), and
/// `C`/`c` (conjugate transpose) arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Operation {
    /// Non-transpose operation.
    NonTranspose = sys::cublasOperation_t::CUBLAS_OP_N as _,
    /// Transpose operation.
    Transpose = sys::cublasOperation_t::CUBLAS_OP_T as _,
    /// Conjugate transpose operation.
    ConjugateTranspose = sys::cublasOperation_t::CUBLAS_OP_C as _,
    Conjugate = sys::cublasOperation_t::CUBLAS_OP_CONJG as _,
}

impl Operation {
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

impl_enum_conversion!(sys::cublasOperation_t, Operation);

/// Specifies the GEMM algorithm for matrix-matrix multiplication on GPU architectures up to `sm_75`.
/// On `sm_80` and newer GPU architectures, this value has no effect.
///
/// `Algo0` through `Algo23` explicitly choose legacy GEMM algorithms. The
/// deprecated `Algo*TensorOp` values explicitly choose legacy Tensor Core GEMM
/// algorithms and allow reduced-precision [`ComputeType::F32FastF16`] kernels
/// for backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum GemmAlgorithm {
    Default = sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT as _,
    Algo0 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0 as _,
    Algo1 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1 as _,
    Algo2 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2 as _,
    Algo3 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3 as _,
    Algo4 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4 as _,
    Algo5 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5 as _,
    Algo6 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6 as _,
    Algo7 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7 as _,
    Algo8 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8 as _,
    Algo9 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9 as _,
    Algo10 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10 as _,
    Algo11 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11 as _,
    Algo12 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12 as _,
    Algo13 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13 as _,
    Algo14 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14 as _,
    Algo15 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15 as _,
    Algo16 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO16 as _,
    Algo17 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO17 as _,
    Algo18 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO18 as _,
    Algo19 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO19 as _,
    Algo20 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO20 as _,
    Algo21 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO21 as _,
    Algo22 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO22 as _,
    Algo23 = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO23 as _,
    #[deprecated]
    Algo0TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP as _,
    #[deprecated]
    Algo1TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1_TENSOR_OP as _,
    #[deprecated]
    Algo2TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2_TENSOR_OP as _,
    #[deprecated]
    Algo3TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3_TENSOR_OP as _,
    #[deprecated]
    Algo4TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4_TENSOR_OP as _,
    #[deprecated]
    Algo5TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5_TENSOR_OP as _,
    #[deprecated]
    Algo6TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6_TENSOR_OP as _,
    #[deprecated]
    Algo7TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7_TENSOR_OP as _,
    #[deprecated]
    Algo8TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8_TENSOR_OP as _,
    #[deprecated]
    Algo9TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9_TENSOR_OP as _,
    #[deprecated]
    Algo10TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10_TENSOR_OP as _,
    #[deprecated]
    Algo11TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11_TENSOR_OP as _,
    #[deprecated]
    Algo12TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12_TENSOR_OP as _,
    #[deprecated]
    Algo13TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13_TENSOR_OP as _,
    #[deprecated]
    Algo14TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14_TENSOR_OP as _,
    #[deprecated]
    Algo15TensorOp = sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP as _,
    /// `EXPERIMENTAL` Benchmarks available algorithms and chooses the optimal one for the given problem configuration.
    /// The solution is cached in the cuBLAS handle so later calls with the same problem size use the cached configuration.
    /// To avoid overwriting application data, cuBLAS allocates memory matching the output size.
    /// Benchmarking is not supported during stream capture; [`crate::error::Status::NotSupported`] is returned if no configuration was found in the cache for the given problem size.
    Autotune = sys::cublasGemmAlgo_t::CUBLAS_GEMM_AUTOTUNE as _,
}

impl_enum_conversion!(i32, sys::cublasGemmAlgo_t, GemmAlgorithm);

/// [`ComputeType`] is used in [`gemm_ex`](crate::blas::level3::gemm_ex) and [`matmul`](crate::lt::matmul::matmul) (including all batched and strided batched variants) to choose compute precision modes as defined below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeType {
    F16 = sys::cublasComputeType_t::CUBLAS_COMPUTE_16F as _,
    F16Pedantic = sys::cublasComputeType_t::CUBLAS_COMPUTE_16F_PEDANTIC as _,
    F32 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F as _,
    F32Pedantic = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC as _,
    F32FastF16 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F as _,
    F32FastBf16 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF as _,
    F32FastTf32 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 as _,
    F32EmulatedBf16x9 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_EMULATED_16BFX9 as _,
    F64 = sys::cublasComputeType_t::CUBLAS_COMPUTE_64F as _,
    F64Pedantic = sys::cublasComputeType_t::CUBLAS_COMPUTE_64F_PEDANTIC as _,
    F64EmulatedFixedPoint = sys::cublasComputeType_t::CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT as _,
    I32 = sys::cublasComputeType_t::CUBLAS_COMPUTE_32I as _,
    I32Pedantic = sys::cublasComputeType_t::CUBLAS_COMPUTE_32I_PEDANTIC as _,
}

impl_enum_conversion!(sys::cublasComputeType_t, ComputeType);

/// [`EmulationStrategy`] is used with
/// [`Context::set_emulation_strategy`](crate::context::Context::set_emulation_strategy) to choose
/// how to leverage floating point emulation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EmulationStrategy {
    /// Default emulation strategy; equivalent to [`EmulationStrategy::Performant`]
    /// unless the `CUBLAS_EMULATION_STRATEGY` environment variable is set.
    Default = sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_DEFAULT as _,
    /// Uses emulation whenever it provides a performance benefit.
    Performant = sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_PERFORMANT as _,
    /// Uses emulation whenever possible.
    Eager = sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_EAGER as _,
}

impl_enum_conversion!(sys::cublasEmulationStrategy_t, EmulationStrategy);

impl_enum_display!(PointerMode, {
    PointerMode::Host => "CUBLAS_POINTER_MODE_HOST",
    PointerMode::Device => "CUBLAS_POINTER_MODE_DEVICE",
});

impl_enum_display!(AtomicsMode, {
    AtomicsMode::NotAllowed => "CUBLAS_ATOMICS_NOT_ALLOWED",
    AtomicsMode::Allowed => "CUBLAS_ATOMICS_ALLOWED",
});

impl Display for MathMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        let base = *self - Self::DISALLOW_REDUCED_PRECISION_REDUCTION;
        if base.is_empty() || base == Self::DEFAULT {
            parts.push("CUBLAS_DEFAULT_MATH");
        } else if base == Self::TENSOR_OP {
            parts.push("CUBLAS_TENSOR_OP_MATH");
        } else if base == Self::PEDANTIC {
            parts.push("CUBLAS_PEDANTIC_MATH");
        } else if base == Self::TF32_TENSOR_OP {
            parts.push("CUBLAS_TF32_TENSOR_OP_MATH");
        } else if base == Self::FP32_EMULATED_BF16X9 {
            parts.push("CUBLAS_FP32_EMULATED_BF16X9_MATH");
        } else if base == Self::FP64_EMULATED_FIXED_POINT {
            parts.push("CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH");
        } else {
            return write!(f, "CUBLAS_MATH_UNKNOWN({})", self.bits());
        }
        if self.contains(Self::DISALLOW_REDUCED_PRECISION_REDUCTION) {
            parts.push("CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION");
        }
        f.write_str(&parts.join(" | "))
    }
}

impl_enum_display!(FillMode, {
    FillMode::Lower => "CUBLAS_FILL_MODE_LOWER",
    FillMode::Upper => "CUBLAS_FILL_MODE_UPPER",
    FillMode::Full => "CUBLAS_FILL_MODE_FULL",
});

impl_enum_display!(DiagonalType, {
    DiagonalType::NonUnit => "CUBLAS_DIAG_NON_UNIT",
    DiagonalType::Unit => "CUBLAS_DIAG_UNIT",
});

impl_enum_display!(SideMode, {
    SideMode::Left => "CUBLAS_SIDE_LEFT",
    SideMode::Right => "CUBLAS_SIDE_RIGHT",
});

impl_enum_display!(Operation, {
    Operation::NonTranspose => "CUBLAS_OP_N",
    Operation::Transpose => "CUBLAS_OP_T",
    Operation::ConjugateTranspose => "CUBLAS_OP_C",
    Operation::Conjugate => "CUBLAS_OP_CONJG",
});

impl_enum_display!(GemmAlgorithm, {
    Self::Default => "CUBLAS_GEMM_DFALT",
    Self::Algo0 => "CUBLAS_GEMM_ALGO0",
    Self::Algo1 => "CUBLAS_GEMM_ALGO1",
    Self::Algo2 => "CUBLAS_GEMM_ALGO2",
    Self::Algo3 => "CUBLAS_GEMM_ALGO3",
    Self::Algo4 => "CUBLAS_GEMM_ALGO4",
    Self::Algo5 => "CUBLAS_GEMM_ALGO5",
    Self::Algo6 => "CUBLAS_GEMM_ALGO6",
    Self::Algo7 => "CUBLAS_GEMM_ALGO7",
    Self::Algo8 => "CUBLAS_GEMM_ALGO8",
    Self::Algo9 => "CUBLAS_GEMM_ALGO9",
    Self::Algo10 => "CUBLAS_GEMM_ALGO10",
    Self::Algo11 => "CUBLAS_GEMM_ALGO11",
    Self::Algo12 => "CUBLAS_GEMM_ALGO12",
    Self::Algo13 => "CUBLAS_GEMM_ALGO13",
    Self::Algo14 => "CUBLAS_GEMM_ALGO14",
    Self::Algo15 => "CUBLAS_GEMM_ALGO15",
    Self::Algo16 => "CUBLAS_GEMM_ALGO16",
    Self::Algo17 => "CUBLAS_GEMM_ALGO17",
    Self::Algo18 => "CUBLAS_GEMM_ALGO18",
    Self::Algo19 => "CUBLAS_GEMM_ALGO19",
    Self::Algo20 => "CUBLAS_GEMM_ALGO20",
    Self::Algo21 => "CUBLAS_GEMM_ALGO21",
    Self::Algo22 => "CUBLAS_GEMM_ALGO22",
    Self::Algo23 => "CUBLAS_GEMM_ALGO23",
    Self::Algo0TensorOp => "CUBLAS_GEMM_ALGO0_TENSOR_OP",
    Self::Algo1TensorOp => "CUBLAS_GEMM_ALGO1_TENSOR_OP",
    Self::Algo2TensorOp => "CUBLAS_GEMM_ALGO2_TENSOR_OP",
    Self::Algo3TensorOp => "CUBLAS_GEMM_ALGO3_TENSOR_OP",
    Self::Algo4TensorOp => "CUBLAS_GEMM_ALGO4_TENSOR_OP",
    Self::Algo5TensorOp => "CUBLAS_GEMM_ALGO5_TENSOR_OP",
    Self::Algo6TensorOp => "CUBLAS_GEMM_ALGO6_TENSOR_OP",
    Self::Algo7TensorOp => "CUBLAS_GEMM_ALGO7_TENSOR_OP",
    Self::Algo8TensorOp => "CUBLAS_GEMM_ALGO8_TENSOR_OP",
    Self::Algo9TensorOp => "CUBLAS_GEMM_ALGO9_TENSOR_OP",
    Self::Algo10TensorOp => "CUBLAS_GEMM_ALGO10_TENSOR_OP",
    Self::Algo11TensorOp => "CUBLAS_GEMM_ALGO11_TENSOR_OP",
    Self::Algo12TensorOp => "CUBLAS_GEMM_ALGO12_TENSOR_OP",
    Self::Algo13TensorOp => "CUBLAS_GEMM_ALGO13_TENSOR_OP",
    Self::Algo14TensorOp => "CUBLAS_GEMM_ALGO14_TENSOR_OP",
    Self::Algo15TensorOp => "CUBLAS_GEMM_ALGO15_TENSOR_OP",
    Self::Autotune => "CUBLAS_GEMM_AUTOTUNE",
});

impl_enum_display!(ComputeType, {
    ComputeType::F16 => "CUBLAS_COMPUTE_16F",
    ComputeType::F16Pedantic => "CUBLAS_COMPUTE_16F_PEDANTIC",
    ComputeType::F32 => "CUBLAS_COMPUTE_32F",
    ComputeType::F32Pedantic => "CUBLAS_COMPUTE_32F_PEDANTIC",
    ComputeType::F32FastF16 => "CUBLAS_COMPUTE_32F_FAST_16F",
    ComputeType::F32FastBf16 => "CUBLAS_COMPUTE_32F_FAST_16BF",
    ComputeType::F32FastTf32 => "CUBLAS_COMPUTE_32F_FAST_TF32",
    ComputeType::F32EmulatedBf16x9 => "CUBLAS_COMPUTE_32F_EMULATED_16BFX9",
    ComputeType::F64 => "CUBLAS_COMPUTE_64F",
    ComputeType::F64Pedantic => "CUBLAS_COMPUTE_64F_PEDANTIC",
    ComputeType::F64EmulatedFixedPoint => "CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT",
    ComputeType::I32 => "CUBLAS_COMPUTE_32I",
    ComputeType::I32Pedantic => "CUBLAS_COMPUTE_32I_PEDANTIC",
});

impl_enum_display!(EmulationStrategy, {
    EmulationStrategy::Default => "CUBLAS_EMULATION_STRATEGY_DEFAULT",
    EmulationStrategy::Performant => "CUBLAS_EMULATION_STRATEGY_PERFORMANT",
    EmulationStrategy::Eager => "CUBLAS_EMULATION_STRATEGY_EAGER",
});
