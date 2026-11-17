use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cublas_sys as sys;

singe_core::native_enum! {
    /// Indicates whether scalar values are read from host memory or device memory.
    ///
    /// If an operation uses several scalar values, all of them must use the same pointer mode.
    /// Each operation selects the mode from the location of its [`Scalar`](crate::scalar::Scalar) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PointerMode: sys::cublasPointerMode_t {
        /// Scalars are read from host memory.
        Host = CUBLAS_POINTER_MODE_HOST,
        /// Scalars are read from device memory.
        Device = CUBLAS_POINTER_MODE_DEVICE,
    }
}

singe_core::native_enum! {
    /// Indicates whether cuBLAS operations that have an alternate implementation using atomics can use it.
    ///
    /// The atomics mode can be set and queried using [`Context::set_atomics_mode`](crate::context::Context::set_atomics_mode) and [`Context::atomics_mode`](crate::context::Context::atomics_mode), respectively.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum AtomicsMode: sys::cublasAtomicsMode_t {
        /// The usage of atomics is not allowed.
        NotAllowed = CUBLAS_ATOMICS_NOT_ALLOWED,
        /// The usage of atomics is allowed.
        Allowed = CUBLAS_ATOMICS_ALLOWED,
    }
}

bitflags::bitflags! {
    /// [`MathMode`] is used with [`Context::set_math_mode`](crate::context::Context::set_math_mode) to choose compute precision modes.
    ///
    /// The values can be combined with [`MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MathMode: u32 {
        /// Default and highest-performance mode, using compute and intermediate storage precisions with at least the requested mantissa and exponent bit counts.
        /// Tensor Cores are used whenever possible.
        const DEFAULT = sys::cublasMath_t::CUBLAS_DEFAULT_MATH.0;
        /// Uses the prescribed precision and standardized arithmetic for all calculation phases, primarily for numerical robustness studies, testing, and debugging.
        /// May be slower than the other modes.
        const PEDANTIC = sys::cublasMath_t::CUBLAS_PEDANTIC_MATH.0;
        /// Enable acceleration of single-precision operations using TF32 tensor cores.
        /// Input conversions round to nearest even.
        const TF32_TENSOR_OP = sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH.0;
        /// Enable acceleration of single-precision operations using the BF16x9 algorithm.
        /// See [`EmulationStrategy`] for floating-point emulation controls.
        /// For single precision GEMM operations, cuBLAS uses the [`ComputeType::F32EmulatedBf16x9`] compute type.
        const FP32_EMULATED_BF16X9 = sys::cublasMath_t::CUBLAS_FP32_EMULATED_BF16X9_MATH.0;
        /// Enable acceleration of double-precision operations using fixed-point emulation algorithms.
        /// See [`EmulationStrategy`] for floating-point emulation controls.
        const FP64_EMULATED_FIXED_POINT =
            sys::cublasMath_t::CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH.0;
        /// Forces reductions during matrix multiplications to use the accumulator type, not the output type, for mixed precision operations whose output precision is lower than the compute type precision.
        const DISALLOW_REDUCED_PRECISION_REDUCTION =
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION.0;
    }
}

singe_core::native_enum! {
    /// Indicates whether the lower or upper part of the dense matrix is filled and used.
    ///
    /// This corresponds to BLAS `L`/`l` (lower) and `U`/`u` (upper) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FillMode: sys::cublasFillMode_t {
        /// The lower part of the matrix is filled.
        Lower = CUBLAS_FILL_MODE_LOWER,
        /// The upper part of the matrix is filled.
        Upper = CUBLAS_FILL_MODE_UPPER,
        /// The full matrix is filled.
        Full = CUBLAS_FILL_MODE_FULL,
    }
}

singe_core::native_enum! {
    /// Whether a triangular matrix stores its diagonal or treats every diagonal element as one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DiagonalType: sys::cublasDiagType_t {
        /// The matrix diagonal has non-unit elements.
        NonUnit = CUBLAS_DIAG_NON_UNIT,
        /// The matrix diagonal has unit elements.
        Unit = CUBLAS_DIAG_UNIT,
    }
}

singe_core::native_enum! {
    /// Side on which a matrix appears in a BLAS matrix equation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SideMode: sys::cublasSideMode_t {
        /// The matrix is on the left side in the equation.
        Left = CUBLAS_SIDE_LEFT,
        /// The matrix is on the right side in the equation.
        Right = CUBLAS_SIDE_RIGHT,
    }
}

singe_core::native_enum! {
    /// Selects the operation to perform with a dense matrix.
    ///
    /// This corresponds to BLAS `N`/`n` (non-transpose), `T`/`t` (transpose), and `C`/`c` (conjugate transpose) arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Operation: sys::cublasOperation_t {
        /// Non-transpose operation.
        NonTranspose = CUBLAS_OP_N,
        /// Transpose operation.
        Transpose = CUBLAS_OP_T,
        /// Conjugate transpose operation.
        ConjugateTranspose = CUBLAS_OP_C,
        /// Conjugation without transposition, which cuBLAS does not support yet.
        Conjugate = CUBLAS_OP_CONJG,
    }
}

impl Operation {
    /// Conjugate-transpose operation; an alias for [`Self::ConjugateTranspose`].
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

singe_core::native_enum! {
    /// Specifies the GEMM algorithm for matrix-matrix multiplication on GPU architectures up to `sm_75`.
    /// On `sm_80` and newer GPU architectures, this value has no effect.
    ///
    /// `Algo0` through `Algo23` explicitly choose legacy GEMM algorithms.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum GemmAlgorithm: sys::cublasGemmAlgo_t {
        /// Heuristic selection of the native GEMM algorithm.
        /// This selector has no effect on Ampere or later GPU architectures.
        Default = CUBLAS_GEMM_DFALT,
        /// Selects cuBLAS algorithm `0`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo0 = CUBLAS_GEMM_ALGO0,
        /// Selects cuBLAS algorithm `1`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo1 = CUBLAS_GEMM_ALGO1,
        /// Selects cuBLAS algorithm `2`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo2 = CUBLAS_GEMM_ALGO2,
        /// Selects cuBLAS algorithm `3`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo3 = CUBLAS_GEMM_ALGO3,
        /// Selects cuBLAS algorithm `4`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo4 = CUBLAS_GEMM_ALGO4,
        /// Selects cuBLAS algorithm `5`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo5 = CUBLAS_GEMM_ALGO5,
        /// Selects cuBLAS algorithm `6`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo6 = CUBLAS_GEMM_ALGO6,
        /// Selects cuBLAS algorithm `7`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo7 = CUBLAS_GEMM_ALGO7,
        /// Selects cuBLAS algorithm `8`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo8 = CUBLAS_GEMM_ALGO8,
        /// Selects cuBLAS algorithm `9`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo9 = CUBLAS_GEMM_ALGO9,
        /// Selects cuBLAS algorithm `10`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo10 = CUBLAS_GEMM_ALGO10,
        /// Selects cuBLAS algorithm `11`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo11 = CUBLAS_GEMM_ALGO11,
        /// Selects cuBLAS algorithm `12`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo12 = CUBLAS_GEMM_ALGO12,
        /// Selects cuBLAS algorithm `13`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo13 = CUBLAS_GEMM_ALGO13,
        /// Selects cuBLAS algorithm `14`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo14 = CUBLAS_GEMM_ALGO14,
        /// Selects cuBLAS algorithm `15`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo15 = CUBLAS_GEMM_ALGO15,
        /// Selects cuBLAS algorithm `16`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo16 = CUBLAS_GEMM_ALGO16,
        /// Selects cuBLAS algorithm `17`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo17 = CUBLAS_GEMM_ALGO17,
        /// Selects cuBLAS algorithm `18`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo18 = CUBLAS_GEMM_ALGO18,
        /// Selects cuBLAS algorithm `19`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo19 = CUBLAS_GEMM_ALGO19,
        /// Selects cuBLAS algorithm `20`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo20 = CUBLAS_GEMM_ALGO20,
        /// Selects cuBLAS algorithm `21`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo21 = CUBLAS_GEMM_ALGO21,
        /// Selects cuBLAS algorithm `22`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo22 = CUBLAS_GEMM_ALGO22,
        /// Selects cuBLAS algorithm `23`.
        ///
        /// This selection has no effect on NVIDIA Ampere GPUs or later architectures.
        Algo23 = CUBLAS_GEMM_ALGO23,
        /// Experimental: benchmarks the available algorithms and chooses the optimal one for the given problem configuration.
        /// The solution is cached in the cuBLAS handle so later calls with the same problem size use the cached configuration.
        /// To avoid overwriting application data, cuBLAS allocates memory matching the output size.
        /// Benchmarking is not supported during stream capture; [`crate::error::Status::NotSupported`] is returned if no configuration was found in the cache for the given problem size.
        Autotune = CUBLAS_GEMM_AUTOTUNE,
    }
}

singe_core::native_enum! {
    /// [`ComputeType`] is used in [`gemm_ex`](crate::blas::extension::gemm_ex) and `cuBLASLt` matrix multiplication (including all batched and strided batched variants) to choose the compute precision.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ComputeType: sys::cublasComputeType_t {
        /// This is the default and highest-performance mode for 16-bit half precision floating point and all compute and intermediate storage precisions with at least 16-bit half precision.
        /// Tensor Cores will be used whenever possible.
        F16 = CUBLAS_COMPUTE_16F,
        /// This mode uses 16-bit half precision floating point standardized arithmetic for all phases of calculations and is primarily intended for numerical robustness studies, testing, and debugging.
        /// This mode might not be as performant as the other modes since it disables use of tensor cores.
        F16Pedantic = CUBLAS_COMPUTE_16F_PEDANTIC,
        /// This is the default 32-bit single precision floating point and uses compute and intermediate storage precisions of at least 32-bits.
        F32 = CUBLAS_COMPUTE_32F,
        /// Uses 32-bit single precision floating point arithmetic for all phases of calculations and also disables algorithmic optimizations such as Gaussian complexity reduction (3M).
        F32Pedantic = CUBLAS_COMPUTE_32F_PEDANTIC,
        /// Allows the library to use Tensor Cores with automatic down-conversion and 16-bit half-precision compute for 32-bit input and output matrices.
        F32FastF16 = CUBLAS_COMPUTE_32F_FAST_16F,
        /// Allows the library to use Tensor Cores with automatic down-conversion and bfloat16 compute for 32-bit input and output matrices.
        /// See [Alternate Floating Point](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-altfp) section for more details on bfloat16.
        F32FastBf16 = CUBLAS_COMPUTE_32F_FAST_16BF,
        /// Allows the library to use Tensor Cores with TF32 compute for 32-bit floating point input and output matrices.
        /// Note that input conversions round to nearest even.
        /// See [Alternate Floating Point](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-altfp) section for more details on TF32 compute.
        F32FastTf32 = CUBLAS_COMPUTE_32F_FAST_TF32,
        /// Allows the library to use the `BF16x9` floating point emulation algorithm for 32-bit floating point arithmetic.
        /// See [Floating Point Emulation](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#floating-point-emulation) for more details.
        F32EmulatedBf16x9 = CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
        /// This is the default 64-bit double precision floating point and uses compute and intermediate storage precisions of at least 64-bits.
        F64 = CUBLAS_COMPUTE_64F,
        /// Uses 64-bit double precision floating point arithmetic for all phases of calculations and also disables algorithmic optimizations such as Gaussian complexity reduction (3M).
        F64Pedantic = CUBLAS_COMPUTE_64F_PEDANTIC,
        /// Allows the library to use fixed-point emulation algorithms for 64-bit double precision floating point arithmetic.
        /// See [Floating Point Emulation](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#floating-point-emulation) for more details.
        F64EmulatedFixedPoint = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
        /// This is the default 32-bit integer mode and uses compute and intermediate storage precisions of at least 32-bits.
        I32 = CUBLAS_COMPUTE_32I,
        /// Uses 32-bit integer arithmetic for all phases of calculations.
        I32Pedantic = CUBLAS_COMPUTE_32I_PEDANTIC,
    }
}

singe_core::native_enum! {
    /// [`EmulationStrategy`] is used with [`Context::set_emulation_strategy`](crate::context::Context::set_emulation_strategy) to choose how to leverage floating point emulation algorithms.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum EmulationStrategy: sys::cublasEmulationStrategy_t {
        /// Default emulation strategy; equivalent to [`EmulationStrategy::Performant`] unless the `CUBLAS_EMULATION_STRATEGY` environment variable is set.
        Default = CUBLAS_EMULATION_STRATEGY_DEFAULT,
        /// Uses emulation whenever it provides a performance benefit.
        Performant = CUBLAS_EMULATION_STRATEGY_PERFORMANT,
        /// Uses emulation whenever possible.
        Eager = CUBLAS_EMULATION_STRATEGY_EAGER,
    }
}

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
        parser::to_writer(self, f)
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
