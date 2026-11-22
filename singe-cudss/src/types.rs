use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, Sub},
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{UnknownEnumValue, impl_enum_display, try_enum_from_raw};
use singe_cuda::{
    types::{Complex32, Complex64},
    view::{AnyBitPatternDeviceRepr, NoUninitDeviceRepr},
};
use singe_cuda_sys::library_types::cudaDataType_t;
use singe_cudss_sys as sys;

/// Extended-precision real value stored as a double-double pair.
///
/// This type maps to [`DataType::F64Mp2`] and has the layout cuDSS documents for double-double values: two `f64` values aligned to 16 bytes.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    PartialOrd,
    singe_cuda::DeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
)]
#[repr(C)]
#[repr(align(16))]
pub struct Fp64Mp2 {
    /// Leading component of the unevaluated sum.
    pub hi: f64,
    /// Trailing correction to the leading component.
    pub lo: f64,
}

const _: () = {
    assert!(size_of::<Fp64Mp2>() == 2 * size_of::<f64>());
    assert!(align_of::<Fp64Mp2>() == 2 * size_of::<f64>());
};

impl Fp64Mp2 {
    /// Stores the high and low components without renormalizing them.
    #[must_use]
    pub const fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// Represents `value` exactly with a zero trailing component.
    #[must_use]
    pub const fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Returns the absolute value after rounding `hi + lo` to `f64`.
    #[must_use]
    pub fn abs(self) -> f64 {
        (self.hi + self.lo).abs()
    }
}

impl Add for Fp64Mp2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let hi = self.hi + rhs.hi;
        let a_hi = hi - rhs.hi;
        let b_hi = hi - a_hi;
        let a_delta = self.hi - a_hi;
        let b_delta = rhs.hi - b_hi;
        let lo = self.lo + rhs.lo + a_delta + b_delta;
        let normalized_hi = hi + lo;
        let diff = normalized_hi - hi;
        Self::new(normalized_hi, lo - diff)
    }
}

impl Sub for Fp64Mp2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + Self::new(-rhs.hi, -rhs.lo)
    }
}

/// Data type for matrix values, offsets, and indices.
///
/// The discriminants are CUDA's `cudaDataType_t` values, except for [`Self::Unset`] and [`Self::F64Mp2`], which only cuDSS defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// Unset data type sentinel.
    ///
    /// This is not valid for user-facing API calls.
    Unset = sys::CUDSS_DATA_TYPE_UNSET.0,
    /// 32-bit real single-precision floating point (`f32`).
    F32 = cudaDataType_t::CUDA_R_32F.0,
    /// 64-bit real double-precision floating point (`f64`).
    F64 = cudaDataType_t::CUDA_R_64F.0,
    /// 32-bit complex single-precision floating point ([`Complex32`]).
    C32 = cudaDataType_t::CUDA_C_32F.0,
    /// 64-bit complex double-precision floating point ([`Complex64`]).
    C64 = cudaDataType_t::CUDA_C_64F.0,
    /// 128-bit real double-double floating point ([`Fp64Mp2`]).
    ///
    /// The value is stored as the unevaluated sum of two normalized `f64` values.
    /// cuDSS marks its support for this type as experimental.
    F64Mp2 = sys::CUDSS_R_64F_MP2.0,
    /// 32-bit signed integer (`int32_t`).
    /// Used for index and offset arrays.
    I32 = cudaDataType_t::CUDA_R_32I.0,
    /// 64-bit signed integer (`int64_t`).
    /// Used for index and offset arrays.
    I64 = cudaDataType_t::CUDA_R_64I.0,
}

impl From<DataType> for cudaDataType_t {
    fn from(value: DataType) -> Self {
        Self(value.into())
    }
}

impl TryFrom<cudaDataType_t> for DataType {
    type Error = UnknownEnumValue;

    fn try_from(value: cudaDataType_t) -> Result<Self, UnknownEnumValue> {
        try_enum_from_raw("DataType", value.0)
    }
}

impl_enum_display!(DataType, {
    Self::Unset => "CUDSS_DATA_TYPE_UNSET",
    Self::F32 => "CUDSS_R_32F",
    Self::F64 => "CUDSS_R_64F",
    Self::C32 => "CUDSS_C_32F",
    Self::C64 => "CUDSS_C_64F",
    Self::F64Mp2 => "CUDSS_R_64F_MP2",
    Self::I32 => "CUDSS_R_32I",
    Self::I64 => "CUDSS_R_64I",
});

/// Rust element types that map directly to a cuDSS value data type.
pub trait DataTypeLike: private::Sealed + AnyBitPatternDeviceRepr + NoUninitDeviceRepr {
    /// cuDSS data type used for this Rust element type.
    const DATA_TYPE: DataType;
}

mod private {
    use singe_cuda::types::{Complex32, Complex64};

    use crate::types::Fp64Mp2;

    #[allow(
        unnameable_types,
        reason = "sealed supertrait that callers must not name or implement"
    )]
    pub trait Sealed {}

    impl Sealed for f32 {}

    impl Sealed for f64 {}

    impl Sealed for Complex32 {}

    impl Sealed for Complex64 {}

    impl Sealed for Fp64Mp2 {}
}

impl DataTypeLike for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

impl DataTypeLike for f64 {
    const DATA_TYPE: DataType = DataType::F64;
}

impl DataTypeLike for Complex32 {
    const DATA_TYPE: DataType = DataType::C32;
}

impl DataTypeLike for Complex64 {
    const DATA_TYPE: DataType = DataType::C64;
}

impl DataTypeLike for Fp64Mp2 {
    const DATA_TYPE: DataType = DataType::F64Mp2;
}

singe_core::native_enum! {
    /// Solver configuration parameter names.
    ///
    /// These values identify settings passed to [`Config::set_raw`](crate::config::Config::set_raw) or [`Config::get_raw`](crate::config::Config::get_raw).
    /// Prefer the named `Config` methods when one exists.
    /// Raw access requires the exact native value representation documented for the selected parameter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ConfigParameter: sys::cudssConfigParam_t {
        /// Reordering algorithm used during analysis.
        ///
        /// Value type: [`ReorderingAlgorithm`].
        /// Default: [`ReorderingAlgorithm::Default`].
        /// [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`] require general matrices.
        /// [`ReorderingAlgorithm::Amd`] does not support 64-bit matrix indices.
        ReorderingAlgorithm = CUDSS_CONFIG_REORDERING_ALG,
        /// Factorization algorithm used during numerical factorization.
        ///
        /// Value type: [`FactorizationAlgorithm`].
        /// Default: [`FactorizationAlgorithm::Default`].
        FactorizationAlgorithm = CUDSS_CONFIG_FACTORIZATION_ALG,
        /// Solve algorithm used during the solve phase.
        ///
        /// Value type: [`SolveAlgorithm`].
        /// Default: [`SolveAlgorithm::Default`].
        SolveAlgorithm = CUDSS_CONFIG_SOLVE_ALG,
        /// Optional matching algorithm used before reordering.
        ///
        /// Matching computes a column permutation, and some algorithms also compute scaling vectors.
        /// It can improve accuracy for ill-conditioned or badly scaled matrices, but it adds analysis overhead and can change factor sparsity.
        ///
        /// For matrix types other than [`MatrixType::General`], matching preserves symmetry unless [`Self::ForceNonsymmetricMatching`] is set.
        ///
        /// Value type: [`MatchingAlgorithm`].
        /// Default: [`MatchingAlgorithm::Auto`].
        /// Matching is not supported with non-uniform batches, uniform-batch settings, [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`], Schur complement mode, or [`PivotEpsilonAlgorithm::Scaled`].
        /// [`MatchingAlgorithm::Auto`] disables matching for these combinations; another explicit algorithm makes analysis return [`Status::NotSupported`](crate::error::Status::NotSupported).
        MatchingAlgorithm = CUDSS_CONFIG_MATCHING_ALG,
        /// Matrix modifier used during the solve phase.
        ///
        /// Value type: [`SolveMode`].
        /// Default: [`SolveMode::Normal`].
        /// Transposed solves are not supported with Schur complement mode, [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`], or non-uniform batches.
        SolveMode = CUDSS_CONFIG_SOLVE_MODE,
        /// Maximum number of iterative refinement steps.
        ///
        /// Value type: `i32`.
        /// Default: `0`.
        IterativeRefinementSteps = CUDSS_CONFIG_IR_N_STEPS,
        /// Iterative refinement tolerance.
        ///
        /// When both [`Self::IterativeRefinementSteps`] and this value are greater than zero, cuDSS performs iterative refinement with convergence checking.
        ///
        /// In each iteration, the relative residual norm is computed as $||r||_2 / ||b||_2$ where $r = b - Ax$.
        ///
        /// The refinement terminates early if the residual norm is less than the specified tolerance.
        ///
        /// If this value is zero and [`Self::IterativeRefinementSteps`] is greater than zero, cuDSS performs exactly that many iterations without convergence checking.
        ///
        /// The actual iteration count can be retrieved with [`DataParameter::IterativeRefinementSteps`].
        /// If the tolerance is not met, cuDSS returns [`Status::IrFailed`](crate::error::Status::IrFailed).
        ///
        /// Value type: `f64`.
        /// Default: `0.0`, which disables tolerance checking.
        IterativeRefinementTolerance = CUDSS_CONFIG_IR_TOL,
        /// Pivoting strategy used during factorization.
        ///
        /// Value type: [`PivotType`].
        /// Default: [`PivotType::Auto`].
        /// With [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`], this controls global pivoting.
        /// With nested dissection, AMD, or natural order, it controls local 1x1 pivoting.
        PivotType = CUDSS_CONFIG_PIVOT_TYPE,
        /// Threshold used to decide whether a diagonal element is pivoted.
        ///
        /// Value type: `f64`.
        /// Default: `1.0`.
        /// This setting is supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        /// With [`PivotType::BunchKaufman`], an explicitly set value replaces the α constant that chooses between 1x1 and 2x2 pivots, by default `(1 + √17) / 8`.
        PivotThreshold = CUDSS_CONFIG_PIVOT_THRESHOLD,
        /// Pivoting epsilon.
        ///
        /// By default, this is the absolute value to test and replace small diagonal elements encountered during numerical factorization.
        ///
        /// If [`Self::PivotEpsilonAlgorithm`] is [`PivotEpsilonAlgorithm::Scaled`], cuDSS scales this value before use.
        ///
        /// Value type: `f64`.
        /// Defaults are `1e-5` for single precision and `1e-13` for double precision.
        PivotEpsilon = CUDSS_CONFIG_PIVOT_EPSILON,
        /// Upper limit on the number of nonzero entries in LU factors.
        ///
        /// Value type: `i64`.
        /// Default: `-1`, which lets cuDSS use its internal estimate.
        /// This is only relevant for general matrices with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        MaxLuNnz = CUDSS_CONFIG_MAX_LU_NNZ,
        /// Enables hybrid host/device memory mode.
        ///
        /// Value type: `i32` boolean.
        /// Default: `0` (disabled).
        /// Enable this before [`Phase::ANALYSIS`]; otherwise analysis must be rerun.
        /// This mode keeps part of the internal factor data on the host while factorization and solve still run on the GPU.
        ///
        /// Not supported with [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`], or [`FactorizationAlgorithm::Multiblock`].
        HybridMemoryMode = CUDSS_CONFIG_HYBRID_MEMORY_MODE,
        /// User-defined device memory limit (number of bytes) for the hybrid memory mode.
        ///
        /// This setting only affects execution when [`Self::HybridMemoryMode`] is enabled.
        /// In multi-GPU mode it must be set separately for each selected device.
        ///
        /// Value type: `i64`.
        /// Default: `-1`, which uses cuDSS' internal heuristic.
        HybridDeviceMemoryLimit = CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
        /// Enables `cudaHostRegister()` use in hybrid memory mode.
        ///
        /// Value type: `i32` boolean.
        /// Default: `1`, which lets cuDSS register host memory when the device supports it.
        UseCudaRegisterMemory = CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY,
        /// Number of threads to be used by cuDSS in [MT mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-threaded-mt-mode).
        /// This setting only affects execution when multi-threaded mode is enabled.
        ///
        /// Value type: `i32`.
        /// Default: `-1`, which uses the maximum thread count reported by cuDSS.
        HostThreads = CUDSS_CONFIG_HOST_NTHREADS,
        /// Enables hybrid host/device execute mode.
        ///
        /// Value type: `i32` boolean.
        /// Default: `0` (disabled).
        /// Enable this before [`Phase::ANALYSIS`].
        /// When enabled, some work may run on the CPU and input matrix, right-hand side, and solution buffers may be host pointers.
        ///
        /// Not supported with [`Self::HybridMemoryMode`], [`Self::DeterministicMode`], batched matrices, or non-trivial leading dimensions for the right-hand side or solution.
        HybridExecuteMode = CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
        /// Algorithm used to compute replacement values for small pivots.
        ///
        /// Value type: [`PivotEpsilonAlgorithm`].
        /// Default: [`PivotEpsilonAlgorithm::Default`].
        /// [`PivotEpsilonAlgorithm::Scaled`] is not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        PivotEpsilonAlgorithm = CUDSS_CONFIG_PIVOT_EPSILON_ALG,
        /// Minimum nested-dissection level count.
        ///
        /// Value type: `i32`.
        /// Default: `10`.
        /// This only affects [`ReorderingAlgorithm::NestedDissection`] and [`ReorderingAlgorithm::Default`].
        /// In MGMN mode, cuDSS may increase the value to satisfy process-count requirements.
        NestedDissectionLevels = CUDSS_CONFIG_ND_NLEVELS,
        /// Number of systems in a uniform batch.
        ///
        /// A uniform batch has the same sparsity pattern for every matrix, but may use different values.
        /// Set this before [`Phase::ANALYSIS`].
        ///
        /// Value type: `i32`.
        /// Default: `1`.
        /// Not supported with hybrid memory mode, hybrid execute mode, MGMN mode, [`ReorderingAlgorithm::BtfColamd`], or [`ReorderingAlgorithm::Colamd`].
        UniformBatchSize = CUDSS_CONFIG_UBATCH_SIZE,
        /// Uniform-batch system index to process.
        ///
        /// Value type: `i32`.
        /// Default: `-1`, which processes all systems at once.
        /// Non-negative values select a single 0-based system index and must be less than [`Self::UniformBatchSize`].
        UniformBatchIndex = CUDSS_CONFIG_UBATCH_INDEX,
        /// Enable or disable superpanel optimization.
        ///
        /// Value type: `i32` boolean.
        /// Default: `1` (enabled).
        /// Set this before [`Phase::ANALYSIS`]; otherwise analysis must be rerun.
        UseSuperpanels = CUDSS_CONFIG_USE_SUPERPANELS,
        /// Number of devices used by a multi-GPU context.
        ///
        /// Value type: `i32`.
        /// Default: `1`.
        /// cuDSS currently supports at most 16 devices.
        DeviceCount = CUDSS_CONFIG_DEVICE_COUNT,
        /// Device indices used by a multi-GPU context.
        ///
        /// Value type: `&[i32]`.
        /// By default, cuDSS uses devices `0..DeviceCount`.
        /// The current CUDA device must match the first selected device, or device `0` when default indices are used.
        DeviceIndices = CUDSS_CONFIG_DEVICE_INDICES,
        /// Enables Schur complement mode.
        ///
        /// Value type: `i32` boolean.
        /// Default: `0` (disabled).
        /// Not supported with [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`], multi-GPU or MGMN mode, [`FactorizationAlgorithm::Multiblock`], batches, or matching.
        SchurMode = CUDSS_CONFIG_SCHUR_MODE,
        /// Enables deterministic mode.
        ///
        /// When run on GPUs with the same architecture and SM count, deterministic mode produces bitwise-identical results for bitwise-identical input data and solver settings.
        ///
        /// Deterministic mode may use slower kernels.
        /// The setting is part of the analyzed state; after a change, rerun [`Phase::ANALYSIS`], or later phases return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        ///
        /// Value type: `i32` boolean.
        /// Default: `0` (disabled).
        /// Supported only for a single GPU, a single right-hand side, no hybrid modes, no iterative refinement, and no [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        DeterministicMode = CUDSS_CONFIG_DETERMINISTIC_MODE,
        /// Controls the balance of the nested dissection partition tree.
        /// The value represents the allowed percentage of imbalance between the two children at each level of the partition tree.
        /// Possible values are from 1 to 100.
        ///
        /// Smaller values produce a more balanced partition tree.
        ///
        /// Value type: `i32`.
        /// Default: `20`.
        /// For multi-GPU and MGMN modes, NVIDIA recommends smaller values such as `2..=5`.
        NestedDissectionUnbalanceFactor = CUDSS_CONFIG_ND_UBFACTOR,
        /// Mantissa bits that floating-point emulation must preserve during factorization.
        ///
        /// A positive value makes factorization use emulated kernels with at least that many mantissa bits; values from `53` to `55` recover close to full double precision.
        /// Analysis, solve, and iterative refinement keep the regular kernels.
        /// Emulation applies only to [`DataType::F64`] and [`DataType::C64`], and is ignored on GPUs with compute capability 8.0 or earlier and 9.0.
        ///
        /// Value type: `i32`.
        /// Default: `0`, which disables emulation.
        /// Not supported with [`PivotType::BunchKaufman`] when superpanels are formed.
        EmulationMantissaBitCount = CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT,
        /// Forces nonsymmetric matching for matrix types other than [`MatrixType::General`].
        ///
        /// This has no effect when matching is disabled.
        ///
        /// Value type: `i32` boolean.
        /// Default: `0`, which preserves symmetry.
        /// Not supported together with [`PivotType::BunchKaufman`] and an explicit matching algorithm other than [`MatchingAlgorithm::Auto`].
        ForceNonsymmetricMatching = CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING,
        /// Stores double-precision factors in single precision after factorization.
        ///
        /// Set this before [`Phase::FACTORIZATION`] and keep it unchanged through every solve that reuses those factors.
        /// Single-precision right-hand side and solution matrices run the whole solve in single precision; double-precision ones run a mixed-precision solve.
        ///
        /// Value type: [`DowncastFactorsMode`].
        /// Default: [`DowncastFactorsMode::Disabled`].
        /// Supported for [`DataType::F64`] and [`DataType::C64`] matrices.
        /// Not supported with [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`], [`Self::DeterministicMode`], [`Self::SchurMode`], or transposed solves in single precision.
        DowncastFactors = CUDSS_CONFIG_DOWNCAST_FACTORS,
        /// Memory optimizations cuDSS may apply.
        ///
        /// The setting applies to the factorization or refactorization phase it is set before.
        /// [`DataParameter::MemoryEstimates`] reports the requirement before any optimization.
        ///
        /// Value type: [`MemoryOptimization`].
        /// Default: [`MemoryOptimization::empty`].
        MemoryOptimizationSetting = CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING,
    }
}

singe_core::native_enum! {
    /// Solver data parameter names.
    ///
    /// These values identify data slots passed to [`SolverState::set_raw`](crate::solver_state::SolverState::set_raw), [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw), or the slice/device variants on [`SolverState`](crate::solver_state::SolverState).
    /// Prefer the named `SolverState` methods when one exists.
    /// Raw access requires the native value representation and memory space documented for the selected parameter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DataParameter: sys::cudssDataParam_t {
        /// Host-readable execution information code.
        ///
        /// This reports device-side asynchronous errors, or host-side errors in hybrid execute mode.
        /// A value of `0` means success.
        /// For positive-definite matrix types, a nonzero value may be the 1-based index of the first non-positive minor in the reordered matrix.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `i32`.
        Info = CUDSS_DATA_INFO,
        /// Number of nonzero entries in LU factors.
        ///
        /// For non-uniform batches, this is accumulated across the batch.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `i64`.
        LuNnz = CUDSS_DATA_LU_NNZ,
        /// Number of pivots encountered during factorization.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: the system matrix index type.
        NumberOfPivots = CUDSS_DATA_NPIVOTS,
        /// Positive and negative indices of inertia for the system matrix `A` (two integer values).
        /// Valid only for symmetric/Hermitian non positive-definite matrix types.
        ///
        /// For non-uniform batches, cuDSS returns the accumulated counts across the batch.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice).
        /// Memory: host.
        /// Value type: two values of the system matrix index type.
        Inertia = CUDSS_DATA_INERTIA,
        /// Row permutation `P` after reordering, such that `A[P,Q]` is factorized.
        ///
        /// Not supported for non-uniform batches.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        PermutationReorderRow = CUDSS_DATA_PERM_REORDER_ROW,
        /// Column permutation `Q` after reordering, such that `A[P,Q]` is factorized.
        ///
        /// Not supported for non-uniform batches.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        PermutationReorderColumn = CUDSS_DATA_PERM_REORDER_COL,
        /// Final row permutation `P`, including reordering and pivoting.
        ///
        /// Not supported for non-uniform batches.
        /// Only supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        PermutationRow = CUDSS_DATA_PERM_ROW,
        /// Final column permutation `Q`, including reordering and pivoting.
        ///
        /// Not supported for non-uniform batches.
        /// Only supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        PermutationColumn = CUDSS_DATA_PERM_COL,
        /// Diagonal of the factorized matrix in original matrix order.
        ///
        /// The values include matching, reordering, and pivoting effects.
        /// For non-uniform batches, cuDSS returns accumulated values across the batch.
        /// Only supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix value type.
        Diagonal = CUDSS_DATA_DIAG,
        /// User permutation used instead of running a reordering algorithm.
        ///
        /// Disable the user permutation by setting a null pointer with zero size.
        /// The provided buffer must contain `n` indices, where `n` is the system matrix row or column count.
        /// The index type must match the system matrix.
        /// The values must represent a valid permutation vector for the matrix index base.
        ///
        /// cuDSS copies the data into an internal buffer.
        ///
        /// Not supported for non-uniform batches or with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        ///
        /// Set with [`SolverState::set_raw_slice`](crate::solver_state::SolverState::set_raw_slice).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        UserPermutation = CUDSS_DATA_USER_PERM,
        /// Minimum device memory, in bytes, required by hybrid memory mode.
        ///
        /// cuDSS refuses this query until analysis has completed.
        /// In multi-GPU mode, the value is reported for the current CUDA device.
        /// In MGMN mode, the query is collective and every rank must make it.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `i64`.
        HybridDeviceMemoryMin = CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
        /// Device communicator for [MGMN mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-gpu-multi-node-mgmn-mode).
        /// The concrete communicator type must match the device backend selected by [`Context::set_communication_layer`](crate::context::Context::set_communication_layer).
        /// This communicator is used for communication involving device memory buffers.
        ///
        /// Set with [`SolverState::set_raw`](crate::solver_state::SolverState::set_raw).
        /// Memory: host.
        /// Value type: `*mut c_void`.
        CommunicationDevice = CUDSS_DATA_COMM_DEVICE,
        /// Host communicator for [MGMN mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-gpu-multi-node-mgmn-mode).
        /// The concrete communicator type must match the host backend selected by [`Context::set_communication_layer`](crate::context::Context::set_communication_layer).
        /// This communicator is used for communication involving host memory buffers.
        ///
        /// Set with [`SolverState::set_raw`](crate::solver_state::SolverState::set_raw).
        /// Memory: host.
        /// Value type: `*mut c_void`.
        CommunicationHost = CUDSS_DATA_COMM_HOST,
        /// Host and device memory estimates, in bytes, for the analyzed solve.
        ///
        /// The query is available once analysis knows the fill-in of the factors, including after a failed analysis, and returns [`Status::NotSupported`](crate::error::Status::NotSupported) before that.
        /// Estimates depend on the configuration, input matrix, and right-hand side count used during analysis; changing those inputs can make the estimates stale.
        ///
        /// The returned array uses these positions:
        ///
        /// * `0`: permanent device memory
        /// * `1`: peak device memory
        /// * `2`: permanent host memory
        /// * `3`: peak host memory
        /// * `4`: minimum device memory for hybrid memory mode
        /// * `5`: maximum host memory for hybrid memory mode
        /// * `6`: the completeness of the estimate, a [`MemoryEstimateState`]
        /// * `7..=15`: reserved
        ///
        /// Positions `4` and `5` are `0` until the estimate is [`MemoryEstimateState::Final`].
        /// In multi-GPU mode, the estimate is reported for the current CUDA device.
        /// In MGMN mode, the query is collective and every rank must make it.
        ///
        /// Read with [`SolverState::memory_estimates`](crate::solver_state::SolverState::memory_estimates).
        /// Memory: host.
        /// Value type: `[i64; 16]`.
        MemoryEstimates = CUDSS_DATA_MEMORY_ESTIMATES,
        /// Matching column permutation `Q`, such that `A[:,Q]` is reordered and then factorized.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        PermutationMatching = CUDSS_DATA_PERM_MATCHING,
        /// Row scaling for the factorized matrix, in original row order.
        ///
        /// Only supported when matching is enabled and the chosen matching algorithm computes scaling.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: real `f32` or `f64`, matching the absolute value type of the system matrix.
        ScaleRow = CUDSS_DATA_SCALE_ROW,
        /// Column scaling for the factorized matrix, in original column order.
        ///
        /// Only supported when matching is enabled and the chosen matching algorithm computes scaling.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: real `f32` or `f64`, matching the absolute value type of the system matrix.
        ScaleColumn = CUDSS_DATA_SCALE_COL,
        /// Number of superpanels in the matrix.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: the system matrix index type.
        SuperpanelCount = CUDSS_DATA_NSUPERPANELS,
        /// User-provided Schur complement indices.
        ///
        /// The provided buffer must contain `n` integer values, where `n` is the system matrix row or column count.
        /// Values equal to `1` mark rows/columns that belong to the Schur complement; `0` marks the rest.
        /// cuDSS copies the data into an internal buffer.
        ///
        /// Set with [`SolverState::set_raw_slice`](crate::solver_state::SolverState::set_raw_slice).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        UserSchurIndices = CUDSS_DATA_USER_SCHUR_INDICES,
        /// Schur complement shape as `(rows, columns, sparse_nnz)`.
        ///
        /// Query this to allocate a dense or sparse Schur complement [`Matrix`](crate::matrix::Matrix), then pass that matrix with [`Self::SchurMatrix`] for cuDSS to fill.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `[i64; 3]`.
        SchurShape = CUDSS_DATA_SCHUR_SHAPE,
        /// Schur complement matrix to be filled by cuDSS.
        ///
        /// The value, offset, and index data types must match the corresponding system matrix types.
        /// The Schur complement matrix cannot be distributed or batched.
        /// Dense Schur complements must be column-major.
        /// Sparse Schur complements must use [`MatrixType::General`] and [`MatrixViewType::Full`].
        ///
        /// Set with [`SolverState::set_raw`](crate::solver_state::SolverState::set_raw).
        /// Memory: host, with matrix buffers on host or device.
        /// Value type: [`Matrix`](crate::matrix::Matrix) handle.
        SchurMatrix = CUDSS_DATA_SCHUR_MATRIX,
        /// User-provided nested-dissection partition tree.
        ///
        /// This is used with [`Self::UserPermutation`] to skip reordering.
        /// Disable it by setting a null pointer with zero size.
        ///
        /// Prefer reusing a tree returned by [`Self::NestedDissectionPartitionTree`] rather than constructing one manually.
        ///
        /// Set with [`SolverState::set_raw_slice`](crate::solver_state::SolverState::set_raw_slice).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        UserNestedDissectionPartitionTree = CUDSS_DATA_USER_ND_PARTITION_TREE,
        /// Nested-dissection partition tree computed during reordering.
        ///
        /// The array has `2^nlevels - 1` elements, where `nlevels` is [`ConfigParameter::NestedDissectionLevels`].
        /// It can be reused with [`Self::UserPermutation`] and [`Self::UserNestedDissectionPartitionTree`] for another matrix with the same sparsity structure.
        ///
        /// Read with [`SolverState::get_raw_slice`](crate::solver_state::SolverState::get_raw_slice) or [`SolverState::get_device`](crate::solver_state::SolverState::get_device).
        /// Memory: host or device.
        /// Value type: the system matrix index type.
        NestedDissectionPartitionTree = CUDSS_DATA_ND_PARTITION_TREE,
        /// User-provided host interrupt pointer.
        ///
        /// Setting a non-null pointer enables host-side interruption checks in [`Context::execute_async`](crate::context::Context::execute_async).
        /// If the pointed-to value is nonzero, cuDSS stops early and returns [`Status::ExecutionFailed`](crate::error::Status::ExecutionFailed).
        /// To resume after an interruption, set the value at that location back to `0` and call again.
        /// Disable the feature by setting a null pointer.
        ///
        /// Since most device kernels inside cuDSS are launched asynchronously, execution on the GPU may continue some time after the host execution has been interrupted.
        ///
        /// Set with [`SolverState::set_raw`](crate::solver_state::SolverState::set_raw).
        /// Memory: host.
        /// Default: null pointer.
        /// Value type: pointer to a 4-byte integer.
        ///
        /// When setting the flag from another thread, use an atomic store with release semantics.
        /// A plain write to a non-atomic object from another thread is a data race and undefined behavior.
        ///
        /// Not supported in multi-GPU or MGMN mode, or with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        UserHostInterrupt = CUDSS_DATA_USER_HOST_INTERRUPT,
        /// Actual number of iterative refinement steps performed during solve.
        ///
        /// The value can be less than [`ConfigParameter::IterativeRefinementSteps`] if the configured tolerance is reached.
        /// The value is 0 if iterative refinement is disabled or if the initial solution already satisfies the tolerance.
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `i32`.
        IterativeRefinementSteps = CUDSS_DATA_IR_N_STEPS,
        /// Systems of a uniform batch that factorization and solve process.
        ///
        /// The mask holds one integer per system; a nonzero element selects the system at that index.
        /// It is an alternative to [`ConfigParameter::UniformBatchIndex`], and the two are mutually exclusive.
        /// Set it before [`Phase::ANALYSIS`] or the first factorization; a mask introduced after a factorization that ran without one is not supported.
        /// A solve leaves the solution entries of unselected systems unchanged, and fails with [`Status::NotSupported`](crate::error::Status::NotSupported) if it selects a system no factorization since the last analysis covered.
        /// Clear it by setting a null pointer with zero size.
        ///
        /// Set with [`SolverState::set_uniform_batch_mask`](crate::solver_state::SolverState::set_uniform_batch_mask).
        /// Memory: host.
        /// Value type: [`ConfigParameter::UniformBatchSize`] values of `i32`.
        /// Default: no mask, which selects all systems.
        /// Not supported with hybrid modes, multi-GPU or MGMN mode, [`ReorderingAlgorithm::BtfColamd`], or [`ReorderingAlgorithm::Colamd`].
        UniformBatchMask = CUDSS_DATA_UBATCH_MASK,
        /// Floating-point operation count for the factorization phase.
        ///
        /// Available after [`Phase::ANALYSIS`].
        ///
        /// Read with [`SolverState::get_raw`](crate::solver_state::SolverState::get_raw).
        /// Memory: host.
        /// Value type: `f64`.
        Flops = CUDSS_DATA_FLOPS,
    }
}

bitflags::bitflags! {
    /// Solver phases passed to [`Context::execute_async`](crate::context::Context::execute_async).
    ///
    /// Phases can be combined with `|`, for example `Phase::FACTORIZATION | Phase::SOLVE`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Phase: u32 {
        /// Reordering.
        const REORDERING = sys::cudssPhase_t::CUDSS_PHASE_REORDERING.0;
        /// Symbolic factorization.
        ///
        /// Do not call symbolic factorization twice without a reordering phase between the calls.
        const SYMBOLIC_FACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_SYMBOLIC_FACTORIZATION.0;
        /// Reordering and symbolic factorization combined.
        const ANALYSIS = sys::cudssPhase_t::CUDSS_PHASE_ANALYSIS.0;
        /// Numerical factorization.
        const FACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_FACTORIZATION.0;
        /// Numerical re-factorization.
        ///
        /// This is distinct from [`Self::FACTORIZATION`] only with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const REFACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_REFACTORIZATION.0;
        /// Applies the reordering permutation to the right-hand side before forward substitution.
        ///
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_FORWARD_PERMUTATION = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_FWD_PERM.0;
        /// Forward substitution sub-step of the solve phase.
        ///
        /// This includes local permutations from partial pivoting.
        /// Use [`DataParameter::PermutationReorderRow`] and [`DataParameter::PermutationRow`] to account for that effect when needed.
        ///
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_FORWARD = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_FWD.0;
        /// Diagonal solve sub-step for symmetric or Hermitian matrix types.
        ///
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_DIAGONAL = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_DIAG.0;
        /// Backward substitution sub-step of the solve phase.
        ///
        /// This includes local permutations from partial pivoting.
        /// Use [`DataParameter::PermutationReorderRow`] and [`DataParameter::PermutationRow`] to account for that effect when needed.
        ///
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_BACKWARD = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_BWD.0;
        /// Applies the inverse reordering permutation after backward substitution.
        ///
        /// If matching with scaling is enabled, this phase also applies the inverse matching permutation and inverse scaling.
        ///
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_BACKWARD_PERMUTATION = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_BWD_PERM.0;
        /// Iterative refinement.
        ///
        /// Controlled by [`ConfigParameter::IterativeRefinementSteps`] and [`ConfigParameter::IterativeRefinementTolerance`].
        /// Solve sub-phases are not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_REFINEMENT = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_REFINEMENT.0;
        /// Full solve phase.
        ///
        /// This combines forward permutation, forward substitution, diagonal solve, backward substitution, backward permutation, and optional iterative refinement.
        ///
        /// Combining solve sub-phases is preferred over calling them separately.
        const SOLVE = sys::cudssPhase_t::CUDSS_PHASE_SOLVE.0;
    }
}

impl Display for Phase {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// Matrix type for sparse system matrices.
    ///
    /// The matrix type describes mathematical properties of the stored matrix and determines the factorization cuDSS computes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MatrixType: sys::cudssMatrixType_t {
        /// General matrix.
        ///
        /// cuDSS computes an `LDU` factorization with optional local or global pivoting.
        General = CUDSS_MTYPE_GENERAL,
        /// Real symmetric matrix.
        /// `LDL^T` factorization will be computed with optional local pivoting.
        Symmetric = CUDSS_MTYPE_SYMMETRIC,
        /// Complex Hermitian matrix.
        /// `LDL^H` factorization will be computed with optional local pivoting.
        Hermitian = CUDSS_MTYPE_HERMITIAN,
        /// Symmetric positive-definite matrix.
        ///
        /// cuDSS computes a Cholesky factorization with optional local pivoting.
        /// If cuDSS encounters a non-positive minor, [`DataParameter::Info`] can report its 1-based reordered index.
        Spd = CUDSS_MTYPE_SPD,
        /// Hermitian positive-definite matrix.
        ///
        /// cuDSS computes a complex Cholesky factorization with optional local pivoting.
        /// If cuDSS encounters a non-positive minor, [`DataParameter::Info`] can report its 1-based reordered index.
        Hpd = CUDSS_MTYPE_HPD,
    }
}

singe_core::native_enum! {
    /// Matrix view for sparse matrices.
    ///
    /// This controls which triangular part of a symmetric or Hermitian sparse matrix cuDSS reads.
    /// The view is ignored for [`MatrixType::General`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MatrixViewType: sys::cudssMatrixViewType_t {
        /// Full matrix \[default\].
        Full = CUDSS_MVIEW_FULL,
        /// Lower-triangular matrix (including the diagonal).
        /// All values above the main diagonal will be ignored.
        Lower = CUDSS_MVIEW_LOWER,
        /// Upper-triangular matrix (including the diagonal).
        /// All values below the main diagonal will be ignored.
        Upper = CUDSS_MVIEW_UPPER,
    }
}

singe_core::native_enum! {
    /// Indexing base for sparse matrix indices.
    ///
    /// cuDSS uses the sparse input matrix index base for index data returned by queries such as [`DataParameter::PermutationReorderRow`] and [`DataParameter::PermutationReorderColumn`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum IndexBase: sys::cudssIndexBase_t {
        /// Zero-based indexing \[default\].
        Zero = CUDSS_BASE_ZERO,
        /// One-based indexing.
        One = CUDSS_BASE_ONE,
    }
}

singe_core::native_enum! {
    /// Dense matrix layout.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Layout: sys::cudssLayout_t {
        /// Column-major layout \[default\].
        ColumnMajor = CUDSS_LAYOUT_COL_MAJOR,
        /// Row-major layout.
        /// Currently not supported.
        RowMajor = CUDSS_LAYOUT_ROW_MAJOR,
    }
}

singe_core::native_enum! {
    /// Reordering algorithm used during analysis.
    ///
    /// Used with [`ConfigParameter::ReorderingAlgorithm`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ReorderingAlgorithm: sys::cudssReorderingAlg_t {
        /// Let cuDSS choose the reordering algorithm.
        ///
        /// Currently equivalent to [`Self::NestedDissection`].
        Default = CUDSS_REORDERING_ALG_DEFAULT,
        /// Block triangular form (BTF) combined with COLAMD.
        /// Supports global pivoting.
        BtfColamd = CUDSS_REORDERING_ALG_BTF_COLAMD,
        /// COLAMD with trivial block structure.
        /// Supports global pivoting.
        Colamd = CUDSS_REORDERING_ALG_COLAMD,
        /// Approximate minimum degree (AMD) reordering.
        Amd = CUDSS_REORDERING_ALG_AMD,
        /// Nested dissection algorithm based on METIS.
        NestedDissection = CUDSS_REORDERING_ALG_NESTED_DISSECTION,
        /// Uses natural (identity) order for the internal ordering when no user permutation is supplied.
        None = CUDSS_REORDERING_ALG_NONE,
    }
}

singe_core::native_enum! {
    /// Factorization algorithm for numerical factorization.
    ///
    /// Used with [`ConfigParameter::FactorizationAlgorithm`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FactorizationAlgorithm: sys::cudssFactorizationAlg_t {
        /// Let cuDSS choose the factorization algorithm.
        ///
        /// Currently equivalent to [`Self::General`].
        Default = CUDSS_FACTORIZATION_ALG_DEFAULT,
        /// Multiblock factorization algorithm.
        ///
        /// This is supported only for [`DataType::F64`] and is intended for extremely sparse matrices.
        Multiblock = CUDSS_FACTORIZATION_ALG_MULTIBLOCK,
        /// General factorization algorithm.
        General = CUDSS_FACTORIZATION_ALG_GENERAL,
    }
}

singe_core::native_enum! {
    /// Pivot epsilon algorithm used when replacing small pivots.
    ///
    /// Used with [`ConfigParameter::PivotEpsilonAlgorithm`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PivotEpsilonAlgorithm: sys::cudssPivotEpsilonAlg_t {
        /// Let cuDSS choose the pivot epsilon algorithm.
        ///
        /// Currently equivalent to [`Self::Static`].
        Default = CUDSS_PIVOT_EPSILON_ALG_DEFAULT,
        /// Replaces pivots below the threshold with a signed epsilon scaled by the maximum element in the corresponding row and column of the original matrix.
        ///
        /// Not supported with [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        Scaled = CUDSS_PIVOT_EPSILON_ALG_SCALED,
        /// Pivots below the threshold are replaced by the appropriately signed epsilon (no scaling).
        /// Recommended for most cases.
        Static = CUDSS_PIVOT_EPSILON_ALG_STATIC,
    }
}

singe_core::native_enum! {
    /// Solve algorithm.
    ///
    /// Used with [`ConfigParameter::SolveAlgorithm`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SolveAlgorithm: sys::cudssSolveAlg_t {
        /// Let the library choose the solve algorithm (recommended).
        Default = CUDSS_SOLVE_ALG_DEFAULT,
        /// Triangular solve only: forward/back substitution using the existing L/U factors without modifying them.
        General = CUDSS_SOLVE_ALG_GENERAL,
    }
}

singe_core::native_enum! {
    /// Matching algorithm for optional column permutation before reordering.
    ///
    /// Used with [`ConfigParameter::MatchingAlgorithm`].
    /// Except for [`Self::None`] and [`Self::Auto`], the algorithms are based on MC64 from HSL.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MatchingAlgorithm: sys::cudssMatchingAlg_t {
        /// Matching is disabled.
        None = CUDSS_MATCHING_ALG_NONE,
        /// Column permutation that maximizes the number of diagonal entries.
        ///
        /// MC64 job 1.
        /// This does not use matrix values and is not generally recommended.
        MaxDiagonalCount = CUDSS_MATCHING_ALG_MAX_DIAG_COUNT,
        /// Column permutation that maximizes the smallest value on the diagonal.
        ///
        /// MC64 job 2.
        MaxMinDiagonal = CUDSS_MATCHING_ALG_MAX_MIN_DIAG,
        /// Alternate algorithm to maximize the smallest value on the diagonal.
        ///
        /// MC64 job 3.
        /// May differ in performance from [`Self::MaxMinDiagonal`].
        MaxMinDiagonalAlternate = CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT,
        /// Column permutation that maximizes the sum of diagonal entries.
        ///
        /// MC64 job 4.
        MaxDiagSum = CUDSS_MATCHING_ALG_MAX_DIAG_SUM,
        /// Column permutation that maximizes the product of diagonal entries.
        ///
        /// MC64 job 5.
        /// Also computes row and column scaling so nonzero diagonal entries are `1` in absolute value and off-diagonal entries are at most `1`.
        /// This is usually the most useful matching option for accuracy and requires matrix values during analysis.
        MaxDiagProduct = CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT,
        /// Let cuDSS choose the matching algorithm.
        ///
        /// Resolves to [`Self::None`] for [`MatrixType::Spd`] and [`MatrixType::Hpd`], and to [`Self::MaxDiagProduct`] for the other matrix types.
        /// Falls back to [`Self::None`] when analysis has no matrix values or the configuration does not support matching.
        Auto = CUDSS_MATCHING_ALG_AUTO,
    }
}

singe_core::native_enum! {
    /// Pivoting strategy for numerical factorization.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PivotType: sys::cudssPivotType_t {
        /// Let cuDSS choose the pivoting strategy.
        ///
        /// cuDSS currently chooses based on the reordering algorithm and matrix type:
        ///
        /// * [`ReorderingAlgorithm::NestedDissection`], [`ReorderingAlgorithm::Amd`], and [`ReorderingAlgorithm::None`]: [`Self::BunchKaufman`] for symmetric or Hermitian indefinite matrices where supported, otherwise [`Self::Diagonal`]; [`Self::LocalBlock`] for general matrices.
        /// * [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`]: [`Self::GlobalColumn`].
        Auto = CUDSS_PIVOT_AUTO,
        /// No pivoting.
        ///
        /// Used only with [`ReorderingAlgorithm::NestedDissection`] and [`ReorderingAlgorithm::Amd`].
        None = CUDSS_PIVOT_NONE,
        /// Global column pivoting.
        ///
        /// Used only with [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`].
        /// Searches for pivot elements within entire columns of the sub-matrix.
        GlobalColumn = CUDSS_PIVOT_GLOBAL_COL,
        /// Global row pivoting.
        ///
        /// Used only with [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`].
        /// Searches for pivot elements within entire rows of the sub-matrix.
        GlobalRow = CUDSS_PIVOT_GLOBAL_ROW,
        /// Diagonal pivoting.
        ///
        /// Used only with [`ReorderingAlgorithm::NestedDissection`] and [`ReorderingAlgorithm::Amd`].
        /// For symmetric/Hermitian indefinite matrices, searches for pivot elements only within the diagonal of the supernode.
        Diagonal = CUDSS_PIVOT_DIAGONAL,
        /// Complete block pivoting.
        ///
        /// Used only with [`ReorderingAlgorithm::NestedDissection`] and [`ReorderingAlgorithm::Amd`].
        /// For general matrices, searches for pivot elements within the entire diagonal block of the supernode.
        LocalBlock = CUDSS_PIVOT_LOCAL_BLOCK,
        /// Bunch-Kaufman pivoting with 1x1 and 2x2 pivots.
        ///
        /// For symmetric and Hermitian indefinite matrices.
        /// [`ConfigParameter::PivotThreshold`] overrides its α constant, which must be in `(0, 1]`.
        /// Solves with Bunch-Kaufman pivoting cannot be stream-captured.
        BunchKaufman = CUDSS_PIVOT_BUNCH_KAUFMAN,
    }
}

singe_core::native_enum! {
    /// Matrix modifier used during the solve phase.
    ///
    /// Used with [`ConfigParameter::SolveMode`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SolveMode: sys::cudssSolveMode_t {
        /// Solves `A x = b`.
        Normal = CUDSS_SOLVE_MODE_NORMAL,
        /// Solves `A^T x = b`.
        ///
        /// This is the normal solve for [`MatrixType::Symmetric`] and [`MatrixType::Spd`].
        /// Not supported for [`MatrixType::Hermitian`] and [`MatrixType::Hpd`].
        Transpose = CUDSS_SOLVE_MODE_TRANSPOSE,
        /// Solves `A^H x = b`.
        ///
        /// Supported for complex [`MatrixType::General`] matrices, and the normal solve for [`MatrixType::Hermitian`] and [`MatrixType::Hpd`].
        /// Not supported for real data types.
        ConjugateTranspose = CUDSS_SOLVE_MODE_CONJUGATE_TRANSPOSE,
    }
}

singe_core::native_enum! {
    /// Precision cuDSS keeps the numeric factors in after factorization.
    ///
    /// Used with [`ConfigParameter::DowncastFactors`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DowncastFactorsMode: sys::cudssDowncastFactorsMode_t {
        /// Keeps the factors in full precision.
        Disabled = CUDSS_DOWNCAST_FACTORS_MODE_DISABLED,
        /// Stores the factors in single precision, keeping the diagonal block of every supernode in full precision.
        KeepBlockDiagonal = CUDSS_DOWNCAST_FACTORS_MODE_KEEP_BLOCK_DIAGONAL,
        /// Stores the factors in single precision, keeping only the diagonal, one value per column, in full precision.
        ///
        /// With [`PivotType::BunchKaufman`], the off-diagonal of each 2x2 diagonal block also stays in full precision.
        KeepStrictDiagonal = CUDSS_DOWNCAST_FACTORS_MODE_KEEP_STRICT_DIAGONAL,
    }
}

bitflags::bitflags! {
    /// Memory optimizations cuDSS may apply, each trading some capability for a lower memory footprint.
    ///
    /// Used with [`ConfigParameter::MemoryOptimizationSetting`].
    /// The empty set requests none of them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryOptimization: u32 {
        /// Releases, at the end of factorization, the state only a further factorization reads.
        ///
        /// Solves may be slower.
        /// A later [`Phase::FACTORIZATION`] or [`Phase::REFACTORIZATION`] on the same solver state returns [`Status::NotSupported`](crate::error::Status::NotSupported) until [`Phase::ANALYSIS`] runs again.
        /// Not supported with a partial uniform-batch factorization, [`ReorderingAlgorithm::BtfColamd`], or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_ONLY = sys::cudssMemoryOptimizationFlags_t::CUDSS_MEMORY_OPTIMIZATION_SOLVE_ONLY.0;
    }
}

impl Display for MemoryOptimization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// Completeness of a memory estimate, by how far analysis had progressed when cuDSS produced it.
    ///
    /// Reported in [`MemoryEstimates::state`](crate::solver_state::MemoryEstimates::state).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum MemoryEstimateState: sys::cudssMemoryEstimateState_t, unknown = Unknown(u32) {
        /// The device and host estimates include the factors, but analysis has not yet sized every workspace.
        ///
        /// The value is neither a lower nor an upper bound, and the hybrid memory mode estimates are `0`.
        FillKnown = CUDSS_MEMORY_ESTIMATE_STATE_FILL_KNOWN,
        /// Analysis completed, and every estimate is final.
        Final = CUDSS_MEMORY_ESTIMATE_STATE_FINAL,
    }
}

bitflags::bitflags! {
    /// Matrix format flags reported by cuDSS matrix descriptors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MatrixFormat: u32 {
        /// Matrix is dense (applied to a single matrix and a batch equally).
        const DENSE = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_DENSE.0;
        /// Matrix is in CSR format.
        ///
        /// Applies to a single matrix and to a batch.
        /// cuDSS supports only 3-array CSR.
        const CSR = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_CSR.0;
        /// Matrix object represents a batch of matrices.
        const BATCH = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_BATCH.0;
        /// Matrix object represents a distributed matrix.
        ///
        /// Format flags can be combined.
        /// For example, a batch of CSR matrices is reported as `MatrixFormat::CSR | MatrixFormat::BATCH`.
        const DISTRIBUTED = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_DISTRIBUTED.0;
    }
}

impl Display for MatrixFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

impl_enum_display!(ConfigParameter, {
    ConfigParameter::ReorderingAlgorithm => "CUDSS_CONFIG_REORDERING_ALG",
    ConfigParameter::FactorizationAlgorithm => "CUDSS_CONFIG_FACTORIZATION_ALG",
    ConfigParameter::SolveAlgorithm => "CUDSS_CONFIG_SOLVE_ALG",
    ConfigParameter::MatchingAlgorithm => "CUDSS_CONFIG_MATCHING_ALG",
    ConfigParameter::SolveMode => "CUDSS_CONFIG_SOLVE_MODE",
    ConfigParameter::IterativeRefinementSteps => "CUDSS_CONFIG_IR_N_STEPS",
    ConfigParameter::IterativeRefinementTolerance => "CUDSS_CONFIG_IR_TOL",
    ConfigParameter::PivotType => "CUDSS_CONFIG_PIVOT_TYPE",
    ConfigParameter::PivotThreshold => "CUDSS_CONFIG_PIVOT_THRESHOLD",
    ConfigParameter::PivotEpsilon => "CUDSS_CONFIG_PIVOT_EPSILON",
    ConfigParameter::MaxLuNnz => "CUDSS_CONFIG_MAX_LU_NNZ",
    ConfigParameter::HybridMemoryMode => "CUDSS_CONFIG_HYBRID_MEMORY_MODE",
    ConfigParameter::HybridDeviceMemoryLimit => "CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT",
    ConfigParameter::UseCudaRegisterMemory => "CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY",
    ConfigParameter::HostThreads => "CUDSS_CONFIG_HOST_NTHREADS",
    ConfigParameter::HybridExecuteMode => "CUDSS_CONFIG_HYBRID_EXECUTE_MODE",
    ConfigParameter::PivotEpsilonAlgorithm => "CUDSS_CONFIG_PIVOT_EPSILON_ALG",
    ConfigParameter::NestedDissectionLevels => "CUDSS_CONFIG_ND_NLEVELS",
    ConfigParameter::UniformBatchSize => "CUDSS_CONFIG_UBATCH_SIZE",
    ConfigParameter::UniformBatchIndex => "CUDSS_CONFIG_UBATCH_INDEX",
    ConfigParameter::UseSuperpanels => "CUDSS_CONFIG_USE_SUPERPANELS",
    ConfigParameter::DeviceCount => "CUDSS_CONFIG_DEVICE_COUNT",
    ConfigParameter::DeviceIndices => "CUDSS_CONFIG_DEVICE_INDICES",
    ConfigParameter::SchurMode => "CUDSS_CONFIG_SCHUR_MODE",
    ConfigParameter::DeterministicMode => "CUDSS_CONFIG_DETERMINISTIC_MODE",
    ConfigParameter::NestedDissectionUnbalanceFactor => "CUDSS_CONFIG_ND_UBFACTOR",
    ConfigParameter::EmulationMantissaBitCount => "CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT",
    ConfigParameter::ForceNonsymmetricMatching => "CUDSS_CONFIG_FORCE_NONSYMMETRIC_MATCHING",
    ConfigParameter::DowncastFactors => "CUDSS_CONFIG_DOWNCAST_FACTORS",
    ConfigParameter::MemoryOptimizationSetting => "CUDSS_CONFIG_MEMORY_OPTIMIZATION_SETTING",
});

impl_enum_display!(DataParameter, {
    DataParameter::Info => "CUDSS_DATA_INFO",
    DataParameter::LuNnz => "CUDSS_DATA_LU_NNZ",
    DataParameter::NumberOfPivots => "CUDSS_DATA_NPIVOTS",
    DataParameter::Inertia => "CUDSS_DATA_INERTIA",
    DataParameter::PermutationReorderRow => "CUDSS_DATA_PERM_REORDER_ROW",
    DataParameter::PermutationReorderColumn => "CUDSS_DATA_PERM_REORDER_COL",
    DataParameter::PermutationRow => "CUDSS_DATA_PERM_ROW",
    DataParameter::PermutationColumn => "CUDSS_DATA_PERM_COL",
    DataParameter::Diagonal => "CUDSS_DATA_DIAG",
    DataParameter::UserPermutation => "CUDSS_DATA_USER_PERM",
    DataParameter::HybridDeviceMemoryMin => "CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN",
    DataParameter::CommunicationDevice => "CUDSS_DATA_COMM_DEVICE",
    DataParameter::CommunicationHost => "CUDSS_DATA_COMM_HOST",
    DataParameter::MemoryEstimates => "CUDSS_DATA_MEMORY_ESTIMATES",
    DataParameter::PermutationMatching => "CUDSS_DATA_PERM_MATCHING",
    DataParameter::ScaleRow => "CUDSS_DATA_SCALE_ROW",
    DataParameter::ScaleColumn => "CUDSS_DATA_SCALE_COL",
    DataParameter::SuperpanelCount => "CUDSS_DATA_NSUPERPANELS",
    DataParameter::UserSchurIndices => "CUDSS_DATA_USER_SCHUR_INDICES",
    DataParameter::SchurShape => "CUDSS_DATA_SCHUR_SHAPE",
    DataParameter::SchurMatrix => "CUDSS_DATA_SCHUR_MATRIX",
    DataParameter::UserNestedDissectionPartitionTree => "CUDSS_DATA_USER_ND_PARTITION_TREE",
    DataParameter::NestedDissectionPartitionTree => "CUDSS_DATA_ND_PARTITION_TREE",
    DataParameter::UserHostInterrupt => "CUDSS_DATA_USER_HOST_INTERRUPT",
    DataParameter::IterativeRefinementSteps => "CUDSS_DATA_IR_N_STEPS",
    DataParameter::UniformBatchMask => "CUDSS_DATA_UBATCH_MASK",
    DataParameter::Flops => "CUDSS_DATA_FLOPS",
});

impl_enum_display!(MatrixType, {
    MatrixType::General => "CUDSS_MTYPE_GENERAL",
    MatrixType::Symmetric => "CUDSS_MTYPE_SYMMETRIC",
    MatrixType::Hermitian => "CUDSS_MTYPE_HERMITIAN",
    MatrixType::Spd => "CUDSS_MTYPE_SPD",
    MatrixType::Hpd => "CUDSS_MTYPE_HPD",
});

impl_enum_display!(MatrixViewType, {
    MatrixViewType::Full => "CUDSS_MVIEW_FULL",
    MatrixViewType::Lower => "CUDSS_MVIEW_LOWER",
    MatrixViewType::Upper => "CUDSS_MVIEW_UPPER",
});

impl_enum_display!(IndexBase, {
    IndexBase::Zero => "CUDSS_BASE_ZERO",
    IndexBase::One => "CUDSS_BASE_ONE",
});

impl_enum_display!(Layout, {
    Layout::ColumnMajor => "CUDSS_LAYOUT_COL_MAJOR",
    Layout::RowMajor => "CUDSS_LAYOUT_ROW_MAJOR",
});

impl_enum_display!(ReorderingAlgorithm, {
    ReorderingAlgorithm::Default => "CUDSS_REORDERING_ALG_DEFAULT",
    ReorderingAlgorithm::BtfColamd => "CUDSS_REORDERING_ALG_BTF_COLAMD",
    ReorderingAlgorithm::Colamd => "CUDSS_REORDERING_ALG_COLAMD",
    ReorderingAlgorithm::Amd => "CUDSS_REORDERING_ALG_AMD",
    ReorderingAlgorithm::NestedDissection => "CUDSS_REORDERING_ALG_NESTED_DISSECTION",
    ReorderingAlgorithm::None => "CUDSS_REORDERING_ALG_NONE",
});

impl_enum_display!(FactorizationAlgorithm, {
    FactorizationAlgorithm::Default => "CUDSS_FACTORIZATION_ALG_DEFAULT",
    FactorizationAlgorithm::Multiblock => "CUDSS_FACTORIZATION_ALG_MULTIBLOCK",
    FactorizationAlgorithm::General => "CUDSS_FACTORIZATION_ALG_GENERAL",
});

impl_enum_display!(PivotEpsilonAlgorithm, {
    PivotEpsilonAlgorithm::Default => "CUDSS_PIVOT_EPSILON_ALG_DEFAULT",
    PivotEpsilonAlgorithm::Scaled => "CUDSS_PIVOT_EPSILON_ALG_SCALED",
    PivotEpsilonAlgorithm::Static => "CUDSS_PIVOT_EPSILON_ALG_STATIC",
});

impl_enum_display!(SolveAlgorithm, {
    SolveAlgorithm::Default => "CUDSS_SOLVE_ALG_DEFAULT",
    SolveAlgorithm::General => "CUDSS_SOLVE_ALG_GENERAL",
});

impl_enum_display!(MatchingAlgorithm, {
    MatchingAlgorithm::None => "CUDSS_MATCHING_ALG_NONE",
    MatchingAlgorithm::MaxDiagonalCount => "CUDSS_MATCHING_ALG_MAX_DIAG_COUNT",
    MatchingAlgorithm::MaxMinDiagonal => "CUDSS_MATCHING_ALG_MAX_MIN_DIAG",
    MatchingAlgorithm::MaxMinDiagonalAlternate => "CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT",
    MatchingAlgorithm::MaxDiagSum => "CUDSS_MATCHING_ALG_MAX_DIAG_SUM",
    MatchingAlgorithm::MaxDiagProduct => "CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT",
    MatchingAlgorithm::Auto => "CUDSS_MATCHING_ALG_AUTO",
});

impl_enum_display!(PivotType, {
    PivotType::Auto => "CUDSS_PIVOT_AUTO",
    PivotType::None => "CUDSS_PIVOT_NONE",
    PivotType::GlobalColumn => "CUDSS_PIVOT_GLOBAL_COL",
    PivotType::GlobalRow => "CUDSS_PIVOT_GLOBAL_ROW",
    PivotType::Diagonal => "CUDSS_PIVOT_DIAGONAL",
    PivotType::LocalBlock => "CUDSS_PIVOT_LOCAL_BLOCK",
    PivotType::BunchKaufman => "CUDSS_PIVOT_BUNCH_KAUFMAN",
});

impl_enum_display!(SolveMode, {
    SolveMode::Normal => "CUDSS_SOLVE_MODE_NORMAL",
    SolveMode::Transpose => "CUDSS_SOLVE_MODE_TRANSPOSE",
    SolveMode::ConjugateTranspose => "CUDSS_SOLVE_MODE_CONJUGATE_TRANSPOSE",
});

impl_enum_display!(DowncastFactorsMode, {
    DowncastFactorsMode::Disabled => "CUDSS_DOWNCAST_FACTORS_MODE_DISABLED",
    DowncastFactorsMode::KeepBlockDiagonal => "CUDSS_DOWNCAST_FACTORS_MODE_KEEP_BLOCK_DIAGONAL",
    DowncastFactorsMode::KeepStrictDiagonal => "CUDSS_DOWNCAST_FACTORS_MODE_KEEP_STRICT_DIAGONAL",
});

impl Display for MemoryEstimateState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FillKnown => f.write_str("CUDSS_MEMORY_ESTIMATE_STATE_FILL_KNOWN"),
            Self::Final => f.write_str("CUDSS_MEMORY_ESTIMATE_STATE_FINAL"),
            Self::Unknown(code) => write!(f, "unknown memory estimate state {code}"),
        }
    }
}
