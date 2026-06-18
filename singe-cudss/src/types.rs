use std::ops::{Add, Sub};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::types::{Complex32, Complex64};
use singe_cudss_sys as sys;

/// Extended-precision real value stored as a double-double pair.
///
/// This type maps to [`DataType::F64F64`] and has the same layout as cuDSS'
/// double-double scalar type.
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
#[repr(C)]
#[repr(align(16))]
pub struct Fp64Mp2 {
    pub hi: f64,
    pub lo: f64,
}

const _: () = {
    ["Fp64Mp2 size"][size_of::<sys::cudss_fp64mp2_t>() - size_of::<Fp64Mp2>()];
    ["Fp64Mp2 alignment"][align_of::<sys::cudss_fp64mp2_t>() - align_of::<Fp64Mp2>()];
};

impl Fp64Mp2 {
    pub const fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    pub const fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    pub fn abs(self) -> f64 {
        (self.hi + self.lo).abs()
    }

    pub const fn from_raw(raw: sys::cudss_fp64mp2_t) -> Self {
        Self {
            hi: raw.hi,
            lo: raw.lo,
        }
    }

    pub const fn into_raw(self) -> sys::cudss_fp64mp2_t {
        sys::cudss_fp64mp2_t {
            hi: self.hi,
            lo: self.lo,
        }
    }
}

impl From<sys::cudss_fp64mp2_t> for Fp64Mp2 {
    fn from(raw: sys::cudss_fp64mp2_t) -> Self {
        Self::from_raw(raw)
    }
}

impl From<Fp64Mp2> for sys::cudss_fp64mp2_t {
    fn from(value: Fp64Mp2) -> Self {
        value.into_raw()
    }
}

impl Add for Fp64Mp2 {
    type Output = Fp64Mp2;

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
    type Output = Fp64Mp2;

    fn sub(self, rhs: Self) -> Self::Output {
        self + Self::new(-rhs.hi, -rhs.lo)
    }
}

/// Data type for matrix values, offsets, and indices.
///
/// The real and complex floating-point values use the same discriminants as the
/// corresponding CUDA data-type constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataType {
    /// Unset data type sentinel.
    ///
    /// This is not valid for user-facing API calls.
    Unset = sys::cudssDataType_t::CUDSS_DATA_TYPE_UNSET as _,
    /// 32-bit real single-precision floating point (`f32`).
    F32 = sys::cudssDataType_t::CUDSS_R_32F as _,
    /// 64-bit real double-precision floating point (`f64`).
    F64 = sys::cudssDataType_t::CUDSS_R_64F as _,
    /// 32-bit complex single-precision floating point ([`Complex32`]).
    C32 = sys::cudssDataType_t::CUDSS_C_32F as _,
    /// 64-bit complex double-precision floating point ([`Complex64`]).
    C64 = sys::cudssDataType_t::CUDSS_C_64F as _,
    /// 128-bit real double-double floating point ([`Fp64Mp2`]).
    ///
    /// The value is stored as the unevaluated sum of two `f64` values.
    F64F64 = sys::cudssDataType_t::CUDSS_R_64F_64F as _,
    /// 32-bit signed integer (`int32_t`). Used for index and offset arrays.
    I32 = sys::cudssDataType_t::CUDSS_R_32I as _,
    /// 64-bit signed integer (`int64_t`). Used for index and offset arrays.
    I64 = sys::cudssDataType_t::CUDSS_R_64I as _,
}

impl_enum_conversion!(sys::cudssDataType_t, DataType);

impl_enum_display!(DataType, {
    Self::Unset => "CUDSS_DATA_TYPE_UNSET",
    Self::F32 => "CUDSS_R_32F",
    Self::F64 => "CUDSS_R_64F",
    Self::C32 => "CUDSS_C_32F",
    Self::C64 => "CUDSS_C_64F",
    Self::F64F64 => "CUDSS_R_64F_64F",
    Self::I32 => "CUDSS_R_32I",
    Self::I64 => "CUDSS_R_64I",
});

/// Rust element types that map directly to a cuDSS value data type.
pub trait DataTypeLike: private::Sealed + Copy + 'static {
    /// cuDSS data type used for this Rust element type.
    const DATA_TYPE: DataType;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for singe_cuda::types::Complex32 {}
    impl Sealed for singe_cuda::types::Complex64 {}
    impl Sealed for super::Fp64Mp2 {}
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
    const DATA_TYPE: DataType = DataType::F64F64;
}

/// Solver configuration parameter names.
///
/// These values identify settings passed to [`Config::set`](crate::config::Config::set)
/// or [`Config::get`](crate::config::Config::get). Prefer the named
/// `Config` methods when one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ConfigParameter {
    /// Reordering algorithm used during analysis.
    ///
    /// Value type: [`ReorderingAlgorithm`]. Default: [`ReorderingAlgorithm::Default`].
    /// [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`]
    /// require general matrices. [`ReorderingAlgorithm::Amd`] does not support
    /// 64-bit matrix indices.
    ReorderingAlgorithm = sys::cudssConfigParam_t::CUDSS_CONFIG_REORDERING_ALG as _,
    /// Factorization algorithm used during numerical factorization.
    ///
    /// Value type: [`FactorizationAlgorithm`]. Default:
    /// [`FactorizationAlgorithm::Default`].
    FactorizationAlgorithm = sys::cudssConfigParam_t::CUDSS_CONFIG_FACTORIZATION_ALG as _,
    /// Solve algorithm used during the solve phase.
    ///
    /// Value type: [`SolveAlgorithm`]. Default: [`SolveAlgorithm::Default`].
    SolveAlgorithm = sys::cudssConfigParam_t::CUDSS_CONFIG_SOLVE_ALG as _,
    /// Optional matching algorithm used before reordering.
    ///
    /// Matching computes a column permutation, and some algorithms also compute
    /// scaling vectors. It can improve accuracy for ill-conditioned or badly
    /// scaled matrices, but it adds analysis overhead and can change factor
    /// sparsity.
    ///
    /// Value type: [`MatchingAlgorithm`]. Default: [`MatchingAlgorithm::None`].
    /// Matching is not supported with [`ReorderingAlgorithm::BtfColamd`],
    /// [`ReorderingAlgorithm::Colamd`], [`DataType::F64F64`], or distributed
    /// matrices.
    MatchingAlgorithm = sys::cudssConfigParam_t::CUDSS_CONFIG_MATCHING_ALG as _,
    /// Matrix solve modifier.
    ///
    /// Value type: `i32`. Default: `0` for no modifier. Other values are not
    /// currently supported.
    SolveMode = sys::cudssConfigParam_t::CUDSS_CONFIG_SOLVE_MODE as _,
    /// Maximum number of iterative refinement steps.
    ///
    /// Value type: `i32`. Default: `0`.
    IterativeRefinementSteps = sys::cudssConfigParam_t::CUDSS_CONFIG_IR_N_STEPS as _,
    /// Iterative refinement tolerance.
    ///
    /// When both [`Self::IterativeRefinementSteps`] and this value are greater
    /// than zero, cuDSS performs iterative refinement with convergence checking.
    ///
    /// In each iteration, the relative residual norm is computed as $||r||_2 / ||b||_2$ where $r = b - Ax$.
    ///
    /// The refinement terminates early if the residual norm is less than the specified tolerance.
    ///
    /// If this value is zero and [`Self::IterativeRefinementSteps`] is greater
    /// than zero, cuDSS performs exactly that many iterations without convergence
    /// checking.
    ///
    /// The actual iteration count can be retrieved with
    /// [`DataParameter::IterativeRefinementSteps`]. If the tolerance is not met,
    /// cuDSS returns [`Status::IrFailed`](crate::error::Status::IrFailed).
    ///
    /// Value type: `f64`. Default: `0.0`, which disables tolerance checking.
    IterativeRefinementTolerance = sys::cudssConfigParam_t::CUDSS_CONFIG_IR_TOL as _,
    /// Pivoting strategy used during factorization.
    ///
    /// Value type: [`PivotType`]. Default: [`PivotType::Auto`]. With
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`],
    /// this controls global pivoting. With nested dissection, AMD, or natural
    /// order, it controls local 1x1 pivoting.
    PivotType = sys::cudssConfigParam_t::CUDSS_CONFIG_PIVOT_TYPE as _,
    /// Threshold used to decide whether a diagonal element is pivoted.
    ///
    /// Value type: `f64`. Default: `1.0`. This setting is only supported with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    PivotThreshold = sys::cudssConfigParam_t::CUDSS_CONFIG_PIVOT_THRESHOLD as _,
    /// Pivoting epsilon.
    ///
    /// By default, this is the absolute value to test and replace small diagonal elements encountered during numerical factorization.
    ///
    /// If [`Self::PivotEpsilonAlgorithm`] is
    /// [`PivotEpsilonAlgorithm::Scaled`], cuDSS scales this value before use.
    ///
    /// Value type: `f64`. Defaults are `1e-5` for single precision and `1e-13`
    /// for double precision.
    PivotEpsilon = sys::cudssConfigParam_t::CUDSS_CONFIG_PIVOT_EPSILON as _,
    /// Upper limit on the number of nonzero entries in LU factors.
    ///
    /// Value type: `i64`. Default: `-1`, which lets cuDSS use its internal
    /// estimate. This is only relevant for general matrices with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    MaxLuNnz = sys::cudssConfigParam_t::CUDSS_CONFIG_MAX_LU_NNZ as _,
    /// Enables hybrid host/device memory mode.
    ///
    /// Value type: `i32` boolean. Default: `0` (disabled). Enable this before
    /// [`Phase::ANALYSIS`]; otherwise analysis must be rerun. This mode keeps
    /// part of the internal factor data on the host while factorization and
    /// solve still run on the GPU.
    ///
    /// Not supported with [`ReorderingAlgorithm::BtfColamd`],
    /// [`ReorderingAlgorithm::Colamd`], or [`FactorizationAlgorithm::Multiblock`].
    HybridMemoryMode = sys::cudssConfigParam_t::CUDSS_CONFIG_HYBRID_MEMORY_MODE as _,
    /// User-defined device memory limit (number of bytes) for the hybrid memory mode.
    ///
    /// This setting only affects execution when [`Self::HybridMemoryMode`] is
    /// enabled. In multi-GPU mode it must be set separately for each selected
    /// device.
    ///
    /// Value type: `i64`. Default: `-1`, which uses cuDSS' internal heuristic.
    HybridDeviceMemoryLimit = sys::cudssConfigParam_t::CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT as _,
    /// Enables `cudaHostRegister()` use in hybrid memory mode.
    ///
    /// Value type: `i32` boolean. Default: `1`, which lets cuDSS register host
    /// memory when the device supports it.
    UseCudaRegisterMemory = sys::cudssConfigParam_t::CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY as _,
    /// Number of threads to be used by cuDSS in [MT mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-threaded-mt-mode).
    /// This setting only affects execution when multi-threaded mode is enabled.
    ///
    /// Value type: `i32`. Default: `-1`, which uses the maximum thread count
    /// reported by cuDSS.
    HostThreads = sys::cudssConfigParam_t::CUDSS_CONFIG_HOST_NTHREADS as _,
    /// Enables hybrid host/device execute mode.
    ///
    /// Value type: `i32` boolean. Default: `0` (disabled). Enable this before
    /// [`Phase::ANALYSIS`]. When enabled, some work may run on the CPU and input
    /// matrix, right-hand side, and solution buffers may be host pointers.
    ///
    /// Not supported with [`Self::HybridMemoryMode`], [`Self::DeterministicMode`],
    /// batched matrices, or non-trivial leading dimensions for the right-hand
    /// side or solution.
    HybridExecuteMode = sys::cudssConfigParam_t::CUDSS_CONFIG_HYBRID_EXECUTE_MODE as _,
    /// Algorithm used to compute replacement values for small pivots.
    ///
    /// Value type: [`PivotEpsilonAlgorithm`]. Default:
    /// [`PivotEpsilonAlgorithm::Default`]. [`PivotEpsilonAlgorithm::Scaled`]
    /// is not supported with [`ReorderingAlgorithm::BtfColamd`] or
    /// [`ReorderingAlgorithm::Colamd`].
    PivotEpsilonAlgorithm = sys::cudssConfigParam_t::CUDSS_CONFIG_PIVOT_EPSILON_ALG as _,
    /// Minimum nested-dissection level count.
    ///
    /// Value type: `i32`. Default: `10`. This only affects
    /// [`ReorderingAlgorithm::NestedDissection`] and
    /// [`ReorderingAlgorithm::Default`]. In MGMN mode, cuDSS may increase the
    /// value to satisfy process-count requirements.
    NestedDissectionLevels = sys::cudssConfigParam_t::CUDSS_CONFIG_ND_NLEVELS as _,
    /// Number of systems in a uniform batch.
    ///
    /// A uniform batch has the same sparsity pattern for every matrix, but may
    /// use different values. Set this before [`Phase::ANALYSIS`].
    ///
    /// Value type: `i32`. Default: `1`. Not supported with hybrid memory mode,
    /// hybrid execute mode, MGMN mode, [`ReorderingAlgorithm::BtfColamd`], or
    /// [`ReorderingAlgorithm::Colamd`].
    UniformBatchSize = sys::cudssConfigParam_t::CUDSS_CONFIG_UBATCH_SIZE as _,
    /// Uniform-batch system index to process.
    ///
    /// Value type: `i32`. Default: `-1`, which processes all systems at once.
    /// Non-negative values select a single 0-based system index and must be
    /// less than [`Self::UniformBatchSize`].
    UniformBatchIndex = sys::cudssConfigParam_t::CUDSS_CONFIG_UBATCH_INDEX as _,
    /// Enable or disable superpanel optimization.
    ///
    /// Value type: `i32` boolean. Default: `1` (enabled). Set this before
    /// [`Phase::ANALYSIS`]; otherwise analysis must be rerun.
    UseSuperpanels = sys::cudssConfigParam_t::CUDSS_CONFIG_USE_SUPERPANELS as _,
    /// Number of devices used by a multi-GPU context.
    ///
    /// Value type: `i32`. Default: `1`. cuDSS currently supports at most 16
    /// devices.
    DeviceCount = sys::cudssConfigParam_t::CUDSS_CONFIG_DEVICE_COUNT as _,
    /// Device indices used by a multi-GPU context.
    ///
    /// Value type: `&[i32]`. By default, cuDSS uses devices
    /// `0..DeviceCount`. The current CUDA device must match the first selected
    /// device, or device `0` when default indices are used.
    DeviceIndices = sys::cudssConfigParam_t::CUDSS_CONFIG_DEVICE_INDICES as _,
    /// Enables Schur complement mode.
    ///
    /// Value type: `i32` boolean. Default: `0` (disabled). Not supported with
    /// [`ReorderingAlgorithm::BtfColamd`], [`ReorderingAlgorithm::Colamd`],
    /// multi-GPU or MGMN mode, [`FactorizationAlgorithm::Multiblock`], batches,
    /// or matching.
    SchurMode = sys::cudssConfigParam_t::CUDSS_CONFIG_SCHUR_MODE as _,
    /// Enables deterministic mode.
    ///
    /// When run on GPUs with the same architecture and SM count, deterministic
    /// mode produces bitwise-identical results for bitwise-identical input data
    /// and solver settings.
    ///
    /// Deterministic mode may use slower kernels.
    ///
    /// Value type: `i32` boolean. Default: `0` (disabled). Supported only for a
    /// single GPU, a single right-hand side, no hybrid modes, no iterative
    /// refinement, and no [`ReorderingAlgorithm::BtfColamd`] or
    /// [`ReorderingAlgorithm::Colamd`].
    DeterministicMode = sys::cudssConfigParam_t::CUDSS_CONFIG_DETERMINISTIC_MODE as _,
    /// Controls the balance of the nested dissection partition tree. The value represents the allowed percentage of imbalance between the two children at each level of the partition tree. Possible values are from 1 to 100.
    ///
    /// Smaller values produce a more balanced partition tree.
    ///
    /// Value type: `i32`. Default: `20`. For multi-GPU and MGMN modes, NVIDIA
    /// recommends smaller values such as `2..=5`.
    NestedDissectionUnbalanceFactor = sys::cudssConfigParam_t::CUDSS_CONFIG_ND_UBFACTOR as _,
}

impl_enum_conversion!(sys::cudssConfigParam_t, ConfigParameter);

/// Solver data parameter names.
///
/// These values identify data slots passed to [`Data::set`](crate::data::Data::set),
/// [`Data::get`](crate::data::Data::get), or the slice/device variants on
/// [`Data`](crate::data::Data). Prefer the named `Data` methods when one exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DataParameter {
    /// Host-readable execution information code.
    ///
    /// This reports device-side asynchronous errors, or host-side errors in
    /// hybrid execute mode. A value of `0` means success. For positive-definite
    /// matrix types, a nonzero value may be the 1-based index of the first
    /// non-positive minor in the reordered matrix.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type: `i32`.
    Info = sys::cudssDataParam_t::CUDSS_DATA_INFO as _,
    /// Number of nonzero entries in LU factors.
    ///
    /// For non-uniform batches, this is accumulated across the batch.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type: `i64`.
    LuNnz = sys::cudssDataParam_t::CUDSS_DATA_LU_NNZ as _,
    /// Number of pivots encountered during factorization.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type:
    /// the system matrix index type.
    NumberOfPivots = sys::cudssDataParam_t::CUDSS_DATA_NPIVOTS as _,
    /// Positive and negative indices of inertia for the system matrix `A` (two integer values). Valid only for symmetric/Hermitian non positive-definite matrix types.
    ///
    /// For non-uniform batches, cuDSS returns the accumulated counts across the
    /// batch.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice). Memory: host.
    /// Value type: two values of the system matrix index type.
    Inertia = sys::cudssDataParam_t::CUDSS_DATA_INERTIA as _,
    /// Row permutation `P` after reordering, such that `A[P,Q]` is factorized.
    ///
    /// Not supported for non-uniform batches.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    PermutationReorderRow = sys::cudssDataParam_t::CUDSS_DATA_PERM_REORDER_ROW as _,
    /// Column permutation `Q` after reordering, such that `A[P,Q]` is factorized.
    ///
    /// Not supported for non-uniform batches.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    PermutationReorderColumn = sys::cudssDataParam_t::CUDSS_DATA_PERM_REORDER_COL as _,
    /// Final row permutation `P`, including reordering and pivoting.
    ///
    /// Not supported for non-uniform batches. Only supported with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    PermutationRow = sys::cudssDataParam_t::CUDSS_DATA_PERM_ROW as _,
    /// Final column permutation `Q`, including reordering and pivoting.
    ///
    /// Not supported for non-uniform batches. Only supported with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    PermutationColumn = sys::cudssDataParam_t::CUDSS_DATA_PERM_COL as _,
    /// Diagonal of the factorized matrix in original matrix order.
    ///
    /// The values include matching, reordering, and pivoting effects. For
    /// non-uniform batches, cuDSS returns accumulated values across the batch.
    /// Only supported with [`ReorderingAlgorithm::BtfColamd`] or
    /// [`ReorderingAlgorithm::Colamd`].
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix value type.
    Diagonal = sys::cudssDataParam_t::CUDSS_DATA_DIAG as _,
    /// User permutation used instead of running a reordering algorithm.
    ///
    /// Disable the user permutation by setting a null pointer with zero size.
    /// The provided buffer must contain `n` indices, where `n` is the system
    /// matrix row or column count. The index type must match the system matrix.
    /// The values must represent a valid permutation vector for the matrix index
    /// base.
    ///
    /// cuDSS copies the data into an internal buffer.
    ///
    /// Not supported for non-uniform batches or with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    ///
    /// Set with [`Data::set_slice`](crate::data::Data::set_slice). Memory: host or
    /// device. Value type: the system matrix index type.
    UserPermutation = sys::cudssDataParam_t::CUDSS_DATA_USER_PERM as _,
    /// Minimum device memory, in bytes, required by hybrid memory mode.
    ///
    /// Query this after analysis. In multi-GPU mode, the value is reported for
    /// the current CUDA device.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type: `i64`.
    HybridDeviceMemoryMin = sys::cudssDataParam_t::CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN as _,
    /// Device communicator for [MGMN mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-gpu-multi-node-mgmn-mode).
    /// The concrete communicator type must match the device backend selected by
    /// [`Context::set_communication_layer`](crate::context::Context::set_communication_layer).
    /// This communicator is used for communication involving device memory buffers.
    ///
    /// Set with [`Data::set`](crate::data::Data::set). Memory: host. Value type: `*mut c_void`.
    CommunicationDevice = sys::cudssDataParam_t::CUDSS_DATA_COMM_DEVICE as _,
    /// Host communicator for [MGMN mode](https://docs.nvidia.com/cuda/cudss/advanced_features.html#multi-gpu-multi-node-mgmn-mode).
    /// The concrete communicator type must match the host backend selected by
    /// [`Context::set_communication_layer`](crate::context::Context::set_communication_layer).
    /// This communicator is used for communication involving host memory buffers.
    ///
    /// Set with [`Data::set`](crate::data::Data::set). Memory: host. Value type: `*mut c_void`.
    CommunicationHost = sys::cudssDataParam_t::CUDSS_DATA_COMM_HOST as _,
    /// Host and device memory estimates, in bytes, for the analyzed solve.
    ///
    /// Query this after analysis. Estimates depend on the configuration, input
    /// matrix, and right-hand side count used during analysis; changing those
    /// inputs can make the estimates stale.
    ///
    /// The returned array uses these positions:
    ///
    /// * `0`: permanent device memory
    /// * `1`: peak device memory
    /// * `2`: permanent host memory
    /// * `3`: peak host memory
    /// * `4`: minimum device memory for hybrid memory mode
    /// * `5`: maximum host memory for hybrid memory mode
    /// * `6..=15`: reserved
    ///
    /// In multi-GPU mode, the estimate is reported for the current CUDA device.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type:
    /// `[i64; 16]`.
    MemoryEstimates = sys::cudssDataParam_t::CUDSS_DATA_MEMORY_ESTIMATES as _,
    /// Matching column permutation `Q`, such that `A[:,Q]` is reordered and then factorized.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    PermutationMatching = sys::cudssDataParam_t::CUDSS_DATA_PERM_MATCHING as _,
    /// Row scaling for the factorized matrix, in original row order.
    ///
    /// Only supported when matching is enabled and the chosen matching algorithm
    /// computes scaling.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: real `f32` or `f64`, matching the absolute value type of the
    /// system matrix.
    ScaleRow = sys::cudssDataParam_t::CUDSS_DATA_SCALE_ROW as _,
    /// Column scaling for the factorized matrix, in original column order.
    ///
    /// Only supported when matching is enabled and the chosen matching algorithm
    /// computes scaling.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: real `f32` or `f64`, matching the absolute value type of the
    /// system matrix.
    ScaleColumn = sys::cudssDataParam_t::CUDSS_DATA_SCALE_COL as _,
    /// Number of superpanels in the matrix.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type:
    /// the system matrix index type.
    SuperpanelCount = sys::cudssDataParam_t::CUDSS_DATA_NSUPERPANELS as _,
    /// User-provided Schur complement indices.
    ///
    /// The provided buffer must contain `n` integer values, where `n` is the
    /// system matrix row or column count. Values equal to `1` mark rows/columns
    /// that belong to the Schur complement; `0` marks the rest. cuDSS copies the
    /// data into an internal buffer.
    ///
    /// Set with [`Data::set_slice`](crate::data::Data::set_slice). Memory: host or
    /// device. Value type: the system matrix index type.
    UserSchurIndices = sys::cudssDataParam_t::CUDSS_DATA_USER_SCHUR_INDICES as _,
    /// Schur complement shape as `(rows, columns, sparse_nnz)`.
    ///
    /// Query this to allocate a dense or sparse Schur complement
    /// [`Matrix`](crate::matrix::Matrix), then pass that matrix with
    /// [`Self::SchurMatrix`] for cuDSS to fill.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type:
    /// `[i64; 3]`.
    SchurShape = sys::cudssDataParam_t::CUDSS_DATA_SCHUR_SHAPE as _,
    /// Schur complement matrix to be filled by cuDSS.
    ///
    /// The value, offset, and index data types must match the corresponding
    /// system matrix types. The Schur complement matrix cannot be distributed or
    /// batched. Dense Schur complements must be column-major. Sparse Schur
    /// complements must use [`MatrixType::General`] and [`MatrixViewType::Full`].
    ///
    /// Set with [`Data::set`](crate::data::Data::set). Memory: host, with matrix
    /// buffers on host or device. Value type: [`Matrix`](crate::matrix::Matrix)
    /// handle.
    SchurMatrix = sys::cudssDataParam_t::CUDSS_DATA_SCHUR_MATRIX as _,
    /// User-provided nested-dissection partition tree.
    ///
    /// This is used with [`Self::UserPermutation`] to skip reordering. Disable
    /// it by setting a null pointer with zero size.
    ///
    /// Prefer reusing a tree returned by [`Self::NestedDissectionPartitionTree`]
    /// rather than constructing one manually.
    ///
    /// Set with [`Data::set_slice`](crate::data::Data::set_slice). Memory: host or
    /// device. Value type: the system matrix index type.
    UserNestedDissectionPartitionTree =
        sys::cudssDataParam_t::CUDSS_DATA_USER_ND_PARTITION_TREE as _,
    /// Nested-dissection partition tree computed during reordering.
    ///
    /// The array has `2^nlevels - 1` elements, where `nlevels` is
    /// [`ConfigParameter::NestedDissectionLevels`]. It can be reused with
    /// [`Self::UserPermutation`] and [`Self::UserNestedDissectionPartitionTree`]
    /// for another matrix with the same sparsity structure.
    ///
    /// Read with [`Data::get_slice`](crate::data::Data::get_slice) or
    /// [`Data::get_device`](crate::data::Data::get_device). Memory: host or device.
    /// Value type: the system matrix index type.
    NestedDissectionPartitionTree = sys::cudssDataParam_t::CUDSS_DATA_ND_PARTITION_TREE as _,
    /// User-provided host interrupt pointer.
    ///
    /// Setting a non-null pointer enables host-side interruption checks in
    /// [`Context::execute`](crate::context::Context::execute). If the pointed-to
    /// value is nonzero, cuDSS stops early and returns
    /// [`Status::ExecutionFailed`](crate::error::Status::ExecutionFailed).
    /// To resume after an interruption, set the value at that location back to `0` and call again.
    /// Disable the feature by setting a null pointer.
    ///
    /// Since most device kernels inside cuDSS are launched asynchronously,
    /// execution on the GPU may continue some time after the host execution has been interrupted.
    ///
    /// Set with [`Data::set`](crate::data::Data::set). Memory: host. Default:
    /// null pointer. Value type: pointer to a 4-byte integer.
    ///
    /// When setting the flag from another thread, use an atomic store with
    /// release semantics. A plain write to a non-atomic object from another
    /// thread is a data race and undefined behavior.
    ///
    /// Not supported in multi-GPU or MGMN mode, or with
    /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
    UserHostInterrupt = sys::cudssDataParam_t::CUDSS_DATA_USER_HOST_INTERRUPT as _,
    /// Actual number of iterative refinement steps performed during solve.
    ///
    /// The value can be less than
    /// [`ConfigParameter::IterativeRefinementSteps`] if the configured tolerance
    /// is reached.
    /// The value is 0 if iterative refinement is disabled or if the initial solution already satisfies the tolerance.
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type: `i32`.
    IterativeRefinementSteps = sys::cudssDataParam_t::CUDSS_DATA_IR_N_STEPS as _,
    /// A bitmask specifying the subset of problems to be factorized or solved in the uniform batch case. Each bit position corresponds to the index of a matrix in the uniform batch; setting a bit to `1` includes that matrix in the operation.
    ///
    /// This can be used instead of [`ConfigParameter::UniformBatchIndex`] to
    /// select multiple specific systems in one call.
    ///
    /// This may be set after [`Phase::FACTORIZATION`] to solve only a subset of
    /// systems.
    ///
    /// Set with [`Data::set`](crate::data::Data::set). Memory: host. Value type: `i64`. Default: `-1`, which
    /// selects all systems. Not supported with hybrid modes, MGMN mode,
    /// [`ReorderingAlgorithm::BtfColamd`], or [`ReorderingAlgorithm::Colamd`].
    UniformBatchMask = sys::cudssDataParam_t::CUDSS_DATA_UBATCH_MASK as _,
    /// Floating-point operation count for the factorization phase.
    ///
    /// Available after [`Phase::ANALYSIS`].
    ///
    /// Read with [`Data::get`](crate::data::Data::get). Memory: host. Value type: `f64`.
    Flops = sys::cudssDataParam_t::CUDSS_DATA_FLOPS as _,
}

impl_enum_conversion!(sys::cudssDataParam_t, DataParameter);

bitflags::bitflags! {
    /// Solver phases passed to [`Context::execute`](crate::context::Context::execute).
    ///
    /// Phases can be combined with `|`, for example
    /// `Phase::FACTORIZATION | Phase::SOLVE`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Phase: u32 {
        /// Reordering.
        const REORDERING = sys::cudssPhase_t::CUDSS_PHASE_REORDERING as _;
        /// Symbolic factorization.
        ///
        /// Do not call symbolic factorization twice without a reordering phase
        /// between the calls.
        const SYMBOLIC_FACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_SYMBOLIC_FACTORIZATION as _;
        /// Reordering and symbolic factorization combined.
        const ANALYSIS = sys::cudssPhase_t::CUDSS_PHASE_ANALYSIS as _;
        /// Numerical factorization.
        const FACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_FACTORIZATION as _;
        /// Numerical re-factorization.
        ///
        /// This is distinct from [`Self::FACTORIZATION`] only with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const REFACTORIZATION = sys::cudssPhase_t::CUDSS_PHASE_REFACTORIZATION as _;
        /// Applies the reordering permutation to the right-hand side before forward substitution.
        ///
        /// Solve sub-phases are not supported with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_FORWARD_PERMUTATION = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_FWD_PERM as _;
        /// Forward substitution sub-step of the solve phase.
        ///
        /// This includes local permutations from partial pivoting. Use
        /// [`DataParameter::PermutationReorderRow`] and
        /// [`DataParameter::PermutationRow`] to account for that effect when
        /// needed.
        ///
        /// Solve sub-phases are not supported with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_FORWARD = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_FWD as _;
        /// Diagonal solve sub-step for symmetric or Hermitian matrix types.
        ///
        /// Solve sub-phases are not supported with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_DIAGONAL = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_DIAG as _;
        /// Backward substitution sub-step of the solve phase.
        ///
        /// This includes local permutations from partial pivoting. Use
        /// [`DataParameter::PermutationReorderRow`] and
        /// [`DataParameter::PermutationRow`] to account for that effect when
        /// needed.
        ///
        /// Solve sub-phases are not supported with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_BACKWARD = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_BWD as _;
        /// Applies the inverse reordering permutation after backward substitution.
        ///
        /// If matching with scaling is enabled, this phase also applies the
        /// inverse matching permutation and inverse scaling.
        ///
        /// Solve sub-phases are not supported with
        /// [`ReorderingAlgorithm::BtfColamd`] or [`ReorderingAlgorithm::Colamd`].
        const SOLVE_BACKWARD_PERMUTATION = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_BWD_PERM as _;
        /// Iterative refinement.
        ///
        /// Controlled by [`ConfigParameter::IterativeRefinementSteps`] and
        /// [`ConfigParameter::IterativeRefinementTolerance`]. Solve sub-phases
        /// are not supported with [`ReorderingAlgorithm::BtfColamd`] or
        /// [`ReorderingAlgorithm::Colamd`].
        const SOLVE_REFINEMENT = sys::cudssPhase_t::CUDSS_PHASE_SOLVE_REFINEMENT as _;
        /// Full solve phase.
        ///
        /// This combines forward permutation, forward substitution, diagonal
        /// solve, backward substitution, backward permutation, and optional
        /// iterative refinement.
        ///
        /// Combining solve sub-phases is preferred over calling them separately.
        const SOLVE = sys::cudssPhase_t::CUDSS_PHASE_SOLVE as _;
    }
}

/// Matrix type for sparse system matrices.
///
/// The matrix type describes mathematical properties of the stored matrix and
/// determines the factorization cuDSS computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MatrixType {
    /// General matrix.
    ///
    /// cuDSS computes an `LDU` factorization with optional local or global
    /// pivoting.
    General = sys::cudssMatrixType_t::CUDSS_MTYPE_GENERAL as _,
    /// Real symmetric matrix. `LDL^T` factorization will be computed with optional local pivoting.
    Symmetric = sys::cudssMatrixType_t::CUDSS_MTYPE_SYMMETRIC as _,
    /// Complex Hermitian matrix. `LDL^H` factorization will be computed with optional local pivoting.
    Hermitian = sys::cudssMatrixType_t::CUDSS_MTYPE_HERMITIAN as _,
    /// Symmetric positive-definite matrix.
    ///
    /// cuDSS computes a Cholesky factorization with optional local pivoting.
    /// If cuDSS encounters a non-positive minor, [`DataParameter::Info`] can
    /// report its 1-based reordered index.
    Spd = sys::cudssMatrixType_t::CUDSS_MTYPE_SPD as _,
    /// Hermitian positive-definite matrix.
    ///
    /// cuDSS computes a complex Cholesky factorization with optional local
    /// pivoting. If cuDSS encounters a non-positive minor,
    /// [`DataParameter::Info`] can report its 1-based reordered index.
    Hpd = sys::cudssMatrixType_t::CUDSS_MTYPE_HPD as _,
}

impl_enum_conversion!(sys::cudssMatrixType_t, MatrixType);

/// Matrix view for sparse matrices.
///
/// This controls which triangular part of a symmetric or Hermitian sparse
/// matrix cuDSS reads. The view is ignored for [`MatrixType::General`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MatrixViewType {
    /// Full matrix \[default\].
    Full = sys::cudssMatrixViewType_t::CUDSS_MVIEW_FULL as _,
    /// Lower-triangular matrix (including the diagonal). All values above the main diagonal will be ignored.
    Lower = sys::cudssMatrixViewType_t::CUDSS_MVIEW_LOWER as _,
    /// Upper-triangular matrix (including the diagonal). All values below the main diagonal will be ignored.
    Upper = sys::cudssMatrixViewType_t::CUDSS_MVIEW_UPPER as _,
}

impl_enum_conversion!(sys::cudssMatrixViewType_t, MatrixViewType);

/// Indexing base for sparse matrix indices.
///
/// cuDSS uses the sparse input matrix index base for index data returned by queries such
/// as [`DataParameter::PermutationReorderRow`] and
/// [`DataParameter::PermutationReorderColumn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum IndexBase {
    /// Zero-based indexing \[default\].
    Zero = sys::cudssIndexBase_t::CUDSS_BASE_ZERO as _,
    /// One-based indexing.
    One = sys::cudssIndexBase_t::CUDSS_BASE_ONE as _,
}

impl_enum_conversion!(sys::cudssIndexBase_t, IndexBase);

/// Dense matrix layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Layout {
    /// Column-major layout \[default\].
    ColumnMajor = sys::cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR as _,
    /// Row-major layout. Currently not supported.
    RowMajor = sys::cudssLayout_t::CUDSS_LAYOUT_ROW_MAJOR as _,
}

impl_enum_conversion!(sys::cudssLayout_t, Layout);

/// Reordering algorithm used during analysis.
///
/// Used with [`ConfigParameter::ReorderingAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ReorderingAlgorithm {
    /// Let cuDSS choose the reordering algorithm.
    ///
    /// Currently equivalent to [`Self::NestedDissection`].
    Default = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_DEFAULT as _,
    /// Block triangular form (BTF) combined with COLAMD. Supports global pivoting.
    BtfColamd = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_BTF_COLAMD as _,
    /// COLAMD with trivial block structure. Supports global pivoting.
    Colamd = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_COLAMD as _,
    /// Approximate minimum degree (AMD) reordering.
    Amd = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_AMD as _,
    /// Nested dissection algorithm based on METIS.
    NestedDissection = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_NESTED_DISSECTION as _,
    /// Uses natural (identity) order for the internal ordering when no user permutation is supplied.
    None = sys::cudssReorderingAlg_t::CUDSS_REORDERING_ALG_NONE as _,
}

impl_enum_conversion!(sys::cudssReorderingAlg_t, ReorderingAlgorithm);

/// Factorization algorithm for numerical factorization.
///
/// Used with [`ConfigParameter::FactorizationAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FactorizationAlgorithm {
    /// Let cuDSS choose the factorization algorithm.
    ///
    /// Currently equivalent to [`Self::General`].
    Default = sys::cudssFactorizationAlg_t::CUDSS_FACTORIZATION_ALG_DEFAULT as _,
    /// Multiblock factorization algorithm.
    ///
    /// This is supported only for [`DataType::F64`] and is intended for
    /// extremely sparse matrices.
    Multiblock = sys::cudssFactorizationAlg_t::CUDSS_FACTORIZATION_ALG_MULTIBLOCK as _,
    /// General factorization algorithm.
    General = sys::cudssFactorizationAlg_t::CUDSS_FACTORIZATION_ALG_GENERAL as _,
}

impl_enum_conversion!(sys::cudssFactorizationAlg_t, FactorizationAlgorithm);

/// Pivot epsilon algorithm used when replacing small pivots.
///
/// Used with [`ConfigParameter::PivotEpsilonAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PivotEpsilonAlgorithm {
    /// Let cuDSS choose the pivot epsilon algorithm.
    ///
    /// Currently equivalent to [`Self::Static`].
    Default = sys::cudssPivotEpsilonAlg_t::CUDSS_PIVOT_EPSILON_ALG_DEFAULT as _,
    /// Replaces pivots below the threshold with a signed epsilon scaled by the
    /// maximum element in the corresponding row and column of the original
    /// matrix.
    ///
    /// Not supported with [`ReorderingAlgorithm::BtfColamd`] or
    /// [`ReorderingAlgorithm::Colamd`].
    Scaled = sys::cudssPivotEpsilonAlg_t::CUDSS_PIVOT_EPSILON_ALG_SCALED as _,
    /// Pivots below the threshold are replaced by the appropriately signed epsilon (no scaling). Recommended for most cases.
    Static = sys::cudssPivotEpsilonAlg_t::CUDSS_PIVOT_EPSILON_ALG_STATIC as _,
}

impl_enum_conversion!(sys::cudssPivotEpsilonAlg_t, PivotEpsilonAlgorithm);

/// Solve algorithm.
///
/// Used with [`ConfigParameter::SolveAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SolveAlgorithm {
    /// Let the library choose the solve algorithm (recommended).
    Default = sys::cudssSolveAlg_t::CUDSS_SOLVE_ALG_DEFAULT as _,
    /// Triangular solve only: forward/back substitution using the existing L/U factors without modifying them.
    General = sys::cudssSolveAlg_t::CUDSS_SOLVE_ALG_GENERAL as _,
}

impl_enum_conversion!(sys::cudssSolveAlg_t, SolveAlgorithm);

/// Matching algorithm for optional column permutation before reordering.
///
/// Used with [`ConfigParameter::MatchingAlgorithm`]. Except for [`Self::None`]
/// and [`Self::Auto`], the algorithms are based on MC64 from HSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MatchingAlgorithm {
    /// Matching is disabled.
    None = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_NONE as _,
    /// Column permutation that maximizes the number of diagonal entries.
    ///
    /// MC64 job 1. This does not use matrix values and is not generally
    /// recommended.
    MaxDiagonalCount = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_MAX_DIAG_COUNT as _,
    /// Column permutation that maximizes the smallest value on the diagonal.
    ///
    /// MC64 job 2.
    MaxMinDiagonal = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_MAX_MIN_DIAG as _,
    /// Alternate algorithm to maximize the smallest value on the diagonal.
    ///
    /// MC64 job 3. May differ in performance from [`Self::MaxMinDiagonal`].
    MaxMinDiagonalAlternate = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT as _,
    /// Column permutation that maximizes the sum of diagonal entries.
    ///
    /// MC64 job 4.
    MaxDiagSum = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_MAX_DIAG_SUM as _,
    /// Column permutation that maximizes the product of diagonal entries.
    ///
    /// MC64 job 5. Also computes row and column scaling so nonzero diagonal
    /// entries are `1` in absolute value and off-diagonal entries are at most
    /// `1`. This is usually the most useful matching option for accuracy and
    /// requires matrix values during analysis.
    MaxDiagProduct = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT as _,
    /// Let cuDSS choose the matching algorithm.
    ///
    /// Currently equivalent to [`Self::MaxDiagProduct`].
    Auto = sys::cudssMatchingAlg_t::CUDSS_MATCHING_ALG_AUTO as _,
}

impl_enum_conversion!(sys::cudssMatchingAlg_t, MatchingAlgorithm);

/// Pivoting strategy for numerical factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PivotType {
    /// Let cuDSS choose the pivoting strategy.
    ///
    /// cuDSS currently chooses based on the reordering algorithm and matrix type:
    ///
    /// * [`ReorderingAlgorithm::NestedDissection`] and
    ///   [`ReorderingAlgorithm::Amd`]: [`Self::Diagonal`] for symmetric or
    ///   Hermitian indefinite matrices, [`Self::LocalBlock`] for general
    ///   matrices.
    /// * [`ReorderingAlgorithm::BtfColamd`] and [`ReorderingAlgorithm::Colamd`]:
    ///   [`Self::GlobalColumn`].
    Auto = sys::cudssPivotType_t::CUDSS_PIVOT_AUTO as _,
    /// No pivoting.
    ///
    /// Used only with [`ReorderingAlgorithm::NestedDissection`] and
    /// [`ReorderingAlgorithm::Amd`].
    None = sys::cudssPivotType_t::CUDSS_PIVOT_NONE as _,
    /// Global column pivoting.
    ///
    /// Used only with [`ReorderingAlgorithm::BtfColamd`] and
    /// [`ReorderingAlgorithm::Colamd`].
    /// Searches for pivot elements within entire columns of the sub-matrix.
    GlobalColumn = sys::cudssPivotType_t::CUDSS_PIVOT_GLOBAL_COL as _,
    /// Global row pivoting.
    ///
    /// Used only with [`ReorderingAlgorithm::BtfColamd`] and
    /// [`ReorderingAlgorithm::Colamd`].
    /// Searches for pivot elements within entire rows of the sub-matrix.
    GlobalRow = sys::cudssPivotType_t::CUDSS_PIVOT_GLOBAL_ROW as _,
    /// Diagonal pivoting.
    ///
    /// Used only with [`ReorderingAlgorithm::NestedDissection`] and
    /// [`ReorderingAlgorithm::Amd`].
    /// For symmetric/Hermitian indefinite matrices, searches for pivot elements
    /// only within the diagonal of the supernode.
    Diagonal = sys::cudssPivotType_t::CUDSS_PIVOT_DIAGONAL as _,
    /// Complete block pivoting.
    ///
    /// Used only with [`ReorderingAlgorithm::NestedDissection`] and
    /// [`ReorderingAlgorithm::Amd`].
    /// For general matrices, searches for pivot elements within the entire
    /// diagonal block of the supernode.
    LocalBlock = sys::cudssPivotType_t::CUDSS_PIVOT_LOCAL_BLOCK as _,
    /// Bunch-Kaufman pivoting. Reserved for future use. Currently not supported.
    BunchKaufman = sys::cudssPivotType_t::CUDSS_PIVOT_BUNCH_KAUFMAN as _,
}

impl_enum_conversion!(sys::cudssPivotType_t, PivotType);

bitflags::bitflags! {
    /// Matrix format flags reported by cuDSS matrix descriptors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MatrixFormat: u32 {
        /// Matrix is dense (applied to a single matrix and a batch equally).
        const DENSE = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_DENSE as _;
        /// Matrix is in CSR format.
        ///
        /// Applies to a single matrix and to a batch. cuDSS supports only
        /// 3-array CSR.
        const CSR = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_CSR as _;
        /// Matrix object represents a batch of matrices.
        const BATCH = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_BATCH as _;
        /// Matrix object represents a distributed matrix.
        ///
        /// Format flags can be combined. For example, a batch of CSR matrices is
        /// reported as `MatrixFormat::CSR | MatrixFormat::BATCH`.
        const DISTRIBUTED = sys::cudssMatrixFormat_t::CUDSS_MFORMAT_DISTRIBUTED as _;
    }
}
