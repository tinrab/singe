use std::{borrow::Cow, result::Result as StdResult};

#[cfg(feature = "cutile")]
use cuda_core::DriverError;
#[cfg(feature = "cutile")]
use cutile::{error::Error as CutileError, prelude::DeviceError as CutileDeviceError};
#[cfg(feature = "cuda_13_4")]
use singe_cuda::error::Error as CudaError;
use thiserror::Error;

/// Kernel validation, specialization, tuning, and device-execution failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The kernel specialization has not been compiled for the current CUDA context.
    #[error("kernel specialization has not been prepared in this cuda context")]
    KernelNotPrepared,
    /// The kernel compilation cache mutex is poisoned.
    #[error("kernel compilation cache lock is poisoned")]
    KernelCachePoisoned,

    /// A logical length is invalid for the requested operation.
    #[error("invalid length")]
    InvalidLength,
    /// Related buffers have incompatible lengths.
    #[error("length mismatch")]
    LengthMismatch,
    /// A named buffer does not have the exact required element count.
    #[error("length mismatch for {name}: expected {expected}, actual {actual}")]
    LengthMismatchFor {
        /// Name of the buffer whose storage requirement failed.
        name: Cow<'static, str>,
        /// Required exact element count.
        expected: usize,
        /// Supplied storage length in elements.
        actual: usize,
    },
    /// A named buffer has fewer elements than required.
    #[error("length too short for {name}: minimum {minimum}, actual {actual}")]
    LengthTooShort {
        /// Name of the buffer whose storage requirement failed.
        name: Cow<'static, str>,
        /// Minimum required element count.
        minimum: usize,
        /// Supplied storage length in elements.
        actual: usize,
    },
    /// A strided tensor can address elements beyond the supplied storage.
    #[error("rank reach mismatch for {name}: rank {rank}, required {required}, actual {actual}")]
    RankReachMismatch {
        /// Name of the buffer whose storage requirement failed.
        name: Cow<'static, str>,
        /// Rank of the strided tensor.
        rank: usize,
        /// Element count needed to cover the largest addressed offset.
        required: usize,
        /// Supplied storage length in elements.
        actual: usize,
    },
    /// A dimension or length cannot fit the native signed 32-bit interface.
    #[error("length exceeds i32")]
    LengthExceedsI32,
    /// A required raw device pointer is null.
    #[error("null pointer")]
    NullPointer,
    /// Checked dimension, offset, or storage-size arithmetic overflowed.
    #[error("size overflow")]
    SizeOverflow,
    /// The transform does not support the requested FFT length.
    #[error("unsupported fft size for {transform}: n = {n}")]
    UnsupportedFftSize {
        /// Name of the FFT transform.
        transform: String,
        /// Requested transform length.
        n: usize,
    },
    /// The operation does not support the selected axis.
    #[error("unsupported axis for {op}: axis = {axis}")]
    UnsupportedAxis {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Requested zero-based tensor axis.
        axis: usize,
    },
    /// The operation does not support the requested row or feature width.
    #[error("unsupported width for {op}: width = {width}")]
    UnsupportedWidth {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Requested row or feature width in elements.
        width: usize,
    },
    /// The operation does not support the requested convolution kernel size.
    #[error("unsupported kernel size for {op}: kernel_size = {kernel_size}")]
    UnsupportedKernelSize {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Requested convolution kernel size.
        kernel_size: usize,
    },
    /// The operation does not support the requested launch-block count.
    #[error("unsupported block count for {op}: blocks = {blocks}")]
    UnsupportedBlockCount {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Requested number of launch blocks.
        blocks: usize,
    },
    /// A required operation parameter was not supplied.
    #[error("missing parameter for {op}: {parameter}")]
    MissingParameter {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Name of the missing or unsupported parameter.
        parameter: String,
    },
    /// An operation parameter has an unsupported value.
    #[error("unsupported parameter for {op}: {parameter} = {value}")]
    UnsupportedParameter {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Name of the missing or unsupported parameter.
        parameter: String,
        /// Rejected parameter value.
        value: usize,
    },
    /// The operation cannot execute the requested combination of settings.
    #[error("unsupported configuration for {op}: {reason}")]
    UnsupportedConfiguration {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
        /// Explanation of the unsupported combination of settings.
        reason: String,
    },
    /// Candidate generation or the candidate limit left no tunings to evaluate.
    #[error("no tune candidates for {op}")]
    NoTuneCandidates {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
    },
    /// Every evaluated tuning candidate failed or was rejected.
    #[error("no successful tune candidates for {op}")]
    NoSuccessfulTuneCandidates {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
    },
    /// The cached tuning has another Rust type than the caller requested.
    #[error("tune cache type mismatch for {op}")]
    TuneCacheTypeMismatch {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
    },
    /// No tuning is cached for this device and problem key.
    #[error("tune cache miss for {op}")]
    TuneCacheMiss {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
    },
    /// The requested tuning work or cache miss cannot be handled during CUDA graph capture.
    #[error("autotune during cuda graph capture for {op}")]
    AutotuneDuringCapture {
        /// Operation whose validation, execution, or tuning failed.
        op: String,
    },

    /// A CUDA wrapper operation failed.
    #[cfg(feature = "cuda_13_4")]
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// Kernel compilation or cuTile runtime preparation failed.
    #[cfg(feature = "cutile")]
    #[error("cutile error: {0}")]
    Cutile(#[from] CutileError),
    /// Execution of a cuTile device operation failed.
    #[cfg(feature = "cutile")]
    #[error("cutile device error: {0}")]
    CutileDevice(#[from] CutileDeviceError),
    /// A cuTile CUDA driver operation failed.
    #[cfg(feature = "cutile")]
    #[error("cutile driver error: {0}")]
    CutileDriver(#[from] DriverError),
}

/// Result of kernel validation, preparation, tuning, or execution.
pub type Result<T> = StdResult<T, Error>;
