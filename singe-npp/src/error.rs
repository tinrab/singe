#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_npp_sys as sys;
use thiserror::Error;

use crate::types::Size;

status_enum! {
    /// Status returned by NPP calls.
    ///
    /// NPP statuses are signed: negative values are errors and positive values are warnings,
    /// so raw codes are `i32` rather than the `u32` used by other wrappers.
    pub enum Status: i32 => sys::NppStatus, unknown = "UNKNOWN_NPP_STATUS" {
        /// The native call succeeded.
        Success = NPP_SUCCESS,
        /// Not supported mode error.
        NotSupportedModeError = NPP_NOT_SUPPORTED_MODE_ERROR,
        /// Application and NPP library version mismatch.
        LibraryVersionMismatchError = NPP_LIBRARY_VERSION_MISMATCH_ERROR,
        /// Invalid host memory pointer error.
        InvalidHostPointerError = NPP_INVALID_HOST_POINTER_ERROR,
        /// Invalid device memory pointer error.
        InvalidDevicePointerError = NPP_INVALID_DEVICE_POINTER_ERROR,
        /// Color look up table bitsize error.
        LutPaletteBitsizeError = NPP_LUT_PALETTE_BITSIZE_ERROR,
        /// ZeroCrossing mode not supported error.
        ZeroCrossingModeNotSupportedError = NPP_ZC_MODE_NOT_SUPPORTED_ERROR,
        /// Not sufficient CUDA compute capability error.
        NotSufficientComputeCapability = NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY,
        /// Texture bind error.
        TextureBindError = NPP_TEXTURE_BIND_ERROR,
        /// Wrong intersection region of interest error.
        WrongIntersectionRoiError = NPP_WRONG_INTERSECTION_ROI_ERROR,
        /// Haar classifier pixel match error.
        HaarClassifierPixelMatchError = NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR,
        /// Memory free request error.
        MemfreeError = NPP_MEMFREE_ERROR,
        /// Memory set request error.
        MemsetError = NPP_MEMSET_ERROR,
        /// Memory copy request error.
        MemcpyError = NPP_MEMCPY_ERROR,
        /// Memory alignment error.
        AlignmentError = NPP_ALIGNMENT_ERROR,
        /// CUDA kernel execution error, most commonly CUDA kernel launch error.
        CudaKernelExecutionError = NPP_CUDA_KERNEL_EXECUTION_ERROR,
        /// Non CTX function support has been deprecated.
        StreamContextError = NPP_STREAM_CTX_ERROR,
        /// Unsupported round mode.
        RoundModeNotSupportedError = NPP_ROUND_MODE_NOT_SUPPORTED_ERROR,
        /// Image pixels are constant for quality index.
        QualityIndexError = NPP_QUALITY_INDEX_ERROR,
        /// One of the output image dimensions is less than 1 pixel.
        ResizeNoOperationError = NPP_RESIZE_NO_OPERATION_ERROR,
        /// Number overflows the upper or lower limit of the data type.
        OverflowError = NPP_OVERFLOW_ERROR,
        /// Step value is not pixel multiple.
        NotEvenStepError = NPP_NOT_EVEN_STEP_ERROR,
        /// Number of levels for histogram is less than 2.
        HistogramNumberOfLevelsError = NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR,
        /// Number of levels for LUT is less than 2.
        LutNumberOfLevelsError = NPP_LUT_NUMBER_OF_LEVELS_ERROR,
        /// Processed data is corrupted.
        CorruptedDataError = NPP_CORRUPTED_DATA_ERROR,
        /// Wrong order of the destination channels.
        ChannelOrderError = NPP_CHANNEL_ORDER_ERROR,
        /// All values of the mask are zero.
        ZeroMaskValueError = NPP_ZERO_MASK_VALUE_ERROR,
        /// The quadrangle is nonconvex or degenerates into triangle, line or point.
        QuadrangleError = NPP_QUADRANGLE_ERROR,
        /// Size of the rectangle region is less than or equal to 1.
        RectangleError = NPP_RECTANGLE_ERROR,
        /// Unallowable values of the transformation coefficients.
        CoefficientError = NPP_COEFFICIENT_ERROR,
        /// Bad or unsupported number of channels.
        NumberOfChannelsError = NPP_NUMBER_OF_CHANNELS_ERROR,
        /// Channel of interest is not 1, 2, or 3.
        ChannelOfInterestError = NPP_COI_ERROR,
        /// Divisor is equal to zero.
        DivisorError = NPP_DIVISOR_ERROR,
        /// Illegal channel index.
        ChannelError = NPP_CHANNEL_ERROR,
        /// Stride is less than the row length.
        StrideError = NPP_STRIDE_ERROR,
        /// A resize scale factor is invalid.
        ResizeFactorError = NPP_RESIZE_FACTOR_ERROR,
        /// The interpolation mode is invalid or unsupported for this operation.
        InterpolationError = NPP_INTERPOLATION_ERROR,
        /// The requested mirror or flip direction is invalid.
        MirrorFlipError = NPP_MIRROR_FLIP_ERROR,
        /// An argument is invalid for the requested operation.
        BadArgumentError = NPP_BAD_ARGUMENT_ERROR,
        /// The operation has insufficient memory.
        NoMemoryError = NPP_NO_MEMORY_ERROR,
        /// The requested operation is not implemented.
        NotImplementedError = NPP_NOT_IMPLEMENTED_ERROR,
        /// An unspecified NPP error occurred.
        Error = NPP_ERROR,
        /// Reserved native NPP error status.
        ErrorReserved = NPP_ERROR_RESERVED,
        /// A required pointer argument is null.
        NullPointerError = NPP_NULL_POINTER_ERROR,
        /// A value range is invalid.
        RangeError = NPP_RANGE_ERROR,
        /// A size argument is invalid.
        SizeError = NPP_SIZE_ERROR,
        /// Step is less or equal zero.
        StepError = NPP_STEP_ERROR,
        /// Anchor point is outside mask.
        AnchorError = NPP_ANCHOR_ERROR,
        /// Lower bound is larger than upper bound.
        MaskSizeError = NPP_MASK_SIZE_ERROR,
        /// The zero-order image moment is zero, so normalization by that moment is undefined.
        Moment00ZeroError = NPP_MOMENT_00_ZERO_ERROR,
        /// The threshold level is negative where a nonnegative level is required.
        ThresholdNegativeLevelError = NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR,
        /// The threshold arguments are invalid.
        ThresholdError = NPP_THRESHOLD_ERROR,
        /// The supplied execution context does not match the required CUDA context.
        ContextMatchError = NPP_CONTEXT_MATCH_ERROR,
        /// The FFT flags are invalid.
        FftFlagError = NPP_FFT_FLAG_ERROR,
        /// The FFT order is outside the supported range.
        FftOrderError = NPP_FFT_ORDER_ERROR,
        /// The scaling exponent is outside the supported range.
        ScaleRangeError = NPP_SCALE_RANGE_ERROR,
        /// Data type is incorrect or not supported.
        DataTypeError = NPP_DATA_TYPE_ERROR,
        /// An argument or computed value is outside the permitted range.
        OutOfRangeError = NPP_OUT_OFF_RANGE_ERROR,
        /// A zero divisor prevents the operation from completing.
        DivideByZeroError = NPP_DIVIDE_BY_ZERO_ERROR,
        /// An allocation required by the operation failed.
        MemoryAllocationErr = NPP_MEMORY_ALLOCATION_ERR,
        /// Indicates that no operation was performed.
        NoOperationWarning = NPP_NO_OPERATION_WARNING,
        /// Divisor is zero however does not terminate the execution.
        DivideByZeroWarning = NPP_DIVIDE_BY_ZERO_WARNING,
        /// Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties.
        ///
        /// First 3 vertices are used, the fourth vertex discarded.
        AffineQuadIncorrectWarning = NPP_AFFINE_QUAD_INCORRECT_WARNING,
        /// The given ROI has no intersection with either the source or destination ROI.
        ///
        /// Thus no operation was performed.
        WrongIntersectionRoiWarning = NPP_WRONG_INTERSECTION_ROI_WARNING,
        /// The given quadrangle has no intersection with either the source or destination ROI.
        ///
        /// Thus no operation was performed.
        WrongIntersectionQuadWarning = NPP_WRONG_INTERSECTION_QUAD_WARNING,
        /// Image size isn't multiple of two.
        ///
        /// Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.
        DoubleSizeWarning = NPP_DOUBLE_SIZE_WARNING,
        /// Speed reduction due to uncoalesced memory accesses warning.
        MisalignedDstRoiWarning = NPP_MISALIGNED_DST_ROI_WARNING,
    }
}

impl Status {
    /// Static description of this status.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::NotSupportedModeError => "not supported mode error",
            Self::LibraryVersionMismatchError => "library version mismatch error",
            Self::InvalidHostPointerError => "invalid host pointer error",
            Self::InvalidDevicePointerError => "invalid device pointer error",
            Self::LutPaletteBitsizeError => "lut palette bitsize error",
            Self::ZeroCrossingModeNotSupportedError => "zc mode not supported error",
            Self::NotSufficientComputeCapability => "not sufficient compute capability",
            Self::TextureBindError => "texture bind error",
            Self::WrongIntersectionRoiError => "wrong intersection roi error",
            Self::HaarClassifierPixelMatchError => "haar classifier pixel match error",
            Self::MemfreeError => "memfree error",
            Self::MemsetError => "memset error",
            Self::MemcpyError => "memcpy error",
            Self::AlignmentError => "alignment error",
            Self::CudaKernelExecutionError => "cuda kernel execution error",
            Self::StreamContextError => "stream ctx error",
            Self::RoundModeNotSupportedError => "round mode not supported error",
            Self::QualityIndexError => "quality index error",
            Self::ResizeNoOperationError => "resize no operation error",
            Self::OverflowError => "overflow error",
            Self::NotEvenStepError => "not even step error",
            Self::HistogramNumberOfLevelsError => "histogram number of levels error",
            Self::LutNumberOfLevelsError => "lut number of levels error",
            Self::CorruptedDataError => "corrupted data error",
            Self::ChannelOrderError => "channel order error",
            Self::ZeroMaskValueError => "zero mask value error",
            Self::QuadrangleError => "quadrangle error",
            Self::RectangleError => "rectangle error",
            Self::CoefficientError => "coefficient error",
            Self::NumberOfChannelsError => "number of channels error",
            Self::ChannelOfInterestError => "coi error",
            Self::DivisorError => "divisor error",
            Self::ChannelError => "channel error",
            Self::StrideError => "stride error",
            Self::ResizeFactorError => "resize factor error",
            Self::InterpolationError => "interpolation error",
            Self::MirrorFlipError => "mirror flip error",
            Self::BadArgumentError => "bad argument error",
            Self::NoMemoryError => "no memory error",
            Self::NotImplementedError => "not implemented error",
            Self::Error => "error",
            Self::ErrorReserved => "error reserved",
            Self::NullPointerError => "null pointer error",
            Self::RangeError => "range error",
            Self::SizeError => "size error",
            Self::StepError => "step error",
            Self::AnchorError => "anchor error",
            Self::MaskSizeError => "mask size error",
            Self::Moment00ZeroError => "moment 00 zero error",
            Self::ThresholdNegativeLevelError => "threshold negative level error",
            Self::ThresholdError => "threshold error",
            Self::ContextMatchError => "context match error",
            Self::FftFlagError => "fft flag error",
            Self::FftOrderError => "fft order error",
            Self::ScaleRangeError => "scale range error",
            Self::DataTypeError => "data type error",
            Self::OutOfRangeError => "out off range error",
            Self::DivideByZeroError => "divide by zero error",
            Self::MemoryAllocationErr => "memory allocation err",
            Self::NoOperationWarning => "no operation warning",
            Self::DivideByZeroWarning => "divide by zero warning",
            Self::AffineQuadIncorrectWarning => "affine quad incorrect warning",
            Self::WrongIntersectionRoiWarning => "wrong intersection roi warning",
            Self::WrongIntersectionQuadWarning => "wrong intersection quad warning",
            Self::DoubleSizeWarning => "double size warning",
            Self::MisalignedDstRoiWarning => "misaligned dst roi warning",
            Self::Unknown(_) => "unknown npp error",
        }
    }
}

/// Errors from NPP operations, CUDA resources, and wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Another owner retains a resource that must be consumed exclusively.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,

    /// An error propagated from the CUDA wrapper.
    #[error("cuda error: {0}")]
    Cuda(
        /// Underlying CUDA error.
        #[from]
        CudaError,
    ),

    /// A native NPP status represented as a wrapper error.
    ///
    /// `try_ffi!` reports positive warning statuses here as well as negative native errors.
    #[error("npp error ({code}): {message}")]
    Npp {
        /// Typed native status, retaining unknown numeric codes.
        code: Status,
        /// Human-readable description associated with the status.
        message: String,
    },

    /// A string intended for a C API contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native operation returned an unexpected null handle or pointer.
    #[error("unexpected null handle")]
    NullHandle,

    /// A named parameter or computed value violates the required range or format.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
    },

    /// A named value is zero or negative where a positive value is required.
    #[error("`{name}` must be greater than 0")]
    ValueNotPositive {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
    },

    /// A length, capacity, or numeric bound differs from the required value.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
        /// Required length, capacity, or bound reported by the operation.
        expected: usize,
        /// Observed length, capacity, or value.
        actual: usize,
    },

    /// A two-dimensional extent differs from the required extent.
    #[error("`{name}` size mismatch (expected {expected}, got {actual})")]
    SizeMismatch {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
        /// Expected two-dimensional extent.
        expected: Size,
        /// Observed two-dimensional extent.
        actual: Size,
    },

    /// A stream or memory region belongs to a different CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// The selected operation is unavailable for these inputs.
    #[error("unsupported operation: `{name}`")]
    UnsupportedOperation {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
    },

    /// A required named parameter was not supplied.
    #[error("missing required parameter: `{name}`")]
    MissingParameter {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
    },
}

/// Result returned by NPP wrapper and pipeline operations.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::NppStatus> for Error {
    fn from(status: sys::NppStatus) -> Self {
        let code = Status::from(status);
        Self::from(code)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Npp {
            code,
            message: code.description().to_string(),
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

/// Converts a raw NPP status-returning expression into a crate [`Result`](crate::error::Result).
///
/// Evaluates the expression once. Only `NPP_SUCCESS` produces `Ok(())`; every other status, including positive NPP warnings, becomes an `Error::Npp` result.
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_npp_sys::NppStatus::NPP_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
