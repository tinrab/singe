#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::error::Error as CudaError;
use singe_npp_sys as sys;
use thiserror::Error;

use crate::types::Size;

singe_core::wrapper_status! {
    library = "npp",
    error = Error::Npp,
    raw = sys::NppStatus,
    success = NPP_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NPP_STATUS",
    /// Status returned by NPP calls.
    ///
    /// NPP statuses are signed: negative values are errors and positive values are warnings, so raw codes are `i32` rather than the `u32` used by other wrappers.
    pub enum Status: i32 {
        /// The native call succeeded.
        Success = NPP_SUCCESS => "success",
        /// Not supported mode error.
        NotSupportedModeError = NPP_NOT_SUPPORTED_MODE_ERROR => "not supported mode error",
        /// Application and NPP library version mismatch.
        LibraryVersionMismatchError = NPP_LIBRARY_VERSION_MISMATCH_ERROR => "library version mismatch error",
        /// Invalid host memory pointer error.
        InvalidHostPointerError = NPP_INVALID_HOST_POINTER_ERROR => "invalid host pointer error",
        /// Invalid device memory pointer error.
        InvalidDevicePointerError = NPP_INVALID_DEVICE_POINTER_ERROR => "invalid device pointer error",
        /// Color look up table bitsize error.
        LutPaletteBitsizeError = NPP_LUT_PALETTE_BITSIZE_ERROR => "lut palette bitsize error",
        /// ZeroCrossing mode not supported error.
        ZeroCrossingModeNotSupportedError = NPP_ZC_MODE_NOT_SUPPORTED_ERROR => "zc mode not supported error",
        /// Not sufficient CUDA compute capability error.
        NotSufficientComputeCapability = NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY => "not sufficient compute capability",
        /// Texture bind error.
        TextureBindError = NPP_TEXTURE_BIND_ERROR => "texture bind error",
        /// Wrong intersection region of interest error.
        WrongIntersectionRoiError = NPP_WRONG_INTERSECTION_ROI_ERROR => "wrong intersection roi error",
        /// Haar classifier pixel match error.
        HaarClassifierPixelMatchError = NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR => "haar classifier pixel match error",
        /// Memory free request error.
        MemfreeError = NPP_MEMFREE_ERROR => "memfree error",
        /// Memory set request error.
        MemsetError = NPP_MEMSET_ERROR => "memset error",
        /// Memory copy request error.
        MemcpyError = NPP_MEMCPY_ERROR => "memcpy error",
        /// Memory alignment error.
        AlignmentError = NPP_ALIGNMENT_ERROR => "alignment error",
        /// CUDA kernel execution error, most commonly CUDA kernel launch error.
        CudaKernelExecutionError = NPP_CUDA_KERNEL_EXECUTION_ERROR => "cuda kernel execution error",
        /// Non CTX function support has been deprecated.
        StreamContextError = NPP_STREAM_CTX_ERROR => "stream ctx error",
        /// Unsupported round mode.
        RoundModeNotSupportedError = NPP_ROUND_MODE_NOT_SUPPORTED_ERROR => "round mode not supported error",
        /// Image pixels are constant for quality index.
        QualityIndexError = NPP_QUALITY_INDEX_ERROR => "quality index error",
        /// One of the output image dimensions is less than 1 pixel.
        ResizeNoOperationError = NPP_RESIZE_NO_OPERATION_ERROR => "resize no operation error",
        /// Number overflows the upper or lower limit of the data type.
        OverflowError = NPP_OVERFLOW_ERROR => "overflow error",
        /// Step value is not pixel multiple.
        NotEvenStepError = NPP_NOT_EVEN_STEP_ERROR => "not even step error",
        /// Number of levels for histogram is less than 2.
        HistogramNumberOfLevelsError = NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR => "histogram number of levels error",
        /// Number of levels for LUT is less than 2.
        LutNumberOfLevelsError = NPP_LUT_NUMBER_OF_LEVELS_ERROR => "lut number of levels error",
        /// Processed data is corrupted.
        CorruptedDataError = NPP_CORRUPTED_DATA_ERROR => "corrupted data error",
        /// Wrong order of the destination channels.
        ChannelOrderError = NPP_CHANNEL_ORDER_ERROR => "channel order error",
        /// All values of the mask are zero.
        ZeroMaskValueError = NPP_ZERO_MASK_VALUE_ERROR => "zero mask value error",
        /// The quadrangle is nonconvex or degenerates into triangle, line or point.
        QuadrangleError = NPP_QUADRANGLE_ERROR => "quadrangle error",
        /// Size of the rectangle region is less than or equal to 1.
        RectangleError = NPP_RECTANGLE_ERROR => "rectangle error",
        /// Unallowable values of the transformation coefficients.
        CoefficientError = NPP_COEFFICIENT_ERROR => "coefficient error",
        /// Bad or unsupported number of channels.
        NumberOfChannelsError = NPP_NUMBER_OF_CHANNELS_ERROR => "number of channels error",
        /// Channel of interest is not 1, 2, or 3.
        ChannelOfInterestError = NPP_COI_ERROR => "coi error",
        /// Divisor is equal to zero.
        DivisorError = NPP_DIVISOR_ERROR => "divisor error",
        /// Illegal channel index.
        ChannelError = NPP_CHANNEL_ERROR => "channel error",
        /// Stride is less than the row length.
        StrideError = NPP_STRIDE_ERROR => "stride error",
        /// A resize scale factor is invalid.
        ResizeFactorError = NPP_RESIZE_FACTOR_ERROR => "resize factor error",
        /// The interpolation mode is invalid or unsupported for this operation.
        InterpolationError = NPP_INTERPOLATION_ERROR => "interpolation error",
        /// The requested mirror or flip direction is invalid.
        MirrorFlipError = NPP_MIRROR_FLIP_ERROR => "mirror flip error",
        /// An argument is invalid for the requested operation.
        BadArgumentError = NPP_BAD_ARGUMENT_ERROR => "bad argument error",
        /// The operation has insufficient memory.
        NoMemoryError = NPP_NO_MEMORY_ERROR => "no memory error",
        /// The requested operation is not implemented.
        NotImplementedError = NPP_NOT_IMPLEMENTED_ERROR => "not implemented error",
        /// An unspecified NPP error occurred.
        Error = NPP_ERROR => "error",
        /// Reserved native NPP error status.
        ErrorReserved = NPP_ERROR_RESERVED => "error reserved",
        /// A required pointer argument is null.
        NullPointerError = NPP_NULL_POINTER_ERROR => "null pointer error",
        /// A value range is invalid.
        RangeError = NPP_RANGE_ERROR => "range error",
        /// A size argument is invalid.
        SizeError = NPP_SIZE_ERROR => "size error",
        /// Step is less or equal zero.
        StepError = NPP_STEP_ERROR => "step error",
        /// Anchor point is outside mask.
        AnchorError = NPP_ANCHOR_ERROR => "anchor error",
        /// Lower bound is larger than upper bound.
        MaskSizeError = NPP_MASK_SIZE_ERROR => "mask size error",
        /// The zero-order image moment is zero, so normalization by that moment is undefined.
        Moment00ZeroError = NPP_MOMENT_00_ZERO_ERROR => "moment 00 zero error",
        /// The threshold level is negative where a nonnegative level is required.
        ThresholdNegativeLevelError = NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR => "threshold negative level error",
        /// The threshold arguments are invalid.
        ThresholdError = NPP_THRESHOLD_ERROR => "threshold error",
        /// The supplied execution context does not match the required CUDA context.
        ContextMatchError = NPP_CONTEXT_MATCH_ERROR => "context match error",
        /// The FFT flags are invalid.
        FftFlagError = NPP_FFT_FLAG_ERROR => "fft flag error",
        /// The FFT order is outside the supported range.
        FftOrderError = NPP_FFT_ORDER_ERROR => "fft order error",
        /// The scaling exponent is outside the supported range.
        ScaleRangeError = NPP_SCALE_RANGE_ERROR => "scale range error",
        /// Data type is incorrect or not supported.
        DataTypeError = NPP_DATA_TYPE_ERROR => "data type error",
        /// An argument or computed value is outside the permitted range.
        OutOfRangeError = NPP_OUT_OFF_RANGE_ERROR => "out off range error",
        /// A zero divisor prevents the operation from completing.
        DivideByZeroError = NPP_DIVIDE_BY_ZERO_ERROR => "divide by zero error",
        /// An allocation required by the operation failed.
        MemoryAllocationErr = NPP_MEMORY_ALLOCATION_ERR => "memory allocation err",
        /// Indicates that no operation was performed.
        NoOperationWarning = NPP_NO_OPERATION_WARNING => "no operation warning",
        /// Divisor is zero however does not terminate the execution.
        DivideByZeroWarning = NPP_DIVIDE_BY_ZERO_WARNING => "divide by zero warning",
        /// Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties.
        ///
        /// First 3 vertices are used, the fourth vertex discarded.
        AffineQuadIncorrectWarning = NPP_AFFINE_QUAD_INCORRECT_WARNING => "affine quad incorrect warning",
        /// The given ROI has no intersection with either the source or destination ROI.
        ///
        /// Thus no operation was performed.
        WrongIntersectionRoiWarning = NPP_WRONG_INTERSECTION_ROI_WARNING => "wrong intersection roi warning",
        /// The given quadrangle has no intersection with either the source or destination ROI.
        ///
        /// Thus no operation was performed.
        WrongIntersectionQuadWarning = NPP_WRONG_INTERSECTION_QUAD_WARNING => "wrong intersection quad warning",
        /// Image size isn't multiple of two.
        ///
        /// Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.
        DoubleSizeWarning = NPP_DOUBLE_SIZE_WARNING => "double size warning",
        /// Speed reduction due to uncoalesced memory accesses warning.
        MisalignedDstRoiWarning = NPP_MISALIGNED_DST_ROI_WARNING => "misaligned dst roi warning",
    }
}

/// Errors from NPP operations, CUDA resources, and wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// NPP or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// Another owner retains a resource that must be consumed exclusively.
    #[error("`{name}` is still shared")]
    OwnershipShared {
        /// Name of the shared resource.
        name: Cow<'static, str>,
    },

    /// An error propagated from the CUDA wrapper.
    #[error(transparent)]
    Cuda(
        /// Underlying CUDA error.
        #[from]
        CudaError,
    ),

    /// A native NPP status represented as a wrapper error.
    ///
    /// `try_ffi!` reports positive warning statuses here as well as negative native errors.
    #[error("npp error ({code}): {}", code.description())]
    Npp {
        /// Typed native status, retaining unknown numeric codes.
        code: Status,
    },

    /// A string intended for a C API contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native operation returned an unexpected null handle or pointer.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A named parameter or computed value violates the required range or format.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the parameter, resource, or quantity involved.
        name: Cow<'static, str>,
    },

    /// A named value is zero or negative where a positive value is required.
    #[error("`{name}` must be positive")]
    NotPositive {
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

impl From<singe_core::UnknownEnumValue> for Error {
    fn from(error: singe_core::UnknownEnumValue) -> Self {
        Self::UnknownEnumValue {
            name: error.name.into(),
            value: error.value,
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

// Converts a raw NPP status into this crate's `Result`.
// Only `NPP_SUCCESS` is `Ok(())`; positive NPP warnings become `Error::Npp` too.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
