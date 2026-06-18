#![allow(deprecated)]

use std::{ffi::NulError, fmt};

use singe_cuda::error::Error as CudaError;
use singe_npp_sys as sys;
use thiserror::Error;

use crate::types::Size;

macro_rules! npp_statuses {
    ($($variant:ident => $raw:ident,)*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Status {
            $($variant,)*
            Unknown(i32),
        }

        impl Status {
            pub fn description(self) -> String {
                match sys::NppStatus::try_from(self.raw()) {
                    Ok(status) => format!("{status:?}")
                        .trim_start_matches("NPP_")
                        .replace('_', " ")
                        .to_ascii_lowercase(),
                    Err(_) => String::from("unknown npp error"),
                }
            }

            pub const fn raw(self) -> i32 {
                match self {
                    $(Self::$variant => sys::NppStatus::$raw as _,)*
                    Self::Unknown(code) => code,
                }
            }
        }

        impl TryFrom<i32> for Status {
            type Error = i32;

            fn try_from(code: i32) -> std::result::Result<Self, i32> {
                match code {
                    $(code if code == sys::NppStatus::$raw as i32 => Ok(Self::$variant),)*
                    code => Err(code),
                }
            }
        }

        impl From<sys::NppStatus> for Status {
            fn from(status: sys::NppStatus) -> Self {
                Self::try_from(status as i32).unwrap_or_else(Self::Unknown)
            }
        }

        impl TryFrom<Status> for sys::NppStatus {
            type Error = Status;

            fn try_from(status: Status) -> std::result::Result<Self, Status> {
                match status {
                    $(Status::$variant => Ok(Self::$raw),)*
                    Status::Unknown(_) => Err(status),
                }
            }
        }

        impl fmt::Display for Status {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Self::$variant => f.write_str(stringify!($raw)),)*
                    Self::Unknown(code) => write!(f, "UNKNOWN_NPP_STATUS({code})"),
                }
            }
        }
    };
}

npp_statuses! {
    Success => NPP_SUCCESS,
    NotSupportedModeError => NPP_NOT_SUPPORTED_MODE_ERROR,
    LibraryVersionMismatchError => NPP_LIBRARY_VERSION_MISMATCH_ERROR,
    InvalidHostPointerError => NPP_INVALID_HOST_POINTER_ERROR,
    InvalidDevicePointerError => NPP_INVALID_DEVICE_POINTER_ERROR,
    LutPaletteBitsizeError => NPP_LUT_PALETTE_BITSIZE_ERROR,
    ZeroCrossingModeNotSupportedError => NPP_ZC_MODE_NOT_SUPPORTED_ERROR,
    NotSufficientComputeCapability => NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY,
    TextureBindError => NPP_TEXTURE_BIND_ERROR,
    WrongIntersectionRoiError => NPP_WRONG_INTERSECTION_ROI_ERROR,
    HaarClassifierPixelMatchError => NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR,
    MemfreeError => NPP_MEMFREE_ERROR,
    MemsetError => NPP_MEMSET_ERROR,
    MemcpyError => NPP_MEMCPY_ERROR,
    AlignmentError => NPP_ALIGNMENT_ERROR,
    CudaKernelExecutionError => NPP_CUDA_KERNEL_EXECUTION_ERROR,
    StreamContextError => NPP_STREAM_CTX_ERROR,
    RoundModeNotSupportedError => NPP_ROUND_MODE_NOT_SUPPORTED_ERROR,
    QualityIndexError => NPP_QUALITY_INDEX_ERROR,
    ResizeNoOperationError => NPP_RESIZE_NO_OPERATION_ERROR,
    OverflowError => NPP_OVERFLOW_ERROR,
    NotEvenStepError => NPP_NOT_EVEN_STEP_ERROR,
    HistogramNumberOfLevelsError => NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR,
    LutNumberOfLevelsError => NPP_LUT_NUMBER_OF_LEVELS_ERROR,
    CorruptedDataError => NPP_CORRUPTED_DATA_ERROR,
    ChannelOrderError => NPP_CHANNEL_ORDER_ERROR,
    ZeroMaskValueError => NPP_ZERO_MASK_VALUE_ERROR,
    QuadrangleError => NPP_QUADRANGLE_ERROR,
    RectangleError => NPP_RECTANGLE_ERROR,
    CoefficientError => NPP_COEFFICIENT_ERROR,
    NumberOfChannelsError => NPP_NUMBER_OF_CHANNELS_ERROR,
    ChannelOfInterestError => NPP_COI_ERROR,
    DivisorError => NPP_DIVISOR_ERROR,
    ChannelError => NPP_CHANNEL_ERROR,
    StrideError => NPP_STRIDE_ERROR,
    ResizeFactorError => NPP_RESIZE_FACTOR_ERROR,
    InterpolationError => NPP_INTERPOLATION_ERROR,
    MirrorFlipError => NPP_MIRROR_FLIP_ERROR,
    BadArgumentError => NPP_BAD_ARGUMENT_ERROR,
    NoMemoryError => NPP_NO_MEMORY_ERROR,
    NotImplementedError => NPP_NOT_IMPLEMENTED_ERROR,
    Error => NPP_ERROR,
    ErrorReserved => NPP_ERROR_RESERVED,
    NullPointerError => NPP_NULL_POINTER_ERROR,
    RangeError => NPP_RANGE_ERROR,
    SizeError => NPP_SIZE_ERROR,
    StepError => NPP_STEP_ERROR,
    AnchorError => NPP_ANCHOR_ERROR,
    MaskSizeError => NPP_MASK_SIZE_ERROR,
    Moment00ZeroError => NPP_MOMENT_00_ZERO_ERROR,
    ThresholdNegativeLevelError => NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR,
    ThresholdError => NPP_THRESHOLD_ERROR,
    ContextMatchError => NPP_CONTEXT_MATCH_ERROR,
    FftFlagError => NPP_FFT_FLAG_ERROR,
    FftOrderError => NPP_FFT_ORDER_ERROR,
    ScaleRangeError => NPP_SCALE_RANGE_ERROR,
    DataTypeError => NPP_DATA_TYPE_ERROR,
    OutOfRangeError => NPP_OUT_OFF_RANGE_ERROR,
    DivideByZeroError => NPP_DIVIDE_BY_ZERO_ERROR,
    MemoryAllocationErr => NPP_MEMORY_ALLOCATION_ERR,
    NoOperationWarning => NPP_NO_OPERATION_WARNING,
    DivideByZeroWarning => NPP_DIVIDE_BY_ZERO_WARNING,
    AffineQuadIncorrectWarning => NPP_AFFINE_QUAD_INCORRECT_WARNING,
    WrongIntersectionRoiWarning => NPP_WRONG_INTERSECTION_ROI_WARNING,
    WrongIntersectionQuadWarning => NPP_WRONG_INTERSECTION_QUAD_WARNING,
    DoubleSizeWarning => NPP_DOUBLE_SIZE_WARNING,
    MisalignedDstRoiWarning => NPP_MISALIGNED_DST_ROI_WARNING,
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("npp error ({code}): {message}")]
    Npp { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("`{name}` must be greater than 0")]
    ValueNotPositive { name: String },

    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("`{name}` size mismatch (expected {expected}, got {actual})")]
    SizeMismatch {
        name: String,
        expected: Size,
        actual: Size,
    },

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("unsupported operation: `{name}`")]
    UnsupportedOperation { name: String },

    #[error("missing required parameter: `{name}`")]
    MissingParameter { name: String },
}

pub type Result<T> = std::result::Result<T, Error>;

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
            message: code.description(),
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

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
