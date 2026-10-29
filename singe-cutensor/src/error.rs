use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda::{data_type::DataType, error::Error as CudaError};
#[cfg(feature = "mp")]
use singe_nccl::error::Error as NcclError;
use thiserror::Error;

use crate::{
    sys,
    types::{OperationDescriptorAttribute, PlanAttribute, PlanPreferenceAttribute},
};

status_enum! {
    /// cuTENSOR status returned by library calls.
    pub enum Status: u32 => sys::cutensorStatus_t, unknown = "UNKNOWN_CUTENSOR_STATUS" {
        /// The operation completed successfully.
        Success = CUTENSOR_STATUS_SUCCESS,
        /// The opaque data structure was not initialized.
        NotInitialized = CUTENSOR_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed inside the cuTENSOR library.
        AllocFailed = CUTENSOR_STATUS_ALLOC_FAILED,
        /// An unsupported value or parameter was passed to the operation.
        InvalidValue = CUTENSOR_STATUS_INVALID_VALUE,
        /// The device is not ready, or the target architecture is not supported.
        ArchitectureMismatch = CUTENSOR_STATUS_ARCH_MISMATCH,
        /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        MappingError = CUTENSOR_STATUS_MAPPING_ERROR,
        /// The GPU program failed to execute.
        /// A kernel launch failure on the GPU is a common cause.
        ExecutionFailed = CUTENSOR_STATUS_EXECUTION_FAILED,
        /// An internal cuTENSOR error occurred.
        InternalError = CUTENSOR_STATUS_INTERNAL_ERROR,
        /// The requested operation is not supported.
        NotSupported = CUTENSOR_STATUS_NOT_SUPPORTED,
        /// The requested operation requires a license, and an error was detected
        /// when checking the current licensing.
        LicenseError = CUTENSOR_STATUS_LICENSE_ERROR,
        /// A call to cuBLAS did not succeed.
        CublasError = CUTENSOR_STATUS_CUBLAS_ERROR,
        /// An unknown CUDA error occurred.
        CudaError = CUTENSOR_STATUS_CUDA_ERROR,
        /// The provided workspace was insufficient.
        InsufficientWorkspace = CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE,
        /// Indicates that the driver version is insufficient.
        InsufficientDriver = CUTENSOR_STATUS_INSUFFICIENT_DRIVER,
        /// Indicates an error related to file I/O.
        IoError = CUTENSOR_STATUS_IO_ERROR,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::cutensorStatus_t::try_from(self) else {
            return "unknown cutensor error";
        };
        // SAFETY: `cutensorGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::cutensorGetErrorString(raw), "unknown cutensor error") }
    }
}

/// Errors from cuTENSOR, CUDA, NCCL, and checked wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// An NCCL operation used by multi-process execution failed.
    #[cfg(feature = "mp")]
    #[error("nccl error: {0}")]
    Nccl(#[from] NcclError),

    /// A native cuTENSOR call returned an error status.
    #[error("cutensor error ({code}): {message}")]
    Cutensor {
        /// Native status returned by the failed call.
        code: Status,
        /// Description of the native status.
        message: String,
    },

    /// A string passed to a native API contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A dependent resource still retains ownership required to be unique.
    #[error("resource is shared and cannot be consumed")]
    OwnershipShared,

    /// A native library or CUDA runtime version is invalid.
    #[error("invalid `{name}` version")]
    InvalidVersion {
        /// Name of the invalid argument, resource, or operation.
        name: &'static str,
    },

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// Related slices have different lengths.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
        /// Required number of entries.
        expected: usize,
        /// Observed number of entries.
        actual: usize,
    },

    /// A tensor's mode-label count does not equal its rank.
    #[error("tensor mode length mismatch for tensor rank {rank} (got {mode_length})")]
    TensorModeMismatch {
        /// Required tensor rank or layout entry count.
        rank: u32,
        /// Number of supplied mode labels.
        mode_length: usize,
    },

    /// A tensor's stride count does not match the required layout.
    #[error("tensor stride length mismatch for tensor rank {rank} (got {stride_length})")]
    TensorStrideMismatch {
        /// Required tensor rank or layout entry count.
        rank: u32,
        /// Number of supplied strides.
        stride_length: usize,
    },

    /// Tensor descriptors that must agree describe different layouts.
    #[error("tensor descriptors must match for `{name}`")]
    DescriptorMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// Tensor mode labels that must agree do not match.
    #[error("tensor modes must match for `{name}`")]
    ModeMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// The tensor descriptor and sample allocation have different data types.
    #[error("tensor data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    TensorMemoryDataTypeMismatch {
        /// Data type required by the descriptor.
        descriptor: DataType,
        /// Data type of the supplied memory or scalar.
        memory: DataType,
    },

    /// The operation's scalar type and supplied scalar representation differ.
    #[error("scalar data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    ScalarDataTypeMismatch {
        /// Data type required by the descriptor.
        descriptor: DataType,
        /// Data type of the supplied memory or scalar.
        memory: DataType,
    },

    /// The operation does not support the supplied scalar data type.
    #[error("unsupported scalar data type {data_type} for `{name}`")]
    UnsupportedScalarDataType {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
        /// Unsupported scalar data type.
        data_type: DataType,
    },

    /// The number of supplied pointers differs from the required count.
    #[error("`{name}` pointer count mismatch (expected {expected}, got {actual})")]
    PointerCountMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
        /// Required number of entries.
        expected: usize,
        /// Observed number of entries.
        actual: usize,
    },

    /// The supplied workspace is smaller than the native byte requirement.
    #[error("insufficient workspace size: {actual} bytes provided (required {required} bytes)")]
    InsufficientWorkspaceSize {
        /// Required workspace capacity in bytes.
        required: u64,
        /// Supplied workspace capacity in bytes.
        actual: u64,
    },

    /// The workspace size does not satisfy its required byte alignment.
    #[error("workspace size is not aligned to {required_alignment} bytes (size is {size} bytes)")]
    WorkspaceMisaligned {
        /// Required workspace-size alignment in bytes.
        required_alignment: u32,
        /// Supplied workspace size in bytes.
        size: usize,
    },

    /// The stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Resources belong to different single-GPU cuTENSOR handles.
    #[error("cutensor contexts must match for `{name}`")]
    ContextMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// The plan was created for a different operation kind.
    #[error("plan operation kind mismatch (expected {expected}, got {actual})")]
    PlanOperationMismatch {
        /// Required operation kind.
        expected: String,
        /// Observed operation kind.
        actual: String,
    },

    /// Resources belong to different multi-GPU cuTENSOR handles.
    #[error("cutensormg contexts must match for `{name}`")]
    MgContextMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// A device does not participate in the multi-GPU context.
    #[error("device {device} does not participate in the cutensormg context")]
    MgDeviceNotInContext {
        /// Device ordinal absent from the multi-GPU context.
        device: i32,
    },

    /// An operation does not support multi-GPU tensor storage on the host.
    #[error("cutensormg host tensor memory is not supported by `{name}`")]
    MgHostTensorMemoryUnsupported {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// Resources belong to different multi-process cuTENSOR handles.
    #[error("cutensormp contexts must match for `{name}`")]
    MpContextMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// A plan attribute buffer has the wrong byte size.
    #[error("invalid plan attribute size for {attr:?}: {actual} (expected {expected})")]
    PlanInvalidAttributeSize {
        /// Attribute whose buffer size is invalid.
        attr: PlanAttribute,
        /// Required byte count.
        expected: usize,
        /// Observed byte count.
        actual: usize,
    },

    /// An operation attribute buffer has the wrong byte size.
    #[error(
        "invalid operation descriptor attribute size for {attr:?}: {actual} (expected {expected})"
    )]
    OperationDescriptorInvalidAttributeSize {
        /// Attribute whose buffer size is invalid.
        attr: OperationDescriptorAttribute,
        /// Required byte count.
        expected: usize,
        /// Observed byte count.
        actual: usize,
    },

    /// A plan-preference attribute buffer has the wrong byte size.
    #[error("invalid plan preference attribute size for {attr:?}: {actual} (expected {expected})")]
    PlanPreferenceInvalidAttributeSize {
        /// Attribute whose buffer size is invalid.
        attr: PlanPreferenceAttribute,
        /// Required byte count.
        expected: usize,
        /// Observed byte count.
        actual: usize,
    },
}

/// Result of a cuTENSOR wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl From<sys::cutensorStatus_t> for Error {
    fn from(status: sys::cutensorStatus_t) -> Self {
        let code = Status::from(status);
        Self::from(code)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cutensor {
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

/// Converts a raw cuTENSOR status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cutensor_sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::cutensorStatus_t::CUTENSOR_STATUS_INVALID_VALUE);
        assert_eq!(status, Status::InvalidValue);
        assert_eq!(
            status.raw(),
            sys::cutensorStatus_t::CUTENSOR_STATUS_INVALID_VALUE as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidValue.to_string(),
            "CUTENSOR_STATUS_INVALID_VALUE"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_CUTENSOR_STATUS(9999)");
    }
}
