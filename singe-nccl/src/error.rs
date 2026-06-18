use std::ffi;

use singe_cuda::error::Error as CudaError;
use singe_nccl_sys as sys;
use thiserror::Error;

use crate::types::Status;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("nccl error ({code}): {message}")]
    Nccl { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("unexpected null pointer")]
    NullPointer,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("`{name}` length mismatch")]
    LengthMismatch { name: String },

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("communicator is in {state} state")]
    InvalidCommunicatorState { state: String },

    #[error("communicator builder mode does not support {operation}")]
    InvalidBuilderMode { operation: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<sys::ncclResult_t> for Error {
    fn from(code: sys::ncclResult_t) -> Self {
        Self::from(Status::from(code))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nccl {
            code: status,
            message: status.description(),
        }
    }
}

impl From<ffi::NulError> for Error {
    fn from(_: ffi::NulError) -> Self {
        Self::InteriorNul
    }
}
