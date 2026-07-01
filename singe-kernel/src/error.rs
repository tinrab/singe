use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid length")]
    InvalidLength,
    #[error("length mismatch")]
    LengthMismatch,
    #[error("length mismatch for {name}: expected {expected}, actual {actual}")]
    LengthMismatchFor {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("length too short for {name}: minimum {minimum}, actual {actual}")]
    LengthTooShort {
        name: String,
        minimum: usize,
        actual: usize,
    },
    #[error("rank reach mismatch for {name}: rank {rank}, required {required}, actual {actual}")]
    RankReachMismatch {
        name: String,
        rank: usize,
        required: usize,
        actual: usize,
    },
    #[error("length exceeds i32")]
    LengthExceedsI32,
    #[error("null pointer")]
    NullPointer,
    #[error("size overflow")]
    SizeOverflow,
    #[error("unsupported fft size for {transform}: n = {n}")]
    UnsupportedFftSize { transform: String, n: usize },
    #[error("unsupported axis for {op}: axis = {axis}")]
    UnsupportedAxis { op: String, axis: usize },
    #[error("unsupported width for {op}: width = {width}")]
    UnsupportedWidth { op: String, width: usize },
    #[error("unsupported kernel size for {op}: kernel_size = {kernel_size}")]
    UnsupportedKernelSize { op: String, kernel_size: usize },
    #[error("unsupported block count for {op}: blocks = {blocks}")]
    UnsupportedBlockCount { op: String, blocks: usize },
    #[error("missing parameter for {op}: {parameter}")]
    MissingParameter { op: String, parameter: String },
    #[error("unsupported parameter for {op}: {parameter} = {value}")]
    UnsupportedParameter {
        op: String,
        parameter: String,
        value: usize,
    },
    #[error("unsupported configuration for {op}: {reason}")]
    UnsupportedConfiguration { op: String, reason: String },

    #[cfg(feature = "cuda_13_3")]
    #[error("cuda error: {0}")]
    Cuda(#[from] singe_cuda::error::Error),

    #[cfg(feature = "cutile")]
    #[error("cutile error: {0}")]
    Cutile(#[from] cutile::error::Error),
    #[cfg(feature = "cutile")]
    #[error("cutile device error: {0}")]
    CutileDevice(#[from] cutile::prelude::DeviceError),
    #[cfg(feature = "cutile")]
    #[error("cutile driver error: {0}")]
    CutileDriver(#[from] cuda_core::DriverError),
}

pub type Result<T> = std::result::Result<T, Error>;
