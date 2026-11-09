mod capture;
mod checkpoint;
mod compile;

use std::{
    borrow::Cow,
    collections::TryReserveError,
    fmt::{self, Display, Formatter},
    io::Error as IoError,
    result::Result as StdResult,
};

pub use singe_core::ErrorChain;

pub use self::{capture::CaptureError, checkpoint::CheckpointError, compile::CompileError};
#[cfg(feature = "cuda")]
use cutile_ir::Error as TileIrError;
use safetensors::SafeTensorError;
use singe_codegen::{emit::EmitError, interpret::InterpretError, schedule::ScheduleError};
#[cfg(feature = "cuda")]
use singe_cublas::error::Error as CublasError;
#[cfg(feature = "cuda")]
use singe_cuda::error::Error as CudaError;
#[cfg(feature = "cuda")]
use singe_cudnn::error::Error as CudnnError;
#[cfg(feature = "cuda")]
use singe_cufft::error::Error as CufftError;
#[cfg(feature = "cuda")]
use singe_cutensor::error::Error as CutensorError;
use singe_kernel::error::Error as KernelError;
use thiserror::Error;

use crate::{data_type::DataType, device::Device};

/// Crate-wide result type.
pub type Result<T> = StdResult<T, Error>;

/// Errors returned by tensor, graph, module, checkpoint, storage, and backend APIs.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A graph operation failed with source format provenance.
    #[error("{location}")]
    GraphSource {
        /// Source-format location associated with the failed graph operation.
        location: String,
        /// Underlying tensor, graph, or backend error.
        #[source]
        source: Box<Self>,
    },
    /// Generating, compiling, tuning, or caching a kernel failed.
    #[error(transparent)]
    Compile(#[from] CompileError),

    /// Reading or writing a checkpoint failed.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    /// Recording eager operations into a graph failed.
    #[error(transparent)]
    Capture(#[from] CaptureError),

    /// A lock was poisoned by a thread that panicked while holding it.
    #[error("`{what}` lock poisoned by a panicked thread")]
    Poisoned {
        /// What the lock guards.
        what: Cow<'static, str>,
    },

    /// Host staging storage could not be reserved.
    #[error("host allocation failed")]
    HostAllocation(#[from] TryReserveError),

    /// Shape metadata did not match an expected shape.
    #[error("shape mismatch for `{name}`: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Name of the argument or value being validated.
        name: Cow<'static, str>,
        /// Expected dimensions.
        expected: Vec<i64>,
        /// Actual dimensions.
        actual: Vec<i64>,
    },

    /// Tensor layout metadata did not match an expected layout.
    #[error(
        "layout mismatch for `{name}`: expected dimensions {expected_dimensions:?} strides \
         {expected_strides:?} offset {expected_offset}, got dimensions {actual_dimensions:?} \
         strides {actual_strides:?} offset {actual_offset}"
    )]
    LayoutMismatch {
        /// Name of the argument or value being validated.
        name: Cow<'static, str>,
        /// Expected dimensions.
        expected_dimensions: Box<[i64]>,
        /// Expected strides.
        expected_strides: Box<[i64]>,
        /// Expected storage element offset.
        expected_offset: usize,
        /// Actual dimensions.
        actual_dimensions: Box<[i64]>,
        /// Actual strides.
        actual_strides: Box<[i64]>,
        /// Actual storage element offset.
        actual_offset: usize,
    },

    /// Tensor or storage data type did not match the expected type.
    #[error("data type mismatch for `{name}`: expected {expected}, got {actual}")]
    DataTypeMismatch {
        /// Name of the argument or value being validated.
        name: Cow<'static, str>,
        /// Expected data type.
        expected: DataType,
        /// Actual data type.
        actual: DataType,
    },

    /// Tensor or storage device did not match the expected device.
    #[error("device mismatch for `{name}`: expected {expected}, got {actual}")]
    DeviceMismatch {
        /// Name of the argument or value being validated.
        name: Cow<'static, str>,
        /// Expected device.
        expected: Device,
        /// Actual device.
        actual: Device,
    },

    /// Sequence or buffer length did not match the expected length.
    #[error("length mismatch for `{name}`: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Name of the argument or value being validated.
        name: Cow<'static, str>,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Dimension index was invalid for a tensor rank.
    #[error("invalid dimension for `{name}`: got {value}, rank is {rank}")]
    InvalidDimension {
        /// Name of the dimension argument.
        name: Cow<'static, str>,
        /// Supplied dimension value.
        value: i64,
        /// Tensor rank used for validation.
        rank: i64,
    },

    /// Shape metadata is invalid or would overflow internal calculations.
    #[error("invalid shape")]
    InvalidShape,

    /// Tensor rank did not satisfy an operation-specific requirement.
    #[error(
        "{operation} expects {expected_rank}D {layout} tensor for {name}, got rank \
         {actual_rank} shape {shape:?}"
    )]
    InvalidTensorRank {
        /// Operation being validated.
        operation: String,
        /// Name of the tensor argument.
        name: Cow<'static, str>,
        /// Expected tensor rank.
        expected_rank: usize,
        /// Actual tensor rank.
        actual_rank: usize,
        /// Required layout description.
        layout: String,
        /// Actual tensor dimensions.
        shape: Vec<i64>,
    },

    /// Stride metadata is invalid or would overflow address calculations.
    #[error("invalid stride")]
    InvalidStride,

    /// Named value is out of its supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the argument being validated.
        name: Cow<'static, str>,
    },

    /// Named argument is invalid.
    #[error("invalid argument for `{name}`")]
    InvalidArgument {
        /// Name of the invalid argument.
        name: Cow<'static, str>,
    },

    /// Byte buffer length is not a whole number of typed elements.
    #[error("byte length mismatch for `{name}`: expected multiple of {element_size}, got {actual}")]
    ByteLengthMismatch {
        /// Name of the buffer being validated.
        name: Cow<'static, str>,
        /// Size in bytes of one element.
        element_size: usize,
        /// Actual buffer length in bytes.
        actual: usize,
    },

    /// Byte buffer cannot be viewed as the requested aligned element type.
    #[error("unaligned byte buffer for `{name}`: prefix {prefix} bytes, suffix {suffix} bytes")]
    UnalignedByteBuffer {
        /// Name of the buffer being validated.
        name: Cow<'static, str>,
        /// Number of unaligned prefix bytes.
        prefix: usize,
        /// Number of unaligned suffix bytes.
        suffix: usize,
    },

    /// Operation, device, data type, or backend capability is not supported.
    #[error("unsupported {target}: {operation} ({details})")]
    Unsupported {
        /// Operation name.
        operation: Cow<'static, str>,
        /// Unsupported target category.
        target: UnsupportedTarget,
        /// Human-readable support diagnostics.
        details: Cow<'static, str>,
    },

    /// A library refused a step planning decided it runs, which nothing else then runs.
    #[error("{library} refused a step planning decided it runs")]
    LibraryRefused {
        /// The library that refused.
        library: String,
        /// What the library reported.
        #[source]
        source: Box<Self>,
    },

    /// A tuning record names parameters or a schedule its kernel no longer takes, as after a change to the kernel's candidates.
    #[error(
        "tuning records chose parameters for `{key}` that its kernel no longer takes; tune again, or plan without records"
    )]
    StaleRecord {
        /// The record's key.
        key: String,
    },

    /// A library refused the configuration tuning records chose, as it may on another GPU or library version.
    #[error(
        "{library} refused the configuration tuning records chose for `{key}`; tune again here, or plan without records"
    )]
    ConfigurationRefused {
        /// The library that refused.
        library: String,
        /// The record's key.
        key: String,
        /// What the library reported.
        #[source]
        source: Box<Self>,
    },

    /// Tensor index is out of bounds.
    #[error("index out of bounds")]
    IndexOutOfBounds,

    /// Tensor dimension is out of bounds.
    #[error("dimension out of bounds")]
    DimensionOutOfBounds,

    /// Shapes cannot be broadcast together.
    #[error("broadcast mismatch for `{name}`: left {left:?}, right {right:?}")]
    BroadcastMismatch {
        /// Name of the broadcast validation.
        name: Cow<'static, str>,
        /// Left shape dimensions.
        left: Vec<i64>,
        /// Right shape dimensions.
        right: Vec<i64>,
    },

    /// Operation requires contiguous tensor storage.
    #[error("contiguous required")]
    ContiguousRequired,

    /// CUDA allocation failed with contextual request metadata.
    #[error(
        "cuda allocation failed on {device}: requested {byte_count} bytes for data type \
         {data_type} and shape {shape:?}"
    )]
    AllocationFailed {
        /// Device where allocation was attempted.
        device: Device,
        /// Requested byte count.
        byte_count: usize,
        /// Requested element data type.
        data_type: DataType,
        /// Requested tensor shape when available.
        shape: Option<Vec<i64>>,
    },

    /// A CUDA allocation would exceed the device's memory limit even after releasing cached blocks.
    #[error(
        "cuda memory limit of {limit} bytes on {device} exceeded: requested {requested} bytes with {active} bytes active"
    )]
    MemoryLimitExceeded {
        /// Device whose limit was exceeded.
        device: Device,
        /// Configured limit in bytes.
        limit: usize,
        /// Requested byte count.
        requested: usize,
        /// Bytes held by live tensors.
        active: usize,
    },

    /// A graph node the operation in `context` looked up was missing.
    #[error("missing graph node in `{context}`")]
    MissingNode {
        /// The operation that found the graph node missing.
        context: Cow<'static, str>,
    },

    /// A output the operation in `context` looked up was missing.
    #[error("missing output in `{context}`")]
    MissingOutput {
        /// The operation that found the output missing.
        context: Cow<'static, str>,
    },

    /// A input the operation in `context` looked up was missing.
    #[error("missing input in `{context}`")]
    MissingInput {
        /// The operation that found the input missing.
        context: Cow<'static, str>,
    },

    /// Duplicate input name was supplied.
    #[error("duplicate input")]
    DuplicateInput,

    /// Input name was not declared by the graph.
    #[error("unexpected input `{0}`")]
    UnexpectedInput(String),

    /// Required concat input tensor was missing.
    #[error("missing concat input")]
    MissingConcatInput,

    /// Graph contains a cycle.
    #[error("cycle detected in graph")]
    GraphCycleDetected,

    /// Meta tensors describe values without holding them.
    #[error("meta tensors have no values to read or copy")]
    MetaTensorData,

    #[cfg(feature = "cuda")]
    /// CUDA driver/runtime wrapper error.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[cfg(feature = "cuda")]
    /// cuBLAS wrapper error.
    #[error(transparent)]
    Cublas(#[from] CublasError),

    #[cfg(feature = "cuda")]
    /// cuFFT wrapper error.
    #[error(transparent)]
    Cufft(#[from] CufftError),

    #[cfg(feature = "cuda")]
    /// cuDNN wrapper error.
    #[error(transparent)]
    Cudnn(#[from] CudnnError),

    #[cfg(feature = "cuda")]
    /// cuTENSOR wrapper error.
    #[error(transparent)]
    Cutensor(#[from] CutensorError),

    // A graph could not be exported as a program, for each of these reasons.
    #[error("cuda program export failed: {}", failures.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    Export { failures: Vec<ExportFailure> },

    /// A generated kernel failed.
    #[error("kernel")]
    Kernel(#[from] KernelError),
}

// Errors of the libraries behind `CompileError` and `CheckpointError` convert through them, so `?` reaches the crate error directly.
macro_rules! through {
    ($($(#[$attribute:meta])* $source:ty => $via:ident),+ $(,)?) => {
        $(
            $(#[$attribute])*
            impl From<$source> for Error {
                fn from(error: $source) -> Self {
                    Self::from($via::from(error))
                }
            }
        )+
    };
}

through!(
    EmitError => CompileError,
    ScheduleError => CompileError,
    InterpretError => CompileError,
    #[cfg(feature = "cuda")]
    TileIrError => CompileError,
    serde_json::Error => CompileError,
    IoError => CheckpointError,
    SafeTensorError => CheckpointError,
);

impl Error {
    /// Creates a byte length mismatch error.
    pub fn byte_length_mismatch(
        name: impl Into<Cow<'static, str>>,
        element_size: usize,
        actual: usize,
    ) -> Self {
        Self::ByteLengthMismatch {
            name: name.into(),
            element_size,
            actual,
        }
    }

    /// Creates an unaligned byte buffer error.
    pub fn unaligned_byte_buffer(
        name: impl Into<Cow<'static, str>>,
        prefix: usize,
        suffix: usize,
    ) -> Self {
        Self::UnalignedByteBuffer {
            name: name.into(),
            prefix,
            suffix,
        }
    }
}

impl Error {
    /// Creates a structured shape mismatch error.
    pub fn shape_mismatch(
        name: impl Into<Cow<'static, str>>,
        expected: impl Into<Vec<i64>>,
        actual: impl Into<Vec<i64>>,
    ) -> Self {
        Self::ShapeMismatch {
            name: name.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Creates a structured layout mismatch error.
    pub fn layout_mismatch(
        name: impl Into<Cow<'static, str>>,
        expected_dimensions: impl Into<Vec<i64>>,
        expected_strides: impl Into<Vec<i64>>,
        expected_offset: usize,
        actual_dimensions: impl Into<Vec<i64>>,
        actual_strides: impl Into<Vec<i64>>,
        actual_offset: usize,
    ) -> Self {
        Self::LayoutMismatch {
            name: name.into(),
            expected_dimensions: expected_dimensions.into().into_boxed_slice(),
            expected_strides: expected_strides.into().into_boxed_slice(),
            expected_offset,
            actual_dimensions: actual_dimensions.into().into_boxed_slice(),
            actual_strides: actual_strides.into().into_boxed_slice(),
            actual_offset,
        }
    }

    /// Creates a structured data type mismatch error.
    pub fn data_type_mismatch(
        name: impl Into<Cow<'static, str>>,
        expected: DataType,
        actual: DataType,
    ) -> Self {
        Self::DataTypeMismatch {
            name: name.into(),
            expected,
            actual,
        }
    }

    /// Creates a structured device mismatch error.
    pub fn device_mismatch(
        name: impl Into<Cow<'static, str>>,
        expected: Device,
        actual: Device,
    ) -> Self {
        Self::DeviceMismatch {
            name: name.into(),
            expected,
            actual,
        }
    }

    /// Creates a structured length mismatch error.
    pub fn length_mismatch(
        name: impl Into<Cow<'static, str>>,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self::LengthMismatch {
            name: name.into(),
            expected,
            actual,
        }
    }

    /// Creates a structured broadcast mismatch error.
    pub fn broadcast_mismatch(
        name: impl Into<Cow<'static, str>>,
        left: impl Into<Vec<i64>>,
        right: impl Into<Vec<i64>>,
    ) -> Self {
        Self::BroadcastMismatch {
            name: name.into(),
            left: left.into(),
            right: right.into(),
        }
    }
}

impl Error {
    // A refusal by `library` of a step planning decided it runs.
    #[cfg(feature = "cuda")]
    pub(crate) fn library_refused(library: &str, source: Self) -> Self {
        Self::LibraryRefused {
            library: library.to_owned(),
            source: Box::new(source),
        }
    }

    /// Creates a structured unsupported-operation error.
    pub fn unsupported_operation(operation: impl Into<Cow<'static, str>>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
            target: UnsupportedTarget::Operation,
            details: "unsupported operation".into(),
        }
    }

    /// Creates a structured unsupported-operation error with diagnostic details.
    pub fn unsupported_operation_with_details(
        operation: impl Into<Cow<'static, str>>,
        details: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::Unsupported {
            operation: operation.into(),
            target: UnsupportedTarget::Operation,
            details: details.into(),
        }
    }

    /// Creates a structured unsupported-device error.
    pub fn unsupported_device(details: impl Into<Cow<'static, str>>) -> Self {
        Self::Unsupported {
            operation: "device".into(),
            target: UnsupportedTarget::Device,
            details: details.into(),
        }
    }

    /// Creates a CUDA-disabled unsupported-device error.
    #[must_use]
    pub fn cuda_support_disabled() -> Self {
        Self::unsupported_device("cuda support is not enabled")
    }

    /// Creates a negative CUDA device id error.
    #[must_use]
    pub fn negative_cuda_device_id(id: i32) -> Self {
        Self::unsupported_device(format!("cuda device id {id} is negative"))
    }

    /// Creates a no-visible-CUDA-devices error.
    #[must_use]
    pub fn no_cuda_devices() -> Self {
        Self::unsupported_device("no cuda devices are available")
    }

    /// Creates an out-of-range CUDA device id error.
    #[must_use]
    pub fn cuda_device_out_of_range(id: i32, count: i32) -> Self {
        Self::unsupported_device(format!(
            "cuda device id {id} is out of range; visible device count is {count}"
        ))
    }

    /// Creates a structured unsupported-data-type error.
    pub fn unsupported_data_type(operation: impl Into<Cow<'static, str>>) -> Self {
        Self::Unsupported {
            operation: operation.into(),
            target: UnsupportedTarget::DataType,
            details: "unsupported data type".into(),
        }
    }
}

impl Error {
    /// Creates a structured invalid dimension error.
    pub fn invalid_dimension(name: impl Into<Cow<'static, str>>, value: i64, rank: i64) -> Self {
        Self::InvalidDimension {
            name: name.into(),
            value,
            rank,
        }
    }

    /// Creates a named invalid argument error.
    pub fn invalid_argument(name: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidArgument { name: name.into() }
    }

    pub fn missing_node(context: impl Into<Cow<'static, str>>) -> Self {
        Self::MissingNode {
            context: context.into(),
        }
    }

    pub fn missing_input(context: impl Into<Cow<'static, str>>) -> Self {
        Self::MissingInput {
            context: context.into(),
        }
    }

    pub fn missing_output(context: impl Into<Cow<'static, str>>) -> Self {
        Self::MissingOutput {
            context: context.into(),
        }
    }

    pub(crate) fn poisoned(what: impl Into<Cow<'static, str>>) -> Self {
        Self::Poisoned { what: what.into() }
    }

    /// Creates a named out-of-range error.
    pub fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

impl Error {
    pub(crate) fn with_graph_source(self, location: Option<&str>) -> Self {
        match location {
            Some(location) => Self::GraphSource {
                location: location.to_owned(),
                source: Box::new(self),
            },
            None => self,
        }
    }
}

/// Category of unsupported capability reported by Singe APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedTarget {
    /// Tensor element data type is unsupported.
    DataType,
    /// Tensor storage or execution device is unsupported.
    Device,
    /// Operation or operation lowering path is unsupported.
    Operation,
    /// Backend operation family is unavailable.
    BackendCapability,
}

impl Display for UnsupportedTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DataType => "data type",
            Self::Device => "device",
            Self::Operation => "operation",
            Self::BackendCapability => "backend capability",
        })
    }
}

// An operation a program could not place, and why.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("`{operation}`: {reason}")]
pub struct ExportFailure {
    pub operation: String,
    pub reason: ExportFailureReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExportFailureReason {
    #[error("no allowed provider runs it")]
    NoProvider,
    #[error("it needs compute capability {}.{}, which {target} lacks", minimum.0, minimum.1)]
    ComputeCapability { minimum: (u32, u32), target: String },
}
