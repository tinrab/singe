use std::io::Error as IoError;

#[cfg(feature = "cuda")]
use cutile_ir::Error as TileIrError;
use singe_codegen::{emit::EmitError, interpret::InterpretError, schedule::ScheduleError};
#[cfg(feature = "cuda")]
use singe_cuda::error::Error as CudaError;
use thiserror::Error;

/// Errors from generating, compiling, tuning, and caching kernels.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CompileError {
    /// An emitter could not write code for a generated kernel.
    #[error("kernel emission")]
    Emit(#[from] EmitError),

    /// A schedule cannot launch a generated kernel.
    #[error("kernel schedule")]
    Schedule(#[from] ScheduleError),

    /// The CPU interpreter could not run a generated kernel, as for an integer division by zero.
    #[error("kernel interpretation")]
    Interpret(#[from] InterpretError),

    /// Tile IR bytecode for a generated kernel could not be written.
    #[cfg(feature = "cuda")]
    #[error("tile ir bytecode")]
    TileBytecode(#[from] TileIrError),

    /// NVRTC rejected a generated kernel's CUDA C++; `log` is its compilation log.
    #[cfg(feature = "cuda")]
    #[error("nvrtc failed to compile `{kernel}`: {log}")]
    Nvrtc {
        kernel: String,
        log: String,
        #[source]
        source: Box<CudaError>,
    },

    /// `tileiras` rejected a generated kernel's Tile IR.
    #[error("tileiras failed to compile `{kernel}`: {message}")]
    Tileiras { kernel: String, message: String },

    /// `tileiras` could not run, or its input or output files could not be written or read.
    #[error("tileiras io")]
    TileirasIo(#[source] IoError),

    /// A file in the generated-kernel cache directory could not be read or written.
    #[error("kernel cache io")]
    KernelCacheIo(#[source] IoError),

    /// A tuning file could not be read or written.
    #[error("tuning file io")]
    TuningIo(#[source] IoError),

    /// A tuning file is malformed.
    #[error("tuning file format")]
    TuningFormat(#[from] serde_json::Error),
}
