use std::{
    fmt::{self, Display, Formatter},
    sync::Arc,
};

use thiserror::Error;

use crate::{
    ast::Span,
    diagnostic::SourceDiagnostic,
    emulator::{CompileError, ExecutionReport},
};

/// Why the emulator did not accept a command.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum SubmitError {
    /// The emulator is shutting down or closed.
    #[error("device is closed")]
    Closed,
    /// An earlier fault stopped the emulator. Downloads are still accepted.
    #[error("device is faulted")]
    Faulted,
    /// [`EmulatorOptions::queued_commands`](crate::emulator::EmulatorOptions::queued_commands)
    /// commands are already waiting.
    #[error("command queue is full")]
    QueueFull,
    /// The command needs more staging storage than remains.
    #[error("staging storage limit exceeded")]
    StagingLimit,
    /// The command needs more execution storage than remains.
    #[error("execution-state storage limit exceeded")]
    ExecutionStateLimit,
    /// A buffer, stream, event, or kernel belongs to another emulator.
    #[error("resource belongs to another emulator")]
    ForeignResource,
    /// An argument is invalid, such as a launch argument that does not match its
    /// parameter.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The module layout or kernel failed validation.
    #[error(transparent)]
    Compile(#[from] CompileError),
}

/// Where a lane was executing when it faulted or stopped at a breakpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultLocation {
    /// Identifier of the launch.
    pub launch_id: u64,
    /// Stream the launch ran on.
    pub stream_id: u64,
    /// Source byte range of the instruction.
    pub span: Span,
    /// Name of the function containing the instruction.
    pub function: String,
    /// CTA coordinates in the grid.
    pub cta: [u32; 3],
    /// Thread coordinates in the CTA.
    pub lane: [u32; 3],
    /// Index of the instruction within its function.
    pub instruction: usize,
    /// Function names from the entry point to the current function.
    pub call_stack: Vec<String>,
}

impl Display for FaultLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [cx, cy, cz] = self.cta;
        let [lx, ly, lz] = self.lane;
        write!(
            f,
            "`{}` instruction {} (bytes {}..{}), cta ({cx}, {cy}, {cz}), lane ({lx}, {ly}, {lz})",
            self.function, self.instruction, self.span.start, self.span.end
        )
    }
}

/// A lane that was waiting when no lane could make progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StalledLane {
    /// CTA coordinates in the grid.
    pub cta: [u32; 3],
    /// Thread coordinates in the CTA.
    pub lane: [u32; 3],
}

/// Why an accepted command failed.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ExecutionError {
    /// The submission's result was already taken.
    #[error("submission result already consumed")]
    ResultConsumed,
    /// A launch failed after it started.
    #[error("{cause}")]
    Launch {
        /// Counts for the part of the launch that ran.
        report: Arc<ExecutionReport>,
        /// The failure.
        #[source]
        cause: Arc<Self>,
    },
    /// A cancelling shutdown stopped the command.
    #[error("command cancelled")]
    Cancelled,
    /// The worker thread panicked or stopped before the command finished.
    #[error("worker failed")]
    WorkerFailed,
    /// An earlier fault stopped the emulator before this command ran.
    #[error("device is faulted")]
    Faulted,
    /// The allocation exceeds the device storage limit, or host memory could not be
    /// reserved.
    #[error("memory limit exceeded")]
    MemoryLimit,
    /// The address window of a state space (named in the error) has no free range left.
    #[error("{0} address space exhausted")]
    AddressSpaceExhausted(&'static str),
    /// The launch needs more execution storage than remains.
    #[error("execution-state storage limit exceeded")]
    ExecutionStateLimit,
    /// The launch reached
    /// [`EmulatorOptions::launch_instructions`](crate::emulator::EmulatorOptions::launch_instructions).
    #[error("instruction budget exhausted")]
    InstructionLimit,
    /// A lane exceeded
    /// [`EmulatorOptions::call_depth`](crate::emulator::EmulatorOptions::call_depth).
    #[error("call depth exhausted")]
    CallDepth,
    /// A memory access was out of bounds, misaligned, or used the wrong state space.
    #[error("invalid memory access: {0}")]
    Memory(String),
    /// A lane read a register or memory that holds no defined value.
    #[error("indeterminate read: {0}")]
    Indeterminate(String),
    /// A lane executed `trap`.
    #[error("trap instruction")]
    Trap,
    /// The kernel broke a PTX execution rule, such as an out-of-range barrier number
    /// or a `ret` from a `.noreturn` function.
    #[error("invalid execution: {0}")]
    Invalid(String),
    /// Every remaining lane is waiting at a barrier or collective that cannot complete.
    #[error("scheduler stalled with {} waiting lanes", .0.len())]
    Stall(Vec<StalledLane>),
    /// A lane failure with the location where it happened.
    #[error("{cause} at {location}")]
    Lane {
        /// Where the lane was executing.
        location: Arc<FaultLocation>,
        /// The failure.
        cause: Arc<Self>,
    },
    /// The kernel failed validation.
    #[error(transparent)]
    Compile(#[from] CompileError),
}

impl ExecutionError {
    /// Returns a display of this error that quotes the source line of the failing
    /// instruction. Errors without a source location display normally.
    #[must_use]
    pub fn with_source<'a>(&'a self, source: &'a str) -> SourceDiagnostic<'a> {
        let span = match self {
            Self::Launch { cause, .. } => return cause.with_source(source),
            Self::Lane { location, .. } => Some(location.span),
            Self::Compile(error) => Some(error.span),
            _ => None,
        };
        SourceDiagnostic::new(self, source, span)
    }

    /// Returns the report of the failed launch, if this error comes from a launch.
    #[must_use]
    pub fn launch_report(&self) -> Option<&ExecutionReport> {
        match self {
            Self::Launch { report, .. } => Some(report),
            _ => None,
        }
    }

    /// Returns the underlying error, skipping the launch and lane wrappers.
    #[must_use]
    pub fn root_cause(&self) -> &Self {
        match self {
            Self::Launch { cause, .. } | Self::Lane { cause, .. } => cause.root_cause(),
            _ => self,
        }
    }
}
