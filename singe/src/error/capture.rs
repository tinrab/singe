use thiserror::Error;

/// Errors from recording eager operations into a graph.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CaptureError {
    /// The capture session was still borrowed when exclusive access was required.
    #[error("capture session still borrowed")]
    SessionStillBorrowed,

    /// A capture session is already active on this thread.
    #[error("capture already active")]
    AlreadyActive,

    /// Capture produced no output tensors.
    #[error("missing capture output")]
    MissingOutput,

    /// No capture session is active on this thread.
    #[error("capture not active")]
    NotActive,

    /// A trace consumed a tensor without a recorded or explicit binding.
    #[error("unrecorded capture tensor; register an input, state, or constant binding")]
    UnrecordedTensor,

    /// Host data observation would silently specialize the trace.
    #[error("host readback of a tracked capture value requires value specialization")]
    Readback,
}
