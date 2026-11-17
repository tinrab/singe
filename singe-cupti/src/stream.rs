use singe_cuda::stream::Stream;
use singe_cupti_sys as sys;

use crate::{context::Context, error::Result, try_ffi, types::StreamId};

/// Which stream the null stream handle names in the traced program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DefaultStream {
    /// The legacy default stream, which synchronizes with other blocking streams.
    #[default]
    Legacy,
    /// The per-thread default stream, used by programs compiled with `--default-stream per-thread`.
    PerThread,
}

impl Context {
    /// Returns the context-local CUPTI ID for a CUDA stream.
    ///
    /// `default_stream` must match how the program treats the null stream.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidStream`] if the stream is invalid or does not belong to the context.
    pub fn stream_id(&self, stream: &Stream, default_stream: DefaultStream) -> Result<StreamId> {
        let mut stream_id = 0;
        // SAFETY: this context and `stream` keep their native handles alive for the call, and `stream_id` is a valid output location.
        unsafe {
            try_ffi!(sys::cuptiGetStreamIdEx(
                self.as_raw(),
                stream.as_raw() as sys::CUstream,
                u8::from(default_stream == DefaultStream::PerThread),
                &raw mut stream_id,
            ))?;
        }
        Ok(StreamId::from(u64::from(stream_id)))
    }
}
