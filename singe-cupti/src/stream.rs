use singe_cuda::stream::Stream;
use singe_cupti_sys as sys;

use crate::{context::Context, error::Result, try_ffi, types::StreamId};

impl Context {
    /// Returns the context-local CUPTI ID for a CUDA stream.
    ///
    /// Set `per_thread_stream` to match whether the program uses CUDA per-thread default streams.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidStream`] if the stream is invalid or does not belong to the context.
    pub fn stream_id(&self, stream: &Stream, per_thread_stream: bool) -> Result<StreamId> {
        let mut stream_id = 0;
        unsafe {
            try_ffi!(sys::cuptiGetStreamIdEx(
                self.as_raw(),
                stream.as_raw() as sys::CUstream,
                per_thread_stream as u8,
                &mut stream_id,
            ))?;
        }
        Ok(StreamId::from(u64::from(stream_id)))
    }

    // /// Returns the context-local CUPTI ID for a CUDA stream.
    // ///
    // /// This wraps CUPTI's deprecated stream ID query. Prefer [`Context::stream_id_extended`] when per-thread default stream behavior matters.
    // ///
    // /// # Errors
    // ///
    // /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    // /// - Returns [`crate::error::Status::InvalidStream`] if the stream is invalid or does not belong to the context.
    // #[deprecated]
    // pub fn stream_id(&self, stream: &Stream) -> Result<StreamId> {
    //     let mut stream_id = 0;
    //     unsafe {
    //         try_ffi!(sys::cuptiGetStreamId(
    //             self.as_raw(),
    //             stream.as_raw() as sys::CUstream,
    //             &mut stream_id,
    //         ))?;
    //     }
    //     Ok(StreamId::from(u64::from(stream_id)))
    // }
}
