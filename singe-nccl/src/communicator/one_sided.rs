use std::{ops::Range, ptr};

use singe_cuda::stream::Stream;

use crate::{
    communicator::{Communicator, Window},
    error::{Error, Result},
    sys, try_ffi,
    utility::to_i32,
};

/// Describes how many signal operations to wait for
/// from a particular rank on a given signal index and context.
#[derive(Debug, Clone, Copy)]
pub struct WaitSignal {
    /// Number of signal operations to wait for.
    pub count: i32,
    /// Target peer to wait for signals from.
    pub peer: i32,
    /// Signal index identifier.
    pub signal: i32,
    /// Context identifier.
    pub context: i32,
}

impl Window<'_> {
    /// Returns the registered window size, in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Returns the user pointer associated with this window.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// NCCL window-pointer query errors are propagated.
    pub fn user_pointer(&self) -> Result<*mut u8> {
        self.communicator.bind()?;
        let mut pointer = ptr::null_mut();
        // SAFETY: `ncclWinGetUserPtr` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclWinGetUserPtr(
                self.communicator.as_raw(),
                self.raw,
                &raw mut pointer
            ))?;
        }
        Ok(pointer.cast())
    }

    /// One-sided communication operation that writes data from the local buffer to a
    /// remote peer's registered memory window without explicit participation from the
    /// target process.
    ///
    /// `source` is a byte range in this registered window; `offset` is the byte offset
    /// in the peer's corresponding registration of `destination`.
    ///
    /// # Safety
    /// The remote window must cover the destination range. The caller must coordinate
    /// remote access, keep both windows registered until all peers finish, and keep
    /// signal and context indices within the communicator's configured limits.
    /// CUDA graph executions using these windows must finish before deregistration.
    ///
    /// # Errors
    ///
    /// Returns an error if the windows belong to different communicators, the source byte range exceeds the local window, a byte offset overflows, or the peer, signal, or context index is invalid.
    /// Communicator-state checks, stream ownership, CUDA context binding, and NCCL transfer errors are propagated.
    pub unsafe fn put_signal(
        &self,
        source: Range<u64>,
        destination: &Window<'_>,
        peer: i32,
        offset: u64,
        signal: i32,
        context: i32,
        stream: &Stream,
    ) -> Result<()> {
        if !ptr::eq(self.communicator, destination.communicator) {
            return Err(Error::LengthMismatch {
                name: "window communicator".into(),
            });
        }
        let end_out_of_bounds = source.end > self.byte_len;
        if source.start > source.end
            || end_out_of_bounds
            || offset.checked_add(source.end - source.start).is_none()
        {
            return Err(Error::out_of_range("window byte range"));
        }
        self.communicator.validate_signal(peer, signal, context)?;
        self.communicator.ensure_stream(stream)?;
        let start =
            usize::try_from(source.start).map_err(|_| Error::out_of_range("window offset"))?;
        // SAFETY: `ncclPutSignal` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclPutSignal(
                self.pointer.add(start).cast(),
                source.end - source.start,
                sys::ncclDataType_t::ncclUint8,
                peer,
                destination.raw,
                offset,
                signal,
                context,
                0,
                self.communicator.as_raw(),
                stream.as_raw()
            ))?;
        }
        Ok(())
    }
}

impl Communicator {
    fn validate_signal(&self, peer: i32, signal: i32, context: i32) -> Result<()> {
        if peer < 0 || peer >= self.rank_count()? || signal < 0 || context < 0 {
            return Err(Error::out_of_range("signal descriptor"));
        }
        Ok(())
    }

    /// Sends a signal to the specified peer without transferring data.
    /// `signal` and `context` must refer to resources configured on every rank.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer, signal, or context index is invalid, the communicator is not active, the stream belongs to another CUDA context, CUDA context binding fails, or NCCL rejects the signal.
    pub fn signal(&self, peer: i32, signal: i32, context: i32, stream: &Stream) -> Result<()> {
        self.ensure_stream(stream)?;
        self.validate_signal(peer, signal, context)?;
        // SAFETY: `ncclSignal` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::ncclSignal(
                peer,
                signal,
                context,
                0,
                self.as_raw(),
                stream.as_raw()
            ))?;
        }
        Ok(())
    }

    /// Waits for signals as described in the signal descriptor array.
    /// Each descriptor indicates how many signals to expect from
    /// a specific peer on a particular signal index and context.
    ///
    /// # Errors
    ///
    /// Returns an error if a wait count is not positive, a peer, signal, or context index is invalid, or the number of waits exceeds the native integer range.
    /// Communicator-state checks, stream ownership, CUDA context binding, and NCCL wait errors are propagated.
    pub fn wait_signals(&self, signals: &[WaitSignal], stream: &Stream) -> Result<()> {
        self.ensure_stream(stream)?;
        let count = to_i32(signals.len(), "signal descriptors")?;
        let mut descriptors = signals
            .iter()
            .map(|signal| {
                self.validate_signal(signal.peer, signal.signal, signal.context)?;
                if signal.count <= 0 {
                    return Err(Error::out_of_range("signal count"));
                }
                Ok(sys::ncclWaitSignalDesc_t {
                    opCnt: signal.count,
                    peer: signal.peer,
                    sigIdx: signal.signal,
                    ctx: signal.context,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: `ncclWaitSignal` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclWaitSignal(
                count,
                descriptors.as_mut_ptr(),
                self.as_raw(),
                stream.as_raw()
            ))?;
        }
        Ok(())
    }
}
