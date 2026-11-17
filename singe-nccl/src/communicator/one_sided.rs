use std::{ops::Range, ptr};

use singe_cuda::stream::Stream;

use crate::{
    communicator::{Communicator, Window},
    error::{Error, Result},
    sys, try_ffi,
    utility::{to_i32, to_usize},
};

/// Describes how many signal operations to wait for from a particular rank on a given signal index and context.
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
        // SAFETY: the window borrows its communicator, so both handles are live, the context is bound, and `pointer` is a local output slot.
        unsafe {
            try_ffi!(sys::ncclWinGetUserPtr(
                self.communicator.as_raw(),
                self.raw,
                &raw mut pointer
            ))?;
        }
        Ok(pointer.cast())
    }

    /// Enqueues a one-sided write from this window to a peer's registered window, followed by a signal, without the peer's participation.
    ///
    /// `source` is a byte range in this registered window; `offset` is the byte offset in the peer's corresponding registration of `destination`.
    ///
    /// # Safety
    ///
    /// The remote window must cover the destination range.
    /// The caller must coordinate remote access, keep both windows registered until all peers finish, and keep signal and context indices within the communicator's configured limits.
    /// CUDA graph executions using these windows must finish before deregistration.
    ///
    /// # Errors
    ///
    /// Returns an error if the windows belong to different communicators, the source byte range exceeds the local window, a byte offset overflows, or the peer, signal, or context index is invalid.
    /// Communicator-state checks, stream ownership, CUDA context binding, and NCCL transfer errors are propagated.
    pub unsafe fn put_signal_async(
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
            return Err(Error::ContextMismatch {
                name: "destination window".into(),
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
        let start = to_usize(source.start, "window offset")?;
        // SAFETY: `source` was checked within `byte_len`, so `pointer.add(start)` stays in the registered window, both windows share this live communicator, and `ensure_stream` matched the stream context.
        // The caller guarantees the remote range and the signal and context limits.
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

    /// Queues a signal to the specified peer on `stream` without transferring data, and returns before it completes.
    ///
    /// # Safety
    ///
    /// `signal` and `context` must be within the limits configured on every rank.
    /// The caller must order this signal with the peer's matching wait and keep the communicator alive until the signal completes, including in CUDA graph replays.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer, signal, or context index is negative or the peer is out of range, the communicator is not active, the stream belongs to another CUDA context, CUDA context binding fails, or NCCL rejects the signal.
    pub unsafe fn signal_async(
        &self,
        peer: i32,
        signal: i32,
        context: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.validate_signal(peer, signal, context)?;
        // SAFETY: `ncclSignal` receives the live communicator and a stream checked to share its context.
        // The caller guarantees the signal and context indices are configured.
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

    /// Queues a wait on `stream` for the signals the descriptors describe, and returns before they arrive.
    /// Each descriptor indicates how many signals to expect from a specific peer on a particular signal index and context.
    ///
    /// # Safety
    ///
    /// Every signal and context index must be within the limits configured on every rank.
    /// The peers must eventually send the expected signals, or work queued on `stream` after the wait never runs.
    /// Keep the communicator alive until the wait completes, including in CUDA graph replays.
    ///
    /// # Errors
    ///
    /// Returns an error if a wait count is not positive, a peer, signal, or context index is invalid, or the number of waits exceeds the native integer range.
    /// Communicator-state checks, stream ownership, CUDA context binding, and NCCL wait errors are propagated.
    pub unsafe fn wait_signals_async(&self, signals: &[WaitSignal], stream: &Stream) -> Result<()> {
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
        // SAFETY: `ncclWaitSignal` receives the live communicator, a stream checked to share its context, and `count` descriptors that outlive the call.
        // The caller guarantees the signal and context indices are configured.
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
