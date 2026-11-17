//! Staged JPEG decoding with a chosen backend.

use std::{mem::ManuallyDrop, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    context::{Context, JpegState},
    decode::DecodeParams,
    error::{Error, Result},
    image::Image,
    stream::JpegStream,
    try_ffi,
    types::Backend,
};

/// Decoder for nvJPEG's staged decoding, tied to one backend.
///
/// Staged decoding splits decoding into a host stage, a transfer to the device, and a device stage, so the stages of different images can overlap.
/// The states it creates keep it alive.
#[derive(Debug)]
pub struct JpegDecoder {
    owner: Arc<DecoderOwner>,
}

/// Native staged decoder with its library handle, from [`JpegDecoder::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawJpegDecoder {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegJpegDecoder_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

// Shared with the states the decoder creates, which nvJPEG ties to the decoder.
#[derive(Debug)]
pub(crate) struct DecoderOwner {
    raw: sys::nvjpegJpegDecoder_t,
    handle: Context,
}

// SAFETY: nvJPEG's thread-safety rules require separate bitstream, state, and buffer handles per thread but not separate decoder handles, so the decoder may move between threads.
// The retained `Context` is `Send`, and the owner holds no other state.
unsafe impl Send for DecoderOwner {}

// SAFETY: the same rules let several threads use one decoder handle at once, and the owner has no interior mutability.
unsafe impl Sync for DecoderOwner {}

impl JpegDecoder {
    /// Takes ownership of a native decoder created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawJpegDecoder) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(DecoderOwner {
                raw: raw.handle,
                handle: raw.context,
            }),
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    ///
    /// # Errors
    ///
    /// Returns the decoder unchanged if a state created by it is still alive.
    pub fn into_raw(self) -> StdResult<RawJpegDecoder, Self> {
        match Arc::try_unwrap(self.owner) {
            Ok(owner) => {
                let owner = ManuallyDrop::new(owner);
                Ok(RawJpegDecoder {
                    handle: owner.raw,
                    // SAFETY: `owner` is never dropped, so its context moves out exactly once.
                    context: unsafe { ptr::read(&raw const owner.handle) },
                })
            }
            Err(owner) => Err(Self { owner }),
        }
    }

    pub(crate) fn handle(&self) -> &Context {
        &self.owner.handle
    }

    /// Returns the CUDA context of the decoder.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle().cuda_context()
    }

    /// Creates a decoder for `handle` that uses `backend`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the decoder.
    pub fn create(handle: &Context, backend: Backend) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `handle` owns a live library handle whose context was bound above, and `raw` is a local that receives the decoder.
        unsafe {
            try_ffi!(sys::nvjpegDecoderCreate(
                handle.as_raw(),
                backend.into(),
                &raw mut raw
            ))?;
        }
        // SAFETY: successful creation returns a decoder that only this wrapper owns.
        unsafe {
            Self::from_raw(RawJpegDecoder {
                handle: raw,
                context: handle.clone(),
            })
        }
    }

    /// Returns whether this decoder can decode the JPEG in `stream` with `params`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `stream` or `params` was created with another [`Context`], or nvJPEG fails the query.
    /// An unsupported JPEG returns `Ok(false)`, not an error.
    pub fn jpeg_supported(&self, stream: &JpegStream, params: &DecodeParams) -> Result<bool> {
        let handle = self.handle();
        handle.bind()?;
        handle.ensure_handle(&stream.handle, "stream")?;
        handle.ensure_handle(&params.handle, "params")?;

        let mut unsupported = 0;
        // SAFETY: the decoder, `stream`, and `params` own live handles of this context, as checked above, and `unsupported` is a local that receives the answer.
        unsafe {
            try_ffi!(sys::nvjpegDecoderJpegSupported(
                self.as_raw(),
                stream.as_raw(),
                params.as_raw(),
                &raw mut unsupported
            ))?;
        }
        Ok(unsupported == 0)
    }

    /// Creates a decoder state for this decoder's backend.
    ///
    /// The state keeps this decoder alive.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn create_state(&self) -> Result<JpegState> {
        let handle = self.handle();
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `handle` and the decoder own live handles whose context was bound above, and `raw` is a local that receives the state.
        unsafe {
            try_ffi!(sys::nvjpegDecoderStateCreate(
                handle.as_raw(),
                self.as_raw(),
                &raw mut raw
            ))?;
        }
        JpegState::from_created_raw(raw, handle, Some(Arc::clone(&self.owner)))
    }

    /// Returns the native handle, which this decoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvjpegJpegDecoder_t {
        self.owner.raw
    }

    pub(crate) fn ensure_state(&self, state: &JpegState) -> Result<()> {
        match &state.decoder {
            Some(owner) if Arc::ptr_eq(owner, &self.owner) => Ok(()),
            _ => Err(Error::ContextMismatch {
                name: "state".into(),
            }),
        }
    }
}

impl Context {
    /// Returns whether batched decoding can decode the JPEG in `stream`.
    ///
    /// Parse the header with [`JpegStream::parse_header`] first.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `stream` was created with another [`Context`], or nvJPEG fails the query.
    /// An unsupported JPEG returns `Ok(false)`, not an error.
    pub fn batched_decode_supported(&self, stream: &JpegStream) -> Result<bool> {
        self.bind()?;
        self.ensure_handle(&stream.handle, "stream")?;

        let mut unsupported = 0;
        // SAFETY: `self` and `stream` own live handles of this context, as checked above, and `unsupported` is a local that receives the answer.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedSupported(
                self.as_raw(),
                stream.as_raw(),
                &raw mut unsupported
            ))?;
        }
        Ok(unsupported == 0)
    }

    /// Returns whether batched decoding with `params`, such as a region, can decode the JPEG in `stream`.
    ///
    /// Parse the header with [`JpegStream::parse_header`] first.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `stream` or `params` was created with another [`Context`], or nvJPEG fails the query.
    /// An unsupported JPEG returns `Ok(false)`, not an error.
    pub fn batched_decode_supported_with_params(
        &self,
        stream: &JpegStream,
        params: &DecodeParams,
    ) -> Result<bool> {
        self.bind()?;
        self.ensure_handle(&stream.handle, "stream")?;
        self.ensure_handle(&params.handle, "params")?;

        let mut unsupported = 0;
        // SAFETY: `self`, `stream`, and `params` own live handles of this context, as checked above, and `unsupported` is a local that receives the answer.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedSupportedEx(
                self.as_raw(),
                stream.as_raw(),
                params.as_raw(),
                &raw mut unsupported
            ))?;
        }
        Ok(unsupported == 0)
    }

    /// Enqueues all three decoding stages for the JPEG in `stream` on `cuda_stream`, writing to `destination`.
    ///
    /// It is equivalent to [`decode_jpeg_host`](Self::decode_jpeg_host), [`decode_jpeg_transfer_to_device_async`](Self::decode_jpeg_transfer_to_device_async), and [`decode_jpeg_device_async`](Self::decode_jpeg_device_async) in one call.
    /// `stream` must hold a JPEG parsed with [`JpegStream::parse`].
    /// `state` must have a device buffer attached; without a pinned buffer, the host stage uses heap memory.
    ///
    /// # Safety
    ///
    /// `destination` must describe device-accessible buffers large enough for the image and `params`, valid until the work on `cuda_stream` finishes.
    /// `state` must have been created by `decoder`.
    /// Everything passed in must stay alive until the work finishes, even on failure or unwinding.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_stream` belongs to another CUDA context, the context cannot be bound, an argument was created with another [`Context`], or nvJPEG rejects or fails the decode.
    pub unsafe fn decode_jpeg_async(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        stream: &JpegStream,
        destination: &mut Image,
        params: &DecodeParams,
        cuda_stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(cuda_stream)?;
        self.ensure_handle(decoder.handle(), "decoder")?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_handle(&stream.handle, "stream")?;
        self.ensure_handle(&params.handle, "params")?;

        // SAFETY: every handle belongs to this context, as checked above; the caller guarantees `destination` is large enough and that the operands outlive the queued work.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpeg(
                self.as_raw(),
                decoder.as_raw(),
                state.as_raw(),
                stream.as_raw(),
                destination.as_raw_mut(),
                params.as_raw(),
                cuda_stream.as_raw()
            ))
        }
    }

    /// Runs the host stage of staged decoding for the JPEG in `stream`, using `params`.
    ///
    /// This stage runs on the CPU only and finishes before returning, so the GPU can be set up later.
    /// `stream` must hold a JPEG parsed with [`JpegStream::parse`].
    /// It allocates its memory from `state`'s pinned buffer if one is attached, and from heap memory otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, an argument was created with another [`Context`], `state` was not created by `decoder`, or nvJPEG rejects or fails the stage.
    pub fn decode_jpeg_host(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        params: &DecodeParams,
        stream: &JpegStream,
    ) -> Result<()> {
        self.bind()?;
        self.ensure_handle(decoder.handle(), "decoder")?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_handle(&params.handle, "params")?;
        self.ensure_handle(&stream.handle, "stream")?;
        decoder.ensure_state(state)?;

        // SAFETY: every handle is live and belongs to this context, as checked above, and the host stage finishes before the call returns.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegHost(
                self.as_raw(),
                decoder.as_raw(),
                state.as_raw(),
                params.as_raw(),
                stream.as_raw()
            ))
        }
    }

    /// Runs the transfer stage of staged decoding and waits for `cuda_stream`.
    ///
    /// See [`decode_jpeg_transfer_to_device_async`](Self::decode_jpeg_transfer_to_device_async).
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_stream` belongs to another CUDA context, the context cannot be bound, an argument was created with another [`Context`], `state` was not created by `decoder`, nvJPEG fails the stage, or waiting for `cuda_stream` fails.
    pub fn decode_jpeg_transfer_to_device(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        stream: &JpegStream,
        cuda_stream: &CudaStream,
    ) -> Result<()> {
        decoder.ensure_state(state)?;

        // SAFETY: `complete` waits for `cuda_stream` on every path, so the borrowed decoder, state, and stream outlive the queued work.
        cuda_stream.complete(|| unsafe {
            self.decode_jpeg_transfer_to_device_async(decoder, state, stream, cuda_stream)
        })
    }

    /// Runs the transfer stage of staged decoding, copying the host stage's results to the device on `cuda_stream`.
    ///
    /// Call it after [`decode_jpeg_host`](Self::decode_jpeg_host) succeeds, with the same `decoder`, `state`, and `stream`.
    /// `state` must have a device buffer attached, which nvJPEG grows if needed.
    /// The stage does part of its work on the CPU before returning and enqueues the rest.
    ///
    /// # Safety
    ///
    /// `state`, `decoder`, `stream`, the attached buffers, and `cuda_stream` must stay alive until the work finishes, even on failure.
    /// Nothing else may use `state` at the same time.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_stream` belongs to another CUDA context, the context cannot be bound, an argument was created with another [`Context`], or nvJPEG rejects or fails the stage.
    pub unsafe fn decode_jpeg_transfer_to_device_async(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        stream: &JpegStream,
        cuda_stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(cuda_stream)?;
        self.ensure_handle(decoder.handle(), "decoder")?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_handle(&stream.handle, "stream")?;

        // SAFETY: every handle belongs to this context, as checked above, and the caller keeps them and the attached buffers alive until the queued copy finishes.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegTransferToDevice(
                self.as_raw(),
                decoder.as_raw(),
                state.as_raw(),
                stream.as_raw(),
                cuda_stream.as_raw()
            ))
        }
    }

    /// Enqueues the device stage of staged decoding on `cuda_stream`, writing to `destination`.
    ///
    /// Call it after [`decode_jpeg_transfer_to_device_async`](Self::decode_jpeg_transfer_to_device_async) succeeds, with the same `decoder` and `state`.
    /// This stage needs neither the parsed stream nor the pinned buffer, so both can be reused for the next image.
    ///
    /// # Safety
    ///
    /// `destination` must describe device-accessible buffers large enough for the image and the settings given to the host stage, valid until the work on `cuda_stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `cuda_stream` belongs to another CUDA context, the context cannot be bound, an argument was created with another [`Context`], or nvJPEG rejects or fails the stage.
    pub unsafe fn decode_jpeg_device_async(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        destination: &mut Image,
        cuda_stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(cuda_stream)?;
        self.ensure_handle(decoder.handle(), "decoder")?;
        self.ensure_handle(&state.handle, "state")?;

        // SAFETY: every handle belongs to this context, as checked above; the caller guarantees `destination` is large enough and that the operands outlive the queued work.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegDevice(
                self.as_raw(),
                decoder.as_raw(),
                state.as_raw(),
                destination.as_raw_mut(),
                cuda_stream.as_raw()
            ))
        }
    }
}

impl Drop for DecoderOwner {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure("bind the cuda context to destroy", "nvjpeg decoder", &error);
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the last owner drops `DecoderOwner`, after every state that used the decoder, and destroys the handle once.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegDecoderDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg decoder", &error);
            }
        }
    }
}
