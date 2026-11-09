//! Staged JPEG decoding with a chosen backend.

use std::{mem, ptr, sync::Arc};

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
    utility::complete,
};

/// Decoder for nvJPEG's staged decoding, tied to one backend.
///
/// Staged decoding splits decoding into a host stage, a transfer to the device, and a device stage, so the stages of different images can overlap.
#[derive(Debug)]
pub struct JpegDecoder {
    raw: sys::nvjpegJpegDecoder_t,
    pub(crate) handle: Context,
}

impl JpegDecoder {
    /// Takes ownership of a native decoder created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpegJpegDecoder_t, handle: &Context) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Releases the native handle without destroying it.
    ///
    /// Destroy it, or pass it back to [`from_raw`](Self::from_raw), while its [`Context`] is alive.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvjpegJpegDecoder_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }

    /// Returns the CUDA context of the decoder.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates a decoder for `handle` that uses `backend`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the decoder.
    pub fn create(handle: &Context, backend: Backend) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegDecoderCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegDecoderCreate(
                handle.as_raw(),
                backend.into(),
                &raw mut raw
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Returns whether this decoder can decode the JPEG in `stream` with `params`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `stream` or `params` was created with another [`Context`], or nvJPEG fails the query.
    /// An unsupported JPEG returns `Ok(false)`, not an error.
    pub fn jpeg_supported(&self, stream: &JpegStream, params: &DecodeParams) -> Result<bool> {
        self.handle.bind()?;
        self.handle.ensure_handle(&stream.handle)?;
        self.handle.ensure_handle(&params.handle)?;

        let mut unsupported = 0;
        // SAFETY: `nvjpegDecoderJpegSupported` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegDecoderJpegSupported(
                self.raw,
                stream.as_raw(),
                params.as_raw(),
                &raw mut unsupported
            ))?;
        }
        Ok(unsupported == 0)
    }

    /// Creates a decoder state for this decoder's backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn create_state(&self) -> Result<JpegState> {
        self.handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegDecoderStateCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegDecoderStateCreate(
                self.handle.as_raw(),
                self.raw,
                &raw mut raw
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: nvJPEG just created `raw` for this handle, and nothing else owns it.
        unsafe { JpegState::from_raw_with_handle(raw, &self.handle) }
    }

    /// Returns the native handle, which this decoder still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegJpegDecoder_t {
        self.raw
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
        self.ensure_handle(&stream.handle)?;

        let mut unsupported = 0;
        // SAFETY: `nvjpegDecodeBatchedSupported` receives live handles owned by the wrappers and valid output locations.
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
        self.ensure_handle(&stream.handle)?;
        self.ensure_handle(&params.handle)?;

        let mut unsupported = 0;
        // SAFETY: `nvjpegDecodeBatchedSupportedEx` receives live handles owned by the wrappers and valid output locations.
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
        self.ensure_handle(&decoder.handle)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&stream.handle)?;
        self.ensure_handle(&params.handle)?;

        // SAFETY: `nvjpegDecodeJpeg` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpeg(
                self.as_raw(),
                decoder.raw,
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
    /// Returns an error if the CUDA context cannot be bound, an argument was created with another [`Context`], or nvJPEG rejects or fails the stage.
    pub fn decode_jpeg_host(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        params: &DecodeParams,
        stream: &JpegStream,
    ) -> Result<()> {
        self.bind()?;
        self.ensure_handle(&decoder.handle)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&params.handle)?;
        self.ensure_handle(&stream.handle)?;

        // SAFETY: `nvjpegDecodeJpegHost` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegHost(
                self.as_raw(),
                decoder.raw,
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
    /// Returns an error if `cuda_stream` belongs to another CUDA context, the context cannot be bound, an argument was created with another [`Context`], nvJPEG fails the stage, or waiting for `cuda_stream` fails.
    pub fn decode_jpeg_transfer_to_device(
        &self,
        decoder: &JpegDecoder,
        state: &mut JpegState,
        stream: &JpegStream,
        cuda_stream: &CudaStream,
    ) -> Result<()> {
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(cuda_stream, || unsafe {
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
        self.ensure_handle(&decoder.handle)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&stream.handle)?;

        // SAFETY: `nvjpegDecodeJpegTransferToDevice` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegTransferToDevice(
                self.as_raw(),
                decoder.raw,
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
        self.ensure_handle(&decoder.handle)?;
        self.ensure_handle(&state.handle)?;

        // SAFETY: `nvjpegDecodeJpegDevice` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeJpegDevice(
                self.as_raw(),
                decoder.raw,
                state.as_raw(),
                destination.as_raw_mut(),
                cuda_stream.as_raw()
            ))
        }
    }
}

impl Drop for JpegDecoder {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegDecoderDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegDecoderDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg decoder: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
