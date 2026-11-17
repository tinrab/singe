use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr, slice,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    stream::Stream as CudaStream,
    view::{DeviceOutputView, DeviceRegionProvider},
};
use singe_nvjpeg_sys as sys;

use crate::{
    context::Context,
    encode::{EncodeInput, Quality},
    error::{Error, Result},
    stream::JpegStream,
    try_ffi,
    types::{ChromaSubsampling, EncoderBackend, JpegEncoding},
    utility::{to_positive_i32, to_u64, to_usize},
};

/// State that an encoder keeps, including the intermediate buffers and the last encoded bitstream.
#[derive(Debug)]
pub struct EncoderState {
    raw: sys::nvjpegEncoderState_t,
    pub(crate) handle: Context,
}

/// Native encoder state with its library handle, from [`EncoderState::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawEncoderState {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegEncoderState_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

/// Encode settings: quality, encoding process, Huffman tables, chroma subsampling, and restart interval.
#[derive(Debug)]
pub struct EncodeParams {
    raw: sys::nvjpegEncoderParams_t,
    pub(crate) handle: Context,
}

/// Native encode settings with their library handle, from [`EncodeParams::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawEncodeParams {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegEncoderParams_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

// SAFETY: nvJPEG requires encoder state to be used by one thread at a time, not by a fixed thread, and every encode takes `&mut EncoderState`.
unsafe impl Send for EncoderState {}

// SAFETY: nvJPEG requires encode settings to be used by one thread at a time, not by a fixed thread, and every change takes `&mut EncodeParams`.
unsafe impl Send for EncodeParams {}

impl EncoderState {
    /// Takes ownership of native encoder state created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawEncoderState) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw: raw.handle,
            handle: raw.context,
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    #[must_use]
    pub fn into_raw(self) -> RawEncoderState {
        let this = ManuallyDrop::new(self);
        RawEncoderState {
            handle: this.raw,
            // SAFETY: `this` is never dropped, so its context moves out exactly once.
            context: unsafe { ptr::read(&raw const this.handle) },
        }
    }

    /// Returns the CUDA context of the encoder state.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates encoder state for `handle` that uses `backend`, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or fails to create the state, or waiting for `stream` fails.
    pub fn create(handle: &Context, backend: EncoderBackend, stream: &CudaStream) -> Result<Self> {
        handle.ensure_stream(stream)?;

        let mut raw = ptr::null_mut();
        stream.complete(|| -> Result<()> {
            // SAFETY: `handle` owns a live library handle and `stream` belongs to its bound context, as checked above, and `raw` is a local that receives the state.
            unsafe {
                try_ffi!(sys::nvjpegEncoderStateCreateWithBackend(
                    handle.as_raw(),
                    &raw mut raw,
                    backend.into(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegEncoderState_t {
        self.raw
    }
}

impl EncodeParams {
    /// Takes ownership of native encode settings created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawEncodeParams) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw: raw.handle,
            handle: raw.context,
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    #[must_use]
    pub fn into_raw(self) -> RawEncodeParams {
        let this = ManuallyDrop::new(self);
        RawEncodeParams {
            handle: this.raw,
            // SAFETY: `this` is never dropped, so its context moves out exactly once.
            context: unsafe { ptr::read(&raw const this.handle) },
        }
    }

    /// Returns the CUDA context of the encode settings.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates encode settings for `handle` with nvJPEG's defaults, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG fails to create the settings, or waiting for `stream` fails.
    pub fn create(handle: &Context, stream: &CudaStream) -> Result<Self> {
        handle.ensure_stream(stream)?;

        let mut raw = ptr::null_mut();
        stream.complete(|| -> Result<()> {
            // SAFETY: `handle` owns a live library handle and `stream` belongs to its bound context, as checked above, and `raw` is a local that receives the settings.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsCreate(
                    handle.as_raw(),
                    &raw mut raw,
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let mut params = Self {
            raw,
            handle: handle.clone(),
        };
        // Establish the documented sampling default before any encode.
        params.set_sampling_factors(ChromaSubsampling::Css444, stream)?;
        Ok(params)
    }

    /// Sets the quality, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_quality(&mut self, quality: Quality, stream: &CudaStream) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream` belongs to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsSetQuality(
                    self.raw,
                    quality.value(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Sets the JPEG encoding process, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_encoding(
        &mut self,
        encoding: JpegEncoding,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream` belongs to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsSetEncoding(
                    self.raw,
                    encoding.into(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Sets whether to use optimized Huffman tables, which give smaller files of the same quality but encode more slowly, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_optimized_huffman(
        &mut self,
        optimized: bool,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream` belongs to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsSetOptimizedHuffman(
                    self.raw,
                    i32::from(optimized),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Sets the chroma subsampling of the output, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_sampling_factors(
        &mut self,
        subsampling: ChromaSubsampling,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream` belongs to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsSetSamplingFactors(
                    self.raw,
                    subsampling.into(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Sets the number of minimum coded units between restart markers, and waits for `stream`.
    ///
    /// Restart markers need a hardware encoder; without one, the setting is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the setting, or waiting for `stream` fails.
    pub fn set_restart_interval(
        &mut self,
        interval: u32,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream` belongs to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsSetRestartInterval(
                    self.raw,
                    interval,
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Copies the JFIF, APP, EXT, and COM markers from the parsed JPEG in `stream_info`, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, `state` or `stream_info` was created with another [`Context`], nvJPEG fails to copy, or waiting for `stream` fails.
    pub fn copy_metadata(
        &mut self,
        state: &mut EncoderState,
        stream_info: &JpegStream,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;
        self.handle.ensure_handle(&state.handle, "state")?;
        self.handle
            .ensure_handle(&stream_info.handle, "stream_info")?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self`, `state`, and `stream_info` own live handles of one context and `stream` belongs to it, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsCopyMetadata(
                    state.as_raw(),
                    self.raw,
                    stream_info.as_raw(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Copies the quantization tables from the parsed JPEG in `stream_info`, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, `stream_info` was created with another [`Context`], nvJPEG fails to copy, or waiting for `stream` fails.
    pub fn copy_quantization_tables(
        &mut self,
        stream_info: &JpegStream,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;
        self.handle
            .ensure_handle(&stream_info.handle, "stream_info")?;

        stream.complete(|| -> Result<()> {
            // SAFETY: `&mut self` owns the live settings handle, `stream_info` and `stream` belong to its context, as checked above, and `complete` waits for `stream`.
            unsafe {
                try_ffi!(sys::nvjpegEncoderParamsCopyQuantizationTables(
                    self.raw,
                    stream_info.as_raw(),
                    stream.as_raw()
                ))?;
            }
            Ok(())
        })?;
        Ok(self)
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegEncoderParams_t {
        self.raw
    }
}

impl Context {
    /// Returns the largest size, in bytes, that a `width` by `height` image can encode to with `params`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `params` was created with another [`Context`], a dimension does not fit a positive `i32`, nvJPEG fails the query, or the size does not fit `usize`.
    pub fn encode_buffer_size(
        &self,
        params: &EncodeParams,
        width: impl TryInto<i32>,
        height: impl TryInto<i32>,
    ) -> Result<usize> {
        self.bind()?;
        self.ensure_handle(&params.handle, "params")?;

        let width = to_positive_i32(width, "width")?;
        let height = to_positive_i32(height, "height")?;
        let mut length = 0;
        // SAFETY: `params` belongs to this live handle, as checked above, the dimensions are positive, and `length` is a local that receives the size.
        unsafe {
            try_ffi!(sys::nvjpegEncodeGetBufferSize(
                self.as_raw(),
                params.as_raw(),
                width,
                height,
                &raw mut length,
            ))?;
        }
        to_usize(length, "length")
    }

    /// Enqueues encoding of `input` in any [`InputFormat`](crate::types::InputFormat) on `stream`, with `params`, and keeps the result in `state`.
    ///
    /// Work that writes the pixels is ordered before the encoding on `stream`.
    /// Read the result with [`retrieve_bitstream`](Self::retrieve_bitstream).
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible pixels with its format, subsampling, and size, which must stay valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` or `params` was created with another [`Context`], the input's memory is in another CUDA context, a dimension does not fit an `i32`, or nvJPEG fails.
    pub unsafe fn encode_async(
        &self,
        state: &mut EncoderState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        input.prepare(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_handle(&params.handle, "params")?;

        // SAFETY: `state`, `params`, `stream`, and the input's memory belong to this context, as checked above, and the input's writers are ordered before the call; the caller keeps the pixels valid until the work finishes.
        unsafe {
            try_ffi!(sys::nvjpegEncode(
                self.as_raw(),
                state.as_raw(),
                params.as_raw(),
                input.source.as_raw(),
                input.input_subsampling.into(),
                input.input_format.into(),
                input.width_i32()?,
                input.height_i32()?,
                stream.as_raw()
            ))
        }
    }

    /// Returns the size, in bytes, of the encoded data in `state`, after waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, `state` was created with another [`Context`], nvJPEG fails, the size does not fit `usize`, or waiting for `stream` fails.
    pub fn bitstream_size(&self, state: &mut EncoderState, stream: &CudaStream) -> Result<usize> {
        // SAFETY: `complete` waits for `stream` on every path, so `state` outlives the queued work, and no output buffer is passed.
        stream.complete(|| unsafe { self.retrieve_bitstream_async(state, None, stream) })
    }

    /// Copies the encoded data in `state` into `output` and returns the written part, after waiting for `stream`.
    ///
    /// The call waits for `stream` even if it fails.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, `state` was created with another [`Context`], `output` is too small, a size does not fit, nvJPEG fails, or waiting for `stream` fails.
    pub fn retrieve_bitstream<'a>(
        &self,
        state: &mut EncoderState,
        output: &'a mut [MaybeUninit<u8>],
        stream: &CudaStream,
    ) -> Result<&'a mut [u8]> {
        // SAFETY: `complete` waits for `stream` on every path, so `state` and the exclusively borrowed `output` outlive the queued copy.
        let length = stream
            .complete(|| unsafe { self.retrieve_bitstream_async(state, Some(output), stream) })?;
        if length > output.len() {
            return Err(Error::BufferTooSmall {
                name: "host bitstream".into(),
                required: length,
                actual: output.len(),
            });
        }
        // SAFETY: the finished copy initialized the first `length` bytes of `output`, and `length` was checked against its size above.
        Ok(unsafe { slice::from_raw_parts_mut(output.as_mut_ptr().cast(), length) })
    }

    /// Enqueues a copy of the encoded data in `state` into host memory `output`, and returns its size.
    ///
    /// nvJPEG fails if the data does not fit in `output`; on success it writes only the returned number of bytes.
    /// With `None`, it returns the size of the data without copying it.
    ///
    /// # Safety
    ///
    /// `state`, `output`, and `stream` must stay alive, and `output` must not be used by anything else, until the work on `stream` finishes, even on failure.
    /// Do not read `output` before then.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` was created with another [`Context`], `output` is too small, a size does not fit, or nvJPEG fails.
    pub unsafe fn retrieve_bitstream_async(
        &self,
        state: &mut EncoderState,
        output: Option<&mut [MaybeUninit<u8>]>,
        stream: &CudaStream,
    ) -> Result<usize> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        let mut length = to_u64(
            output.as_ref().map_or(0, |buffer| buffer.len()),
            "output length",
        )?;
        let data = output.map_or(ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast());
        // SAFETY: `state` and `stream` belong to this context, as checked above; `data` is null or points to `length` writable bytes the caller keeps exclusive until the copy finishes, and `length` is a local.
        unsafe {
            try_ffi!(sys::nvjpegEncodeRetrieveBitstream(
                self.as_raw(),
                state.as_raw(),
                data,
                &raw mut length,
                stream.as_raw()
            ))?;
        }
        to_usize(length, "length")
    }

    /// Enqueues a copy of the encoded data in `state` into device memory `output`, and returns its size.
    ///
    /// nvJPEG fails if the data does not fit in `output`; on success it writes only the returned number of bytes.
    /// With `None`, it returns the size of the data without copying it.
    ///
    /// # Safety
    ///
    /// `state`, `output`, and `stream` must stay alive, and `output` must not be used by anything else, until the work on `stream` finishes, even on failure and for every replay of a captured graph.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` was created with another [`Context`], `output` is in another CUDA context or is too small, a size does not fit, or nvJPEG fails.
    pub unsafe fn retrieve_bitstream_device_async(
        &self,
        state: &mut EncoderState,
        output: Option<DeviceOutputView<'_, u8>>,
        stream: &CudaStream,
    ) -> Result<usize> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        if let Some(view) = &output {
            stream.join_regions(&[view.region()])?;
        }

        let mut length = to_u64(
            output.as_ref().map_or(0, DeviceOutputView::len),
            "output length",
        )?;
        let data = output.map_or(ptr::null_mut(), |mut view| view.as_mut_ptr());
        // SAFETY: `state`, `stream`, and the output's memory belong to this context, as checked above; `data` is null or points to `length` writable device bytes the caller keeps exclusive until the copy finishes.
        unsafe {
            try_ffi!(sys::nvjpegEncodeRetrieveBitstreamDevice(
                self.as_raw(),
                state.as_raw(),
                data,
                &raw mut length,
                stream.as_raw()
            ))?;
        }
        to_usize(length, "length")
    }
}

impl Drop for EncoderState {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg encoder state",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the state owns its handle exclusively and is destroyed once; safe encodes wait for their work, and raw callers keep the state alive until theirs finishes.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegEncoderStateDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg encoder state", &error);
            }
        }
    }
}

impl Drop for EncodeParams {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg encoder params",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the settings own their handle exclusively and are destroyed once; safe encodes wait for their work, and raw callers keep the settings alive until theirs finishes.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegEncoderParamsDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg encoder params", &error);
            }
        }
    }
}
