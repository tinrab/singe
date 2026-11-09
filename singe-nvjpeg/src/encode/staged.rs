use std::{
    mem::{self, MaybeUninit},
    ptr, slice,
    sync::Arc,
};

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
    utility::{complete, to_positive_i32, to_u64, to_usize},
};

/// State that an encoder keeps, including the intermediate buffers and the last encoded bitstream.
#[derive(Debug)]
pub struct EncoderState {
    raw: sys::nvjpegEncoderState_t,
    pub(crate) handle: Context,
}

/// Encode settings: quality, encoding process, Huffman tables, chroma subsampling, and restart interval.
#[derive(Debug)]
pub struct EncodeParams {
    raw: sys::nvjpegEncoderParams_t,
    pub(crate) handle: Context,
}

impl EncoderState {
    /// Takes ownership of native encoder state created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpegEncoderState_t, handle: &Context) -> Result<Self> {
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
    pub fn into_raw(mut self) -> sys::nvjpegEncoderState_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }

    /// Returns the CUDA context of the encoder state.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates encoder state for `handle` with the default encoder backend, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG fails to create the state, or waiting for `stream` fails.
    pub fn create(handle: &Context, stream: &CudaStream) -> Result<Self> {
        handle.ensure_stream(stream)?;

        let mut raw = ptr::null_mut();
        complete(stream, || {
            // SAFETY: `nvjpegEncoderStateCreate` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::nvjpegEncoderStateCreate(
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
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Creates encoder state for `handle` that uses `backend`, and waits for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the CUDA context cannot be bound, nvJPEG rejects the backend or fails to create the state, or waiting for `stream` fails.
    pub fn create_with_backend(
        handle: &Context,
        backend: EncoderBackend,
        stream: &CudaStream,
    ) -> Result<Self> {
        handle.ensure_stream(stream)?;

        let mut raw = ptr::null_mut();
        complete(stream, || {
            // SAFETY: `nvjpegEncoderStateCreateWithBackend` receives live handles owned by the wrappers and valid output locations.
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
    /// Takes ownership of native encode settings created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpegEncoderParams_t, handle: &Context) -> Result<Self> {
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
    pub fn into_raw(mut self) -> sys::nvjpegEncoderParams_t {
        mem::replace(&mut self.raw, ptr::null_mut())
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
        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsCreate` receives live handles owned by the wrappers and valid output locations.
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
    /// It also fails if `quality` is not a valid [`Quality`].
    pub fn set_quality(
        &mut self,
        quality: impl TryInto<Quality>,
        stream: &CudaStream,
    ) -> Result<&mut Self> {
        self.handle.ensure_stream(stream)?;

        let quality = quality.try_into().map_err(|_| Error::OutOfRange {
            name: "quality".into(),
        })?;
        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsSetQuality` receives live handles owned by the wrappers.
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

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsSetEncoding` receives live handles owned by the wrappers.
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

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsSetOptimizedHuffman` receives live handles owned by the wrappers.
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

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsSetSamplingFactors` receives live handles owned by the wrappers.
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

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsSetRestartInterval` receives live handles owned by the wrappers.
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
        self.handle.ensure_handle(&state.handle)?;
        self.handle.ensure_handle(&stream_info.handle)?;

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsCopyMetadata` receives live handles owned by the wrappers.
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
        self.handle.ensure_handle(&stream_info.handle)?;

        complete(stream, || {
            // SAFETY: `nvjpegEncoderParamsCopyQuantizationTables` receives live handles owned by the wrappers.
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
        self.ensure_handle(&params.handle)?;

        let width = to_positive_i32(width, "width")?;
        let height = to_positive_i32(height, "height")?;
        let mut length = 0;
        // SAFETY: `nvjpegEncodeGetBufferSize` receives live handles owned by the wrappers and valid output locations.
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

    /// Enqueues encoding of an RGB or BGR `input` on `stream`, with `params`, and keeps the result in `state`.
    ///
    /// Read the result with [`retrieve_bitstream`](Self::retrieve_bitstream).
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible RGB or BGR pixels, which must stay valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` or `params` was created with another [`Context`], a dimension does not fit an `i32`, the input's memory is in another context, or nvJPEG fails.
    pub unsafe fn encode_image_async(
        &self,
        state: &mut EncoderState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        input.prepare(stream)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&params.handle)?;

        // SAFETY: `nvjpegEncodeImage` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegEncodeImage(
                self.as_raw(),
                state.as_raw(),
                params.as_raw(),
                input.source.as_raw(),
                input.input_format.into(),
                input.width_i32()?,
                input.height_i32()?,
                stream.as_raw()
            ))
        }
    }

    /// Enqueues encoding of a YUV `input` on `stream`, with `params`, and keeps the result in `state`.
    ///
    /// Read the result with [`retrieve_bitstream`](Self::retrieve_bitstream).
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible YUV pixels with its subsampling and size, which must stay valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` or `params` was created with another [`Context`], a dimension does not fit an `i32`, the input's memory is in another context, or nvJPEG fails.
    pub unsafe fn encode_yuv_async(
        &self,
        state: &mut EncoderState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        input.prepare(stream)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&params.handle)?;

        // SAFETY: `nvjpegEncodeYUV` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegEncodeYUV(
                self.as_raw(),
                state.as_raw(),
                params.as_raw(),
                input.source.as_raw(),
                input.input_subsampling.into(),
                input.width_i32()?,
                input.height_i32()?,
                stream.as_raw()
            ))
        }
    }

    /// Enqueues encoding of `input` in any [`InputFormat`](crate::types::InputFormat) on `stream`, with `params`, and keeps the result in `state`.
    ///
    /// It can replace [`encode_image_async`](Self::encode_image_async) and [`encode_yuv_async`](Self::encode_yuv_async), and is the only one that accepts [`InputFormat::Nv12`](crate::types::InputFormat::Nv12).
    /// Read the result with [`retrieve_bitstream`](Self::retrieve_bitstream).
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible pixels with its format, subsampling, and size.
    /// Work that writes the pixels must be ordered before this on `stream`, and the pixels must stay valid until the work finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, `state` or `params` was created with another [`Context`], a dimension does not fit an `i32`, or nvJPEG fails.
    pub unsafe fn encode_async(
        &self,
        state: &mut EncoderState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_handle(&params.handle)?;

        // SAFETY: `nvjpegEncode` receives live handles owned by the wrappers.
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
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.retrieve_bitstream_async(state, None, stream)
        })
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
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        let length = complete(stream, || unsafe {
            self.retrieve_bitstream_async(state, Some(output), stream)
        })?;
        if length > output.len() {
            return Err(Error::BufferTooSmall {
                name: "host bitstream".into(),
                required: length,
                actual: output.len(),
            });
        }
        // SAFETY: the completed native call produced exactly this byte prefix.
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
        self.ensure_handle(&state.handle)?;
        let mut length = to_u64(
            output.as_ref().map_or(0, |buffer| buffer.len()),
            "output length",
        )?;
        let data = output.map_or(ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast());
        // SAFETY: `nvjpegEncodeRetrieveBitstream` receives live handles owned by the wrappers and valid output locations.
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
        self.ensure_handle(&state.handle)?;
        if let Some(view) = &output {
            stream.join_regions(&[view.region()])?;
        }

        let mut length = to_u64(
            output.as_ref().map_or(0, DeviceOutputView::len),
            "output length",
        )?;
        let data = output.map_or(ptr::null_mut(), |mut view| view.as_mut_ptr());
        // SAFETY: `nvjpegEncodeRetrieveBitstreamDevice` receives live handles owned by the wrappers and valid output locations.
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
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegEncoderStateDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegEncoderStateDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg encoder state: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for EncodeParams {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegEncoderParamsDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegEncoderParamsDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg encoder params: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
