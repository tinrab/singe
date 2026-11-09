//! JPEG 2000 encoding.

use std::{
    fs,
    mem::MaybeUninit,
    path::Path,
    ptr, slice,
    sync::{Arc, Mutex},
};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, stream::Stream as CudaStream,
};
use singe_nvjpeg_sys::k2 as sys;

use crate::{
    error::{Error, Result},
    image::Size,
    k2::{
        BitstreamType, ColorSpace, Image, ImageComponentInfo, ImageFormat, MAX_IMAGE_COMPONENTS,
        ProgressionOrder, QualityType,
        image::{EncodeInput, validate_plane},
        validate_code_block_size, validate_component_precision, validate_dimensions,
        validate_resolutions,
    },
    try_ffi_k2,
    utility::{checked_mul, complete, to_u32, to_u64, to_usize},
};

/// Quality setting for lossy JPEG 2000 encoding: how to interpret the value, and the value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeQuality {
    pub(crate) quality_type: QualityType,
    pub(crate) value: f64,
}

/// Settings for a JPEG 2000 [`Encoder`]: input layout and quality.
///
/// The encoder applies them to its native [`EncodeParams`], together with the [`EncodeConfig`] set with [`Encoder::set_encode_config`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    pub(crate) input_format: Option<ImageFormat>,
    pub(crate) quality: Option<EncodeQuality>,
}

/// Builder for an [`EncodeConfig`].
#[derive(Debug, Clone)]
pub struct EncodeConfigBuilder {
    pub(crate) size: Size,
    pub(crate) stream_type: BitstreamType,
    pub(crate) color_space: ColorSpace,
    pub(crate) component_info: Vec<ImageComponentInfo>,
    pub(crate) tiling: Option<Size>,
    pub(crate) progression_order: Option<ProgressionOrder>,
    pub(crate) resolutions: Option<u32>,
    pub(crate) code_block_size: Option<Size>,
    pub(crate) irreversible: Option<bool>,
    pub(crate) multi_component_transform: Option<bool>,
    pub(crate) high_throughput: Option<bool>,
}

impl EncodeQuality {
    /// Creates a quality setting of `quality_type` with `value`.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite, a target PSNR or quantization step is not positive, or a Q-factor is not between 1 and 100.
    pub fn create(quality_type: QualityType, value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::OutOfRange {
                name: "quality_value".into(),
            });
        }

        match quality_type {
            QualityType::TargetPsnr | QualityType::QuantizationStep if value <= 0.0 => {
                return Err(Error::OutOfRange {
                    name: "quality_value".into(),
                });
            }
            QualityType::QFactor if !(1.0..=100.0).contains(&value) => {
                return Err(Error::OutOfRange {
                    name: "quality_value".into(),
                });
            }
            _ => {}
        }
        Ok(Self {
            quality_type,
            value,
        })
    }

    /// Returns how the value is interpreted.
    #[must_use]
    pub const fn quality_type(self) -> QualityType {
        self.quality_type
    }

    /// Returns the value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

impl EncodeOptions {
    /// Creates options with nvJPEG 2000's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            input_format: None,
            quality: None,
        }
    }

    /// Sets whether the input is planar or interleaved.
    #[must_use]
    pub const fn with_input_format(mut self, input_format: ImageFormat) -> Self {
        self.input_format = Some(input_format);
        self
    }

    /// Sets the quality for lossy encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite, a target PSNR or quantization step is not positive, or a Q-factor is not between 1 and 100.
    pub fn with_quality(mut self, quality_type: QualityType, value: f64) -> Result<Self> {
        self.quality = Some(EncodeQuality::create(quality_type, value)?);
        Ok(self)
    }

    /// Sets a JPEG-style quality from 1 to 100.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite or not between 1 and 100.
    pub fn with_q_factor(self, value: f64) -> Result<Self> {
        self.with_quality(QualityType::QFactor, value)
    }

    /// Sets a target peak signal-to-noise ratio.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite or not positive.
    pub fn with_target_psnr(self, value: f64) -> Result<Self> {
        self.with_quality(QualityType::TargetPsnr, value)
    }

    /// Sets a quantization step.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is not finite or not positive.
    pub fn with_quantization_step(self, value: f64) -> Result<Self> {
        self.with_quality(QualityType::QuantizationStep, value)
    }

    /// Returns the input layout, if set.
    #[must_use]
    pub const fn input_format(self) -> Option<ImageFormat> {
        self.input_format
    }

    /// Returns the quality, if set.
    #[must_use]
    pub const fn quality(self) -> Option<EncodeQuality> {
        self.quality
    }

    fn apply_without_config(self, params: &mut EncodeParams) -> Result<()> {
        if let Some(input_format) = self.input_format {
            params.set_input_format(input_format)?;
        }
        Ok(())
    }

    fn apply_after_config(self, params: &mut EncodeParams) -> Result<()> {
        if let Some(quality) = self.quality {
            params.specify_quality(quality.quality_type, quality.value)?;
        }
        Ok(())
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeConfigBuilder {
    /// Creates a builder for an image of `size`, written as `stream_type` in `color_space`.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn create(size: Size, stream_type: BitstreamType, color_space: ColorSpace) -> Result<Self> {
        validate_dimensions(size, "image")?;
        Ok(Self {
            size,
            stream_type,
            color_space,
            component_info: Vec::new(),
            tiling: None,
            progression_order: None,
            resolutions: None,
            code_block_size: None,
            irreversible: None,
            multi_component_transform: None,
            high_throughput: None,
        })
    }

    /// Adds a component.
    #[must_use]
    pub fn component(mut self, component: ImageComponentInfo) -> Self {
        self.component_info.push(component);
        self
    }

    /// Adds several components.
    #[must_use]
    pub fn components(mut self, components: impl IntoIterator<Item = ImageComponentInfo>) -> Self {
        self.component_info.extend(components);
        self
    }

    /// Adds one 8-bit unsigned component.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn grayscale_u8(mut self) -> Result<Self> {
        self.color_space = ColorSpace::Gray;
        self.component_info = vec![ImageComponentInfo::u8(self.size)?];
        self.multi_component_transform = Some(false);
        Ok(self)
    }

    /// Adds one unsigned component with `precision` bits, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn grayscale_u16(mut self, precision: u8) -> Result<Self> {
        self.color_space = ColorSpace::Gray;
        self.component_info = vec![ImageComponentInfo::u16(self.size, precision)?];
        self.multi_component_transform = Some(false);
        Ok(self)
    }

    /// Adds one signed component with `precision` bits, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn grayscale_i16(mut self, precision: u8) -> Result<Self> {
        self.color_space = ColorSpace::Gray;
        self.component_info = vec![ImageComponentInfo::i16(self.size, precision)?];
        self.multi_component_transform = Some(false);
        Ok(self)
    }

    /// Adds three 8-bit unsigned components.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn rgb_u8(mut self) -> Result<Self> {
        let component = ImageComponentInfo::u8(self.size)?;
        self.color_space = ColorSpace::Srgb;
        self.component_info = vec![component; 3];
        self.multi_component_transform = Some(true);
        Ok(self)
    }

    /// Adds three unsigned components with `precision` bits, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn rgb_u16(mut self, precision: u8) -> Result<Self> {
        let component = ImageComponentInfo::u16(self.size, precision)?;
        self.color_space = ColorSpace::Srgb;
        self.component_info = vec![component; 3];
        self.multi_component_transform = Some(true);
        Ok(self)
    }

    /// Adds three signed components with `precision` bits, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn rgb_i16(mut self, precision: u8) -> Result<Self> {
        let component = ImageComponentInfo::i16(self.size, precision)?;
        self.color_space = ColorSpace::Srgb;
        self.component_info = vec![component; 3];
        self.multi_component_transform = Some(true);
        Ok(self)
    }

    /// Splits the image into tiles of `size`.
    ///
    /// nvJPEG 2000 checks the tile size against the image when the configuration is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if a tile dimension is zero.
    pub fn tiling(mut self, size: Size) -> Result<Self> {
        validate_dimensions(size, "tile")?;
        self.tiling = Some(size);
        Ok(self)
    }

    /// Sets the progression order.
    #[must_use]
    pub const fn progression_order(mut self, order: ProgressionOrder) -> Self {
        self.progression_order = Some(order);
        self
    }

    /// Sets the number of resolution levels.
    ///
    /// nvJPEG 2000 checks it against the image and tile size when the configuration is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if `resolutions` is zero.
    pub fn resolutions(mut self, resolutions: u32) -> Result<Self> {
        validate_resolutions(resolutions)?;
        self.resolutions = Some(resolutions);
        Ok(self)
    }

    /// Sets the code block size.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is not 32 or 64.
    pub fn code_block_size(mut self, size: Size) -> Result<Self> {
        validate_code_block_size(size)?;
        self.code_block_size = Some(size);
        Ok(self)
    }

    /// Sets whether to use the irreversible wavelet transform, which is lossy.
    #[must_use]
    pub const fn irreversible(mut self, irreversible: bool) -> Self {
        self.irreversible = Some(irreversible);
        self
    }

    /// Uses the reversible wavelet transform, for lossless encoding.
    #[must_use]
    pub const fn lossless(self) -> Self {
        self.irreversible(false)
    }

    /// Uses the irreversible wavelet transform, for lossy encoding.
    #[must_use]
    pub const fn lossy(self) -> Self {
        self.irreversible(true)
    }

    /// Sets whether to apply the multiple-component transform; enable it for RGB input and disable it for YCC or grayscale.
    #[must_use]
    pub const fn multi_component_transform(mut self, enabled: bool) -> Self {
        self.multi_component_transform = Some(enabled);
        self
    }

    /// Sets whether to use High-Throughput JPEG 2000 (HTJ2K) block coding.
    #[must_use]
    pub const fn high_throughput(mut self, enabled: bool) -> Self {
        self.high_throughput = Some(enabled);
        self
    }

    /// Creates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero, there are no components or more than four, a component's bits per sample are not between 1 and 16, a tile dimension or the number of resolution levels is zero, or the code block size is not 32 or 64.
    pub fn build(self) -> Result<EncodeConfig> {
        let mut config = EncodeConfig::create(
            self.size.width,
            self.size.height,
            self.stream_type,
            self.color_space,
            self.component_info,
        )?;

        if let Some(size) = self.tiling {
            config.set_tiling(size.width, size.height)?;
        }

        if let Some(order) = self.progression_order {
            config.set_progression_order(order);
        }

        if let Some(resolutions) = self.resolutions {
            config.set_resolutions(resolutions)?;
        }

        if let Some(size) = self.code_block_size {
            config.set_code_block_size(size.width, size.height)?;
        }

        if let Some(irreversible) = self.irreversible {
            config.set_irreversible(irreversible);
        }

        if let Some(enabled) = self.multi_component_transform {
            config.set_multi_component_transform(enabled);
        }

        if let Some(enabled) = self.high_throughput {
            config.set_high_throughput(enabled);
        }

        Ok(config)
    }
}

/// nvJPEG 2000 encoder handle, for the lower-level encoding functions.
///
/// It keeps its CUDA context alive.
#[derive(Debug, Clone)]
pub struct EncoderHandle {
    inner: Arc<EncoderHandleInner>,
}

#[derive(Debug)]
struct EncoderHandleInner {
    pub(crate) raw: sys::nvjpeg2kEncoder_t,
    cuda_context: Arc<CudaContext>,
    streams: Mutex<Vec<CudaStream>>,
}

/// JPEG 2000 encoder.
///
/// It owns its handle, state, and settings, and reuses them across calls.
#[derive(Debug)]
pub struct Encoder {
    pub(crate) handle: EncoderHandle,
    pub(crate) state: EncodeState,
    pub(crate) params: EncodeParams,
    pub(crate) options: EncodeOptions,
}

/// Builder for a JPEG 2000 [`Encoder`] with custom options or a configuration.
#[derive(Debug)]
pub struct EncoderBuilder {
    pub(crate) options: EncodeOptions,
    pub(crate) config: Option<EncodeConfig>,
}

/// State that a JPEG 2000 encoder keeps, including the last encoded data.
#[derive(Debug)]
pub struct EncodeState {
    pub(crate) raw: sys::nvjpeg2kEncodeState_t,
    pub(crate) handle: EncoderHandle,
}

/// Native nvJPEG 2000 encode parameters for staged encoding: an [`EncodeConfig`], input layout, and quality.
#[derive(Debug)]
pub struct EncodeParams {
    pub(crate) raw: sys::nvjpeg2kEncodeParams_t,
    pub(crate) handle: EncoderHandle,
    #[expect(
        clippy::vec_box,
        reason = "native config and component pointers must remain stable across updates"
    )]
    configs: Vec<Box<EncodeConfig>>,
    active_config: Option<usize>,
    input_format: Option<ImageFormat>,
}

/// Image size, components, tiling, and coding settings for JPEG 2000 encoding.
///
/// These are the values written to the SIZ and COD headers; fields nvJPEG 2000 does not support keep the values it requires.
#[derive(Debug)]
pub struct EncodeConfig {
    pub(crate) raw: sys::nvjpeg2kEncodeConfig_t,
    pub(crate) component_info: Vec<sys::nvjpeg2kImageComponentInfo_t>,
}

// SAFETY: native encoder handles can be shared while per-operation state stays
// exclusive. Native calls and destruction bind their retained actual context.
unsafe impl Send for EncoderHandleInner {}

unsafe impl Sync for EncoderHandleInner {}

impl EncoderHandle {
    /// Returns the CUDA context of the handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.cuda_context
    }

    /// Makes the handle's CUDA context current on this thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot make the context current.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Checks that `stream` belongs to the handle's CUDA context, makes the context current, and keeps `stream` alive until the handle and its resources are destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or the context cannot be made current.
    pub fn ensure_stream(&self, stream: &CudaStream) -> Result<()> {
        if stream.context() != self.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()?;
        let mut streams = self.inner.streams.lock().map_err(|_| Error::LockPoisoned {
            name: "nvjpeg streams",
        })?;
        if !streams
            .iter()
            .any(|owner| owner.as_raw() == stream.as_raw())
        {
            streams.push(stream.clone());
        }

        drop(streams);
        Ok(())
    }

    pub(crate) fn ensure_handle(&self, other: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &other.inner) {
            return Err(Error::HandleMismatch);
        }
        Ok(())
    }

    pub(crate) fn ensure_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if self.cuda_context() != context {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        Ok(())
    }

    /// Creates an encoder handle in `cuda_context` with default settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the handle.
    pub fn create_simple(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kEncoderCreateSimple` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncoderCreateSimple(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(EncoderHandleInner {
                raw,
                cuda_context: Arc::clone(cuda_context),
                streams: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Creates encoder state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the state.
    pub fn create_state(&self) -> Result<EncodeState> {
        self.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kEncodeStateCreate` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeStateCreate(self.inner.raw, &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(EncodeState {
            raw,
            handle: self.clone(),
        })
    }

    /// Enqueues encoding of `input` on `stream`, with `params`, and keeps the result in `state`.
    ///
    /// Read the result with [`retrieve_bitstream`](Self::retrieve_bitstream).
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible pixels that stay valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or `params` belongs to another CUDA context, the context cannot be bound, `state` was created with another handle, or nvJPEG 2000 fails.
    pub unsafe fn encode_async(
        &self,
        state: &mut EncodeState,
        params: &EncodeParams,
        input: &Image,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;
        self.ensure_context(params.cuda_context())?;

        // SAFETY: `nvjpeg2kEncode` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncode(
                self.inner.raw,
                state.raw,
                params.raw,
                input.as_raw(),
                stream.as_raw()
            ))
        }
    }

    /// Encodes `input` with `params` and returns the JPEG 2000 data in host memory, waiting for `stream`.
    ///
    /// Work that writes `input` is ordered before the encoding, and the call waits for the GPU work even if it fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration or input layout has not been set, `input` does not match the configuration in layout, component count, size, pitch, or memory size, interleaved components differ in size, a size overflows, the CUDA contexts differ, `state` was created with another [`EncoderHandle`], `stream` is being captured, the CUDA context cannot be bound, or encoding, copying the data, or waiting for `stream` fails.
    pub fn encode_to_vec(
        &self,
        state: &mut EncodeState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<Vec<u8>> {
        self.ensure_stream(stream)?;
        params.validate_input(input)?;
        complete(stream, || {
            stream.join_regions(&input.regions)?;
            // SAFETY: initialized views and checked configuration establish the native layout.
            unsafe { self.encode_async(state, params, &input.image, stream) }
        })?;
        let length = self.bitstream_size(state, stream)?;
        let mut output = Vec::with_capacity(length);
        let produced = self
            .retrieve_bitstream(state, output.spare_capacity_mut(), stream)?
            .len();
        // SAFETY: checked completed retrieval produced this prefix of spare capacity.
        unsafe {
            output.set_len(produced);
        }
        Ok(output)
    }

    /// Encodes `input` with `params` and writes the JPEG 2000 data to the file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode_to_vec`](Self::encode_to_vec) fails or the file cannot be written.
    pub fn encode_to_file(
        &self,
        state: &mut EncodeState,
        params: &EncodeParams,
        input: &EncodeInput<'_>,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<()> {
        fs::write(path, self.encode_to_vec(state, params, input, stream)?)?;
        Ok(())
    }

    /// Returns the size, in bytes, of the encoded data in `state`, after waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the context cannot be bound, `state` was created with another handle, nvJPEG 2000 fails, the size does not fit `usize`, or waiting for `stream` fails.
    pub fn bitstream_size(&self, state: &mut EncodeState, stream: &CudaStream) -> Result<usize> {
        let mut length = 0;
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.retrieve_bitstream_async(state, None, &mut length, stream)
        })?;
        to_usize(length, "bitstream length")
    }

    /// Copies the encoded data in `state` into `output` and returns the written part, after waiting for `stream`.
    ///
    /// Bytes after the returned part are not written.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or is being captured, the context cannot be bound, `state` was created with another handle, `output` is too small or too long for nvJPEG 2000, nvJPEG 2000 fails, it reports more data than `output` holds, or waiting for `stream` fails.
    pub fn retrieve_bitstream<'a>(
        &self,
        state: &mut EncodeState,
        output: &'a mut [MaybeUninit<u8>],
        stream: &CudaStream,
    ) -> Result<&'a mut [u8]> {
        let mut length = to_u64(output.len(), "output length")?;
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe {
            self.retrieve_bitstream_async(state, Some(output), &mut length, stream)
        })?;
        let length = to_usize(length, "bitstream length")?;
        if length > output.len() {
            return Err(Error::BufferTooSmall {
                name: "host bitstream".into(),
                required: length,
                actual: output.len(),
            });
        }
        // SAFETY: native retrieval and checked completion establish these initialized bytes.
        Ok(unsafe { slice::from_raw_parts_mut(output.as_mut_ptr().cast(), length) })
    }

    /// Enqueues a copy of the encoded data in `state` into `output`, and sets `length` to its size.
    ///
    /// This sets `length` to the size of `output`, or 0 for `None`, before calling nvJPEG 2000. With `None`, nvJPEG 2000 only reports the size, and does so before returning.
    /// With `Some`, the copy fails if `output` is too small, and the bytes are ready only after the work on `stream` finishes.
    ///
    /// # Safety
    ///
    /// `state`, `output`, and `length` must stay alive, and be used by nothing else, until the work on `stream` finishes, even on failure and for every replay of a captured graph.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `state` was created with another handle, `output` is too long for nvJPEG 2000 or too small, or nvJPEG 2000 fails.
    pub unsafe fn retrieve_bitstream_async(
        &self,
        state: &mut EncodeState,
        output: Option<&mut [MaybeUninit<u8>]>,
        length: &mut u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;
        if let Some(buffer) = &output {
            *length = to_u64(buffer.len(), "output length")?;
        } else {
            *length = 0;
        }

        let data = output.map_or(ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast());
        // SAFETY: `nvjpeg2kEncodeRetrieveBitstream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeRetrieveBitstream(
                self.as_raw(),
                state.raw,
                data,
                length,
                stream.as_raw()
            ))
        }
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvjpeg2kEncoder_t {
        self.inner.raw
    }
}

impl Encoder {
    /// Returns a builder for an encoder with custom options or a configuration.
    #[must_use]
    pub const fn builder() -> EncoderBuilder {
        EncoderBuilder::new()
    }

    /// Creates an encoder in `cuda_context` with the default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails to create the handle, state, or settings, or it rejects the input layout.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Self::builder().create(cuda_context)
    }

    /// Creates an encoder in `cuda_context` with `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails to create the handle, state, or settings, or it rejects the input layout.
    pub fn create_with_options(
        cuda_context: &Arc<CudaContext>,
        options: EncodeOptions,
    ) -> Result<Self> {
        Self::builder().options(options).create(cuda_context)
    }

    /// Creates an encoder that uses an existing `handle` and the default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails to create the handle, state, or settings, or it rejects the input layout.
    pub fn from_handle(handle: EncoderHandle) -> Result<Self> {
        Self::from_handle_with_options(handle, EncodeOptions::new())
    }

    /// Creates an encoder that uses an existing `handle` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails to create the handle, state, or settings, or it rejects the input layout.
    pub fn from_handle_with_options(handle: EncoderHandle, options: EncodeOptions) -> Result<Self> {
        let state = handle.create_state()?;
        let mut params = EncodeParams::create(&handle)?;
        options.apply_without_config(&mut params)?;
        Ok(Self {
            handle,
            state,
            params,
            options,
        })
    }

    /// Returns the encoder handle.
    #[must_use]
    pub const fn handle(&self) -> &EncoderHandle {
        &self.handle
    }

    /// Returns the encode settings.
    #[must_use]
    pub const fn params(&self) -> &EncodeParams {
        &self.params
    }

    /// Returns the encode options.
    #[must_use]
    pub const fn options(&self) -> EncodeOptions {
        self.options
    }

    /// Returns the encode settings mutably.
    pub fn params_mut(&mut self) -> &mut EncodeParams {
        &mut self.params
    }

    /// Sets the encode configuration, then applies the current quality again.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 rejects the configuration, or applying the quality fails.
    /// If only the quality fails, the new configuration stays in effect.
    pub fn set_encode_config(&mut self, config: EncodeConfig) -> Result<&mut Self> {
        self.params.set_encode_config(config)?;
        self.options.apply_after_config(&mut self.params)?;
        Ok(self)
    }

    /// Sets whether the input is planar or interleaved.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the layout.
    /// After a rejection, the layout is unset and safe encoding fails until it is set again.
    pub fn set_input_format(&mut self, format: ImageFormat) -> Result<&mut Self> {
        self.params.set_input_format(format)?;
        self.options = self.options.with_input_format(format);
        Ok(self)
    }

    /// Sets the quality for lossy encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if [`EncodeQuality::create`] rejects the value, the CUDA context cannot be bound, or nvJPEG 2000 rejects the quality for the current configuration.
    pub fn specify_quality(
        &mut self,
        quality_type: QualityType,
        quality_value: f64,
    ) -> Result<&mut Self> {
        let quality = EncodeQuality::create(quality_type, quality_value)?;
        self.params
            .specify_quality(quality.quality_type(), quality.value())?;
        self.options.quality = Some(quality);
        Ok(self)
    }

    /// Sets a JPEG-style quality from 1 to 100.
    ///
    /// # Errors
    ///
    /// Returns an error if [`EncodeQuality::create`] rejects the value, the CUDA context cannot be bound, or nvJPEG 2000 rejects the quality for the current configuration.
    pub fn set_q_factor(&mut self, value: f64) -> Result<&mut Self> {
        self.specify_quality(QualityType::QFactor, value)
    }

    /// Sets a target peak signal-to-noise ratio.
    ///
    /// # Errors
    ///
    /// Returns an error if [`EncodeQuality::create`] rejects the value, the CUDA context cannot be bound, or nvJPEG 2000 rejects the quality for the current configuration.
    pub fn set_target_psnr(&mut self, value: f64) -> Result<&mut Self> {
        self.specify_quality(QualityType::TargetPsnr, value)
    }

    /// Sets a quantization step.
    ///
    /// # Errors
    ///
    /// Returns an error if [`EncodeQuality::create`] rejects the value, the CUDA context cannot be bound, or nvJPEG 2000 rejects the quality for the current configuration.
    pub fn set_quantization_step(&mut self, value: f64) -> Result<&mut Self> {
        self.specify_quality(QualityType::QuantizationStep, value)
    }

    /// Enqueues encoding of `input` on `stream` with the current settings.
    ///
    /// # Safety
    ///
    /// `input` must describe initialized, device-accessible pixels that stay valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` or the settings belong to another CUDA context, the context cannot be bound, or nvJPEG 2000 fails.
    pub unsafe fn encode_async(&mut self, input: &Image, stream: &CudaStream) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.handle
                .encode_async(&mut self.state, &self.params, input, stream)
        }
    }

    /// Encodes `input` and returns the JPEG 2000 data in host memory, waiting for `stream`.
    ///
    /// Work that writes `input` is ordered before the encoding, and the call waits for the GPU work even if it fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration or input layout has not been set, `input` does not match the configuration in layout, component count, size, pitch, or memory size, interleaved components differ in size, a size overflows, the CUDA contexts differ, `state` was created with another [`EncoderHandle`], `stream` is being captured, the CUDA context cannot be bound, or encoding, copying the data, or waiting for `stream` fails.
    pub fn encode_to_vec(
        &mut self,
        input: &EncodeInput<'_>,
        stream: &CudaStream,
    ) -> Result<Vec<u8>> {
        self.handle
            .encode_to_vec(&mut self.state, &self.params, input, stream)
    }

    /// Encodes `input` and writes the JPEG 2000 data to the file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`encode_to_vec`](Self::encode_to_vec) fails or the file cannot be written.
    pub fn encode_to_file(
        &mut self,
        input: &EncodeInput<'_>,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.handle
            .encode_to_file(&mut self.state, &self.params, input, path, stream)
    }
}

impl EncoderBuilder {
    /// Creates a builder with nvJPEG 2000's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            options: EncodeOptions::new(),
            config: None,
        }
    }

    /// Sets the encode options.
    #[must_use]
    pub const fn options(mut self, options: EncodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets the encode configuration to apply when the encoder is created.
    #[must_use]
    pub fn encode_config(mut self, config: EncodeConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Creates the encoder in `cuda_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails to create the handle, state, or settings, or it rejects the input layout, configuration, or quality.
    pub fn create(mut self, cuda_context: &Arc<CudaContext>) -> Result<Encoder> {
        let handle = EncoderHandle::create_simple(cuda_context)?;
        let mut encoder = Encoder::from_handle_with_options(handle, self.options)?;
        if let Some(config) = self.config.take() {
            encoder.set_encode_config(config)?;
        }
        Ok(encoder)
    }
}

impl Default for EncoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodeParams {
    /// Returns the CUDA context of the settings.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates encode settings for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create them.
    pub fn create(handle: &EncoderHandle) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kEncodeParamsCreate` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeParamsCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
            configs: Vec::new(),
            active_config: None,
            input_format: Some(ImageFormat::Planar),
        })
    }

    /// Sets the image size, components, and coding settings written to the headers.
    ///
    /// Encoding always needs a configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the configuration.
    /// After a rejection, no configuration is in effect, and safe encoding fails until one is set.
    pub fn set_encode_config(&mut self, config: EncodeConfig) -> Result<&mut Self> {
        self.handle.bind()?;

        let mut config = Box::new(config);
        let raw = config.as_raw_mut();
        self.configs.push(config);
        self.active_config = None;
        // SAFETY: `nvjpeg2kEncodeParamsSetEncodeConfig` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeParamsSetEncodeConfig(self.raw, raw))?;
        }

        self.active_config = Some(self.configs.len() - 1);
        Ok(self)
    }

    /// Sets the quality for lossy encoding, replacing any earlier quality.
    ///
    /// Set the configuration with [`set_encode_config`](Self::set_encode_config) first.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the quality: when no configuration is set, `quality_value` is negative, encoding is reversible, [`QualityType::TargetPsnr`] is used with HTJ2K, [`QualityType::QFactor`] is above 100, or [`QualityType::QFactor`] is used for sRGB input without the multiple-component transform.
    pub fn specify_quality(
        &mut self,
        quality_type: QualityType,
        quality_value: f64,
    ) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpeg2kEncodeParamsSpecifyQuality` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeParamsSpecifyQuality(
                self.raw,
                quality_type.into(),
                quality_value
            ))?;
        }
        Ok(self)
    }

    /// Sets whether the input is planar or interleaved.
    ///
    /// Interleaved input needs every component to have the same size, and a plane with a pitch of at least `width * components * bytes_per_sample` bytes, with 1 byte per sample for 8-bit samples and 2 otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the layout.
    /// After a rejection, the layout is unset and safe encoding fails until it is set again.
    pub fn set_input_format(&mut self, format: ImageFormat) -> Result<&mut Self> {
        self.handle.bind()?;

        self.input_format = None;
        // SAFETY: `nvjpeg2kEncodeParamsSetInputFormat` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kEncodeParamsSetInputFormat(
                self.raw,
                format.into()
            ))?;
        }

        self.input_format = Some(format);
        Ok(self)
    }

    /// Returns the native handle, which this value still owns.
    ///
    /// If code outside this crate changes the input layout or configuration through it, use the raw encode functions afterwards, because the settings tracked here no longer match.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpeg2kEncodeParams_t {
        self.raw
    }

    fn validate_input(&self, input: &EncodeInput<'_>) -> Result<()> {
        let index = self
            .active_config
            .ok_or(Error::UnsupportedEncodeConfiguration {
                reason: "encode configuration is not established".into(),
            })?;
        let config = &self.configs[index];
        let infos = config.components().collect::<Vec<_>>();
        let format = self
            .input_format
            .ok_or(Error::UnsupportedEncodeConfiguration {
                reason: "encode input format is not established".into(),
            })?;
        if format != input.format || input.image.component_count() != infos.len() {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "input type, component count or layout does not match encode configuration"
                    .into(),
            });
        }

        for (index, region) in input.regions.iter().enumerate() {
            let expected = infos[index].size();
            if format == ImageFormat::Interleaved
                && infos.iter().any(|info| info.size() != expected)
            {
                return Err(Error::UnsupportedEncodeConfiguration {
                    reason: "interleaved components must have equal dimensions".into(),
                });
            }

            if input.sizes[index].width() < expected.width()
                || input.sizes[index].height() < expected.height()
            {
                return Err(Error::OutOfRange {
                    name: "encode input dimensions".into(),
                });
            }

            let samples = if format == ImageFormat::Interleaved {
                infos.len()
            } else {
                1
            };
            let row_bytes = checked_mul(
                checked_mul(expected.width() as usize, samples, "input samples")?,
                input.image.pixel_type().element_size(),
                "input row bytes",
            )?;
            validate_plane(
                *region,
                expected,
                input.image.pitch_in_bytes(index)?,
                row_bytes,
            )?;
        }
        Ok(())
    }
}

impl EncodeConfig {
    /// Returns a builder for an image of `size`, written as `stream_type` in `color_space`.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn builder(
        size: Size,
        stream_type: BitstreamType,
        color_space: ColorSpace,
    ) -> Result<EncodeConfigBuilder> {
        EncodeConfigBuilder::create(size, stream_type, color_space)
    }

    /// Creates a configuration for a `width` by `height` image with the components in `component_info`, written as `stream_type` in `color_space`.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero, there are no components or more than four, or a component's bits per sample are not between 1 and 16.
    pub fn create(
        width: u32,
        height: u32,
        stream_type: BitstreamType,
        color_space: ColorSpace,
        component_info: Vec<ImageComponentInfo>,
    ) -> Result<Self> {
        validate_dimensions(Size::new(width, height), "image")?;
        if component_info.is_empty() {
            return Err(Error::Empty {
                name: "component_info".into(),
            });
        }

        if component_info.len() > MAX_IMAGE_COMPONENTS {
            return Err(Error::OutOfRange {
                name: "component_info".into(),
            });
        }

        for component in &component_info {
            validate_dimensions(component.size(), "component")?;
            validate_component_precision(component.precision())?;
        }

        let component_info = component_info
            .into_iter()
            .map(ImageComponentInfo::into_raw)
            .collect::<Vec<_>>();
        let mut config = Self {
            raw: sys::nvjpeg2kEncodeConfig_t::default(),
            component_info,
        };
        config.raw.stream_type = stream_type.into();
        config.raw.color_space = color_space.into();
        config.raw.image_width = width;
        config.raw.image_height = height;
        config.raw.num_components = to_u32(config.component_info.len(), "component count")?;
        config.raw.image_comp_info = config.component_info.as_mut_ptr();
        config.raw.prog_order = ProgressionOrder::Lrcp.into();
        config.raw.num_layers = 1;
        config.raw.num_resolutions = 1;
        config.raw.code_block_w = 64;
        config.raw.code_block_h = 64;
        Ok(config)
    }

    /// Splits the image into tiles of `width` by `height`, which should be smaller than the image.
    ///
    /// nvJPEG 2000 checks the tile size against the image when the configuration is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if a tile dimension is zero.
    pub fn set_tiling(&mut self, width: u32, height: u32) -> Result<&mut Self> {
        validate_dimensions(Size::new(width, height), "tile")?;
        self.raw.enable_tiling = 1;
        self.raw.tile_width = width;
        self.raw.tile_height = height;
        Ok(self)
    }

    /// Encodes the image as a single tile.
    pub fn clear_tiling(&mut self) -> &mut Self {
        self.raw.enable_tiling = 0;
        self.raw.tile_width = 0;
        self.raw.tile_height = 0;
        self
    }

    /// Sets the progression order.
    pub fn set_progression_order(&mut self, order: ProgressionOrder) -> &mut Self {
        self.raw.prog_order = order.into();
        self
    }

    /// Sets the number of resolution levels, which must not exceed what the image and tile size allow.
    ///
    /// nvJPEG 2000 checks the limit when the configuration is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if `resolutions` is zero.
    pub fn set_resolutions(&mut self, resolutions: u32) -> Result<&mut Self> {
        validate_resolutions(resolutions)?;
        self.raw.num_resolutions = resolutions;
        Ok(self)
    }

    /// Sets the code block size.
    ///
    /// # Errors
    ///
    /// Returns an error if `width` or `height` is not 32 or 64.
    pub fn set_code_block_size(&mut self, width: u32, height: u32) -> Result<&mut Self> {
        validate_code_block_size(Size::new(width, height))?;
        self.raw.code_block_w = width;
        self.raw.code_block_h = height;
        Ok(self)
    }

    /// Sets whether to use the irreversible wavelet transform, which is lossy, or the reversible one, which is lossless.
    pub fn set_irreversible(&mut self, irreversible: bool) -> &mut Self {
        self.raw.irreversible = u32::from(irreversible);
        self
    }

    /// Sets whether to apply the multiple-component transform; enable it for RGB input and disable it for YCC or grayscale.
    pub fn set_multi_component_transform(&mut self, enabled: bool) -> &mut Self {
        self.raw.mct_mode = u32::from(enabled);
        self
    }

    /// Sets whether to use High-Throughput JPEG 2000 (HTJ2K) block coding.
    ///
    /// This also sets the profile and encode-mode header fields HTJ2K needs, or restores the standard values.
    ///
    /// # Panics
    ///
    /// Panics if nvJPEG 2000's HTJ2K profile value does not fit the 16-bit profile field, which would mean the bindings are out of date.
    pub fn set_high_throughput(&mut self, enabled: bool) -> &mut Self {
        if enabled {
            self.raw.rsiz = u16::try_from(sys::NVJPEG2K_RSIZ_HT)
                .expect("the JPEG 2000 profile fits its ABI field");
            self.raw.encode_modes = sys::NVJPEG2K_MODE_HT;
        } else {
            self.raw.rsiz = 0;
            self.raw.encode_modes = 0;
        }
        self
    }

    /// Returns the image size in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        Size::new(self.raw.image_width, self.raw.image_height)
    }

    /// Returns the container.
    #[must_use]
    pub fn stream_type(&self) -> BitstreamType {
        BitstreamType::from(self.raw.stream_type)
    }

    /// Returns the color space.
    #[must_use]
    pub fn color_space(&self) -> ColorSpace {
        ColorSpace::from(self.raw.color_space)
    }

    /// Returns the number of components.
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_info.len()
    }

    /// Returns an iterator over the components.
    pub fn components(&self) -> impl ExactSizeIterator<Item = ImageComponentInfo> + '_ {
        self.component_info
            .iter()
            .copied()
            .map(ImageComponentInfo::from_raw)
    }

    /// Returns the tile size, or `None` if the image is one tile.
    #[must_use]
    pub const fn tiling(&self) -> Option<Size> {
        if self.raw.enable_tiling != 0 {
            Some(Size::new(self.raw.tile_width, self.raw.tile_height))
        } else {
            None
        }
    }

    /// Returns the progression order.
    #[must_use]
    pub fn progression_order(&self) -> ProgressionOrder {
        ProgressionOrder::from(self.raw.prog_order)
    }

    /// Returns the number of quality layers.
    #[must_use]
    pub const fn layer_count(&self) -> u32 {
        self.raw.num_layers
    }

    /// Returns the number of resolution levels.
    #[must_use]
    pub const fn resolution_count(&self) -> u32 {
        self.raw.num_resolutions
    }

    /// Returns the code block size.
    #[must_use]
    pub const fn code_block_size(&self) -> Size {
        Size::new(self.raw.code_block_w, self.raw.code_block_h)
    }

    /// Returns whether the irreversible wavelet transform is used.
    #[must_use]
    pub const fn irreversible(&self) -> bool {
        self.raw.irreversible != 0
    }

    /// Returns whether the multiple-component transform is applied.
    #[must_use]
    pub const fn multi_component_transform(&self) -> bool {
        self.raw.mct_mode != 0
    }

    /// Returns whether HTJ2K block coding is used.
    #[must_use]
    pub const fn high_throughput(&self) -> bool {
        self.raw.rsiz as u32 == sys::NVJPEG2K_RSIZ_HT
            && self.raw.encode_modes == sys::NVJPEG2K_MODE_HT
    }

    /// Returns a mutable pointer to the native configuration.
    pub fn as_raw_mut(&mut self) -> *mut sys::nvjpeg2kEncodeConfig_t {
        self.raw.image_comp_info = self.component_info.as_mut_ptr();
        &raw mut self.raw
    }
}

impl Clone for EncodeConfig {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            raw: self.raw,
            component_info: self.component_info.clone(),
        };
        cloned.raw.image_comp_info = cloned.component_info.as_mut_ptr();
        cloned
    }
}

impl Drop for EncoderHandleInner {
    fn drop(&mut self) {
        if self.cuda_context.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kEncoderDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kEncoderDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k encoder: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for EncodeState {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kEncodeStateDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kEncodeStateDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k encode state: {}",
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
            // SAFETY: `nvjpeg2kEncodeParamsDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kEncodeParamsDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k encode params: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
