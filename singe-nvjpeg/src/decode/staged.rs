use std::{ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    context::{Context, JpegState},
    decode::{DecodeOptions, Rectangle, to_positive_i32, validate_batch},
    error::{Error, Result},
    image::Image,
    try_ffi,
    types::{ChromaSubsampling, ExifOrientation, OutputFormat, ScaleFactor},
    utility::{to_i32, to_non_zero_u64, to_u64},
};

/// Decode settings for nvJPEG's staged and batched decoding: output format, region, CMYK conversion, scaling, and EXIF orientation.
#[derive(Debug)]
pub struct DecodeParams {
    raw: sys::nvjpegDecodeParams_t,
    pub(crate) handle: Context,
    pub(crate) options: DecodeOptions,
}

impl DecodeParams {
    /// Returns the CUDA context of the parameters.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates decode settings for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create them.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegDecodeParamsCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsCreate(handle.as_raw(), &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
            options: DecodeOptions::new().with_output_format(OutputFormat::Unchanged),
        })
    }

    /// Sets the output format; [`OutputFormat::Unchanged`] by default.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG rejects the format.
    pub fn set_output_format(&mut self, format: OutputFormat) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpegDecodeParamsSetOutputFormat` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetOutputFormat(
                self.raw,
                format.into()
            ))?;
        }

        self.options.output_format = format;
        Ok(self)
    }

    /// Decodes only the region at (`offset_x`, `offset_y`) of size `width` by `height`.
    ///
    /// The region must lie inside the image, and with [`OutputFormat::Yuv`] or [`OutputFormat::Unchanged`] its offsets must be multiples of the largest chroma subsampling factor.
    /// Region decoding is off by default and not supported by [`Backend::Hardware`](crate::types::Backend::Hardware).
    /// With an EXIF orientation set, the region is in rotated coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a value does not fit an `i32`, an offset is negative, a size is not positive, an end coordinate overflows, or nvJPEG rejects the region.
    /// The image bounds and subsampling rules are checked when decoding.
    pub fn set_roi(
        &mut self,
        offset_x: impl TryInto<i32>,
        offset_y: impl TryInto<i32>,
        width: impl TryInto<i32>,
        height: impl TryInto<i32>,
    ) -> Result<&mut Self> {
        self.handle.bind()?;

        let roi = Rectangle::create(
            to_i32(offset_x, "offset_x")?,
            to_i32(offset_y, "offset_y")?,
            to_i32(width, "width")?,
            to_i32(height, "height")?,
        )?;
        // SAFETY: `nvjpegDecodeParamsSetROI` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetROI(
                self.raw, roi.x, roi.y, roi.width, roi.height
            ))?;
        }

        self.options.roi = Some(roi);
        Ok(self)
    }

    /// Decodes the whole image again, after [`set_roi`](Self::set_roi).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG rejects the change.
    pub fn clear_roi(&mut self) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpegDecodeParamsSetROI` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetROI(self.raw, 0, 0, -1, -1))?;
        }

        self.options.roi = None;
        Ok(self)
    }

    /// Sets whether to treat four-component images as CMYK and convert them to RGB or YUV; off by default.
    ///
    /// The conversion is subtractive and matches how `OpenCV` handles four-component JPEG files.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG rejects the setting.
    pub fn set_allow_cmyk(&mut self, allow: bool) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpegDecodeParamsSetAllowCMYK` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetAllowCMYK(
                self.raw,
                i32::from(allow)
            ))?;
        }

        self.options.allow_cmyk = allow;
        Ok(self)
    }

    /// Sets how much to scale the output down.
    ///
    /// Only [`Backend::Hardware`](crate::types::Backend::Hardware) supports scaling.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG rejects the scale factor.
    pub fn set_scale_factor(&mut self, scale_factor: ScaleFactor) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpegDecodeParamsSetScaleFactor` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetScaleFactor(
                self.raw,
                scale_factor.into()
            ))?;
        }

        self.options.scale_factor = Some(scale_factor);
        Ok(self)
    }

    /// Rotates and flips the output as `orientation` says.
    ///
    /// Allocate the output for the rotated size.
    /// [`ExifOrientation::Unknown`] is treated as [`ExifOrientation::Normal`].
    /// A region set with [`set_roi`](Self::set_roi) is in rotated coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG rejects the orientation.
    pub fn set_exif_orientation(&mut self, orientation: ExifOrientation) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpegDecodeParamsSetExifOrientation` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeParamsSetExifOrientation(
                self.raw,
                orientation.into()
            ))?;
        }

        self.options.exif_orientation = Some(orientation);
        Ok(self)
    }

    /// Returns the native handle, which this value still owns.
    ///
    /// If code outside this crate changes the settings through it, use the raw decode functions afterwards, because the settings tracked here no longer match.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegDecodeParams_t {
        self.raw
    }
}

impl Context {
    /// Enqueues decoding of the JPEG in `data` into `destination` on `stream`, using `state`.
    ///
    /// nvJPEG chooses the backend for each image; use the staged decoding of a [`Decoder`](crate::decode::Decoder) with a backend to choose it yourself.
    ///
    /// # Safety
    ///
    /// `destination` must describe device-accessible buffers large enough for `output_format`, valid and used by nothing else until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `state` was created with another [`Context`], `data` is empty or too long for nvJPEG, or nvJPEG fails.
    pub unsafe fn decode_async(
        &self,
        state: &mut JpegState,
        data: &[u8],
        output_format: OutputFormat,
        destination: &mut Image,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;

        // SAFETY: `nvjpegDecode` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvjpegDecode(
                self.as_raw(),
                state.as_raw(),
                data.as_ptr(),
                to_non_zero_u64(data.len(), "data length")?,
                output_format.into(),
                destination.as_raw_mut(),
                stream.as_raw()
            ))
        }
    }

    /// Prepares `state` for batched decoding with a batch size, CPU thread limit, and output format.
    ///
    /// Call it once before decoding batches, and not while a batch is being decoded.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `state` was created with another [`Context`], `batch_size` or `max_cpu_threads` does not fit a positive `i32`, or nvJPEG fails.
    pub fn decode_batched_initialize(
        &self,
        state: &mut JpegState,
        batch_size: impl TryInto<i32>,
        max_cpu_threads: impl TryInto<i32>,
        output_format: OutputFormat,
    ) -> Result<()> {
        self.bind()?;
        self.ensure_handle(&state.handle)?;

        let batch_size = to_positive_i32(batch_size, "batch_size")?;
        let max_cpu_threads = to_positive_i32(max_cpu_threads, "max_cpu_threads")?;
        // SAFETY: `nvjpegDecodeBatchedInitialize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedInitialize(
                self.as_raw(),
                state.as_raw(),
                batch_size,
                max_cpu_threads,
                output_format.into()
            ))
        }
    }

    /// Allocates device memory in `state` ahead of batched decoding, for images up to `width` by `height`.
    ///
    /// This avoids repeated allocations when images differ in size or subsampling.
    /// Only the hardware backends use it, and nvJPEG resizes the memory later if an image is larger.
    /// Use [`ChromaSubsampling::Css444`] when the images' subsampling varies, so the memory can be reused.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `state` was created with another [`Context`], `batch_size`, `width`, or `height` does not fit a positive `i32`, or nvJPEG fails.
    pub fn decode_batched_preallocate(
        &self,
        state: &mut JpegState,
        batch_size: impl TryInto<i32>,
        width: impl TryInto<i32>,
        height: impl TryInto<i32>,
        subsampling: ChromaSubsampling,
        output_format: OutputFormat,
    ) -> Result<()> {
        self.bind()?;
        self.ensure_handle(&state.handle)?;

        let batch_size = to_positive_i32(batch_size, "batch_size")?;
        let width = to_positive_i32(width, "width")?;
        let height = to_positive_i32(height, "height")?;
        // SAFETY: `nvjpegDecodeBatchedPreAllocate` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedPreAllocate(
                self.as_raw(),
                state.as_raw(),
                batch_size,
                width,
                height,
                subsampling.into(),
                output_format.into()
            ))
        }
    }

    /// Sets external Huffman and quantization tables in `state` for later batches.
    ///
    /// Use it for JPEG data from TIFF files, whose tables are stored separately.
    /// The tables apply to every image in the batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `state` was created with another [`Context`], the tables are too long for nvJPEG, or nvJPEG rejects them.
    pub fn decode_batched_parse_jpeg_tables(
        &self,
        state: &mut JpegState,
        tables: Option<&[u8]>,
    ) -> Result<()> {
        self.bind()?;
        self.ensure_handle(&state.handle)?;

        let (ptr, len) = tables.map_or((ptr::null(), 0), |tables| (tables.as_ptr(), tables.len()));
        // SAFETY: `nvjpegDecodeBatchedParseJpegTables` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedParseJpegTables(
                self.as_raw(),
                state.as_raw(),
                ptr,
                to_u64(len, "tables length")?
            ))
        }
    }

    /// Enqueues decoding of `inputs` into `destinations` on `stream`, in the output format set by [`decode_batched_initialize`](Self::decode_batched_initialize).
    ///
    /// # Safety
    ///
    /// Each destination must describe device-accessible buffers large enough for that output format, valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `state` was created with another [`Context`], `destinations` and `inputs` have different lengths, an input is empty or too long for nvJPEG, or nvJPEG fails.
    pub unsafe fn decode_batched_async(
        &self,
        state: &mut JpegState,
        inputs: &[&[u8]],
        destinations: &mut [Image],
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;

        validate_batch(inputs.len(), destinations.len(), "destinations")?;
        let data = inputs
            .iter()
            .map(|input| input.as_ptr())
            .collect::<Vec<_>>();
        let lengths = inputs
            .iter()
            .map(|input| to_non_zero_u64(input.len(), "input length"))
            .collect::<Result<Vec<_>>>()?;
        let mut raw_destinations = destinations
            .iter_mut()
            // SAFETY: the caller upholds this function's safety contract.
            .map(|destination| unsafe { *destination.as_raw_mut() })
            .collect::<Vec<_>>();

        // SAFETY: `nvjpegDecodeBatched` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatched(
                self.as_raw(),
                state.as_raw(),
                data.as_ptr(),
                lengths.as_ptr(),
                raw_destinations.as_mut_ptr(),
                stream.as_raw()
            ))
        }
    }

    /// Enqueues decoding of `inputs` into `destinations` on `stream`, with separate decode settings, such as a region, from `params` for each image.
    ///
    /// # Safety
    ///
    /// Each destination must describe device-accessible buffers large enough for its image's settings, valid until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `state` or a [`DecodeParams`] was created with another [`Context`], `inputs`, `destinations`, and `params` have different lengths, an input is empty or too long for nvJPEG, or nvJPEG fails.
    pub unsafe fn decode_batched_with_params_async(
        &self,
        state: &mut JpegState,
        inputs: &[&[u8]],
        destinations: &mut [Image],
        params: &mut [&mut DecodeParams],
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle)?;

        validate_batch(inputs.len(), destinations.len(), "destinations")?;
        validate_batch(inputs.len(), params.len(), "params")?;
        for params in params.iter() {
            self.ensure_handle(&params.handle)?;
        }

        let data = inputs
            .iter()
            .map(|input| input.as_ptr())
            .collect::<Vec<_>>();
        let lengths = inputs
            .iter()
            .map(|input| to_non_zero_u64(input.len(), "input length"))
            .collect::<Result<Vec<_>>>()?;
        let mut raw_destinations = destinations
            .iter_mut()
            // SAFETY: the caller upholds this function's safety contract.
            .map(|destination| unsafe { *destination.as_raw_mut() })
            .collect::<Vec<_>>();
        let mut raw_params = params
            .iter_mut()
            .map(|param| param.as_raw())
            .collect::<Vec<_>>();

        // SAFETY: `nvjpegDecodeBatchedEx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvjpegDecodeBatchedEx(
                self.as_raw(),
                state.as_raw(),
                data.as_ptr(),
                lengths.as_ptr(),
                raw_destinations.as_mut_ptr(),
                raw_params.as_mut_ptr(),
                stream.as_raw()
            ))
        }
    }
}

impl Drop for DecodeParams {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegDecodeParamsDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegDecodeParamsDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg decode params: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
