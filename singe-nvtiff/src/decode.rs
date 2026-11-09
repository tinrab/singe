//! TIFF decoding into device memory.

use std::{mem::ManuallyDrop, process, ptr, result::Result as StdResult, sync::Arc};

use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::{BorrowedStream, Stream as CudaStream},
};
use singe_nvtiff_sys as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    image::{DecodeOutputLayout, IfdOffset, Image, ImageInfo, ImagePlane},
    stream::TiffStream,
    try_ffi,
    types::OutputFormat,
    utility::{cleanup_stream, to_i32, to_u32},
};

/// TIFF decoder.
///
/// It keeps its CUDA context and stream alive, and creates and destroys its native state on that stream.
/// The [`DecodeOptions`] set on it select the output format and the images or regions to decode.
#[derive(Debug)]
pub struct Decoder {
    owner: Arc<DecoderOwner>,
    params: DecodeParams,
    options: DecodeOptions,
    stream: CudaStream,
}

// Capture cleanup retains only the native owner, never the owning stream.
#[derive(Debug)]
struct DecoderOwner {
    raw: sys::nvtiffDecoder_t,
    stream: BorrowedStream,
}

// SAFETY: native mutation requires an exclusive Decoder borrow. Final cleanup binds
// the retained context and the stream wrapper keeps this raw stream live.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
unsafe impl Send for DecoderOwner {}

unsafe impl Sync for DecoderOwner {}

impl Drop for DecoderOwner {
    fn drop(&mut self) {
        if self.stream.context().bind().is_err() {
            return;
        }
        // SAFETY: `nvtiffDecoderDestroy` receives live handles owned by the wrappers, and no queued work uses them.
        if let Err(error) =
            unsafe { try_ffi!(sys::nvtiffDecoderDestroy(self.raw, self.stream.as_raw())) }
        {
            eprintln!(
                "failed to destroy nvtiff decoder: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

/// Output format and images or regions to decode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecodeOptions {
    output_format: OutputFormat,
    regions: Vec<DecodeRegion>,
}

/// Native decode settings, kept in sync with a decoder's [`DecodeOptions`].
#[derive(Debug)]
struct DecodeParams {
    raw: sys::nvtiffDecodeParams_t,
}

/// Rectangular region of an image, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rectangle {
    /// Horizontal offset in pixels.
    pub x: u32,
    /// Vertical offset in pixels.
    pub y: u32,
    /// Width in pixels; 0 extends the region to the right edge.
    pub width: u32,
    /// Height in pixels; 0 extends the region to the bottom edge.
    pub height: u32,
}

/// Image, and optionally a region of it, to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeRegion {
    /// Offset of the image's directory; 0 selects the first image.
    pub ifd_offset: IfdOffset,
    /// Region to decode, or `None` for the whole image.
    pub rectangle: Option<Rectangle>,
}

/// Device outputs for [`Decoder::decode_async`], one per decoded image or region.
///
/// It owns the native description array passed to nvTIFF, so the array stays valid while raw work runs.
#[derive(Debug)]
pub struct DecodeOutputs<'a> {
    images: Vec<Image<'a>>,
    raw: Vec<sys::nvtiffImage_t>,
}

impl Decoder {
    /// Creates a decoder on `stream` with the default allocators and options.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the decoder, or waiting for `stream` fails.
    pub fn create(stream: &CudaStream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffDecoderCreateSimple` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffDecoderCreateSimple(
                    &raw mut raw,
                    stream.as_raw()
                ))
            })
        })??;
        Self::from_created_raw(raw, stream)
    }

    /// Creates a decoder on `stream` with custom allocators and the default options.
    ///
    /// # Safety
    ///
    /// The callbacks must allocate and free memory in stream order, as nvTIFF expects.
    /// The callbacks and their context data must stay valid until the decoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the decoder, or waiting for `stream` fails.
    pub unsafe fn create_with_allocators(
        mut allocators: AllocatorSet,
        stream: &CudaStream,
    ) -> Result<Self> {
        stream.ensure_not_capturing()?;
        let mut raw = ptr::null_mut();
        stream.sync_scope(|_| {
            // SAFETY: `nvtiffDecoderCreate` receives live handles owned by the wrappers and valid output locations.
            Ok(unsafe {
                try_ffi!(sys::nvtiffDecoderCreate(
                    &raw mut raw,
                    allocators.device_raw(),
                    allocators.pinned_raw(),
                    stream.as_raw()
                ))
            })
        })??;
        Self::from_created_raw(raw, stream)
    }

    fn from_created_raw(raw: sys::nvtiffDecoder_t, stream: &CudaStream) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let owner = Arc::new(DecoderOwner {
            raw,
            stream: cleanup_stream(stream),
        });
        Ok(Self {
            owner,
            params: DecodeParams::create()?,
            options: DecodeOptions::new(),
            stream: stream.clone(),
        })
    }

    /// Takes ownership of a native decoder, which is destroyed on `stream`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, owned by nobody else, and created in `stream`'s CUDA context.
    /// All work submitted with it must have finished.
    /// Custom allocator data must stay valid until the decoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null or nvTIFF fails to create the decode settings.
    pub unsafe fn from_raw(raw: sys::nvtiffDecoder_t, stream: &CudaStream) -> Result<Self> {
        Self::from_created_raw(raw, stream)
    }

    /// Releases the native handle without destroying it.
    ///
    /// After the submitted work finishes, destroy the handle on [`stream`](Self::stream) or pass it back to [`from_raw`](Self::from_raw).
    /// The handle does not keep the CUDA context or stream alive, so both must outlive it.
    ///
    /// # Errors
    ///
    /// Returns an error containing the decoder unchanged if something else still shares the native handle, such as cleanup for a captured graph.
    ///
    /// # Panics
    ///
    /// Panics if the handle becomes shared after the check, which indicates a bug in this crate.
    pub fn into_raw(self) -> StdResult<sys::nvtiffDecoder_t, Self> {
        if Arc::strong_count(&self.owner) != 1 {
            return Err(self);
        }

        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each field moves out exactly once.
        let (owner, params, options, stream) = unsafe {
            (
                ptr::read(&raw const this.owner),
                ptr::read(&raw const this.params),
                ptr::read(&raw const this.options),
                ptr::read(&raw const this.stream),
            )
        };
        drop((params, options, stream));
        let owner =
            ManuallyDrop::new(Arc::into_inner(owner).expect("the native owner is uniquely held"));
        // SAFETY: the owner is never dropped, so its stream moves out exactly once.
        drop(unsafe { ptr::read(&raw const owner.stream) });
        Ok(owner.raw)
    }

    /// Returns the native handle, which this decoder still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvtiffDecoder_t {
        self.owner.raw
    }

    /// Returns the CUDA context of the decoder.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Returns the stream that creates and destroys the decoder.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Returns the decode options.
    #[must_use]
    pub const fn options(&self) -> &DecodeOptions {
        &self.options
    }

    /// Sets the output format and the images or regions to decode.
    ///
    /// If the call fails, the previous options stay in effect.
    ///
    /// # Errors
    ///
    /// Returns an error if there are more regions than fit a `u32`, a region offset does not fit an `i32`, or nvTIFF rejects the format or regions.
    pub fn set_options(&mut self, options: DecodeOptions) -> Result<&mut Self> {
        let params = DecodeParams::create()?;
        params.set_output_format(options.output_format)?;
        params.set_regions(&options.regions)?;
        self.params = params;
        self.options = options;
        Ok(self)
    }

    /// Checks whether nvTIFF can decode the images in `tiff` with the decoder's options.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, or nvTIFF does not support the image, regions, output format, or compression.
    pub fn check_supported(&mut self, tiff: &TiffStream<'_>) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        // SAFETY: `nvtiffDecodeCheckSupported` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvtiffDecodeCheckSupported(
                tiff.as_raw(),
                self.as_raw(),
                self.params.raw,
                ptr::null()
            ))
        }
    }

    /// Checks whether nvTIFF can decode the images in `tiff` into `outputs` with the decoder's options.
    ///
    /// # Errors
    ///
    /// Returns an error if the outputs are invalid, as described for [`decode_into`](Self::decode_into).
    /// It also fails if stream capture is active or invalidated, the CUDA context cannot be bound, or nvTIFF does not support the request.
    pub fn check_supported_with_outputs(
        &mut self,
        tiff: &TiffStream<'_>,
        outputs: &[Image<'_>],
    ) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        self.validate_outputs(tiff, outputs)?;
        let raw = raw_images(outputs);
        // SAFETY: `nvtiffDecodeCheckSupported` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::nvtiffDecodeCheckSupported(
                tiff.as_raw(),
                self.as_raw(),
                self.params.raw,
                raw.as_ptr()
            ))
        }
    }

    /// Enqueues decoding of the requested images in `tiff` into `outputs` on `stream`.
    ///
    /// Palette and `YCbCr` images are converted to RGB, and images with several samples per pixel are written planar.
    ///
    /// # Safety
    ///
    /// The output memory must not be used by anything else until the work finishes.
    /// `outputs`, `tiff`, the output memory, and this decoder must stay alive until the work on `stream` finishes, even on failure, and for every replay of a captured graph.
    /// Each request must finish before the decoder is reused or destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if the outputs are invalid, as described for [`decode_into`](Self::decode_into).
    /// It also fails if `stream` belongs to another CUDA context, the context cannot be bound, or nvTIFF rejects the request.
    pub unsafe fn decode_async(
        &mut self,
        tiff: &TiffStream<'_>,
        outputs: &mut DecodeOutputs<'_>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.validate_outputs(tiff, &outputs.images)?;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.submit(tiff, &mut outputs.raw, stream) }
    }

    /// Decodes the requested images in `tiff` into `outputs` and waits for the work.
    ///
    /// Each output must have one plane.
    /// Only the image rows are written; the padding between rows stays uninitialized.
    /// The call waits for the GPU work even if it fails or unwinds.
    ///
    /// # Errors
    ///
    /// Returns an error if there is not one output per region, an output has more than one plane or is in another CUDA context, a region is invalid, an output is too small, or a pitch or sample is misaligned.
    /// It also fails if reading the image information or computing a size fails, stream capture is active or invalidated, the CUDA context cannot be bound, or decoding or waiting for the stream fails.
    pub fn decode_into(&mut self, tiff: &TiffStream<'_>, outputs: &mut [Image<'_>]) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        self.validate_outputs(tiff, outputs)?;
        let mut raw = raw_images(outputs);
        let regions = outputs
            .iter()
            .flat_map(|image| image.planes().iter().map(ImagePlane::region))
            .collect::<Vec<_>>();
        let stream = self.stream.clone();
        stream.sync_scope(|_| {
            Ok((|| -> Result<()> {
                stream.join_regions(&regions)?;
                // SAFETY: byte extents and contexts were validated. All host arrays,
                // exclusive output borrows, and dependencies outlive scope completion.
                unsafe { self.submit(tiff, &mut raw, &stream) }
            })())
        })??;
        Ok(())
    }

    /// Allocates tightly packed device memory, decodes one image or region of `tiff` into it, and waits for the work.
    ///
    /// # Errors
    ///
    /// Returns an error if the options select more than one region, the region or image layout is invalid, a size overflows, stream capture is active or invalidated, reading the image information fails, the allocation fails, or decoding fails.
    pub fn decode_to_device(&mut self, tiff: &TiffStream<'_>) -> Result<DeviceMemory<u8>> {
        self.stream.ensure_not_capturing()?;
        let region = self.options.single_region()?;
        let info = tiff.image_info(region.ifd_offset)?;
        let layout = self.options.output_layout(&info, region)?;
        let mut output = DeviceMemory::<u8>::create_uninit(self.context(), layout.size_bytes)?;
        let pitch = layout.size_bytes / layout.height as usize;
        {
            let plane = ImagePlane::from_buffer(&mut output, pitch)?;
            let image = Image::from_planes([plane])?;
            self.decode_into(tiff, &mut [image])?;
        }
        // SAFETY: successful completed dense decode writes every logical byte,
        // and this allocation has no inter-row or trailing padding.
        Ok(unsafe { output.assume_init_prefix(layout.size_bytes)? })
    }

    /// Enqueues the native decode.
    ///
    /// # Safety
    ///
    /// The contract of [`decode_async`](Self::decode_async) applies, with `raw` describing validated outputs.
    unsafe fn submit(
        &mut self,
        tiff: &TiffStream<'_>,
        raw: &mut [sys::nvtiffImage_t],
        stream: &CudaStream,
    ) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context().bind()?;
        // SAFETY: `nvtiffDecode` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvtiffDecode(
                tiff.as_raw(),
                self.as_raw(),
                self.params.raw,
                raw.as_mut_ptr(),
                stream.as_raw()
            ))
        }
    }

    fn validate_outputs(&self, tiff: &TiffStream<'_>, outputs: &[Image<'_>]) -> Result<()> {
        let expected = self.options.output_count();
        if outputs.len() != expected {
            return Err(Error::LengthMismatch {
                name: "outputs".into(),
                expected,
                actual: outputs.len(),
            });
        }

        for (index, image) in outputs.iter().enumerate() {
            if image.planes().len() != 1 {
                return Err(Error::LengthMismatch {
                    name: "output planes".into(),
                    expected: 1,
                    actual: image.planes().len(),
                });
            }

            let region = self
                .options
                .regions
                .get(index)
                .copied()
                .unwrap_or(DecodeRegion::full_image(IfdOffset::new(0)));
            let layout = self
                .options
                .output_layout(&tiff.image_info(region.ifd_offset)?, region)?;
            let plane = &image.planes()[0];
            if plane.context() != self.context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            let row_bytes = layout.size_bytes / layout.height as usize;
            let sample_bytes = (layout.bits_per_sample as usize).div_ceil(8).max(1);
            if plane.pitch_bytes() < row_bytes
                || !plane.pitch_bytes().is_multiple_of(sample_bytes)
                || !plane.region().as_ptr().addr().is_multiple_of(sample_bytes)
            {
                return Err(Error::OutOfRange {
                    name: "output pitch or sample alignment".into(),
                });
            }

            let required = (layout.height as usize - 1)
                .checked_mul(plane.pitch_bytes())
                .and_then(|bytes| bytes.checked_add(row_bytes))
                .ok_or_else(|| Error::OutOfRange {
                    name: "output plane size".into(),
                })?;
            if required > plane.region().byte_len() {
                return Err(Error::BufferTooSmall {
                    name: "output plane".into(),
                    required,
                    actual: plane.region().byte_len(),
                });
            }
        }
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
        }
    }
}

impl DecodeOptions {
    /// Creates options that decode the first whole image, with [`OutputFormat::UnchangedInterleaved`] output.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            output_format: OutputFormat::UnchangedInterleaved,
            regions: Vec::new(),
        }
    }

    /// Returns the output format.
    #[must_use]
    pub const fn output_format(&self) -> OutputFormat {
        self.output_format
    }

    /// Sets the output format.
    ///
    /// The output buffer size and pitch depend on the format:
    ///
    /// - [`UnchangedInterleaved`](OutputFormat::UnchangedInterleaved): `height * ceil(width * bits_per_pixel / 8)` bytes, with the pitch a multiple of `ceil(bits_per_sample / 8)` bytes and `ceil(width * bits_per_pixel / 8)` by default.
    /// - [`RgbInterleavedU8`](OutputFormat::RgbInterleavedU8): `width * height * 3` bytes, with any pitch.
    /// - [`RgbInterleavedU16`](OutputFormat::RgbInterleavedU16): `width * height * 6` bytes, with the pitch a multiple of 2 bytes.
    #[must_use]
    pub const fn with_output_format(mut self, output_format: OutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    /// Returns the regions to decode; empty means the first whole image.
    #[must_use]
    pub fn regions(&self) -> &[DecodeRegion] {
        &self.regions
    }

    /// Adds a region to decode.
    #[must_use]
    pub fn with_region(mut self, region: DecodeRegion) -> Self {
        self.regions.push(region);
        self
    }

    /// Replaces the regions to decode; an empty collection restores the first whole image.
    ///
    /// A whole image at any directory offset is a region without a rectangle.
    /// The regions are not tied to a TIFF stream: their offsets are read against the stream passed when decoding.
    /// Offsets, bounds, and codec support are checked by [`Decoder::check_supported`] and decoding.
    #[must_use]
    pub fn with_regions(mut self, regions: impl IntoIterator<Item = DecodeRegion>) -> Self {
        self.regions = regions.into_iter().collect();
        self
    }

    /// Returns the number of outputs a decode needs: the number of regions, or 1 for the default.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.regions.len().max(1)
    }

    fn single_region(&self) -> Result<DecodeRegion> {
        match self.regions.as_slice() {
            [] => Ok(DecodeRegion::full_image(IfdOffset::new(0))),
            [region] => Ok(*region),
            regions => Err(Error::LengthMismatch {
                name: "regions".into(),
                expected: 1,
                actual: regions.len(),
            }),
        }
    }

    fn output_layout(
        &self,
        image_info: &ImageInfo,
        region: DecodeRegion,
    ) -> Result<DecodeOutputLayout> {
        let (width, height) = if let Some(rectangle) = region.rectangle {
            let available_width = image_info
                .width()
                .checked_sub(rectangle.x)
                .ok_or_else(|| Error::out_of_range("region x"))?;
            let available_height = image_info
                .height()
                .checked_sub(rectangle.y)
                .ok_or_else(|| Error::out_of_range("region y"))?;
            let width = if rectangle.width == 0 {
                available_width
            } else {
                rectangle.width
            };
            let height = if rectangle.height == 0 {
                available_height
            } else {
                rectangle.height
            };
            if width == 0 || width > available_width {
                return Err(Error::out_of_range("region width"));
            }

            if height == 0 || height > available_height {
                return Err(Error::out_of_range("region height"));
            }
            (width, height)
        } else {
            (image_info.width(), image_info.height())
        };
        image_info.decode_layout_for_extent(self.output_format, width, height)
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeParams {
    fn create() -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `nvtiffDecodeParamsCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::nvtiffDecodeParamsCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { raw })
    }

    fn set_output_format(&self, format: OutputFormat) -> Result<()> {
        // SAFETY: `nvtiffDecodeParamsSetOutputFormat` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvtiffDecodeParamsSetOutputFormat(
                self.raw,
                format.into()
            ))
        }
    }

    fn set_regions(&self, regions: &[DecodeRegion]) -> Result<()> {
        if regions.is_empty() {
            return Ok(());
        }

        let raw = regions
            .iter()
            .map(|region| region.to_raw())
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: `nvtiffDecodeParamsSetRegions` receives live handles owned by the wrappers
        // and copies the borrowed regions.
        unsafe {
            try_ffi!(sys::nvtiffDecodeParamsSetRegions(
                self.raw,
                raw.as_ptr(),
                to_u32(raw.len(), "regions")?,
            ))
        }
    }
}

impl Drop for DecodeParams {
    fn drop(&mut self) {
        // SAFETY: `nvtiffDecodeParamsDestroy` receives live handles owned by the wrappers.
        if let Err(error) = unsafe { try_ffi!(sys::nvtiffDecodeParamsDestroy(self.raw)) } {
            eprintln!(
                "failed to destroy nvtiff decode params: {}",
                singe_core::ErrorChain(&error)
            );
        }
    }
}

impl Rectangle {
    /// Creates a region at (`x`, `y`) of the given size, without checking it against an image.
    ///
    /// A width or height of 0 extends the region to that edge of the image.
    ///
    /// # Errors
    ///
    /// Returns an error if an offset or size is negative or does not fit a `u32`.
    pub fn create(
        x: impl TryInto<u32>,
        y: impl TryInto<u32>,
        width: impl TryInto<u32>,
        height: impl TryInto<u32>,
    ) -> Result<Self> {
        Ok(Self {
            x: to_u32(x, "x")?,
            y: to_u32(y, "y")?,
            width: to_u32(width, "width")?,
            height: to_u32(height, "height")?,
        })
    }
}

impl DecodeRegion {
    /// Selects the whole image whose directory is at `ifd_offset`.
    #[must_use]
    pub const fn full_image(ifd_offset: IfdOffset) -> Self {
        Self {
            ifd_offset,
            rectangle: None,
        }
    }

    /// Selects `rectangle` of the image whose directory is at `ifd_offset`.
    #[must_use]
    pub const fn region(ifd_offset: IfdOffset, rectangle: Rectangle) -> Self {
        Self {
            ifd_offset,
            rectangle: Some(rectangle),
        }
    }

    fn to_raw(self) -> Result<sys::nvtiffDecodeRegion_t> {
        let rectangle = self.rectangle.unwrap_or(Rectangle {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        });
        Ok(sys::nvtiffDecodeRegion_t {
            ifd_offset: self.ifd_offset.raw() as sys::size_t,
            offset_x: to_i32(rectangle.x, "region x")?,
            offset_y: to_i32(rectangle.y, "region y")?,
            width: rectangle.width,
            height: rectangle.height,
        })
    }
}

impl<'a> DecodeOutputs<'a> {
    /// Collects the outputs, one per decoded image or region.
    #[must_use]
    pub fn new(images: impl IntoIterator<Item = Image<'a>>) -> Self {
        let images = images.into_iter().collect::<Vec<_>>();
        let raw = raw_images(&images);
        Self { images, raw }
    }

    /// Returns the outputs.
    #[must_use]
    pub fn images(&self) -> &[Image<'a>] {
        &self.images
    }

    /// Returns the number of outputs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Returns whether there are no outputs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Consumes the value and returns the outputs.
    #[must_use]
    pub fn into_images(self) -> Vec<Image<'a>> {
        self.images
    }

    /// Returns the native description array passed to nvTIFF.
    #[must_use]
    pub fn as_raw(&self) -> &[sys::nvtiffImage_t] {
        &self.raw
    }
}

fn raw_images(images: &[Image<'_>]) -> Vec<sys::nvtiffImage_t> {
    images.iter().map(|image| *image.as_raw()).collect()
}
