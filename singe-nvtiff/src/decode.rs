//! TIFF decoding into device memory.

use std::{mem::ManuallyDrop, ptr, result::Result as StdResult, sync::Arc};

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::{
    context::Context,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::{AsStream, BorrowedStream, Stream as CudaStream},
};
use singe_nvtiff_sys as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    image::{DecodeOutputLayout, IfdOffset, Image, ImageInfo, ImagePlane},
    stream::TiffStream,
    try_ffi,
    types::OutputFormat,
    utility::{checked_pitched_extent, to_i32, to_size_t, to_u32, to_usize},
};

/// TIFF decoder.
///
/// It keeps its CUDA context and stream alive, and creates, runs, and destroys its native state on that stream.
/// The [`DecodeOptions`] set on it select the output format and the images or regions to decode.
/// It is `Send` and not `Sync`.
#[derive(Debug)]
pub struct Decoder {
    owner: Arc<DecoderOwner>,
    params: DecodeParams,
    options: DecodeOptions,
    stream: CudaStream,
}

/// Native decoder handle with the stream it is destroyed on, from [`Decoder::into_raw`].
///
/// Dropping it leaks the native decoder.
#[derive(Debug)]
pub struct RawDecoder {
    /// Native decoder handle, owned exclusively.
    pub handle: sys::nvtiffDecoder_t,
    /// Stream that creates and destroys the decoder, retaining its CUDA context.
    pub stream: CudaStream,
}

// Capture cleanup retains only the native owner, never the owning stream.
#[derive(Debug)]
struct DecoderOwner {
    raw: sys::nvtiffDecoder_t,
    stream: BorrowedStream,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the borrowed cleanup stream is only used to destroy the native owner"
)]
// SAFETY: an nvTIFF decoder has no thread affinity and is used through `&mut Decoder` only.
// The destroying thread binds the retained context first, and the stream wrapper or capture cleanup keeps the borrowed stream live.
unsafe impl Send for DecoderOwner {}

// SAFETY: shared references reach only the handle value; the native object is used through the owning wrapper's exclusive borrows and destroyed in `Drop`, which needs ownership.
unsafe impl Sync for DecoderOwner {}

impl Drop for DecoderOwner {
    fn drop(&mut self) {
        if let Err(error) = self.stream.context().bind() {
            report_cleanup_failure("bind the cuda context to destroy", "nvtiff decoder", &error);
            return;
        }

        // SAFETY: the last owner destroys the decoder once, with its context bound; the safe calls waited for their work, raw callers must finish theirs before dropping, and capture cleanup releases this owner only after capture ends.
        let destroyed =
            unsafe { try_ffi!(sys::nvtiffDecoderDestroy(self.raw, self.stream.as_raw())) };
        if let Err(error) = destroyed {
            report_cleanup_failure("destroy", "nvtiff decoder", &error);
        }
    }
}

/// Output format and images or regions to decode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecodeOptions {
    output_format: OutputFormat,
    regions: Vec<DecodeRegion>,
}

// Native decode settings, kept in sync with a decoder's `DecodeOptions`.
#[derive(Debug)]
struct DecodeParams {
    raw: sys::nvtiffDecodeParams_t,
}

// SAFETY: decode settings have no thread affinity, and nvTIFF reads them only during calls made through an exclusive `Decoder` borrow.
unsafe impl Send for DecodeParams {}

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
    /// Creates a decoder on `stream` with `allocators` and the default options.
    ///
    /// [`AllocatorSet::new`] selects nvTIFF's default allocators.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, the CUDA context cannot be bound, nvTIFF fails to create the decoder, or waiting for `stream` fails.
    pub fn create(mut allocators: AllocatorSet, stream: &CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        stream.complete(|| {
            stream.context().bind()?;
            // SAFETY: `nvtiffDecoderCreate` receives a live stream of the bound context, a writable output location, and allocators that are null for nvTIFF's defaults or come from unsafe constructors that guarantee callbacks following nvTIFF's contract and outliving the decoder.
            unsafe {
                try_ffi!(sys::nvtiffDecoderCreate(
                    &raw mut raw,
                    allocators.device_raw(),
                    allocators.pinned_raw(),
                    stream.as_raw()
                ))
            }
        })?;
        Self::from_created_raw(raw, stream)
    }

    fn from_created_raw(raw: sys::nvtiffDecoder_t, stream: &CudaStream) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let owner = Arc::new(DecoderOwner {
            raw,
            // SAFETY: the wrapper retains `stream` until this owner drops, directly or through the stream's capture cleanup.
            stream: unsafe { stream.to_unretained() },
        });
        Ok(Self {
            owner,
            params: DecodeParams::create()?,
            options: DecodeOptions::new(),
            stream: stream.clone(),
        })
    }

    /// Takes ownership of a native decoder, which is destroyed on `raw.stream`.
    ///
    /// The decoder starts with the default [`DecodeOptions`].
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, owned by nobody else, and created in `raw.stream`'s CUDA context.
    /// All work submitted with it must have finished.
    /// Custom allocator data must stay valid until the decoder has been destroyed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    /// Returns an error if nvTIFF fails to create the decode settings.
    pub unsafe fn from_raw(raw: RawDecoder) -> Result<Self> {
        let RawDecoder { handle, stream } = raw;
        Self::from_created_raw(handle, &stream)
    }

    /// Releases the native handle with its stream, without destroying it.
    ///
    /// After the submitted work finishes, destroy the handle on `stream` or pass it back to [`from_raw`](Self::from_raw).
    /// The decode options and native decode settings are released.
    ///
    /// # Errors
    ///
    /// Returns the decoder unchanged if something else still shares the native handle, such as cleanup for a captured graph.
    pub fn into_raw(self) -> StdResult<RawDecoder, Self> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the owner moves out once, and the `Err` arm writes it back before `this` is used again.
        let owner = unsafe { ptr::read(&raw const this.owner) };
        match Arc::try_unwrap(owner) {
            Ok(owner) => {
                let owner = ManuallyDrop::new(owner);
                // SAFETY: `this` is never dropped, so each remaining field moves out exactly once.
                let (params, options, stream) = unsafe {
                    (
                        ptr::read(&raw const this.params),
                        ptr::read(&raw const this.options),
                        ptr::read(&raw const this.stream),
                    )
                };
                drop((params, options));
                // SAFETY: `owner` is never dropped, so its borrowed stream moves out exactly once.
                drop(unsafe { ptr::read(&raw const owner.stream) });
                Ok(RawDecoder {
                    handle: owner.raw,
                    stream,
                })
            }
            Err(owner) => {
                // SAFETY: `this.owner` was moved out above, so writing it back restores the only copy.
                unsafe {
                    ptr::write(&raw mut this.owner, owner);
                }
                Err(ManuallyDrop::into_inner(this))
            }
        }
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

    /// Returns the stream the decoder runs on.
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
    /// Returns an error if stream capture is active or invalidated, or nvTIFF does not support the image, regions, output format, or compression.
    pub fn check_supported(&mut self, tiff: &TiffStream<'_>) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        // SAFETY: `tiff`, the decoder, and its settings are live owned handles, and a null output array asks nvTIFF to check the request without outputs.
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
    /// It also fails if stream capture is active or invalidated, or nvTIFF does not support the request.
    pub fn check_supported_with_outputs(
        &mut self,
        tiff: &TiffStream<'_>,
        outputs: &[Image<'_>],
    ) -> Result<()> {
        self.stream.ensure_not_capturing()?;
        self.validate_outputs(tiff, outputs)?;
        let raw = raw_images(outputs);
        // SAFETY: `tiff`, the decoder, and its settings are live owned handles, and `raw` holds one validated description per region whose arrays `outputs` keeps alive for the call.
        unsafe {
            try_ffi!(sys::nvtiffDecodeCheckSupported(
                tiff.as_raw(),
                self.as_raw(),
                self.params.raw,
                raw.as_ptr()
            ))
        }
    }

    /// Enqueues decoding of the requested images in `tiff` into `outputs` on the decoder's stream.
    ///
    /// The work is ordered after the queued writers of the output memory.
    /// Palette and `YCbCr` images are converted to RGB, and images with several samples per pixel are written in the layout of the selected [`OutputFormat`].
    ///
    /// # Safety
    ///
    /// The output memory must not be used by anything else until the work finishes.
    /// `outputs`, `tiff`, the output memory, and this decoder must stay alive until the work on the decoder's stream finishes, even on failure, and for every replay of a captured graph.
    /// Each request must finish before the decoder is reused or destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if the outputs are invalid, as described for [`decode_into`](Self::decode_into).
    /// It also fails if the context cannot be bound, the ordering after the output writers cannot be enqueued, or nvTIFF rejects the request.
    pub unsafe fn decode_async(
        &mut self,
        tiff: &TiffStream<'_>,
        outputs: &mut DecodeOutputs<'_>,
    ) -> Result<()> {
        // SAFETY: `outputs.raw` was built from `outputs.images` and describes them, and the caller meets `decode_async`'s exclusivity and lifetime contract, which `enqueue_decode` requires.
        unsafe { self.enqueue_decode(tiff, &outputs.images, &mut outputs.raw) }
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
        let mut raw = raw_images(outputs);
        let stream = self.stream.clone();
        stream.complete(|| {
            // SAFETY: `outputs` are exclusive borrows described by `raw`, and they, `tiff`, and this decoder outlive `complete`, which waits for the stream on every path.
            unsafe { self.enqueue_decode(tiff, outputs, &mut raw) }
        })
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
        let pitch = layout.size_bytes / to_usize(layout.height, "height")?;
        {
            let plane = ImagePlane::from_buffer(&mut output, pitch)?;
            let image = Image::from_planes([plane])?;
            self.decode_into(tiff, &mut [image])?;
        }
        // SAFETY: `decode_into` returned after the decode finished and wrote every row, and the pitch equals the row size, so the `size_bytes` prefix has no padding left uninitialized.
        Ok(unsafe { output.assume_init_prefix(layout.size_bytes)? })
    }

    // Validates the outputs, orders the decode after their writers, and enqueues it on the decoder's stream.
    // Safety: the contract of `decode_async` applies, with `raw` describing `images`.
    unsafe fn enqueue_decode(
        &mut self,
        tiff: &TiffStream<'_>,
        images: &[Image<'_>],
        raw: &mut [sys::nvtiffImage_t],
    ) -> Result<()> {
        self.validate_outputs(tiff, images)?;
        let regions = images
            .iter()
            .flat_map(|image| image.planes().iter().map(ImagePlane::region))
            .collect::<Vec<_>>();
        self.context().bind()?;
        self.stream.join_regions(&regions)?;
        // SAFETY: the outputs were validated against the decoder's options and context, the decode is ordered after their writers, and the caller keeps every participant alive until the work finishes.
        unsafe {
            try_ffi!(sys::nvtiffDecode(
                tiff.as_raw(),
                self.as_raw(),
                self.params.raw,
                raw.as_mut_ptr(),
                self.stream.as_raw()
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

            let height = to_usize(layout.height, "height")?;
            let row_bytes = layout.size_bytes / height;
            let sample_bytes = usize::from(layout.bits_per_sample).div_ceil(8).max(1);
            if plane.pitch_bytes() < row_bytes
                || !plane.pitch_bytes().is_multiple_of(sample_bytes)
                || !plane.region().as_ptr().addr().is_multiple_of(sample_bytes)
            {
                return Err(Error::OutOfRange {
                    name: "output pitch or sample alignment".into(),
                });
            }

            let required = checked_pitched_extent(
                height,
                plane.pitch_bytes(),
                row_bytes,
                "output plane size",
            )?;
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
        if let Err(error) = self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
        {
            abort_after_cleanup_failure("retain until capture ends", "nvtiff decoder", &error);
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
        // SAFETY: `raw` is a writable handle location that nvTIFF fills on success.
        unsafe {
            try_ffi!(sys::nvtiffDecodeParamsCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { raw })
    }

    fn set_output_format(&self, format: OutputFormat) -> Result<()> {
        // SAFETY: `self.raw` is a live settings handle owned by this value, and `format` converts to a valid native format.
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
        // SAFETY: `self.raw` is a live settings handle owned by this value, and `raw` holds the passed count of regions, which nvTIFF deep-copies before returning.
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
        // SAFETY: `self.raw` is owned by this value and destroyed once, when the decoder replaces or drops it; safe decodes have finished by then, and raw callers finish each request before reusing or dropping the decoder.
        if let Err(error) = unsafe { try_ffi!(sys::nvtiffDecodeParamsDestroy(self.raw)) } {
            report_cleanup_failure("destroy", "nvtiff decode params", &error);
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
            ifd_offset: to_size_t(self.ifd_offset.raw(), "ifd_offset")?,
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
