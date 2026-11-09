//! JPEG encoding.

mod encoder;
mod staged;

use std::{mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::Stream as CudaStream,
    view::{DeviceRegion, DeviceView},
};

pub use self::{
    encoder::{Encoder, EncoderBuilder},
    staged::{EncodeParams, EncoderState},
};
use crate::{
    error::{Error, Result},
    image::{Image, ImageView, MAX_COMPONENTS, Size},
    types::{ChromaSubsampling, InputFormat, JpegEncoding},
    utility::{ceil_div_u32, ceil_to_multiple_u32, to_i32},
};

/// Settings for an [`Encoder`]; unset values keep nvJPEG's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncodeOptions {
    quality: Option<Quality>,
    encoding: Option<JpegEncoding>,
    optimized_huffman: Option<bool>,
    sampling: Option<ChromaSubsampling>,
    restart_interval: Option<u32>,
}

/// JPEG quality from 1 (worst) to 100 (best).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quality(i32);

/// Image to encode, in device memory.
///
/// The layout depends on the [`InputFormat`]:
///
/// - planar RGB or BGR: three full-size planes;
/// - interleaved RGB or BGR: one plane with three bytes per pixel;
/// - YUV: one plane for grayscale and three otherwise, with the chroma planes' sizes rounded up for the subsampling;
/// - NV12: a full-size luma plane, and an interleaved chroma plane whose width in bytes is rounded up to even and whose height is half the image height, rounded up.
///
/// Pitches are in bytes.
#[derive(Debug, Clone)]
pub struct EncodeInput<'a> {
    context: &'a Arc<CudaContext>,
    regions: Vec<DeviceRegion<'a>>,
    source: Image,
    size: Size,
    input_subsampling: ChromaSubsampling,
    input_format: InputFormat,
}

/// Encoded JPEG data in device memory.
///
/// The allocation may be larger than the data; only the encoded bytes are exposed.
#[derive(Debug)]
pub struct EncodedDeviceBitstream {
    data: DeviceMemory<u8>,
}

impl Quality {
    /// Creates a quality from `value`.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` does not fit an `i32` or is not between 1 and 100.
    pub fn create(value: impl TryInto<i32>) -> Result<Self> {
        let value = to_i32(value, "quality")?;
        if !(1..=100).contains(&value) {
            return Err(Error::OutOfRange {
                name: "quality".into(),
            });
        }
        Ok(Self(value))
    }

    /// Returns the quality as an integer.
    #[must_use]
    pub const fn value(self) -> i32 {
        self.0
    }
}

impl TryFrom<i32> for Quality {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        Self::create(value)
    }
}

impl TryFrom<u32> for Quality {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Self::create(value)
    }
}

impl TryFrom<usize> for Quality {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self> {
        Self::create(value)
    }
}

impl From<Quality> for i32 {
    fn from(value: Quality) -> Self {
        value.value()
    }
}

impl EncodeOptions {
    /// Creates options that keep every nvJPEG default.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quality: None,
            encoding: None,
            optimized_huffman: None,
            sampling: None,
            restart_interval: None,
        }
    }

    /// Sets the quality.
    ///
    /// # Errors
    ///
    /// Returns an error if `quality` is not a valid [`Quality`].
    pub fn with_quality(mut self, quality: impl TryInto<Quality>) -> Result<Self> {
        self.quality = Some(quality.try_into().map_err(|_| Error::OutOfRange {
            name: "quality".into(),
        })?);
        Ok(self)
    }

    /// Sets the JPEG encoding process, such as baseline or progressive.
    #[must_use]
    pub const fn with_encoding(mut self, encoding: JpegEncoding) -> Self {
        self.encoding = Some(encoding);
        self
    }

    /// Sets whether to use optimized Huffman tables, which give smaller files of the same quality but encode more slowly.
    #[must_use]
    pub const fn with_optimized_huffman(mut self, optimized: bool) -> Self {
        self.optimized_huffman = Some(optimized);
        self
    }

    /// Sets the chroma subsampling of the output.
    #[must_use]
    pub const fn with_sampling(mut self, sampling: ChromaSubsampling) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// Sets the number of minimum coded units between restart markers.
    #[must_use]
    pub const fn with_restart_interval(mut self, interval: u32) -> Self {
        self.restart_interval = Some(interval);
        self
    }

    /// Returns the quality, if set.
    #[must_use]
    pub const fn quality(self) -> Option<Quality> {
        self.quality
    }

    /// Returns the encoding process, if set.
    #[must_use]
    pub const fn encoding(self) -> Option<JpegEncoding> {
        self.encoding
    }

    /// Returns whether optimized Huffman tables are used, if set.
    #[must_use]
    pub const fn optimized_huffman(self) -> Option<bool> {
        self.optimized_huffman
    }

    /// Returns the output chroma subsampling, if set.
    #[must_use]
    pub const fn sampling(self) -> Option<ChromaSubsampling> {
        self.sampling
    }

    /// Returns the restart interval, if set.
    #[must_use]
    pub const fn restart_interval(self) -> Option<u32> {
        self.restart_interval
    }

    fn apply_to(self, params: &mut EncodeParams, stream: &CudaStream) -> Result<()> {
        if let Some(quality) = self.quality {
            params.set_quality(quality, stream)?;
        }

        if let Some(encoding) = self.encoding {
            params.set_encoding(encoding, stream)?;
        }

        if let Some(optimized) = self.optimized_huffman {
            params.set_optimized_huffman(optimized, stream)?;
        }

        if let Some(sampling) = self.sampling {
            params.set_sampling_factors(sampling, stream)?;
        }

        if let Some(interval) = self.restart_interval {
            params.set_restart_interval(interval, stream)?;
        }
        Ok(())
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> EncodeInput<'a> {
    /// Creates an input from a native image description and the memory regions it points into.
    ///
    /// See [`EncodeInput`] for the layout of each format.
    ///
    /// # Safety
    ///
    /// Every pixel must be initialized in the layout of `input_format` and readable for `'a`.
    /// Work that writes the pixels must be ordered before every encode that reads them.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or does not fit an `i32`, the subsampling is unknown for YUV or not 4:2:0 for NV12, the planes do not match the format, a required pointer is null or an unused one is set, the planes are in different CUDA contexts or do not match the descriptor, or a pitch or plane is too small.
    pub unsafe fn from_raw_parts(
        source: Image,
        regions: Vec<DeviceRegion<'a>>,
        size: Size,
        input_subsampling: ChromaSubsampling,
        input_format: InputFormat,
    ) -> Result<Self> {
        validate_encode_input_metadata(size, input_subsampling, input_format)?;
        validate_encode_input_descriptor(&source, size, input_format, input_subsampling)?;
        let expected = expected_encode_channel_count(input_format, input_subsampling);
        if regions.len() != expected {
            return Err(Error::LengthMismatch {
                name: "input regions".into(),
                expected,
                actual: regions.len(),
            });
        }

        let context = regions[0].context();
        for (index, region) in regions.iter().enumerate() {
            if region.context() != context {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if region.as_ptr() != source.channel_ptr(index)? {
                return Err(Error::OutOfRange {
                    name: "input region pointer".into(),
                });
            }

            let height = match input_format {
                InputFormat::Yuv | InputFormat::Nv12 => {
                    expected_encode_plane_size(index, size, input_format, input_subsampling)?
                        .height()
                }
                _ => size.height(),
            } as usize;
            let row_bytes =
                expected_encode_channel_row_bytes(index, size, input_format, input_subsampling)?;
            let required = (height - 1)
                .checked_mul(source.channel_pitch(index)?)
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or(Error::OutOfRange {
                    name: "input plane extent".into(),
                })?;
            if region.byte_len() < required {
                return Err(Error::BufferTooSmall {
                    name: "input plane".into(),
                    required,
                    actual: region.byte_len(),
                });
            }
        }
        Ok(Self {
            context,
            regions,
            source,
            size,
            input_subsampling,
            input_format,
        })
    }

    /// Creates an input from the planes of `view`.
    ///
    /// See [`EncodeInput`] for the layout of each format.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or does not fit an `i32`, the subsampling is unknown for YUV or not 4:2:0 for NV12, the planes do not match the format, a required pointer is null or an unused one is set, the planes are in different CUDA contexts or do not match the descriptor, or a pitch or plane is too small, or the format is interleaved or does not match the plane sizes.
    pub fn from_view<const C: usize>(
        view: &ImageView<'a, u8, C>,
        input_subsampling: ChromaSubsampling,
        input_format: InputFormat,
    ) -> Result<Self> {
        validate_planar_input_format(input_format)?;
        validate_device_plane_count(C, input_format, input_subsampling)?;
        let size = infer_input_size_from_view(view)?;
        for (index, plane) in view.planes().iter().enumerate() {
            validate_encode_plane_size(index, plane.size(), size, input_format, input_subsampling)?;
        }

        let regions = view
            .planes()
            .iter()
            .map(super::image::Plane::region)
            .collect();
        // SAFETY: read-only planes establish initialized pixels and source borrows.
        unsafe {
            Self::from_raw_parts(
                view.descriptor()?,
                regions,
                size,
                input_subsampling,
                input_format,
            )
        }
    }

    /// Creates an input from separate planes, with each plane's size and pitch.
    ///
    /// See [`EncodeInput`] for the layout of each format.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or does not fit an `i32`, the subsampling is unknown for YUV or not 4:2:0 for NV12, the planes do not match the format, a required pointer is null or an unused one is set, the planes are in different CUDA contexts or do not match the descriptor, or a pitch or plane is too small, or the format is interleaved or does not match the plane sizes.
    pub fn from_planes<const C: usize>(
        planes: [DeviceView<'a, u8>; C],
        plane_sizes: [Size; C],
        row_bytes: [usize; C],
        size: Size,
        input_subsampling: ChromaSubsampling,
        input_format: InputFormat,
    ) -> Result<Self> {
        validate_plane_count(C)?;
        validate_planar_input_format(input_format)?;
        validate_device_plane_count(C, input_format, input_subsampling)?;
        let mut channels = [ptr::null_mut(); MAX_COMPONENTS];
        let mut pitches = [0; MAX_COMPONENTS];
        for index in 0..C {
            validate_encode_plane_size(
                index,
                plane_sizes[index],
                size,
                input_format,
                input_subsampling,
            )?;
            channels[index] = planes[index].as_ptr().cast_mut();
            pitches[index] = row_bytes[index];
        }

        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        let source = unsafe { Image::from_raw_parts(channels, pitches)? };
        let regions = planes.iter().map(DeviceView::region).collect();
        // SAFETY: initialized views retain every source allocation borrow.
        unsafe { Self::from_raw_parts(source, regions, size, input_subsampling, input_format) }
    }

    /// Creates an input from three full-size RGB or BGR planes, in the channel order of `input_format`.
    ///
    /// `row_bytes` holds each plane's pitch in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_format` is not planar RGB or BGR, a dimension is zero or too large, the planes are in different CUDA contexts, a pitch is shorter than a row, a size overflows, or a plane is too small.
    pub fn from_planar_rgb(
        planes: [DeviceView<'a, u8>; 3],
        size: Size,
        row_bytes: [usize; 3],
        input_format: InputFormat,
    ) -> Result<Self> {
        if !matches!(input_format, InputFormat::Rgb | InputFormat::Bgr) {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "planar RGB input requires RGB or BGR format".into(),
            });
        }
        Self::from_planes(
            planes,
            [size; 3],
            row_bytes,
            size,
            ChromaSubsampling::Unknown,
            input_format,
        )
    }

    /// Creates an input from interleaved RGB or BGR pixels, three bytes each.
    ///
    /// `row_bytes` is the pitch in bytes, including any padding.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_format` is not interleaved RGB or BGR, a dimension is zero or too large, the pointer or pitch is invalid, a size overflows, or `view` is too small.
    pub fn from_interleaved(
        view: DeviceView<'a, u8>,
        size: Size,
        row_bytes: usize,
        input_format: InputFormat,
    ) -> Result<Self> {
        if !matches!(
            input_format,
            InputFormat::RgbInterleaved | InputFormat::BgrInterleaved
        ) {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "interleaved input requires RGBI or BGRI format".into(),
            });
        }

        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        let source = unsafe {
            Image::from_raw_parts(
                [
                    view.as_ptr().cast_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ],
                [row_bytes, 0, 0, 0],
            )?
        };
        // SAFETY: the view grants initialized readable bytes for the full source borrow.
        unsafe { Self::image(source, vec![view.region()], size, input_format) }
    }

    /// Creates an RGB or BGR input from a native image description and the memory regions it points into.
    ///
    /// # Safety
    ///
    /// The requirements of [`from_raw_parts`](Self::from_raw_parts) apply.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_format` is YUV or NV12, or for the reasons [`from_raw_parts`](Self::from_raw_parts) gives.
    pub unsafe fn image(
        source: Image,
        regions: Vec<DeviceRegion<'a>>,
        size: Size,
        input_format: InputFormat,
    ) -> Result<Self> {
        if matches!(input_format, InputFormat::Yuv | InputFormat::Nv12) {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "RGB input helper requires an RGB or BGR format".into(),
            });
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::from_raw_parts(
                source,
                regions,
                size,
                ChromaSubsampling::Unknown,
                input_format,
            )
        }
    }

    /// Creates a planar YUV input from a native image description and the memory regions it points into.
    ///
    /// # Safety
    ///
    /// The requirements of [`from_raw_parts`](Self::from_raw_parts) apply.
    ///
    /// # Errors
    ///
    /// Returns an error for the reasons [`from_raw_parts`](Self::from_raw_parts) gives.
    pub unsafe fn yuv(
        source: Image,
        regions: Vec<DeviceRegion<'a>>,
        size: Size,
        subsampling: ChromaSubsampling,
    ) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_parts(source, regions, size, subsampling, InputFormat::Yuv) }
    }

    /// Returns the native image description, with the plane pointers and pitches.
    #[must_use]
    pub const fn source(&self) -> &Image {
        &self.source
    }

    /// Returns the image size in pixels.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Returns the chroma subsampling of the input.
    #[must_use]
    pub const fn input_subsampling(&self) -> ChromaSubsampling {
        self.input_subsampling
    }

    /// Returns the pixel layout of the input.
    #[must_use]
    pub const fn input_format(&self) -> InputFormat {
        self.input_format
    }

    /// Returns the CUDA context of the input's memory.
    #[must_use]
    pub fn cuda_context(&self) -> &'a Arc<CudaContext> {
        self.context
    }

    /// Returns the memory regions of the planes.
    #[must_use]
    pub fn regions(&self) -> &[DeviceRegion<'a>] {
        &self.regions
    }

    fn prepare(&self, stream: &CudaStream) -> Result<()> {
        Ok(stream.join_regions(&self.regions)?)
    }

    fn width_i32(&self) -> Result<i32> {
        to_i32(self.size.width(), "width")
    }

    fn height_i32(&self) -> Result<i32> {
        to_i32(self.size.height(), "height")
    }
}

impl EncodedDeviceBitstream {
    /// Creates a bitstream from the first `len` bytes of `data`.
    ///
    /// # Safety
    ///
    /// The work that wrote those bytes must have finished.
    unsafe fn create(data: DeviceMemory<MaybeUninit<u8>>, len: usize) -> Result<Self> {
        if len > data.len() {
            return Err(Error::BufferTooSmall {
                name: "data".into(),
                required: len,
                actual: data.len(),
            });
        }
        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            data: unsafe { data.assume_init_prefix(len)? },
        })
    }

    /// Returns the encoded bytes.
    #[must_use]
    pub const fn data(&self) -> &DeviceMemory<u8> {
        &self.data
    }

    /// Returns the encoded bytes as device memory.
    #[must_use]
    pub fn into_data(self) -> DeviceMemory<u8> {
        self.data
    }

    /// Returns the size of the encoded data in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the encoded data is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the size of the whole allocation in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.allocation().byte_len()
    }

    /// Copies the encoded bytes to host memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy fails.
    pub fn copy_to_host_vec(&self) -> Result<Vec<u8>> {
        Ok(self.data.copy_to_host_vec()?)
    }
}

fn validate_plane_count(count: usize) -> Result<()> {
    if count == 0 {
        return Err(Error::Empty {
            name: "planes".into(),
        });
    }

    if count > MAX_COMPONENTS {
        return Err(Error::OutOfRange {
            name: "planes".into(),
        });
    }
    Ok(())
}

fn validate_planar_input_format(input_format: InputFormat) -> Result<()> {
    if matches!(
        input_format,
        InputFormat::Rgb | InputFormat::Bgr | InputFormat::Yuv | InputFormat::Nv12
    ) {
        Ok(())
    } else {
        Err(Error::UnsupportedEncodeConfiguration {
            reason: "device planes encode input does not accept interleaved formats".into(),
        })
    }
}

fn validate_encode_input_metadata(
    size: Size,
    input_subsampling: ChromaSubsampling,
    input_format: InputFormat,
) -> Result<()> {
    if size.is_empty() {
        return Err(Error::Empty {
            name: "size".into(),
        });
    }

    to_i32(size.width(), "width")?;
    to_i32(size.height(), "height")?;
    match input_format {
        InputFormat::Yuv => {
            if input_subsampling == ChromaSubsampling::Unknown {
                return Err(Error::UnsupportedEncodeConfiguration {
                    reason: "YUV encode input requires known chroma subsampling".into(),
                });
            }
        }
        InputFormat::Nv12 => {
            if input_subsampling != ChromaSubsampling::Css420 {
                return Err(Error::UnsupportedEncodeConfiguration {
                    reason: "NV12 encode input requires 4:2:0 chroma subsampling".into(),
                });
            }
        }
        InputFormat::Rgb
        | InputFormat::Bgr
        | InputFormat::RgbInterleaved
        | InputFormat::BgrInterleaved => {}
    }
    Ok(())
}

fn validate_encode_input_descriptor(
    source: &Image,
    size: Size,
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> Result<()> {
    let expected = expected_encode_channel_count(input_format, input_subsampling);
    for index in 0..expected {
        if !source.has_channel(index)? {
            return Err(Error::NullPointer {
                name: "channel".into(),
            });
        }

        let actual_pitch = source.channel_pitch(index)?;
        let required_pitch =
            expected_encode_channel_row_bytes(index, size, input_format, input_subsampling)?;
        if actual_pitch < required_pitch {
            return Err(Error::OutOfRange {
                name: "pitch".into(),
            });
        }
    }

    for index in expected..MAX_COMPONENTS {
        if source.has_channel(index)? {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "encode input descriptor has channels not used by format".into(),
            });
        }
    }
    Ok(())
}

fn expected_encode_channel_row_bytes(
    index: usize,
    size: Size,
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> Result<usize> {
    match input_format {
        InputFormat::Rgb | InputFormat::Bgr => Ok(size.width() as usize),
        InputFormat::RgbInterleaved | InputFormat::BgrInterleaved => (size.width() as usize)
            .checked_mul(3)
            .ok_or(Error::OutOfRange {
                name: "row byte length".into(),
            }),
        InputFormat::Yuv | InputFormat::Nv12 => {
            Ok(
                expected_encode_plane_size(index, size, input_format, input_subsampling)?.width()
                    as usize,
            )
        }
    }
}

fn expected_encode_channel_count(
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> usize {
    match input_format {
        InputFormat::RgbInterleaved | InputFormat::BgrInterleaved => 1,
        InputFormat::Nv12 => 2,
        InputFormat::Yuv if input_subsampling == ChromaSubsampling::Gray => 1,
        InputFormat::Rgb | InputFormat::Bgr | InputFormat::Yuv => 3,
    }
}

fn validate_device_plane_count(
    count: usize,
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> Result<()> {
    validate_plane_count(count)?;
    let expected = match input_format {
        InputFormat::Nv12 => 2,
        InputFormat::Yuv if input_subsampling == ChromaSubsampling::Gray => 1,
        InputFormat::Rgb | InputFormat::Bgr | InputFormat::Yuv => 3,
        InputFormat::RgbInterleaved | InputFormat::BgrInterleaved => {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "device planes encode input does not accept interleaved formats".into(),
            });
        }
    };
    if count != expected {
        return Err(Error::LengthMismatch {
            name: "planes".into(),
            expected,
            actual: count,
        });
    }
    Ok(())
}

fn infer_input_size_from_view<const C: usize>(view: &ImageView<'_, u8, C>) -> Result<Size> {
    let [first, ..] = view.planes().as_slice() else {
        return Err(Error::Empty {
            name: "planes".into(),
        });
    };
    Ok(first.size())
}

fn validate_encode_plane_size(
    index: usize,
    actual: Size,
    size: Size,
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> Result<()> {
    let expected = expected_encode_plane_size(index, size, input_format, input_subsampling)?;
    if actual != expected {
        return Err(Error::UnsupportedEncodeConfiguration {
            reason: "encode input plane dimensions do not match format and subsampling".into(),
        });
    }
    Ok(())
}

fn expected_encode_plane_size(
    index: usize,
    size: Size,
    input_format: InputFormat,
    input_subsampling: ChromaSubsampling,
) -> Result<Size> {
    match input_format {
        InputFormat::Rgb | InputFormat::Bgr => Ok(size),
        InputFormat::Yuv => expected_yuv_plane_size(index, size, input_subsampling),
        InputFormat::Nv12 => match index {
            0 => Ok(size),
            1 => Ok(Size::new(
                ceil_to_multiple_u32(size.width(), 2, "nv12 uv width")?,
                ceil_div_u32(size.height(), 2),
            )),
            _ => Err(Error::OutOfRange {
                name: "plane".into(),
            }),
        },
        InputFormat::RgbInterleaved | InputFormat::BgrInterleaved => {
            Err(Error::UnsupportedEncodeConfiguration {
                reason: "device planes encode input does not accept interleaved formats".into(),
            })
        }
    }
}

fn expected_yuv_plane_size(
    index: usize,
    size: Size,
    subsampling: ChromaSubsampling,
) -> Result<Size> {
    if index == 0 {
        return Ok(size);
    }

    let (horizontal, vertical) = match subsampling {
        ChromaSubsampling::Css444 => (1, 1),
        ChromaSubsampling::Css422 => (2, 1),
        ChromaSubsampling::Css420 => (2, 2),
        ChromaSubsampling::Css440 => (1, 2),
        ChromaSubsampling::Css411 => (4, 1),
        ChromaSubsampling::Css410 => (4, 2),
        ChromaSubsampling::Css410V => (2, 4),
        ChromaSubsampling::Gray => {
            return Err(Error::OutOfRange {
                name: "plane".into(),
            });
        }
        ChromaSubsampling::Unknown => {
            return Err(Error::UnsupportedEncodeConfiguration {
                reason: "YUV encode input requires known chroma subsampling".into(),
            });
        }
    };
    Ok(Size::new(
        ceil_div_u32(size.width(), horizontal),
        ceil_div_u32(size.height(), vertical),
    ))
}
