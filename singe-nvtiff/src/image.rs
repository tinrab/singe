//! Image information, directory offsets, and decode outputs.

use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    ptr::NonNull,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    view::{DeviceOutput, DeviceRegion},
};
use singe_nvtiff_sys as sys;

use crate::{
    error::{Error, Result},
    types::{
        Compression, ImageGeometryType, ImageTypeFlags, OutputFormat, PhotometricInterpretation,
        PlanarConfig, SampleFormat,
    },
    utility::{to_size_t, to_u32, to_usize},
};

/// Offset of an image file directory (IFD) in a TIFF file.
///
/// Each image has its own directory; offset 0 selects the first image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct IfdOffset(usize);

/// Information about an image in a TIFF file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    raw: sys::nvtiffImageInfo_t,
}

/// Image information checked for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncodeImageInfo {
    raw: sys::nvtiffImageInfo_t,
}

/// Strip or tile layout of an image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ImageGeometry {
    raw: sys::nvtiffImageGeometry_t,
}

/// Size and format of a decoded image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeOutputLayout {
    /// Output format.
    pub format: OutputFormat,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Size of a tightly packed output in bytes.
    pub size_bytes: usize,
    /// Samples per pixel.
    pub samples_per_pixel: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
}

/// Device output of one decoded image.
///
/// It owns the pointer and pitch arrays passed to nvTIFF and borrows the plane memory.
#[derive(Debug)]
pub struct Image<'a> {
    raw: sys::nvtiffImage_t,
    _plane_data: Vec<*mut c_void>,
    _plane_pitch_bytes: Vec<sys::size_t>,
    planes: Vec<ImagePlane<'a>>,
}

/// Device memory for one plane of a decoded image.
#[derive(Debug)]
pub struct ImagePlane<'a> {
    region: DeviceRegion<'a>,
    pitch_bytes: usize,
}

impl IfdOffset {
    /// Wraps an offset without checking it against a file.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the offset in bytes.
    #[must_use]
    pub const fn raw(self) -> usize {
        self.0
    }
}

impl From<usize> for IfdOffset {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl TryFrom<i32> for IfdOffset {
    type Error = Error;

    fn try_from(value: i32) -> StdResult<Self, Self::Error> {
        Ok(Self::new(to_usize(value, "ifd_offset")?))
    }
}

impl Display for IfdOffset {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl ImageInfo {
    /// Wraps native image information.
    #[must_use]
    pub const fn from_raw(raw: sys::nvtiffImageInfo_t) -> Self {
        Self { raw }
    }

    /// Returns the native image information.
    #[must_use]
    pub const fn as_raw(&self) -> &sys::nvtiffImageInfo_t {
        &self.raw
    }

    /// Returns the image type from the `NewSubfileType` tag.
    #[must_use]
    pub const fn image_type(&self) -> ImageTypeFlags {
        ImageTypeFlags::from_bits_retain(self.raw.image_type)
    }

    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.raw.image_width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.raw.image_height
    }

    /// Returns the number of bits per pixel.
    #[must_use]
    pub const fn bits_per_pixel(&self) -> u16 {
        self.raw.bits_per_pixel
    }

    /// Returns the number of samples per pixel.
    #[must_use]
    pub const fn samples_per_pixel(&self) -> u16 {
        self.raw.samples_per_pixel
    }

    /// Returns the number of bits of sample `index`, or `None` if there is no such sample.
    #[must_use]
    pub fn bits_per_sample(&self, index: usize) -> Option<u16> {
        self.raw
            .bits_per_sample
            .get(..self.samples_per_pixel() as usize)
            .and_then(|samples| samples.get(index))
            .copied()
    }

    /// Returns the format of sample `index`, or `None` if there is no such sample.
    pub fn sample_format(&self, index: usize) -> Option<SampleFormat> {
        self.raw
            .sample_format
            .get(..self.samples_per_pixel() as usize)
            .and_then(|formats| formats.get(index))
            .copied()
            .map(SampleFormat::from)
    }

    /// Returns the color space of the pixel values.
    #[must_use]
    pub fn photometric_interpretation(&self) -> PhotometricInterpretation {
        self.raw.photometric_int.into()
    }

    /// Returns the planar configuration.
    #[must_use]
    pub fn planar_config(&self) -> PlanarConfig {
        self.raw.planar_config.into()
    }

    /// Returns the compression.
    #[must_use]
    pub fn compression(&self) -> Compression {
        self.raw.compression.into()
    }

    /// Returns the output layout that [`Decoder::decode_to_device`](crate::decode::Decoder::decode_to_device) uses for the whole image.
    ///
    /// Palette images decode to [`OutputFormat::RgbInterleavedU16`], JPEG-compressed `YCbCr` images to [`OutputFormat::RgbInterleavedU8`], and others to [`OutputFormat::UnchangedInterleaved`].
    ///
    /// # Errors
    ///
    /// Returns an error if the width or height is 0, the size overflows `usize`, or unchanged output is requested for an image without sample sizes.
    pub fn default_decode_layout(&self) -> Result<DecodeOutputLayout> {
        let format = match (self.photometric_interpretation(), self.compression()) {
            (PhotometricInterpretation::Palette, _) => OutputFormat::RgbInterleavedU16,
            (PhotometricInterpretation::Ycbcr, Compression::Jpeg) => OutputFormat::RgbInterleavedU8,
            _ => OutputFormat::UnchangedInterleaved,
        };
        self.decode_layout(format)
    }

    /// Returns the output layout of the whole image in `format`.
    ///
    /// # Errors
    ///
    /// Returns an error if the width or height is 0, the size overflows `usize`, or unchanged output is requested for an image without sample sizes.
    pub fn decode_layout(&self, format: OutputFormat) -> Result<DecodeOutputLayout> {
        self.decode_layout_for_extent(format, self.width(), self.height())
    }

    /// Returns the output layout of a `width` by `height` region in `format`.
    ///
    /// # Errors
    ///
    /// Returns an error if the width or height is 0, the size overflows `usize`, or unchanged output is requested for an image without sample sizes.
    pub fn decode_layout_for_extent(
        &self,
        format: OutputFormat,
        width: u32,
        height: u32,
    ) -> Result<DecodeOutputLayout> {
        let pixels = checked_pixels(width, height)?;
        let (size_bytes, samples_per_pixel, bits_per_sample) = match format {
            OutputFormat::UnchangedInterleaved => {
                let row_bits = (width as usize)
                    .checked_mul(self.bits_per_pixel() as usize)
                    .ok_or_else(|| Error::OutOfRange {
                        name: "decoded row size".into(),
                    })?;
                let size_bytes = row_bits
                    .div_ceil(8)
                    .checked_mul(height as usize)
                    .ok_or_else(|| Error::OutOfRange {
                        name: "decoded image size".into(),
                    })?;
                let bits_per_sample = self
                    .bits_per_sample(0)
                    .ok_or_else(|| Error::out_of_range("bits per sample"))?;
                (size_bytes, self.samples_per_pixel(), bits_per_sample)
            }
            OutputFormat::RgbInterleavedU8 => (
                pixels.checked_mul(3).ok_or_else(|| Error::OutOfRange {
                    name: "decoded image size".into(),
                })?,
                3,
                8,
            ),
            OutputFormat::RgbInterleavedU16 => (
                pixels
                    .checked_mul(3)
                    .and_then(|size| size.checked_mul(size_of::<u16>()))
                    .ok_or_else(|| Error::OutOfRange {
                        name: "decoded image size".into(),
                    })?,
                3,
                16,
            ),
        };

        Ok(DecodeOutputLayout {
            format,
            width,
            height,
            size_bytes,
            samples_per_pixel,
            bits_per_sample,
        })
    }

    /// Checks that the image can be encoded and returns its encoding information.
    ///
    /// # Errors
    ///
    /// Returns an error if the image does not have 1 or 3 samples per pixel, has a width or height of 0, is not contiguous, has an unknown sample format, or its sample sizes are 0 or do not add up to the pixel size.
    /// It also fails if a size overflows.
    pub fn to_encode_info(&self) -> Result<EncodeImageInfo> {
        EncodeImageInfo::try_from(self)
    }
}

impl EncodeImageInfo {
    /// Wraps native image information without checking it.
    ///
    /// # Safety
    ///
    /// `raw` must meet nvTIFF's encoding requirements and describe a contiguous image with a nonzero size and 1 or 3 samples per pixel.
    /// The nonzero sample sizes must add up to `bits_per_pixel`.
    #[must_use]
    pub const unsafe fn from_raw(raw: sys::nvtiffImageInfo_t) -> Self {
        Self { raw }
    }

    /// Returns the native image information.
    #[must_use]
    pub const fn as_raw(&self) -> &sys::nvtiffImageInfo_t {
        &self.raw
    }
}

impl TryFrom<&ImageInfo> for EncodeImageInfo {
    type Error = Error;

    fn try_from(value: &ImageInfo) -> StdResult<Self, Self::Error> {
        let samples = value.samples_per_pixel();
        if samples != 1 && samples != 3 {
            return Err(Error::OutOfRange {
                name: "samples_per_pixel".into(),
            });
        }

        checked_pixels(value.width(), value.height())?;
        if value.planar_config() != PlanarConfig::Contiguous {
            return Err(Error::OutOfRange {
                name: "encoder planar configuration".into(),
            });
        }

        let mut pixel_bits = 0_u16;
        for index in 0..samples as usize {
            let bits = value.bits_per_sample(index).unwrap_or(0);
            if bits == 0
                || value
                    .sample_format(index)
                    .is_none_or(|format| format == SampleFormat::Unknown)
            {
                return Err(Error::OutOfRange {
                    name: "encoder sample format".into(),
                });
            }

            pixel_bits = pixel_bits
                .checked_add(bits)
                .ok_or_else(|| Error::OutOfRange {
                    name: "encoder pixel size".into(),
                })?;
        }

        if pixel_bits != value.bits_per_pixel() {
            return Err(Error::OutOfRange {
                name: "encoder pixel size".into(),
            });
        }
        Ok(Self {
            raw: *value.as_raw(),
        })
    }
}

impl ImageGeometry {
    /// Wraps a native image layout.
    #[must_use]
    pub const fn from_raw(raw: sys::nvtiffImageGeometry_t) -> Self {
        Self { raw }
    }

    /// Returns the native image layout.
    #[must_use]
    pub const fn as_raw(&self) -> &sys::nvtiffImageGeometry_t {
        &self.raw
    }

    /// Returns whether the image is stored in strips or tiles.
    #[must_use]
    pub fn geometry_type(&self) -> ImageGeometryType {
        self.raw.type_.into()
    }

    /// Returns the number of strips or tiles.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.raw.num_striles
    }

    /// Returns the height of each strip or tile in rows.
    #[must_use]
    pub const fn strile_height(&self) -> u32 {
        self.raw.strile_height
    }

    /// Returns the width of each strip or tile in pixels.
    #[must_use]
    pub const fn strile_width(&self) -> u32 {
        self.raw.strile_width
    }

    /// Returns the depth of each strip or tile.
    #[must_use]
    pub const fn strile_depth(&self) -> u32 {
        self.raw.strile_depth
    }

    /// Returns the number of strips or tiles in each plane.
    #[must_use]
    pub const fn striles_per_plane(&self) -> u32 {
        self.raw.num_striles_per_plane
    }
}

fn checked_pixels(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(Error::OutOfRange {
            name: "image dimensions".into(),
        });
    }
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::OutOfRange {
            name: "image dimensions".into(),
        })
}

impl<'a> Image<'a> {
    /// Creates an output from its planes.
    ///
    /// # Errors
    ///
    /// Returns an error if there are more planes than fit a `u32`, or a pitch does not fit the native type.
    pub fn from_planes(planes: impl IntoIterator<Item = ImagePlane<'a>>) -> Result<Self> {
        let planes = planes.into_iter().collect::<Vec<_>>();
        let num_planes = to_u32(planes.len(), "planes")?;
        let mut plane_data = planes
            .iter()
            .map(|plane| plane.data().as_ptr())
            .collect::<Vec<_>>();
        let mut plane_pitch_bytes = planes
            .iter()
            .map(|plane| to_size_t(plane.pitch_bytes, "pitch_bytes"))
            .collect::<Result<Vec<_>>>()?;
        let raw = sys::nvtiffImage_t {
            plane_data: plane_data.as_mut_ptr(),
            plane_pitch_bytes: plane_pitch_bytes.as_mut_ptr(),
            num_planes,
        };
        Ok(Self {
            raw,
            _plane_data: plane_data,
            _plane_pitch_bytes: plane_pitch_bytes,
            planes,
        })
    }

    /// Returns the planes.
    #[must_use]
    pub fn planes(&self) -> &[ImagePlane<'a>] {
        &self.planes
    }

    /// Returns the native image description.
    #[must_use]
    pub const fn as_raw(&self) -> &sys::nvtiffImage_t {
        &self.raw
    }
}

impl<'a> ImagePlane<'a> {
    /// Creates a plane over `buffer`, which may be uninitialized, with rows `pitch_bytes` apart.
    ///
    /// # Errors
    ///
    /// Returns an error if the pitch is 0 or larger than the memory, the memory pointer is null, or the buffer range is out of bounds.
    /// It also fails if the pitch does not fit `usize` or the native type.
    pub fn from_buffer<B: DeviceOutput<u8> + ?Sized>(
        buffer: &'a mut B,
        pitch_bytes: impl TryInto<usize>,
    ) -> Result<Self> {
        // SAFETY: DeviceOutput and the exclusive borrow establish write access.
        unsafe { Self::from_raw_parts(buffer.region().slice(..buffer.len())?, pitch_bytes) }
    }

    /// Creates a plane over a device memory region with rows `pitch_bytes` apart.
    ///
    /// # Safety
    ///
    /// The whole region must be writable and used by nothing else for `'a`, and until any raw decode into it finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the pitch is 0 or larger than the memory, or the memory pointer is null.
    /// It also fails if the pitch does not fit `usize` or the native type.
    pub unsafe fn from_raw_parts(
        region: DeviceRegion<'a>,
        pitch_bytes: impl TryInto<usize>,
    ) -> Result<Self> {
        let pitch_bytes = to_usize(pitch_bytes, "pitch_bytes")?;
        to_size_t(pitch_bytes, "pitch_bytes")?;
        if pitch_bytes == 0 || region.byte_len() < pitch_bytes || region.as_ptr().is_null() {
            return Err(Error::OutOfRange {
                name: "output plane".into(),
            });
        }
        Ok(Self {
            region,
            pitch_bytes,
        })
    }

    /// Returns the memory region of the plane.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'a> {
        self.region
    }

    /// Returns the CUDA context of the memory.
    #[must_use]
    pub fn context(&self) -> &'a Arc<Context> {
        self.region.context()
    }

    /// Returns a pointer to the plane memory.
    #[must_use]
    pub fn data(&self) -> NonNull<c_void> {
        // SAFETY: construction rejects null output pointers.
        unsafe { NonNull::new_unchecked(self.region.as_ptr().cast()) }
    }

    /// Returns the distance between rows in bytes.
    #[must_use]
    pub const fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }
}
