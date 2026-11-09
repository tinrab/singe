//! JPEG decoding, one image at a time or in batches.

mod batched;
mod single;
mod staged;

use std::{mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_nvjpeg_sys as sys;

pub use self::{
    batched::{BatchConfig, BatchPrealloc, BatchedDecoder, BatchedDecoderBuilder},
    single::{Decoder, DecoderBuilder},
    staged::DecodeParams,
};
use crate::{
    error::{Error, Result},
    image::{Image, ImageViewMut, MAX_COMPONENTS, OutputPlaneSpec, OutputSpec, Size},
    types::{ExifOrientation, JpegInfo, OutputFormat, ScaleFactor},
    utility::to_i32,
};

/// Region of an image to decode, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rectangle {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// Settings for a [`Decoder`]: output format, region, CMYK conversion, scaling, and EXIF orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeOptions {
    output_format: OutputFormat,
    roi: Option<Rectangle>,
    allow_cmyk: bool,
    scale_factor: Option<ScaleFactor>,
    exif_orientation: Option<ExifOrientation>,
}

/// Decoded image in device memory, with one plane.
#[derive(Debug)]
pub struct DecodedImage<T = u8> {
    data: DeviceMemory<T>,
    plane: OutputPlaneSpec,
}

/// One plane of a decoded image in device memory.
#[derive(Debug)]
pub struct DecodedPlane<T = u8> {
    data: DeviceMemory<T>,
    spec: OutputPlaneSpec,
}

/// Decoded image in device memory, with as many planes as its output format has.
#[derive(Debug)]
pub struct DecodedOutput<T = u8> {
    format: OutputFormat,
    planes: Vec<DecodedPlane<T>>,
}

impl Rectangle {
    /// Creates a region at (`x`, `y`) with the given width and height.
    ///
    /// The image bounds are checked when the region is used.
    ///
    /// # Errors
    ///
    /// Returns an error if an offset is negative, a size is not positive, or an offset plus its size overflows `i32`.
    pub fn create(x: i32, y: i32, width: i32, height: i32) -> Result<Self> {
        let rect = Self {
            x,
            y,
            width,
            height,
        };
        rect.validate()?;
        Ok(rect)
    }

    /// Returns the horizontal offset in pixels.
    #[must_use]
    pub const fn x(self) -> i32 {
        self.x
    }

    /// Returns the vertical offset in pixels.
    #[must_use]
    pub const fn y(self) -> i32 {
        self.y
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(self) -> i32 {
        self.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(self) -> i32 {
        self.height
    }

    /// Returns the horizontal end, `x + width`, in pixels.
    #[must_use]
    pub const fn end_x(self) -> i32 {
        self.x + self.width
    }

    /// Returns the vertical end, `y + height`, in pixels.
    #[must_use]
    pub const fn end_y(self) -> i32 {
        self.y + self.height
    }

    fn validate(self) -> Result<()> {
        if self.x < 0 {
            return Err(Error::OutOfRange {
                name: "roi x".into(),
            });
        }

        if self.y < 0 {
            return Err(Error::OutOfRange {
                name: "roi y".into(),
            });
        }

        if self.width <= 0 {
            return Err(Error::OutOfRange {
                name: "roi width".into(),
            });
        }

        if self.height <= 0 {
            return Err(Error::OutOfRange {
                name: "roi height".into(),
            });
        }

        self.x.checked_add(self.width).ok_or(Error::OutOfRange {
            name: "roi width".into(),
        })?;
        self.y.checked_add(self.height).ok_or(Error::OutOfRange {
            name: "roi height".into(),
        })?;
        Ok(())
    }
}

impl DecodeOptions {
    /// Creates options with nvJPEG's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            output_format: OutputFormat::Unchanged,
            roi: None,
            allow_cmyk: false,
            scale_factor: None,
            exif_orientation: None,
        }
    }

    /// Sets the output format.
    #[must_use]
    pub const fn with_output_format(mut self, output_format: OutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    /// Decodes only the region `roi`.
    #[must_use]
    pub const fn with_roi(mut self, roi: Rectangle) -> Self {
        self.roi = Some(roi);
        self
    }

    /// Sets whether to treat four-component images as CMYK and convert them.
    #[must_use]
    pub const fn with_allow_cmyk(mut self, allow_cmyk: bool) -> Self {
        self.allow_cmyk = allow_cmyk;
        self
    }

    /// Sets how much the hardware decoder scales the output down.
    #[must_use]
    pub const fn with_scale_factor(mut self, scale_factor: ScaleFactor) -> Self {
        self.scale_factor = Some(scale_factor);
        self
    }

    /// Sets the EXIF orientation to apply to the output.
    #[must_use]
    pub const fn with_exif_orientation(mut self, exif_orientation: ExifOrientation) -> Self {
        self.exif_orientation = Some(exif_orientation);
        self
    }

    /// Returns the output format.
    #[must_use]
    pub const fn output_format(self) -> OutputFormat {
        self.output_format
    }

    /// Returns the region to decode, or `None` for the whole image.
    #[must_use]
    pub const fn roi(self) -> Option<Rectangle> {
        self.roi
    }

    /// Returns whether four-component images are treated as CMYK.
    #[must_use]
    pub const fn allow_cmyk(self) -> bool {
        self.allow_cmyk
    }

    /// Returns the scale factor, if one is set.
    #[must_use]
    pub const fn scale_factor(self) -> Option<ScaleFactor> {
        self.scale_factor
    }

    /// Returns the EXIF orientation, if one is set.
    #[must_use]
    pub const fn exif_orientation(self) -> Option<ExifOrientation> {
        self.exif_orientation
    }

    fn requires_parameterized_decode(self) -> bool {
        self.roi.is_some()
            || self.allow_cmyk
            || self.scale_factor.is_some()
            || self.exif_orientation.is_some()
    }

    /// Returns the size and layout of each output plane for an image described by `info`.
    ///
    /// # Errors
    ///
    /// Returns an error if a scale factor or EXIF orientation is set, a region is set with the [`Unchanged`](OutputFormat::Unchanged) or [`Yuv`](OutputFormat::Yuv) format, the region extends past the image, [`Nv12`](OutputFormat::Nv12) is requested for an image without 4:2:0 subsampling, `info` is invalid, or a size overflows.
    pub fn output_spec(self, info: &JpegInfo) -> Result<OutputSpec> {
        if self.scale_factor.is_some() || self.exif_orientation.is_some() {
            return Err(Error::UnsupportedDecodeConfiguration {
                reason: "scaled or oriented output geometry is not supported by output spec".into(),
            });
        }

        if let Some(roi) = self.roi {
            return roi_output_spec(info, self.output_format, roi);
        }
        OutputSpec::for_jpeg(info, self.output_format)
    }

    fn apply_to(self, params: &mut DecodeParams) -> Result<()> {
        params.set_output_format(self.output_format)?;
        if let Some(roi) = self.roi {
            params.set_roi(roi.x, roi.y, roi.width, roi.height)?;
        }

        if self.allow_cmyk {
            params.set_allow_cmyk(true)?;
        }

        if let Some(scale_factor) = self.scale_factor {
            params.set_scale_factor(scale_factor)?;
        }

        if let Some(exif_orientation) = self.exif_orientation {
            params.set_exif_orientation(exif_orientation)?;
        }
        Ok(())
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DecodedImage<T> {
    /// Returns the decoded pixels.
    #[must_use]
    pub const fn data(&self) -> &DeviceMemory<T> {
        &self.data
    }

    /// Returns the decoded pixels mutably.
    pub const fn data_mut(&mut self) -> &mut DeviceMemory<T> {
        &mut self.data
    }

    /// Returns the layout of [`data`](Self::data).
    #[must_use]
    pub const fn plane(&self) -> OutputPlaneSpec {
        self.plane
    }

    /// Returns the decoded pixels and their layout.
    #[must_use]
    pub fn into_parts(self) -> (DeviceMemory<T>, OutputPlaneSpec) {
        (self.data, self.plane)
    }
}

impl<T> DecodedPlane<T> {
    /// Returns the decoded pixels.
    #[must_use]
    pub const fn data(&self) -> &DeviceMemory<T> {
        &self.data
    }

    /// Returns the decoded pixels mutably.
    pub const fn data_mut(&mut self) -> &mut DeviceMemory<T> {
        &mut self.data
    }

    /// Returns the layout of [`data`](Self::data).
    #[must_use]
    pub const fn spec(&self) -> OutputPlaneSpec {
        self.spec
    }

    /// Returns the decoded pixels and their layout.
    #[must_use]
    pub fn into_parts(self) -> (DeviceMemory<T>, OutputPlaneSpec) {
        (self.data, self.spec)
    }
}

impl DecodedOutput {
    /// Allocates uninitialized device memory for every plane in `spec`.
    ///
    /// # Errors
    ///
    /// Returns an error if `spec` has no planes, a plane's size overflows, or an allocation fails.
    pub fn create_uninit(
        context: &Arc<CudaContext>,
        spec: OutputSpec,
    ) -> Result<DecodedOutput<MaybeUninit<u8>>> {
        let planes = spec
            .present_planes()
            .map(|plane| {
                Ok(DecodedPlane {
                    data: plane.create_uninit_buffer(context)?,
                    spec: plane,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if planes.is_empty() {
            return Err(Error::Empty {
                name: "planes".into(),
            });
        }
        Ok(DecodedOutput {
            format: spec.format(),
            planes,
        })
    }
}

impl<T> DecodedOutput<T> {
    /// Returns the output format.
    #[must_use]
    pub const fn format(&self) -> OutputFormat {
        self.format
    }

    /// Returns the decoded planes.
    #[must_use]
    pub fn planes(&self) -> &[DecodedPlane<T>] {
        &self.planes
    }

    /// Returns the decoded planes mutably.
    pub fn planes_mut(&mut self) -> &mut [DecodedPlane<T>] {
        &mut self.planes
    }

    /// Returns the only plane, for single-plane formats such as [`OutputFormat::RgbInterleaved`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if the output does not have exactly one plane.
    pub fn single_plane(&self) -> Result<&DecodedPlane<T>> {
        if self.planes.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "planes".into(),
                expected: 1,
                actual: self.planes.len(),
            });
        }
        Ok(&self.planes[0])
    }

    /// Returns the only plane mutably.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if the output does not have exactly one plane.
    pub fn single_plane_mut(&mut self) -> Result<&mut DecodedPlane<T>> {
        if self.planes.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "planes".into(),
                expected: 1,
                actual: self.planes.len(),
            });
        }
        Ok(&mut self.planes[0])
    }

    /// Returns the output format and the decoded planes.
    #[must_use]
    pub fn into_parts(self) -> (OutputFormat, Vec<DecodedPlane<T>>) {
        (self.format, self.planes)
    }
}

fn validate_batch(expected: usize, actual: usize, name: &'static str) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            name: name.into(),
            expected,
            actual,
        })
    }
}

fn to_positive_i32(value: impl TryInto<i32>, name: &'static str) -> Result<i32> {
    let value = to_i32(value, name)?;
    if value <= 0 {
        return Err(Error::OutOfRange { name: name.into() });
    }
    Ok(value)
}

fn image_descriptors_from_views<T, const C: usize>(
    destinations: &mut [&mut ImageViewMut<'_, T, C>],
) -> Result<Vec<Image>> {
    destinations
        .iter_mut()
        .map(|destination| destination.descriptor())
        .collect()
}

fn image_descriptors_from_decoded_images<T>(outputs: &mut [DecodedImage<T>]) -> Result<Vec<Image>> {
    outputs
        .iter_mut()
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        .map(|output| unsafe {
            Image::from_raw_parts(
                [
                    output.data.as_mut_ptr().cast(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ],
                [output.plane.row_bytes(), 0, 0, 0],
            )
        })
        .collect()
}

fn image_descriptor_from_decoded_output<T>(output: &mut DecodedOutput<T>) -> Result<Image> {
    if output.planes.is_empty() {
        return Err(Error::Empty {
            name: "planes".into(),
        });
    }

    if output.planes.len() > MAX_COMPONENTS {
        return Err(Error::OutOfRange {
            name: "planes".into(),
        });
    }

    let mut channels = [ptr::null_mut(); MAX_COMPONENTS];
    let mut pitches = [0; MAX_COMPONENTS];
    for (index, plane) in output.planes.iter_mut().enumerate() {
        channels[index] = plane.data.as_mut_ptr().cast();
        pitches[index] = plane.spec.row_bytes();
    }
    // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
    unsafe { Image::from_raw_parts(channels, pitches) }
}

fn image_descriptors_from_decoded_outputs<T>(
    outputs: &mut [DecodedOutput<T>],
) -> Result<Vec<Image>> {
    outputs
        .iter_mut()
        .map(image_descriptor_from_decoded_output)
        .collect()
}

fn roi_output_spec(info: &JpegInfo, format: OutputFormat, roi: Rectangle) -> Result<OutputSpec> {
    let image = info.first_component().ok_or(Error::Empty {
        name: "components".into(),
    })?;
    let size = roi.size_within(Size::new(image.width(), image.height()))?;
    let component_count = info.component_count();
    match format {
        OutputFormat::Y
        | OutputFormat::Rgb
        | OutputFormat::Bgr
        | OutputFormat::RgbInterleaved
        | OutputFormat::BgrInterleaved
        | OutputFormat::UnchangedInterleaved
        | OutputFormat::UnchangedInterleavedU16
        | OutputFormat::Nv12
        | OutputFormat::Yuy2 => {
            let mut widths = [0; sys::NVJPEG_MAX_COMPONENT as usize];
            let mut heights = [0; sys::NVJPEG_MAX_COMPONENT as usize];
            let width = to_i32(size.width(), "roi width")?;
            let height = to_i32(size.height(), "roi height")?;
            for index in 0..component_count {
                widths[index] = width;
                heights[index] = height;
            }

            let roi_info =
                JpegInfo::from_raw(component_count, info.subsampling(), widths, heights)?;
            OutputSpec::for_jpeg(&roi_info, format)
        }
        OutputFormat::Unchanged | OutputFormat::Yuv => Err(Error::UnsupportedDecodeConfiguration {
            reason: "roi output spec for planar source components is not supported".into(),
        }),
    }
}

impl Rectangle {
    fn size_within(self, image: Size) -> Result<Size> {
        if self.x < 0 {
            return Err(Error::OutOfRange {
                name: "roi x".into(),
            });
        }

        if self.y < 0 {
            return Err(Error::OutOfRange {
                name: "roi y".into(),
            });
        }

        if self.width <= 0 {
            return Err(Error::OutOfRange {
                name: "roi width".into(),
            });
        }

        if self.height <= 0 {
            return Err(Error::OutOfRange {
                name: "roi height".into(),
            });
        }

        let end_x = (self.x as u32)
            .checked_add(self.width as u32)
            .ok_or(Error::OutOfRange {
                name: "roi width".into(),
            })?;
        let end_y = (self.y as u32)
            .checked_add(self.height as u32)
            .ok_or(Error::OutOfRange {
                name: "roi height".into(),
            })?;
        if end_x > image.width() {
            return Err(Error::OutOfRange {
                name: "roi width".into(),
            });
        }

        if end_y > image.height() {
            return Err(Error::OutOfRange {
                name: "roi height".into(),
            });
        }
        Ok(Size::new(self.width as u32, self.height as u32))
    }
}

impl DecodedImage<MaybeUninit<u8>> {
    /// Marks the plane as initialized.
    ///
    /// # Safety
    ///
    /// Every byte of the plane must be initialized, and the work that wrote it must have finished before it is read.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    #[must_use]
    pub unsafe fn assume_init(self) -> DecodedImage {
        DecodedImage {
            // SAFETY: the caller upholds this function's safety contract.
            data: unsafe { self.data.assume_init() },
            plane: self.plane,
        }
    }
}

impl DecodedPlane<MaybeUninit<u8>> {
    /// Marks the plane as initialized.
    ///
    /// # Safety
    ///
    /// Every byte of the plane must be initialized, and the work that wrote it must have finished before it is read.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    #[must_use]
    pub unsafe fn assume_init(self) -> DecodedPlane {
        DecodedPlane {
            // SAFETY: the caller upholds this function's safety contract.
            data: unsafe { self.data.assume_init() },
            spec: self.spec,
        }
    }
}

impl DecodedOutput<MaybeUninit<u8>> {
    /// Marks every plane as initialized.
    ///
    /// # Safety
    ///
    /// Every byte of every plane must be initialized, and the work that wrote them must have finished before they are read.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    #[must_use]
    pub unsafe fn assume_init(self) -> DecodedOutput {
        DecodedOutput {
            format: self.format,
            planes: self
                .planes
                .into_iter()
                // SAFETY: the caller upholds this function's safety contract.
                .map(|plane| unsafe { plane.assume_init() })
                .collect(),
        }
    }
}
