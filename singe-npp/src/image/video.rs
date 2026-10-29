//! Video color conversion and UYVP transcoding.
//!
//! Safe conversion functions complete all borrowed work before returning, including error and unwind paths.
//! They accept write-only destinations and reject stream capture.
//! Public raw enqueue functions are available under [`crate::image::raw::video`].
//!
//! Source pixel offsets accepted by these entry points must currently be `(0, 0)`.
//! Plane views count stored samples, while [`VideoView::size`](crate::image::video::VideoView::size) counts pixels.
//!
//! Some NPP builds declare UYVP transcoders without exporting them from the shared
//! library. Those entry points return [`crate::error::Status::NotSupportedModeError`]
//! when the loaded library does not provide them.

mod half;
mod memory;
pub(crate) mod raw;
mod semiplanar;
mod uyvp;

use std::sync::Arc;

pub use memory::Frame;
use singe_cuda::{context::Context as CudaContext, types::f16};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    types::{ColorSpace, Point, Size},
};

/// Pixel organization of a video frame. Plane widths describe stored samples;
/// the frame size separately describes the number of image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFormat {
    /// Three-channel packed RGB.
    Rgb,
    /// Three-channel packed BGR.
    Bgr,
    /// Four-channel packed RGBA.
    Rgba,
    /// Four-channel packed BGRA.
    Bgra,
    /// Three full-resolution RGB planes.
    RgbPlanar,
    /// Three YUV planes with 4:2:0 chroma subsampling.
    Yuv420,
    /// Y and interleaved UV planes with 4:2:0 chroma subsampling.
    Nv12,
    /// Y and interleaved UV planes with 4:2:2 chroma subsampling.
    Nv16,
    /// 10-bit Y and UV in the high ten bits of 16-bit samples, with 4:2:0 chroma subsampling.
    P010,
    /// The three-plane P010 input accepted by the NPP UYVP transcoder.
    P010Planar,
    /// 16-bit Y and UV with 4:2:0 chroma subsampling.
    P016,
    /// 10-bit Y and UV in 16-bit samples, with 4:2:2 chroma subsampling.
    P210,
    /// Three full-resolution YUV planes.
    I444,
    /// Packed full-resolution YUV.
    Yuv444,
    /// UYVP 10-bit packed YCbCr 4:2:2: five bytes for two pixels.
    Uyvp10,
    /// Packed 8-bit UYVY 4:2:2.
    Uyvy,
    /// Packed 16-bit UYVY 4:2:2.
    Uyvy16,
    /// V210 10-bit packed 4:2:2: sixteen bytes for six pixels.
    V210,
    /// Y410 10-bit packed 4:4:4: four bytes per pixel.
    Y410,
}

/// Borrowed video frame whose plane capacities cover its logical pixel extent.
#[derive(Debug)]
pub struct VideoView<'a, T> {
    format: VideoFormat,
    size: Size,
    planes: Vec<ImageView<'a, T, C1>>,
}

/// Exclusively borrowed writable video frame.
#[derive(Debug)]
pub struct VideoViewMut<'a, T> {
    format: VideoFormat,
    size: Size,
    planes: Vec<ImageViewMut<'a, T, C1>>,
}

/// Exclusively writable stored samples with checked video geometry and no initialization claim.
#[derive(Debug)]
pub struct VideoOutputView<'a, T> {
    format: VideoFormat,
    size: Size,
    planes: Vec<ImageOutputView<'a, T, C1>>,
}

impl<'a, T> VideoView<'a, T> {
    /// Creates a video view whose planes cover the logical frame extent.
    ///
    /// Plane widths count stored `T` samples, while `size` counts pixels. Byte pitches describe row padding.
    /// Plane capacities may exceed the format's required stored-sample dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if pixel dimensions are nonpositive, `T` is zero-sized, subsampled dimensions are incompatible with the format, a packed pixel group is incomplete, or a row-size calculation overflows.
    /// The plane count must match the format. Each plane must have enough stored samples, rows, and byte pitch for the logical frame. All planes must belong to the same CUDA context.
    /// Subsampled formats require even widths; vertically subsampled formats also require even heights. `VideoFormat::V210` additionally requires a width divisible by six.
    pub fn create(
        format: VideoFormat,
        size: Size,
        planes: Vec<ImageView<'a, T, C1>>,
    ) -> Result<Self> {
        validate_planes::<T>(
            format,
            size,
            planes.iter().map(|plane| (plane.size(), plane.step())),
        )?;
        validate_contexts(planes.iter().map(ImageView::context))?;
        Ok(Self {
            format,
            size,
            planes,
        })
    }

    /// Returns the logical frame width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// Returns the frame's pixel and plane organization.
    #[must_use]
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Returns the planes in the order specified by the video format.
    ///
    /// Each plane width counts stored samples rather than frame pixels.
    #[must_use]
    pub fn planes(&self) -> &[ImageView<'a, T, C1>] {
        &self.planes
    }
}

impl<'a, T> VideoViewMut<'a, T> {
    /// Creates a video view whose planes cover the logical frame extent.
    ///
    /// Plane widths count stored `T` samples, while `size` counts pixels. Byte pitches describe row padding.
    /// Plane capacities may exceed the format's required stored-sample dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if pixel dimensions are nonpositive, `T` is zero-sized, subsampled dimensions are incompatible with the format, a packed pixel group is incomplete, or a row-size calculation overflows.
    /// The plane count must match the format. Each plane must have enough stored samples, rows, and byte pitch for the logical frame. All planes must belong to the same CUDA context.
    /// Subsampled formats require even widths; vertically subsampled formats also require even heights. `VideoFormat::V210` additionally requires a width divisible by six.
    pub fn create(
        format: VideoFormat,
        size: Size,
        planes: Vec<ImageViewMut<'a, T, C1>>,
    ) -> Result<Self> {
        validate_planes::<T>(
            format,
            size,
            planes.iter().map(|plane| (plane.size(), plane.step())),
        )?;
        validate_contexts(planes.iter().map(ImageViewMut::context))?;
        Ok(Self {
            format,
            size,
            planes,
        })
    }

    /// Returns the logical frame width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// Returns the frame's pixel and plane organization.
    #[must_use]
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Reborrows the frame as write-only plane views while preserving its validated format and geometry.
    ///
    /// The returned view makes no initialization claim about subsequent writes.
    pub fn output(&mut self) -> VideoOutputView<'_, T> {
        VideoOutputView {
            format: self.format,
            size: self.size,
            planes: self.planes.iter_mut().map(ImageViewMut::output).collect(),
        }
    }
}

impl<'a, T> VideoOutputView<'a, T> {
    /// Creates a video view whose planes cover the logical frame extent.
    ///
    /// Plane widths count stored `T` samples, while `size` counts pixels. Byte pitches describe row padding.
    /// Plane capacities may exceed the format's required stored-sample dimensions. This view makes no initialization claim.
    ///
    /// # Errors
    ///
    /// Returns an error if pixel dimensions are nonpositive, `T` is zero-sized, subsampled dimensions are incompatible with the format, a packed pixel group is incomplete, or a row-size calculation overflows.
    /// The plane count must match the format. Each plane must have enough stored samples, rows, and byte pitch for the logical frame. All planes must belong to the same CUDA context.
    /// Subsampled formats require even widths; vertically subsampled formats also require even heights. `VideoFormat::V210` additionally requires a width divisible by six.
    pub fn create(
        format: VideoFormat,
        size: Size,
        planes: Vec<ImageOutputView<'a, T, C1>>,
    ) -> Result<Self> {
        validate_planes::<T>(
            format,
            size,
            planes.iter().map(|plane| (plane.size(), plane.step())),
        )?;
        validate_contexts(planes.iter().map(ImageOutputView::context))?;
        Ok(Self {
            format,
            size,
            planes,
        })
    }

    /// Returns the logical frame width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// Returns the frame's pixel and plane organization.
    #[must_use]
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Returns the planes in the order specified by the video format.
    ///
    /// Each plane width counts stored samples rather than frame pixels.
    #[must_use]
    pub fn planes(&self) -> &[ImageOutputView<'a, T, C1>] {
        &self.planes
    }

    /// Borrows one writable plane while preserving the frame's validated geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is outside the stored plane list.
    pub fn plane_mut(&mut self, index: usize) -> Result<ImageOutputView<'_, T, C1>> {
        self.planes
            .get_mut(index)
            .map(ImageOutputView::output)
            .ok_or_else(|| invalid("video plane index"))
    }
}

fn validate_contexts<'a>(contexts: impl Iterator<Item = &'a Arc<CudaContext>>) -> Result<()> {
    let mut selected = None;
    for context in contexts {
        if let Some(selected) = selected
            && selected != context
        {
            return Err(Error::StreamContextMismatch);
        }
        selected = Some(context);
    }
    Ok(())
}

fn invalid(name: &str) -> Error {
    Error::out_of_range(name.to_owned())
}

fn plane_dimensions<T>(format: VideoFormat, size: Size) -> Result<Vec<(usize, usize)>> {
    let width = usize::try_from(size.width).map_err(|_| invalid("video width"))?;
    let height = usize::try_from(size.height).map_err(|_| invalid("video height"))?;
    let sample_size = size_of::<T>();
    if width == 0 || height == 0 || sample_size == 0 {
        return Err(invalid("video size"));
    }
    let subsampled = matches!(
        format,
        VideoFormat::Yuv420
            | VideoFormat::Nv12
            | VideoFormat::Nv16
            | VideoFormat::P010
            | VideoFormat::P010Planar
            | VideoFormat::P016
            | VideoFormat::P210
            | VideoFormat::Uyvp10
            | VideoFormat::Uyvy
            | VideoFormat::Uyvy16
            | VideoFormat::V210
    );
    if subsampled && width % 2 != 0 {
        return Err(invalid("video chroma width"));
    }
    let vertical = matches!(
        format,
        VideoFormat::Yuv420
            | VideoFormat::Nv12
            | VideoFormat::P010
            | VideoFormat::P010Planar
            | VideoFormat::P016
    );
    if vertical && height % 2 != 0 {
        return Err(invalid("video chroma height"));
    }
    let row = |samples: usize| {
        width
            .checked_mul(samples)
            .and_then(|value| value.checked_mul(sample_size))
            .ok_or_else(|| invalid("video row size"))
    };
    let dimensions = match format {
        VideoFormat::Rgb | VideoFormat::Bgr | VideoFormat::Yuv444 => vec![(row(3)?, height)],
        VideoFormat::Rgba | VideoFormat::Bgra => vec![(row(4)?, height)],
        VideoFormat::RgbPlanar | VideoFormat::I444 => vec![(row(1)?, height); 3],
        VideoFormat::Yuv420 | VideoFormat::P010Planar => vec![
            (row(1)?, height),
            (row(1)? / 2, height / 2),
            (row(1)? / 2, height / 2),
        ],
        VideoFormat::Nv12 | VideoFormat::P010 | VideoFormat::P016 => {
            vec![(row(1)?, height), (row(1)?, height / 2)]
        }
        VideoFormat::Nv16 | VideoFormat::P210 => vec![(row(1)?, height); 2],
        VideoFormat::Uyvp10 => vec![(
            width.checked_mul(5).ok_or_else(|| invalid("uyvp row"))? / 2,
            height,
        )],
        VideoFormat::Uyvy | VideoFormat::Uyvy16 => vec![(row(2)?, height)],
        VideoFormat::V210 => {
            if width % 6 != 0 {
                return Err(invalid("v210 pixel group"));
            }
            vec![(
                width
                    .checked_div(6)
                    .and_then(|value| value.checked_mul(16))
                    .ok_or_else(|| invalid("v210 row"))?,
                height,
            )]
        }
        VideoFormat::Y410 => vec![(
            width.checked_mul(4).ok_or_else(|| invalid("y410 row"))?,
            height,
        )],
    };
    Ok(dimensions)
}

fn validate_planes<T>(
    format: VideoFormat,
    size: Size,
    planes: impl Iterator<Item = (Size, i32)>,
) -> Result<()> {
    let dimensions = plane_dimensions::<T>(format, size)?;
    let sample_size = size_of::<T>();
    let planes: Vec<_> = planes.collect();
    if planes.len() != dimensions.len() {
        return Err(Error::LengthMismatch {
            name: "video planes".into(),
            expected: dimensions.len(),
            actual: planes.len(),
        });
    }
    for ((size, step), (row_bytes, rows)) in planes.into_iter().zip(dimensions) {
        let width = usize::try_from(size.width).map_err(|_| invalid("plane row"))?;
        let capacity = width
            .checked_mul(sample_size)
            .ok_or_else(|| invalid("plane row"))?;
        if capacity < row_bytes
            || (size.height as usize) < rows
            || step < 0
            || (step as usize) < row_bytes
        {
            return Err(invalid("video plane capacity"));
        }
    }
    Ok(())
}

fn prepare_conversion<S, D>(
    context: &StreamContext,
    source: &VideoView<'_, S>,
    destination: &VideoOutputView<'_, D>,
) -> Result<()> {
    context.stream().ensure_not_capturing()?;
    for plane in source.planes() {
        context.prepare_region(plane)?;
    }
    for plane in destination.planes() {
        context.prepare_region(plane)?;
    }
    Ok(())
}

fn validate_conversion<S, D>(
    source: &VideoView<'_, S>,
    source_format: VideoFormat,
    destination: &VideoOutputView<'_, D>,
    destination_format: VideoFormat,
    offset: Point,
    source_shared_step: bool,
    destination_shared_step: bool,
) -> Result<()> {
    if source.format != source_format || destination.format != destination_format {
        return Err(invalid("video format"));
    }
    if offset.x != 0
        || offset.y != 0
        || offset
            .x
            .checked_add(destination.size.width)
            .is_none_or(|right| right > source.size.width)
        || offset
            .y
            .checked_add(destination.size.height)
            .is_none_or(|bottom| bottom > source.size.height)
    {
        return Err(invalid("video source region"));
    }
    if source_shared_step
        && source
            .planes
            .iter()
            .any(|plane| plane.step() != source.planes[0].step())
    {
        return Err(invalid("video source pitch"));
    }
    if destination_shared_step
        && destination
            .planes
            .iter()
            .any(|plane| plane.step() != destination.planes[0].step())
    {
        return Err(invalid("video destination pitch"));
    }
    Ok(())
}

macro_rules! completed_video_conversion {
    ($(#[$metadata:meta])* $name:ident, $source:ty, $destination:ty $(, $argument:ident: $kind:ty)* $(,)?) => {
        $(#[$metadata])*
        pub fn $name(
            context: &StreamContext,
            source: &VideoView<'_, $source>,
            destination: &mut VideoOutputView<'_, $destination>,
            $($argument: $kind,)*
        ) -> Result<()> {
            context.sync_scope(|| {
                for plane in source.planes() { context.prepare_region(plane)?; }
                for plane in destination.planes() { context.prepare_region(plane)?; }
                // SAFETY: checked plane borrows and their actual allocation metadata survive native completion.
                unsafe { raw::$name(context, source, destination $(, $argument)*) }
            })
        }
    };
}

completed_video_conversion!(
    /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB BT.601 full-range color conversion (ITU-T T.871 / JFIF).
    ///
    /// Coefficients derived from BT.601 luma weights (Kr=0.299, Kb=0.114) with full digital range Y=\[0..255\], Cb/Cr=\[0..255\] (offset 128). Use this variant for cameras that signal full-range BT.601 NV12 (e.g. OAK-D VPU encoder output), as distinct from the unspecified-range default `nppiNV12ToRGB_8u_P2C3R` and the BT.709 variants.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    nv12_to_rgb_601_8u_p2c3r,
    u8,
    u8,
);

completed_video_conversion!(
    /// 2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR BT.601 full-range color conversion (ITU-T T.871 / JFIF).
    ///
    /// Coefficients derived from BT.601 luma weights (Kr=0.299, Kb=0.114) with full digital range Y=\[0..255\], Cb/Cr=\[0..255\] (offset 128).
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgr`. The destination pixel extent must fit within the source extent.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    nv12_to_bgr_601_8u_p2c3r,
    u8,
    u8,
);

/// Encodes normalized half-precision Rgb channels as packed 10-bit UYVP.
/// Completes integer-codec composition before releasing temporary storage.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn rgb_16f_to_uyvp_10u_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    half::encode(
        context,
        source,
        destination,
        VideoFormat::Rgb,
        source_offset,
        color_space,
    )
}

/// Encodes normalized half-precision `RgbPlanar` channels as packed 10-bit UYVP.
/// Completes integer-codec composition before releasing temporary storage.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn rgb_16f_to_uyvp_10u_p3c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    half::encode(
        context,
        source,
        destination,
        VideoFormat::RgbPlanar,
        source_offset,
        color_space,
    )
}

/// Encodes normalized half-precision Rgba channels as packed 10-bit UYVP.
/// Completes integer-codec composition before releasing temporary storage.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn rgba_16f_to_uyvp_10u_ac4c3r(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    half::encode(
        context,
        source,
        destination,
        VideoFormat::Rgba,
        source_offset,
        color_space,
    )
}

completed_video_conversion!(
/// UYVP to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_rgb_8u_c3ac4r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u8,
);

/// Decodes packed 10-bit UYVP to normalized half-precision Rgb channels.
/// Uses the integer color decoder and explicit half conversion; native half decoding does not produce pixels in NPP 13.2.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn uyvp_10u_to_rgb_16f_c3c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    half::decode(
        context,
        source,
        destination,
        VideoFormat::Rgb,
        source_offset,
        color_space,
        f16::ZERO,
    )
}

/// Decodes packed 10-bit UYVP to normalized half-precision RGBA channels.
///
/// Copies the supplied half-precision alpha bits exactly.
/// Uses the integer color decoder and explicit half conversion; native half decoding does not produce pixels in NPP 13.2.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn uyvp_10u_to_rgba_16f_c3ac4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: f16,
) -> Result<()> {
    half::decode(
        context,
        source,
        destination,
        VideoFormat::Rgba,
        source_offset,
        color_space,
        alpha,
    )
}

/// Decodes packed 10-bit UYVP to normalized half-precision `RgbPlanar` channels.
/// Uses the integer color decoder and explicit half conversion; native half decoding does not produce pixels in NPP 13.2.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
/// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
/// All source planes must have equal byte strides.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn uyvp_10u_to_rgb_16f_c3p3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    source_offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    half::decode(
        context,
        source,
        destination,
        VideoFormat::RgbPlanar,
        source_offset,
        color_space,
        f16::ZERO,
    )
}

completed_video_conversion!(
/// 4:2:0 conversions.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    nv12_8u_to_uyvp_10u_p2c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// P010 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::P010Planar` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    p010_16u_to_uyvp_10u_p3c3r, u16, u8,
    source_offset: Point,
);

pub use raw::uyvp_10u_to_p208_8u_c3c3r;

completed_video_conversion!(
/// UYVP to NV16 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Nv16`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_nv16_8u_c3p2r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// UYVP to V210 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::V210`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_v210_10u_c3c3r, u8, u8,
    source_offset: Point,
);

pub use raw::p208_8u_to_uyvp_10u_c3c3r;

completed_video_conversion!(
/// NV16 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Nv16` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    nv16_8u_to_uyvp_10u_p2c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// V210 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::V210` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    v210_10u_to_uyvp_10u_c3c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// P210 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::P210` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    p210_16u_to_uyvp_10u_p2c3r, u16, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// 4:4:4 conversions.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::I444`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_i444_8u_c3p3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// UYVP to YUV444 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Yuv444`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_yuv444_8u_c3c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// UYVP to Y410 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvp10` and the destination format must be `VideoFormat::Y410`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvp_10u_to_y410_10u_c3c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// I444 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::I444` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    i444_8u_to_uyvp_10u_p3c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// YUV444 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Yuv444` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    yuv444_8u_to_uyvp_10u_c3c3r, u8, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// Y410 to UYVP color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Y410` and the destination format must be `VideoFormat::Uyvp10`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    y410_10u_to_uyvp_10u_c3c3r, u16, u8,
    source_offset: Point,
);

completed_video_conversion!(
/// RGB to UYVY color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_8u_to_uyvy_8u_c3c3r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGB to UYVY color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_8u_to_uyvy_8u_p3c3r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGBA to UYVY color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvy`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgba_8u_to_uyvy_8u_ac4c3r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGB to UYVY16 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_16u_to_uyvy16_16u_c3c3r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGB to UYVY16 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_16u_to_uyvy16_16u_p3c3r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGBA to UYVY16 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Rgba` and the destination format must be `VideoFormat::Uyvy16`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgba_16u_to_uyvy16_16u_ac4c3r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// UYVY to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy_8u_to_rgb_8u_c3c3r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// UYVY to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy_8u_to_rgb_8u_c3ac4r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u8,
);

completed_video_conversion!(
/// UYVY to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy_8u_to_rgb_8u_c3p3r, u8, u8,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// UYVY16 to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy16_16u_to_rgb_16u_c3c3r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// UYVY16 to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy16_16u_to_rgb_16u_c3ac4r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
    alpha: u16,
);

completed_video_conversion!(
/// UYVY16 to RGB color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Uyvy16` and the destination format must be `VideoFormat::RgbPlanar`. The destination pixel extent must fit within the source extent.
    /// `source_offset` must be `(0, 0)`; source-region endpoint arithmetic must not overflow.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    uyvy16_16u_to_rgb_16u_c3p3r, u16, u16,
    source_offset: Point,
    color_space: ColorSpace,
);

/// Decodes full-range Nv12 into Rgba channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn nv12_to_rgb_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    semiplanar::decode_u8(
        context,
        source,
        destination,
        VideoFormat::Rgba,
        color_space,
        alpha,
    )
}

/// Decodes full-range Nv12 into Bgra channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgra`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn nv12_to_bgr_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    semiplanar::decode_u8(
        context,
        source,
        destination,
        VideoFormat::Bgra,
        color_space,
        alpha,
    )
}

/// Decodes full-range Nv12 into Rgb channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn nv12_to_rgb_bt2020_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    semiplanar::decode_u8(
        context,
        source,
        destination,
        VideoFormat::Rgb,
        ColorSpace::Bt2020Hdtv,
        0,
    )
}

/// Decodes full-range Nv12 into Bgr channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::Nv12` and the destination format must be `VideoFormat::Bgr`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn nv12_to_bgr_bt2020_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
) -> Result<()> {
    semiplanar::decode_u8(
        context,
        source,
        destination,
        VideoFormat::Bgr,
        ColorSpace::Bt2020Hdtv,
        0,
    )
}

/// Decodes full-range P010 into Rgb channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p010_to_rgb_8u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
) -> Result<()> {
    semiplanar::decode_u16_to_u8(
        context,
        source,
        destination,
        VideoFormat::P010,
        VideoFormat::Rgb,
        color_space,
        0,
    )
}

/// Decodes full-range P010 into Rgba channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p010_to_rgb_8u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    semiplanar::decode_u16_to_u8(
        context,
        source,
        destination,
        VideoFormat::P010,
        VideoFormat::Rgba,
        color_space,
        alpha,
    )
}

/// Decodes full-range P010 into Rgb channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p010_to_rgb_16u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
) -> Result<()> {
    semiplanar::decode_u16(
        context,
        source,
        destination,
        VideoFormat::P010,
        VideoFormat::Rgb,
        color_space,
        0,
    )
}

/// Decodes full-range P010 into Rgba channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P010` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p010_to_rgb_16u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    semiplanar::decode_u16(
        context,
        source,
        destination,
        VideoFormat::P010,
        VideoFormat::Rgba,
        color_space,
        alpha,
    )
}

/// Decodes full-range P016 into Rgb channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P016` and the destination format must be `VideoFormat::Rgb`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p016_to_rgb_16u_p2c3r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
) -> Result<()> {
    semiplanar::decode_u16(
        context,
        source,
        destination,
        VideoFormat::P016,
        VideoFormat::Rgb,
        color_space,
        0,
    )
}

/// Decodes full-range P016 into Rgba channels.
/// Completes color-twist conversion and any alpha insertion before returning.
///
/// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
/// Pitched row padding is not produced.
/// Uses the full-range color-twist implementation with independently supplied source-plane strides.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
/// The source format must be `VideoFormat::P016` and the destination format must be `VideoFormat::Rgba`. The destination pixel extent must fit within the source extent.
/// All destination planes must have equal byte strides.
/// `ColorSpace::TranscodeOnly` is not accepted for these RGB color conversions.
/// Temporary image allocation, scalar conversion, temporary view validation, and checked sample-width arithmetic can also fail.
/// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
pub fn p016_to_rgb_16u_p2c4r(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    semiplanar::decode_u16(
        context,
        source,
        destination,
        VideoFormat::P016,
        VideoFormat::Rgba,
        color_space,
        alpha,
    )
}

completed_video_conversion!(
/// RGB to NV12 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Rgb` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_to_nv12_8u_c3p2r, u8, u8,
    color_space: ColorSpace,
);

completed_video_conversion!(
/// RGB to NV12 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::RgbPlanar` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
    /// All source planes must have equal byte strides.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    rgb_to_nv12_8u_p3p2r, u8, u8,
    color_space: ColorSpace,
);

completed_video_conversion!(
    /// YUV420 to NV12 color conversion.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams, and completes borrowed native work before returning.
    /// Pitched row padding is not produced.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a source or destination plane belongs to another CUDA context, or NPP rejects the conversion.
    /// The source format must be `VideoFormat::Yuv420` and the destination format must be `VideoFormat::Nv12`. The destination pixel extent must fit within the source extent.
    /// All destination planes must have equal byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    yuv420_to_nv12_8u_p3p2r,
    u8,
    u8,
);
