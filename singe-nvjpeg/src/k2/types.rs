//! JPEG 2000 formats, image information, and tile selections.

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::view::AnyBitPatternDeviceRepr;
use singe_nvjpeg_sys::k2 as sys;

use crate::{
    error::{Error, Result},
    image::Size,
};

/// nvJPEG 2000 implementation; only one exists so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Backend {
    /// The default implementation.
    Default = sys::nvjpeg2kBackend_t::NVJPEG2K_BACKEND_DEFAULT as _,
}

impl_enum_conversion!(sys::nvjpeg2kBackend_t, Backend);

/// Color space of a JP2 file, as defined by the JP2 format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum ColorSpace {
    /// A color space nvJPEG 2000 does not support.
    NotSupported = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_NOT_SUPPORTED as _,
    /// Unknown.
    Unknown = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_UNKNOWN as _,
    /// sRGB.
    Srgb = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_SRGB as _,
    /// Grayscale.
    Gray = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_GRAY as _,
    /// sYCC.
    Sycc = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_SYCC as _,
}

impl_enum_conversion!(i32, sys::nvjpeg2kColorSpace_t, ColorSpace);

/// Type of each sample in a JPEG 2000 image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ImageType {
    /// 8-bit unsigned integer.
    U8 = sys::nvjpeg2kImageType_t::NVJPEG2K_UINT8 as _,
    /// 16-bit unsigned integer.
    U16 = sys::nvjpeg2kImageType_t::NVJPEG2K_UINT16 as _,
    /// 16-bit signed integer.
    I16 = sys::nvjpeg2kImageType_t::NVJPEG2K_INT16 as _,
}

impl_enum_conversion!(sys::nvjpeg2kImageType_t, ImageType);

/// Rust sample types that JPEG 2000 images can hold: `u8`, `u16`, and `i16`.
///
/// The trait is sealed.
pub trait ImagePixel: private::Sealed + AnyBitPatternDeviceRepr {
    /// nvJPEG 2000 sample type for this Rust type.
    const IMAGE_TYPE: ImageType;
}

impl ImagePixel for u8 {
    const IMAGE_TYPE: ImageType = ImageType::U8;
}

impl ImagePixel for u16 {
    const IMAGE_TYPE: ImageType = ImageType::U16;
}

impl ImagePixel for i16 {
    const IMAGE_TYPE: ImageType = ImageType::I16;
}

mod private {
    pub trait Sealed {}

    impl Sealed for u8 {}

    impl Sealed for u16 {}

    impl Sealed for i16 {}
}

/// How an image's components are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ImageFormat {
    /// Each component in its own plane.
    Planar = sys::nvjpeg2kImageFormat_t::NVJPEG2K_FORMAT_PLANAR as _,
    /// All components interleaved in one plane.
    ///
    /// Not supported when the components have different sizes.
    Interleaved = sys::nvjpeg2kImageFormat_t::NVJPEG2K_FORMAT_INTERLEAVED as _,
}

impl_enum_conversion!(sys::nvjpeg2kImageFormat_t, ImageFormat);

/// Container for JPEG 2000 data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BitstreamType {
    /// A bare JPEG 2000 codestream (`.j2k`).
    J2k = sys::nvjpeg2kBitstreamType_t::NVJPEG2K_STREAM_J2K as _,
    /// A JP2 file (`.jp2`).
    Jp2 = sys::nvjpeg2kBitstreamType_t::NVJPEG2K_STREAM_JP2 as _,
}

impl_enum_conversion!(sys::nvjpeg2kBitstreamType_t, BitstreamType);

/// Order of packets in a JPEG 2000 codestream, as defined by the standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ProgressionOrder {
    /// Layer, resolution, component, position.
    Lrcp = sys::nvjpeg2kProgOrder_t::NVJPEG2K_LRCP as _,
    /// Resolution, layer, component, position.
    Rlcp = sys::nvjpeg2kProgOrder_t::NVJPEG2K_RLCP as _,
    /// Resolution, position, component, layer.
    Rpcl = sys::nvjpeg2kProgOrder_t::NVJPEG2K_RPCL as _,
    /// Position, component, resolution, layer.
    Pcrl = sys::nvjpeg2kProgOrder_t::NVJPEG2K_PCRL as _,
    /// Component, position, resolution, layer.
    Cprl = sys::nvjpeg2kProgOrder_t::NVJPEG2K_CPRL as _,
}

impl_enum_conversion!(sys::nvjpeg2kProgOrder_t, ProgressionOrder);

/// How lossy encoding interprets its quality value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum QualityType {
    /// Target peak signal-to-noise ratio, a positive number; higher values give higher quality.
    ///
    /// The high-throughput (HT) encoder does not support it.
    TargetPsnr = sys::nvjpeg2kQualityType_t::NVJPEG2K_QUALITY_TYPE_TARGET_PSNR as _,
    /// Quantization step, the positive amount that pixel data is divided by; higher values give lower quality.
    QuantizationStep = sys::nvjpeg2kQualityType_t::NVJPEG2K_QUALITY_TYPE_QUANTIZATION_STEP as _,
    /// JPEG-style quality from 1.0 (worst) to 100.0 (best).
    ///
    /// The input must be YCC, grayscale, or RGB with the multiple-component transform enabled.
    QFactor = sys::nvjpeg2kQualityType_t::NVJPEG2K_QUALITY_TYPE_Q_FACTOR as _,
}

impl_enum_conversion!(sys::nvjpeg2kQualityType_t, QualityType);

/// Size, tiling, and component count of a parsed JPEG 2000 image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageInfo {
    pub(crate) size: Size,
    pub(crate) tile_size: Size,
    pub(crate) tile_count: Size,
    pub(crate) components: u32,
}

/// Size, bit depth, and signedness of one component of a JPEG 2000 image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageComponentInfo {
    pub(crate) size: Size,
    pub(crate) precision: u8,
    pub(crate) signed: bool,
}

/// Tile of a JPEG 2000 image to decode, and how many resolution levels to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileDecode {
    pub(crate) tile_id: u32,
    pub(crate) resolution_levels: u32,
}

impl ImageInfo {
    pub(crate) fn from_raw(raw: sys::nvjpeg2kImageInfo_t) -> Self {
        Self {
            size: Size::new(raw.image_width, raw.image_height),
            tile_size: Size::new(raw.tile_width, raw.tile_height),
            tile_count: Size::new(raw.num_tiles_x, raw.num_tiles_y),
            components: raw.num_components,
        }
    }

    /// Returns the image size in pixels.
    #[must_use]
    pub const fn size(self) -> Size {
        self.size
    }

    /// Returns the nominal tile size in pixels.
    #[must_use]
    pub const fn tile_size(self) -> Size {
        self.tile_size
    }

    /// Returns the number of tiles across and down.
    #[must_use]
    pub const fn tile_count(self) -> Size {
        self.tile_count
    }

    /// Returns the number of components.
    #[must_use]
    pub const fn component_count(self) -> u32 {
        self.components
    }
}

impl ImageComponentInfo {
    /// Creates component information for encoding, with `precision` bits per sample.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn create(size: Size, precision: u8, signed: bool) -> Result<Self> {
        validate_dimensions(size, "component")?;
        validate_component_precision(precision)?;
        Ok(Self {
            size,
            precision,
            signed,
        })
    }

    /// Creates information for an 8-bit unsigned component.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero.
    pub fn u8(size: Size) -> Result<Self> {
        Self::create(size, 8, false)
    }

    /// Creates information for an unsigned component with `precision` bits per sample, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn u16(size: Size, precision: u8) -> Result<Self> {
        Self::create(size, precision, false)
    }

    /// Creates information for a signed component with `precision` bits per sample, stored as 16 bits.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or `precision` is not between 1 and 16.
    pub fn i16(size: Size, precision: u8) -> Result<Self> {
        Self::create(size, precision, true)
    }

    /// Returns the component size in pixels.
    #[must_use]
    pub const fn size(self) -> Size {
        self.size
    }

    /// Returns the bits per sample.
    #[must_use]
    pub const fn precision(self) -> u8 {
        self.precision
    }

    /// Returns whether the samples are signed.
    #[must_use]
    pub const fn signed(self) -> bool {
        self.signed
    }

    /// Returns the sample type that decoding this component produces.
    ///
    /// # Errors
    ///
    /// Returns an error if the bits per sample are zero or more than 16.
    pub fn decode_pixel_type(self) -> Result<ImageType> {
        if self.precision == 0 {
            return Err(Error::OutOfRange {
                name: "precision".into(),
            });
        }
        if self.signed {
            if self.precision <= 16 {
                Ok(ImageType::I16)
            } else {
                Err(Error::OutOfRange {
                    name: "precision".into(),
                })
            }
        } else if self.precision <= 8 {
            Ok(ImageType::U8)
        } else if self.precision <= 16 {
            Ok(ImageType::U16)
        } else {
            Err(Error::OutOfRange {
                name: "precision".into(),
            })
        }
    }

    pub(crate) fn from_raw(raw: sys::nvjpeg2kImageComponentInfo_t) -> Self {
        Self {
            size: Size::new(raw.component_width, raw.component_height),
            precision: raw.precision,
            signed: raw.sgn != 0,
        }
    }

    pub(crate) fn into_raw(self) -> sys::nvjpeg2kImageComponentInfo_t {
        sys::nvjpeg2kImageComponentInfo_t {
            component_width: self.size.width,
            component_height: self.size.height,
            precision: self.precision,
            sgn: u8::from(self.signed),
        }
    }
}

impl TileDecode {
    /// Selects tile `tile_id` and the number of resolution levels to decode.
    ///
    /// The decoder checks the tile index and the number of levels when the selection is used.
    ///
    /// # Errors
    ///
    /// Returns an error if `resolution_levels` is zero.
    pub fn create(tile_id: u32, resolution_levels: u32) -> Result<Self> {
        if resolution_levels == 0 {
            return Err(Error::Empty {
                name: "resolution_levels".into(),
            });
        }
        Ok(Self {
            tile_id,
            resolution_levels,
        })
    }

    /// Returns the tile index, in raster order.
    #[must_use]
    pub const fn tile_id(self) -> u32 {
        self.tile_id
    }

    /// Returns the number of resolution levels to decode.
    #[must_use]
    pub const fn resolution_levels(self) -> u32 {
        self.resolution_levels
    }
}

impl ImageType {
    pub(crate) const fn element_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
        }
    }
}

pub(crate) fn validate_component_precision(precision: u8) -> Result<()> {
    if (1..=16).contains(&precision) {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            name: "precision".into(),
        })
    }
}

pub(crate) fn validate_dimensions(size: Size, name: &'static str) -> Result<()> {
    if size.is_empty() {
        Err(Error::Empty { name: name.into() })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_resolutions(resolutions: u32) -> Result<()> {
    if resolutions == 0 {
        Err(Error::Empty {
            name: "resolutions".into(),
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_code_block_size(size: Size) -> Result<()> {
    if !matches!(size.width, 32 | 64) {
        return Err(Error::OutOfRange {
            name: "code_block_width".into(),
        });
    }

    if !matches!(size.height, 32 | 64) {
        return Err(Error::OutOfRange {
            name: "code_block_height".into(),
        });
    }
    Ok(())
}

pub(crate) fn decode_pixel_type(component_infos: &[ImageComponentInfo]) -> Result<ImageType> {
    let (first, rest) = component_infos.split_first().ok_or(Error::Empty {
        name: "component_infos".into(),
    })?;
    let pixel_type = first.decode_pixel_type()?;
    for component in rest {
        if component.decode_pixel_type()? != pixel_type {
            return Err(Error::UnsupportedDecodeConfiguration {
                reason:
                    "mixed JPEG 2000 component pixel types are not supported by decode_to_device"
                        .into(),
            });
        }
    }
    Ok(pixel_type)
}

impl_enum_display!(Backend, {
    Backend::Default => "NVJPEG2K_BACKEND_DEFAULT",
});

impl_enum_display!(ColorSpace, {
    ColorSpace::NotSupported => "NVJPEG2K_COLORSPACE_NOT_SUPPORTED",
    ColorSpace::Unknown => "NVJPEG2K_COLORSPACE_UNKNOWN",
    ColorSpace::Srgb => "NVJPEG2K_COLORSPACE_SRGB",
    ColorSpace::Gray => "NVJPEG2K_COLORSPACE_GRAY",
    ColorSpace::Sycc => "NVJPEG2K_COLORSPACE_SYCC",
});

impl_enum_display!(ImageType, {
    ImageType::U8 => "NVJPEG2K_UINT8",
    ImageType::U16 => "NVJPEG2K_UINT16",
    ImageType::I16 => "NVJPEG2K_INT16",
});

impl_enum_display!(ImageFormat, {
    ImageFormat::Planar => "NVJPEG2K_FORMAT_PLANAR",
    ImageFormat::Interleaved => "NVJPEG2K_FORMAT_INTERLEAVED",
});

impl_enum_display!(BitstreamType, {
    BitstreamType::J2k => "NVJPEG2K_STREAM_J2K",
    BitstreamType::Jp2 => "NVJPEG2K_STREAM_JP2",
});

impl_enum_display!(ProgressionOrder, {
    ProgressionOrder::Lrcp => "NVJPEG2K_LRCP",
    ProgressionOrder::Rlcp => "NVJPEG2K_RLCP",
    ProgressionOrder::Rpcl => "NVJPEG2K_RPCL",
    ProgressionOrder::Pcrl => "NVJPEG2K_PCRL",
    ProgressionOrder::Cprl => "NVJPEG2K_CPRL",
});

impl_enum_display!(QualityType, {
    QualityType::TargetPsnr => "NVJPEG2K_QUALITY_TYPE_TARGET_PSNR",
    QualityType::QuantizationStep => "NVJPEG2K_QUALITY_TYPE_QUANTIZATION_STEP",
    QualityType::QFactor => "NVJPEG2K_QUALITY_TYPE_Q_FACTOR",
});
