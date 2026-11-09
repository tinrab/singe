//! TIFF formats, tags, and metadata values.

use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_nvtiff_sys as sys;

use crate::error::{Error, Result};

bitflags::bitflags! {
    /// Image type, from the `NewSubfileType` (254) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageTypeFlags: u32 {
        /// Reduced-resolution version of another image.
        const REDUCED_IMAGE = sys::nvtiffImageType::NVTIFF_IMAGETYPE_REDUCED_IMAGE as _;
        /// Page of a multi-page image.
        const PAGE = sys::nvtiffImageType::NVTIFF_IMAGETYPE_PAGE as _;
        /// Transparency mask of another image.
        const MASK = sys::nvtiffImageType::NVTIFF_IMAGETYPE_MASK as _;
    }
}

impl Display for ImageTypeFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// TIFF file variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Variant {
    /// Regular TIFF, up to 4 GB.
    Regular = sys::nvtiffVariant_t::NVTIFF_REGULAR_TIFF as _,
    /// `BigTIFF`, larger than 4 GB.
    Big = sys::nvtiffVariant_t::NVTIFF_BIG_TIFF as _,
}

impl_enum_conversion!(sys::nvtiffVariant_t, Variant);

impl_enum_display!(Variant, {
    Self::Regular => "NVTIFF_REGULAR_TIFF",
    Self::Big => "NVTIFF_BIG_TIFF",
});

/// `GeoTIFF` key, as defined by the `GeoTIFF` specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GeoKey {
    /// Model type.
    GtModelType = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GT_MODEL_TYPE as _,
    /// Raster type.
    GtRasterType = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GT_RASTER_TYPE as _,
    /// Citation.
    GtCitation = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GT_CITATION as _,
    /// Geodetic CRS.
    GeodeticCrs = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEODETIC_CRS as _,
    /// Geodetic citation.
    GeodeticCitation = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEODETIC_CITATION as _,
    /// Geodetic datum.
    GeodeticDatum = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEODETIC_DATUM as _,
    /// Prime meridian.
    PrimeMeridian = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PRIME_MERIDIAN as _,
    /// Geographic linear units.
    GeogLinearUnits = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEOG_LINEAR_UNITS as _,
    /// Geographic linear unit size.
    GeogLinearUnitSize = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEOG_LINEAR_UNIT_SIZE as _,
    /// Geographic angular units.
    GeogAngularUnits = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEOG_ANGULAR_UNITS as _,
    /// Geographic angular unit size.
    GeogAngularUnitSize = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEOG_ANGULAR_UNIT_SIZE as _,
    /// Ellipsoid.
    Ellipsoid = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_ELLIPSOID as _,
    /// Ellipsoid semi-major axis.
    EllipsoidSemiMajorAxis = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_ELLIPSOID_SEMI_MAJOR_AXIS as _,
    /// Ellipsoid semi-minor axis.
    EllipsoidSemiMinorAxis = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_ELLIPSOID_SEMI_MINOR_AXIS as _,
    /// Ellipsoid inverse flattening.
    EllipsoidInvFlattening = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_ELLIPSOID_INV_FLATTENING as _,
    /// Geographic azimuth units.
    GeogAzimuthUnits = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_GEOG_AZIMUTH_UNITS as _,
    /// Prime meridian longitude.
    PrimeMeridianLong = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PRIME_MERIDIAN_LONG as _,
    /// Projected CRS.
    ProjectedCrs = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJECTED_CRS as _,
    /// Projected citation.
    ProjectedCitation = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJECTED_CITATION as _,
    /// Projection.
    Projection = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJECTION as _,
    /// Projection method.
    ProjMethod = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_METHOD as _,
    /// Projection linear units.
    ProjLinearUnits = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_LINEAR_UNITS as _,
    /// Projection linear unit size.
    ProjLinearUnitSize = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_LINEAR_UNIT_SIZE as _,
    /// Standard parallel 1.
    ProjStdParallel1 = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_STD_PARALLEL1 as _,
    /// Standard parallel 2.
    ProjStdParallel2 = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_STD_PARALLEL2 as _,
    /// Natural origin longitude.
    ProjNatOriginLong = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LONG as _,
    /// Natural origin latitude.
    ProjNatOriginLat = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LAT as _,
    /// False easting.
    ProjFalseEasting = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_EASTING as _,
    /// False northing.
    ProjFalseNorthing = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_NORTHING as _,
    /// False origin longitude.
    ProjFalseOriginLong = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LONG as _,
    /// False origin latitude.
    ProjFalseOriginLat = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LAT as _,
    /// False origin easting.
    ProjFalseOriginEasting = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_EASTING as _,
    /// False origin northing.
    ProjFalseOriginNorthing = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_NORTHING as _,
    /// Center longitude.
    ProjCenterLong = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_CENTER_LONG as _,
    /// Center latitude.
    ProjCenterLat = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_CENTER_LAT as _,
    /// Center easting.
    ProjCenterEasting = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_CENTER_EASTING as _,
    /// Center northing.
    ProjCenterNorthing = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_CENTER_NORTHING as _,
    /// Scale at natural origin.
    ProjScaleAtNatOrigin = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN as _,
    /// Scale at center.
    ProjScaleAtCenter = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_SCALE_AT_CENTER as _,
    /// Azimuth angle.
    ProjAzimuthAngle = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_AZIMUTH_ANGLE as _,
    /// Straight vertical pole longitude.
    ProjStraightVertPoleLong = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG as _,
    /// Vertical CRS.
    Vertical = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_VERTICAL as _,
    /// Vertical citation.
    VerticalCitation = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_VERTICAL_CITATION as _,
    /// Vertical datum.
    VerticalDatum = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_VERTICAL_DATUM as _,
    /// Vertical units.
    VerticalUnits = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_VERTICAL_UNITS as _,
    /// Start of the private key range.
    Base = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_BASE as _,
    /// End of the key range.
    End = sys::nvtiffGeoKey_t::NVTIFF_GEOKEY_END as _,
}

impl_enum_conversion!(sys::nvtiffGeoKey_t, GeoKey);

impl_enum_display!(GeoKey, {
    Self::GtModelType => "NVTIFF_GEOKEY_GT_MODEL_TYPE",
    Self::GtRasterType => "NVTIFF_GEOKEY_GT_RASTER_TYPE",
    Self::GtCitation => "NVTIFF_GEOKEY_GT_CITATION",
    Self::GeodeticCrs => "NVTIFF_GEOKEY_GEODETIC_CRS",
    Self::GeodeticCitation => "NVTIFF_GEOKEY_GEODETIC_CITATION",
    Self::GeodeticDatum => "NVTIFF_GEOKEY_GEODETIC_DATUM",
    Self::PrimeMeridian => "NVTIFF_GEOKEY_PRIME_MERIDIAN",
    Self::GeogLinearUnits => "NVTIFF_GEOKEY_GEOG_LINEAR_UNITS",
    Self::GeogLinearUnitSize => "NVTIFF_GEOKEY_GEOG_LINEAR_UNIT_SIZE",
    Self::GeogAngularUnits => "NVTIFF_GEOKEY_GEOG_ANGULAR_UNITS",
    Self::GeogAngularUnitSize => "NVTIFF_GEOKEY_GEOG_ANGULAR_UNIT_SIZE",
    Self::Ellipsoid => "NVTIFF_GEOKEY_ELLIPSOID",
    Self::EllipsoidSemiMajorAxis => "NVTIFF_GEOKEY_ELLIPSOID_SEMI_MAJOR_AXIS",
    Self::EllipsoidSemiMinorAxis => "NVTIFF_GEOKEY_ELLIPSOID_SEMI_MINOR_AXIS",
    Self::EllipsoidInvFlattening => "NVTIFF_GEOKEY_ELLIPSOID_INV_FLATTENING",
    Self::GeogAzimuthUnits => "NVTIFF_GEOKEY_GEOG_AZIMUTH_UNITS",
    Self::PrimeMeridianLong => "NVTIFF_GEOKEY_PRIME_MERIDIAN_LONG",
    Self::ProjectedCrs => "NVTIFF_GEOKEY_PROJECTED_CRS",
    Self::ProjectedCitation => "NVTIFF_GEOKEY_PROJECTED_CITATION",
    Self::Projection => "NVTIFF_GEOKEY_PROJECTION",
    Self::ProjMethod => "NVTIFF_GEOKEY_PROJ_METHOD",
    Self::ProjLinearUnits => "NVTIFF_GEOKEY_PROJ_LINEAR_UNITS",
    Self::ProjLinearUnitSize => "NVTIFF_GEOKEY_PROJ_LINEAR_UNIT_SIZE",
    Self::ProjStdParallel1 => "NVTIFF_GEOKEY_PROJ_STD_PARALLEL1",
    Self::ProjStdParallel2 => "NVTIFF_GEOKEY_PROJ_STD_PARALLEL2",
    Self::ProjNatOriginLong => "NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LONG",
    Self::ProjNatOriginLat => "NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LAT",
    Self::ProjFalseEasting => "NVTIFF_GEOKEY_PROJ_FALSE_EASTING",
    Self::ProjFalseNorthing => "NVTIFF_GEOKEY_PROJ_FALSE_NORTHING",
    Self::ProjFalseOriginLong => "NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LONG",
    Self::ProjFalseOriginLat => "NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LAT",
    Self::ProjFalseOriginEasting => "NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_EASTING",
    Self::ProjFalseOriginNorthing => "NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_NORTHING",
    Self::ProjCenterLong => "NVTIFF_GEOKEY_PROJ_CENTER_LONG",
    Self::ProjCenterLat => "NVTIFF_GEOKEY_PROJ_CENTER_LAT",
    Self::ProjCenterEasting => "NVTIFF_GEOKEY_PROJ_CENTER_EASTING",
    Self::ProjCenterNorthing => "NVTIFF_GEOKEY_PROJ_CENTER_NORTHING",
    Self::ProjScaleAtNatOrigin => "NVTIFF_GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN",
    Self::ProjScaleAtCenter => "NVTIFF_GEOKEY_PROJ_SCALE_AT_CENTER",
    Self::ProjAzimuthAngle => "NVTIFF_GEOKEY_PROJ_AZIMUTH_ANGLE",
    Self::ProjStraightVertPoleLong => "NVTIFF_GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG",
    Self::Vertical => "NVTIFF_GEOKEY_VERTICAL",
    Self::VerticalCitation => "NVTIFF_GEOKEY_VERTICAL_CITATION",
    Self::VerticalDatum => "NVTIFF_GEOKEY_VERTICAL_DATUM",
    Self::VerticalUnits => "NVTIFF_GEOKEY_VERTICAL_UNITS",
    Self::Base => "NVTIFF_GEOKEY_BASE",
    Self::End => "NVTIFF_GEOKEY_END",
});

/// Type of a `GeoTIFF` key value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GeoKeyDataType {
    /// 16-bit unsigned integer.
    Short = sys::nvtiffGeoKeyDataType_t::NVTIFF_GEOKEY_TYPE_SHORT as _,
    /// ASCII string.
    Ascii = sys::nvtiffGeoKeyDataType_t::NVTIFF_GEOKEY_TYPE_ASCII as _,
    /// 64-bit IEEE floating point.
    Double = sys::nvtiffGeoKeyDataType_t::NVTIFF_GEOKEY_TYPE_DOUBLE as _,
    /// Unknown type.
    Unknown = sys::nvtiffGeoKeyDataType_t::NVTIFF_GEOKEY_TYPE_UNKNOWN as _,
}

impl_enum_conversion!(sys::nvtiffGeoKeyDataType_t, GeoKeyDataType);

impl_enum_display!(GeoKeyDataType, {
    Self::Short => "NVTIFF_GEOKEY_TYPE_SHORT",
    Self::Ascii => "NVTIFF_GEOKEY_TYPE_ASCII",
    Self::Double => "NVTIFF_GEOKEY_TYPE_DOUBLE",
    Self::Unknown => "NVTIFF_GEOKEY_TYPE_UNKNOWN",
});

/// Type of a TIFF tag value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum TagDataType {
    /// 8-bit unsigned integer.
    Byte = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_BYTE as _,
    /// NUL-terminated ASCII string.
    Ascii = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_ASCII as _,
    /// 16-bit unsigned integer.
    Short = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SHORT as _,
    /// 32-bit unsigned integer.
    Long = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_LONG as _,
    /// Two LONGs: numerator and denominator.
    Rational = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_RATIONAL as _,
    /// 8-bit signed integer.
    Sbyte = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SBYTE as _,
    /// Byte of any value.
    Undefined = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_UNDEFINED as _,
    /// 16-bit signed integer.
    SignedShort = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SSHORT as _,
    /// 32-bit signed integer.
    Slong = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SLONG as _,
    /// Two SLONGs: numerator and denominator.
    SignedRational = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SRATIONAL as _,
    /// 32-bit IEEE floating point.
    Float = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_FLOAT as _,
    /// 64-bit IEEE floating point.
    Double = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_DOUBLE as _,
    /// `BigTIFF` 64-bit unsigned integer.
    Long8 = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_LONG8 as _,
    /// `BigTIFF` 64-bit signed integer.
    Slong8 = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_SLONG8 as _,
    /// `BigTIFF` 64-bit unsigned IFD offset.
    Ifd8 = sys::nvtiffTagDataType_t::NVTIFF_TAG_TYPE_IFD8 as _,
}

impl_enum_conversion!(sys::nvtiffTagDataType_t, TagDataType);

impl_enum_display!(TagDataType, {
    Self::Byte => "NVTIFF_TAG_TYPE_BYTE",
    Self::Ascii => "NVTIFF_TAG_TYPE_ASCII",
    Self::Short => "NVTIFF_TAG_TYPE_SHORT",
    Self::Long => "NVTIFF_TAG_TYPE_LONG",
    Self::Rational => "NVTIFF_TAG_TYPE_RATIONAL",
    Self::Sbyte => "NVTIFF_TAG_TYPE_SBYTE",
    Self::Undefined => "NVTIFF_TAG_TYPE_UNDEFINED",
    Self::SignedShort => "NVTIFF_TAG_TYPE_SSHORT",
    Self::Slong => "NVTIFF_TAG_TYPE_SLONG",
    Self::SignedRational => "NVTIFF_TAG_TYPE_SRATIONAL",
    Self::Float => "NVTIFF_TAG_TYPE_FLOAT",
    Self::Double => "NVTIFF_TAG_TYPE_DOUBLE",
    Self::Long8 => "NVTIFF_TAG_TYPE_LONG8",
    Self::Slong8 => "NVTIFF_TAG_TYPE_SLONG8",
    Self::Ifd8 => "NVTIFF_TAG_TYPE_IFD8",
});

/// TIFF tag that nvTIFF names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Tag {
    /// `GeoTIFF` `ModelPixelScaleTag` (33550).
    ModelPixelScale = sys::NVTIFF_TAG_MODEL_PIXEL_SCALE,
    /// `GeoTIFF` `ModelTiepointTag` (33922).
    ModelTiePoint = sys::NVTIFF_TAG_MODEL_TIE_POINT,
    /// `GeoTIFF` `ModelTransformationTag` (34264).
    ModelTransformation = sys::NVTIFF_TAG_MODEL_TRANSFORMATION,
}

impl Tag {
    /// Returns the tag ID.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self as u16
    }
}

impl From<Tag> for u16 {
    fn from(tag: Tag) -> Self {
        tag.raw()
    }
}

impl_enum_display!(Tag, {
    Self::ModelPixelScale => "NVTIFF_TAG_MODEL_PIXEL_SCALE",
    Self::ModelTiePoint => "NVTIFF_TAG_MODEL_TIE_POINT",
    Self::ModelTransformation => "NVTIFF_TAG_MODEL_TRANSFORMATION",
});

/// Sample format, from the `SampleFormat` (339) TIFF tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SampleFormat {
    /// Unknown sample format.
    Unknown = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_UNKNOWN as _,
    /// Unsigned integer.
    Uint = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_UINT as _,
    /// Signed integer.
    Int = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_INT as _,
    /// IEEE floating point.
    IeeeFloat = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_IEEEFP as _,
    /// Undefined data.
    Void = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_VOID as _,
    /// Complex signed integer.
    ComplexInt = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_COMPLEXINT as _,
    /// Complex IEEE floating point.
    ComplexIeeeFloat = sys::nvtiffSampleFormat_t::NVTIFF_SAMPLEFORMAT_COMPLEXIEEEFP as _,
}

impl_enum_conversion!(sys::nvtiffSampleFormat_t, SampleFormat);

impl_enum_display!(SampleFormat, {
    Self::Unknown => "NVTIFF_SAMPLEFORMAT_UNKNOWN",
    Self::Uint => "NVTIFF_SAMPLEFORMAT_UINT",
    Self::Int => "NVTIFF_SAMPLEFORMAT_INT",
    Self::IeeeFloat => "NVTIFF_SAMPLEFORMAT_IEEEFP",
    Self::Void => "NVTIFF_SAMPLEFORMAT_VOID",
    Self::ComplexInt => "NVTIFF_SAMPLEFORMAT_COMPLEXINT",
    Self::ComplexIeeeFloat => "NVTIFF_SAMPLEFORMAT_COMPLEXIEEEFP",
});

/// Color space, from the `PhotometricInterpretation` (262) TIFF tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum PhotometricInterpretation {
    /// Unknown photometric interpretation.
    Unknown = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_UNKNOWN as _,
    /// Grayscale where 0 is white.
    MinisWhite = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_MINISWHITE as _,
    /// Grayscale where 0 is black.
    MinisBlack = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_MINISBLACK as _,
    /// RGB.
    Rgb = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_RGB as _,
    /// Indexes into a color map.
    Palette = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_PALETTE as _,
    /// Transparency mask.
    Mask = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_MASK as _,
    /// Color separations, usually CMYK.
    Separated = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_SEPARATED as _,
    /// `YCbCr`.
    Ycbcr = sys::nvtiffPhotometricInt_t::NVTIFF_PHOTOMETRIC_YCBCR as _,
}

impl_enum_conversion!(i32, sys::nvtiffPhotometricInt_t, PhotometricInterpretation,);

impl_enum_display!(PhotometricInterpretation, {
    Self::Unknown => "NVTIFF_PHOTOMETRIC_UNKNOWN",
    Self::MinisWhite => "NVTIFF_PHOTOMETRIC_MINISWHITE",
    Self::MinisBlack => "NVTIFF_PHOTOMETRIC_MINISBLACK",
    Self::Rgb => "NVTIFF_PHOTOMETRIC_RGB",
    Self::Palette => "NVTIFF_PHOTOMETRIC_PALETTE",
    Self::Mask => "NVTIFF_PHOTOMETRIC_MASK",
    Self::Separated => "NVTIFF_PHOTOMETRIC_SEPARATED",
    Self::Ycbcr => "NVTIFF_PHOTOMETRIC_YCBCR",
});

/// Sample layout, from the `PlanarConfiguration` (284) TIFF tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanarConfig {
    /// Unknown planar configuration.
    Unknown = sys::nvtiffPlanarConfig_t::NVTIFF_PLANARCONFIG_UNKNOWN as _,
    /// Samples of each pixel stored together.
    Contiguous = sys::nvtiffPlanarConfig_t::NVTIFF_PLANARCONFIG_CONTIG as _,
    /// Each sample stored in its own plane.
    Separate = sys::nvtiffPlanarConfig_t::NVTIFF_PLANARCONFIG_SEPARATE as _,
}

impl_enum_conversion!(sys::nvtiffPlanarConfig_t, PlanarConfig);

impl_enum_display!(PlanarConfig, {
    Self::Unknown => "NVTIFF_PLANARCONFIG_UNKNOWN",
    Self::Contiguous => "NVTIFF_PLANARCONFIG_CONTIG",
    Self::Separate => "NVTIFF_PLANARCONFIG_SEPARATE",
});

/// Compression, from the Compression (259) TIFF tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Compression {
    /// Unknown compression.
    Unknown = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_UNKNOWN as _,
    /// No compression.
    None = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_NONE as _,
    /// LZW.
    Lzw = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_LZW as _,
    /// JPEG.
    Jpeg = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_JPEG as _,
    /// Adobe Deflate.
    AdobeDeflate = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_ADOBE_DEFLATE as _,
    /// Deflate.
    Deflate = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_DEFLATE as _,
    /// Aperio JPEG 2000 with `YCbCr`.
    AperioJp2000Ycc = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_APERIO_JP2000_YCC as _,
    /// Aperio JPEG 2000 with RGB.
    AperioJp2000Rgb = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_APERIO_JP2000_RGB as _,
    /// JPEG 2000.
    Jp2000 = sys::nvtiffCompression_t::NVTIFF_COMPRESSION_JP2000 as _,
}

impl_enum_conversion!(sys::nvtiffCompression_t, Compression);

impl_enum_display!(Compression, {
    Self::Unknown => "NVTIFF_COMPRESSION_UNKNOWN",
    Self::None => "NVTIFF_COMPRESSION_NONE",
    Self::Lzw => "NVTIFF_COMPRESSION_LZW",
    Self::Jpeg => "NVTIFF_COMPRESSION_JPEG",
    Self::AdobeDeflate => "NVTIFF_COMPRESSION_ADOBE_DEFLATE",
    Self::Deflate => "NVTIFF_COMPRESSION_DEFLATE",
    Self::AperioJp2000Ycc => "NVTIFF_COMPRESSION_APERIO_JP2000_YCC",
    Self::AperioJp2000Rgb => "NVTIFF_COMPRESSION_APERIO_JP2000_RGB",
    Self::Jp2000 => "NVTIFF_COMPRESSION_JP2000",
});

/// Whether an image is stored in strips or tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ImageGeometryType {
    /// Strips of rows.
    Striped = sys::nvtiffImageGeometryType_t::NVTIFF_IMAGE_STRIPED as _,
    /// Rectangular tiles.
    Tiled = sys::nvtiffImageGeometryType_t::NVTIFF_IMAGE_TILED as _,
}

impl_enum_conversion!(sys::nvtiffImageGeometryType_t, ImageGeometryType);

impl_enum_display!(ImageGeometryType, {
    Self::Striped => "NVTIFF_IMAGE_STRIPED",
    Self::Tiled => "NVTIFF_IMAGE_TILED",
});

/// Decode output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum OutputFormat {
    /// Samples as stored, interleaved.
    UnchangedInterleaved = sys::nvtiffOutputFormat_t::NVTIFF_OUTPUT_UNCHANGED_I as _,
    /// Interleaved RGB with 8 bits per sample.
    RgbInterleavedU8 = sys::nvtiffOutputFormat_t::NVTIFF_OUTPUT_RGB_I_UINT8 as _,
    /// Interleaved RGB with 16 bits per sample.
    RgbInterleavedU16 = sys::nvtiffOutputFormat_t::NVTIFF_OUTPUT_RGB_I_UINT16 as _,
}

impl_enum_conversion!(sys::nvtiffOutputFormat_t, OutputFormat);

impl_enum_display!(OutputFormat, {
    Self::UnchangedInterleaved => "NVTIFF_OUTPUT_UNCHANGED_I",
    Self::RgbInterleavedU8 => "NVTIFF_OUTPUT_RGB_I_UINT8",
    Self::RgbInterleavedU16 => "NVTIFF_OUTPUT_RGB_I_UINT16",
});

/// Type and size of a TIFF tag value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagInfo {
    /// Value type.
    pub data_type: TagDataType,
    /// Size of one element in bytes.
    pub size: u32,
    /// Number of elements.
    pub count: u32,
}

impl TagInfo {
    /// Returns the size of the whole value in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the size overflows `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        value_byte_len(self.size, self.count)
    }
}

/// Type and size of a `GeoTIFF` key value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeoKeyInfo {
    /// Size of one element in bytes.
    pub size: u32,
    /// Number of elements.
    pub count: u32,
    /// Value type.
    pub data_type: GeoKeyDataType,
}

impl GeoKeyInfo {
    /// Returns the size of the whole value in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the size overflows `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        value_byte_len(self.size, self.count)
    }
}

fn value_byte_len(size: u32, count: u32) -> Result<usize> {
    usize::try_from(u64::from(size) * u64::from(count))
        .map_err(|_| Error::out_of_range("value size"))
}

/// TIFF tag value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TagValue {
    /// BYTE values.
    Byte(Vec<u8>),
    /// SHORT values.
    Short(Vec<u16>),
    /// LONG values.
    Long(Vec<u32>),
    /// DOUBLE values.
    Double(Vec<f64>),
    /// ASCII string.
    Ascii(String),
    /// Value of another type, as bytes.
    Raw {
        /// Value type.
        data_type: TagDataType,
        /// Value bytes.
        bytes: Vec<u8>,
    },
}

/// `GeoTIFF` key value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GeoKeyValue {
    /// SHORT values.
    Short(Vec<u16>),
    /// DOUBLE values.
    Double(Vec<f64>),
    /// ASCII string.
    Ascii(String),
    /// Value of an unknown type, as bytes.
    Unknown(Vec<u8>),
}
