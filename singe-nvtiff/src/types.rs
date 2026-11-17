//! TIFF formats, tags, and metadata values.

use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_nvtiff_sys as sys;

use crate::{error::Result, utility::to_usize};

bitflags::bitflags! {
    /// Image type, from the `NewSubfileType` (254) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageTypeFlags: u32 {
        /// Reduced-resolution version of another image.
        const REDUCED_IMAGE = sys::nvtiffImageType::NVTIFF_IMAGETYPE_REDUCED_IMAGE.0;
        /// Page of a multi-page image.
        const PAGE = sys::nvtiffImageType::NVTIFF_IMAGETYPE_PAGE.0;
        /// Transparency mask of another image.
        const MASK = sys::nvtiffImageType::NVTIFF_IMAGETYPE_MASK.0;
    }
}

impl Display for ImageTypeFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// TIFF file variant.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Variant: sys::nvtiffVariant_t {
        /// Regular TIFF, up to 4 GB.
        Regular = NVTIFF_REGULAR_TIFF,
        /// `BigTIFF`, larger than 4 GB.
        Big = NVTIFF_BIG_TIFF,
    }
}

impl_enum_display!(Variant, {
    Self::Regular => "NVTIFF_REGULAR_TIFF",
    Self::Big => "NVTIFF_BIG_TIFF",
});

singe_core::native_enum! {
    /// `GeoTIFF` key, as defined by the `GeoTIFF` specification.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum GeoKey: sys::nvtiffGeoKey_t {
        /// Model type.
        GtModelType = NVTIFF_GEOKEY_GT_MODEL_TYPE,
        /// Raster type.
        GtRasterType = NVTIFF_GEOKEY_GT_RASTER_TYPE,
        /// Citation.
        GtCitation = NVTIFF_GEOKEY_GT_CITATION,
        /// Geodetic CRS.
        GeodeticCrs = NVTIFF_GEOKEY_GEODETIC_CRS,
        /// Geodetic citation.
        GeodeticCitation = NVTIFF_GEOKEY_GEODETIC_CITATION,
        /// Geodetic datum.
        GeodeticDatum = NVTIFF_GEOKEY_GEODETIC_DATUM,
        /// Prime meridian.
        PrimeMeridian = NVTIFF_GEOKEY_PRIME_MERIDIAN,
        /// Geographic linear units.
        GeogLinearUnits = NVTIFF_GEOKEY_GEOG_LINEAR_UNITS,
        /// Geographic linear unit size.
        GeogLinearUnitSize = NVTIFF_GEOKEY_GEOG_LINEAR_UNIT_SIZE,
        /// Geographic angular units.
        GeogAngularUnits = NVTIFF_GEOKEY_GEOG_ANGULAR_UNITS,
        /// Geographic angular unit size.
        GeogAngularUnitSize = NVTIFF_GEOKEY_GEOG_ANGULAR_UNIT_SIZE,
        /// Ellipsoid.
        Ellipsoid = NVTIFF_GEOKEY_ELLIPSOID,
        /// Ellipsoid semi-major axis.
        EllipsoidSemiMajorAxis = NVTIFF_GEOKEY_ELLIPSOID_SEMI_MAJOR_AXIS,
        /// Ellipsoid semi-minor axis.
        EllipsoidSemiMinorAxis = NVTIFF_GEOKEY_ELLIPSOID_SEMI_MINOR_AXIS,
        /// Ellipsoid inverse flattening.
        EllipsoidInvFlattening = NVTIFF_GEOKEY_ELLIPSOID_INV_FLATTENING,
        /// Geographic azimuth units.
        GeogAzimuthUnits = NVTIFF_GEOKEY_GEOG_AZIMUTH_UNITS,
        /// Prime meridian longitude.
        PrimeMeridianLong = NVTIFF_GEOKEY_PRIME_MERIDIAN_LONG,
        /// Projected CRS.
        ProjectedCrs = NVTIFF_GEOKEY_PROJECTED_CRS,
        /// Projected citation.
        ProjectedCitation = NVTIFF_GEOKEY_PROJECTED_CITATION,
        /// Projection.
        Projection = NVTIFF_GEOKEY_PROJECTION,
        /// Projection method.
        ProjMethod = NVTIFF_GEOKEY_PROJ_METHOD,
        /// Projection linear units.
        ProjLinearUnits = NVTIFF_GEOKEY_PROJ_LINEAR_UNITS,
        /// Projection linear unit size.
        ProjLinearUnitSize = NVTIFF_GEOKEY_PROJ_LINEAR_UNIT_SIZE,
        /// Standard parallel 1.
        ProjStdParallel1 = NVTIFF_GEOKEY_PROJ_STD_PARALLEL1,
        /// Standard parallel 2.
        ProjStdParallel2 = NVTIFF_GEOKEY_PROJ_STD_PARALLEL2,
        /// Natural origin longitude.
        ProjNatOriginLong = NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LONG,
        /// Natural origin latitude.
        ProjNatOriginLat = NVTIFF_GEOKEY_PROJ_NAT_ORIGIN_LAT,
        /// False easting.
        ProjFalseEasting = NVTIFF_GEOKEY_PROJ_FALSE_EASTING,
        /// False northing.
        ProjFalseNorthing = NVTIFF_GEOKEY_PROJ_FALSE_NORTHING,
        /// False origin longitude.
        ProjFalseOriginLong = NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LONG,
        /// False origin latitude.
        ProjFalseOriginLat = NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_LAT,
        /// False origin easting.
        ProjFalseOriginEasting = NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_EASTING,
        /// False origin northing.
        ProjFalseOriginNorthing = NVTIFF_GEOKEY_PROJ_FALSE_ORIGIN_NORTHING,
        /// Center longitude.
        ProjCenterLong = NVTIFF_GEOKEY_PROJ_CENTER_LONG,
        /// Center latitude.
        ProjCenterLat = NVTIFF_GEOKEY_PROJ_CENTER_LAT,
        /// Center easting.
        ProjCenterEasting = NVTIFF_GEOKEY_PROJ_CENTER_EASTING,
        /// Center northing.
        ProjCenterNorthing = NVTIFF_GEOKEY_PROJ_CENTER_NORTHING,
        /// Scale at natural origin.
        ProjScaleAtNatOrigin = NVTIFF_GEOKEY_PROJ_SCALE_AT_NAT_ORIGIN,
        /// Scale at center.
        ProjScaleAtCenter = NVTIFF_GEOKEY_PROJ_SCALE_AT_CENTER,
        /// Azimuth angle.
        ProjAzimuthAngle = NVTIFF_GEOKEY_PROJ_AZIMUTH_ANGLE,
        /// Straight vertical pole longitude.
        ProjStraightVertPoleLong = NVTIFF_GEOKEY_PROJ_STRAIGHT_VERT_POLE_LONG,
        /// Vertical CRS.
        Vertical = NVTIFF_GEOKEY_VERTICAL,
        /// Vertical citation.
        VerticalCitation = NVTIFF_GEOKEY_VERTICAL_CITATION,
        /// Vertical datum.
        VerticalDatum = NVTIFF_GEOKEY_VERTICAL_DATUM,
        /// Vertical units.
        VerticalUnits = NVTIFF_GEOKEY_VERTICAL_UNITS,
        /// Start of the private key range.
        Base = NVTIFF_GEOKEY_BASE,
        /// End of the key range.
        End = NVTIFF_GEOKEY_END,
    }
}

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

singe_core::native_enum! {
    /// Type of a `GeoTIFF` key value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum GeoKeyDataType: sys::nvtiffGeoKeyDataType_t {
        /// 16-bit unsigned integer.
        Short = NVTIFF_GEOKEY_TYPE_SHORT,
        /// ASCII string.
        Ascii = NVTIFF_GEOKEY_TYPE_ASCII,
        /// 64-bit IEEE floating point.
        Double = NVTIFF_GEOKEY_TYPE_DOUBLE,
        /// Unknown type.
        Unknown = NVTIFF_GEOKEY_TYPE_UNKNOWN,
    }
}

impl_enum_display!(GeoKeyDataType, {
    Self::Short => "NVTIFF_GEOKEY_TYPE_SHORT",
    Self::Ascii => "NVTIFF_GEOKEY_TYPE_ASCII",
    Self::Double => "NVTIFF_GEOKEY_TYPE_DOUBLE",
    Self::Unknown => "NVTIFF_GEOKEY_TYPE_UNKNOWN",
});

singe_core::native_enum! {
    /// Type of a TIFF tag value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum TagDataType: sys::nvtiffTagDataType_t {
        /// 8-bit unsigned integer.
        Byte = NVTIFF_TAG_TYPE_BYTE,
        /// NUL-terminated ASCII string.
        Ascii = NVTIFF_TAG_TYPE_ASCII,
        /// 16-bit unsigned integer.
        Short = NVTIFF_TAG_TYPE_SHORT,
        /// 32-bit unsigned integer.
        Long = NVTIFF_TAG_TYPE_LONG,
        /// Two LONGs: numerator and denominator.
        Rational = NVTIFF_TAG_TYPE_RATIONAL,
        /// 8-bit signed integer.
        Sbyte = NVTIFF_TAG_TYPE_SBYTE,
        /// Byte of any value.
        Undefined = NVTIFF_TAG_TYPE_UNDEFINED,
        /// 16-bit signed integer.
        SignedShort = NVTIFF_TAG_TYPE_SSHORT,
        /// 32-bit signed integer.
        Slong = NVTIFF_TAG_TYPE_SLONG,
        /// Two SLONGs: numerator and denominator.
        SignedRational = NVTIFF_TAG_TYPE_SRATIONAL,
        /// 32-bit IEEE floating point.
        Float = NVTIFF_TAG_TYPE_FLOAT,
        /// 64-bit IEEE floating point.
        Double = NVTIFF_TAG_TYPE_DOUBLE,
        /// `BigTIFF` 64-bit unsigned integer.
        Long8 = NVTIFF_TAG_TYPE_LONG8,
        /// `BigTIFF` 64-bit signed integer.
        Slong8 = NVTIFF_TAG_TYPE_SLONG8,
        /// `BigTIFF` 64-bit unsigned IFD offset.
        Ifd8 = NVTIFF_TAG_TYPE_IFD8,
    }
}

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
        // Every variant is a TIFF tag ID, which is 16-bit, so the narrowing keeps the value.
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

singe_core::native_enum! {
    /// Sample format, from the `SampleFormat` (339) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SampleFormat: sys::nvtiffSampleFormat_t {
        /// Unknown sample format.
        Unknown = NVTIFF_SAMPLEFORMAT_UNKNOWN,
        /// Unsigned integer.
        Uint = NVTIFF_SAMPLEFORMAT_UINT,
        /// Signed integer.
        Int = NVTIFF_SAMPLEFORMAT_INT,
        /// IEEE floating point.
        IeeeFloat = NVTIFF_SAMPLEFORMAT_IEEEFP,
        /// Undefined data.
        Void = NVTIFF_SAMPLEFORMAT_VOID,
        /// Complex signed integer.
        ComplexInt = NVTIFF_SAMPLEFORMAT_COMPLEXINT,
        /// Complex IEEE floating point.
        ComplexIeeeFloat = NVTIFF_SAMPLEFORMAT_COMPLEXIEEEFP,
    }
}

impl_enum_display!(SampleFormat, {
    Self::Unknown => "NVTIFF_SAMPLEFORMAT_UNKNOWN",
    Self::Uint => "NVTIFF_SAMPLEFORMAT_UINT",
    Self::Int => "NVTIFF_SAMPLEFORMAT_INT",
    Self::IeeeFloat => "NVTIFF_SAMPLEFORMAT_IEEEFP",
    Self::Void => "NVTIFF_SAMPLEFORMAT_VOID",
    Self::ComplexInt => "NVTIFF_SAMPLEFORMAT_COMPLEXINT",
    Self::ComplexIeeeFloat => "NVTIFF_SAMPLEFORMAT_COMPLEXIEEEFP",
});

singe_core::native_enum! {
    /// Color space, from the `PhotometricInterpretation` (262) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum PhotometricInterpretation: sys::nvtiffPhotometricInt_t {
        /// Unknown photometric interpretation.
        Unknown = NVTIFF_PHOTOMETRIC_UNKNOWN,
        /// Grayscale where 0 is white.
        MinisWhite = NVTIFF_PHOTOMETRIC_MINISWHITE,
        /// Grayscale where 0 is black.
        MinisBlack = NVTIFF_PHOTOMETRIC_MINISBLACK,
        /// RGB.
        Rgb = NVTIFF_PHOTOMETRIC_RGB,
        /// Indexes into a color map.
        Palette = NVTIFF_PHOTOMETRIC_PALETTE,
        /// Transparency mask.
        Mask = NVTIFF_PHOTOMETRIC_MASK,
        /// Color separations, usually CMYK.
        Separated = NVTIFF_PHOTOMETRIC_SEPARATED,
        /// `YCbCr`.
        Ycbcr = NVTIFF_PHOTOMETRIC_YCBCR,
    }
}

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

singe_core::native_enum! {
    /// Sample layout, from the `PlanarConfiguration` (284) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PlanarConfig: sys::nvtiffPlanarConfig_t {
        /// Unknown planar configuration.
        Unknown = NVTIFF_PLANARCONFIG_UNKNOWN,
        /// Samples of each pixel stored together.
        Contiguous = NVTIFF_PLANARCONFIG_CONTIG,
        /// Each sample stored in its own plane.
        Separate = NVTIFF_PLANARCONFIG_SEPARATE,
    }
}

impl_enum_display!(PlanarConfig, {
    Self::Unknown => "NVTIFF_PLANARCONFIG_UNKNOWN",
    Self::Contiguous => "NVTIFF_PLANARCONFIG_CONTIG",
    Self::Separate => "NVTIFF_PLANARCONFIG_SEPARATE",
});

singe_core::native_enum! {
    /// Compression, from the Compression (259) TIFF tag.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Compression: sys::nvtiffCompression_t {
        /// Unknown compression.
        Unknown = NVTIFF_COMPRESSION_UNKNOWN,
        /// No compression.
        None = NVTIFF_COMPRESSION_NONE,
        /// LZW.
        Lzw = NVTIFF_COMPRESSION_LZW,
        /// JPEG.
        Jpeg = NVTIFF_COMPRESSION_JPEG,
        /// Adobe Deflate.
        AdobeDeflate = NVTIFF_COMPRESSION_ADOBE_DEFLATE,
        /// Deflate.
        Deflate = NVTIFF_COMPRESSION_DEFLATE,
        /// Aperio JPEG 2000 with `YCbCr`.
        AperioJp2000Ycc = NVTIFF_COMPRESSION_APERIO_JP2000_YCC,
        /// Aperio JPEG 2000 with RGB.
        AperioJp2000Rgb = NVTIFF_COMPRESSION_APERIO_JP2000_RGB,
        /// JPEG 2000.
        Jp2000 = NVTIFF_COMPRESSION_JP2000,
    }
}

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

singe_core::native_enum! {
    /// Whether an image is stored in strips or tiles.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ImageGeometryType: sys::nvtiffImageGeometryType_t {
        /// Strips of rows.
        Striped = NVTIFF_IMAGE_STRIPED,
        /// Rectangular tiles.
        Tiled = NVTIFF_IMAGE_TILED,
    }
}

impl_enum_display!(ImageGeometryType, {
    Self::Striped => "NVTIFF_IMAGE_STRIPED",
    Self::Tiled => "NVTIFF_IMAGE_TILED",
});

singe_core::native_enum! {
    /// Decode output format.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum OutputFormat: sys::nvtiffOutputFormat_t {
        /// Samples as stored, interleaved.
        UnchangedInterleaved = NVTIFF_OUTPUT_UNCHANGED_I,
        /// Interleaved RGB with 8 bits per sample.
        RgbInterleavedU8 = NVTIFF_OUTPUT_RGB_I_UINT8,
        /// Interleaved RGB with 16 bits per sample.
        RgbInterleavedU16 = NVTIFF_OUTPUT_RGB_I_UINT16,
    }
}

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
    to_usize(u64::from(size) * u64::from(count), "value size")
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
