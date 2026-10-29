use std::{
    fmt::{self, Debug, Display, Formatter},
    mem::{align_of, offset_of, size_of},
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    types::{Complex, Complex32, Complex64, f16},
    view::{AnyBitPatternDeviceRepr, NoUninitDeviceRepr},
};
use singe_npp_sys as sys;

use crate::error::{Error, Result};

/// Data types for nppiPlus functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum DataType {
    /// 8-bit unsigned integer data type.
    U8 = sys::NppDataType::NPP_8U as _,
    /// 8-bit signed integer data type.
    I8 = sys::NppDataType::NPP_8S as _,
    /// 16-bit unsigned integer data type.
    U16 = sys::NppDataType::NPP_16U as _,
    /// 16-bit signed integer data type.
    I16 = sys::NppDataType::NPP_16S as _,
    /// 32-bit unsigned integer data type.
    U32 = sys::NppDataType::NPP_32U as _,
    /// 32-bit signed integer data type.
    I32 = sys::NppDataType::NPP_32S as _,
    /// 64-bit unsigned integer data type.
    U64 = sys::NppDataType::NPP_64U as _,
    /// 64-bit signed integer data type.
    I64 = sys::NppDataType::NPP_64S as _,
    /// 16-bit original CUDA floating point data type.
    F16 = sys::NppDataType::NPP_16F as _,
    /// 32-bit floating point data type.
    F32 = sys::NppDataType::NPP_32F as _,
    /// 64-bit double precision floating point data type.
    F64 = sys::NppDataType::NPP_64F as _,
}

impl_enum_conversion!(sys::NppDataType, DataType);

impl_enum_display!(DataType, {
    DataType::U8 => "NPP_8U",
    DataType::I8 => "NPP_8S",
    DataType::U16 => "NPP_16U",
    DataType::I16 => "NPP_16S",
    DataType::U32 => "NPP_32U",
    DataType::I32 => "NPP_32S",
    DataType::U64 => "NPP_64U",
    DataType::I64 => "NPP_64S",
    DataType::F16 => "NPP_16F",
    DataType::F32 => "NPP_32F",
    DataType::F64 => "NPP_64F",
});

/// A Rust scalar or complex representation matching an NPP datatype.
///
/// # Safety
///
/// Size, alignment, complexity, and the reported datatype must match the NPP
/// element ABI. All valid values must contain initialized bytes, and all
/// initialized NPP storage patterns must be valid Rust values.
pub unsafe trait DataTypeLike: AnyBitPatternDeviceRepr + NoUninitDeviceRepr + Debug {
    /// Returns the native datatype of this scalar or complex component.
    fn data_type() -> DataType;

    /// Returns whether each element contains real and imaginary components.
    fn is_complex() -> bool;

    /// Returns the Rust type spelling used by this dispatch implementation.
    fn rust_type_name() -> &'static str;
}

macro_rules! impl_data_type {
    ($ty:ty, $data_type:ident, $is_complex:expr) => {
        unsafe impl DataTypeLike for $ty {
            fn data_type() -> DataType {
                DataType::$data_type
            }

            fn is_complex() -> bool {
                $is_complex
            }

            fn rust_type_name() -> &'static str {
                stringify!($ty)
            }
        }
    };
}

/// Image channel counts for nppiPlus functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ImageChannels {
    /// Single channel per pixel data.
    C1 = sys::NppiChannels::NPP_CH_1 as _,
    /// Two channel per pixel data.
    C2 = sys::NppiChannels::NPP_CH_2 as _,
    /// Three channel per pixel data.
    C3 = sys::NppiChannels::NPP_CH_3 as _,
    /// Four channel per pixel data.
    C4 = sys::NppiChannels::NPP_CH_4 as _,
    /// Four channel per pixel data with alpha.
    AC4 = sys::NppiChannels::NPP_CH_A4 as _,
    /// Two channel single channel per plane pixel data.
    P2 = sys::NppiChannels::NPP_CH_P2 as _,
    /// Three channel single channel per plane pixel data.
    P3 = sys::NppiChannels::NPP_CH_P3 as _,
    /// Four channel single channel per plane pixel data.
    P4 = sys::NppiChannels::NPP_CH_P4 as _,
}

impl_enum_conversion!(sys::NppiChannels, ImageChannels);

impl_enum_display!(ImageChannels, {
    ImageChannels::C1 => "NPP_CH_1",
    ImageChannels::C2 => "NPP_CH_2",
    ImageChannels::C3 => "NPP_CH_3",
    ImageChannels::C4 => "NPP_CH_4",
    ImageChannels::AC4 => "NPP_CH_A4",
    ImageChannels::P2 => "NPP_CH_P2",
    ImageChannels::P3 => "NPP_CH_P3",
    ImageChannels::P4 => "NPP_CH_P4",
});

///  Color space enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ColorSpace {
    ///< no color space defined. used for simple transcoding.
    TranscodeOnly = sys::NppiColorSpace::NPP_TRANSCODE_ONLY as _,
    ///< BT601 colorspace used within JPEG images covering 0..255.
    Bt601Jpeg = sys::NppiColorSpace::NPP_BT601JPEG as _,
    ///< BT709 High Definition TV colorspace.
    Bt709Hdtv = sys::NppiColorSpace::NPP_BT709HDTV as _,
    ///< BT2020 High Definition TV colorspace.
    Bt2020Hdtv = sys::NppiColorSpace::NPP_BT2020HDTV as _,
}

impl_enum_conversion!(sys::NppiColorSpace, ColorSpace);

impl_enum_display!(ColorSpace, {
    ColorSpace::TranscodeOnly => "NPP_TRANSCODE_ONLY",
    ColorSpace::Bt601Jpeg => "NPP_BT601JPEG",
    ColorSpace::Bt709Hdtv => "NPP_BT709HDTV",
    ColorSpace::Bt2020Hdtv => "NPP_BT2020HDTV",
});

/// Complex image or signal samples with signed 16-bit real and imaginary components.
pub type ComplexI16 = Complex<i16>;

/// Complex image or signal samples with unsigned 32-bit real and imaginary components.
pub type ComplexU32 = Complex<u32>;

/// Complex image or signal samples with signed 32-bit real and imaginary components.
pub type ComplexI32 = Complex<i32>;

/// Complex image or signal samples with signed 64-bit real and imaginary components.
pub type ComplexI64 = Complex<i64>;

impl_data_type!(u8, U8, false);

impl_data_type!(i8, I8, false);

impl_data_type!(u16, U16, false);

impl_data_type!(i16, I16, false);

impl_data_type!(u32, U32, false);

impl_data_type!(i32, I32, false);

impl_data_type!(u64, U64, false);

impl_data_type!(i64, I64, false);

impl_data_type!(f16, F16, false);

impl_data_type!(f32, F32, false);

impl_data_type!(f64, F64, false);

impl_data_type!(ComplexI16, I16, true);

impl_data_type!(ComplexU32, U32, true);

impl_data_type!(ComplexI32, I32, true);

impl_data_type!(ComplexI64, I64, true);

impl_data_type!(Complex32, F32, true);

impl_data_type!(Complex64, F64, true);

/// Converts a Rust value to its corresponding native NPP argument representation.
///
/// The provided scalar implementations return the value unchanged; complex implementations
/// construct the corresponding native real and imaginary fields.
pub trait IntoNpp {
    /// Native argument type produced by this value conversion.
    type Npp;

    /// Consumes this value and constructs its corresponding native NPP argument.
    fn into_npp(self) -> Self::Npp;
}

macro_rules! impl_into_npp_identity {
    ($($ty:ty),* $(,)?) => {
        $(
            impl IntoNpp for $ty {
                type Npp = Self;

                fn into_npp(self) -> Self::Npp {
                    self
                }
            }
        )*
    };
}

impl_into_npp_identity!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64,);

impl IntoNpp for ComplexI16 {
    type Npp = sys::Npp16sc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp16sc {
            re: self.re,
            im: self.im,
        }
    }
}

impl IntoNpp for ComplexU32 {
    type Npp = sys::Npp32uc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp32uc {
            re: self.re,
            im: self.im,
        }
    }
}

impl IntoNpp for ComplexI32 {
    type Npp = sys::Npp32sc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp32sc {
            re: self.re,
            im: self.im,
        }
    }
}

impl IntoNpp for ComplexI64 {
    type Npp = sys::Npp64sc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp64sc {
            re: self.re,
            im: self.im,
        }
    }
}

impl IntoNpp for Complex32 {
    type Npp = sys::Npp32fc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp32fc {
            re: self.re,
            im: self.im,
        }
    }
}

impl IntoNpp for Complex64 {
    type Npp = sys::Npp64fc;

    fn into_npp(self) -> Self::Npp {
        sys::Npp64fc {
            re: self.re,
            im: self.im,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Native contour boundary direction flags for union-find labeled regions.
    ///
    /// Single directions describe neighboring contour pixels. `ANY_*` masks combine
    /// the cardinal direction with its two adjacent diagonals.
    pub struct ContourDirection: u8 {
        /// Contour direction south east.
        const SOUTH_EAST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_SOUTH_EAST);
        /// Contour direction south.
        const SOUTH = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_SOUTH);
        /// Contour direction south west.
        const SOUTH_WEST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_SOUTH_WEST);
        /// Contour direction west.
        const WEST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_WEST);
        /// Contour direction east.
        const EAST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_EAST);
        /// Contour direction north east.
        const NORTH_EAST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_NORTH_EAST);
        /// Contour direction north.
        const NORTH = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_NORTH);
        /// Contour direction north west.
        const NORTH_WEST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_NORTH_WEST);
        /// North, northeast, or northwest contour direction.
        const ANY_NORTH = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_ANY_NORTH);
        /// West, northwest, or southwest contour direction.
        const ANY_WEST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_ANY_WEST);
        /// South, southeast, or southwest contour direction.
        const ANY_SOUTH = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_ANY_SOUTH);
        /// East, northeast, or southeast contour direction.
        const ANY_EAST = contour_direction_bits(sys::NPP_CONTOUR_DIRECTION_ANY_EAST);
    }
}

impl Display for ContourDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

const _: () = {
    // Radial profiles write directly through a native NppiProfileData pointer.
    assert!(size_of::<ProfileStatistics>() == size_of::<sys::NppiProfileData>());
    assert!(align_of::<ProfileStatistics>() == align_of::<sys::NppiProfileData>());
    assert!(offset_of!(ProfileStatistics, pixels) == offset_of!(sys::NppiProfileData, nPixels));
    assert!(
        offset_of!(ProfileStatistics, mean_intensity)
            == offset_of!(sys::NppiProfileData, nMeanIntensity)
    );
    assert!(
        offset_of!(ProfileStatistics, standard_deviation_intensity)
            == offset_of!(sys::NppiProfileData, nStdDevIntensity)
    );

    // Flood fill writes directly into ConnectedRegion through a native pointer.
    assert!(size_of::<ConnectedRegion>() == size_of::<sys::NppiConnectedRegion>());
    assert!(align_of::<ConnectedRegion>() == align_of::<sys::NppiConnectedRegion>());
    assert!(
        offset_of!(ConnectedRegion, bounding_box)
            == offset_of!(sys::NppiConnectedRegion, oBoundingBox)
    );
    assert!(
        offset_of!(ConnectedRegion, connected_pixel_count)
            == offset_of!(sys::NppiConnectedRegion, nConnectedPixelCount)
    );
    assert!(
        offset_of!(ConnectedRegion, seed_pixel_value)
            == offset_of!(sys::NppiConnectedRegion, aSeedPixelValue)
    );

    ["Npp16f size"][size_of::<sys::Npp16f>() - size_of::<f16>()];
    ["Npp16f alignment"][align_of::<sys::Npp16f>() - align_of::<f16>()];

    ["Npp16sc size"][size_of::<sys::Npp16sc>() - size_of::<ComplexI16>()];
    ["Npp16sc alignment"][align_of::<sys::Npp16sc>() - align_of::<ComplexI16>()];
    ["Npp16sc re field"][offset_of!(ComplexI16, re) - offset_of!(sys::Npp16sc, re)];
    ["Npp16sc im field"][offset_of!(ComplexI16, im) - offset_of!(sys::Npp16sc, im)];

    ["Npp32uc size"][size_of::<sys::Npp32uc>() - size_of::<ComplexU32>()];
    ["Npp32uc alignment"][align_of::<sys::Npp32uc>() - align_of::<ComplexU32>()];
    ["Npp32uc re field"][offset_of!(ComplexU32, re) - offset_of!(sys::Npp32uc, re)];
    ["Npp32uc im field"][offset_of!(ComplexU32, im) - offset_of!(sys::Npp32uc, im)];

    ["Npp32sc size"][size_of::<sys::Npp32sc>() - size_of::<ComplexI32>()];
    ["Npp32sc alignment"][align_of::<sys::Npp32sc>() - align_of::<ComplexI32>()];
    ["Npp32sc re field"][offset_of!(ComplexI32, re) - offset_of!(sys::Npp32sc, re)];
    ["Npp32sc im field"][offset_of!(ComplexI32, im) - offset_of!(sys::Npp32sc, im)];

    ["Npp64sc size"][size_of::<sys::Npp64sc>() - size_of::<ComplexI64>()];
    ["Npp64sc alignment"][align_of::<sys::Npp64sc>() - align_of::<ComplexI64>()];
    ["Npp64sc re field"][offset_of!(ComplexI64, re) - offset_of!(sys::Npp64sc, re)];
    ["Npp64sc im field"][offset_of!(ComplexI64, im) - offset_of!(sys::Npp64sc, im)];

    ["Npp32fc size"][size_of::<sys::Npp32fc>() - size_of::<Complex32>()];
    ["Npp32fc alignment"][align_of::<sys::Npp32fc>() - align_of::<Complex32>()];
    ["Npp32fc re field"][offset_of!(Complex32, re) - offset_of!(sys::Npp32fc, re)];
    ["Npp32fc im field"][offset_of!(Complex32, im) - offset_of!(sys::Npp32fc, im)];

    ["Npp64fc size"][size_of::<sys::Npp64fc>() - size_of::<Complex64>()];
    ["Npp64fc alignment"][align_of::<sys::Npp64fc>() - align_of::<Complex64>()];
    ["Npp64fc re field"][offset_of!(Complex64, re) - offset_of!(sys::Npp64fc, re)];
    ["Npp64fc im field"][offset_of!(Complex64, im) - offset_of!(sys::Npp64fc, im)];
};

/// Width and height in the native NPP two-dimensional size format.
///
/// Raw construction and native conversion preserve nonpositive dimensions; [`Self::create`] and [`Self::validate`] require positive values.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Hash,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct Size {
    /// Horizontal extent.
    pub width: i32,
    /// Vertical extent.
    pub height: i32,
}

impl Size {
    /// Creates a size with positive width and height.
    ///
    /// # Panics
    ///
    /// Panics if either dimension is zero or negative.
    #[must_use]
    pub const fn new(width: i32, height: i32) -> Self {
        assert!(width > 0, "width must be greater than 0");
        assert!(height > 0, "height must be greater than 0");
        Self { width, height }
    }

    /// Stores raw width and height values, preserving zero and negative native dimensions.
    #[must_use]
    pub const fn from_raw_parts(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    /// Creates a size after validating positive width and height.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is zero or negative.
    pub fn create(width: i32, height: i32) -> Result<Self> {
        let size = Self::from_raw_parts(width, height);
        size.validate()?;
        Ok(size)
    }

    /// Returns whether both width and height are greater than zero.
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Checks that both dimensions are positive.
    ///
    /// # Errors
    ///
    /// Returns an error identifying the first nonpositive dimension.
    pub fn validate(self) -> Result<()> {
        if self.width <= 0 {
            return Err(Error::ValueNotPositive {
                name: "width".into(),
            });
        }
        if self.height <= 0 {
            return Err(Error::ValueNotPositive {
                name: "height".into(),
            });
        }
        Ok(())
    }

    /// Returns an origin-based rectangle with the stored width and height.
    #[must_use]
    pub fn rectangle(self) -> Rectangle {
        Rectangle {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }

    /// Returns the largest centered square using the smaller stored dimension as its side length.
    ///
    /// For positive dimensions, odd excess pixels remain on the right or bottom side.
    ///
    /// # Panics
    ///
    /// When integer overflow checks are enabled, panics if subtraction of unvalidated raw dimensions overflows `i32`.
    #[must_use]
    pub fn centered_rectangle(self) -> Rectangle {
        let side = self.width.min(self.height);
        Rectangle {
            x: ((self.width - side) / 2),
            y: ((self.height - side) / 2),
            width: side,
            height: side,
        }
    }
}

impl From<Size> for sys::NppiSize {
    fn from(value: Size) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl From<sys::NppiSize> for Size {
    fn from(value: sys::NppiSize) -> Self {
        Self::from_raw_parts(value.width, value.height)
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Position and dimensions in the native NPP rectangle format.
///
/// Geometry operations treat `width` and `height` as extents from the upper-left origin.
/// Connected-region output uses the endpoint convention documented on [`ConnectedRegion`](crate::types::ConnectedRegion).
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Hash,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
pub struct Rectangle {
    /// Horizontal coordinate of the upper-left origin.
    pub x: i32,
    /// Vertical coordinate of the upper-left origin.
    pub y: i32,
    /// Horizontal extent from the origin.
    pub width: i32,
    /// Vertical extent from the origin.
    pub height: i32,
}

impl Rectangle {
    /// Returns the stored width and height as a raw size.
    #[must_use]
    pub fn size(self) -> Size {
        Size::from_raw_parts(self.width, self.height)
    }

    /// Checks positive dimensions and representable endpoints, allowing negative source origins.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is nonpositive or computing the right or bottom endpoint overflows `i32`.
    pub fn validate(self) -> Result<()> {
        self.endpoints().map(|_| ())
    }

    /// Checks that the entire rectangle lies within the supplied image size.
    ///
    /// # Errors
    ///
    /// Returns an error if the image size or rectangle dimensions are nonpositive, an endpoint calculation overflows, the origin is negative, or an endpoint exceeds the image dimensions.
    pub fn validate_within(self, size: Size) -> Result<()> {
        size.validate()?;
        let (right, bottom) = self.endpoints()?;
        if self.x < 0 || self.y < 0 || right > size.width || bottom > size.height {
            return Err(Error::out_of_range("image roi"));
        }
        Ok(())
    }

    fn endpoints(self) -> Result<(i32, i32)> {
        self.size().validate()?;
        self.x
            .checked_add(self.width)
            .zip(self.y.checked_add(self.height))
            .ok_or_else(|| Error::out_of_range("image roi endpoints"))
    }
}

impl From<Rectangle> for sys::NppiRect {
    fn from(value: Rectangle) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<sys::NppiRect> for Rectangle {
    fn from(value: sys::NppiRect) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

/// A two-dimensional point with signed integer coordinates.
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Hash,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: i32,
    /// Vertical coordinate.
    pub y: i32,
}

impl Point {
    /// The point at coordinates `(0, 0)`.
    pub const ZERO: Self = Self::new(0, 0);

    /// Creates a point from the supplied signed coordinates.
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl From<Point> for sys::NppiPoint {
    fn from(value: Point) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint> for Point {
    fn from(value: sys::NppiPoint) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

/// A two-dimensional point with single-precision coordinates.
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    PartialEq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
pub struct Point32f {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl From<Point32f> for sys::NppiPoint32f {
    fn from(value: Point32f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint32f> for Point32f {
    fn from(value: sys::NppiPoint32f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

/// A two-dimensional point with double-precision coordinates.
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    PartialEq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
pub struct Point64f {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

impl From<Point64f> for sys::NppiPoint64f {
    fn from(value: Point64f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<sys::NppiPoint64f> for Point64f {
    fn from(value: sys::NppiPoint64f) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<[f64; 2]> for Point64f {
    fn from(value: [f64; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Point64f> for [f64; 2] {
    fn from(value: Point64f) -> Self {
        [value.x, value.y]
    }
}

impl Point64f {
    /// Returns the coordinates in `[x, y]` order.
    #[must_use]
    pub const fn to_array(self) -> [f64; 2] {
        [self.x, self.y]
    }
}

/// Alias for a double-precision [`Point64f`].
pub type PointF64 = Point64f;

/// A two-dimensional polar point with single-precision coordinates.
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    PartialEq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
pub struct PointPolar {
    /// Radial coordinate.
    pub rho: f32,
    /// Angular coordinate.
    pub theta: f32,
}

impl From<PointPolar> for sys::NppPointPolar {
    fn from(value: PointPolar) -> Self {
        Self {
            rho: value.rho,
            theta: value.theta,
        }
    }
}

impl From<sys::NppPointPolar> for PointPolar {
    fn from(value: sys::NppPointPolar) -> Self {
        Self {
            rho: value.rho,
            theta: value.theta,
        }
    }
}

/// Two double-precision coordinate pairs describing lower and upper bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BoundF64 {
    /// Lower coordinate pair.
    pub lower: PointF64,
    /// Upper coordinate pair.
    pub upper: PointF64,
}

impl From<[[f64; 2]; 2]> for BoundF64 {
    fn from(value: [[f64; 2]; 2]) -> Self {
        Self {
            lower: value[0].into(),
            upper: value[1].into(),
        }
    }
}

impl From<BoundF64> for [[f64; 2]; 2] {
    fn from(value: BoundF64) -> Self {
        [value.lower.into(), value.upper.into()]
    }
}

impl BoundF64 {
    /// Returns a read-only pointer to two coordinate pairs in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    #[must_use]
    pub const fn as_ptr(&self) -> *const [f64; 2] {
        (&raw const self.lower).cast()
    }

    /// Returns a writable pointer to two coordinate pairs in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    pub const fn as_mut_ptr(&mut self) -> *mut [f64; 2] {
        (&raw mut self.lower).cast()
    }
}

/// Four ordered vertices with double-precision coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct QuadrangleF64 {
    /// Four vertices in the order required by the consuming geometry operation.
    pub points: [PointF64; 4],
}

impl From<[[f64; 2]; 4]> for QuadrangleF64 {
    fn from(value: [[f64; 2]; 4]) -> Self {
        Self {
            points: value.map(PointF64::from),
        }
    }
}

impl From<QuadrangleF64> for [[f64; 2]; 4] {
    fn from(value: QuadrangleF64) -> Self {
        value.points.map(Into::into)
    }
}

impl QuadrangleF64 {
    /// Returns a read-only pointer to four coordinate pairs in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    #[must_use]
    pub const fn as_ptr(&self) -> *const [f64; 2] {
        self.points.as_ptr().cast()
    }

    /// Returns a writable pointer to four coordinate pairs in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    pub fn as_mut_ptr(&mut self) -> *mut [f64; 2] {
        self.points.as_mut_ptr().cast()
    }
}

/// A two-row, three-column double-precision affine transformation matrix.
///
/// The default matrix is zero.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(transparent)]
pub struct AffineCoefficients {
    /// Two rows of three affine coefficients.
    pub values: [[f64; 3]; 2],
}

impl From<[[f64; 3]; 2]> for AffineCoefficients {
    fn from(value: [[f64; 3]; 2]) -> Self {
        Self { values: value }
    }
}

impl From<AffineCoefficients> for [[f64; 3]; 2] {
    fn from(value: AffineCoefficients) -> Self {
        value.values
    }
}

impl AffineCoefficients {
    /// Returns a read-only pointer to two rows of three coefficients in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    #[must_use]
    pub const fn as_ptr(&self) -> *const [f64; 3] {
        self.values.as_ptr()
    }

    /// Returns a writable pointer to two rows of three coefficients in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    pub fn as_mut_ptr(&mut self) -> *mut [f64; 3] {
        self.values.as_mut_ptr()
    }
}

/// A three-row, three-column double-precision perspective transformation matrix.
///
/// The default matrix is zero.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(transparent)]
pub struct PerspectiveCoefficients {
    /// Three rows of three perspective coefficients.
    pub values: [[f64; 3]; 3],
}

impl From<[[f64; 3]; 3]> for PerspectiveCoefficients {
    fn from(value: [[f64; 3]; 3]) -> Self {
        Self { values: value }
    }
}

impl From<PerspectiveCoefficients> for [[f64; 3]; 3] {
    fn from(value: PerspectiveCoefficients) -> Self {
        value.values
    }
}

impl PerspectiveCoefficients {
    /// Returns a read-only pointer to three rows of three coefficients in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    #[must_use]
    pub const fn as_ptr(&self) -> *const [f64; 3] {
        self.values.as_ptr()
    }

    /// Returns a writable pointer to three rows of three coefficients in native array layout.
    ///
    /// The pointer borrows this value; retain its storage and required access permissions through native use.
    pub fn as_mut_ptr(&mut self) -> *mut [f64; 3] {
        self.values.as_mut_ptr()
    }
}

/// Configuration for a histogram-of-oriented-gradients descriptor.
///
/// Cell, block, and detection-window sizes are measured in pixels.
/// The default sets every size and count to zero; it does not select a usable HOG configuration.
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct HistogramOfGradientsConfig {
    /// Side length of a square cell, in pixels.
    pub cell_size: i32,
    /// Side length of a square histogram block, in pixels.
    pub histogram_block_size: i32,
    /// Required number of histogram bins.
    pub histogram_bins: i32,
    /// Detection-window width and height, in pixels.
    pub detection_window_size: Size,
}

impl Default for HistogramOfGradientsConfig {
    fn default() -> Self {
        Self {
            cell_size: 0,
            histogram_block_size: 0,
            histogram_bins: 0,
            detection_window_size: Size::from_raw_parts(0, 0),
        }
    }
}

impl From<HistogramOfGradientsConfig> for sys::NppiHOGConfig {
    fn from(value: HistogramOfGradientsConfig) -> Self {
        Self {
            cellSize: value.cell_size,
            histogramBlockSize: value.histogram_block_size,
            nHistogramBins: value.histogram_bins,
            detectionWindowSize: value.detection_window_size.into(),
        }
    }
}

impl From<sys::NppiHOGConfig> for HistogramOfGradientsConfig {
    fn from(value: sys::NppiHOGConfig) -> Self {
        Self {
            cell_size: value.cellSize,
            histogram_block_size: value.histogramBlockSize,
            histogram_bins: value.nHistogramBins,
            detection_window_size: value.detectionWindowSize.into(),
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP Haar classifier descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct HaarClassifier32f {
    /// Number of classifiers.
    pub classifier_count: i32,
    /// Device pointer to packed classifier records, each 40 bytes long.
    pub classifiers: *mut i32,
    /// Byte stride between packed classifier records.
    pub classifier_step: usize,
    /// Dimensions of a packed classifier.
    pub classifier_size: Size,
    /// Device pointer to the classifier counter.
    pub device_counter: *mut i32,
}

impl From<HaarClassifier32f> for sys::NppiHaarClassifier_32f {
    fn from(value: HaarClassifier32f) -> Self {
        Self {
            numClassifiers: value.classifier_count,
            classifiers: value.classifiers,
            classifierStep: value.classifier_step as _,
            classifierSize: value.classifier_size.into(),
            counterDevice: value.device_counter,
        }
    }
}

impl From<sys::NppiHaarClassifier_32f> for HaarClassifier32f {
    fn from(value: sys::NppiHaarClassifier_32f) -> Self {
        Self {
            classifier_count: value.numClassifiers,
            classifiers: value.classifiers,
            classifier_step: usize::try_from(value.classifierStep)
                .expect("classifier step fits the host address space"),
            classifier_size: value.classifierSize.into(),
            device_counter: value.counterDevice,
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP Haar buffer descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct HaarBuffer {
    /// Buffer size passed to the native Haar operation.
    pub size: i32,
    /// Device pointer to the Haar working buffer.
    pub buffer: *mut i32,
}

impl From<HaarBuffer> for sys::NppiHaarBuffer {
    fn from(value: HaarBuffer) -> Self {
        Self {
            haarBufferSize: value.size,
            haarBuffer: value.buffer,
        }
    }
}

impl From<sys::NppiHaarBuffer> for HaarBuffer {
    fn from(value: sys::NppiHaarBuffer) -> Self {
        Self {
            size: value.haarBufferSize,
            buffer: value.haarBuffer,
        }
    }
}

/// Connected pixel region information returned by flood filling.
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
#[repr(C)]
pub struct ConnectedRegion {
    /// Native bounding box stored as left, top, right, and bottom pixel coordinates in `x`, `y`, `width`, and `height`, respectively.
    /// `width` and `height` hold endpoint coordinates here, rather than extents.
    pub bounding_box: Rectangle,
    /// Total number of pixels in connected region.
    pub connected_pixel_count: u32,
    /// Original seed-pixel value for one or three channels.
    pub seed_pixel_value: [u32; 3],
}

impl From<ConnectedRegion> for sys::NppiConnectedRegion {
    fn from(value: ConnectedRegion) -> Self {
        Self {
            oBoundingBox: value.bounding_box.into(),
            nConnectedPixelCount: value.connected_pixel_count,
            aSeedPixelValue: value.seed_pixel_value,
        }
    }
}

impl From<sys::NppiConnectedRegion> for ConnectedRegion {
    fn from(value: sys::NppiConnectedRegion) -> Self {
        Self {
            bounding_box: value.oBoundingBox.into(),
            connected_pixel_count: value.nConnectedPixelCount,
            seed_pixel_value: value.aSeedPixelValue,
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP image descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct ImageDescriptor {
    /// Device pointer to the image data.
    pub data: *mut (),
    /// Byte stride between image rows.
    pub step: i32,
    /// Image width and height, in pixels.
    pub size: Size,
}

impl From<ImageDescriptor> for sys::NppiImageDescriptor {
    fn from(value: ImageDescriptor) -> Self {
        Self {
            pData: value.data.cast(),
            nStep: value.step,
            oSize: value.size.into(),
        }
    }
}

impl From<sys::NppiImageDescriptor> for ImageDescriptor {
    fn from(value: sys::NppiImageDescriptor) -> Self {
        Self {
            data: value.pData.cast(),
            step: value.nStep,
            size: value.oSize.into(),
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP scratch/data buffer descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct BufferDescriptor {
    /// Device pointer to the corresponding per-image buffer.
    pub data: *mut (),
    /// Allocated buffer size in bytes.
    pub size: i32,
}

impl From<BufferDescriptor> for sys::NppiBufferDescriptor {
    fn from(value: BufferDescriptor) -> Self {
        Self {
            pData: value.data.cast(),
            nBufferSize: value.size,
        }
    }
}

impl From<sys::NppiBufferDescriptor> for BufferDescriptor {
    fn from(value: sys::NppiBufferDescriptor) -> Self {
        Self {
            data: value.pData.cast(),
            size: value.nBufferSize,
        }
    }
}

/// Pixel counts, contour location, and bounds of a uniquely labeled connected region.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct CompressedMarkerLabelsInfo {
    /// Total number of pixels in this connected pixel region.
    pub marker_label_pixel_count: u32,
    /// Total number of pixels in this connected pixel region contour.
    pub contour_pixel_count: u32,
    /// Total number of pixels in this connected pixel region contour found during geometry search.
    pub contour_pixels_found: u32,
    /// Image geometric x and y location of the first pixel in the contour.
    pub contour_first_pixel_location: Point,
    /// Native region bounds stored as left, top, right, and bottom pixel coordinates in `x`, `y`, `width`, and `height`, respectively.
    /// `width` and `height` hold endpoint coordinates here, rather than extents.
    pub marker_label_bounding_box: Rectangle,
}

impl From<CompressedMarkerLabelsInfo> for sys::NppiCompressedMarkerLabelsInfo {
    fn from(value: CompressedMarkerLabelsInfo) -> Self {
        Self {
            nMarkerLabelPixelCount: value.marker_label_pixel_count,
            nContourPixelCount: value.contour_pixel_count,
            nContourPixelsFound: value.contour_pixels_found,
            oContourFirstPixelLocation: value.contour_first_pixel_location.into(),
            oMarkerLabelBoundingBox: value.marker_label_bounding_box.into(),
        }
    }
}

impl From<sys::NppiCompressedMarkerLabelsInfo> for CompressedMarkerLabelsInfo {
    fn from(value: sys::NppiCompressedMarkerLabelsInfo) -> Self {
        Self {
            marker_label_pixel_count: value.nMarkerLabelPixelCount,
            contour_pixel_count: value.nContourPixelCount,
            contour_pixels_found: value.nContourPixelsFound,
            contour_first_pixel_location: value.oContourFirstPixelLocation.into(),
            marker_label_bounding_box: value.oMarkerLabelBoundingBox.into(),
        }
    }
}

/// Location and region association of a contour segment in the contour pixel list.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct ContourBlockSegment {
    /// This connected pixel region contour ID.
    pub marker_label_id: u32,
    /// Total number of pixels in this connected pixel region contour.
    pub contour_pixel_count: u32,
    /// Base offset of starting pixel in the overall contour pixel list.
    pub contour_starting_pixel_offset: u32,
    /// Relative block segment number within this particular contour ID.
    pub segment_number: u32,
}

impl From<ContourBlockSegment> for sys::NppiContourBlockSegment {
    fn from(value: ContourBlockSegment) -> Self {
        Self {
            nMarkerLabelID: value.marker_label_id,
            nContourPixelCount: value.contour_pixel_count,
            nContourStartingPixelOffset: value.contour_starting_pixel_offset,
            nSegmentNum: value.segment_number,
        }
    }
}

impl From<sys::NppiContourBlockSegment> for ContourBlockSegment {
    fn from(value: sys::NppiContourBlockSegment) -> Self {
        Self {
            marker_label_id: value.nMarkerLabelID,
            contour_pixel_count: value.nContourPixelCount,
            contour_starting_pixel_offset: value.nContourStartingPixelOffset,
            segment_number: value.nSegmentNum,
        }
    }
}

/// Geometry and traversal indices for a connected region's contour.
///
/// NPP orders the contour counterclockwise relative to the region interior.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct ContourPixelGeometryInfo {
    /// Image geometry X and Y location in requested output order.
    pub contour_ordered_geometry_location: Point,
    /// Image geometry X and Y location of previous contour pixel.
    pub previous_contour_pixel_location: Point,
    /// Image geometry X and Y location of center contour pixel.
    pub contour_center_pixel_location: Point,
    /// Image geometry X and Y location of next contour pixel.
    pub next_contour_pixel_location: Point,
    /// Contour pixel counterclockwise order index in geometry list.
    pub order_index: i32,
    /// Contour pixel clockwise order index in geometry list.
    pub reverse_order_index: i32,
    /// Index of first ordered contour pixel in this subgroup.
    pub first_index: u32,
    /// Index of last ordered contour pixel in this subgroup.
    pub last_index: u32,
    /// Index of next ordered contour pixel in [`ContourPixelGeometryInfo`] list.
    pub next_contour_pixel_index: u32,
    /// Index of previous ordered contour pixel in [`ContourPixelGeometryInfo`] list.
    pub previous_contour_pixel_index: u32,
    /// This test pixel has already been used.
    pub pixel_already_used: u8,
    /// This test pixel is already linked to center pixel.
    pub already_linked: u8,
    /// This pixel has already been output in geometry list.
    pub already_output: u8,
    /// Direction of contour region interior.
    pub contour_interior_direction: u8,
}

impl From<ContourPixelGeometryInfo> for sys::NppiContourPixelGeometryInfo {
    fn from(value: ContourPixelGeometryInfo) -> Self {
        Self {
            oContourOrderedGeometryLocation: value.contour_ordered_geometry_location.into(),
            oContourPrevPixelLocation: value.previous_contour_pixel_location.into(),
            oContourCenterPixelLocation: value.contour_center_pixel_location.into(),
            oContourNextPixelLocation: value.next_contour_pixel_location.into(),
            nOrderIndex: value.order_index,
            nReverseOrderIndex: value.reverse_order_index,
            nFirstIndex: value.first_index,
            nLastIndex: value.last_index,
            nNextContourPixelIndex: value.next_contour_pixel_index,
            nPrevContourPixelIndex: value.previous_contour_pixel_index,
            nPixelAlreadyUsed: value.pixel_already_used,
            nAlreadyLinked: value.already_linked,
            nAlreadyOutput: value.already_output,
            nContourInteriorDirection: value.contour_interior_direction,
        }
    }
}

impl From<sys::NppiContourPixelGeometryInfo> for ContourPixelGeometryInfo {
    fn from(value: sys::NppiContourPixelGeometryInfo) -> Self {
        Self {
            contour_ordered_geometry_location: value.oContourOrderedGeometryLocation.into(),
            previous_contour_pixel_location: value.oContourPrevPixelLocation.into(),
            contour_center_pixel_location: value.oContourCenterPixelLocation.into(),
            next_contour_pixel_location: value.oContourNextPixelLocation.into(),
            order_index: value.nOrderIndex,
            reverse_order_index: value.nReverseOrderIndex,
            first_index: value.nFirstIndex,
            last_index: value.nLastIndex,
            next_contour_pixel_index: value.nNextContourPixelIndex,
            previous_contour_pixel_index: value.nPrevContourPixelIndex,
            pixel_already_used: value.nPixelAlreadyUsed,
            already_linked: value.nAlreadyLinked,
            already_output: value.nAlreadyOutput,
            contour_interior_direction: value.nContourInteriorDirection,
        }
    }
}

/// Neighbor coordinates, connectivity flags, and geometry of a contour pixel.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct ContourPixelDirectionInfo {
    /// Marker label identifying the connected region inside the contour.
    pub marker_label_id: u32,
    /// Input and output direction information for the current center contour pixel.
    pub contour_direction_center_pixel: u8,
    /// Direction information for the region interior at the center contour pixel.
    pub contour_interior_direction_center_pixel: u8,
    /// Direction to directly connected contour pixels, 0 if not connected.
    pub connected: u8,
    /// Native geometry-information validity flag.
    pub geometry_info_is_valid: u8,
    /// Contour-pixel coordinates and traversal indices.
    pub contour_pixel_geometry_info: ContourPixelGeometryInfo,
    /// Coordinates of the east neighbor pixel.
    pub east: Point,
    /// Coordinates of the northeast neighbor pixel.
    pub north_east: Point,
    /// Coordinates of the north neighbor pixel.
    pub north: Point,
    /// Coordinates of the northwest neighbor pixel.
    pub north_west: Point,
    /// Coordinates of the west neighbor pixel.
    pub west: Point,
    /// Coordinates of the southwest neighbor pixel.
    pub south_west: Point,
    /// Coordinates of the south neighbor pixel.
    pub south: Point,
    /// Coordinates of the southeast neighbor pixel.
    pub south_east: Point,
    /// East connected flag.
    pub east_connected: u8,
    /// North east connected flag.
    pub north_east_connected: u8,
    /// North connected flag.
    pub north_connected: u8,
    /// North west connected flag.
    pub north_west_connected: u8,
    /// West connected flag.
    pub west_connected: u8,
    /// South west connected flag.
    pub south_west_connected: u8,
    /// South connected flag.
    pub south_connected: u8,
    /// South east connected flag.
    pub south_east_connected: u8,
}

impl From<ContourPixelDirectionInfo> for sys::NppiContourPixelDirectionInfo {
    fn from(value: ContourPixelDirectionInfo) -> Self {
        Self {
            nMarkerLabelID: value.marker_label_id,
            nContourDirectionCenterPixel: value.contour_direction_center_pixel,
            nContourInteriorDirectionCenterPixel: value.contour_interior_direction_center_pixel,
            nConnected: value.connected,
            nGeometryInfoIsValid: value.geometry_info_is_valid,
            oContourPixelGeometryInfo: value.contour_pixel_geometry_info.into(),
            nEast1: value.east.into(),
            nNorthEast1: value.north_east.into(),
            nNorth1: value.north.into(),
            nNorthWest1: value.north_west.into(),
            nWest1: value.west.into(),
            nSouthWest1: value.south_west.into(),
            nSouth1: value.south.into(),
            nSouthEast1: value.south_east.into(),
            nTest1EastConnected: value.east_connected,
            nTest1NorthEastConnected: value.north_east_connected,
            nTest1NorthConnected: value.north_connected,
            nTest1NorthWestConnected: value.north_west_connected,
            nTest1WestConnected: value.west_connected,
            nTest1SouthWestConnected: value.south_west_connected,
            nTest1SouthConnected: value.south_connected,
            nTest1SouthEastConnected: value.south_east_connected,
        }
    }
}

impl From<sys::NppiContourPixelDirectionInfo> for ContourPixelDirectionInfo {
    fn from(value: sys::NppiContourPixelDirectionInfo) -> Self {
        Self {
            marker_label_id: value.nMarkerLabelID,
            contour_direction_center_pixel: value.nContourDirectionCenterPixel,
            contour_interior_direction_center_pixel: value.nContourInteriorDirectionCenterPixel,
            connected: value.nConnected,
            geometry_info_is_valid: value.nGeometryInfoIsValid,
            contour_pixel_geometry_info: value.oContourPixelGeometryInfo.into(),
            east: value.nEast1.into(),
            north_east: value.nNorthEast1.into(),
            north: value.nNorth1.into(),
            north_west: value.nNorthWest1.into(),
            west: value.nWest1.into(),
            south_west: value.nSouthWest1.into(),
            south: value.nSouth1.into(),
            south_east: value.nSouthEast1.into(),
            east_connected: value.nTest1EastConnected,
            north_east_connected: value.nTest1NorthEastConnected,
            north_connected: value.nTest1NorthConnected,
            north_west_connected: value.nTest1NorthWestConnected,
            west_connected: value.nTest1WestConnected,
            south_west_connected: value.nTest1SouthWestConnected,
            south_connected: value.nTest1SouthConnected,
            south_east_connected: value.nTest1SouthEastConnected,
        }
    }
}

/// Total contour pixel count and the longest contour length in an image.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Default,
    singe_cuda::DeviceRepr,
    singe_cuda::NoUninitDeviceRepr,
    singe_cuda::ZeroableDeviceRepr,
    singe_cuda::AnyBitPatternDeviceRepr,
)]
#[repr(C)]
pub struct ContourTotalsInfo {
    /// Total number of contour pixels in image.
    pub total_image_pixel_contour_count: u32,
    /// Longest per contour pixel count in image.
    pub longest_image_contour_pixel_count: u32,
}

impl From<ContourTotalsInfo> for sys::NppiContourTotalsInfo {
    fn from(value: ContourTotalsInfo) -> Self {
        Self {
            nTotalImagePixelContourCount: value.total_image_pixel_contour_count,
            nLongestImageContourPixelCount: value.longest_image_contour_pixel_count,
        }
    }
}

impl From<sys::NppiContourTotalsInfo> for ContourTotalsInfo {
    fn from(value: sys::NppiContourTotalsInfo) -> Self {
        Self {
            total_image_pixel_contour_count: value.nTotalImagePixelContourCount,
            longest_image_contour_pixel_count: value.nLongestImageContourPixelCount,
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP resize batch descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct ResizeBatchDescriptor {
    /// Device pointer to the source image.
    pub source: *const (),
    /// Byte stride between source-image rows.
    pub source_step: i32,
    /// Device pointer to the destination image.
    pub destination: *mut (),
    /// Byte stride between destination-image rows.
    pub destination_step: i32,
}

impl From<ResizeBatchDescriptor> for sys::NppiResizeBatchCXR {
    fn from(value: ResizeBatchDescriptor) -> Self {
        Self {
            pSrc: value.source.cast(),
            nSrcStep: value.source_step,
            pDst: value.destination.cast(),
            nDstStep: value.destination_step,
        }
    }
}

impl From<sys::NppiResizeBatchCXR> for ResizeBatchDescriptor {
    fn from(value: sys::NppiResizeBatchCXR) -> Self {
        Self {
            source: value.pSrc.cast(),
            source_step: value.nSrcStep,
            destination: value.pDst.cast(),
            destination_step: value.nDstStep,
        }
    }
}

/// Source and destination rectangles for one image in a variable-ROI resize batch.
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ResizeBatchRoiAdvanced {
    /// Per source image rectangle parameters.
    pub source_roi: Rectangle,
    /// Per destination image rectangle parameters.
    pub destination_roi: Rectangle,
}

impl From<ResizeBatchRoiAdvanced> for sys::NppiResizeBatchROI_Advanced {
    fn from(value: ResizeBatchRoiAdvanced) -> Self {
        Self {
            oSrcRectROI: value.source_roi.into(),
            oDstRectROI: value.destination_roi.into(),
        }
    }
}

impl From<sys::NppiResizeBatchROI_Advanced> for ResizeBatchRoiAdvanced {
    fn from(value: sys::NppiResizeBatchROI_Advanced) -> Self {
        Self {
            source_roi: value.oSrcRectROI.into(),
            destination_roi: value.oDstRectROI.into(),
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP mirror batch descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct MirrorBatchDescriptor {
    /// Device pointer to the source image.
    pub source: *const (),
    /// Byte stride between source-image rows.
    pub source_step: i32,
    /// Device pointer to the destination image.
    pub destination: *mut (),
    /// Byte stride between destination-image rows.
    pub destination_step: i32,
}

impl From<MirrorBatchDescriptor> for sys::NppiMirrorBatchCXR {
    fn from(value: MirrorBatchDescriptor) -> Self {
        Self {
            pSrc: value.source.cast(),
            nSrcStep: value.source_step,
            pDst: value.destination.cast(),
            nDstStep: value.destination_step,
        }
    }
}

impl From<sys::NppiMirrorBatchCXR> for MirrorBatchDescriptor {
    fn from(value: sys::NppiMirrorBatchCXR) -> Self {
        Self {
            source: value.pSrc.cast(),
            source_step: value.nSrcStep,
            destination: value.pDst.cast(),
            destination_step: value.nDstStep,
        }
    }
}

#[derive(Debug, PartialOrd, PartialEq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP affine-warp batch descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct WarpAffineBatchDescriptor {
    /// Device pointer to the source image.
    pub source: *const (),
    /// Byte stride between source-image rows.
    pub source_step: i32,
    /// Device pointer to the destination image.
    pub destination: *mut (),
    /// Byte stride between destination-image rows.
    pub destination_step: i32,
    /// Device pointer to the double-precision transformation matrix for this image.
    pub coefficients: *mut f64,
    /// Reserved for NPP's transformed coefficients; `coefficients` supplies the input transformation.
    pub transformed_coefficients: [[f64; 3]; 2],
}

impl From<WarpAffineBatchDescriptor> for sys::NppiWarpAffineBatchCXR {
    fn from(value: WarpAffineBatchDescriptor) -> Self {
        Self {
            pSrc: value.source.cast(),
            nSrcStep: value.source_step,
            pDst: value.destination.cast(),
            nDstStep: value.destination_step,
            pCoeffs: value.coefficients,
            aTransformedCoeffs: value.transformed_coefficients,
        }
    }
}

impl From<sys::NppiWarpAffineBatchCXR> for WarpAffineBatchDescriptor {
    fn from(value: sys::NppiWarpAffineBatchCXR) -> Self {
        Self {
            source: value.pSrc.cast(),
            source_step: value.nSrcStep,
            destination: value.pDst.cast(),
            destination_step: value.nDstStep,
            coefficients: value.pCoeffs,
            transformed_coefficients: value.aTransformedCoeffs,
        }
    }
}

#[derive(Debug, PartialOrd, PartialEq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP perspective-warp batch descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct WarpPerspectiveBatchDescriptor {
    /// Device pointer to the source image.
    pub source: *const (),
    /// Byte stride between source-image rows.
    pub source_step: i32,
    /// Device pointer to the destination image.
    pub destination: *mut (),
    /// Byte stride between destination-image rows.
    pub destination_step: i32,
    /// Device pointer to the double-precision transformation matrix for this image.
    pub coefficients: *mut f64,
    /// Reserved for NPP's transformed coefficients; `coefficients` supplies the input transformation.
    pub transformed_coefficients: [[f64; 3]; 3],
}

impl From<WarpPerspectiveBatchDescriptor> for sys::NppiWarpPerspectiveBatchCXR {
    fn from(value: WarpPerspectiveBatchDescriptor) -> Self {
        Self {
            pSrc: value.source.cast(),
            nSrcStep: value.source_step,
            pDst: value.destination.cast(),
            nDstStep: value.destination_step,
            pCoeffs: value.coefficients,
            aTransformedCoeffs: value.transformed_coefficients,
        }
    }
}

impl From<sys::NppiWarpPerspectiveBatchCXR> for WarpPerspectiveBatchDescriptor {
    fn from(value: sys::NppiWarpPerspectiveBatchCXR) -> Self {
        Self {
            source: value.pSrc.cast(),
            source_step: value.nSrcStep,
            destination: value.pDst.cast(),
            destination_step: value.nDstStep,
            coefficients: value.pCoeffs,
            transformed_coefficients: value.aTransformedCoeffs,
        }
    }
}

#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[non_exhaustive]
#[repr(C)]
/// Non-owning NPP color-twist batch descriptor.
///
/// The descriptor does not own or free its pointers.
/// Keep referenced allocations valid until queued native work completes, including every replay of a captured call.
/// The consuming operation determines pointer access, layout, and memory-space requirements.
pub struct ColorTwistBatchDescriptor {
    /// Device pointer to the source image.
    pub source: *const (),
    /// Byte stride between source-image rows.
    pub source_step: i32,
    /// Device pointer to the destination image.
    pub destination: *mut (),
    /// Byte stride between destination-image rows.
    pub destination_step: i32,
    /// Device pointer to the floating-point color-twist matrix for this image.
    pub twist: *mut f32,
}

impl From<ColorTwistBatchDescriptor> for sys::NppiColorTwistBatchCXR {
    fn from(value: ColorTwistBatchDescriptor) -> Self {
        Self {
            pSrc: value.source.cast(),
            nSrcStep: value.source_step,
            pDst: value.destination.cast(),
            nDstStep: value.destination_step,
            pTwist: value.twist,
        }
    }
}

impl From<sys::NppiColorTwistBatchCXR> for ColorTwistBatchDescriptor {
    fn from(value: sys::NppiColorTwistBatchCXR) -> Self {
        Self {
            source: value.pSrc.cast(),
            source_step: value.nSrcStep,
            destination: value.pDst.cast(),
            destination_step: value.nDstStep,
            twist: value.pTwist,
        }
    }
}

/// Pixel count and intensity statistics for a radial or linear image profile.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Default)]
#[repr(C)]
pub struct ProfileStatistics {
    /// Number of pixels contributing to the profile.
    pub pixels: i32,
    /// Mean pixel intensity of the profile.
    pub mean_intensity: f32,
    /// Standard deviation of pixel intensities in the profile.
    pub standard_deviation_intensity: f32,
}

impl From<ProfileStatistics> for sys::NppiProfileData {
    fn from(value: ProfileStatistics) -> Self {
        Self {
            nPixels: value.pixels,
            nMeanIntensity: value.mean_intensity,
            nStdDevIntensity: value.standard_deviation_intensity,
        }
    }
}

impl From<sys::NppiProfileData> for ProfileStatistics {
    fn from(value: sys::NppiProfileData) -> Self {
        Self {
            pixels: value.nPixels,
            mean_intensity: value.nMeanIntensity,
            standard_deviation_intensity: value.nStdDevIntensity,
        }
    }
}

/// Filtering methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum InterpolationMode {
    /// Undefined filtering interpolation mode.
    Undefined = sys::NppiInterpolationMode::NPPI_INTER_UNDEFINED as _,
    /// Nearest neighbor filtering.
    Nearest = sys::NppiInterpolationMode::NPPI_INTER_NN as _,
    /// Linear interpolation.
    Linear = sys::NppiInterpolationMode::NPPI_INTER_LINEAR as _,
    /// Cubic interpolation.
    Cubic = sys::NppiInterpolationMode::NPPI_INTER_CUBIC as _,
    /// Two-parameter cubic filter (B=1, C=0).
    CubicBspline = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_BSPLINE as _,
    /// Two-parameter cubic filter (B=0, C=1/2).
    CubicCatmullRom = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_CATMULLROM as _,
    /// Two-parameter cubic filter (B=1/2, C=3/10).
    CubicB05C03 = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_B05C03 as _,
    /// Super sampling.
    Super = sys::NppiInterpolationMode::NPPI_INTER_SUPER as _,
    /// Lanczos filtering.
    Lanczos = sys::NppiInterpolationMode::NPPI_INTER_LANCZOS as _,
    /// Generic Lanczos filtering with order 3.
    Lanczos3Advanced = sys::NppiInterpolationMode::NPPI_INTER_LANCZOS3_ADVANCED as _,
    /// Smooth edge filtering.
    SmoothEdge = sys::NppiInterpolationMode::NPPI_SMOOTH_EDGE as _,
}

impl_enum_conversion!(sys::NppiInterpolationMode, InterpolationMode);

impl From<InterpolationMode> for i32 {
    fn from(value: InterpolationMode) -> Self {
        let raw: u32 = value.into();
        raw.cast_signed()
    }
}

impl_enum_display!(InterpolationMode, {
    InterpolationMode::Undefined => "NPPI_INTER_UNDEFINED",
    InterpolationMode::Nearest => "NPPI_INTER_NN",
    InterpolationMode::Linear => "NPPI_INTER_LINEAR",
    InterpolationMode::Cubic => "NPPI_INTER_CUBIC",
    InterpolationMode::CubicBspline => "NPPI_INTER_CUBIC2P_BSPLINE",
    InterpolationMode::CubicCatmullRom => "NPPI_INTER_CUBIC2P_CATMULLROM",
    InterpolationMode::CubicB05C03 => "NPPI_INTER_CUBIC2P_B05C03",
    InterpolationMode::Super => "NPPI_INTER_SUPER",
    InterpolationMode::Lanczos => "NPPI_INTER_LANCZOS",
    InterpolationMode::Lanczos3Advanced => "NPPI_INTER_LANCZOS3_ADVANCED",
    InterpolationMode::SmoothEdge => "NPPI_SMOOTH_EDGE",
});

/// Rounding Modes.
///
/// The enumerated rounding modes are used by a large number of NPP primitives to allow the user to specify the method by which fractional values are converted to integer values. Also see [Rounding Modes](https://docs.nvidia.com/cuda/archive/13.4.2/npp/introduction.html#general_conventions_lb_1rounding_modes).
///
/// For NPP release 5.5 new names for the three rounding modes are introduced that are based on the naming conventions for rounding modes set forth in the IEEE-754 floating-point standard. Developers are encouraged to use the new, longer names to be future proof as the legacy names will be deprecated in subsequent NPP releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum RoundMode {
    /// Round to the nearest even integer.
    ///
    /// All fractional numbers are rounded to their nearest integer. The ambiguous cases (i.e. &lt;integer>.5) are rounded to the closest even integer. E.g.
    ///
    /// * roundNear(0.5) = 0
    /// * roundNear(0.6) = 1
    /// * roundNear(1.5) = 2
    /// * roundNear(-1.5) = -2.
    Near = sys::NppRoundMode::NPP_RND_NEAR as _,
    /// Round according to financial rule.
    ///
    /// All fractional numbers are rounded to their nearest integer. The ambiguous cases (i.e. &lt;integer>.5) are rounded away from zero. E.g.
    ///
    /// * roundFinancial(0.4) = 0
    /// * roundFinancial(0.5) = 1
    /// * roundFinancial(-1.5) = -2.
    Financial = sys::NppRoundMode::NPP_RND_FINANCIAL as _,
    /// Round towards zero (truncation).
    ///
    /// All fractional numbers of the form &lt;integer>.&lt;decimals> are truncated to &lt;integer>.
    ///
    /// * roundZero(1.5) = 1
    /// * roundZero(1.9) = 1
    /// * roundZero(-2.5) = -2.
    Zero = sys::NppRoundMode::NPP_RND_ZERO as _,
}

impl_enum_conversion!(sys::NppRoundMode, RoundMode);

impl_enum_display!(RoundMode, {
    RoundMode::Near => "NPP_RND_NEAR",
    RoundMode::Financial => "NPP_RND_FINANCIAL",
    RoundMode::Zero => "NPP_RND_ZERO",
});

/// Hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum HintAlgorithm {
    /// Hint none, currently these are all ignored.
    None = sys::NppHintAlgorithm::NPP_ALG_HINT_NONE as _,
    /// Hint fast, currently these are all ignored.
    Fast = sys::NppHintAlgorithm::NPP_ALG_HINT_FAST as _,
    /// Hint accurate, currently these are all ignored.
    Accurate = sys::NppHintAlgorithm::NPP_ALG_HINT_ACCURATE as _,
}

impl_enum_conversion!(sys::NppHintAlgorithm, HintAlgorithm);

impl_enum_display!(HintAlgorithm, {
    HintAlgorithm::None => "NPP_ALG_HINT_NONE",
    HintAlgorithm::Fast => "NPP_ALG_HINT_FAST",
    HintAlgorithm::Accurate => "NPP_ALG_HINT_ACCURATE",
});

/// Norm controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ImageNormalization {
    /// maximum.
    Infinity = sys::NppiNorm::nppiNormInf as _,
    /// sum.
    L1 = sys::NppiNorm::nppiNormL1 as _,
    /// square root of sum of squares.
    L2 = sys::NppiNorm::nppiNormL2 as _,
}

impl_enum_conversion!(sys::NppiNorm, ImageNormalization);

impl_enum_display!(ImageNormalization, {
    ImageNormalization::Infinity => "nppiNormInf",
    ImageNormalization::L1 => "nppiNormL1",
    ImageNormalization::L2 => "nppiNormL2",
});

/// Provides control of the type of segment boundaries, if any, added to the image generated by the watershed segmentation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum WatershedSegmentBoundaryType {
    /// Image watershed segment boundary type none.
    None = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE as _,
    /// Image watershed segment boundary type black.
    Black = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK as _,
    /// Image watershed segment boundary type white.
    White = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE as _,
    /// Image watershed segment boundary type contrasting intensity.
    Contrast =
        sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST as _,
    /// Image watershed segment boundary type render boundaries only.
    BoundariesOnly =
        sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY as _,
}

impl_enum_conversion!(
    sys::NppiWatershedSegmentBoundaryType,
    WatershedSegmentBoundaryType
);

impl_enum_display!(WatershedSegmentBoundaryType, {
    WatershedSegmentBoundaryType::None => "NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE",
    WatershedSegmentBoundaryType::Black => "NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK",
    WatershedSegmentBoundaryType::White => "NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE",
    WatershedSegmentBoundaryType::Contrast => "NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST",
    WatershedSegmentBoundaryType::BoundariesOnly => "NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY",
});

/// Pixel comparison control values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ComparisonOperation {
    /// Threshold test for less than.
    Less = sys::NppCmpOp::NPP_CMP_LESS as _,
    /// Threshold test for less than or equal.
    LessEqual = sys::NppCmpOp::NPP_CMP_LESS_EQ as _,
    /// Threshold test for equal.
    Equal = sys::NppCmpOp::NPP_CMP_EQ as _,
    /// Threshold test for greater than or equal.
    GreaterEqual = sys::NppCmpOp::NPP_CMP_GREATER_EQ as _,
    /// Threshold test for greater than.
    Greater = sys::NppCmpOp::NPP_CMP_GREATER as _,
}

impl_enum_conversion!(sys::NppCmpOp, ComparisonOperation);

impl_enum_display!(ComparisonOperation, {
    ComparisonOperation::Less => "NPP_CMP_LESS",
    ComparisonOperation::LessEqual => "NPP_CMP_LESS_EQ",
    ComparisonOperation::Equal => "NPP_CMP_EQ",
    ComparisonOperation::GreaterEqual => "NPP_CMP_GREATER_EQ",
    ComparisonOperation::Greater => "NPP_CMP_GREATER",
});

/// Supported image border modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum BorderType {
    /// Image border type undefined.
    Undefined = sys::NppiBorderType::NPP_BORDER_UNDEFINED as _,
    /// Image border type constant value.
    Constant = sys::NppiBorderType::NPP_BORDER_CONSTANT as _,
    /// Image border type replicate border pixels.
    Replicate = sys::NppiBorderType::NPP_BORDER_REPLICATE as _,
    /// Image border type wrap border pixels.
    Wrap = sys::NppiBorderType::NPP_BORDER_WRAP as _,
    /// Image border type mirror border pixels (IPP mirrored border / `OpenCV` `BORDER_REFLECT_101`): gfedcb|abcdefgh|gfedcba , edge pixel is NOT repeated.
    Mirror = sys::NppiBorderType::NPP_BORDER_MIRROR as _,
    /// Image border type mirror with replication: `fedcba|abcdefgh|hgfedcba`.
    /// The edge pixel is repeated, as in `OpenCV` `BORDER_REFLECT`.
    MirrorReflect = sys::NppiBorderType::NPP_BORDER_MIRROR_REFL as _,
}

impl_enum_conversion!(sys::NppiBorderType, BorderType);

impl_enum_display!(BorderType, {
    BorderType::Undefined => "NPP_BORDER_UNDEFINED",
    BorderType::Constant => "NPP_BORDER_CONSTANT",
    BorderType::Replicate => "NPP_BORDER_REPLICATE",
    BorderType::Wrap => "NPP_BORDER_WRAP",
    BorderType::Mirror => "NPP_BORDER_MIRROR",
    BorderType::MirrorReflect => "NPP_BORDER_MIRROR_REFL",
});

/// Fixed filter-kernel sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MaskSize {
    /// 1 X 3 filter mask size.
    Mask1x3 = sys::NppiMaskSize::NPP_MASK_SIZE_1_X_3 as _,
    /// 1 X 5 filter mask size.
    Mask1x5 = sys::NppiMaskSize::NPP_MASK_SIZE_1_X_5 as _,
    /// 3 X 1 filter mask size, leaving space for more 1 X N type enum values.
    Mask3x1 = sys::NppiMaskSize::NPP_MASK_SIZE_3_X_1 as _,
    /// 5 X 1 filter mask size.
    Mask5x1 = sys::NppiMaskSize::NPP_MASK_SIZE_5_X_1 as _,
    /// 3 X 3 filter mask size, leaving space for more N X 1 type enum values.
    Mask3x3 = sys::NppiMaskSize::NPP_MASK_SIZE_3_X_3 as _,
    /// 5 X 5 filter mask size.
    Mask5x5 = sys::NppiMaskSize::NPP_MASK_SIZE_5_X_5 as _,
    /// 7 X 7 filter mask size.
    Mask7x7 = sys::NppiMaskSize::NPP_MASK_SIZE_7_X_7 as _,
    /// 9 X 9 filter mask size.
    Mask9x9 = sys::NppiMaskSize::NPP_MASK_SIZE_9_X_9 as _,
    /// 11 X 11 filter mask size.
    Mask11x11 = sys::NppiMaskSize::NPP_MASK_SIZE_11_X_11 as _,
    /// 13 X 13 filter mask size.
    Mask13x13 = sys::NppiMaskSize::NPP_MASK_SIZE_13_X_13 as _,
    /// 15 X 15 filter mask size.
    Mask15x15 = sys::NppiMaskSize::NPP_MASK_SIZE_15_X_15 as _,
}

impl_enum_conversion!(sys::NppiMaskSize, MaskSize);

impl_enum_display!(MaskSize, {
    MaskSize::Mask1x3 => "NPP_MASK_SIZE_1_X_3",
    MaskSize::Mask1x5 => "NPP_MASK_SIZE_1_X_5",
    MaskSize::Mask3x1 => "NPP_MASK_SIZE_3_X_1",
    MaskSize::Mask5x1 => "NPP_MASK_SIZE_5_X_1",
    MaskSize::Mask3x3 => "NPP_MASK_SIZE_3_X_3",
    MaskSize::Mask5x5 => "NPP_MASK_SIZE_5_X_5",
    MaskSize::Mask7x7 => "NPP_MASK_SIZE_7_X_7",
    MaskSize::Mask9x9 => "NPP_MASK_SIZE_9_X_9",
    MaskSize::Mask11x11 => "NPP_MASK_SIZE_11_X_11",
    MaskSize::Mask13x13 => "NPP_MASK_SIZE_13_X_13",
    MaskSize::Mask15x15 => "NPP_MASK_SIZE_15_X_15",
});

/// nppiMirror direction controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum Axis {
    /// Flip around horizontal axis in mirror function.
    Horizontal = sys::NppiAxis::NPP_HORIZONTAL_AXIS as _,
    /// Flip around vertical axis in mirror function.
    Vertical = sys::NppiAxis::NPP_VERTICAL_AXIS as _,
    /// Flip around both axes in mirror function.
    Both = sys::NppiAxis::NPP_BOTH_AXIS as _,
}

impl_enum_conversion!(sys::NppiAxis, Axis);

impl_enum_display!(Axis, {
    Axis::Horizontal => "NPP_HORIZONTAL_AXIS",
    Axis::Vertical => "NPP_VERTICAL_AXIS",
    Axis::Both => "NPP_BOTH_AXIS",
});

/// Bayer Grid Position Registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum BayerGridPosition {
    /// Default registration position BGGR.
    Bggr = sys::NppiBayerGridPosition::NPPI_BAYER_BGGR as _,
    /// Registration position RGGB.
    Rggb = sys::NppiBayerGridPosition::NPPI_BAYER_RGGB as _,
    /// Registration position GBRG.
    Gbrg = sys::NppiBayerGridPosition::NPPI_BAYER_GBRG as _,
    /// Registration position GRBG.
    Grbg = sys::NppiBayerGridPosition::NPPI_BAYER_GRBG as _,
}

impl_enum_conversion!(sys::NppiBayerGridPosition, BayerGridPosition);

impl_enum_display!(BayerGridPosition, {
    BayerGridPosition::Bggr => "NPPI_BAYER_BGGR",
    BayerGridPosition::Rggb => "NPPI_BAYER_RGGB",
    BayerGridPosition::Gbrg => "NPPI_BAYER_GBRG",
    BayerGridPosition::Grbg => "NPPI_BAYER_GRBG",
});

/// Differential Filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum DifferentialKernel {
    /// Differential kernel filter type sobel.
    Sobel = sys::NppiDifferentialKernel::NPP_FILTER_SOBEL as _,
    /// Differential kernel filter type scharr.
    Scharr = sys::NppiDifferentialKernel::NPP_FILTER_SCHARR as _,
}

impl_enum_conversion!(sys::NppiDifferentialKernel, DifferentialKernel);

impl_enum_display!(DifferentialKernel, {
    DifferentialKernel::Sobel => "NPP_FILTER_SOBEL",
    DifferentialKernel::Scharr => "NPP_FILTER_SCHARR",
});

/// Alpha composition mode controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum AlphaOperation {
    /// Alpha composition over operation.
    Over = sys::NppiAlphaOp::NPPI_OP_ALPHA_OVER as _,
    /// Alpha composition in operation.
    In = sys::NppiAlphaOp::NPPI_OP_ALPHA_IN as _,
    /// Alpha composition out operation.
    Out = sys::NppiAlphaOp::NPPI_OP_ALPHA_OUT as _,
    /// Alpha composition atop operation.
    Atop = sys::NppiAlphaOp::NPPI_OP_ALPHA_ATOP as _,
    /// Alpha composition xor operation.
    Xor = sys::NppiAlphaOp::NPPI_OP_ALPHA_XOR as _,
    /// Alpha composition plus operation.
    Plus = sys::NppiAlphaOp::NPPI_OP_ALPHA_PLUS as _,
    /// Alpha composition over premultiply operation.
    OverPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL as _,
    /// Alpha composition in premultiply operation.
    InPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL as _,
    /// Alpha composition out premultiply operation.
    OutPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL as _,
    /// Alpha composition atop premultiply operation.
    AtopPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL as _,
    /// Alpha composition xor premultiply operation.
    XorPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_XOR_PREMUL as _,
    /// Alpha composition plus premultiply operation.
    PlusPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL as _,
    /// Alpha composition premultiply operation.
    Premultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_PREMUL as _,
}

impl_enum_conversion!(sys::NppiAlphaOp, AlphaOperation);

impl_enum_display!(AlphaOperation, {
    AlphaOperation::Over => "NPPI_OP_ALPHA_OVER",
    AlphaOperation::In => "NPPI_OP_ALPHA_IN",
    AlphaOperation::Out => "NPPI_OP_ALPHA_OUT",
    AlphaOperation::Atop => "NPPI_OP_ALPHA_ATOP",
    AlphaOperation::Xor => "NPPI_OP_ALPHA_XOR",
    AlphaOperation::Plus => "NPPI_OP_ALPHA_PLUS",
    AlphaOperation::OverPremultiplied => "NPPI_OP_ALPHA_OVER_PREMUL",
    AlphaOperation::InPremultiplied => "NPPI_OP_ALPHA_IN_PREMUL",
    AlphaOperation::OutPremultiplied => "NPPI_OP_ALPHA_OUT_PREMUL",
    AlphaOperation::AtopPremultiplied => "NPPI_OP_ALPHA_ATOP_PREMUL",
    AlphaOperation::XorPremultiplied => "NPPI_OP_ALPHA_XOR_PREMUL",
    AlphaOperation::PlusPremultiplied => "NPPI_OP_ALPHA_PLUS_PREMUL",
    AlphaOperation::Premultiplied => "NPPI_OP_ALPHA_PREMUL",
});

/// Signal sign operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ZeroCrossingType {
    /// sign change.
    SignChange = sys::NppsZCType::nppZCR as _,
    /// sign change XOR.
    SignChangeXor = sys::NppsZCType::nppZCXor as _,
    /// sign change `count_0`.
    SignChangeCountZero = sys::NppsZCType::nppZCC as _,
}

impl_enum_conversion!(sys::NppsZCType, ZeroCrossingType);

impl_enum_display!(ZeroCrossingType, {
    ZeroCrossingType::SignChange => "nppZCR",
    ZeroCrossingType::SignChangeXor => "nppZCXor",
    ZeroCrossingType::SignChangeCountZero => "nppZCC",
});

/// `HuffMan` Table controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum HuffmanTableType {
    /// DC Table.
    Dc = sys::NppiHuffmanTableType::nppiDCTable as _,
    /// AC Table.
    Ac = sys::NppiHuffmanTableType::nppiACTable as _,
}

impl_enum_conversion!(sys::NppiHuffmanTableType, HuffmanTableType);

impl_enum_display!(HuffmanTableType, {
    HuffmanTableType::Dc => "nppiDCTable",
    HuffmanTableType::Ac => "nppiACTable",
});

const fn contour_direction_bits(value: u32) -> u8 {
    assert!(
        value <= u8::MAX as u32,
        "native contour direction exceeds its ABI field"
    );
    value.to_le_bytes()[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_create_rejects_nonpositive_width() {
        assert!(matches!(
            Size::create(0, 1),
            Err(Error::ValueNotPositive { name }) if name == "width"
        ));
    }

    #[test]
    fn test_size_create_rejects_nonpositive_height() {
        assert!(matches!(
            Size::create(1, -1),
            Err(Error::ValueNotPositive { name }) if name == "height"
        ));
    }

    #[test]
    fn test_size_raw_parts_preserves_ffi_values() {
        let size = Size::from_raw_parts(0, -1);
        assert_eq!(size.width, 0);
        assert_eq!(size.height, -1);
        assert!(!size.is_positive());
    }
}
