use std::{
    fmt::Debug,
    fmt::{self, Display, Formatter},
    mem::{align_of, offset_of, size_of},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::types::{Complex, Complex32, Complex64, f16};
use singe_npp_sys as sys;

use crate::error::{Error, Result};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum DataType {
    U8 = sys::NppDataType::NPP_8U as _,
    I8 = sys::NppDataType::NPP_8S as _,
    U16 = sys::NppDataType::NPP_16U as _,
    I16 = sys::NppDataType::NPP_16S as _,
    U32 = sys::NppDataType::NPP_32U as _,
    I32 = sys::NppDataType::NPP_32S as _,
    U64 = sys::NppDataType::NPP_64U as _,
    I64 = sys::NppDataType::NPP_64S as _,
    F16 = sys::NppDataType::NPP_16F as _,
    F32 = sys::NppDataType::NPP_32F as _,
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

pub trait DataTypeLike: Clone + Copy + Debug + 'static {
    fn data_type() -> DataType;

    fn is_complex() -> bool;

    fn rust_type_name() -> &'static str;
}

macro_rules! impl_data_type {
    ($ty:ty, $data_type:ident, $is_complex:expr) => {
        impl DataTypeLike for $ty {
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

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ImageChannels {
    C1 = sys::NppiChannels::NPP_CH_1 as _,
    C2 = sys::NppiChannels::NPP_CH_2 as _,
    C3 = sys::NppiChannels::NPP_CH_3 as _,
    C4 = sys::NppiChannels::NPP_CH_4 as _,
    AC4 = sys::NppiChannels::NPP_CH_A4 as _,
    P2 = sys::NppiChannels::NPP_CH_P2 as _,
    P3 = sys::NppiChannels::NPP_CH_P3 as _,
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

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ColorSpace {
    TranscodeOnly = sys::NppiColorSpace::NPP_TRANSCODE_ONLY as _,
    Bt601Jpeg = sys::NppiColorSpace::NPP_BT601JPEG as _,
    Bt709Hdtv = sys::NppiColorSpace::NPP_BT709HDTV as _,
    Bt2020Hdtv = sys::NppiColorSpace::NPP_BT2020HDTV as _,
}

impl_enum_conversion!(sys::NppiColorSpace, ColorSpace);

impl_enum_display!(ColorSpace, {
    ColorSpace::TranscodeOnly => "NPP_TRANSCODE_ONLY",
    ColorSpace::Bt601Jpeg => "NPP_BT601JPEG",
    ColorSpace::Bt709Hdtv => "NPP_BT709HDTV",
    ColorSpace::Bt2020Hdtv => "NPP_BT2020HDTV",
});

pub type ComplexI16 = Complex<i16>;
pub type ComplexU32 = Complex<u32>;
pub type ComplexI32 = Complex<i32>;
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

pub trait IntoNpp {
    type Npp;

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
    pub struct ContourDirection: u8 {
        const SOUTH_EAST = sys::NPP_CONTOUR_DIRECTION_SOUTH_EAST as u8;
        const SOUTH = sys::NPP_CONTOUR_DIRECTION_SOUTH as u8;
        const SOUTH_WEST = sys::NPP_CONTOUR_DIRECTION_SOUTH_WEST as u8;
        const WEST = sys::NPP_CONTOUR_DIRECTION_WEST as u8;
        const EAST = sys::NPP_CONTOUR_DIRECTION_EAST as u8;
        const NORTH_EAST = sys::NPP_CONTOUR_DIRECTION_NORTH_EAST as u8;
        const NORTH = sys::NPP_CONTOUR_DIRECTION_NORTH as u8;
        const NORTH_WEST = sys::NPP_CONTOUR_DIRECTION_NORTH_WEST as u8;
        const ANY_NORTH = sys::NPP_CONTOUR_DIRECTION_ANY_NORTH as u8;
        const ANY_WEST = sys::NPP_CONTOUR_DIRECTION_ANY_WEST as u8;
        const ANY_SOUTH = sys::NPP_CONTOUR_DIRECTION_ANY_SOUTH as u8;
        const ANY_EAST = sys::NPP_CONTOUR_DIRECTION_ANY_EAST as u8;
    }
}

const _: () = {
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

#[path = "types_geometry.rs"]
mod types_geometry;
pub use types_geometry::*;

#[path = "types_descriptors.rs"]
mod types_descriptors;
pub use types_descriptors::*;

#[path = "types_enums.rs"]
mod types_enums;
pub use types_enums::*;
