use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

/// `cudnnConvolutionMode_t` is an enumerated type used by `cudnnSetConvolution2dDescriptor` to configure a convolution `descriptor`. The filter used for the convolution can be applied in two different ways, corresponding mathematically to a convolution or to a cross-correlation. (A cross-correlation is equivalent to a convolution with its filter rotated by 180 degrees.)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum ConvolutionMode {
    /// In this mode, a convolution operation will be done when applying the filter to the images.
    Convolution = sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION as _,
    /// In this mode, a cross-correlation operation will be done when applying the filter to the images.
    CrossCorrelation = sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION as _,
}

impl_enum_conversion!(sys::cudnnConvolutionMode_t, ConvolutionMode);

impl_enum_display!(ConvolutionMode, {
    Self::Convolution => "CUDNN_CONVOLUTION",
    Self::CrossCorrelation => "CUDNN_CROSS_CORRELATION",
});
