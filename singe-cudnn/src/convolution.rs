use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum ConvolutionMode {
    Convolution = sys::cudnnConvolutionMode_t::CUDNN_CONVOLUTION as _,
    CrossCorrelation = sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION as _,
}

impl_enum_conversion!(sys::cudnnConvolutionMode_t, ConvolutionMode);

impl_enum_display!(ConvolutionMode, {
    Self::Convolution => "CUDNN_CONVOLUTION",
    Self::CrossCorrelation => "CUDNN_CROSS_CORRELATION",
});
