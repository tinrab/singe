use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::impl_enum_display;

use singe_cudnn_sys as sys;

singe_core::native_enum! {
    /// Selects whether a convolution descriptor applies its filter as a true convolution or as a cross-correlation.
    /// A cross-correlation equals a convolution with the filter rotated by 180 degrees.
    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
    )]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ConvolutionMode: sys::cudnnConvolutionMode_t {
        /// Applies the filter as a mathematical convolution.
        Convolution = CUDNN_CONVOLUTION,
        /// Applies the filter as a cross-correlation.
        CrossCorrelation = CUDNN_CROSS_CORRELATION,
    }
}

impl_enum_display!(ConvolutionMode, {
    Self::Convolution => "CUDNN_CONVOLUTION",
    Self::CrossCorrelation => "CUDNN_CROSS_CORRELATION",
});
