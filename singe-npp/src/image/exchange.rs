use singe_cuda::types::{Complex32, f16};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::exchange_validation::*,
    image::view::{
        AC4, C1, C2, C3, C4, ImageView, ImageViewMut, MaskView, PlanarImageView, PlanarImageViewMut,
    },
    try_ffi,
    types::{ComplexI16, ComplexI32, DataTypeLike, HintAlgorithm, IntoNpp, RoundMode},
};

#[macro_use]
#[path = "exchange_direct_macros.rs"]
mod direct_macros;

#[macro_use]
#[path = "exchange_convert_scale_macros.rs"]
mod convert_scale_macros;

#[macro_use]
#[path = "exchange_copy_channel_macros.rs"]
mod copy_channel_macros;

#[macro_use]
#[path = "exchange_transpose_border_macros.rs"]
mod transpose_border_macros;

#[macro_use]
#[path = "exchange_swap_set_macros.rs"]
mod swap_set_macros;

#[path = "exchange_set.rs"]
mod set;
pub use set::*;

#[path = "exchange_copy.rs"]
mod copy;
pub use copy::*;

#[path = "exchange_copy_variants.rs"]
mod copy_variants;
pub use copy_variants::*;

#[path = "exchange_convert.rs"]
mod convert;
pub use convert::*;

#[path = "exchange_channels.rs"]
mod channels;
pub use channels::*;

#[path = "exchange_scale.rs"]
mod scale;
pub use scale::*;

#[path = "exchange_transpose.rs"]
mod transpose;
pub use transpose::*;

#[path = "exchange_swap_channels.rs"]
mod swap_channels;
pub use swap_channels::*;
