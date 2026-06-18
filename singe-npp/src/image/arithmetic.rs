use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, f16},
};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, C4, ImageView, ImageViewMut, MaskView},
    try_ffi,
    types::{AlphaOperation, ComplexI16, ComplexI32, DataTypeLike, IntoNpp, RoundMode},
};

use super::arithmetic_validation::*;

#[macro_use]
#[path = "arithmetic_direct_macros.rs"]
mod direct_macros;

#[macro_use]
#[path = "arithmetic_generic_macros.rs"]
mod generic_macros;

#[path = "arithmetic_alpha.rs"]
mod alpha;
pub use alpha::*;

#[path = "arithmetic_accumulate.rs"]
mod accumulate;
pub use accumulate::*;

#[path = "arithmetic_binary_ops.rs"]
mod binary_ops;
pub use binary_ops::*;

#[path = "arithmetic_unary_ops.rs"]
mod unary_ops;
pub use unary_ops::*;

#[path = "arithmetic_logical_ops.rs"]
mod logical_ops;
pub use logical_ops::*;

#[path = "arithmetic_constant_ops.rs"]
mod constant_ops;
pub use constant_ops::*;

#[path = "arithmetic_binary_dispatch.rs"]
mod binary_dispatch;
pub use binary_dispatch::*;

#[path = "arithmetic_unary_dispatch.rs"]
mod unary_dispatch;
pub use unary_dispatch::*;

#[path = "arithmetic_constant_dispatch.rs"]
mod constant_dispatch;
pub use constant_dispatch::*;
