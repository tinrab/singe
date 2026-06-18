use super::{ImagePipeline, filtering_traits::*};

#[macro_use]
#[path = "filtering_dispatch_kernel_macros.rs"]
mod macros;

#[path = "filtering_dispatch_float_kernel.rs"]
mod float_kernel;
#[path = "filtering_dispatch_integer_kernel.rs"]
mod integer_kernel;
#[path = "filtering_dispatch_typed_kernel.rs"]
mod typed_kernel;
