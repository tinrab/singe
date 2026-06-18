use super::*;

#[path = "arithmetic_constant_dispatch_host_arithmetic.rs"]
mod host_arithmetic;
pub use host_arithmetic::*;

#[path = "arithmetic_constant_dispatch_host_logical.rs"]
mod host_logical;
pub use host_logical::*;

#[path = "arithmetic_constant_dispatch_device_arithmetic.rs"]
mod device_arithmetic;
pub use device_arithmetic::*;

#[path = "arithmetic_constant_dispatch_scale.rs"]
mod scale;
pub use scale::*;
