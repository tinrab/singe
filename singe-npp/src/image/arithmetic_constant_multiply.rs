use super::*;

#[path = "arithmetic_constant_multiply_host.rs"]
mod host;
pub use host::*;

#[path = "arithmetic_constant_multiply_device.rs"]
mod device;
pub use device::*;

#[path = "arithmetic_constant_multiply_host_scale.rs"]
mod host_scale;
pub use host_scale::*;

#[path = "arithmetic_constant_multiply_device_scale.rs"]
mod device_scale;
pub use device_scale::*;
