#[macro_use]
#[path = "operation_impls_float_device_constant_arithmetic_macros.rs"]
mod macros;

#[path = "operation_impls_float_device_constant_arithmetic_f32.rs"]
mod f32_impls;

#[path = "operation_impls_float_device_constant_arithmetic_f16.rs"]
mod f16_impls;
