use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::activation as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::activation as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::activation as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        adapter::TensorAdapter,
        utility::{checked_device_pointer, vector_tile_size},
    },
    error::Result,
};

macro_rules! activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            $kernel::$kernel_fn(out, input).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! scalar_activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            value: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            $kernel::$kernel_fn(out, input, value).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! binary_scalar_activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            first: $ty,
            second: $ty,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            $kernel::$kernel_fn(out, input, first, second).enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! activation_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(activation_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

macro_rules! scalar_activation_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(scalar_activation_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

macro_rules! binary_scalar_activation_fns_for_type {
    ($ty:ty, $kernel:ident, $(
        $name:ident => $kernel_fn:ident
    ),* $(,)?) => {
        $(binary_scalar_activation_fn!($name, $ty, $kernel, $kernel_fn);)*
    };
}

#[cfg(feature = "dtype-f32")]
activation_fns_for_type!(f32, kernel_f32,
    relu_f32 => relu_f32,
    relu6_f32 => relu6_f32,
    sigmoid_f32 => sigmoid_f32,
    silu_f32 => silu_f32,
    gelu_f32 => gelu_f32,
    softplus_f32 => softplus_f32,
    hard_sigmoid_f32 => hard_sigmoid_f32,
    hard_swish_f32 => hard_swish_f32,
    mish_f32 => mish_f32,
    selu_f32 => selu_f32,
    softsign_f32 => softsign_f32,
    tanhshrink_f32 => tanhshrink_f32,
    log_sigmoid_f32 => log_sigmoid_f32,
);
#[cfg(feature = "dtype-f32")]
scalar_activation_fns_for_type!(f32, kernel_f32,
    leaky_relu_f32 => leaky_relu_f32,
    elu_f32 => elu_f32,
    celu_f32 => celu_f32,
    hardshrink_f32 => hardshrink_f32,
    softshrink_f32 => softshrink_f32,
);
#[cfg(feature = "dtype-f32")]
binary_scalar_activation_fns_for_type!(f32, kernel_f32,
    threshold_f32 => threshold_f32,
    hardtanh_f32 => hardtanh_f32,
);

#[cfg(feature = "dtype-f16")]
activation_fns_for_type!(f16, kernel_f16,
    relu_f16 => relu_f16,
    relu6_f16 => relu6_f16,
    sigmoid_f16 => sigmoid_f16,
    silu_f16 => silu_f16,
    gelu_f16 => gelu_f16,
    softplus_f16 => softplus_f16,
    hard_sigmoid_f16 => hard_sigmoid_f16,
    hard_swish_f16 => hard_swish_f16,
    mish_f16 => mish_f16,
    selu_f16 => selu_f16,
    softsign_f16 => softsign_f16,
    tanhshrink_f16 => tanhshrink_f16,
    log_sigmoid_f16 => log_sigmoid_f16,
);
#[cfg(feature = "dtype-f16")]
scalar_activation_fns_for_type!(f16, kernel_f16,
    leaky_relu_f16 => leaky_relu_f16,
    elu_f16 => elu_f16,
    celu_f16 => celu_f16,
    hardshrink_f16 => hardshrink_f16,
    softshrink_f16 => softshrink_f16,
);
#[cfg(feature = "dtype-f16")]
binary_scalar_activation_fns_for_type!(f16, kernel_f16,
    threshold_f16 => threshold_f16,
    hardtanh_f16 => hardtanh_f16,
);

#[cfg(feature = "dtype-f64")]
activation_fns_for_type!(f64, kernel_f64,
    relu_f64 => relu_f64,
    relu6_f64 => relu6_f64,
    sigmoid_f64 => sigmoid_f64,
    silu_f64 => silu_f64,
    gelu_f64 => gelu_f64,
    softplus_f64 => softplus_f64,
    hard_sigmoid_f64 => hard_sigmoid_f64,
    hard_swish_f64 => hard_swish_f64,
    mish_f64 => mish_f64,
    selu_f64 => selu_f64,
    softsign_f64 => softsign_f64,
    tanhshrink_f64 => tanhshrink_f64,
    log_sigmoid_f64 => log_sigmoid_f64,
);
#[cfg(feature = "dtype-f64")]
scalar_activation_fns_for_type!(f64, kernel_f64,
    leaky_relu_f64 => leaky_relu_f64,
    elu_f64 => elu_f64,
    celu_f64 => celu_f64,
    hardshrink_f64 => hardshrink_f64,
    softshrink_f64 => softshrink_f64,
);
#[cfg(feature = "dtype-f64")]
binary_scalar_activation_fns_for_type!(f64, kernel_f64,
    threshold_f64 => threshold_f64,
    hardtanh_f64 => hardtanh_f64,
);

macro_rules! swiglu_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(gate)?;
            let tile = vector_tile_size(len);
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            let gate = TensorAdapter::contiguous_1d(gate, len)?;
            $kernel::$kernel_fn(out, input, gate).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
swiglu_fn!(swiglu_f32, f32, kernel_f32, swiglu_f32);
#[cfg(feature = "dtype-f16")]
swiglu_fn!(swiglu_f16, f16, kernel_f16, swiglu_f16);
#[cfg(feature = "dtype-f64")]
swiglu_fn!(swiglu_f64, f64, kernel_f64, swiglu_f64);
