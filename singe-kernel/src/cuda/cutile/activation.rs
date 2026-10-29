use std::sync::Arc;

#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::CompileOptions,
};

#[cfg(feature = "dtype_f16")]
use crate::cuda::cutile::kernel::f16::activation as kernel_f16;
#[cfg(feature = "dtype_f32")]
use crate::cuda::cutile::kernel::f32::activation as kernel_f32;
#[cfg(feature = "dtype_f64")]
use crate::cuda::cutile::kernel::f64::activation as kernel_f64;
use crate::{
    cuda::{
        autotune::KernelCompileConfig,
        cutile::{
            DeviceOpExt, TileKernel,
            adapter::TensorAdapter,
            utility::{checked_device_pointer, vector_tile_size},
        },
    },
    error::Result,
};

fn vector_compile_options(compile: KernelCompileConfig) -> CompileOptions {
    let mut options = CompileOptions::default();
    if let Some(occupancy) = compile.occupancy {
        options = options.occupancy(occupancy);
    }
    if let Some(num_cta_in_cga) = compile.num_cta_in_cga {
        options = options.num_cta_in_cga(num_cta_in_cga);
    }
    if let Some(max_divisibility) = compile.max_divisibility {
        options = options.max_divisibility(max_divisibility);
    }
    options
}

macro_rules! enqueue_vector_op {
    ($op:expr, $stream:expr, $compile:expr) => {{
        if $compile.is_default() {
            $op.enqueue_on($stream)?;
        } else {
            $op.compile_options(vector_compile_options($compile))
                .enqueue_on($stream)?;
        }
    }};
}

macro_rules! activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            len: usize,
            tile_width: Option<usize>,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let tile = tile_width.unwrap_or_else(|| vector_tile_size(len));
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            enqueue_vector_op!($kernel::$kernel_fn(out, input), stream, compile);
            Ok(())
        }
    };
}

macro_rules! scalar_activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            value: $ty,
            len: usize,
            tile_width: Option<usize>,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let tile = tile_width.unwrap_or_else(|| vector_tile_size(len));
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            enqueue_vector_op!($kernel::$kernel_fn(out, input, value), stream, compile);
            Ok(())
        }
    };
}

macro_rules! binary_scalar_activation_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            first: $ty,
            second: $ty,
            len: usize,
            tile_width: Option<usize>,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            let tile = tile_width.unwrap_or_else(|| vector_tile_size(len));
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            enqueue_vector_op!(
                $kernel::$kernel_fn(out, input, first, second),
                stream,
                compile
            );
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

#[cfg(feature = "dtype_f32")]
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

#[cfg(feature = "dtype_f32")]
scalar_activation_fns_for_type!(f32, kernel_f32,
    leaky_relu_f32 => leaky_relu_f32,
    elu_f32 => elu_f32,
    celu_f32 => celu_f32,
    hardshrink_f32 => hardshrink_f32,
    softshrink_f32 => softshrink_f32,
);

#[cfg(feature = "dtype_f32")]
binary_scalar_activation_fns_for_type!(f32, kernel_f32,
    threshold_f32 => threshold_f32,
    hardtanh_f32 => hardtanh_f32,
);

#[cfg(feature = "dtype_f16")]
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

#[cfg(feature = "dtype_f16")]
scalar_activation_fns_for_type!(f16, kernel_f16,
    leaky_relu_f16 => leaky_relu_f16,
    elu_f16 => elu_f16,
    celu_f16 => celu_f16,
    hardshrink_f16 => hardshrink_f16,
    softshrink_f16 => softshrink_f16,
);

#[cfg(feature = "dtype_f16")]
binary_scalar_activation_fns_for_type!(f16, kernel_f16,
    threshold_f16 => threshold_f16,
    hardtanh_f16 => hardtanh_f16,
);

#[cfg(feature = "dtype_f64")]
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

#[cfg(feature = "dtype_f64")]
scalar_activation_fns_for_type!(f64, kernel_f64,
    leaky_relu_f64 => leaky_relu_f64,
    elu_f64 => elu_f64,
    celu_f64 => celu_f64,
    hardshrink_f64 => hardshrink_f64,
    softshrink_f64 => softshrink_f64,
);

#[cfg(feature = "dtype_f64")]
binary_scalar_activation_fns_for_type!(f64, kernel_f64,
    threshold_f64 => threshold_f64,
    hardtanh_f64 => hardtanh_f64,
);

macro_rules! swiglu_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            len: usize,
            tile_width: Option<usize>,
            compile: KernelCompileConfig,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            checked_device_pointer(&gate)?;
            let tile = tile_width.unwrap_or_else(|| vector_tile_size(len));
            let out = TensorAdapter::contiguous_1d(out, len)?.partition([tile])?;
            let input = TensorAdapter::contiguous_1d(input, len)?;
            let gate = TensorAdapter::contiguous_1d(gate, len)?;
            enqueue_vector_op!($kernel::$kernel_fn(out, input, gate), stream, compile);
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f32")]
swiglu_fn!(swiglu_f32, f32, kernel_f32, swiglu_f32);

#[cfg(feature = "dtype_f16")]
swiglu_fn!(swiglu_f16, f16, kernel_f16, swiglu_f16);

#[cfg(feature = "dtype_f64")]
swiglu_fn!(swiglu_f64, f64, kernel_f64, swiglu_f64);
