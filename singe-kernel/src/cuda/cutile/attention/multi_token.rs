#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use std::sync::Arc;

#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};

#[cfg(any(feature = "dtype_bf16", feature = "dtype_f16", feature = "dtype_f32"))]
use crate::{
    cuda::cutile::{
        DeviceOpExt, attention::MultiTokenAttention, kernel::attention as kernel_attention,
        runtime::tile_kernel::TileKernel, utility::checked_device_pointer,
    },
    error::Result,
};

#[cfg(feature = "dtype_f32")]
/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn multi_token_attention_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    scores: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    seq_len: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    has_bias: bool,
    sparse: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&scores)?;
    checked_device_pointer(&weight)?;
    if has_bias {
        checked_device_pointer(&bias)?;
    }

    let params = MultiTokenAttention::create(
        batch,
        channels_in,
        channels_out,
        seq_len,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        has_bias,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_attention::multi_token_attention_f32(
            out,
            scores,
            weight,
            bias,
            params.batch,
            params.channels_in,
            params.channels_out,
            params.seq_len,
            params.output_seq_len,
            params.kernel_h,
            params.kernel_w,
            params.stride_h,
            params.stride_w,
            params.padding_h,
            params.padding_w,
            params.dilation_h,
            params.dilation_w,
            params.groups,
            params.has_bias,
            i32::from(sparse),
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! multi_token_attention_half_fn {
    ($name:ident, $dtype:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            scores: DevicePointer<$dtype>,
            weight: DevicePointer<$dtype>,
            bias: DevicePointer<$dtype>,
            batch: usize,
            channels_in: usize,
            channels_out: usize,
            seq_len: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            padding_h: usize,
            padding_w: usize,
            dilation_h: usize,
            dilation_w: usize,
            groups: usize,
            has_bias: bool,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&scores)?;
            checked_device_pointer(&weight)?;
            if has_bias {
                checked_device_pointer(&bias)?;
            }

            let params = MultiTokenAttention::create(
                batch,
                channels_in,
                channels_out,
                seq_len,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
                has_bias,
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                kernel_attention::$kernel(
                    out,
                    scores,
                    weight,
                    bias,
                    params.batch,
                    params.channels_in,
                    params.channels_out,
                    params.seq_len,
                    params.output_seq_len,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.padding_h,
                    params.padding_w,
                    params.dilation_h,
                    params.dilation_w,
                    params.groups,
                    params.has_bias,
                    0,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
multi_token_attention_half_fn!(multi_token_attention_f16, f16, multi_token_attention_f16);

#[cfg(feature = "dtype_bf16")]
multi_token_attention_half_fn!(multi_token_attention_bf16, bf16, multi_token_attention_bf16);

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! multi_token_attention_sparse_half_fn {
    ($name:ident, $dtype:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$dtype>,
            scores: DevicePointer<$dtype>,
            weight: DevicePointer<$dtype>,
            bias: DevicePointer<$dtype>,
            batch: usize,
            channels_in: usize,
            channels_out: usize,
            seq_len: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            padding_h: usize,
            padding_w: usize,
            dilation_h: usize,
            dilation_w: usize,
            groups: usize,
            has_bias: bool,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&scores)?;
            checked_device_pointer(&weight)?;
            if has_bias {
                checked_device_pointer(&bias)?;
            }

            let params = MultiTokenAttention::create(
                batch,
                channels_in,
                channels_out,
                seq_len,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
                has_bias,
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                kernel_attention::$kernel(
                    out,
                    scores,
                    weight,
                    bias,
                    params.batch,
                    params.channels_in,
                    params.channels_out,
                    params.seq_len,
                    params.output_seq_len,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.padding_h,
                    params.padding_w,
                    params.dilation_h,
                    params.dilation_w,
                    params.groups,
                    params.has_bias,
                    1,
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
multi_token_attention_sparse_half_fn!(
    multi_token_attention_sparse_f16,
    f16,
    multi_token_attention_f16
);

#[cfg(feature = "dtype_bf16")]
multi_token_attention_sparse_half_fn!(
    multi_token_attention_sparse_bf16,
    bf16,
    multi_token_attention_bf16
);
