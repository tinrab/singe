use std::sync::Arc;

#[cfg(feature = "dtype-f8")]
use cutile::core::f8e4m3fn;
#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::quantization as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::quantization as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::quantization as kernel_f64;
use crate::{
    cuda::{
        cutile::{
            DeviceOpExt,
            adapter::TensorAdapter,
            kernel::common as kernel_common,
            utility::{VectorLaunch, checked_device_pointer, vector_tile_size},
        },
        quantization::Int4Packing,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QuantizedMatmul {
    rows: i32,
    columns: i32,
    reduction: i32,
    activation_row_stride: i32,
    weight_row_stride: i32,
    output_row_stride: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockDequant {
    columns: i32,
    block_size: i32,
    scale_columns: i32,
    len: i32,
    grid: (u32, u32, u32),
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
impl BlockDequant {
    fn create(rows: usize, columns: usize, block_size: usize) -> Result<Self> {
        if rows == 0 || columns == 0 || block_size == 0 {
            return Err(Error::InvalidLength);
        }
        let len = checked_element_count(rows, columns)?;
        let launch = VectorLaunch::create(len)?;
        Ok(Self {
            columns: checked_i32_value(columns)?,
            block_size: checked_i32_value(block_size)?,
            scale_columns: checked_i32_value(columns.div_ceil(block_size))?,
            len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockScaledFp8Matmul {
    rows: i32,
    columns: i32,
    reduction: i32,
    group_n: i32,
    group_k: i32,
    k_groups: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
impl BlockScaledFp8Matmul {
    fn create(
        rows: usize,
        columns: usize,
        reduction: usize,
        group_n: usize,
        group_k: usize,
    ) -> Result<Self> {
        if rows == 0 || columns == 0 || reduction == 0 || group_n == 0 || group_k == 0 {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(rows, columns)?;
        let launch = VectorLaunch::create(output_len)?;
        Ok(Self {
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            group_n: checked_i32_value(group_n)?,
            group_k: checked_i32_value(group_k)?,
            k_groups: checked_i32_value(reduction.div_ceil(group_k))?,
            output_len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct RopeQuantizeFp8 {
    heads: i32,
    rope_dim: i32,
    cos_sin_cache_stride: i32,
    quant_scale: f32,
    len: i32,
    grid: (u32, u32, u32),
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct QkRopeQuantizeFp8 {
    q_heads: i32,
    kv_heads: i32,
    rope_dim: i32,
    no_rope_dim: i32,
    cos_sin_cache_stride: i32,
    quant_scale_q: f32,
    quant_scale_kv: f32,
    q_rope_len: i32,
    k_rope_len: i32,
    q_nope_len: i32,
    total_len: i32,
    grid: (u32, u32, u32),
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
impl RopeQuantizeFp8 {
    fn create(
        tokens: usize,
        heads: usize,
        rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale: f32,
    ) -> Result<Self> {
        if tokens == 0
            || heads == 0
            || rope_dim == 0
            || !rope_dim.is_multiple_of(2)
            || cos_sin_cache_stride < rope_dim
            || !quant_scale.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let len = checked_element_count(checked_element_count(tokens, heads)?, rope_dim)?;
        let launch = VectorLaunch::create(len)?;
        Ok(Self {
            heads: checked_i32_value(heads)?,
            rope_dim: checked_i32_value(rope_dim)?,
            cos_sin_cache_stride: checked_i32_value(cos_sin_cache_stride)?,
            quant_scale,
            len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
impl QkRopeQuantizeFp8 {
    fn create(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        rope_dim: usize,
        no_rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale_q: f32,
        quant_scale_kv: f32,
    ) -> Result<Self> {
        if tokens == 0
            || q_heads == 0
            || kv_heads == 0
            || rope_dim == 0
            || !rope_dim.is_multiple_of(2)
            || cos_sin_cache_stride < rope_dim
            || !quant_scale_q.is_finite()
            || !quant_scale_kv.is_finite()
        {
            return Err(Error::InvalidLength);
        }
        let q_rope_len = checked_element_count(checked_element_count(tokens, q_heads)?, rope_dim)?;
        let k_rope_len = checked_element_count(checked_element_count(tokens, kv_heads)?, rope_dim)?;
        let q_nope_len =
            checked_element_count(checked_element_count(tokens, q_heads)?, no_rope_dim)?;
        let k_nope_len =
            checked_element_count(checked_element_count(tokens, kv_heads)?, no_rope_dim)?;
        let total_len = q_rope_len
            .checked_add(k_rope_len)
            .and_then(|value| value.checked_add(q_nope_len))
            .and_then(|value| value.checked_add(k_nope_len))
            .ok_or(Error::SizeOverflow)?;
        let launch = VectorLaunch::create(total_len)?;
        Ok(Self {
            q_heads: checked_i32_value(q_heads)?,
            kv_heads: checked_i32_value(kv_heads)?,
            rope_dim: checked_i32_value(rope_dim)?,
            no_rope_dim: checked_i32_value(no_rope_dim)?,
            cos_sin_cache_stride: checked_i32_value(cos_sin_cache_stride)?,
            quant_scale_q,
            quant_scale_kv,
            q_rope_len: checked_i32_value(q_rope_len)?,
            k_rope_len: checked_i32_value(k_rope_len)?,
            q_nope_len: checked_i32_value(q_nope_len)?,
            total_len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

impl QuantizedMatmul {
    fn create(
        rows: usize,
        columns: usize,
        reduction: usize,
        activation_row_stride: usize,
        weight_row_stride: usize,
        output_row_stride: usize,
    ) -> Result<Self> {
        if rows == 0
            || columns == 0
            || reduction == 0
            || activation_row_stride < reduction
            || weight_row_stride < reduction
            || output_row_stride < columns
        {
            return Err(Error::InvalidLength);
        }

        let output_len = checked_element_count(rows, columns)?;
        let launch = VectorLaunch::create(output_len)?;
        Ok(Self {
            rows: checked_i32_value(rows)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            activation_row_stride: checked_i32_value(activation_row_stride)?,
            weight_row_stride: checked_i32_value(weight_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            output_len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

macro_rules! dequantize_fn {
    ($name:ident, $out:ty, $input:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out>,
            input: DevicePointer<$input>,
            scale: $out,
            zero_point: $out,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, scale, zero_point, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! quantize_fn {
    ($name:ident, $out:ty, $input:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$out>,
            input: DevicePointer<$input>,
            scale: $input,
            zero_point: $input,
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
            $kernel::$kernel_fn(out, input, scale, zero_point).enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
dequantize_fn!(dequantize_u8_to_f16, f16, u8, kernel_f16, dequantize_u8_f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
dequantize_fn!(dequantize_i8_to_f16, f16, i8, kernel_f16, dequantize_i8_f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
dequantize_fn!(dequantize_u8_to_f32, f32, u8, kernel_f32, dequantize_u8_f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
dequantize_fn!(dequantize_i8_to_f32, f32, i8, kernel_f32, dequantize_i8_f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
dequantize_fn!(dequantize_u8_to_f64, f64, u8, kernel_f64, dequantize_u8_f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
dequantize_fn!(dequantize_i8_to_f64, f64, i8, kernel_f64, dequantize_i8_f64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
quantize_fn!(quantize_f16_to_u8, u8, f16, kernel_f16, quantize_f16_u8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
quantize_fn!(quantize_f16_to_i8, i8, f16, kernel_f16, quantize_f16_i8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
quantize_fn!(quantize_f32_to_u8, u8, f32, kernel_f32, quantize_f32_u8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
quantize_fn!(quantize_f32_to_i8, i8, f32, kernel_f32, quantize_f32_i8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
quantize_fn!(quantize_f64_to_u8, u8, f64, kernel_f64, quantize_f64_u8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
quantize_fn!(quantize_f64_to_i8, i8, f64, kernel_f64, quantize_f64_i8);

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn quantize_f32_to_f8e4m3_block(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    len: usize,
    block_size: usize,
    blocks: usize,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    validate_block_quant(out, scales, input, len, block_size, blocks)?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe {
        kernel_f32::quantize_f32_f8e4m3_block(
            out,
            scales,
            input,
            checked_i32_value(len)?,
            checked_i32_value(block_size)?,
        )
    }
    .grid((blocks as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn dequantize_f8e4m3_block_to_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Result<()> {
    let params = BlockDequant::create(rows, columns, block_size)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(scales)?;
    let input = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(input.cu_deviceptr()) };
    unsafe {
        kernel_f32::dequantize_f8e4m3_f32_block(
            out,
            input,
            scales,
            params.columns,
            params.block_size,
            params.scale_columns,
            params.len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]

pub fn matmul_f8e4m3_f8e4m3_block_scaled_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    activations: DevicePointer<u8>,
    weights: DevicePointer<u8>,
    activation_scales: DevicePointer<f32>,
    weight_scales: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    group_n: usize,
    group_k: usize,
) -> Result<()> {
    let params = BlockScaledFp8Matmul::create(rows, columns, reduction, group_n, group_k)?;
    checked_device_pointer(out)?;
    checked_device_pointer(activations)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(activation_scales)?;
    checked_device_pointer(weight_scales)?;
    let activations =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(activations.cu_deviceptr()) };
    let weights = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(weights.cu_deviceptr()) };
    unsafe {
        kernel_f32::matmul_f8e4m3_f8e4m3_block_scaled_f32(
            out,
            activations,
            weights,
            activation_scales,
            weight_scales,
            params.rows,
            params.columns,
            params.reduction,
            params.group_n,
            params.group_k,
            params.k_groups,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8", feature = "dtype-f16"))]

pub fn matmul_f8e4m3_f8e4m3_block_scaled_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    activations: DevicePointer<u8>,
    weights: DevicePointer<u8>,
    activation_scales: DevicePointer<f32>,
    weight_scales: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    group_n: usize,
    group_k: usize,
) -> Result<()> {
    let params = BlockScaledFp8Matmul::create(rows, columns, reduction, group_n, group_k)?;
    checked_device_pointer(out)?;
    checked_device_pointer(activations)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(activation_scales)?;
    checked_device_pointer(weight_scales)?;
    let activations =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(activations.cu_deviceptr()) };
    let weights = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(weights.cu_deviceptr()) };
    unsafe {
        kernel_f32::matmul_f8e4m3_f8e4m3_block_scaled_f16(
            out,
            activations,
            weights,
            activation_scales,
            weight_scales,
            params.rows,
            params.columns,
            params.reduction,
            params.group_n,
            params.group_k,
            params.k_groups,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8", feature = "dtype-bf16"))]

pub fn matmul_f8e4m3_f8e4m3_block_scaled_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    activations: DevicePointer<u8>,
    weights: DevicePointer<u8>,
    activation_scales: DevicePointer<f32>,
    weight_scales: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    group_n: usize,
    group_k: usize,
) -> Result<()> {
    let params = BlockScaledFp8Matmul::create(rows, columns, reduction, group_n, group_k)?;
    checked_device_pointer(out)?;
    checked_device_pointer(activations)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(activation_scales)?;
    checked_device_pointer(weight_scales)?;
    let activations =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(activations.cu_deviceptr()) };
    let weights = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(weights.cu_deviceptr()) };
    unsafe {
        kernel_f32::matmul_f8e4m3_f8e4m3_block_scaled_bf16(
            out,
            activations,
            weights,
            activation_scales,
            weight_scales,
            params.rows,
            params.columns,
            params.reduction,
            params.group_n,
            params.group_k,
            params.k_groups,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]

pub fn rope_quantize_f8e4m3_f32_interleaved(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    input: DevicePointer<f32>,
    cos_sin_cache: DevicePointer<f32>,
    pos_ids: DevicePointer<u32>,
    tokens: usize,
    heads: usize,
    rope_dim: usize,
    cos_sin_cache_stride: usize,
    quant_scale: f32,
) -> Result<()> {
    let params =
        RopeQuantizeFp8::create(tokens, heads, rope_dim, cos_sin_cache_stride, quant_scale)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(cos_sin_cache)?;
    checked_device_pointer(pos_ids)?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe {
        kernel_f32::rope_quantize_f8e4m3_f32_interleaved(
            out,
            input,
            cos_sin_cache,
            pos_ids,
            params.heads,
            params.rope_dim,
            params.cos_sin_cache_stride,
            params.quant_scale,
            params.len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]

pub fn qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
    stream: &Arc<Stream>,
    q_rope_out: DevicePointer<u8>,
    k_rope_out: DevicePointer<u8>,
    q_nope_out: DevicePointer<u8>,
    k_nope_out: DevicePointer<u8>,
    q_rope: DevicePointer<f32>,
    k_rope: DevicePointer<f32>,
    q_nope: DevicePointer<f32>,
    k_nope: DevicePointer<f32>,
    cos_sin_cache: DevicePointer<f32>,
    pos_ids: DevicePointer<u32>,
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    rope_dim: usize,
    no_rope_dim: usize,
    cos_sin_cache_stride: usize,
    quant_scale_q: f32,
    quant_scale_kv: f32,
) -> Result<()> {
    let params = QkRopeQuantizeFp8::create(
        tokens,
        q_heads,
        kv_heads,
        rope_dim,
        no_rope_dim,
        cos_sin_cache_stride,
        quant_scale_q,
        quant_scale_kv,
    )?;
    checked_device_pointer(q_rope_out)?;
    checked_device_pointer(k_rope_out)?;
    checked_device_pointer(q_rope)?;
    checked_device_pointer(k_rope)?;
    checked_device_pointer(cos_sin_cache)?;
    checked_device_pointer(pos_ids)?;
    if no_rope_dim != 0 {
        checked_device_pointer(q_nope_out)?;
        checked_device_pointer(k_nope_out)?;
        checked_device_pointer(q_nope)?;
        checked_device_pointer(k_nope)?;
    }
    let q_rope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(q_rope_out.cu_deviceptr()) };
    let k_rope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(k_rope_out.cu_deviceptr()) };
    let q_nope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(q_nope_out.cu_deviceptr()) };
    let k_nope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(k_nope_out.cu_deviceptr()) };
    unsafe {
        kernel_f32::qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
            q_rope_out,
            k_rope_out,
            q_nope_out,
            k_nope_out,
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            params.q_heads,
            params.kv_heads,
            params.rope_dim,
            params.no_rope_dim,
            params.cos_sin_cache_stride,
            params.quant_scale_q,
            params.quant_scale_kv,
            params.q_rope_len,
            params.k_rope_len,
            params.q_nope_len,
            params.total_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]

pub fn qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
    stream: &Arc<Stream>,
    q_rope_out: DevicePointer<u8>,
    k_rope_out: DevicePointer<u8>,
    q_rope: DevicePointer<f32>,
    k_rope: DevicePointer<f32>,
    cos_sin_cache: DevicePointer<f32>,
    pos_ids: DevicePointer<u32>,
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    rope_dim: usize,
    cos_sin_cache_stride: usize,
    quant_scale_q: f32,
    quant_scale_kv: f32,
) -> Result<()> {
    let params = QkRopeQuantizeFp8::create(
        tokens,
        q_heads,
        kv_heads,
        rope_dim,
        0,
        cos_sin_cache_stride,
        quant_scale_q,
        quant_scale_kv,
    )?;
    checked_device_pointer(q_rope_out)?;
    checked_device_pointer(k_rope_out)?;
    checked_device_pointer(q_rope)?;
    checked_device_pointer(k_rope)?;
    checked_device_pointer(cos_sin_cache)?;
    checked_device_pointer(pos_ids)?;
    let q_rope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(q_rope_out.cu_deviceptr()) };
    let k_rope_out =
        unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(k_rope_out.cu_deviceptr()) };
    unsafe {
        kernel_f32::qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            q_rope_out,
            k_rope_out,
            q_rope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            params.q_heads,
            params.kv_heads,
            params.rope_dim,
            params.cos_sin_cache_stride,
            params.quant_scale_q,
            params.quant_scale_kv,
            params.q_rope_len,
            params.total_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn quantize_f32_to_f8e4m3_scaled(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    input: DevicePointer<f32>,
    len: usize,
    quant_scale: f32,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if !quant_scale.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    let launch = VectorLaunch::create(len)?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe { kernel_f32::quantize_f32_f8e4m3_scaled(out, input, quant_scale, launch.len_i32) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-u8")]
pub fn unpack_u4_u8(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    packed: DevicePointer<u8>,
    len: usize,
    packing: Int4Packing,
) -> Result<()> {
    validate_unpack_int4(out, packed, len)?;
    let launch = VectorLaunch::create(len)?;
    let high_first = packing.kernel_high_first();
    unsafe { kernel_common::unpack_u4_u8(out, packed, launch.len_i32, high_first) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
pub fn unpack_i4_i8(
    stream: &Arc<Stream>,
    out: DevicePointer<i8>,
    packed: DevicePointer<u8>,
    len: usize,
    packing: Int4Packing,
) -> Result<()> {
    validate_unpack_int4(out, packed, len)?;
    let launch = VectorLaunch::create(len)?;
    let high_first = packing.kernel_high_first();
    unsafe { kernel_common::unpack_i4_i8(out, packed, launch.len_i32, high_first) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
pub fn dequantize_u8_f32_grouped(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    zero_points: DevicePointer<f32>,
    len: usize,
    group_size: usize,
    groups: usize,
) -> Result<()> {
    validate_grouped_dequant(out, input, scales, zero_points, len, group_size, groups)?;
    let launch = VectorLaunch::create(len)?;
    let group_size = checked_i32_value(group_size)?;
    unsafe {
        kernel_f32::dequantize_u8_f32_grouped(
            out,
            input,
            scales,
            zero_points,
            launch.len_i32,
            group_size,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn dequantize_i8_f32_grouped(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    zero_points: DevicePointer<f32>,
    len: usize,
    group_size: usize,
    groups: usize,
) -> Result<()> {
    validate_grouped_dequant(out, input, scales, zero_points, len, group_size, groups)?;
    let launch = VectorLaunch::create(len)?;
    let group_size = checked_i32_value(group_size)?;
    unsafe {
        kernel_f32::dequantize_i8_f32_grouped(
            out,
            input,
            scales,
            zero_points,
            launch.len_i32,
            group_size,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn per_token_group_quant_i8_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Result<()> {
    validate_per_token_group_quant(out, scales, input, groups, group_size, eps)?;
    unsafe {
        kernel_f32::per_token_group_quant_i8_f32(
            out,
            scales,
            input,
            checked_i32_value(groups)?,
            checked_i32_value(group_size)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn per_token_group_quant_i8_f32_column_major_scales(
    stream: &Arc<Stream>,
    out: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<()> {
    let groups = validate_per_token_group_quant_column_major(
        out,
        scales,
        input,
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    unsafe {
        kernel_f32::per_token_group_quant_i8_f32_column_major_scales(
            out,
            scales,
            input,
            checked_i32_value(rows)?,
            checked_i32_value(columns)?,
            checked_i32_value(group_size)?,
            checked_i32_value(scale_column_stride)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn per_token_group_quant_i8_f32_column_major_ue8m0_scales(
    stream: &Arc<Stream>,
    out: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<()> {
    let groups = validate_per_token_group_quant_column_major(
        out,
        scales,
        input,
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    unsafe {
        kernel_f32::per_token_group_quant_i8_f32_column_major_ue8m0_scales(
            out,
            scales,
            input,
            checked_i32_value(rows)?,
            checked_i32_value(columns)?,
            checked_i32_value(group_size)?,
            checked_i32_value(scale_column_stride)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn per_token_group_quant_f8e4m3_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Result<()> {
    validate_per_token_group_quant_f8(out, scales, input, groups, group_size, eps)?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe {
        kernel_f32::per_token_group_quant_f8e4m3_f32(
            out,
            scales,
            input,
            checked_i32_value(groups)?,
            checked_i32_value(group_size)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn per_token_group_quant_f8e4m3_f32_column_major_scales(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<()> {
    let groups = validate_per_token_group_quant_column_major_f8(
        out,
        scales,
        input,
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe {
        kernel_f32::per_token_group_quant_f8e4m3_f32_column_major_scales(
            out,
            scales,
            input,
            checked_i32_value(rows)?,
            checked_i32_value(columns)?,
            checked_i32_value(group_size)?,
            checked_i32_value(scale_column_stride)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
pub fn per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
    stream: &Arc<Stream>,
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<()> {
    let groups = validate_per_token_group_quant_column_major_f8(
        out,
        scales,
        input,
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    let out = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(out.cu_deviceptr()) };
    unsafe {
        kernel_f32::per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
            out,
            scales,
            input,
            checked_i32_value(rows)?,
            checked_i32_value(columns)?,
            checked_i32_value(group_size)?,
            checked_i32_value(scale_column_stride)?,
            eps,
        )
    }
    .grid((groups as u32, 1, 1))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn matmul_i8_i8_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    activations: DevicePointer<i8>,
    weights: DevicePointer<i8>,
    rows: usize,
    columns: usize,
    reduction: usize,
    activation_row_stride: usize,
    weight_row_stride: usize,
    output_row_stride: usize,
    activation_zero_point: i32,
    weight_zero_point: i32,
    output_scale: f32,
) -> Result<()> {
    if !output_scale.is_finite() {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(activations)?;
    checked_device_pointer(weights)?;
    let params = QuantizedMatmul::create(
        rows,
        columns,
        reduction,
        activation_row_stride,
        weight_row_stride,
        output_row_stride,
    )?;
    unsafe {
        kernel_f32::matmul_i8_i8_f32(
            out,
            activations,
            weights,
            params.rows,
            params.columns,
            params.reduction,
            params.activation_row_stride,
            params.weight_row_stride,
            params.output_row_stride,
            activation_zero_point as f32,
            weight_zero_point as f32,
            output_scale,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
pub fn matmul_f32_i8_dequantize_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    activations: DevicePointer<f32>,
    weights: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    zero_points: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    reduction: usize,
    activation_row_stride: usize,
    weight_row_stride: usize,
    output_row_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(activations)?;
    checked_device_pointer(weights)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(zero_points)?;
    let params = QuantizedMatmul::create(
        rows,
        columns,
        reduction,
        activation_row_stride,
        weight_row_stride,
        output_row_stride,
    )?;
    unsafe {
        kernel_f32::matmul_f32_i8_dequantize_f32(
            out,
            activations,
            weights,
            scales,
            zero_points,
            params.rows,
            params.columns,
            params.reduction,
            params.activation_row_stride,
            params.weight_row_stride,
            params.output_row_stride,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
fn validate_per_token_group_quant(
    out: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Result<()> {
    if groups == 0 || group_size == 0 || group_size > 128 || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(groups)?;
    checked_i32_value(group_size)?;
    checked_device_pointer(out)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(input)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
fn validate_per_token_group_quant_column_major(
    out: DevicePointer<i8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<usize> {
    validate_per_token_group_quant_column_major_shape(
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    checked_device_pointer(out)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(input)?;
    checked_element_count(rows, columns / group_size)
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
fn validate_per_token_group_quant_f8(
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Result<()> {
    if groups == 0 || group_size == 0 || group_size > 128 || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(groups)?;
    checked_i32_value(group_size)?;
    checked_device_pointer(out)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(input)?;
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
fn validate_per_token_group_quant_column_major_f8(
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<usize> {
    validate_per_token_group_quant_column_major_shape(
        rows,
        columns,
        group_size,
        scale_column_stride,
        eps,
    )?;
    checked_device_pointer(out)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(input)?;
    checked_element_count(rows, columns / group_size)
}

fn validate_per_token_group_quant_column_major_shape(
    rows: usize,
    columns: usize,
    group_size: usize,
    scale_column_stride: usize,
    eps: f32,
) -> Result<()> {
    if rows == 0
        || columns == 0
        || group_size == 0
        || group_size > 128
        || !columns.is_multiple_of(group_size)
        || scale_column_stride < rows
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::InvalidLength);
    }
    checked_i32_value(rows)?;
    checked_i32_value(columns)?;
    checked_i32_value(group_size)?;
    checked_i32_value(scale_column_stride)?;
    Ok(())
}

fn validate_grouped_dequant<T>(
    out: DevicePointer<f32>,
    input: DevicePointer<T>,
    scales: DevicePointer<f32>,
    zero_points: DevicePointer<f32>,
    len: usize,
    group_size: usize,
    groups: usize,
) -> Result<()> {
    if len == 0 || group_size == 0 || groups == 0 {
        return Err(Error::InvalidLength);
    }
    ensure_grouped_dequant_shape(len, group_size, groups)?;
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(zero_points)?;
    Ok(())
}

fn ensure_grouped_dequant_shape(len: usize, group_size: usize, groups: usize) -> Result<()> {
    let expected_len = checked_element_count(group_size, groups)?;
    if len != expected_len {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
fn validate_block_quant(
    out: DevicePointer<u8>,
    scales: DevicePointer<f32>,
    input: DevicePointer<f32>,
    len: usize,
    block_size: usize,
    blocks: usize,
) -> Result<()> {
    if block_size == 0 || block_size > 128 || blocks == 0 {
        return Err(Error::InvalidLength);
    }
    let expected_blocks = len.div_ceil(block_size);
    if blocks != expected_blocks {
        return Err(Error::LengthMismatch);
    }
    checked_i32_value(block_size)?;
    checked_i32_value(len)?;
    checked_i32_value(blocks)?;
    checked_device_pointer(out)?;
    checked_device_pointer(scales)?;
    checked_device_pointer(input)?;
    Ok(())
}

fn validate_unpack_int4<T>(
    out: DevicePointer<T>,
    packed: DevicePointer<u8>,
    len: usize,
) -> Result<()> {
    if len == 0 {
        return Err(Error::InvalidLength);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(packed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{cpu::quantization as cpu_quantization, error::Result};

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn quantize_f32_to_f8e4m3_block_matches_expected_bytes() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let input_host = vec![1.0f32, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0];
        let expected_out = vec![0x7e, 0x76, 0x00, 0xf6, 0x00, 0x00, 0x00, 0x00];
        let expected_scales = vec![1.0f32 / 448.0, 1.0f32];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[8]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[2]).sync_on(&stream)?;

        quantize_f32_to_f8e4m3_block(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            8,
            4,
            2,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn dequantize_f8e4m3_block_to_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, block_size) = (3usize, 5usize, 2usize);
        let input_host = vec![
            0x38u8, 0x40, 0xb8, 0x30, 0x00, 0xc0, 0x38, 0x40, 0xb8, 0x30, 0x30, 0xb8, 0x38, 0x40,
            0xc0,
        ];
        let scales_host = vec![0.5f32, 2.0, 4.0, 1.5, 0.25, 3.0];
        let expected = cpu_quantization::dequantize_f8e4m3_block_f32(
            &input_host,
            &scales_host,
            rows,
            columns,
            block_size,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let scales = api::copy_host_vec_to_device(&Arc::new(scales_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        dequantize_f8e4m3_block_to_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            scales.device_pointer(),
            rows,
            columns,
            block_size,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-6);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn matmul_f8e4m3_f8e4m3_block_scaled_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction, group_n, group_k) = (2usize, 3usize, 4usize, 2usize, 2usize);
        let activations_host = vec![0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0x38, 0x40, 0x30];
        let weights_host = vec![
            0x38u8, 0x30, 0xb8, 0x40, 0x40, 0x38, 0x30, 0xb8, 0xb8, 0x40, 0x38, 0x30,
        ];
        let activation_scales_host = vec![0.5f32, 1.5, 2.0, 0.25];
        let weight_scales_host = vec![1.0f32, 0.5, 1.25, 2.0];
        let expected = cpu_quantization::matmul_f8e4m3_block_scaled_f32(
            &activations_host,
            &weights_host,
            &activation_scales_host,
            &weight_scales_host,
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        );

        let activations =
            api::copy_host_vec_to_device(&Arc::new(activations_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let activation_scales =
            api::copy_host_vec_to_device(&Arc::new(activation_scales_host)).sync_on(&stream)?;
        let weight_scales =
            api::copy_host_vec_to_device(&Arc::new(weight_scales_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f8e4m3_f8e4m3_block_scaled_f32(
            &stream,
            out.device_pointer(),
            activations.device_pointer(),
            weights.device_pointer(),
            activation_scales.device_pointer(),
            weight_scales.device_pointer(),
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-6);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn matmul_f8e4m3_f8e4m3_block_scaled_f32_handles_tail_groups() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction, group_n, group_k) = (2usize, 5usize, 5usize, 2usize, 3usize);
        let activations_host = vec![0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0xc0, 0x38, 0x40, 0x30, 0xb8];
        let weights_host = vec![
            0x38u8, 0x30, 0xb8, 0x40, 0xc0, 0x40, 0x38, 0x30, 0xb8, 0xc0, 0xb8, 0x40, 0x38, 0x30,
            0xc0, 0x30, 0xb8, 0x40, 0x38, 0xc0, 0xc0, 0x30, 0x38, 0x40, 0xb8,
        ];
        let activation_scales_host = vec![0.5f32, 1.5, 2.0, 0.25];
        let weight_scales_host = vec![1.0f32, 0.5, 1.25, 2.0, 0.75, 1.5];
        let expected = cpu_quantization::matmul_f8e4m3_block_scaled_f32(
            &activations_host,
            &weights_host,
            &activation_scales_host,
            &weight_scales_host,
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        );

        let activations =
            api::copy_host_vec_to_device(&Arc::new(activations_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let activation_scales =
            api::copy_host_vec_to_device(&Arc::new(activation_scales_host)).sync_on(&stream)?;
        let weight_scales =
            api::copy_host_vec_to_device(&Arc::new(weight_scales_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[rows * columns]).sync_on(&stream)?;

        matmul_f8e4m3_f8e4m3_block_scaled_f32(
            &stream,
            out.device_pointer(),
            activations.device_pointer(),
            weights.device_pointer(),
            activation_scales.device_pointer(),
            weight_scales.device_pointer(),
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-6);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8", feature = "dtype-f16"))]
    #[test]
    fn matmul_f8e4m3_f8e4m3_block_scaled_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction, group_n, group_k) = (2usize, 3usize, 4usize, 2usize, 2usize);
        let activations_host = vec![0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0x38, 0x40, 0x30];
        let weights_host = vec![
            0x38u8, 0x30, 0xb8, 0x40, 0x40, 0x38, 0x30, 0xb8, 0xb8, 0x40, 0x38, 0x30,
        ];
        let activation_scales_host = vec![0.5f32, 1.5, 2.0, 0.25];
        let weight_scales_host = vec![1.0f32, 0.5, 1.25, 2.0];
        let expected = cpu_quantization::matmul_f8e4m3_block_scaled_f32(
            &activations_host,
            &weights_host,
            &activation_scales_host,
            &weight_scales_host,
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        );

        let activations =
            api::copy_host_vec_to_device(&Arc::new(activations_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let activation_scales =
            api::copy_host_vec_to_device(&Arc::new(activation_scales_host)).sync_on(&stream)?;
        let weight_scales =
            api::copy_host_vec_to_device(&Arc::new(weight_scales_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[rows * columns]).sync_on(&stream)?;

        matmul_f8e4m3_f8e4m3_block_scaled_f16(
            &stream,
            out.device_pointer(),
            activations.device_pointer(),
            weights.device_pointer(),
            activation_scales.device_pointer(),
            weight_scales.device_pointer(),
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8", feature = "dtype-bf16"))]
    #[test]
    fn matmul_f8e4m3_f8e4m3_block_scaled_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, reduction, group_n, group_k) = (2usize, 3usize, 4usize, 2usize, 2usize);
        let activations_host = vec![0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0x38, 0x40, 0x30];
        let weights_host = vec![
            0x38u8, 0x30, 0xb8, 0x40, 0x40, 0x38, 0x30, 0xb8, 0xb8, 0x40, 0x38, 0x30,
        ];
        let activation_scales_host = vec![0.5f32, 1.5, 2.0, 0.25];
        let weight_scales_host = vec![1.0f32, 0.5, 1.25, 2.0];
        let expected = cpu_quantization::matmul_f8e4m3_block_scaled_f32(
            &activations_host,
            &weights_host,
            &activation_scales_host,
            &weight_scales_host,
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        );

        let activations =
            api::copy_host_vec_to_device(&Arc::new(activations_host)).sync_on(&stream)?;
        let weights = api::copy_host_vec_to_device(&Arc::new(weights_host)).sync_on(&stream)?;
        let activation_scales =
            api::copy_host_vec_to_device(&Arc::new(activation_scales_host)).sync_on(&stream)?;
        let weight_scales =
            api::copy_host_vec_to_device(&Arc::new(weight_scales_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[rows * columns]).sync_on(&stream)?;

        matmul_f8e4m3_f8e4m3_block_scaled_bf16(
            &stream,
            out.device_pointer(),
            activations.device_pointer(),
            weights.device_pointer(),
            activation_scales.device_pointer(),
            weight_scales.device_pointer(),
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        )?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 4e-3);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn rope_quantize_f8e4m3_f32_interleaved_matches_expected_bytes() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, heads, rope_dim, stride) = (2usize, 2usize, 4usize, 4usize);
        let input_host = vec![
            224.0f32, 112.0, -224.0, -112.0, 448.0, 224.0, 112.0, -448.0, -448.0, 224.0, 112.0,
            -224.0, 0.0, 448.0, 224.0, 112.0,
        ];
        let cos_sin_cache_host = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let pos_ids_host = vec![1u32, 0u32];
        let expected_out = vec![
            0xeeu8, 0x76, 0xf6, 0xee, 0xf6, 0x7e, 0x6e, 0xfe, 0xfe, 0x76, 0x76, 0x6e, 0x00, 0x7e,
            0xee, 0x76,
        ];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let cos_sin_cache =
            api::copy_host_vec_to_device(&Arc::new(cos_sin_cache_host)).sync_on(&stream)?;
        let pos_ids = api::copy_host_vec_to_device(&Arc::new(pos_ids_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[expected_out.len()]).sync_on(&stream)?;

        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            heads,
            rope_dim,
            stride,
            1.0,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn qk_rope_quantize_f8e4m3_single_kernel_matches_composed() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, q_heads, kv_heads, rope_dim, no_rope_dim, stride) =
            (2usize, 2usize, 1usize, 4usize, 3usize, 4usize);
        let q_rope_host = vec![
            224.0f32, 112.0, -224.0, -112.0, 448.0, 224.0, 112.0, -448.0, -448.0, 224.0, 112.0,
            -224.0, 0.0, 448.0, 224.0, 112.0,
        ];
        let k_rope_host = vec![-112.0f32, 224.0, 448.0, -224.0, 112.0, -448.0, 0.0, 224.0];
        let q_nope_host = vec![
            -896.0f32, -224.0, -112.0, 0.0, 112.0, 224.0, 448.0, 896.0, 56.0, -56.0, 336.0, -336.0,
        ];
        let k_nope_host = vec![224.0f32, -224.0, 0.0, 112.0, -112.0, 896.0];
        let cos_sin_cache_host = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let pos_ids_host = vec![1u32, 0u32];

        let q_rope = api::copy_host_vec_to_device(&Arc::new(q_rope_host)).sync_on(&stream)?;
        let k_rope = api::copy_host_vec_to_device(&Arc::new(k_rope_host)).sync_on(&stream)?;
        let q_nope = api::copy_host_vec_to_device(&Arc::new(q_nope_host)).sync_on(&stream)?;
        let k_nope = api::copy_host_vec_to_device(&Arc::new(k_nope_host)).sync_on(&stream)?;
        let cos_sin_cache =
            api::copy_host_vec_to_device(&Arc::new(cos_sin_cache_host)).sync_on(&stream)?;
        let pos_ids = api::copy_host_vec_to_device(&Arc::new(pos_ids_host)).sync_on(&stream)?;

        let fused_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let fused_k_rope = api::zeros::<u8>(&[tokens * kv_heads * rope_dim]).sync_on(&stream)?;
        let fused_q_nope = api::zeros::<u8>(&[tokens * q_heads * no_rope_dim]).sync_on(&stream)?;
        let fused_k_nope = api::zeros::<u8>(&[tokens * kv_heads * no_rope_dim]).sync_on(&stream)?;
        let expected_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let expected_k_rope = api::zeros::<u8>(&[tokens * kv_heads * rope_dim]).sync_on(&stream)?;
        let expected_q_nope =
            api::zeros::<u8>(&[tokens * q_heads * no_rope_dim]).sync_on(&stream)?;
        let expected_k_nope =
            api::zeros::<u8>(&[tokens * kv_heads * no_rope_dim]).sync_on(&stream)?;

        qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
            &stream,
            fused_q_rope.device_pointer(),
            fused_k_rope.device_pointer(),
            fused_q_nope.device_pointer(),
            fused_k_nope.device_pointer(),
            q_rope.device_pointer(),
            k_rope.device_pointer(),
            q_nope.device_pointer(),
            k_nope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            kv_heads,
            rope_dim,
            no_rope_dim,
            stride,
            1.0,
            1.0,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_q_rope.device_pointer(),
            q_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            rope_dim,
            stride,
            1.0,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_k_rope.device_pointer(),
            k_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            kv_heads,
            rope_dim,
            stride,
            1.0,
        )?;
        quantize_f32_to_f8e4m3_scaled(
            &stream,
            expected_q_nope.device_pointer(),
            q_nope.device_pointer(),
            tokens * q_heads * no_rope_dim,
            1.0,
        )?;
        quantize_f32_to_f8e4m3_scaled(
            &stream,
            expected_k_nope.device_pointer(),
            k_nope.device_pointer(),
            tokens * kv_heads * no_rope_dim,
            1.0,
        )?;

        assert_eq!(
            fused_q_rope.to_host_vec().sync_on(&stream)?,
            expected_q_rope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_k_rope.to_host_vec().sync_on(&stream)?,
            expected_k_rope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_q_nope.to_host_vec().sync_on(&stream)?,
            expected_q_nope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_k_nope.to_host_vec().sync_on(&stream)?,
            expected_k_nope.to_host_vec().sync_on(&stream)?
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn qk_rope_quantize_f8e4m3_rope_only_matches_composed() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, q_heads, kv_heads, rope_dim, stride) =
            (2usize, 2usize, 1usize, 4usize, 4usize);
        let q_scale = 0.5f32;
        let kv_scale = 2.0f32;
        let q_rope_host = vec![
            224.0f32, 112.0, -224.0, -112.0, 448.0, 224.0, 112.0, -448.0, -448.0, 224.0, 112.0,
            -224.0, 0.0, 448.0, 224.0, 112.0,
        ];
        let k_rope_host = vec![-112.0f32, 224.0, 448.0, -224.0, 112.0, -448.0, 0.0, 224.0];
        let cos_sin_cache_host = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let pos_ids_host = vec![1u32, 0u32];

        let q_rope = api::copy_host_vec_to_device(&Arc::new(q_rope_host)).sync_on(&stream)?;
        let k_rope = api::copy_host_vec_to_device(&Arc::new(k_rope_host)).sync_on(&stream)?;
        let cos_sin_cache =
            api::copy_host_vec_to_device(&Arc::new(cos_sin_cache_host)).sync_on(&stream)?;
        let pos_ids = api::copy_host_vec_to_device(&Arc::new(pos_ids_host)).sync_on(&stream)?;

        let fused_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let fused_k_rope = api::zeros::<u8>(&[tokens * kv_heads * rope_dim]).sync_on(&stream)?;
        let expected_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let expected_k_rope = api::zeros::<u8>(&[tokens * kv_heads * rope_dim]).sync_on(&stream)?;

        qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            &stream,
            fused_q_rope.device_pointer(),
            fused_k_rope.device_pointer(),
            q_rope.device_pointer(),
            k_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            kv_heads,
            rope_dim,
            stride,
            q_scale,
            kv_scale,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_q_rope.device_pointer(),
            q_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            rope_dim,
            stride,
            q_scale,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_k_rope.device_pointer(),
            k_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            kv_heads,
            rope_dim,
            stride,
            kv_scale,
        )?;

        assert_eq!(
            fused_q_rope.to_host_vec().sync_on(&stream)?,
            expected_q_rope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_k_rope.to_host_vec().sync_on(&stream)?,
            expected_k_rope.to_host_vec().sync_on(&stream)?
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn qk_rope_quantize_f8e4m3_single_kernel_matches_mla_composed() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, q_heads, rope_dim, no_rope_dim, stride) =
            (2usize, 3usize, 4usize, 5usize, 4usize);
        let q_scale = 0.5f32;
        let kv_scale = 2.0f32;
        let q_rope_host = vec![
            224.0f32, 112.0, -224.0, -112.0, 448.0, 224.0, 112.0, -448.0, -448.0, 224.0, 112.0,
            -224.0, 0.0, 448.0, 224.0, 112.0, -112.0, 56.0, -56.0, 112.0, 336.0, -336.0, 224.0,
            -224.0,
        ];
        let k_rope_host = vec![-112.0f32, 224.0, 448.0, -224.0, 112.0, -448.0, 0.0, 224.0];
        let q_nope_host = vec![
            -896.0f32, -224.0, -112.0, 0.0, 112.0, 224.0, 448.0, 896.0, 56.0, -56.0, 336.0, -336.0,
            672.0, -672.0, 14.0, -14.0, 28.0, -28.0, 42.0, -42.0, 84.0, -84.0, 168.0, -168.0,
            252.0, -252.0, 504.0, -504.0, 1008.0, -1008.0,
        ];
        let k_nope_host = vec![
            224.0f32, -224.0, 0.0, 112.0, -112.0, 896.0, -896.0, 56.0, -56.0, 28.0,
        ];
        let cos_sin_cache_host = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let pos_ids_host = vec![1u32, 0u32];

        let q_rope = api::copy_host_vec_to_device(&Arc::new(q_rope_host)).sync_on(&stream)?;
        let k_rope = api::copy_host_vec_to_device(&Arc::new(k_rope_host)).sync_on(&stream)?;
        let q_nope = api::copy_host_vec_to_device(&Arc::new(q_nope_host)).sync_on(&stream)?;
        let k_nope = api::copy_host_vec_to_device(&Arc::new(k_nope_host)).sync_on(&stream)?;
        let cos_sin_cache =
            api::copy_host_vec_to_device(&Arc::new(cos_sin_cache_host)).sync_on(&stream)?;
        let pos_ids = api::copy_host_vec_to_device(&Arc::new(pos_ids_host)).sync_on(&stream)?;

        let fused_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let fused_k_rope = api::zeros::<u8>(&[tokens * rope_dim]).sync_on(&stream)?;
        let fused_q_nope = api::zeros::<u8>(&[tokens * q_heads * no_rope_dim]).sync_on(&stream)?;
        let fused_k_nope = api::zeros::<u8>(&[tokens * no_rope_dim]).sync_on(&stream)?;
        let expected_q_rope = api::zeros::<u8>(&[tokens * q_heads * rope_dim]).sync_on(&stream)?;
        let expected_k_rope = api::zeros::<u8>(&[tokens * rope_dim]).sync_on(&stream)?;
        let expected_q_nope =
            api::zeros::<u8>(&[tokens * q_heads * no_rope_dim]).sync_on(&stream)?;
        let expected_k_nope = api::zeros::<u8>(&[tokens * no_rope_dim]).sync_on(&stream)?;

        qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
            &stream,
            fused_q_rope.device_pointer(),
            fused_k_rope.device_pointer(),
            fused_q_nope.device_pointer(),
            fused_k_nope.device_pointer(),
            q_rope.device_pointer(),
            k_rope.device_pointer(),
            q_nope.device_pointer(),
            k_nope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            1,
            rope_dim,
            no_rope_dim,
            stride,
            q_scale,
            kv_scale,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_q_rope.device_pointer(),
            q_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            q_heads,
            rope_dim,
            stride,
            q_scale,
        )?;
        rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            expected_k_rope.device_pointer(),
            k_rope.device_pointer(),
            cos_sin_cache.device_pointer(),
            pos_ids.device_pointer(),
            tokens,
            1,
            rope_dim,
            stride,
            kv_scale,
        )?;
        quantize_f32_to_f8e4m3_scaled(
            &stream,
            expected_q_nope.device_pointer(),
            q_nope.device_pointer(),
            tokens * q_heads * no_rope_dim,
            q_scale,
        )?;
        quantize_f32_to_f8e4m3_scaled(
            &stream,
            expected_k_nope.device_pointer(),
            k_nope.device_pointer(),
            tokens * no_rope_dim,
            kv_scale,
        )?;

        assert_eq!(
            fused_q_rope.to_host_vec().sync_on(&stream)?,
            expected_q_rope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_k_rope.to_host_vec().sync_on(&stream)?,
            expected_k_rope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_q_nope.to_host_vec().sync_on(&stream)?,
            expected_q_nope.to_host_vec().sync_on(&stream)?
        );
        assert_eq!(
            fused_k_nope.to_host_vec().sync_on(&stream)?,
            expected_k_nope.to_host_vec().sync_on(&stream)?
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn quantize_f32_to_f8e4m3_scaled_matches_expected_bytes() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let input_host = vec![-224.0f32, -112.0, 0.0, 112.0, 224.0, 896.0, -896.0];
        let expected_out = vec![0xf6u8, 0xee, 0x00, 0x6e, 0x76, 0x7e, 0xfe];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[expected_out.len()]).sync_on(&stream)?;

        quantize_f32_to_f8e4m3_scaled(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            expected_out.len(),
            1.0,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
    #[test]
    fn per_token_group_quant_i8_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let group_size = 5usize;
        let eps = 1e-10f32;
        let input_host = vec![
            -127.0f32, -64.0, 0.0, 64.0, 127.0, 127.0, 1.5, 2.5, -1.5, -2.5, 127.0, -127.0, 32.0,
            -32.0, 0.0,
        ];
        let (expected_out, expected_scales) =
            cpu_quantization::per_token_group_quant_i8_f32(&input_host, group_size, eps);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<i8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_i8_f32(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            expected_scales.len(),
            group_size,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
    #[test]
    fn per_token_group_quant_i8_f32_column_major_scales() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, group_size, scale_column_stride) = (2usize, 5usize, 5usize, 3usize);
        let eps = 1e-10f32;
        let input_host = vec![127.0f32, 1.5, 2.5, -1.5, -2.5, -127.0, 1.5, 2.5, -1.5, -2.5];
        let expected_out = vec![127i8, 2, 2, -2, -2, -127, 2, 2, -2, -2];
        let expected_scales = vec![1.0f32, 1.0, 0.0];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<i8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_i8_f32_column_major_scales(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            rows,
            columns,
            group_size,
            scale_column_stride,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
    #[test]
    fn per_token_group_quant_i8_f32_column_major_ue8m0_scales() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, group_size, scale_column_stride) = (2usize, 5usize, 5usize, 3usize);
        let eps = 1e-10f32;
        let input_host = vec![127.0f32, 1.5, 2.5, -1.5, -2.5, -127.0, 1.5, 2.5, -1.5, -2.5];
        let expected_out = vec![127i8, 2, 2, -2, -2, -127, 2, 2, -2, -2];
        let expected_scales = vec![1.0f32, 1.0, 0.0];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<i8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_i8_f32_column_major_ue8m0_scales(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            rows,
            columns,
            group_size,
            scale_column_stride,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn per_token_group_quant_f8e4m3_f32_matches_expected_bytes() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let group_size = 5usize;
        let eps = 1e-10f32;
        let input_host = vec![
            -448.0f32, -224.0, 0.0, 224.0, 448.0, 0.0, 0.0, 0.0, 0.0, 0.0, 448.0, -448.0, 112.0,
            -112.0, 0.0,
        ];
        let expected_out = vec![
            0xfeu8, 0xf6, 0x00, 0x76, 0x7e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7e, 0xfe, 0x6e, 0xee,
            0x00,
        ];
        let expected_scales = vec![1.0f32, eps / 448.0, 1.0f32];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_f8e4m3_f32(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            expected_scales.len(),
            group_size,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn per_token_group_quant_f8e4m3_f32_column_major_scales_matches_expected_bytes() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, group_size, scale_column_stride) = (2usize, 6usize, 3usize, 3usize);
        let eps = 1e-10f32;
        let input_host = vec![
            -448.0f32, 0.0, 448.0, 224.0, -224.0, 0.0, 0.0, 0.0, 0.0, 112.0, -112.0, 0.0,
        ];
        let expected_out = vec![
            0xfeu8, 0x00, 0x7e, 0x7e, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x7e, 0xfe, 0x00,
        ];
        let expected_scales = vec![1.0f32, eps / 448.0, 0.0, 0.5, 0.25, 0.0];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_f8e4m3_f32_column_major_scales(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            rows,
            columns,
            group_size,
            scale_column_stride,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-f8"))]
    #[test]
    fn per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales_matches_expected_bytes()
    -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (rows, columns, group_size, scale_column_stride) = (2usize, 6usize, 3usize, 3usize);
        let eps = 1e-10f32;
        let input_host = vec![
            672.0f32, 448.0, -448.0, 448.0, -448.0, 0.0, 224.0, -224.0, 0.0, 112.0, -112.0, 0.0,
        ];
        let expected_out = vec![
            0x7au8, 0x76, 0xf6, 0x7e, 0xfe, 0x00, 0x7e, 0xfe, 0x00, 0x7e, 0xfe, 0x00,
        ];
        let expected_scales = vec![2.0f32, 0.5, 0.0, 1.0, 0.25, 0.0];

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let out = api::zeros::<u8>(&[expected_out.len()]).sync_on(&stream)?;
        let scales = api::zeros::<f32>(&[expected_scales.len()]).sync_on(&stream)?;

        per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
            &stream,
            out.device_pointer(),
            scales.device_pointer(),
            input.device_pointer(),
            rows,
            columns,
            group_size,
            scale_column_stride,
            eps,
        )?;

        assert_eq!(out.to_host_vec().sync_on(&stream)?, expected_out);
        singe_core::assert_close!(
            &scales.to_host_vec().sync_on(&stream)?,
            &expected_scales,
            1e-8,
        );
        Ok(())
    }
}
