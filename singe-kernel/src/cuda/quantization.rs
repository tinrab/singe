//! Quantize, dequantize, int4 unpacking, fp8 helpers, and quantized matmul.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::{
        autotune::{self, KernelCompileConfig, TuneOptions, TuneReport, TuningChoice},
        interop::input_pointer,
        utility::{
            VectorConfig, VectorTuning, resolve_vector_tuning, validate_vector_lengths,
            vector_tile_candidates, vector_tune_key,
        },
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value, ensure_len},
};

/// Layout of the float32 dequantization scales emitted beside FP8 values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ScaleLayout {
    RowMajor,
    ColumnMajor { column_stride: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8ScaleRounding {
    Linear,
    PowerOfTwo,
}

/// Contiguous activation rows, quantized in channel groups of 32, 64, or 128.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8GroupQuantConfig {
    pub rows: usize,
    pub columns: usize,
    pub group_size: usize,
    pub eps: f32,
    pub layout: Fp8ScaleLayout,
    pub rounding: Fp8ScaleRounding,
}

impl Fp8GroupQuantConfig {
    pub fn create(rows: usize, columns: usize, group_size: usize) -> Result<Self> {
        let config = Self {
            rows,
            columns,
            group_size,
            eps: 1e-4,
            layout: Fp8ScaleLayout::RowMajor,
            rounding: Fp8ScaleRounding::Linear,
        };
        config.scale_len()?;
        Ok(config)
    }

    pub fn scale_len(self) -> Result<usize> {
        if self.columns == 0
            || ![32, 64, 128].contains(&self.group_size)
            || !self.columns.is_multiple_of(self.group_size)
            || !self.eps.is_finite()
            || self.eps <= 0.0
        {
            return Err(Error::InvalidLength);
        }
        checked_i32_value(checked_element_count(self.rows, self.columns)?)?;
        if self.rows == 0 {
            return Ok(0);
        }
        let groups = self.columns / self.group_size;
        let len = match self.layout {
            Fp8ScaleLayout::RowMajor => checked_element_count(self.rows, groups)?,
            Fp8ScaleLayout::ColumnMajor { column_stride } => {
                if column_stride < self.rows {
                    return Err(Error::InvalidLength);
                }
                (groups - 1)
                    .checked_mul(column_stride)
                    .and_then(|n| n.checked_add(self.rows))
                    .ok_or(Error::SizeOverflow)?
            }
        };
        checked_i32_value(len)?;
        Ok(len)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int4Packing {
    LowFirst,
    HighFirst,
}

impl Int4Packing {
    pub(crate) const fn kernel_high_first(self) -> i32 {
        match self {
            Self::LowFirst => 0,
            Self::HighFirst => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizedMatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub activation_row_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
    pub activation_zero_point: i32,
    pub weight_zero_point: i32,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DequantizedWeightMatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub activation_row_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScaledFp8MatmulConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub group_n: usize,
    pub group_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Per-token group quantization with round-to-nearest-even int8 conversion.
///
/// Each contiguous group uses `scale = max(max(abs(group)), eps) / 127`.
/// Values are divided by that scale, clamped to `[-128, 127]`, and rounded to nearest even before conversion to `i8`.
pub struct PerTokenGroupQuantConfig {
    pub group_size: usize,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Column-major scale layout variant of per-token group quantization.
///
/// Quantized values are written in row-major input order. Scales are stored at
/// `group * scale_column_stride + row`.
/// Int8 values use round-to-nearest-even.
pub struct PerTokenGroupQuantColumnMajorConfig {
    pub rows: usize,
    pub columns: usize,
    pub group_size: usize,
    pub scale_column_stride: usize,
    pub eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeQuantizeFp8Config {
    pub tokens: usize,
    pub heads: usize,
    pub rope_dim: usize,
    pub cos_sin_cache_stride: usize,
    pub quant_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Interleaved Q/K `RoPE` FP8 quantization layout.
///
/// Q and K `RoPE` channels are rotated from `cos_sin_cache` using `pos_ids`,
/// scaled independently, clamped to the finite f8e4m3 range, and converted to
/// byte-backed FP8 output. Optional no-RoPE channels are scaled and quantized without rotation.
pub struct QkRopeQuantizeFp8Config {
    pub tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub rope_dim: usize,
    pub no_rope_dim: usize,
    pub cos_sin_cache_stride: usize,
    pub quant_scale_q: f32,
    pub quant_scale_kv: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// MLA variant of interleaved Q/K `RoPE` FP8 quantization.
///
/// This layout has multiple query heads and a single shared K head. `RoPE`
/// channels are rotated and quantized like [`QkRopeQuantizeFp8Config`], while
/// optional no-RoPE channels are scaled directly.
pub struct MlaRopeQuantizeFp8Config {
    pub tokens: usize,
    pub q_heads: usize,
    pub rope_dim: usize,
    pub no_rope_dim: usize,
    pub cos_sin_cache_stride: usize,
    pub quant_scale_q: f32,
    pub quant_scale_kv: f32,
}

impl DequantizedWeightMatmulConfig {
    #[must_use]
    pub fn row_major(rows: usize, columns: usize, reduction: usize) -> Self {
        Self {
            rows,
            columns,
            reduction,
            activation_row_stride: reduction,
            weight_row_stride: reduction,
            output_row_stride: columns,
        }
    }
}

impl BlockScaledFp8MatmulConfig {
    #[must_use]
    pub fn create(
        rows: usize,
        columns: usize,
        reduction: usize,
        group_n: usize,
        group_k: usize,
    ) -> Self {
        Self {
            rows,
            columns,
            reduction,
            group_n,
            group_k,
        }
    }
}

impl PerTokenGroupQuantConfig {
    #[must_use]
    pub fn create(group_size: usize, eps: f32) -> Self {
        Self { group_size, eps }
    }
}

impl PerTokenGroupQuantColumnMajorConfig {
    #[must_use]
    pub fn create(
        rows: usize,
        columns: usize,
        group_size: usize,
        scale_column_stride: usize,
        eps: f32,
    ) -> Self {
        Self {
            rows,
            columns,
            group_size,
            scale_column_stride,
            eps,
        }
    }
}

impl RopeQuantizeFp8Config {
    #[must_use]
    pub fn interleaved(
        tokens: usize,
        heads: usize,
        rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale: f32,
    ) -> Self {
        Self {
            tokens,
            heads,
            rope_dim,
            cos_sin_cache_stride,
            quant_scale,
        }
    }
}

impl QkRopeQuantizeFp8Config {
    #[must_use]
    pub fn row_major_interleaved(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        rope_dim: usize,
        no_rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale_q: f32,
        quant_scale_kv: f32,
    ) -> Self {
        Self {
            tokens,
            q_heads,
            kv_heads,
            rope_dim,
            no_rope_dim,
            cos_sin_cache_stride,
            quant_scale_q,
            quant_scale_kv,
        }
    }

    #[must_use]
    pub fn row_major_interleaved_rope_only(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale_q: f32,
        quant_scale_kv: f32,
    ) -> Self {
        Self {
            tokens,
            q_heads,
            kv_heads,
            rope_dim,
            no_rope_dim: 0,
            cos_sin_cache_stride,
            quant_scale_q,
            quant_scale_kv,
        }
    }
}

impl MlaRopeQuantizeFp8Config {
    #[must_use]
    pub fn row_major_interleaved(
        tokens: usize,
        q_heads: usize,
        rope_dim: usize,
        no_rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale_q: f32,
        quant_scale_kv: f32,
    ) -> Self {
        Self {
            tokens,
            q_heads,
            rope_dim,
            no_rope_dim,
            cos_sin_cache_stride,
            quant_scale_q,
            quant_scale_kv,
        }
    }

    #[must_use]
    pub fn row_major_interleaved_rope_only(
        tokens: usize,
        q_heads: usize,
        rope_dim: usize,
        cos_sin_cache_stride: usize,
        quant_scale_q: f32,
        quant_scale_kv: f32,
    ) -> Self {
        Self {
            tokens,
            q_heads,
            rope_dim,
            no_rope_dim: 0,
            cos_sin_cache_stride,
            quant_scale_q,
            quant_scale_kv,
        }
    }
}

impl QuantizedMatmulConfig {
    #[must_use]
    pub fn row_major(
        rows: usize,
        columns: usize,
        reduction: usize,
        activation_zero_point: i32,
        weight_zero_point: i32,
        output_scale: f32,
    ) -> Self {
        Self {
            rows,
            columns,
            reduction,
            activation_row_stride: reduction,
            weight_row_stride: reduction,
            output_row_stride: columns,
            activation_zero_point,
            weight_zero_point,
            output_scale,
        }
    }
}

fn vector_launch_parts(tuning: Option<VectorTuning>) -> (Option<usize>, KernelCompileConfig) {
    tuning.map_or_else(
        || (None, KernelCompileConfig::default()),
        |tuning| (Some(tuning.tile_width), tuning.compile),
    )
}

fn launch_quantize_vector<F>(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: VectorConfig,
    mut launch: F,
) -> Result<()>
where
    F: FnMut(Option<VectorTuning>) -> Result<()>,
{
    if config.length == 0 {
        return Ok(());
    }
    match config.tuning {
        TuningChoice::Default => return launch(None),
        TuningChoice::Explicit(tuning) => return launch(Some(tuning)),
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = vector_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_vector_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        vector_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch(Some(*tuning))
            })
        },
    )?;
    launch(tuning)
}

fn tune_quantize_vector<F>(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: VectorConfig,
    options: TuneOptions,
    mut launch: F,
) -> Result<TuneReport<VectorTuning>>
where
    F: FnMut(VectorTuning) -> Result<()>,
{
    let key = vector_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        vector_tile_candidates(),
        |tuning| launch(*tuning),
    )?;
    autotune::cache().insert(key, report.best);
    launch(report.best)?;
    Ok(report)
}

macro_rules! dequantize_fn {
    ($name:ident, $out:ty, $input:ty) => {
        /// Writes `scale * (input - zero_point)` for every input element.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out>,
            input: &impl DeviceInput<$input>,
            scale: $out,
            zero_point: $out,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::quantization::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    scale,
                    zero_point,
                    len,
                )
            }
        }
    };
}

macro_rules! quantize_fn {
    ($name:ident, $tune_name:ident, $out:ty, $input:ty) => {
        /// Scales each input by `1 / scale`, adds `zero_point`, and converts to the output type.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out>,
            input: &impl DeviceInput<$input>,
            scale: $input,
            zero_point: $input,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_quantize_vector(
                stream,
                stringify!($name),
                stringify!($input),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::quantization::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            scale,
                            zero_point,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Tunes elementwise quantization using the supplied output storage.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$out>,
            input: &impl DeviceInput<$input>,
            scale: $input,
            zero_point: $input,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_quantize_vector(
                stream,
                stringify!($name),
                stringify!($input),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::quantization::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        scale,
                        zero_point,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
dequantize_fn!(dequantize_u8_to_f16, f16, u8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
dequantize_fn!(dequantize_i8_to_f16, f16, i8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
dequantize_fn!(dequantize_u8_to_f32, f32, u8);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
dequantize_fn!(dequantize_i8_to_f32, f32, i8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
dequantize_fn!(dequantize_u8_to_f64, f64, u8);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
dequantize_fn!(dequantize_i8_to_f64, f64, i8);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
quantize_fn!(quantize_f16_to_u8, tune_quantize_f16_to_u8, u8, f16);

#[cfg(all(feature = "dtype_f16", feature = "dtype_i8"))]
quantize_fn!(quantize_f16_to_i8, tune_quantize_f16_to_i8, i8, f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
quantize_fn!(quantize_f32_to_u8, tune_quantize_f32_to_u8, u8, f32);

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
quantize_fn!(quantize_f32_to_i8, tune_quantize_f32_to_i8, i8, f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
quantize_fn!(quantize_f64_to_u8, tune_quantize_f64_to_u8, u8, f64);

#[cfg(all(feature = "dtype_f64", feature = "dtype_i8"))]
quantize_fn!(quantize_f64_to_i8, tune_quantize_f64_to_i8, i8, f64);

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn quantize_f32_to_f8e4m3_block(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    block_size: usize,
) -> Result<()> {
    let len = input.len();
    if len == 0 {
        ensure_len(out.len(), 0)?;
        return Ok(());
    }
    let blocks = validate_block_quant(out.len(), scales.len(), len, block_size)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::quantize_f32_to_f8e4m3_block(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            len,
            block_size,
            blocks,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Reconstructs floating-point values using the configured quantization scales and zero points.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn dequantize_f8e4m3_block_to_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<u8>,
    scales: &impl DeviceInput<f32>,
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Result<()> {
    validate_block_dequant(
        out.len(),
        input.len(),
        scales.len(),
        rows,
        columns,
        block_size,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::dequantize_f8e4m3_block_to_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(scales),
            rows,
            columns,
            block_size,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Applies rotary position encoding and quantizes the selected query and key ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn rope_quantize_f8e4m3_f32_interleaved(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<f32>,
    cos_sin_cache: &impl DeviceInput<f32>,
    pos_ids: &impl DeviceInput<u32>,
    config: RopeQuantizeFp8Config,
) -> Result<()> {
    validate_rope_quantize_fp8(
        out.len(),
        input.len(),
        cos_sin_cache.len(),
        pos_ids.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::rope_quantize_f8e4m3_f32_interleaved(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(cos_sin_cache),
            input_pointer(pos_ids),
            config.tokens,
            config.heads,
            config.rope_dim,
            config.cos_sin_cache_stride,
            config.quant_scale,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn quantize_f32_to_f8e4m3_scaled(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<f32>,
    quant_scale: f32,
) -> Result<()> {
    ensure_len(out.len(), input.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::quantize_f32_to_f8e4m3_scaled(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input.len(),
            quant_scale,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Rotates and quantizes Q/K `RoPE` channels, plus optional no-RoPE channels.
///
/// Inputs and outputs are row-major flattened tensors.
/// Q uses `q_heads` and K uses `kv_heads`.
/// When `no_rope_dim` is zero, this launches the RoPE-only kernel and ignores the no-RoPE buffers after length validation.
/// Applies rotary position encoding and quantizes the selected query and key ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
pub unsafe fn qk_rope_quantize_f8e4m3_f32_interleaved(
    stream: &Stream,
    q_rope_out: &mut impl DeviceOutput<u8>,
    k_rope_out: &mut impl DeviceOutput<u8>,
    q_nope_out: &mut impl DeviceOutput<u8>,
    k_nope_out: &mut impl DeviceOutput<u8>,
    q_rope: &impl DeviceInput<f32>,
    k_rope: &impl DeviceInput<f32>,
    q_nope: &impl DeviceInput<f32>,
    k_nope: &impl DeviceInput<f32>,
    cos_sin_cache: &impl DeviceInput<f32>,
    pos_ids: &impl DeviceInput<u32>,
    config: QkRopeQuantizeFp8Config,
) -> Result<()> {
    validate_qk_rope_quantize_fp8(
        q_rope_out.len(),
        k_rope_out.len(),
        q_nope_out.len(),
        k_nope_out.len(),
        q_rope.len(),
        k_rope.len(),
        q_nope.len(),
        k_nope.len(),
        cos_sin_cache.len(),
        pos_ids.len(),
        config,
    )?;
    if config.no_rope_dim != 0 {
        let stream = stream.to_cuda_core()?;
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe {
            cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
                &stream,
                DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(q_nope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(k_nope_out.as_mut_ptr() as _),
                input_pointer(q_rope),
                input_pointer(k_rope),
                input_pointer(q_nope),
                input_pointer(k_nope),
                input_pointer(cos_sin_cache),
                input_pointer(pos_ids),
                config.tokens,
                config.q_heads,
                config.kv_heads,
                config.rope_dim,
                config.no_rope_dim,
                config.cos_sin_cache_stride,
                config.quant_scale_q,
                config.quant_scale_kv,
            )
        };
    }
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            &stream,
            DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
            input_pointer(q_rope),
            input_pointer(k_rope),
            input_pointer(cos_sin_cache),
            input_pointer(pos_ids),
            config.tokens,
            config.q_heads,
            config.kv_heads,
            config.rope_dim,
            config.cos_sin_cache_stride,
            config.quant_scale_q,
            config.quant_scale_kv,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// RoPE-only form of [`qk_rope_quantize_f8e4m3_f32_interleaved`].
/// Applies rotary position encoding and quantizes the selected query and key ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
    stream: &Stream,
    q_rope_out: &mut impl DeviceOutput<u8>,
    k_rope_out: &mut impl DeviceOutput<u8>,
    q_rope: &impl DeviceInput<f32>,
    k_rope: &impl DeviceInput<f32>,
    cos_sin_cache: &impl DeviceInput<f32>,
    pos_ids: &impl DeviceInput<u32>,
    config: QkRopeQuantizeFp8Config,
) -> Result<()> {
    validate_qk_rope_quantize_fp8_rope_only(
        q_rope_out.len(),
        k_rope_out.len(),
        q_rope.len(),
        k_rope.len(),
        cos_sin_cache.len(),
        pos_ids.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            &stream,
            DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
            input_pointer(q_rope),
            input_pointer(k_rope),
            input_pointer(cos_sin_cache),
            input_pointer(pos_ids),
            config.tokens,
            config.q_heads,
            config.kv_heads,
            config.rope_dim,
            config.cos_sin_cache_stride,
            config.quant_scale_q,
            config.quant_scale_kv,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// MLA form of interleaved Q/K `RoPE` FP8 quantization.
///
/// K is treated as a single shared head while Q keeps `q_heads`.
/// Applies rotary position encoding and quantizes the selected query and key ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
pub unsafe fn mla_rope_quantize_f8e4m3_f32_interleaved(
    stream: &Stream,
    q_rope_out: &mut impl DeviceOutput<u8>,
    k_rope_out: &mut impl DeviceOutput<u8>,
    q_nope_out: &mut impl DeviceOutput<u8>,
    k_nope_out: &mut impl DeviceOutput<u8>,
    q_rope: &impl DeviceInput<f32>,
    k_rope: &impl DeviceInput<f32>,
    q_nope: &impl DeviceInput<f32>,
    k_nope: &impl DeviceInput<f32>,
    cos_sin_cache: &impl DeviceInput<f32>,
    pos_ids: &impl DeviceInput<u32>,
    config: MlaRopeQuantizeFp8Config,
) -> Result<()> {
    validate_mla_rope_quantize_fp8(
        q_rope_out.len(),
        k_rope_out.len(),
        q_nope_out.len(),
        k_nope_out.len(),
        q_rope.len(),
        k_rope.len(),
        q_nope.len(),
        k_nope.len(),
        cos_sin_cache.len(),
        pos_ids.len(),
        config,
    )?;
    if config.no_rope_dim != 0 {
        let stream = stream.to_cuda_core()?;
        // SAFETY: the caller upholds this function's safety contract.
        return unsafe {
            cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_single_kernel(
                &stream,
                DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(q_nope_out.as_mut_ptr() as _),
                DevicePointer::from_cu_deviceptr(k_nope_out.as_mut_ptr() as _),
                input_pointer(q_rope),
                input_pointer(k_rope),
                input_pointer(q_nope),
                input_pointer(k_nope),
                input_pointer(cos_sin_cache),
                input_pointer(pos_ids),
                config.tokens,
                config.q_heads,
                1,
                config.rope_dim,
                config.no_rope_dim,
                config.cos_sin_cache_stride,
                config.quant_scale_q,
                config.quant_scale_kv,
            )
        };
    }
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            &stream,
            DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
            input_pointer(q_rope),
            input_pointer(k_rope),
            input_pointer(cos_sin_cache),
            input_pointer(pos_ids),
            config.tokens,
            config.q_heads,
            1,
            config.rope_dim,
            config.cos_sin_cache_stride,
            config.quant_scale_q,
            config.quant_scale_kv,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// RoPE-only form of [`mla_rope_quantize_f8e4m3_f32_interleaved`].
/// Applies rotary position encoding and quantizes the selected query and key ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn mla_rope_quantize_f8e4m3_f32_interleaved_rope_only(
    stream: &Stream,
    q_rope_out: &mut impl DeviceOutput<u8>,
    k_rope_out: &mut impl DeviceOutput<u8>,
    q_rope: &impl DeviceInput<f32>,
    k_rope: &impl DeviceInput<f32>,
    cos_sin_cache: &impl DeviceInput<f32>,
    pos_ids: &impl DeviceInput<u32>,
    config: MlaRopeQuantizeFp8Config,
) -> Result<()> {
    validate_mla_rope_quantize_fp8_rope_only(
        q_rope_out.len(),
        k_rope_out.len(),
        q_rope.len(),
        k_rope.len(),
        cos_sin_cache.len(),
        pos_ids.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::qk_rope_quantize_f8e4m3_f32_interleaved_rope_only(
            &stream,
            DevicePointer::from_cu_deviceptr(q_rope_out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(k_rope_out.as_mut_ptr() as _),
            input_pointer(q_rope),
            input_pointer(k_rope),
            input_pointer(cos_sin_cache),
            input_pointer(pos_ids),
            config.tokens,
            config.q_heads,
            1,
            config.rope_dim,
            config.cos_sin_cache_stride,
            config.quant_scale_q,
            config.quant_scale_kv,
        )
    }
}

#[cfg(feature = "dtype_u8")]
/// Unpacks each 4-bit input value into its own output byte.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn unpack_u4_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    packed: &impl DeviceInput<u8>,
    packing: Int4Packing,
) -> Result<()> {
    let len = validate_unpack_int4(out.len(), packed.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::unpack_u4_u8(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(packed),
            len,
            packing,
        )
    }
}

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
/// Unpacks each 4-bit input value into its own output byte.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn unpack_i4_i8(
    stream: &Stream,
    out: &mut impl DeviceOutput<i8>,
    packed: &impl DeviceInput<u8>,
    packing: Int4Packing,
) -> Result<()> {
    let len = validate_unpack_int4(out.len(), packed.len())?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::unpack_i4_i8(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(packed),
            len,
            packing,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
/// Reconstructs floating-point values using the configured quantization scales and zero points.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn dequantize_u8_f32_per_channel(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<u8>,
    scales: &impl DeviceInput<f32>,
    zero_points: &impl DeviceInput<f32>,
    channels: usize,
    channel_size: usize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        dequantize_u8_f32_grouped(
            stream,
            out,
            input,
            scales,
            zero_points,
            channel_size,
            channels,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Reconstructs floating-point values using the configured quantization scales and zero points.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn dequantize_i8_f32_per_channel(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<i8>,
    scales: &impl DeviceInput<f32>,
    zero_points: &impl DeviceInput<f32>,
    channels: usize,
    channel_size: usize,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        dequantize_i8_f32_grouped(
            stream,
            out,
            input,
            scales,
            zero_points,
            channel_size,
            channels,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
/// Reconstructs floating-point values using the configured quantization scales and zero points.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn dequantize_u8_f32_grouped(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<u8>,
    scales: &impl DeviceInput<f32>,
    zero_points: &impl DeviceInput<f32>,
    group_size: usize,
    groups: usize,
) -> Result<()> {
    let len = validate_grouped_dequant(
        out.len(),
        input.len(),
        scales.len(),
        zero_points.len(),
        group_size,
        groups,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::dequantize_u8_f32_grouped(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(scales),
            input_pointer(zero_points),
            len,
            group_size,
            groups,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Reconstructs floating-point values using the configured quantization scales and zero points.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn dequantize_i8_f32_grouped(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<i8>,
    scales: &impl DeviceInput<f32>,
    zero_points: &impl DeviceInput<f32>,
    group_size: usize,
    groups: usize,
) -> Result<()> {
    let len = validate_grouped_dequant(
        out.len(),
        input.len(),
        scales.len(),
        zero_points.len(),
        group_size,
        groups,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::dequantize_i8_f32_grouped(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            input_pointer(scales),
            input_pointer(zero_points),
            len,
            group_size,
            groups,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_i8_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<i8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantConfig,
) -> Result<()> {
    let groups = validate_per_token_group_quant(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_i8_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            groups,
            config.group_size,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_i8_f32_column_major_scales(
    stream: &Stream,
    out: &mut impl DeviceOutput<i8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantColumnMajorConfig,
) -> Result<()> {
    validate_per_token_group_quant_column_major(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_i8_f32_column_major_scales(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            config.rows,
            config.columns,
            config.group_size,
            config.scale_column_stride,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_i8_f32_column_major_ue8m0_scales(
    stream: &Stream,
    out: &mut impl DeviceOutput<i8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantColumnMajorConfig,
) -> Result<()> {
    validate_per_token_group_quant_column_major(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_i8_f32_column_major_ue8m0_scales(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            config.rows,
            config.columns,
            config.group_size,
            config.scale_column_stride,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_f8e4m3_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantConfig,
) -> Result<()> {
    let groups = validate_per_token_group_quant(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_f8e4m3_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            groups,
            config.group_size,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_f8e4m3_f32_column_major_scales(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantColumnMajorConfig,
) -> Result<()> {
    validate_per_token_group_quant_column_major(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_f8e4m3_f32_column_major_scales(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            config.rows,
            config.columns,
            config.group_size,
            config.scale_column_stride,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Quantizes input values and writes the configured output and scale ranges.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    scales: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: PerTokenGroupQuantColumnMajorConfig,
) -> Result<()> {
    validate_per_token_group_quant_column_major(out.len(), scales.len(), input.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::per_token_group_quant_f8e4m3_f32_column_major_ue8m0_scales(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(scales.as_mut_ptr() as _),
            input_pointer(input),
            config.rows,
            config.columns,
            config.group_size,
            config.scale_column_stride,
            config.eps,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Computes a matrix product from quantized inputs using the configured scales.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn matmul_i8_i8_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    activations: &impl DeviceInput<i8>,
    weights: &impl DeviceInput<i8>,
    config: QuantizedMatmulConfig,
) -> Result<()> {
    validate_quantized_matmul(out.len(), activations.len(), weights.len(), config)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::matmul_i8_i8_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(activations),
            input_pointer(weights),
            config.rows,
            config.columns,
            config.reduction,
            config.activation_row_stride,
            config.weight_row_stride,
            config.output_row_stride,
            config.activation_zero_point,
            config.weight_zero_point,
            config.output_scale,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_i8"))]
/// Computes a matrix product from quantized inputs using the configured scales.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn matmul_f32_i8_dequantize_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    activations: &impl DeviceInput<f32>,
    weights: &impl DeviceInput<i8>,
    scales: &impl DeviceInput<f32>,
    zero_points: &impl DeviceInput<f32>,
    config: DequantizedWeightMatmulConfig,
) -> Result<()> {
    validate_dequantized_weight_matmul(
        out.len(),
        activations.len(),
        weights.len(),
        scales.len(),
        zero_points.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::matmul_f32_i8_dequantize_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(activations),
            input_pointer(weights),
            input_pointer(scales),
            input_pointer(zero_points),
            config.rows,
            config.columns,
            config.reduction,
            config.activation_row_stride,
            config.weight_row_stride,
            config.output_row_stride,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
/// Computes a matrix product from quantized inputs using the configured scales.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    activations: &impl DeviceInput<u8>,
    weights: &impl DeviceInput<u8>,
    activation_scales: &impl DeviceInput<f32>,
    weight_scales: &impl DeviceInput<f32>,
    config: BlockScaledFp8MatmulConfig,
) -> Result<()> {
    validate_block_scaled_fp8_matmul(
        out.len(),
        activations.len(),
        weights.len(),
        activation_scales.len(),
        weight_scales.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::matmul_f8e4m3_f8e4m3_block_scaled_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(activations),
            input_pointer(weights),
            input_pointer(activation_scales),
            input_pointer(weight_scales),
            config.rows,
            config.columns,
            config.reduction,
            config.group_n,
            config.group_k,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8", feature = "dtype_f16"))]
/// Computes a matrix product from quantized inputs using the configured scales.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    activations: &impl DeviceInput<u8>,
    weights: &impl DeviceInput<u8>,
    activation_scales: &impl DeviceInput<f32>,
    weight_scales: &impl DeviceInput<f32>,
    config: BlockScaledFp8MatmulConfig,
) -> Result<()> {
    validate_block_scaled_fp8_matmul(
        out.len(),
        activations.len(),
        weights.len(),
        activation_scales.len(),
        weight_scales.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::matmul_f8e4m3_f8e4m3_block_scaled_f16(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(activations),
            input_pointer(weights),
            input_pointer(activation_scales),
            input_pointer(weight_scales),
            config.rows,
            config.columns,
            config.reduction,
            config.group_n,
            config.group_k,
        )
    }
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8", feature = "dtype_bf16"))]
/// Computes a matrix product from quantized inputs using the configured scales.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn matmul_f8e4m3_f8e4m3_block_scaled_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    activations: &impl DeviceInput<u8>,
    weights: &impl DeviceInput<u8>,
    activation_scales: &impl DeviceInput<f32>,
    weight_scales: &impl DeviceInput<f32>,
    config: BlockScaledFp8MatmulConfig,
) -> Result<()> {
    validate_block_scaled_fp8_matmul(
        out.len(),
        activations.len(),
        weights.len(),
        activation_scales.len(),
        weight_scales.len(),
        config,
    )?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::quantization::matmul_f8e4m3_f8e4m3_block_scaled_bf16(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(activations),
            input_pointer(weights),
            input_pointer(activation_scales),
            input_pointer(weight_scales),
            config.rows,
            config.columns,
            config.reduction,
            config.group_n,
            config.group_k,
        )
    }
}

#[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
fn validate_per_token_group_quant(
    out_len: usize,
    scales_len: usize,
    input_len: usize,
    config: PerTokenGroupQuantConfig,
) -> Result<usize> {
    if input_len == 0
        || config.group_size == 0
        || config.group_size > 128
        || !config.eps.is_finite()
        || config.eps <= 0.0
    {
        return Err(Error::InvalidLength);
    }
    if !input_len.is_multiple_of(config.group_size) {
        return Err(Error::LengthMismatch);
    }
    let groups = input_len / config.group_size;
    ensure_len(out_len, input_len)?;
    ensure_len(scales_len, groups)?;
    Ok(groups)
}

#[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
fn validate_per_token_group_quant_column_major(
    out_len: usize,
    scales_len: usize,
    input_len: usize,
    config: PerTokenGroupQuantColumnMajorConfig,
) -> Result<()> {
    if config.rows == 0
        || config.columns == 0
        || config.group_size == 0
        || config.group_size > 128
        || !config.columns.is_multiple_of(config.group_size)
        || config.scale_column_stride < config.rows
        || !config.eps.is_finite()
        || config.eps <= 0.0
    {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    let groups_per_row = config.columns / config.group_size;
    let scale_len = strided_matrix_reach(groups_per_row, config.rows, config.scale_column_stride)?;
    ensure_len(scales_len, scale_len)?;
    Ok(())
}

fn validate_grouped_dequant(
    out_len: usize,
    input_len: usize,
    scales_len: usize,
    zero_points_len: usize,
    group_size: usize,
    groups: usize,
) -> Result<usize> {
    if group_size == 0 || groups == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(group_size, groups)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(scales_len, groups)?;
    ensure_len(zero_points_len, groups)?;
    Ok(len)
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_block_quant(
    out_len: usize,
    scales_len: usize,
    len: usize,
    block_size: usize,
) -> Result<usize> {
    if len == 0 || block_size == 0 || block_size > 128 {
        return Err(Error::InvalidLength);
    }
    ensure_len(out_len, len)?;
    let blocks = len.div_ceil(block_size);
    ensure_len(scales_len, blocks)?;
    Ok(blocks)
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_block_dequant(
    out_len: usize,
    input_len: usize,
    scales_len: usize,
    rows: usize,
    columns: usize,
    block_size: usize,
) -> Result<()> {
    if rows == 0 || columns == 0 || block_size == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, columns)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    let scale_rows = rows.div_ceil(block_size);
    let scale_columns = columns.div_ceil(block_size);
    ensure_len(
        scales_len,
        checked_element_count(scale_rows, scale_columns)?,
    )?;
    Ok(())
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_block_scaled_fp8_matmul(
    out_len: usize,
    activations_len: usize,
    weights_len: usize,
    activation_scales_len: usize,
    weight_scales_len: usize,
    config: BlockScaledFp8MatmulConfig,
) -> Result<()> {
    if config.rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.group_n == 0
        || config.group_k == 0
    {
        return Err(Error::InvalidLength);
    }
    ensure_len(out_len, checked_element_count(config.rows, config.columns)?)?;
    ensure_len(
        activations_len,
        checked_element_count(config.rows, config.reduction)?,
    )?;
    ensure_len(
        weights_len,
        checked_element_count(config.columns, config.reduction)?,
    )?;
    let k_groups = config.reduction.div_ceil(config.group_k);
    let n_groups = config.columns.div_ceil(config.group_n);
    ensure_len(
        activation_scales_len,
        checked_element_count(config.rows, k_groups)?,
    )?;
    ensure_len(
        weight_scales_len,
        checked_element_count(n_groups, k_groups)?,
    )?;
    Ok(())
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_rope_quantize_fp8(
    out_len: usize,
    input_len: usize,
    cos_sin_cache_len: usize,
    pos_ids_len: usize,
    config: RopeQuantizeFp8Config,
) -> Result<()> {
    if config.tokens == 0
        || config.heads == 0
        || config.rope_dim == 0
        || !config.rope_dim.is_multiple_of(2)
        || config.cos_sin_cache_stride < config.rope_dim
        || !config.quant_scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(
        checked_element_count(config.tokens, config.heads)?,
        config.rope_dim,
    )?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(pos_ids_len, config.tokens)?;
    if cos_sin_cache_len < config.cos_sin_cache_stride {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
fn validate_qk_rope_quantize_fp8(
    q_rope_out_len: usize,
    k_rope_out_len: usize,
    q_nope_out_len: usize,
    k_nope_out_len: usize,
    q_rope_len: usize,
    k_rope_len: usize,
    q_nope_len: usize,
    k_nope_len: usize,
    cos_sin_cache_len: usize,
    pos_ids_len: usize,
    config: QkRopeQuantizeFp8Config,
) -> Result<()> {
    if config.tokens == 0
        || config.q_heads == 0
        || config.kv_heads == 0
        || config.rope_dim == 0
        || !config.rope_dim.is_multiple_of(2)
        || config.cos_sin_cache_stride < config.rope_dim
        || !config.quant_scale_q.is_finite()
        || !config.quant_scale_kv.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    let q_rope_expected = checked_element_count(
        checked_element_count(config.tokens, config.q_heads)?,
        config.rope_dim,
    )?;
    let k_rope_expected = checked_element_count(
        checked_element_count(config.tokens, config.kv_heads)?,
        config.rope_dim,
    )?;
    let q_nope_expected = checked_element_count(
        checked_element_count(config.tokens, config.q_heads)?,
        config.no_rope_dim,
    )?;
    let k_nope_expected = checked_element_count(
        checked_element_count(config.tokens, config.kv_heads)?,
        config.no_rope_dim,
    )?;
    ensure_len(q_rope_out_len, q_rope_expected)?;
    ensure_len(k_rope_out_len, k_rope_expected)?;
    ensure_len(q_nope_out_len, q_nope_expected)?;
    ensure_len(k_nope_out_len, k_nope_expected)?;
    ensure_len(q_rope_len, q_rope_expected)?;
    ensure_len(k_rope_len, k_rope_expected)?;
    ensure_len(q_nope_len, q_nope_expected)?;
    ensure_len(k_nope_len, k_nope_expected)?;
    ensure_len(pos_ids_len, config.tokens)?;
    if cos_sin_cache_len < config.cos_sin_cache_stride {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_qk_rope_quantize_fp8_rope_only(
    q_rope_out_len: usize,
    k_rope_out_len: usize,
    q_rope_len: usize,
    k_rope_len: usize,
    cos_sin_cache_len: usize,
    pos_ids_len: usize,
    config: QkRopeQuantizeFp8Config,
) -> Result<()> {
    if config.no_rope_dim != 0 {
        return Err(Error::InvalidLength);
    }
    validate_qk_rope_quantize_fp8(
        q_rope_out_len,
        k_rope_out_len,
        0,
        0,
        q_rope_len,
        k_rope_len,
        0,
        0,
        cos_sin_cache_len,
        pos_ids_len,
        config,
    )
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
fn validate_mla_rope_quantize_fp8(
    q_rope_out_len: usize,
    k_rope_out_len: usize,
    q_nope_out_len: usize,
    k_nope_out_len: usize,
    q_rope_len: usize,
    k_rope_len: usize,
    q_nope_len: usize,
    k_nope_len: usize,
    cos_sin_cache_len: usize,
    pos_ids_len: usize,
    config: MlaRopeQuantizeFp8Config,
) -> Result<()> {
    if config.tokens == 0
        || config.q_heads == 0
        || config.rope_dim == 0
        || !config.rope_dim.is_multiple_of(2)
        || config.cos_sin_cache_stride < config.rope_dim
        || !config.quant_scale_q.is_finite()
        || !config.quant_scale_kv.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    let q_rope_expected = checked_element_count(
        checked_element_count(config.tokens, config.q_heads)?,
        config.rope_dim,
    )?;
    let k_rope_expected = checked_element_count(config.tokens, config.rope_dim)?;
    let q_nope_expected = checked_element_count(
        checked_element_count(config.tokens, config.q_heads)?,
        config.no_rope_dim,
    )?;
    let k_nope_expected = checked_element_count(config.tokens, config.no_rope_dim)?;
    ensure_len(q_rope_out_len, q_rope_expected)?;
    ensure_len(k_rope_out_len, k_rope_expected)?;
    ensure_len(q_nope_out_len, q_nope_expected)?;
    ensure_len(k_nope_out_len, k_nope_expected)?;
    ensure_len(q_rope_len, q_rope_expected)?;
    ensure_len(k_rope_len, k_rope_expected)?;
    ensure_len(q_nope_len, q_nope_expected)?;
    ensure_len(k_nope_len, k_nope_expected)?;
    ensure_len(pos_ids_len, config.tokens)?;
    if cos_sin_cache_len < config.cos_sin_cache_stride {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
fn validate_mla_rope_quantize_fp8_rope_only(
    q_rope_out_len: usize,
    k_rope_out_len: usize,
    q_rope_len: usize,
    k_rope_len: usize,
    cos_sin_cache_len: usize,
    pos_ids_len: usize,
    config: MlaRopeQuantizeFp8Config,
) -> Result<()> {
    if config.no_rope_dim != 0 {
        return Err(Error::InvalidLength);
    }
    validate_mla_rope_quantize_fp8(
        q_rope_out_len,
        k_rope_out_len,
        0,
        0,
        q_rope_len,
        k_rope_len,
        0,
        0,
        cos_sin_cache_len,
        pos_ids_len,
        config,
    )
}

fn validate_dequantized_weight_matmul(
    out_len: usize,
    activations_len: usize,
    weights_len: usize,
    scales_len: usize,
    zero_points_len: usize,
    config: DequantizedWeightMatmulConfig,
) -> Result<()> {
    if config.rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.activation_row_stride < config.reduction
        || config.weight_row_stride < config.reduction
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }
    ensure_len(
        activations_len,
        strided_matrix_reach(config.rows, config.reduction, config.activation_row_stride)?,
    )?;
    ensure_len(
        weights_len,
        strided_matrix_reach(config.columns, config.reduction, config.weight_row_stride)?,
    )?;
    ensure_len(scales_len, config.columns)?;
    ensure_len(zero_points_len, config.columns)?;
    ensure_len(
        out_len,
        strided_matrix_reach(config.rows, config.columns, config.output_row_stride)?,
    )?;
    Ok(())
}

fn validate_quantized_matmul(
    out_len: usize,
    activations_len: usize,
    weights_len: usize,
    config: QuantizedMatmulConfig,
) -> Result<()> {
    if config.rows == 0
        || config.columns == 0
        || config.reduction == 0
        || config.activation_row_stride < config.reduction
        || config.weight_row_stride < config.reduction
        || config.output_row_stride < config.columns
        || !config.output_scale.is_finite()
    {
        return Err(Error::InvalidLength);
    }
    ensure_len(
        activations_len,
        strided_matrix_reach(config.rows, config.reduction, config.activation_row_stride)?,
    )?;
    ensure_len(
        weights_len,
        strided_matrix_reach(config.columns, config.reduction, config.weight_row_stride)?,
    )?;
    ensure_len(
        out_len,
        strided_matrix_reach(config.rows, config.columns, config.output_row_stride)?,
    )?;
    Ok(())
}

fn strided_matrix_reach(rows: usize, columns: usize, row_stride: usize) -> Result<usize> {
    let prior_rows = rows.checked_sub(1).ok_or(Error::InvalidLength)?;
    prior_rows
        .checked_mul(row_stride)
        .and_then(|offset| offset.checked_add(columns))
        .ok_or(Error::SizeOverflow)
}

fn validate_unpack_int4(out_len: usize, packed_len: usize) -> Result<usize> {
    if out_len == 0 {
        return Err(Error::InvalidLength);
    }
    let expected_packed_len = out_len.div_ceil(2);
    ensure_len(packed_len, expected_packed_len)?;
    Ok(out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_vector_config_builders_preserve_tuning_choice() {
        let tuning =
            VectorTuning::tile(128).with_compile(KernelCompileConfig::default().occupancy(2));
        assert_eq!(
            VectorConfig::create(17).explicit_tuning(tuning).tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            VectorConfig::create(17).cached_tuning().tuning,
            TuningChoice::Cached
        );

        let options = TuneOptions::create(0, 1, 2);
        assert_eq!(
            VectorConfig::create(17).tune(options).tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            VectorConfig::create(17).retune(options).tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_quantize_uses_shared_vector_tile_candidates() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_quantize_validation_accepts_exact_lengths() -> Result<()> {
        validate_vector_lengths(24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_quantize_validation_rejects_short_input() {
        assert!(validate_vector_lengths(24, 23, VectorConfig::create(24)).is_err());
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_validation_accepts_exact_group_scales() -> Result<()> {
        let config = PerTokenGroupQuantConfig::create(8, 1e-6);
        assert_eq!(validate_per_token_group_quant(24, 3, 24, config)?, 3);
        Ok(())
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_validation_rejects_partial_group() {
        let config = PerTokenGroupQuantConfig::create(8, 1e-6);
        assert!(matches!(
            validate_per_token_group_quant(25, 4, 25, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_validation_rejects_bad_eps() {
        let config = PerTokenGroupQuantConfig::create(8, 0.0);
        assert!(matches!(
            validate_per_token_group_quant(24, 3, 24, config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_column_major_validation_accepts_padded_scale_stride() -> Result<()>
    {
        let config = PerTokenGroupQuantColumnMajorConfig::create(3, 16, 8, 4, 1e-6);
        validate_per_token_group_quant_column_major(48, 7, 48, config)
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_column_major_validation_rejects_short_scale_buffer() {
        let config = PerTokenGroupQuantColumnMajorConfig::create(3, 16, 8, 4, 1e-6);
        assert!(matches!(
            validate_per_token_group_quant_column_major(48, 6, 48, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", any(feature = "dtype_i8", feature = "dtype_f8")))]
    #[test]
    fn test_per_token_group_quant_column_major_validation_rejects_short_scale_stride() {
        let config = PerTokenGroupQuantColumnMajorConfig::create(3, 16, 8, 2, 1e-6);
        assert!(matches!(
            validate_per_token_group_quant_column_major(48, 6, 48, config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_block_scaled_fp8_matmul_validation_accepts_tail_scale_groups() -> Result<()> {
        let config = BlockScaledFp8MatmulConfig::create(3, 5, 7, 2, 3);
        validate_block_scaled_fp8_matmul(15, 21, 35, 9, 9, config)
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_block_scaled_fp8_matmul_validation_rejects_short_weight_scales() {
        let config = BlockScaledFp8MatmulConfig::create(3, 5, 7, 2, 3);
        assert!(matches!(
            validate_block_scaled_fp8_matmul(15, 21, 35, 9, 8, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_block_scaled_fp8_matmul_validation_rejects_zero_scale_group() {
        let config = BlockScaledFp8MatmulConfig::create(3, 5, 7, 0, 3);
        assert!(matches!(
            validate_block_scaled_fp8_matmul(15, 21, 35, 9, 9, config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_quantize_validation_accepts_2d_k_layout() -> Result<()> {
        let config = MlaRopeQuantizeFp8Config::row_major_interleaved(3, 4, 8, 5, 8, 1.0, 0.5);
        validate_mla_rope_quantize_fp8(96, 24, 60, 15, 96, 24, 60, 15, 8, 3, config)
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_qk_rope_quantize_validation_accepts_zero_no_rope() -> Result<()> {
        let config =
            QkRopeQuantizeFp8Config::row_major_interleaved_rope_only(3, 4, 2, 8, 8, 1.0, 0.5);
        validate_qk_rope_quantize_fp8(96, 48, 0, 0, 96, 48, 0, 0, 8, 3, config)
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_qk_rope_only_validation_rejects_nonzero_no_rope_config() {
        let config = QkRopeQuantizeFp8Config::row_major_interleaved(3, 4, 2, 8, 5, 8, 1.0, 0.5);
        assert!(matches!(
            validate_qk_rope_quantize_fp8_rope_only(96, 48, 96, 48, 8, 3, config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_qk_rope_quantize_validation_rejects_nonzero_nope_buffers_for_zero_no_rope() {
        let config = QkRopeQuantizeFp8Config::row_major_interleaved(3, 4, 2, 8, 0, 8, 1.0, 0.5);
        assert!(matches!(
            validate_qk_rope_quantize_fp8(96, 48, 1, 0, 96, 48, 0, 0, 8, 3, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_quantize_validation_accepts_zero_no_rope() -> Result<()> {
        let config =
            MlaRopeQuantizeFp8Config::row_major_interleaved_rope_only(3, 4, 8, 8, 1.0, 0.5);
        validate_mla_rope_quantize_fp8(96, 24, 0, 0, 96, 24, 0, 0, 8, 3, config)
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_only_validation_rejects_nonzero_no_rope_config() {
        let config = MlaRopeQuantizeFp8Config::row_major_interleaved(3, 4, 8, 5, 8, 1.0, 0.5);
        assert!(matches!(
            validate_mla_rope_quantize_fp8_rope_only(96, 24, 96, 24, 8, 3, config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_quantize_validation_rejects_nonzero_nope_buffers_for_zero_no_rope() {
        let config = MlaRopeQuantizeFp8Config::row_major_interleaved(3, 4, 8, 0, 8, 1.0, 0.5);
        assert!(matches!(
            validate_mla_rope_quantize_fp8(96, 24, 1, 0, 96, 24, 0, 0, 8, 3, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_quantize_validation_rejects_3d_k_lengths() {
        let config = MlaRopeQuantizeFp8Config::row_major_interleaved(3, 4, 8, 5, 8, 1.0, 0.5);
        assert!(matches!(
            validate_mla_rope_quantize_fp8(96, 96, 60, 60, 96, 96, 60, 60, 8, 3, config,),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(all(feature = "dtype_f32", feature = "dtype_f8"))]
    #[test]
    fn test_mla_rope_quantize_validation_rejects_bad_scale() {
        let config = MlaRopeQuantizeFp8Config::row_major_interleaved(3, 4, 8, 5, 8, f32::NAN, 0.5);
        assert!(matches!(
            validate_mla_rope_quantize_fp8(96, 24, 60, 15, 96, 24, 60, 15, 8, 3, config),
            Err(Error::InvalidLength)
        ));
    }
}
