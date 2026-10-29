//! Binary/ternary elementwise operations, clamps, masks, and selects.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
use crate::cuda::utility::ensure_binary_lengths;
use crate::{
    cuda::{
        autotune::{self, KernelCompileConfig, TuneOptions, TuneReport, TuningChoice},
        interop::input_pointer,
        utility::{
            VectorConfig, VectorTuning, resolve_vector_tuning, validate_vector_lengths,
            vector_tile_candidates, vector_tune_key,
        },
    },
    error::Result,
    utility::ensure_len,
};

fn vector_launch_parts(tuning: Option<VectorTuning>) -> (Option<usize>, KernelCompileConfig) {
    tuning.map_or_else(
        || (None, KernelCompileConfig::default()),
        |tuning| (Some(tuning.tile_width), tuning.compile),
    )
}

fn validate_binary_vector_lengths(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: VectorConfig,
) -> Result<()> {
    validate_vector_lengths(out_len, lhs_len, config)?;
    ensure_len(rhs_len, config.length)?;
    Ok(())
}

fn validate_ternary_vector_lengths(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    acc_len: usize,
    config: VectorConfig,
) -> Result<()> {
    validate_binary_vector_lengths(out_len, lhs_len, rhs_len, config)?;
    ensure_len(acc_len, config.length)?;
    Ok(())
}

#[allow(dead_code, reason = "only used by some kernel feature combinations")]
fn validate_masked_vector_lengths(
    out_len: usize,
    input_len: usize,
    mask_len: usize,
    config: VectorConfig,
) -> Result<()> {
    validate_vector_lengths(out_len, input_len, config)?;
    ensure_len(mask_len, config.length)?;
    Ok(())
}

fn launch_elementwise_vector<F>(
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

fn tune_elementwise_vector<F>(
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

macro_rules! binary_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: VectorConfig,
        ) -> Result<()> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(lhs),
                            input_pointer(rhs),
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(lhs),
                        input_pointer(rhs),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64",
))]
macro_rules! binary_raw_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out.len(), lhs.len(), rhs.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::elementwise::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_u8")]
macro_rules! binary_bool_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: VectorConfig,
        ) -> Result<()> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(lhs),
                            input_pointer(rhs),
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(lhs),
                        input_pointer(rhs),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_u8")]
macro_rules! binary_bool_raw_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(lhs.len(), len)?;
            ensure_len(rhs.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::elementwise::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    len,
                )
            }
        }
    };
}

macro_rules! ternary_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            acc: &impl DeviceInput<$ty>,
            config: VectorConfig,
        ) -> Result<()> {
            validate_ternary_vector_lengths(out.len(), lhs.len(), rhs.len(), acc.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(lhs),
                            input_pointer(rhs),
                            input_pointer(acc),
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            acc: &impl DeviceInput<$ty>,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_ternary_vector_lengths(out.len(), lhs.len(), rhs.len(), acc.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(lhs),
                        input_pointer(rhs),
                        input_pointer(acc),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `lhs + rhs` elementwise.",
    add_f32,
    tune_add_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `lhs - rhs` elementwise.",
    sub_f32,
    tune_sub_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `lhs * rhs` elementwise.",
    mul_f32,
    tune_mul_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `lhs / rhs` elementwise.",
    div_f32,
    tune_div_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Raises each `lhs` element to its corresponding `rhs` power.",
    pow_f32,
    tune_pow_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes the Euclidean norm of each pair of input elements.",
    hypot_f32,
    tune_hypot_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `(lhs - rhs) * (lhs - rhs)` elementwise.",
    squared_difference_f32,
    tune_squared_difference_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes `log(exp(lhs) + exp(rhs))` elementwise.",
    logaddexp_f32,
    tune_logaddexp_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes the elementwise x-log-y product of `lhs` and `rhs`.",
    xlogy_f32,
    tune_xlogy_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_f32,
    tune_max_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_f32,
    tune_min_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_f32,
    tune_modulo_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_f32,
    tune_remainder_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Computes the two-argument arctangent of corresponding `lhs` and `rhs` elements.",
    atan2_f32,
    tune_atan2_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs == rhs`.",
    equal_f32,
    tune_equal_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs != rhs`.",
    not_equal_f32,
    tune_not_equal_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs < rhs`.",
    less_f32,
    tune_less_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs <= rhs`.",
    less_equal_f32,
    tune_less_equal_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs > rhs`.",
    greater_f32,
    tune_greater_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs >= rhs`.",
    greater_equal_f32,
    tune_greater_equal_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
ternary_float_fn!(
    "Computes the elementwise fused multiply-add of three inputs.",
    fma_f32,
    tune_fma_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_f32,
    tune_equal_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_f32,
    tune_not_equal_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_f32,
    tune_less_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_f32,
    tune_less_equal_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_f32,
    tune_greater_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_f32,
    tune_greater_equal_bool_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `lhs + rhs` elementwise.",
    add_f16,
    tune_add_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `lhs - rhs` elementwise.",
    sub_f16,
    tune_sub_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `lhs * rhs` elementwise.",
    mul_f16,
    tune_mul_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `lhs / rhs` elementwise.",
    div_f16,
    tune_div_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Raises each `lhs` element to its corresponding `rhs` power.",
    pow_f16,
    tune_pow_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes the Euclidean norm of each pair of input elements.",
    hypot_f16,
    tune_hypot_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `(lhs - rhs) * (lhs - rhs)` elementwise.",
    squared_difference_f16,
    tune_squared_difference_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes `log(exp(lhs) + exp(rhs))` elementwise.",
    logaddexp_f16,
    tune_logaddexp_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes the elementwise x-log-y product of `lhs` and `rhs`.",
    xlogy_f16,
    tune_xlogy_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_f16,
    tune_max_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_f16,
    tune_min_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_f16,
    tune_modulo_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_f16,
    tune_remainder_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Computes the two-argument arctangent of corresponding `lhs` and `rhs` elements.",
    atan2_f16,
    tune_atan2_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs == rhs`.",
    equal_f16,
    tune_equal_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs != rhs`.",
    not_equal_f16,
    tune_not_equal_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs < rhs`.",
    less_f16,
    tune_less_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs <= rhs`.",
    less_equal_f16,
    tune_less_equal_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs > rhs`.",
    greater_f16,
    tune_greater_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs >= rhs`.",
    greater_equal_f16,
    tune_greater_equal_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
ternary_float_fn!(
    "Computes the elementwise fused multiply-add of three inputs.",
    fma_f16,
    tune_fma_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_f16,
    tune_equal_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_f16,
    tune_not_equal_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_f16,
    tune_less_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_f16,
    tune_less_equal_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_f16,
    tune_greater_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_f16,
    tune_greater_equal_bool_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `lhs + rhs` elementwise.",
    add_f64,
    tune_add_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `lhs - rhs` elementwise.",
    sub_f64,
    tune_sub_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `lhs * rhs` elementwise.",
    mul_f64,
    tune_mul_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `lhs / rhs` elementwise.",
    div_f64,
    tune_div_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Raises each `lhs` element to its corresponding `rhs` power.",
    pow_f64,
    tune_pow_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes the Euclidean norm of each pair of input elements.",
    hypot_f64,
    tune_hypot_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `(lhs - rhs) * (lhs - rhs)` elementwise.",
    squared_difference_f64,
    tune_squared_difference_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes `log(exp(lhs) + exp(rhs))` elementwise.",
    logaddexp_f64,
    tune_logaddexp_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes the elementwise x-log-y product of `lhs` and `rhs`.",
    xlogy_f64,
    tune_xlogy_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_f64,
    tune_max_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_f64,
    tune_min_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_f64,
    tune_modulo_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_f64,
    tune_remainder_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Computes the two-argument arctangent of corresponding `lhs` and `rhs` elements.",
    atan2_f64,
    tune_atan2_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs == rhs`.",
    equal_f64,
    tune_equal_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs != rhs`.",
    not_equal_f64,
    tune_not_equal_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs < rhs`.",
    less_f64,
    tune_less_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs <= rhs`.",
    less_equal_f64,
    tune_less_equal_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs > rhs`.",
    greater_f64,
    tune_greater_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
binary_float_fn!(
    "Writes the elementwise comparison `lhs >= rhs`.",
    greater_equal_f64,
    tune_greater_equal_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
ternary_float_fn!(
    "Computes the elementwise fused multiply-add of three inputs.",
    fma_f64,
    tune_fma_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_f64,
    tune_equal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_f64,
    tune_not_equal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_f64,
    tune_less_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_f64,
    tune_less_equal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_f64,
    tune_greater_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
binary_bool_float_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_f64,
    tune_greater_equal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_u32,
    u32
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_u8,
    u8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs == rhs`.",
    equal_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs != rhs`.",
    not_equal_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs < rhs`.",
    less_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs <= rhs`.",
    less_equal_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs > rhs`.",
    greater_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
binary_bool_raw_fn!(
    "Writes the byte result of the elementwise comparison `lhs >= rhs`.",
    greater_equal_bool_i64,
    i64
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_u32, u32);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_u32,
    u32
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_u8, u8);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_u8,
    u8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_i8, i8);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_i8,
    i8
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_i32, i32);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_i32,
    i32
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_u64, u64);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_u64,
    u64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Selects the elementwise minimum of `lhs` and `rhs`.",
    min_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Selects the elementwise maximum of `lhs` and `rhs`.",
    max_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs + rhs` elementwise.", add_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs - rhs` elementwise.", sub_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs * rhs` elementwise.", mul_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs / rhs` elementwise.", div_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Computes the elementwise modulo of `lhs` by `rhs`.",
    modulo_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Computes the elementwise remainder of `lhs` divided by `rhs`.",
    remainder_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs & rhs` elementwise.", bitwise_and_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs | rhs` elementwise.", bitwise_or_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!("Computes `lhs ^ rhs` elementwise.", bitwise_xor_i64, i64);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Shifts each `lhs` element left by its corresponding `rhs` count.",
    shift_left_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
binary_raw_fn!(
    "Shifts each `lhs` element right by its corresponding `rhs` count.",
    shift_right_i64,
    i64
);

macro_rules! clamp_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            min: $ty,
            max: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            min,
                            max,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            min: $ty,
            max: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        min,
                        max,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
clamp_float_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_f32,
    tune_clamp_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
clamp_float_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_f64,
    tune_clamp_f64,
    f64
);

#[cfg(feature = "dtype_f16")]
clamp_float_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_f16,
    tune_clamp_f16,
    f16
);

#[cfg(any(
    feature = "dtype_u8",
    feature = "dtype_i8",
    feature = "dtype_u32",
    feature = "dtype_i32",
    feature = "dtype_u64",
    feature = "dtype_i64"
))]
macro_rules! clamp_helper_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            min: $ty,
            max: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::elementwise::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    min,
                    max,
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_u8")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_u8,
    u8
);

#[cfg(feature = "dtype_i8")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_i8,
    i8
);

#[cfg(feature = "dtype_u32")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_u32,
    u32
);

#[cfg(feature = "dtype_i32")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_i32,
    i32
);

#[cfg(feature = "dtype_u64")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_u64,
    u64
);

#[cfg(feature = "dtype_i64")]
clamp_helper_fn!(
    "Clamps each input element between the supplied minimum and maximum.",
    clamp_i64,
    i64
);

macro_rules! lerp_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(lhs),
                            input_pointer(rhs),
                            scalar,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_binary_vector_lengths(out.len(), lhs.len(), rhs.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(lhs),
                        input_pointer(rhs),
                        scalar,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

macro_rules! where_float_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty, $condition:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<$condition>,
            x: &impl DeviceInput<$ty>,
            y: &impl DeviceInput<$ty>,
            config: VectorConfig,
        ) -> Result<()> {
            validate_ternary_vector_lengths(out.len(), condition.len(), x.len(), y.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(condition),
                            input_pointer(x),
                            input_pointer(y),
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<$condition>,
            x: &impl DeviceInput<$ty>,
            y: &impl DeviceInput<$ty>,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_ternary_vector_lengths(out.len(), condition.len(), x.len(), y.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(condition),
                        input_pointer(x),
                        input_pointer(y),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
lerp_float_fn!(
    "Interpolates corresponding input elements using the supplied weight.",
    lerp_f32,
    tune_lerp_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
lerp_float_fn!(
    "Interpolates corresponding input elements using the supplied weight.",
    lerp_f64,
    tune_lerp_f64,
    f64
);

#[cfg(feature = "dtype_f16")]
lerp_float_fn!(
    "Interpolates corresponding input elements using the supplied weight.",
    lerp_f16,
    tune_lerp_f16,
    f16
);

#[cfg(feature = "dtype_f32")]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the condition.",
    where_f32,
    tune_where_f32,
    f32,
    f32
);

#[cfg(feature = "dtype_f64")]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the condition.",
    where_f64,
    tune_where_f64,
    f64,
    f64
);

#[cfg(feature = "dtype_f16")]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the condition.",
    where_f16,
    tune_where_f16,
    f16,
    f16
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_f32,
    tune_where_bool_f32,
    f32,
    u8
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_f64,
    tune_where_bool_f64,
    f64,
    u8
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
where_float_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_f16,
    tune_where_bool_f16,
    f16,
    u8
);

#[cfg(feature = "dtype_u8")]
macro_rules! where_bool_helper_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            x: &impl DeviceInput<$ty>,
            y: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            let len = ensure_binary_lengths(out.len(), x.len(), y.len())?;
            ensure_len(condition.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::elementwise::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(condition),
                    input_pointer(x),
                    input_pointer(y),
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_u8")]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_u8,
    u8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
where_bool_helper_fn!(
    "Selects corresponding elements from two inputs according to the byte condition.",
    where_bool_i64,
    i64
);

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
macro_rules! masked_fill_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            mask: &impl DeviceInput<u8>,
            value: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_masked_vector_lengths(out.len(), input.len(), mask.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::elementwise::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            input_pointer(mask),
                            value,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        /// Times candidate tiles and enqueues the selected operation.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// The access and lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            mask: &impl DeviceInput<u8>,
            value: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_masked_vector_lengths(out.len(), input.len(), mask.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_elementwise_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::elementwise::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        input_pointer(mask),
                        value,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
masked_fill_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_f32,
    tune_masked_fill_f32,
    f32
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
masked_fill_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_f16,
    tune_masked_fill_f16,
    f16
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
masked_fill_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_f64,
    tune_masked_fill_f64,
    f64
);

#[cfg(feature = "dtype_u8")]
macro_rules! masked_fill_helper_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            mask: &impl DeviceInput<u8>,
            value: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            ensure_len(mask.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::elementwise::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(mask),
                    value,
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_u8")]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_u8,
    u8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_i8,
    i8
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_u32,
    u32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_i32,
    i32
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_u64,
    u64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
masked_fill_helper_fn!(
    "Replaces masked input elements with the supplied fill value.",
    masked_fill_i64,
    i64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_vector_config_builders_preserve_tuning_choice() {
        let tuning =
            VectorTuning::tile(256).with_compile(KernelCompileConfig::default().occupancy(2));
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
    fn test_elementwise_uses_shared_vector_tile_candidates() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_binary_validation_accepts_exact_lengths() -> Result<()> {
        validate_binary_vector_lengths(24, 24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_binary_validation_rejects_short_rhs() {
        assert!(validate_binary_vector_lengths(24, 24, 23, VectorConfig::create(24)).is_err());
    }

    #[test]
    fn test_ternary_validation_accepts_exact_lengths() -> Result<()> {
        validate_ternary_vector_lengths(24, 24, 24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_ternary_validation_rejects_short_acc() {
        assert!(validate_ternary_vector_lengths(24, 24, 24, 23, VectorConfig::create(24)).is_err());
    }

    #[test]
    fn test_masked_validation_accepts_exact_lengths() -> Result<()> {
        validate_masked_vector_lengths(24, 24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_masked_validation_rejects_short_mask() {
        assert!(validate_masked_vector_lengths(24, 24, 23, VectorConfig::create(24)).is_err());
    }
}
