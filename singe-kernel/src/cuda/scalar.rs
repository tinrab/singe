//! Scalar arithmetic and scalar comparisons on device slices.

use ::cutile::cuda_async::device_buffer::DevicePointer;
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
            VectorConfig, VectorTuning, ensure_unary_lengths, resolve_vector_tuning,
            validate_vector_lengths, vector_tile_candidates, vector_tune_key,
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

fn launch_scalar_vector<F>(
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

fn tune_scalar_vector<F>(
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

macro_rules! scalar_float_fn {
    ($operation:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_scalar_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::scalar::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            scalar,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_scalar_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::scalar::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
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

macro_rules! scalar_raw_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
        ) -> Result<()> {
            let len = ensure_unary_lengths(out.len(), input.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::scalar::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    scalar,
                    len,
                )
            }
        }
    };
}

macro_rules! scalar_bool_float_fn {
    ($operation:literal, $name:ident, $tune_name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_scalar_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::scalar::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            scalar,
                            config.length,
                            tile_width,
                            compile,
                        )
                    }
                },
            )
        }

        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_scalar_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::scalar::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
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

macro_rules! scalar_bool_raw_fn {
    ($operation:literal, $name:ident, $ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<u8>,
            input: &impl DeviceInput<$ty>,
            scalar: $ty,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::scalar::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    scalar,
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_f32,
    tune_add_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_f32,
    tune_sub_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_f32,
    tune_rsub_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_f32,
    tune_mul_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    scale_f32,
    tune_scale_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_f32,
    tune_div_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_f32,
    tune_rdiv_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_f32,
    tune_modulo_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_f32,
    tune_rmodulo_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `atan2` to each input element and the scalar operand.",
    atan2_scalar_f32,
    tune_atan2_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `ratan2` to each input element and the scalar operand.",
    ratan2_scalar_f32,
    tune_ratan2_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `input ** scalar` elementwise.",
    pow_scalar_f32,
    tune_pow_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `scalar ** input` elementwise.",
    rpow_scalar_f32,
    tune_rpow_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `hypot` to each input element and the scalar operand.",
    hypot_scalar_f32,
    tune_hypot_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `(input - scalar) ** 2` elementwise.",
    squared_difference_scalar_f32,
    tune_squared_difference_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `logaddexp` to each input element and the scalar operand.",
    logaddexp_scalar_f32,
    tune_logaddexp_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `xlogy` to each input element and the scalar operand.",
    xlogy_scalar_f32,
    tune_xlogy_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Applies `rxlogy` to each input element and the scalar operand.",
    rxlogy_scalar_f32,
    tune_rxlogy_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_f32,
    tune_min_scalar_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_float_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_f32,
    tune_max_scalar_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_f32,
    tune_equal_scalar_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_f32,
    tune_not_equal_scalar_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_f32,
    tune_less_scalar_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_f32,
    tune_less_equal_scalar_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_f32,
    tune_greater_scalar_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_f32,
    tune_greater_equal_scalar_bool_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_f16,
    tune_add_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_f16,
    tune_sub_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_f16,
    tune_rsub_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_f16,
    tune_mul_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    scale_f16,
    tune_scale_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_f16,
    tune_div_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_f16,
    tune_rdiv_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_f16,
    tune_modulo_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_f16,
    tune_rmodulo_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `atan2` to each input element and the scalar operand.",
    atan2_scalar_f16,
    tune_atan2_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `ratan2` to each input element and the scalar operand.",
    ratan2_scalar_f16,
    tune_ratan2_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `input ** scalar` elementwise.",
    pow_scalar_f16,
    tune_pow_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `scalar ** input` elementwise.",
    rpow_scalar_f16,
    tune_rpow_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `hypot` to each input element and the scalar operand.",
    hypot_scalar_f16,
    tune_hypot_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `(input - scalar) ** 2` elementwise.",
    squared_difference_scalar_f16,
    tune_squared_difference_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `logaddexp` to each input element and the scalar operand.",
    logaddexp_scalar_f16,
    tune_logaddexp_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `xlogy` to each input element and the scalar operand.",
    xlogy_scalar_f16,
    tune_xlogy_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Applies `rxlogy` to each input element and the scalar operand.",
    rxlogy_scalar_f16,
    tune_rxlogy_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_f16,
    tune_min_scalar_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_float_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_f16,
    tune_max_scalar_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_f16,
    tune_equal_scalar_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_f16,
    tune_not_equal_scalar_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_f16,
    tune_less_scalar_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_f16,
    tune_less_equal_scalar_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_f16,
    tune_greater_scalar_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_f16,
    tune_greater_equal_scalar_bool_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_f64,
    tune_add_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_f64,
    tune_sub_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_f64,
    tune_rsub_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_f64,
    tune_mul_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input * scalar` elementwise.",
    scale_f64,
    tune_scale_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_f64,
    tune_div_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_f64,
    tune_rdiv_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_f64,
    tune_modulo_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_f64,
    tune_rmodulo_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `atan2` to each input element and the scalar operand.",
    atan2_scalar_f64,
    tune_atan2_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `ratan2` to each input element and the scalar operand.",
    ratan2_scalar_f64,
    tune_ratan2_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `input ** scalar` elementwise.",
    pow_scalar_f64,
    tune_pow_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `scalar ** input` elementwise.",
    rpow_scalar_f64,
    tune_rpow_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `hypot` to each input element and the scalar operand.",
    hypot_scalar_f64,
    tune_hypot_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `(input - scalar) ** 2` elementwise.",
    squared_difference_scalar_f64,
    tune_squared_difference_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `logaddexp` to each input element and the scalar operand.",
    logaddexp_scalar_f64,
    tune_logaddexp_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `xlogy` to each input element and the scalar operand.",
    xlogy_scalar_f64,
    tune_xlogy_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Applies `rxlogy` to each input element and the scalar operand.",
    rxlogy_scalar_f64,
    tune_rxlogy_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_f64,
    tune_min_scalar_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_float_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_f64,
    tune_max_scalar_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_f64,
    tune_equal_scalar_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_f64,
    tune_not_equal_scalar_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_f64,
    tune_less_scalar_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_f64,
    tune_less_equal_scalar_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_f64,
    tune_greater_scalar_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
scalar_bool_float_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_f64,
    tune_greater_equal_scalar_bool_f64,
    f64
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_u8,
    u8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input == scalar` elementwise.",
    equal_scalar_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input != scalar` elementwise.",
    not_equal_scalar_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input < scalar` elementwise.",
    less_scalar_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input <= scalar` elementwise.",
    less_equal_scalar_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input > scalar` elementwise.",
    greater_scalar_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
scalar_bool_raw_fn!(
    "Writes the byte result of `input >= scalar` elementwise.",
    greater_equal_scalar_bool_i64,
    i64
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u32")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_u32,
    u32
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `input + scalar` elementwise.", add_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `input - scalar` elementwise.", sub_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `scalar - input` elementwise.", rsub_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `input * scalar` elementwise.", mul_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `input / scalar` elementwise.", div_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!("Computes `scalar / input` elementwise.", rdiv_scalar_u8, u8);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_u8,
    u8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `input + scalar` elementwise.", add_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `input - scalar` elementwise.", sub_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `scalar - input` elementwise.", rsub_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `input * scalar` elementwise.", mul_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `input / scalar` elementwise.", div_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!("Computes `scalar / input` elementwise.", rdiv_scalar_i8, i8);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i8")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_i8,
    i8
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_i32,
    i32
);

#[cfg(feature = "dtype_i32")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_i32,
    i32
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_u64,
    u64
);

#[cfg(feature = "dtype_u64")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_u64,
    u64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `min(input, scalar)` elementwise.",
    min_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `max(input, scalar)` elementwise.",
    max_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input + scalar` elementwise.",
    add_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input - scalar` elementwise.",
    sub_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `scalar - input` elementwise.",
    rsub_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input * scalar` elementwise.",
    mul_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input / scalar` elementwise.",
    div_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `scalar / input` elementwise.",
    rdiv_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Applies `modulo` to each input element and the scalar operand.",
    modulo_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Applies `rmodulo` to each input element and the scalar operand.",
    rmodulo_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input & scalar` elementwise.",
    bitwise_and_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input | scalar` elementwise.",
    bitwise_or_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input ^ scalar` elementwise.",
    bitwise_xor_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input << scalar` elementwise.",
    shift_left_scalar_i64,
    i64
);

#[cfg(feature = "dtype_i64")]
scalar_raw_fn!(
    "Computes `input >> scalar` elementwise.",
    shift_right_scalar_i64,
    i64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_vector_config_builders_preserve_tuning_choice() {
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
    fn test_scalar_uses_shared_vector_tile_candidates() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_scalar_validation_accepts_exact_lengths() -> Result<()> {
        validate_vector_lengths(24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_scalar_validation_rejects_short_input() {
        assert!(validate_vector_lengths(24, 23, VectorConfig::create(24)).is_err());
    }
}
