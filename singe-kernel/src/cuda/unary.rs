//! Single-input math operations and predicates.

#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

use crate::{
    cuda::{
        autotune::{self, TuneOptions, TuneReport, TuningChoice},
        cutile,
        interop::{input_pointer, write_only_pointer},
        utility::{
            VectorConfig, VectorTuning, ensure_unary_lengths, resolve_vector_tuning,
            validate_vector_lengths, vector_tile_candidates, vector_tune_key,
        },
    },
    error::Result,
    utility::ensure_len,
};

macro_rules! unary_float_fn {
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
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = tuning
                        .map(|tuning| (Some(tuning.tile_width), tuning.compile))
                        .unwrap_or((None, Default::default()));
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::unary::$name(
                            &cuda_stream,
                            write_only_pointer(out),
                            input_pointer(input),
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
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::unary::$name(
                        &cuda_stream,
                        write_only_pointer(out),
                        input_pointer(input),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

macro_rules! unary_raw_fn {
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
        ) -> Result<()> {
            let len = ensure_unary_lengths(out.len(), input.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::unary::$name(&stream, write_only_pointer(out), input_pointer(input), len)
            }
        }
    };
}

macro_rules! unary_float_bool_fn {
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
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = tuning
                        .map(|tuning| (Some(tuning.tile_width), tuning.compile))
                        .unwrap_or((None, Default::default()));
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::unary::$name(
                            &cuda_stream,
                            write_only_pointer(out),
                            input_pointer(input),
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
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::unary::$name(
                        &cuda_stream,
                        write_only_pointer(out),
                        input_pointer(input),
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

macro_rules! unary_bool_fn {
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
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::unary::$name(&stream, write_only_pointer(out), input_pointer(input), len)
            }
        }
    };
}

macro_rules! nan_to_num_fn {
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
            nan: $ty,
            posinf: $ty,
            neginf: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = tuning
                        .map(|tuning| (Some(tuning.tile_width), tuning.compile))
                        .unwrap_or((None, Default::default()));
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::unary::$name(
                            &cuda_stream,
                            write_only_pointer(out),
                            input_pointer(input),
                            nan,
                            posinf,
                            neginf,
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
            nan: $ty,
            posinf: $ty,
            neginf: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_unary_float(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::unary::$name(
                        &cuda_stream,
                        write_only_pointer(out),
                        input_pointer(input),
                        nan,
                        posinf,
                        neginf,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

fn launch_unary_float<F>(
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

fn tune_unary_float<F>(
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

#[cfg(feature = "dtype_f32")]
unary_float_fn!("Computes `-input` elementwise.", neg_f32, tune_neg_f32, f32);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `abs(input)` elementwise.",
    abs_f32,
    tune_abs_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `sqrt(input)` elementwise.",
    sqrt_f32,
    tune_sqrt_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `exp` to each input element.",
    exp_f32,
    tune_exp_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `2 ** input` elementwise.",
    exp2_f32,
    tune_exp2_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `exp(input) - 1` elementwise.",
    expm1_f32,
    tune_expm1_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `log` to each input element.",
    log_f32,
    tune_log_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `log2` to each input element.",
    log2_f32,
    tune_log2_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `log10` to each input element.",
    log10_f32,
    tune_log10_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `log(1 + input)` elementwise.",
    log1p_f32,
    tune_log1p_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Converts each input angle from degrees to radians.",
    deg2rad_f32,
    tune_deg2rad_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Converts each input angle from radians to degrees.",
    rad2deg_f32,
    tune_rad2deg_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `sin` to each input element.",
    sin_f32,
    tune_sin_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `cos` to each input element.",
    cos_f32,
    tune_cos_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `tan` to each input element.",
    tan_f32,
    tune_tan_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `sinh` to each input element.",
    sinh_f32,
    tune_sinh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `cosh` to each input element.",
    cosh_f32,
    tune_cosh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `tanh` to each input element.",
    tanh_f32,
    tune_tanh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `asin` to each input element.",
    asin_f32,
    tune_asin_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `acos` to each input element.",
    acos_f32,
    tune_acos_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `atan` to each input element.",
    atan_f32,
    tune_atan_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `ceil` to each input element.",
    ceil_f32,
    tune_ceil_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `floor` to each input element.",
    floor_f32,
    tune_floor_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `trunc` to each input element.",
    trunc_f32,
    tune_trunc_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `frac` to each input element.",
    frac_f32,
    tune_frac_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `1 / input` elementwise.",
    reciprocal_f32,
    tune_reciprocal_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `input * input` elementwise.",
    square_f32,
    tune_square_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Computes `1 / sqrt(input)` elementwise.",
    rsqrt_f32,
    tune_rsqrt_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `cbrt` to each input element.",
    cbrt_f32,
    tune_cbrt_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `asinh` to each input element.",
    asinh_f32,
    tune_asinh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `acosh` to each input element.",
    acosh_f32,
    tune_acosh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Applies `atanh` to each input element.",
    atanh_f32,
    tune_atanh_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
unary_float_fn!(
    "Writes the sign of each input element.",
    sign_f32,
    tune_sign_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
nan_to_num_fn!(
    "Replaces NaNs and infinities with the supplied finite values.",
    nan_to_num_f32,
    tune_nan_to_num_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nan input elements.",
    is_nan_bool_f32,
    tune_is_nan_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for inf input elements.",
    is_inf_bool_f32,
    tune_is_inf_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for pos inf input elements.",
    is_pos_inf_bool_f32,
    tune_is_pos_inf_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for neg inf input elements.",
    is_neg_inf_bool_f32,
    tune_is_neg_inf_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for finite input elements.",
    is_finite_bool_f32,
    tune_is_finite_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for normal input elements.",
    is_normal_bool_f32,
    tune_is_normal_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for subnormal input elements.",
    is_subnormal_bool_f32,
    tune_is_subnormal_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_f32,
    tune_is_zero_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_f32,
    tune_is_nonzero_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_f32,
    tune_is_positive_bool_f32,
    f32
);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_f32,
    tune_is_negative_bool_f32,
    f32
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!("Computes `-input` elementwise.", neg_f16, tune_neg_f16, f16);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `abs(input)` elementwise.",
    abs_f16,
    tune_abs_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `sqrt(input)` elementwise.",
    sqrt_f16,
    tune_sqrt_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `exp` to each input element.",
    exp_f16,
    tune_exp_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `2 ** input` elementwise.",
    exp2_f16,
    tune_exp2_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `exp(input) - 1` elementwise.",
    expm1_f16,
    tune_expm1_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `log` to each input element.",
    log_f16,
    tune_log_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `log2` to each input element.",
    log2_f16,
    tune_log2_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `log10` to each input element.",
    log10_f16,
    tune_log10_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `log(1 + input)` elementwise.",
    log1p_f16,
    tune_log1p_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Converts each input angle from degrees to radians.",
    deg2rad_f16,
    tune_deg2rad_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Converts each input angle from radians to degrees.",
    rad2deg_f16,
    tune_rad2deg_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `sin` to each input element.",
    sin_f16,
    tune_sin_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `cos` to each input element.",
    cos_f16,
    tune_cos_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `tan` to each input element.",
    tan_f16,
    tune_tan_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `sinh` to each input element.",
    sinh_f16,
    tune_sinh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `cosh` to each input element.",
    cosh_f16,
    tune_cosh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `tanh` to each input element.",
    tanh_f16,
    tune_tanh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `asin` to each input element.",
    asin_f16,
    tune_asin_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `acos` to each input element.",
    acos_f16,
    tune_acos_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `atan` to each input element.",
    atan_f16,
    tune_atan_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `ceil` to each input element.",
    ceil_f16,
    tune_ceil_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `floor` to each input element.",
    floor_f16,
    tune_floor_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `trunc` to each input element.",
    trunc_f16,
    tune_trunc_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `frac` to each input element.",
    frac_f16,
    tune_frac_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `1 / input` elementwise.",
    reciprocal_f16,
    tune_reciprocal_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `input * input` elementwise.",
    square_f16,
    tune_square_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Computes `1 / sqrt(input)` elementwise.",
    rsqrt_f16,
    tune_rsqrt_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `cbrt` to each input element.",
    cbrt_f16,
    tune_cbrt_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `asinh` to each input element.",
    asinh_f16,
    tune_asinh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `acosh` to each input element.",
    acosh_f16,
    tune_acosh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Applies `atanh` to each input element.",
    atanh_f16,
    tune_atanh_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
unary_float_fn!(
    "Writes the sign of each input element.",
    sign_f16,
    tune_sign_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
nan_to_num_fn!(
    "Replaces NaNs and infinities with the supplied finite values.",
    nan_to_num_f16,
    tune_nan_to_num_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nan input elements.",
    is_nan_bool_f16,
    tune_is_nan_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for inf input elements.",
    is_inf_bool_f16,
    tune_is_inf_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for pos inf input elements.",
    is_pos_inf_bool_f16,
    tune_is_pos_inf_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for neg inf input elements.",
    is_neg_inf_bool_f16,
    tune_is_neg_inf_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for finite input elements.",
    is_finite_bool_f16,
    tune_is_finite_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for normal input elements.",
    is_normal_bool_f16,
    tune_is_normal_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for subnormal input elements.",
    is_subnormal_bool_f16,
    tune_is_subnormal_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_f16,
    tune_is_zero_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_f16,
    tune_is_nonzero_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_f16,
    tune_is_positive_bool_f16,
    f16
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_f16,
    tune_is_negative_bool_f16,
    f16
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!("Computes `-input` elementwise.", neg_f64, tune_neg_f64, f64);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `abs(input)` elementwise.",
    abs_f64,
    tune_abs_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `sqrt(input)` elementwise.",
    sqrt_f64,
    tune_sqrt_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `exp` to each input element.",
    exp_f64,
    tune_exp_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `2 ** input` elementwise.",
    exp2_f64,
    tune_exp2_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `exp(input) - 1` elementwise.",
    expm1_f64,
    tune_expm1_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `log` to each input element.",
    log_f64,
    tune_log_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `log2` to each input element.",
    log2_f64,
    tune_log2_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `log10` to each input element.",
    log10_f64,
    tune_log10_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `log(1 + input)` elementwise.",
    log1p_f64,
    tune_log1p_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Converts each input angle from degrees to radians.",
    deg2rad_f64,
    tune_deg2rad_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Converts each input angle from radians to degrees.",
    rad2deg_f64,
    tune_rad2deg_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `sin` to each input element.",
    sin_f64,
    tune_sin_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `cos` to each input element.",
    cos_f64,
    tune_cos_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `tan` to each input element.",
    tan_f64,
    tune_tan_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `sinh` to each input element.",
    sinh_f64,
    tune_sinh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `cosh` to each input element.",
    cosh_f64,
    tune_cosh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `tanh` to each input element.",
    tanh_f64,
    tune_tanh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `asin` to each input element.",
    asin_f64,
    tune_asin_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `acos` to each input element.",
    acos_f64,
    tune_acos_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `atan` to each input element.",
    atan_f64,
    tune_atan_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `ceil` to each input element.",
    ceil_f64,
    tune_ceil_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `floor` to each input element.",
    floor_f64,
    tune_floor_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `trunc` to each input element.",
    trunc_f64,
    tune_trunc_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `frac` to each input element.",
    frac_f64,
    tune_frac_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `1 / input` elementwise.",
    reciprocal_f64,
    tune_reciprocal_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `input * input` elementwise.",
    square_f64,
    tune_square_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Computes `1 / sqrt(input)` elementwise.",
    rsqrt_f64,
    tune_rsqrt_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `cbrt` to each input element.",
    cbrt_f64,
    tune_cbrt_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `asinh` to each input element.",
    asinh_f64,
    tune_asinh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `acosh` to each input element.",
    acosh_f64,
    tune_acosh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Applies `atanh` to each input element.",
    atanh_f64,
    tune_atanh_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
unary_float_fn!(
    "Writes the sign of each input element.",
    sign_f64,
    tune_sign_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
nan_to_num_fn!(
    "Replaces NaNs and infinities with the supplied finite values.",
    nan_to_num_f64,
    tune_nan_to_num_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nan input elements.",
    is_nan_bool_f64,
    tune_is_nan_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for inf input elements.",
    is_inf_bool_f64,
    tune_is_inf_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for pos inf input elements.",
    is_pos_inf_bool_f64,
    tune_is_pos_inf_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for neg inf input elements.",
    is_neg_inf_bool_f64,
    tune_is_neg_inf_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for finite input elements.",
    is_finite_bool_f64,
    tune_is_finite_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for normal input elements.",
    is_normal_bool_f64,
    tune_is_normal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for subnormal input elements.",
    is_subnormal_bool_f64,
    tune_is_subnormal_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_f64,
    tune_is_zero_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_f64,
    tune_is_nonzero_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_f64,
    tune_is_positive_bool_f64,
    f64
);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u8"))]
unary_float_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_f64,
    tune_is_negative_bool_f64,
    f64
);

#[cfg(feature = "dtype_i8")]
unary_raw_fn!("Computes `-input` elementwise.", neg_i8, i8);

#[cfg(feature = "dtype_i32")]
unary_raw_fn!("Computes `-input` elementwise.", neg_i32, i32);

#[cfg(feature = "dtype_i64")]
unary_raw_fn!("Computes `-input` elementwise.", neg_i64, i64);

#[cfg(feature = "dtype_i8")]
unary_raw_fn!("Computes `abs(input)` elementwise.", abs_i8, i8);

#[cfg(feature = "dtype_i32")]
unary_raw_fn!("Computes `abs(input)` elementwise.", abs_i32, i32);

#[cfg(feature = "dtype_i64")]
unary_raw_fn!("Computes `abs(input)` elementwise.", abs_i64, i64);

#[cfg(feature = "dtype_i8")]
unary_raw_fn!("Writes the sign of each input element.", sign_i8, i8);

#[cfg(feature = "dtype_i32")]
unary_raw_fn!("Writes the sign of each input element.", sign_i32, i32);

#[cfg(feature = "dtype_i64")]
unary_raw_fn!("Writes the sign of each input element.", sign_i64, i64);

#[cfg(feature = "dtype_u8")]
unary_raw_fn!("Writes the sign of each input element.", sign_u8, u8);

#[cfg(feature = "dtype_u32")]
unary_raw_fn!("Writes the sign of each input element.", sign_u32, u32);

#[cfg(feature = "dtype_u64")]
unary_raw_fn!("Writes the sign of each input element.", sign_u64, u64);

#[cfg(feature = "dtype_u8")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_u8, u8);

#[cfg(feature = "dtype_i8")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_i8, i8);

#[cfg(feature = "dtype_u32")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_u32, u32);

#[cfg(feature = "dtype_i32")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_i32, i32);

#[cfg(feature = "dtype_u64")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_u64, u64);

#[cfg(feature = "dtype_i64")]
unary_raw_fn!("Computes `~input` elementwise.", bitwise_not_i64, i64);

#[cfg(feature = "dtype_u8")]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_u8,
    u8
);

#[cfg(feature = "dtype_u8")]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_u8,
    u8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_i8", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_i8,
    i8
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_u32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_u32,
    u32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_i32", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_i32,
    i32
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_u64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_u64,
    u64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for zero input elements.",
    is_zero_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for nonzero input elements.",
    is_nonzero_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for positive input elements.",
    is_positive_bool_i64,
    i64
);

#[cfg(all(feature = "dtype_i64", feature = "dtype_u8"))]
unary_bool_fn!(
    "Writes a byte predicate for negative input elements.",
    is_negative_bool_i64,
    i64
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::autotune::KernelCompileConfig;

    #[test]
    fn test_vector_config_builders_preserve_tuning_choice() {
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
    fn test_vector_tile_candidates_cover_cu_tile_ladder() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_unary_validation_accepts_exact_lengths() -> Result<()> {
        validate_vector_lengths(24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_unary_validation_rejects_short_input() {
        assert!(validate_vector_lengths(24, 23, VectorConfig::create(24)).is_err());
    }
}
