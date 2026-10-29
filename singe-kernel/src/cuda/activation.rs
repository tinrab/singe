//! Elementwise nonlinearities and gated activations.

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

fn launch_activation_vector<F>(
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

fn tune_activation_vector<F>(
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

fn validate_swiglu_lengths(
    out_len: usize,
    input_len: usize,
    gate_len: usize,
    config: VectorConfig,
) -> Result<()> {
    validate_vector_lengths(out_len, input_len, config)?;
    ensure_len(gate_len, config.length)?;
    Ok(())
}

macro_rules! activation_fn {
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
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::activation::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
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
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::activation::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
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

macro_rules! scalar_activation_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty, $param:ident) => {
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
            $param: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::activation::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            $param,
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
            $param: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::activation::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        $param,
                        config.length,
                        Some(tuning.tile_width),
                        tuning.compile,
                    )
                },
            )
        }
    };
}

macro_rules! binary_scalar_activation_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty, $first:ident, $second:ident) => {
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
            $first: $ty,
            $second: $ty,
            config: VectorConfig,
        ) -> Result<()> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            launch_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                |tuning| {
                    let (tile_width, compile) = vector_launch_parts(tuning);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        cutile::activation::$name(
                            &cuda_stream,
                            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                            input_pointer(input),
                            $first,
                            $second,
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
            $first: $ty,
            $second: $ty,
            config: VectorConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<VectorTuning>> {
            validate_vector_lengths(out.len(), input.len(), config)?;
            let cuda_stream = stream.to_cuda_core()?;
            tune_activation_vector(
                stream,
                stringify!($name),
                stringify!($ty),
                config,
                options,
                // SAFETY: the caller upholds this function's safety contract.
                |tuning| unsafe {
                    cutile::activation::$name(
                        &cuda_stream,
                        DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                        input_pointer(input),
                        $first,
                        $second,
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
activation_fn!(
    "Applies `max(x, 0)` elementwise.",
    relu_f32,
    tune_relu_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Clamps every input element to `[0, 6]`.",
    relu6_f32,
    tune_relu6_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Applies the logistic sigmoid elementwise.",
    sigmoid_f32,
    tune_sigmoid_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes `x * sigmoid(x)` elementwise.",
    silu_f32,
    tune_silu_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Applies the tanh approximation of GELU elementwise.",
    gelu_f32,
    tune_gelu_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Applies softplus elementwise.",
    softplus_f32,
    tune_softplus_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Applies the piecewise linear hard sigmoid elementwise.",
    hard_sigmoid_f32,
    tune_hard_sigmoid_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes `x * hard_sigmoid(x)` elementwise.",
    hard_swish_f32,
    tune_hard_swish_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes `x * tanh(softplus(x))` elementwise.",
    mish_f32,
    tune_mish_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Applies the scaled exponential linear activation elementwise.",
    selu_f32,
    tune_selu_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes `x / (1 + abs(x))` elementwise.",
    softsign_f32,
    tune_softsign_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes `x - tanh(x)` elementwise.",
    tanhshrink_f32,
    tune_tanhshrink_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
activation_fn!(
    "Computes the logarithm of the logistic sigmoid elementwise.",
    log_sigmoid_f32,
    tune_log_sigmoid_f32,
    f32
);

#[cfg(feature = "dtype_f32")]
scalar_activation_fn!(
    "Applies `ReLU` with `negative_slope` for negative inputs.",
    leaky_relu_f32,
    tune_leaky_relu_f32,
    f32,
    negative_slope
);

#[cfg(feature = "dtype_f32")]
scalar_activation_fn!(
    "Applies the exponential linear activation with parameter `alpha`.",
    elu_f32,
    tune_elu_f32,
    f32,
    alpha
);

#[cfg(feature = "dtype_f32")]
scalar_activation_fn!(
    "Applies the continuously differentiable exponential linear activation with parameter `alpha`.",
    celu_f32,
    tune_celu_f32,
    f32,
    alpha
);

#[cfg(feature = "dtype_f32")]
scalar_activation_fn!(
    "Replaces inputs in `[-lambda, lambda]` with zero.",
    hardshrink_f32,
    tune_hardshrink_f32,
    f32,
    lambda
);

#[cfg(feature = "dtype_f32")]
scalar_activation_fn!(
    "Shrinks input magnitudes by `lambda`, clamping the result at zero.",
    softshrink_f32,
    tune_softshrink_f32,
    f32,
    lambda
);

#[cfg(feature = "dtype_f32")]
binary_scalar_activation_fn!(
    "Replaces inputs at or below `threshold` with `value`.",
    threshold_f32,
    tune_threshold_f32,
    f32,
    threshold,
    value
);

#[cfg(feature = "dtype_f32")]
binary_scalar_activation_fn!(
    "Clamps each input to `[min_value, max_value]`.",
    hardtanh_f32,
    tune_hardtanh_f32,
    f32,
    min_value,
    max_value
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies `max(x, 0)` elementwise.",
    relu_f16,
    tune_relu_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Clamps every input element to `[0, 6]`.",
    relu6_f16,
    tune_relu6_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies the logistic sigmoid elementwise.",
    sigmoid_f16,
    tune_sigmoid_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes `x * sigmoid(x)` elementwise.",
    silu_f16,
    tune_silu_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies the tanh approximation of GELU elementwise.",
    gelu_f16,
    tune_gelu_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies softplus elementwise.",
    softplus_f16,
    tune_softplus_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies the piecewise linear hard sigmoid elementwise.",
    hard_sigmoid_f16,
    tune_hard_sigmoid_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes `x * hard_sigmoid(x)` elementwise.",
    hard_swish_f16,
    tune_hard_swish_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes `x * tanh(softplus(x))` elementwise.",
    mish_f16,
    tune_mish_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Applies the scaled exponential linear activation elementwise.",
    selu_f16,
    tune_selu_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes `x / (1 + abs(x))` elementwise.",
    softsign_f16,
    tune_softsign_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes `x - tanh(x)` elementwise.",
    tanhshrink_f16,
    tune_tanhshrink_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
activation_fn!(
    "Computes the logarithm of the logistic sigmoid elementwise.",
    log_sigmoid_f16,
    tune_log_sigmoid_f16,
    f16
);

#[cfg(feature = "dtype_f16")]
scalar_activation_fn!(
    "Applies `ReLU` with `negative_slope` for negative inputs.",
    leaky_relu_f16,
    tune_leaky_relu_f16,
    f16,
    negative_slope
);

#[cfg(feature = "dtype_f16")]
scalar_activation_fn!(
    "Applies the exponential linear activation with parameter `alpha`.",
    elu_f16,
    tune_elu_f16,
    f16,
    alpha
);

#[cfg(feature = "dtype_f16")]
scalar_activation_fn!(
    "Applies the continuously differentiable exponential linear activation with parameter `alpha`.",
    celu_f16,
    tune_celu_f16,
    f16,
    alpha
);

#[cfg(feature = "dtype_f16")]
scalar_activation_fn!(
    "Replaces inputs in `[-lambda, lambda]` with zero.",
    hardshrink_f16,
    tune_hardshrink_f16,
    f16,
    lambda
);

#[cfg(feature = "dtype_f16")]
scalar_activation_fn!(
    "Shrinks input magnitudes by `lambda`, clamping the result at zero.",
    softshrink_f16,
    tune_softshrink_f16,
    f16,
    lambda
);

#[cfg(feature = "dtype_f16")]
binary_scalar_activation_fn!(
    "Replaces inputs at or below `threshold` with `value`.",
    threshold_f16,
    tune_threshold_f16,
    f16,
    threshold,
    value
);

#[cfg(feature = "dtype_f16")]
binary_scalar_activation_fn!(
    "Clamps each input to `[min_value, max_value]`.",
    hardtanh_f16,
    tune_hardtanh_f16,
    f16,
    min_value,
    max_value
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies `max(x, 0)` elementwise.",
    relu_f64,
    tune_relu_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Clamps every input element to `[0, 6]`.",
    relu6_f64,
    tune_relu6_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies the logistic sigmoid elementwise.",
    sigmoid_f64,
    tune_sigmoid_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes `x * sigmoid(x)` elementwise.",
    silu_f64,
    tune_silu_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies the tanh approximation of GELU elementwise.",
    gelu_f64,
    tune_gelu_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies softplus elementwise.",
    softplus_f64,
    tune_softplus_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies the piecewise linear hard sigmoid elementwise.",
    hard_sigmoid_f64,
    tune_hard_sigmoid_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes `x * hard_sigmoid(x)` elementwise.",
    hard_swish_f64,
    tune_hard_swish_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes `x * tanh(softplus(x))` elementwise.",
    mish_f64,
    tune_mish_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Applies the scaled exponential linear activation elementwise.",
    selu_f64,
    tune_selu_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes `x / (1 + abs(x))` elementwise.",
    softsign_f64,
    tune_softsign_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes `x - tanh(x)` elementwise.",
    tanhshrink_f64,
    tune_tanhshrink_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
activation_fn!(
    "Computes the logarithm of the logistic sigmoid elementwise.",
    log_sigmoid_f64,
    tune_log_sigmoid_f64,
    f64
);

#[cfg(feature = "dtype_f64")]
scalar_activation_fn!(
    "Applies `ReLU` with `negative_slope` for negative inputs.",
    leaky_relu_f64,
    tune_leaky_relu_f64,
    f64,
    negative_slope
);

#[cfg(feature = "dtype_f64")]
scalar_activation_fn!(
    "Applies the exponential linear activation with parameter `alpha`.",
    elu_f64,
    tune_elu_f64,
    f64,
    alpha
);

#[cfg(feature = "dtype_f64")]
scalar_activation_fn!(
    "Applies the continuously differentiable exponential linear activation with parameter `alpha`.",
    celu_f64,
    tune_celu_f64,
    f64,
    alpha
);

#[cfg(feature = "dtype_f64")]
scalar_activation_fn!(
    "Replaces inputs in `[-lambda, lambda]` with zero.",
    hardshrink_f64,
    tune_hardshrink_f64,
    f64,
    lambda
);

#[cfg(feature = "dtype_f64")]
scalar_activation_fn!(
    "Shrinks input magnitudes by `lambda`, clamping the result at zero.",
    softshrink_f64,
    tune_softshrink_f64,
    f64,
    lambda
);

#[cfg(feature = "dtype_f64")]
binary_scalar_activation_fn!(
    "Replaces inputs at or below `threshold` with `value`.",
    threshold_f64,
    tune_threshold_f64,
    f64,
    threshold,
    value
);

#[cfg(feature = "dtype_f64")]
binary_scalar_activation_fn!(
    "Clamps each input to `[min_value, max_value]`.",
    hardtanh_f64,
    tune_hardtanh_f64,
    f64,
    min_value,
    max_value
);

#[cfg(feature = "dtype_f32")]
/// Computes `input * silu(gate)` elementwise for the configured vector.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
pub unsafe fn swiglu_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    gate: &impl DeviceInput<f32>,
    config: VectorConfig,
) -> Result<()> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    launch_activation_vector(stream, "swiglu_f32", "f32", config, |tuning| {
        let (tile_width, compile) = vector_launch_parts(tuning);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cutile::activation::swiglu_f32(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                tile_width,
                compile,
            )
        }
    })
}

#[cfg(feature = "dtype_f32")]
/// Times candidate tiles and enqueues the selected operation.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
/// The access and lifetime requirements apply to every candidate launch.
pub unsafe fn tune_swiglu_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    gate: &impl DeviceInput<f32>,
    config: VectorConfig,
    options: TuneOptions,
) -> Result<TuneReport<VectorTuning>> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    tune_activation_vector(
        stream,
        "swiglu_f32",
        "f32",
        config,
        options,
        // SAFETY: the caller upholds this function's safety contract.
        |tuning| unsafe {
            cutile::activation::swiglu_f32(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                Some(tuning.tile_width),
                tuning.compile,
            )
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Computes `input * silu(gate)` elementwise for the configured vector.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
pub unsafe fn swiglu_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    gate: &impl DeviceInput<f16>,
    config: VectorConfig,
) -> Result<()> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    launch_activation_vector(stream, "swiglu_f16", "f16", config, |tuning| {
        let (tile_width, compile) = vector_launch_parts(tuning);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cutile::activation::swiglu_f16(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                tile_width,
                compile,
            )
        }
    })
}

#[cfg(feature = "dtype_f16")]
/// Times candidate tiles and enqueues the selected operation.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
/// The access and lifetime requirements apply to every candidate launch.
pub unsafe fn tune_swiglu_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    gate: &impl DeviceInput<f16>,
    config: VectorConfig,
    options: TuneOptions,
) -> Result<TuneReport<VectorTuning>> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    tune_activation_vector(
        stream,
        "swiglu_f16",
        "f16",
        config,
        options,
        // SAFETY: the caller upholds this function's safety contract.
        |tuning| unsafe {
            cutile::activation::swiglu_f16(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                Some(tuning.tile_width),
                tuning.compile,
            )
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Computes `input * silu(gate)` elementwise for the configured vector.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
pub unsafe fn swiglu_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    gate: &impl DeviceInput<f64>,
    config: VectorConfig,
) -> Result<()> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    launch_activation_vector(stream, "swiglu_f64", "f64", config, |tuning| {
        let (tile_width, compile) = vector_launch_parts(tuning);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cutile::activation::swiglu_f64(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                tile_width,
                compile,
            )
        }
    })
}

#[cfg(feature = "dtype_f64")]
/// Times candidate tiles and enqueues the selected operation.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
/// The access and lifetime requirements apply to every candidate launch.
pub unsafe fn tune_swiglu_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    gate: &impl DeviceInput<f64>,
    config: VectorConfig,
    options: TuneOptions,
) -> Result<TuneReport<VectorTuning>> {
    validate_swiglu_lengths(out.len(), input.len(), gate.len(), config)?;
    let cuda_stream = stream.to_cuda_core()?;
    tune_activation_vector(
        stream,
        "swiglu_f64",
        "f64",
        config,
        options,
        // SAFETY: the caller upholds this function's safety contract.
        |tuning| unsafe {
            cutile::activation::swiglu_f64(
                &cuda_stream,
                DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                input_pointer(input),
                input_pointer(gate),
                config.length,
                Some(tuning.tile_width),
                tuning.compile,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_vector_config_builders_preserve_tuning_choice() {
        let tuning =
            VectorTuning::tile(64).with_compile(KernelCompileConfig::default().occupancy(2));
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
    fn test_activation_uses_shared_vector_tile_candidates() {
        let tile_widths = vector_tile_candidates()
            .into_iter()
            .map(|candidate| candidate.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(tile_widths, vec![32, 64, 128, 256]);
    }

    #[test]
    fn test_swiglu_validation_accepts_exact_lengths() -> Result<()> {
        validate_swiglu_lengths(24, 24, 24, VectorConfig::create(24))
    }

    #[test]
    fn test_swiglu_validation_rejects_short_gate() {
        assert!(validate_swiglu_lengths(24, 24, 23, VectorConfig::create(24)).is_err());
    }
}
