//! Sum, max, norm, and row/axis reductions.

use std::sync::Arc;

#[cfg(feature = "cutile")]
use ::cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream as CutileStream};
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceRepr},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::{
        autotune::{
            self, DeviceFingerprint, KernelCompileConfig, TuneOptions, TuneProblemKey, TuneReport,
            TuningChoice,
        },
        interop::input_pointer,
    },
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReductionConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<ReductionTuning>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReductionTuning {
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgReductionConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<ReductionTuning>,
}

impl ReductionConfig {
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<ReductionTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: ReductionTuning) -> Self {
        self.with_tuning(TuningChoice::Explicit(tuning))
    }

    #[must_use]
    pub const fn cached_tuning(self) -> Self {
        self.with_tuning(TuningChoice::Cached)
    }

    #[must_use]
    pub const fn tune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Tune(options))
    }

    #[must_use]
    pub const fn retune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Retune(options))
    }
}

impl ArgReductionConfig {
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<ReductionTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: ReductionTuning) -> Self {
        self.with_tuning(TuningChoice::Explicit(tuning))
    }

    #[must_use]
    pub const fn cached_tuning(self) -> Self {
        self.with_tuning(TuningChoice::Cached)
    }

    #[must_use]
    pub const fn tune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Tune(options))
    }

    #[must_use]
    pub const fn retune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Retune(options))
    }
}

impl ReductionTuning {
    #[must_use]
    pub const fn tile_width(tile_width: usize) -> Self {
        Self {
            tile_width,
            compile: KernelCompileConfig {
                occupancy: None,
                num_cta_in_cga: None,
                max_divisibility: None,
            },
        }
    }

    #[must_use]
    pub const fn compile(mut self, compile: KernelCompileConfig) -> Self {
        self.compile = compile;
        self
    }
}

fn reduction_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: ReductionConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Reduction {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn arg_reduction_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: ArgReductionConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Reduction {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn reduction_candidates(columns: usize) -> Vec<ReductionTuning> {
    [
        ReductionTuning::tile_width(1),
        ReductionTuning::tile_width(8),
        ReductionTuning::tile_width(32),
        ReductionTuning::tile_width(64),
        ReductionTuning::tile_width(128),
        ReductionTuning::tile_width(256),
        ReductionTuning::tile_width(512),
        ReductionTuning::tile_width(1024),
    ]
    .into_iter()
    .filter(|tuning| tuning.tile_width >= columns)
    .collect()
}

fn arg_reduction_candidates(columns: usize) -> Vec<ReductionTuning> {
    [1, 8, 128, 256, 512, 1024]
        .into_iter()
        .map(ReductionTuning::tile_width)
        .filter(|tuning| tuning.tile_width >= columns)
        .collect()
}

fn validate_reduction<T: DeviceRepr>(
    out: &impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: ReductionConfig,
) -> Result<()> {
    let len = checked_element_count(config.rows, config.columns)?;
    ensure_len(out.len(), config.rows)?;
    ensure_len(input.len(), len)?;
    if config.columns == 0 {
        return Err(Error::InvalidLength);
    }
    if config.columns > 1024 {
        return Err(Error::UnsupportedWidth {
            op: "reduction".into(),
            width: config.columns,
        });
    }
    Ok(())
}

fn validate_arg_reduction<T: DeviceRepr>(
    out: &impl DeviceOutput<i32>,
    input: &impl DeviceInput<T>,
    config: ArgReductionConfig,
) -> Result<()> {
    let len = checked_element_count(config.rows, config.columns)?;
    let out_len = checked_element_count(config.rows, 2)?;
    ensure_len(out.len(), out_len)?;
    ensure_len(input.len(), len)?;
    if config.columns == 0 {
        return Err(Error::InvalidLength);
    }
    if config.columns > 1024 {
        return Err(Error::UnsupportedWidth {
            op: "arg reduction".into(),
            width: config.columns,
        });
    }
    Ok(())
}

fn resolve_reduction_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<ReductionTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<ReductionTuning>>
where
    C: FnOnce() -> Vec<ReductionTuning>,
    F: FnOnce(TuneOptions, Vec<ReductionTuning>) -> Result<TuneReport<ReductionTuning>>,
{
    match choice {
        TuningChoice::Default => Ok(None),
        TuningChoice::Explicit(tuning) => Ok(Some(tuning)),
        TuningChoice::Cached => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                Ok(Some(tuning))
            } else {
                Err(autotune::cache_miss_error(op, stream.capture_status()?))
            }
        }
        TuningChoice::Tune(options) => {
            if let Some(tuning) = autotune::cache().get(&key)? {
                return Ok(Some(tuning));
            }
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
        TuningChoice::Retune(options) => {
            let report = tune(options, candidates())?;
            autotune::cache().insert(key, report.best);
            Ok(Some(report.best))
        }
    }
}

fn launch_reduction_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: ReductionConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<CutileStream>, DevicePointer<T>, DevicePointer<T>, usize, usize) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        config.rows,
        config.columns,
    )
}

fn launch_reduction_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: ReductionConfig,
    tuning: ReductionTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn reduction<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: ReductionConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault:
        FnOnce(&Arc<CutileStream>, DevicePointer<T>, DevicePointer<T>, usize, usize) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_reduction(out, input, config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_reduction_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_reduction_with_tuning(stream, out, input, config, tuning, launch_tuned);
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = reduction_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_reduction_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || reduction_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_reduction_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit reduction tuning return before cache resolution"),
        |tuning| launch_reduction_with_tuning(stream, out, input, config, tuning, launch_tuned),
    )
}

fn tune_reduction<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: ReductionConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<ReductionTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_reduction(out, input, config)?;
    let key = reduction_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        reduction_candidates(config.columns),
        |tuning| launch_reduction_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_reduction_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

fn launch_arg_reduction_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    input: &impl DeviceInput<T>,
    config: ArgReductionConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(&Arc<CutileStream>, DevicePointer<i32>, DevicePointer<T>, usize, usize) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        config.rows,
        config.columns,
    )
}

fn launch_arg_reduction_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    input: &impl DeviceInput<T>,
    config: ArgReductionConfig,
    tuning: ReductionTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<i32>,
        DevicePointer<T>,
        usize,
        usize,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn arg_reduction<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    input: &impl DeviceInput<T>,
    config: ArgReductionConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<i32>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<i32>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_arg_reduction(out, input, config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_arg_reduction_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_arg_reduction_with_tuning(
                stream,
                out,
                input,
                config,
                tuning,
                launch_tuned,
            );
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = arg_reduction_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_reduction_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || arg_reduction_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_arg_reduction_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit arg reduction tuning return before cache resolution"),
        |tuning| launch_arg_reduction_with_tuning(stream, out, input, config, tuning, launch_tuned),
    )
}

fn tune_arg_reduction<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    input: &impl DeviceInput<T>,
    config: ArgReductionConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<ReductionTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<i32>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_arg_reduction(out, input, config)?;
    let key = arg_reduction_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        arg_reduction_candidates(config.columns),
        |tuning| launch_arg_reduction_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_arg_reduction_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

macro_rules! autotuned_reduction_fn {
    ($operation:literal, $name:ident, $tune_name:ident, $tuned_backend:ident, $ty:ty, $dtype:literal) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: ReductionConfig,
        ) -> Result<()> {
            reduction(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::reduction::$name(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::reduction::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }

        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: ReductionConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<ReductionTuning>> {
            tune_reduction(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::reduction::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_f32,
    tune_sum_f32,
    sum_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the mean of every input row.",
    mean_f32,
    tune_mean_f32,
    mean_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the variance of every input row.",
    variance_f32,
    tune_variance_f32,
    variance_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the std of every input row.",
    std_f32,
    tune_std_f32,
    std_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_f32,
    tune_max_f32,
    max_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_f32,
    tune_min_f32,
    min_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_reduction_fn!(
    "Computes the product of every input row.",
    product_f32,
    tune_product_f32,
    product_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_f16,
    tune_sum_f16,
    sum_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the mean of every input row.",
    mean_f16,
    tune_mean_f16,
    mean_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the variance of every input row.",
    variance_f16,
    tune_variance_f16,
    variance_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the std of every input row.",
    std_f16,
    tune_std_f16,
    std_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_f16,
    tune_max_f16,
    max_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_f16,
    tune_min_f16,
    min_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_reduction_fn!(
    "Computes the product of every input row.",
    product_f16,
    tune_product_f16,
    product_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_f64,
    tune_sum_f64,
    sum_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the mean of every input row.",
    mean_f64,
    tune_mean_f64,
    mean_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the variance of every input row.",
    variance_f64,
    tune_variance_f64,
    variance_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the std of every input row.",
    std_f64,
    tune_std_f64,
    std_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_f64,
    tune_max_f64,
    max_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_f64,
    tune_min_f64,
    min_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_reduction_fn!(
    "Computes the product of every input row.",
    product_f64,
    tune_product_f64,
    product_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_u8")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_u8,
    tune_sum_u8,
    sum_u8_tuned,
    u8,
    "u8"
);

#[cfg(feature = "dtype_u8")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_u8,
    tune_max_u8,
    max_u8_tuned,
    u8,
    "u8"
);

#[cfg(feature = "dtype_u8")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_u8,
    tune_min_u8,
    min_u8_tuned,
    u8,
    "u8"
);

#[cfg(feature = "dtype_i8")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_i8,
    tune_sum_i8,
    sum_i8_tuned,
    i8,
    "i8"
);

#[cfg(feature = "dtype_i8")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_i8,
    tune_max_i8,
    max_i8_tuned,
    i8,
    "i8"
);

#[cfg(feature = "dtype_i8")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_i8,
    tune_min_i8,
    min_i8_tuned,
    i8,
    "i8"
);

#[cfg(feature = "dtype_u32")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_u32,
    tune_sum_u32,
    sum_u32_tuned,
    u32,
    "u32"
);

#[cfg(feature = "dtype_u32")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_u32,
    tune_max_u32,
    max_u32_tuned,
    u32,
    "u32"
);

#[cfg(feature = "dtype_u32")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_u32,
    tune_min_u32,
    min_u32_tuned,
    u32,
    "u32"
);

#[cfg(feature = "dtype_i32")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_i32,
    tune_sum_i32,
    sum_i32_tuned,
    i32,
    "i32"
);

#[cfg(feature = "dtype_i32")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_i32,
    tune_max_i32,
    max_i32_tuned,
    i32,
    "i32"
);

#[cfg(feature = "dtype_i32")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_i32,
    tune_min_i32,
    min_i32_tuned,
    i32,
    "i32"
);

#[cfg(feature = "dtype_u64")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_u64,
    tune_sum_u64,
    sum_u64_tuned,
    u64,
    "u64"
);

#[cfg(feature = "dtype_u64")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_u64,
    tune_max_u64,
    max_u64_tuned,
    u64,
    "u64"
);

#[cfg(feature = "dtype_u64")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_u64,
    tune_min_u64,
    min_u64_tuned,
    u64,
    "u64"
);

#[cfg(feature = "dtype_i64")]
autotuned_reduction_fn!(
    "Computes the sum of every input row.",
    sum_i64,
    tune_sum_i64,
    sum_i64_tuned,
    i64,
    "i64"
);

#[cfg(feature = "dtype_i64")]
autotuned_reduction_fn!(
    "Computes the max of every input row.",
    max_i64,
    tune_max_i64,
    max_i64_tuned,
    i64,
    "i64"
);

#[cfg(feature = "dtype_i64")]
autotuned_reduction_fn!(
    "Computes the min of every input row.",
    min_i64,
    tune_min_i64,
    min_i64_tuned,
    i64,
    "i64"
);

macro_rules! autotuned_arg_reduction_fn {
    ($operation:literal, $name:ident, $tune_name:ident, $tuned_backend:ident, $ty:ty, $dtype:literal) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<i32>,
            input: &impl DeviceInput<$ty>,
            config: ArgReductionConfig,
        ) -> Result<()> {
            arg_reduction(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::reduction::$name(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::reduction::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }

        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<i32>,
            input: &impl DeviceInput<$ty>,
            config: ArgReductionConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<ReductionTuning>> {
            tune_arg_reduction(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::reduction::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
autotuned_arg_reduction_fn!(
    "Writes the first extremum index as two i32 words. The first NaN takes precedence.",
    argmax_f32,
    tune_argmax_f32,
    argmax_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f32")]
autotuned_arg_reduction_fn!(
    "Writes the first extremum index as two i32 words. The first NaN takes precedence.",
    argmin_f32,
    tune_argmin_f32,
    argmin_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f16")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_f16,
    tune_argmax_f16,
    argmax_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f16")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_f16,
    tune_argmin_f16,
    argmin_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_f64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_f64,
    tune_argmax_f64,
    argmax_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_f64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_f64,
    tune_argmin_f64,
    argmin_f64_tuned,
    f64,
    "f64"
);

#[cfg(feature = "dtype_u8")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_u8,
    tune_argmax_u8,
    argmax_u8_tuned,
    u8,
    "u8"
);

#[cfg(feature = "dtype_u8")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_u8,
    tune_argmin_u8,
    argmin_u8_tuned,
    u8,
    "u8"
);

#[cfg(feature = "dtype_i8")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_i8,
    tune_argmax_i8,
    argmax_i8_tuned,
    i8,
    "i8"
);

#[cfg(feature = "dtype_i8")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_i8,
    tune_argmin_i8,
    argmin_i8_tuned,
    i8,
    "i8"
);

#[cfg(feature = "dtype_u32")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_u32,
    tune_argmax_u32,
    argmax_u32_tuned,
    u32,
    "u32"
);

#[cfg(feature = "dtype_u32")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_u32,
    tune_argmin_u32,
    argmin_u32_tuned,
    u32,
    "u32"
);

#[cfg(feature = "dtype_i32")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_i32,
    tune_argmax_i32,
    argmax_i32_tuned,
    i32,
    "i32"
);

#[cfg(feature = "dtype_i32")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_i32,
    tune_argmin_i32,
    argmin_i32_tuned,
    i32,
    "i32"
);

#[cfg(feature = "dtype_u64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_u64,
    tune_argmax_u64,
    argmax_u64_tuned,
    u64,
    "u64"
);

#[cfg(feature = "dtype_u64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_u64,
    tune_argmin_u64,
    argmin_u64_tuned,
    u64,
    "u64"
);

#[cfg(feature = "dtype_i64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmax_i64,
    tune_argmax_i64,
    argmax_i64_tuned,
    i64,
    "i64"
);

#[cfg(feature = "dtype_i64")]
autotuned_arg_reduction_fn!(
    "Writes each row's extremum index as a pair of i32 words representing an i64 value.",
    argmin_i64,
    tune_argmin_i64,
    argmin_i64_tuned,
    i64,
    "i64"
);

macro_rules! block_argmax_fn {
    ($operation:literal, $name:ident, $ty:ty, $max_ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            block_max: &mut impl DeviceOutput<$max_ty>,
            block_idx: &mut impl DeviceOutput<i32>,
            input: &impl DeviceInput<$ty>,
        ) -> Result<usize> {
            let len = input.len();
            if len == 0 {
                return Err(Error::InvalidLength);
            }
            let num_blocks = len.div_ceil(256);
            ensure_len(block_max.len(), num_blocks)?;
            ensure_len(block_idx.len(), num_blocks)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::reduction::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(block_max.as_mut_ptr() as _),
                    DevicePointer::from_cu_deviceptr(block_idx.as_mut_ptr() as _),
                    input_pointer(input),
                    len,
                )
            }
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmax_f16,
    f16,
    f32
);

#[cfg(feature = "dtype_f32")]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmax_f32,
    f32,
    f32
);

#[cfg(feature = "dtype_f64")]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmax_f64,
    f64,
    f64
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmin_f16,
    f16,
    f32
);

#[cfg(feature = "dtype_f32")]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmin_f32,
    f32,
    f32
);

#[cfg(feature = "dtype_f64")]
block_argmax_fn!(
    "Writes an extremum and its index for each 256-element input block.",
    block_argmin_f64,
    f64,
    f64
);

macro_rules! reduce_block_argmax_fn {
    ($operation:literal, $name:ident, $max_ty:ty) => {
        #[doc = $operation]
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<i32>,
            block_max: &impl DeviceInput<$max_ty>,
            block_idx: &impl DeviceInput<i32>,
            num_blocks: usize,
        ) -> Result<()> {
            if num_blocks == 0 {
                return Err(Error::InvalidLength);
            }
            ensure_len(out.len(), 1)?;
            ensure_len(block_max.len(), num_blocks)?;
            ensure_len(block_idx.len(), num_blocks)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::reduction::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(block_max),
                    input_pointer(block_idx),
                    num_blocks,
                )
            }
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmax_f16,
    f32
);

#[cfg(feature = "dtype_f32")]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmax_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmax_f64,
    f64
);

#[cfg(all(feature = "dtype_f16", feature = "dtype_f32"))]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmin_f16,
    f32
);

#[cfg(feature = "dtype_f32")]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmin_f32,
    f32
);

#[cfg(feature = "dtype_f64")]
reduce_block_argmax_fn!(
    "Reduces block extrema and writes the selected original input index.",
    reduce_block_argmin_f64,
    f64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_config_builders_preserve_tuning_choice() {
        let options = TuneOptions::create(0, 1, 2);
        let tuning =
            ReductionTuning::tile_width(32).compile(KernelCompileConfig::default().occupancy(4));

        assert_eq!(
            ReductionConfig::create(2, 16)
                .explicit_tuning(tuning)
                .tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            ReductionConfig::create(2, 16).cached_tuning().tuning,
            TuningChoice::Cached
        );
        assert_eq!(
            ReductionConfig::create(2, 16).tune(options).tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            ReductionConfig::create(2, 16).retune(options).tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_arg_reduction_config_builders_preserve_tuning_choice() {
        let options = TuneOptions::create(0, 1, 2);
        let tuning = ReductionTuning::tile_width(128);

        assert_eq!(
            ArgReductionConfig::create(3, 64)
                .explicit_tuning(tuning)
                .tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            ArgReductionConfig::create(3, 64).cached_tuning().tuning,
            TuningChoice::Cached
        );
        assert_eq!(
            ArgReductionConfig::create(3, 64).tune(options).tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            ArgReductionConfig::create(3, 64).retune(options).tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_reduction_candidates_only_include_launchable_widths() {
        let candidates = reduction_candidates(33)
            .into_iter()
            .map(|tuning| tuning.tile_width)
            .collect::<Vec<_>>();

        assert_eq!(candidates, vec![64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_arg_reduction_candidates_use_arg_width_ladder() {
        let candidates = arg_reduction_candidates(33)
            .into_iter()
            .map(|tuning| tuning.tile_width)
            .collect::<Vec<_>>();

        assert_eq!(candidates, vec![128, 256, 512, 1024]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_reduction_tuning_and_config_round_trip_through_json() {
        let tuning =
            ReductionTuning::tile_width(128).compile(KernelCompileConfig::default().occupancy(2));
        let config = ReductionConfig::create(3, 17).explicit_tuning(tuning);

        let tuning_json = serde_json::to_string(&tuning).unwrap();
        let config_json = serde_json::to_string(&config).unwrap();
        let decoded_tuning: ReductionTuning = serde_json::from_str(&tuning_json).unwrap();
        let decoded_config: ReductionConfig = serde_json::from_str(&config_json).unwrap();

        assert_eq!(decoded_tuning, tuning);
        assert_eq!(decoded_config, config);
    }
}
