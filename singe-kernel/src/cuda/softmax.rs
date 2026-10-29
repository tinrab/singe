//! Softmax and log-sum-exp style kernels.

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
            self, DeviceFingerprint, KernelCompileConfig, Tile2d, TuneOptions, TuneProblemKey,
            TuneReport, TuningChoice,
        },
        interop::input_pointer,
    },
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftmaxConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<SoftmaxTuning>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftmaxTuning {
    pub tile: Tile2d,
    pub compile: KernelCompileConfig,
}

impl SoftmaxConfig {
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<SoftmaxTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: SoftmaxTuning) -> Self {
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

impl SoftmaxTuning {
    #[must_use]
    pub const fn tile(rows: usize, columns: usize) -> Self {
        Self {
            tile: Tile2d::create(rows, columns),
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

macro_rules! softmax_fns {
    (
        $ty:ty,
        $dtype:literal,
        $softmax:ident,
        $tune_softmax:ident,
        $softmax_tuned:ident,
        $log_softmax:ident,
        $tune_log_softmax:ident,
        $log_softmax_tuned:ident,
        $softmin:ident,
        $tune_softmin:ident,
        $softmin_tuned:ident
    ) => {
        /// Normalizes the exponentials of row elements to sum to one.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $softmax(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
        ) -> Result<()> {
            softmax(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($softmax),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::softmax::$softmax(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$softmax_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        /// Tunes row normalization using the supplied output storage.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_softmax(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<SoftmaxTuning>> {
            tune_softmax(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($softmax),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$softmax_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        /// Writes each input element minus its row log-sum-exp.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $log_softmax(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
        ) -> Result<()> {
            softmax(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($log_softmax),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::softmax::$log_softmax(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$log_softmax_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied output storage.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_log_softmax(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<SoftmaxTuning>> {
            tune_softmax(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($log_softmax),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$log_softmax_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                    )
                },
            )
        }

        /// Normalizes the exponentials of negated row elements to sum to one.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $softmin(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
        ) -> Result<()> {
            softmax(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($softmin),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::softmax::$softmin(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$softmin_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        /// Tunes row normalization using the supplied output storage.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Writable output and scratch ranges must also be mutually disjoint.
        pub unsafe fn $tune_softmin(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SoftmaxConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<SoftmaxTuning>> {
            tune_softmax(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($softmin),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::softmax::$softmin_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
softmax_fns!(
    f32,
    "f32",
    softmax_f32,
    tune_softmax_f32,
    softmax_f32_tuned,
    log_softmax_f32,
    tune_log_softmax_f32,
    log_softmax_f32_tuned,
    softmin_f32,
    tune_softmin_f32,
    softmin_f32_tuned
);

#[cfg(feature = "dtype_f16")]
softmax_fns!(
    f16,
    "f16",
    softmax_f16,
    tune_softmax_f16,
    softmax_f16_tuned,
    log_softmax_f16,
    tune_log_softmax_f16,
    log_softmax_f16_tuned,
    softmin_f16,
    tune_softmin_f16,
    softmin_f16_tuned
);

#[cfg(feature = "dtype_f64")]
softmax_fns!(
    f64,
    "f64",
    softmax_f64,
    tune_softmax_f64,
    softmax_f64_tuned,
    log_softmax_f64,
    tune_log_softmax_f64,
    log_softmax_f64_tuned,
    softmin_f64,
    tune_softmin_f64,
    softmin_f64_tuned
);

#[cfg(feature = "dtype_f32")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: SoftmaxConfig,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        config,
        "f32",
        "softmax_lse_f32",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::softmax::softmax_lse_f32(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f32_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f32")]
/// Tunes row normalization using the supplied output storage.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn tune_softmax_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    config: SoftmaxConfig,
    options: TuneOptions,
) -> Result<TuneReport<SoftmaxTuning>> {
    tune_softmax_lse(
        stream,
        out,
        input,
        config,
        options,
        "f32",
        "softmax_lse_f32",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f32_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f16>,
    config: SoftmaxConfig,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        config,
        "f16",
        "softmax_lse_f16",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::softmax::softmax_lse_f16(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f16_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Tunes row normalization using the supplied output storage.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn tune_softmax_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f16>,
    config: SoftmaxConfig,
    options: TuneOptions,
) -> Result<TuneReport<SoftmaxTuning>> {
    tune_softmax_lse(
        stream,
        out,
        input,
        config,
        options,
        "f16",
        "softmax_lse_f16",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f16_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    config: SoftmaxConfig,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        config,
        "f64",
        "softmax_lse_f64",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::softmax::softmax_lse_f64(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f64_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Tunes row normalization using the supplied output storage.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn tune_softmax_lse_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    config: SoftmaxConfig,
    options: TuneOptions,
) -> Result<TuneReport<SoftmaxTuning>> {
    tune_softmax_lse(
        stream,
        out,
        input,
        config,
        options,
        "f64",
        "softmax_lse_f64",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_f64_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f32")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_wide_f32(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f16>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_wide_f16(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Writes the log-sum-exp of each input row.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_lse_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    partial: &mut impl DeviceOutput<f64>,
    row_max: &mut impl DeviceOutput<f64>,
    row_sum: &mut impl DeviceOutput<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_lse_wide_f64(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f32")]
/// Normalizes the exponentials of row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_wide_f32(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Normalizes the exponentials of row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_wide_f16(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Normalizes the exponentials of row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmax_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    partial: &mut impl DeviceOutput<f64>,
    row_max: &mut impl DeviceOutput<f64>,
    row_sum: &mut impl DeviceOutput<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmax_wide_f64(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f32")]
/// Writes each input element minus its row log-sum-exp.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn log_softmax_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::log_softmax_wide_f32(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Writes each input element minus its row log-sum-exp.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn log_softmax_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    partial: &mut impl DeviceOutput<f32>,
    row_max: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::log_softmax_wide_f16(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Writes each input element minus its row log-sum-exp.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn log_softmax_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    partial: &mut impl DeviceOutput<f64>,
    row_max: &mut impl DeviceOutput<f64>,
    row_sum: &mut impl DeviceOutput<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::log_softmax_wide_f64(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f32")]
/// Normalizes the exponentials of negated row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmin_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    partial: &mut impl DeviceOutput<f32>,
    row_min: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmin_wide_f32(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f16")]
/// Normalizes the exponentials of negated row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmin_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    partial: &mut impl DeviceOutput<f32>,
    row_min: &mut impl DeviceOutput<f32>,
    row_sum: &mut impl DeviceOutput<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmin_wide_f16(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

#[cfg(feature = "dtype_f64")]
/// Normalizes the exponentials of negated row elements to sum to one.
///
/// # Safety
///
/// Input values must be valid when read; order their producers before use.
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Writable output and scratch ranges must also be mutually disjoint.
pub unsafe fn softmin_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceOutput<f64>,
    input: &impl DeviceInput<f64>,
    partial: &mut impl DeviceOutput<f64>,
    row_min: &mut impl DeviceOutput<f64>,
    row_sum: &mut impl DeviceOutput<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
            cutile::softmax::softmin_wide_f64(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        },
    )
}

fn softmax_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: SoftmaxConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Softmax {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn softmax_candidates(columns: usize) -> Vec<SoftmaxTuning> {
    [
        SoftmaxTuning::tile(4, 8),
        SoftmaxTuning::tile(2, 16),
        SoftmaxTuning::tile(1, 32),
        SoftmaxTuning::tile(1, 64),
        SoftmaxTuning::tile(1, 128),
        SoftmaxTuning::tile(1, 256),
        SoftmaxTuning::tile(1, 512),
        SoftmaxTuning::tile(1, 1024),
    ]
    .into_iter()
    .filter(|tuning| tuning.tile.columns >= columns)
    .collect()
}

fn resolve_softmax_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<SoftmaxTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<SoftmaxTuning>>
where
    C: FnOnce() -> Vec<SoftmaxTuning>,
    F: FnOnce(TuneOptions, Vec<SoftmaxTuning>) -> Result<TuneReport<SoftmaxTuning>>,
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

fn validate_softmax<T: DeviceRepr>(
    out: &impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SoftmaxConfig,
) -> Result<()> {
    if config.columns == 0 {
        return Err(Error::InvalidLength);
    }
    if config.columns > 1024 {
        return Err(Error::UnsupportedWidth {
            op: "softmax".into(),
            width: config.columns,
        });
    }
    let len = checked_element_count(config.rows, config.columns)?;
    ensure_len(out.len(), len)?;
    ensure_len(input.len(), len)?;
    Ok(())
}

fn launch_softmax_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SoftmaxConfig,
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

fn launch_softmax_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SoftmaxConfig,
    tuning: SoftmaxTuning,
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
        tuning.tile.rows,
        tuning.tile.columns,
        tuning.compile,
    )
}

fn softmax<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SoftmaxConfig,
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
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_softmax(out, input, config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_softmax_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_softmax_with_tuning(stream, out, input, config, tuning, launch_tuned);
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = softmax_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_softmax_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || softmax_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_softmax_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit softmax tuning return before cache resolution"),
        |tuning| launch_softmax_with_tuning(stream, out, input, config, tuning, launch_tuned),
    )
}

fn tune_softmax<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SoftmaxConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<SoftmaxTuning>>
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
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_softmax(out, input, config)?;
    let key = softmax_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        softmax_candidates(config.columns),
        |tuning| launch_softmax_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_softmax_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

fn softmax_wide<Input, Scratch, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Input>,
    input: &impl DeviceInput<Input>,
    partial: &mut impl DeviceOutput<Scratch>,
    row_max: &mut impl DeviceOutput<Scratch>,
    row_sum: &mut impl DeviceOutput<Scratch>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Scratch: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<Input>,
        DevicePointer<Input>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        usize,
        usize,
    ) -> Result<()>,
{
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(input.len(), len)?;
    let chunks = cols.div_ceil(1024);
    let partial_len = checked_element_count(rows, chunks)?;
    ensure_len(partial.len(), partial_len)?;
    ensure_len(row_max.len(), rows)?;
    ensure_len(row_sum.len(), rows)?;
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(partial.as_mut_ptr() as _) },
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(row_max.as_mut_ptr() as _) },
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(row_sum.as_mut_ptr() as _) },
        rows,
        cols,
    )
}

fn validate_softmax_lse<Input: DeviceRepr, Output: DeviceRepr>(
    out: &impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    config: SoftmaxConfig,
) -> Result<()> {
    if config.columns == 0 {
        return Err(Error::InvalidLength);
    }
    if config.columns > 1024 {
        return Err(Error::UnsupportedWidth {
            op: "softmax".into(),
            width: config.columns,
        });
    }
    let len = checked_element_count(config.rows, config.columns)?;
    ensure_len(out.len(), config.rows)?;
    ensure_len(input.len(), len)?;
    Ok(())
}

fn launch_softmax_lse_default<Input, Output, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    config: SoftmaxConfig,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        usize,
        usize,
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
    )
}

fn launch_softmax_lse_with_tuning<Input, Output, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    config: SoftmaxConfig,
    tuning: SoftmaxTuning,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        usize,
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
        tuning.tile.rows,
        tuning.tile.columns,
        tuning.compile,
    )
}

fn softmax_lse<Input, Output, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    config: SoftmaxConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<Output>,
            DevicePointer<Input>,
            usize,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_softmax_lse(out, input, config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_softmax_lse_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_softmax_lse_with_tuning(
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

    let key = softmax_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_softmax_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || softmax_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_softmax_lse_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit softmax lse tuning return before cache resolution"),
        |tuning| launch_softmax_lse_with_tuning(stream, out, input, config, tuning, launch_tuned),
    )
}

fn tune_softmax_lse<Input, Output, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    config: SoftmaxConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<SoftmaxTuning>>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<Output>,
            DevicePointer<Input>,
            usize,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_softmax_lse(out, input, config)?;
    let key = softmax_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        softmax_candidates(config.columns),
        |tuning| launch_softmax_lse_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_softmax_lse_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

fn softmax_lse_wide<Input, Output, Scratch, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<Output>,
    input: &impl DeviceInput<Input>,
    partial: &mut impl DeviceOutput<Scratch>,
    row_max: &mut impl DeviceOutput<Scratch>,
    row_sum: &mut impl DeviceOutput<Scratch>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    Scratch: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        usize,
        usize,
    ) -> Result<()>,
{
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), rows)?;
    ensure_len(input.len(), len)?;
    let chunks = cols.div_ceil(1024);
    let partial_len = checked_element_count(rows, chunks)?;
    ensure_len(partial.len(), partial_len)?;
    ensure_len(row_max.len(), rows)?;
    ensure_len(row_sum.len(), rows)?;
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(input),
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(partial.as_mut_ptr() as _) },
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(row_max.as_mut_ptr() as _) },
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(row_sum.as_mut_ptr() as _) },
        rows,
        cols,
    )
}

#[cfg(test)]
mod tests {
    use singe_cuda::{
        context::Context as CudaContext,
        device::Device,
        memory::DeviceMemory,
        stream::{Stream, StreamCaptureMode, StreamFlags},
    };

    // Borrowed eager launches and replay are drained on errors as well as success.
    use super::*;

    #[cfg(feature = "dtype_f32")]
    fn test_stream() -> Result<Option<Stream>> {
        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(None);
        };
        Ok(Some(
            context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?,
        ))
    }

    #[cfg(feature = "dtype_f32")]
    fn softmax_input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| (index as f32 - 3.0) * 0.25)
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn expected_softmax(input: &[f32], columns: usize) -> Vec<f32> {
        let mut expected = Vec::with_capacity(input.len());
        for row in input.chunks_exact(columns) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps = row
                .iter()
                .map(|value| (*value - max).exp())
                .collect::<Vec<_>>();
            let sum = exps.iter().copied().sum::<f32>();
            expected.extend(exps.into_iter().map(|value| value / sum));
        }
        expected
    }

    #[cfg(feature = "dtype_f32")]
    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 1e-5,
                "mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[cfg(feature = "dtype_f32")]
    fn assert_softmax_output(
        out: &DeviceMemory<f32>,
        expected: &[f32],
        stream: &Stream,
    ) -> Result<()> {
        stream.synchronize()?;
        let actual = out.copy_to_host_vec()?;
        assert_close(&actual, expected);
        Ok(())
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_explicit_softmax_tuning_matches_default() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 4;
        let input_host = softmax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_softmax(&input_host, columns);
        let config = SoftmaxConfig::create(rows, columns);
        let mut default_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut tuned_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.sync_scope(|_| {
            // SAFETY: inputs and exclusive output remain borrowed through the draining scope.
            Ok(unsafe { softmax_f32(&stream, &mut default_out, &input, config) })
        })??;
        stream.sync_scope(|_| {
            // SAFETY: inputs and exclusive output remain borrowed through the draining scope.
            Ok(unsafe {
                softmax_f32(
                    &stream,
                    &mut tuned_out,
                    &input,
                    config.explicit_tuning(SoftmaxTuning::tile(4, 8)),
                )
            })
        })??;
        stream.synchronize()?;

        let default_actual = default_out.copy_to_host_vec()?;
        let tuned_actual = tuned_out.copy_to_host_vec()?;
        assert_close(&default_actual, &expected);
        assert_close(&tuned_actual, &default_actual);
        Ok(())
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_tune_and_retune_softmax_relaunch_winning_candidate() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 3;
        let columns = 5;
        let input_host = softmax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_softmax(&input_host, columns);
        let options = TuneOptions::create(0, 1, 2);
        let config = SoftmaxConfig::create(rows, columns);
        let mut tune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut retune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.sync_scope(|_| {
            // SAFETY: inputs and exclusive output remain borrowed through the draining scope.
            Ok(unsafe { softmax_f32(&stream, &mut tune_out, &input, config.tune(options)) })
        })??;
        assert_softmax_output(&tune_out, &expected, &stream)?;

        stream.sync_scope(|_| {
            // SAFETY: inputs and exclusive output remain borrowed through the draining scope.
            Ok(unsafe { softmax_f32(&stream, &mut retune_out, &input, config.retune(options)) })
        })??;
        assert_softmax_output(&retune_out, &expected, &stream)?;
        Ok(())
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_cached_softmax_tuning_can_be_captured() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 6;
        let input_host = softmax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_softmax(&input_host, columns);
        let options = TuneOptions::create(0, 1, 1);
        let config = SoftmaxConfig::create(rows, columns);
        let mut warmup_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut graph_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.sync_scope(|_| {
            // SAFETY: inputs and exclusive output remain borrowed through the draining scope.
            Ok(unsafe { softmax_f32(&stream, &mut warmup_out, &input, config.tune(options)) })
        })??;
        assert_softmax_output(&warmup_out, &expected, &stream)?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: the inputs and exclusive output survive capture and draining replay.
        unsafe { softmax_f32(&stream, &mut graph_out, &input, config.cached_tuning()) }?;
        let graph = stream.end_capture()?.instantiate()?;
        stream.sync_scope(|_| unsafe { graph.launch(&stream) })?;

        assert_softmax_output(&graph_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_softmax_tune_and_retune_reject_capture() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 7;
        let input_host = softmax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let options = TuneOptions::create(0, 1, 1);
        let config = SoftmaxConfig::create(rows, columns);
        let mut out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: tuning rejects capture before enqueue.
        let error =
            unsafe { softmax_f32(&stream, &mut out, &input, config.tune(options)) }.unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "softmax_f32"
        ));

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: tuning rejects capture before enqueue.
        let error =
            unsafe { softmax_f32(&stream, &mut out, &input, config.retune(options)) }.unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "softmax_f32"
        ));
        Ok(())
    }
}
