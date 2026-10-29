//! RMS norm, layer norm, group norm, sparsemax, and related reductions.

use std::{
    fmt::{self, Debug, Formatter},
    sync::Arc,
};

#[cfg(feature = "cutile")]
use ::cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream as CutileStream};
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceInputView, DeviceOutput, DeviceRepr},
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparsemaxConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<SparsemaxTuning>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparsemaxTuning {
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Normalization parameters in the accumulation dtype, independent of buffer storage.
pub struct NormalizationConfig<T> {
    pub rows: usize,
    pub columns: usize,
    pub eps: T,
    pub tuning: TuningChoice<NormalizationTuning>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// RMS normalization parameters in the accumulation dtype.
pub struct RmsNormWeightOffsetConfig<T> {
    pub rows: usize,
    pub columns: usize,
    pub eps: T,
    pub weight_offset: T,
    pub tuning: TuningChoice<NormalizationTuning>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NormalizationTuning {
    pub rows_per_block: usize,
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

impl SparsemaxConfig {
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<SparsemaxTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: SparsemaxTuning) -> Self {
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

impl SparsemaxTuning {
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

impl<T: Copy> NormalizationConfig<T> {
    pub const fn create(rows: usize, columns: usize, eps: T) -> Self {
        Self {
            rows,
            columns,
            eps,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<NormalizationTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: NormalizationTuning) -> Self {
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

impl<T: Copy> RmsNormWeightOffsetConfig<T> {
    pub const fn create(rows: usize, columns: usize, eps: T, weight_offset: T) -> Self {
        Self {
            rows,
            columns,
            eps,
            weight_offset,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<NormalizationTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: NormalizationTuning) -> Self {
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

impl NormalizationTuning {
    #[must_use]
    pub const fn tile_width(tile_width: usize) -> Self {
        Self {
            rows_per_block: 1,
            tile_width,
            compile: KernelCompileConfig {
                occupancy: None,
                num_cta_in_cga: None,
                max_divisibility: None,
            },
        }
    }

    #[must_use]
    pub const fn rows(mut self, rows_per_block: usize) -> Self {
        self.rows_per_block = rows_per_block;
        self
    }

    #[must_use]
    pub const fn compile(mut self, compile: KernelCompileConfig) -> Self {
        self.compile = compile;
        self
    }
}

macro_rules! normalization_fns {
    (
        $ty:ty,
        $scalar:ty,
        $zero:expr,
        $dtype:literal,
        $layer_norm:ident,
        $tune_layer_norm:ident,
        $layer_norm_tuned:ident,
        $layer_norm_weight:ident,
        $tune_layer_norm_weight:ident,
        $layer_norm_bias:ident,
        $tune_layer_norm_bias:ident,
        $layer_norm_affine:ident,
        $tune_layer_norm_affine:ident,
        $rms_norm:ident,
        $tune_rms_norm:ident,
        $rms_norm_tuned:ident,
        $rms_norm_affine:ident,
        $tune_rms_norm_affine:ident,
        $rms_norm_weight_offset:ident,
        $tune_rms_norm_weight_offset:ident
    ) => {
        /// Normalizes each row by its mean and variance.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $layer_norm(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($layer_norm),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$layer_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_layer_norm(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($layer_norm),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row and multiplies by the column weights.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $layer_norm_weight(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                Some(weight),
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($layer_norm_weight),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$layer_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_layer_norm_weight(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                Some(weight),
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($layer_norm_weight),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row and adds the column bias.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $layer_norm_bias(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($layer_norm_bias),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$layer_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_layer_norm_bias(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($layer_norm_bias),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row, multiplies by column weights, and adds column bias.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $layer_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($layer_norm_affine),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$layer_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_layer_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($layer_norm_affine),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$layer_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row by its root mean square.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $rms_norm(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($rms_norm),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$rms_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_rms_norm(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                None::<&DeviceInputView<'_, $ty>>,
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($rms_norm),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row by its root mean square and applies column weights and bias.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $rms_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                config.tuning,
                $dtype,
                stringify!($rms_norm_affine),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$rms_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_rms_norm_affine(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: NormalizationConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                Some(weight),
                Some(bias),
                config.rows,
                config.columns,
                config.eps,
                $zero,
                options,
                $dtype,
                stringify!($rms_norm_affine),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Normalizes each row by its root mean square and scales by `weight + weight_offset`.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $rms_norm_weight_offset(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: RmsNormWeightOffsetConfig<$scalar>,
        ) -> Result<()> {
            normalization_autotuned(
                stream,
                out,
                input,
                Some(weight),
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                config.weight_offset,
                config.tuning,
                $dtype,
                stringify!($rms_norm_weight_offset),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10| unsafe {
                    cutile::normalization::$rms_norm(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                    )
                },
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_rms_norm_weight_offset(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            weight: &impl DeviceInput<$ty>,
            config: RmsNormWeightOffsetConfig<$scalar>,
            options: TuneOptions,
        ) -> Result<TuneReport<NormalizationTuning>> {
            tune_normalization(
                stream,
                out,
                input,
                Some(weight),
                None::<&DeviceInputView<'_, $ty>>,
                config.rows,
                config.columns,
                config.eps,
                config.weight_offset,
                options,
                $dtype,
                stringify!($rms_norm_weight_offset),
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg12| unsafe {
                    cutile::normalization::$rms_norm_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
normalization_fns!(
    f32,
    f32,
    0.0f32,
    "f32",
    layer_norm_f32,
    tune_layer_norm_f32,
    layer_norm_f32_tuned,
    layer_norm_weight_f32,
    tune_layer_norm_weight_f32,
    layer_norm_bias_f32,
    tune_layer_norm_bias_f32,
    layer_norm_affine_f32,
    tune_layer_norm_affine_f32,
    rms_norm_f32,
    tune_rms_norm_f32,
    rms_norm_f32_tuned,
    rms_norm_affine_f32,
    tune_rms_norm_affine_f32,
    rms_norm_weight_offset_f32,
    tune_rms_norm_weight_offset_f32
);

#[cfg(feature = "dtype_f16")]
normalization_fns!(
    f16,
    f32,
    0.0f32,
    "f16",
    layer_norm_f16,
    tune_layer_norm_f16,
    layer_norm_f16_tuned,
    layer_norm_weight_f16,
    tune_layer_norm_weight_f16,
    layer_norm_bias_f16,
    tune_layer_norm_bias_f16,
    layer_norm_affine_f16,
    tune_layer_norm_affine_f16,
    rms_norm_f16,
    tune_rms_norm_f16,
    rms_norm_f16_tuned,
    rms_norm_affine_f16,
    tune_rms_norm_affine_f16,
    rms_norm_weight_offset_f16,
    tune_rms_norm_weight_offset_f16
);

#[cfg(feature = "dtype_bf16")]
normalization_fns!(
    bf16,
    f32,
    0.0f32,
    "bf16",
    layer_norm_bf16,
    tune_layer_norm_bf16,
    layer_norm_bf16_tuned,
    layer_norm_weight_bf16,
    tune_layer_norm_weight_bf16,
    layer_norm_bias_bf16,
    tune_layer_norm_bias_bf16,
    layer_norm_affine_bf16,
    tune_layer_norm_affine_bf16,
    rms_norm_bf16,
    tune_rms_norm_bf16,
    rms_norm_bf16_tuned,
    rms_norm_affine_bf16,
    tune_rms_norm_affine_bf16,
    rms_norm_weight_offset_bf16,
    tune_rms_norm_weight_offset_bf16
);

#[cfg(feature = "dtype_f64")]
normalization_fns!(
    f64,
    f64,
    0.0f64,
    "f64",
    layer_norm_f64,
    tune_layer_norm_f64,
    layer_norm_f64_tuned,
    layer_norm_weight_f64,
    tune_layer_norm_weight_f64,
    layer_norm_bias_f64,
    tune_layer_norm_bias_f64,
    layer_norm_affine_f64,
    tune_layer_norm_affine_f64,
    rms_norm_f64,
    tune_rms_norm_f64,
    rms_norm_f64_tuned,
    rms_norm_affine_f64,
    tune_rms_norm_affine_f64,
    rms_norm_weight_offset_f64,
    tune_rms_norm_weight_offset_f64
);

macro_rules! sparsemax_fn {
    ($name:ident, $tune_name:ident, $tuned_backend:ident, $ty:ty, $dtype:literal) => {
        /// Projects each row onto the probability simplex using sparsemax.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SparsemaxConfig,
        ) -> Result<()> {
            sparsemax(
                stream,
                out,
                input,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::normalization::$name(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::normalization::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }

        /// Tunes row normalization using the supplied writable destination.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: SparsemaxConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<SparsemaxTuning>> {
            tune_sparsemax(
                stream,
                out,
                input,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::normalization::$tuned_backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
sparsemax_fn!(
    sparsemax_f32,
    tune_sparsemax_f32,
    sparsemax_f32_tuned,
    f32,
    "f32"
);

#[cfg(feature = "dtype_f16")]
sparsemax_fn!(
    sparsemax_f16,
    tune_sparsemax_f16,
    sparsemax_f16_tuned,
    f16,
    "f16"
);

#[cfg(feature = "dtype_bf16")]
sparsemax_fn!(
    sparsemax_bf16,
    tune_sparsemax_bf16,
    sparsemax_bf16_tuned,
    bf16,
    "bf16"
);

pub struct BatchNormInferenceConfig<'a, T: DeviceRepr, A> {
    pub weight: Option<&'a dyn DeviceInput<T>>,
    pub bias: Option<&'a dyn DeviceInput<T>>,
    pub running_mean: Option<&'a dyn DeviceInput<T>>,
    pub running_var: Option<&'a dyn DeviceInput<T>>,
    pub feature_count: usize,
    pub spatial_len: usize,
    pub eps: A,
}

impl<T: DeviceRepr, A> Debug for BatchNormInferenceConfig<'_, T, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchNormInferenceConfig")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupNormConfig {
    pub batch: usize,
    pub channels: usize,
    pub groups: usize,
    pub spatial_len: usize,
    pub eps: f32,
}

#[cfg(feature = "dtype_f32")]
/// Normalizes each contiguous feature group by its mean and variance.
///
/// # Safety
///
/// Inputs must be valid when read in the stream's context and disjoint from output.
/// Order preceding producers before every read, including graph replay.
/// Retain all ranges and exclusive output access until completion on every path
/// and throughout replay. Order prior accesses before launch.
pub unsafe fn group_norm_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    weight: &impl DeviceInput<f32>,
    bias: &impl DeviceInput<f32>,
    config: GroupNormConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { group_norm(stream, out, input, weight, bias, config) }
}

#[cfg(feature = "dtype_f16")]
/// Normalizes each contiguous feature group by its mean and variance.
///
/// # Safety
///
/// Inputs must be valid when read in the stream's context and disjoint from output.
/// Order preceding producers before every read, including graph replay.
/// Retain all ranges and exclusive output access until completion on every path
/// and throughout replay. Order prior accesses before launch.
pub unsafe fn group_norm_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    weight: &impl DeviceInput<f16>,
    bias: &impl DeviceInput<f16>,
    config: GroupNormConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { group_norm_f16_impl(stream, out, input, weight, bias, config) }
}

#[cfg(feature = "dtype_bf16")]
/// Normalizes each contiguous feature group by its mean and variance.
///
/// # Safety
///
/// Inputs must be valid when read in the stream's context and disjoint from output.
/// Order preceding producers before every read, including graph replay.
/// Retain all ranges and exclusive output access until completion on every path
/// and throughout replay. Order prior accesses before launch.
pub unsafe fn group_norm_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    input: &impl DeviceInput<bf16>,
    weight: &impl DeviceInput<bf16>,
    bias: &impl DeviceInput<bf16>,
    config: GroupNormConfig,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { group_norm_bf16_impl(stream, out, input, weight, bias, config) }
}

macro_rules! batch_norm_inference_fn {
    ($name:ident, $ty:ty, $scalar:ty) => {
        /// Applies inference batch normalization with optional feature statistics, weights, and bias.
        ///
        /// # Safety
        ///
        /// Inputs must be valid when read in the stream's context and disjoint from output.
        /// Order preceding producers before every read, including graph replay.
        /// Retain all ranges and exclusive output access until completion on every path
        /// and throughout replay. Order prior accesses before launch.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: BatchNormInferenceConfig<'_, $ty, $scalar>,
        ) -> Result<()> {
            batch_norm_inference(
                stream,
                out,
                input,
                config,
                |arg0,
                 arg1,
                 arg2,
                 arg3,
                 arg4,
                 arg5,
                 arg6,
                 arg7,
                 arg8,
                 arg9,
                 arg10,
                 arg11,
                 arg12,
                 arg13,
                 // SAFETY: the caller upholds this function's safety contract.
                 arg14| unsafe {
                    cutile::normalization::$name(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
                        arg12, arg13, arg14,
                    )
                },
            )
        }
    };
}

#[cfg(feature = "dtype_f32")]
batch_norm_inference_fn!(batch_norm_inference_f32, f32, f32);

#[cfg(feature = "dtype_f16")]
batch_norm_inference_fn!(batch_norm_inference_f16, f16, f32);

#[cfg(feature = "dtype_f64")]
batch_norm_inference_fn!(batch_norm_inference_f64, f64, f64);

#[cfg(feature = "dtype_f32")]
/// # Safety
/// The caller must uphold the safety contract of the public wrapper that calls this helper.
unsafe fn group_norm(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    weight: &impl DeviceInput<f32>,
    bias: &impl DeviceInput<f32>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = stream.to_cuda_core()?;
    let input_ptr = input_pointer(input);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::normalization::group_norm_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_ptr,
            input_pointer(weight),
            input_pointer(bias),
            config.batch,
            config.channels,
            config.groups,
            config.spatial_len,
            config.eps,
        )
    }
}

#[cfg(feature = "dtype_f16")]
/// # Safety
/// The caller must uphold the safety contract of the public wrapper that calls this helper.
unsafe fn group_norm_f16_impl(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    weight: &impl DeviceInput<f16>,
    bias: &impl DeviceInput<f16>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = stream.to_cuda_core()?;
    let input_ptr = input_pointer(input);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::normalization::group_norm_f16(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_ptr,
            input_pointer(weight),
            input_pointer(bias),
            config.batch,
            config.channels,
            config.groups,
            config.spatial_len,
            config.eps,
        )
    }
}

#[cfg(feature = "dtype_bf16")]
/// # Safety
/// The caller must uphold the safety contract of the public wrapper that calls this helper.
unsafe fn group_norm_bf16_impl(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    input: &impl DeviceInput<bf16>,
    weight: &impl DeviceInput<bf16>,
    bias: &impl DeviceInput<bf16>,
    config: GroupNormConfig,
) -> Result<()> {
    validate_group_norm(out.len(), input.len(), weight.len(), bias.len(), config)?;

    let stream = stream.to_cuda_core()?;
    let input_ptr = input_pointer(input);
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::normalization::group_norm_bf16(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_ptr,
            input_pointer(weight),
            input_pointer(bias),
            config.batch,
            config.channels,
            config.groups,
            config.spatial_len,
            config.eps,
        )
    }
}

#[cfg(feature = "cutile")]
fn batch_norm_inference<T, A, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: BatchNormInferenceConfig<'_, T, A>,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        bool,
        bool,
        bool,
        bool,
        A,
        usize,
    ) -> Result<()>,
{
    if config.feature_count == 0 || config.spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    let len = out.len();
    ensure_len(input.len(), len)?;
    if let Some(weight) = config.weight {
        ensure_len(weight.len(), config.feature_count)?;
    }
    if let Some(bias) = config.bias {
        ensure_len(bias.len(), config.feature_count)?;
    }
    if let Some(running_mean) = config.running_mean {
        ensure_len(running_mean.len(), config.feature_count)?;
    }
    if let Some(running_var) = config.running_var {
        ensure_len(running_var.len(), config.feature_count)?;
    }
    let feature_period = checked_element_count(config.feature_count, config.spatial_len)?;
    if len % feature_period != 0 {
        return Err(Error::LengthMismatch);
    }

    let stream = stream.to_cuda_core()?;
    let input_ptr = input_pointer(input);
    let weight_ptr = config
        .weight
        .map_or(input_ptr, |weight| input_pointer(weight));
    let bias_ptr = config.bias.map_or(input_ptr, |bias| input_pointer(bias));
    let running_mean_ptr = config
        .running_mean
        .map_or(input_ptr, |running_mean| input_pointer(running_mean));
    let running_var_ptr = config
        .running_var
        .map_or(input_ptr, |running_var| input_pointer(running_var));
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_ptr,
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        config.feature_count,
        config.spatial_len,
        config.weight.is_some(),
        config.bias.is_some(),
        config.running_mean.is_some(),
        config.running_var.is_some(),
        config.eps,
        len,
    )
}

fn sparsemax_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: SparsemaxConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Sparsemax {
            dtype: dtype.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn sparsemax_candidates(columns: usize) -> Vec<SparsemaxTuning> {
    [
        SparsemaxTuning::tile_width(1),
        SparsemaxTuning::tile_width(8),
        SparsemaxTuning::tile_width(32),
        SparsemaxTuning::tile_width(64),
        SparsemaxTuning::tile_width(128),
        SparsemaxTuning::tile_width(256),
        SparsemaxTuning::tile_width(512),
        SparsemaxTuning::tile_width(1024),
        SparsemaxTuning::tile_width(2048),
    ]
    .into_iter()
    .filter(|tuning| tuning.tile_width >= columns)
    .collect()
}

fn resolve_sparsemax_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<SparsemaxTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<SparsemaxTuning>>
where
    C: FnOnce() -> Vec<SparsemaxTuning>,
    F: FnOnce(TuneOptions, Vec<SparsemaxTuning>) -> Result<TuneReport<SparsemaxTuning>>,
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

fn launch_sparsemax_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SparsemaxConfig,
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

fn launch_sparsemax_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SparsemaxConfig,
    tuning: SparsemaxTuning,
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

fn sparsemax<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SparsemaxConfig,
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
    validate_sparsemax(out.len(), input.len(), config.rows, config.columns)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_sparsemax_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_sparsemax_with_tuning(stream, out, input, config, tuning, launch_tuned);
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = sparsemax_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_sparsemax_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || sparsemax_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_sparsemax_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit sparsemax tuning return before cache resolution"),
        |tuning| launch_sparsemax_with_tuning(stream, out, input, config, tuning, launch_tuned),
    )
}

fn tune_sparsemax<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    config: SparsemaxConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<SparsemaxTuning>>
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
    validate_sparsemax(out.len(), input.len(), config.rows, config.columns)?;
    let key = sparsemax_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        sparsemax_candidates(config.columns),
        |tuning| launch_sparsemax_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_sparsemax_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

fn validate_sparsemax(out_len: usize, input_len: usize, rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    Ok(len)
}

fn validate_group_norm(
    out_len: usize,
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    config: GroupNormConfig,
) -> Result<()> {
    if config.batch == 0 || config.channels == 0 || config.groups == 0 || config.spatial_len == 0 {
        return Err(Error::InvalidLength);
    }
    if !config.channels.is_multiple_of(config.groups) {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(
        checked_element_count(config.batch, config.channels)?,
        config.spatial_len,
    )?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(weight_len, config.channels)?;
    ensure_len(bias_len, config.channels)?;
    Ok(())
}

fn normalization_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    rows: usize,
    columns: usize,
    has_weight: bool,
    has_bias: bool,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Normalization {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows,
            columns,
            has_weight,
            has_bias,
        },
    ))
}

fn normalization_candidates(columns: usize) -> Vec<NormalizationTuning> {
    if columns > 8192 {
        return vec![NormalizationTuning::tile_width(128)];
    }
    let width = columns.max(1).next_power_of_two();
    [1, 2, 4, 8]
        .into_iter()
        .filter(|rows| rows * width <= 8192)
        .map(|rows| NormalizationTuning::tile_width(width).rows(rows))
        .collect()
}

fn normalization_candidates_for_dtype(columns: usize, dtype: &str) -> Vec<NormalizationTuning> {
    if dtype != "f64" {
        return normalization_candidates(columns);
    }
    let width = match columns {
        0..=1 => 1,
        2..=8 => 8,
        9..=32 => 32,
        33..=2048 => columns.next_power_of_two(),
        _ => 128,
    };
    vec![NormalizationTuning::tile_width(width)]
}

fn resolve_normalization_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<NormalizationTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<NormalizationTuning>>
where
    C: FnOnce() -> Vec<NormalizationTuning>,
    F: FnOnce(TuneOptions, Vec<NormalizationTuning>) -> Result<TuneReport<NormalizationTuning>>,
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

fn launch_normalization_default<T, A: Copy, W, B, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: Option<&W>,
    bias: Option<&B>,
    rows: usize,
    cols: usize,
    eps: A,
    weight_offset: A,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    W: DeviceInput<T> + ?Sized,
    B: DeviceInput<T> + ?Sized,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        bool,
        bool,
        usize,
        usize,
        A,
        A,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    let input = input_pointer(input);
    let weight_pointer = weight.map_or(input, input_pointer);
    let bias_pointer = bias.map_or(input, input_pointer);
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input,
        weight_pointer,
        bias_pointer,
        weight.is_some(),
        bias.is_some(),
        rows,
        cols,
        eps,
        weight_offset,
    )
}

fn launch_normalization_with_tuning<T, A: Copy, W, B, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: Option<&W>,
    bias: Option<&B>,
    rows: usize,
    cols: usize,
    eps: A,
    weight_offset: A,
    tuning: NormalizationTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    W: DeviceInput<T>,
    B: DeviceInput<T>,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        bool,
        bool,
        usize,
        usize,
        A,
        A,
        Tile2d,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    let input = input_pointer(input);
    let weight_pointer = weight.map_or(input, input_pointer);
    let bias_pointer = bias.map_or(input, input_pointer);
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input,
        weight_pointer,
        bias_pointer,
        weight.is_some(),
        bias.is_some(),
        rows,
        cols,
        eps,
        weight_offset,
        Tile2d::create(tuning.rows_per_block, tuning.tile_width),
        tuning.compile,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "autotuning forwards the full kernel configuration"
)]
fn normalization_autotuned<T, A: Copy, W, B, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: Option<&W>,
    bias: Option<&B>,
    rows: usize,
    cols: usize,
    eps: A,
    weight_offset: A,
    choice: TuningChoice<NormalizationTuning>,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    W: DeviceInput<T>,
    B: DeviceInput<T>,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        bool,
        bool,
        usize,
        usize,
        A,
        A,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            bool,
            bool,
            usize,
            usize,
            A,
            A,
            Tile2d,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_normalization_lengths(
        out.len(),
        input.len(),
        weight.map(DeviceInput::len),
        bias.map(DeviceInput::len),
        rows,
        cols,
    )?;
    match choice {
        TuningChoice::Default => {
            return launch_normalization_default(
                stream,
                out,
                input,
                weight,
                bias,
                rows,
                cols,
                eps,
                weight_offset,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_normalization_with_tuning(
                stream,
                out,
                input,
                weight,
                bias,
                rows,
                cols,
                eps,
                weight_offset,
                tuning,
                launch_tuned,
            );
        }
        TuningChoice::Cached => {}
        TuningChoice::Tune(_) | TuningChoice::Retune(_) => {
            autotune::ensure_not_capturing(stream, op)?;
        }
    }

    let key = normalization_tune_key(
        stream,
        op,
        dtype,
        rows,
        cols,
        weight.is_some(),
        bias.is_some(),
    )?;
    let tuning = resolve_normalization_tuning(
        stream,
        op,
        key.clone(),
        choice,
        || normalization_candidates_for_dtype(cols, dtype),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_normalization_with_tuning(
                    stream,
                    out,
                    input,
                    weight,
                    bias,
                    rows,
                    cols,
                    eps,
                    weight_offset,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit normalization tuning return before cache resolution"),
        |tuning| {
            launch_normalization_with_tuning(
                stream,
                out,
                input,
                weight,
                bias,
                rows,
                cols,
                eps,
                weight_offset,
                tuning,
                launch_tuned,
            )
        },
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "autotuning forwards the full kernel configuration"
)]
fn tune_normalization<T, A: Copy, W, B, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: Option<&W>,
    bias: Option<&B>,
    rows: usize,
    cols: usize,
    eps: A,
    weight_offset: A,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<NormalizationTuning>>
where
    T: DeviceRepr,
    W: DeviceInput<T>,
    B: DeviceInput<T>,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            bool,
            bool,
            usize,
            usize,
            A,
            A,
            Tile2d,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_normalization_lengths(
        out.len(),
        input.len(),
        weight.map(DeviceInput::len),
        bias.map(DeviceInput::len),
        rows,
        cols,
    )?;
    let key = normalization_tune_key(
        stream,
        op,
        dtype,
        rows,
        cols,
        weight.is_some(),
        bias.is_some(),
    )?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        normalization_candidates_for_dtype(cols, dtype),
        |tuning| {
            launch_normalization_with_tuning(
                stream,
                out,
                input,
                weight,
                bias,
                rows,
                cols,
                eps,
                weight_offset,
                *tuning,
                launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_normalization_with_tuning(
        stream,
        out,
        input,
        weight,
        bias,
        rows,
        cols,
        eps,
        weight_offset,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn validate_normalization_lengths(
    out_len: usize,
    input_len: usize,
    weight_len: Option<usize>,
    bias_len: Option<usize>,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    if let Some(weight_len) = weight_len {
        ensure_len(weight_len, cols)?;
    }
    if let Some(bias_len) = bias_len {
        ensure_len(bias_len, cols)?;
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "dtype_f32")]
    use singe_cuda::{
        context::Context as CudaContext,
        device::Device,
        memory::DeviceMemory,
        stream::{Stream, StreamCaptureMode, StreamFlags},
    };

    // Each eager normalization launch is drained before its borrowed resources can drop.
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
    fn sparsemax_input(rows: usize, columns: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.25)
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn expected_sparsemax(input: &[f32], columns: usize) -> Vec<f32> {
        input
            .chunks_exact(columns)
            .flat_map(|row| {
                let mut sorted = row.to_vec();
                sorted.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
                let mut sum = 0.0f32;
                let mut support = 0usize;
                for (index, value) in sorted.iter().copied().enumerate() {
                    sum += value;
                    let k = index + 1;
                    if 1.0 + k as f32 * value > sum {
                        support = k;
                    }
                }
                let tau = (sorted[..support].iter().sum::<f32>() - 1.0) / support as f32;
                row.iter()
                    .map(|value| (value - tau).max(0.0))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn affine_weight(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| 1.0 + index as f32 * 0.125)
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn affine_bias(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|index| -0.25 + index as f32 * 0.05)
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn expected_layer_norm_affine(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        columns: usize,
        eps: f32,
    ) -> Vec<f32> {
        input
            .chunks_exact(columns)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / columns as f32;
                let variance = row
                    .iter()
                    .map(|value| {
                        let diff = value - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / columns as f32;
                let inv_std = (variance + eps).sqrt().recip();
                row.iter()
                    .zip(weight)
                    .zip(bias)
                    .map(|((value, weight), bias)| (value - mean) * inv_std * weight + bias)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn expected_rms_norm_weight_offset(
        input: &[f32],
        weight: &[f32],
        columns: usize,
        eps: f32,
        weight_offset: f32,
    ) -> Vec<f32> {
        input
            .chunks_exact(columns)
            .flat_map(|row| {
                let mean_square =
                    row.iter().map(|value| value * value).sum::<f32>() / columns as f32;
                let inv_rms = (mean_square + eps).sqrt().recip();
                row.iter()
                    .zip(weight)
                    .map(|(value, weight)| value * inv_rms * (weight + weight_offset))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(feature = "dtype_f32")]
    fn assert_sparsemax_output(
        out: &DeviceMemory<f32>,
        expected: &[f32],
        stream: &Stream,
    ) -> Result<()> {
        stream.synchronize()?;
        let actual = out.copy_to_host_vec()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "actual {actual} expected {expected}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "dtype_f32")]
    fn assert_close_output(
        out: &DeviceMemory<f32>,
        expected: &[f32],
        stream: &Stream,
    ) -> Result<()> {
        stream.synchronize()?;
        let actual = out.copy_to_host_vec()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "actual {actual} expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_normalization_config_builders_preserve_tuning_choice() {
        let options = TuneOptions::create(0, 1, 2);
        let tuning = NormalizationTuning::tile_width(64)
            .compile(KernelCompileConfig::default().occupancy(4));

        assert_eq!(
            NormalizationConfig::create(3, 33, 1e-5f32)
                .explicit_tuning(tuning)
                .tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            NormalizationConfig::create(3, 33, 1e-5f32)
                .cached_tuning()
                .tuning,
            TuningChoice::Cached
        );
        assert_eq!(
            NormalizationConfig::create(3, 33, 1e-5f32)
                .tune(options)
                .tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            NormalizationConfig::create(3, 33, 1e-5f32)
                .retune(options)
                .tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_rms_norm_weight_offset_config_builders_preserve_tuning_choice() {
        let options = TuneOptions::create(0, 1, 2);
        let tuning = NormalizationTuning::tile_width(128);

        assert_eq!(
            RmsNormWeightOffsetConfig::create(3, 64, 1e-5f32, 1.0)
                .explicit_tuning(tuning)
                .tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            RmsNormWeightOffsetConfig::create(3, 64, 1e-5f32, 1.0)
                .cached_tuning()
                .tuning,
            TuningChoice::Cached
        );
        assert_eq!(
            RmsNormWeightOffsetConfig::create(3, 64, 1e-5f32, 1.0)
                .tune(options)
                .tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            RmsNormWeightOffsetConfig::create(3, 64, 1e-5f32, 1.0)
                .retune(options)
                .tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_normalization_candidates_only_include_launchable_widths() {
        let candidates = normalization_candidates(129)
            .into_iter()
            .filter(|tuning| tuning.rows_per_block == 1)
            .map(|tuning| tuning.tile_width)
            .collect::<Vec<_>>();

        assert_eq!(candidates, vec![256]);
    }

    #[test]
    fn test_sparsemax_config_builders_preserve_tuning_choice() {
        let options = TuneOptions::create(0, 1, 2);
        let tuning =
            SparsemaxTuning::tile_width(64).compile(KernelCompileConfig::default().occupancy(4));

        assert_eq!(
            SparsemaxConfig::create(3, 33)
                .explicit_tuning(tuning)
                .tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(
            SparsemaxConfig::create(3, 33).cached_tuning().tuning,
            TuningChoice::Cached
        );
        assert_eq!(
            SparsemaxConfig::create(3, 33).tune(options).tuning,
            TuningChoice::Tune(options)
        );
        assert_eq!(
            SparsemaxConfig::create(3, 33).retune(options).tuning,
            TuningChoice::Retune(options)
        );
    }

    #[test]
    fn test_sparsemax_candidates_only_include_non_wide_launchable_widths() {
        let candidates = sparsemax_candidates(33)
            .into_iter()
            .map(|tuning| tuning.tile_width)
            .collect::<Vec<_>>();

        assert_eq!(candidates, vec![64, 128, 256, 512, 1024, 2048]);
        assert!(sparsemax_candidates(2049).is_empty());
    }

    #[test]
    fn test_sparsemax_validation_accepts_exact_lengths() -> Result<()> {
        assert_eq!(validate_sparsemax(15, 15, 3, 5)?, 15);
        Ok(())
    }

    #[test]
    fn test_sparsemax_validation_rejects_zero_rows() {
        assert!(matches!(
            validate_sparsemax(0, 0, 0, 5),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn test_sparsemax_validation_rejects_short_input() {
        assert!(matches!(
            validate_sparsemax(15, 14, 3, 5),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_explicit_sparsemax_tuning_matches_default() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 5;
        let input_host = sparsemax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_sparsemax(&input_host, columns);
        let config = SparsemaxConfig::create(rows, columns);
        let mut default_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut tuned_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe { sparsemax_f32(&stream, &mut default_out, &input, config) })
        })??;
        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                sparsemax_f32(
                    &stream,
                    &mut tuned_out,
                    &input,
                    config.explicit_tuning(SparsemaxTuning::tile_width(8)),
                )
            })
        })??;
        assert_sparsemax_output(&default_out, &expected, &stream)?;
        assert_sparsemax_output(&tuned_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_explicit_layer_norm_tuning_matches_default() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 5;
        let eps = 1e-5f32;
        let input_host = sparsemax_input(rows, columns);
        let weight_host = affine_weight(columns);
        let bias_host = affine_bias(columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let weight = DeviceMemory::from_slice(&stream.context(), &weight_host)?;
        let bias = DeviceMemory::from_slice(&stream.context(), &bias_host)?;
        let expected =
            expected_layer_norm_affine(&input_host, &weight_host, &bias_host, columns, eps);
        let config = NormalizationConfig::create(rows, columns, eps);
        let mut default_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut tuned_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                layer_norm_affine_f32(&stream, &mut default_out, &input, &weight, &bias, config)
            })
        })??;
        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                layer_norm_affine_f32(
                    &stream,
                    &mut tuned_out,
                    &input,
                    &weight,
                    &bias,
                    config.explicit_tuning(NormalizationTuning::tile_width(8)),
                )
            })
        })??;
        assert_close_output(&default_out, &expected, &stream)?;
        assert_close_output(&tuned_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_tune_and_retune_layer_norm_relaunch_winning_candidate() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 6;
        let eps = 1e-5f32;
        let input_host = sparsemax_input(rows, columns);
        let weight_host = affine_weight(columns);
        let bias_host = affine_bias(columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let weight = DeviceMemory::from_slice(&stream.context(), &weight_host)?;
        let bias = DeviceMemory::from_slice(&stream.context(), &bias_host)?;
        let expected =
            expected_layer_norm_affine(&input_host, &weight_host, &bias_host, columns, eps);
        let options = TuneOptions::create(0, 1, 2);
        let config = NormalizationConfig::create(rows, columns, eps);
        let mut tune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut retune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                layer_norm_affine_f32(
                    &stream,
                    &mut tune_out,
                    &input,
                    &weight,
                    &bias,
                    config.tune(options),
                )
            })
        })??;
        assert_close_output(&tune_out, &expected, &stream)?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                layer_norm_affine_f32(
                    &stream,
                    &mut retune_out,
                    &input,
                    &weight,
                    &bias,
                    config.retune(options),
                )
            })
        })??;
        assert_close_output(&retune_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_cached_layer_norm_tuning_can_be_captured() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 7;
        let eps = 1e-5f32;
        let input_host = sparsemax_input(rows, columns);
        let weight_host = affine_weight(columns);
        let bias_host = affine_bias(columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let weight = DeviceMemory::from_slice(&stream.context(), &weight_host)?;
        let bias = DeviceMemory::from_slice(&stream.context(), &bias_host)?;
        let expected =
            expected_layer_norm_affine(&input_host, &weight_host, &bias_host, columns, eps);
        let options = TuneOptions::create(0, 1, 1);
        let config = NormalizationConfig::create(rows, columns, eps);
        let mut warmup_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut graph_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe {
                layer_norm_affine_f32(
                    &stream,
                    &mut warmup_out,
                    &input,
                    &weight,
                    &bias,
                    config.tune(options),
                )
            })
        })??;
        assert_close_output(&warmup_out, &expected, &stream)?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: capture records this access; the scoped replay retains every allocation until completion.
        unsafe {
            layer_norm_affine_f32(
                &stream,
                &mut graph_out,
                &input,
                &weight,
                &bias,
                config.cached_tuning(),
            )
        }?;
        let graph = stream.end_capture()?.instantiate()?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        stream.sync_scope(|_| unsafe { graph.launch(&stream) })?;

        assert_close_output(&graph_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_layer_norm_tune_and_retune_reject_capture() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 8;
        let eps = 1e-5f32;
        let input_host = sparsemax_input(rows, columns);
        let weight_host = affine_weight(columns);
        let bias_host = affine_bias(columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let weight = DeviceMemory::from_slice(&stream.context(), &weight_host)?;
        let bias = DeviceMemory::from_slice(&stream.context(), &bias_host)?;
        let options = TuneOptions::create(0, 1, 1);
        let config = NormalizationConfig::create(rows, columns, eps);
        let mut out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: autotuning rejects capture before submitting any work.
        let error = unsafe {
            layer_norm_affine_f32(
                &stream,
                &mut out,
                &input,
                &weight,
                &bias,
                config.tune(options),
            )
        }
        .unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "layer_norm_affine_f32"
        ));

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: autotuning rejects capture before submitting any work.
        let error = unsafe {
            layer_norm_affine_f32(
                &stream,
                &mut out,
                &input,
                &weight,
                &bias,
                config.retune(options),
            )
        }
        .unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "layer_norm_affine_f32"
        ));
        Ok(())
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_explicit_rms_norm_weight_offset_tuning_matches_host() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 5;
        let eps = 1e-5f32;
        let weight_offset = 1.0f32;
        let input_host = sparsemax_input(rows, columns);
        let weight_host = affine_weight(columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let weight = DeviceMemory::from_slice(&stream.context(), &weight_host)?;
        let expected =
            expected_rms_norm_weight_offset(&input_host, &weight_host, columns, eps, weight_offset);
        let config = RmsNormWeightOffsetConfig::create(rows, columns, eps, weight_offset)
            .explicit_tuning(NormalizationTuning::tile_width(8));
        let mut out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe { rms_norm_weight_offset_f32(&stream, &mut out, &input, &weight, config) })
        })??;
        assert_close_output(&out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_tune_and_retune_sparsemax_relaunch_winning_candidate() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 6;
        let input_host = sparsemax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_sparsemax(&input_host, columns);
        let options = TuneOptions::create(0, 1, 2);
        let config = SparsemaxConfig::create(rows, columns);
        let mut tune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut retune_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe { sparsemax_f32(&stream, &mut tune_out, &input, config.tune(options)) })
        })??;
        assert_sparsemax_output(&tune_out, &expected, &stream)?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe { sparsemax_f32(&stream, &mut retune_out, &input, config.retune(options)) })
        })??;
        assert_sparsemax_output(&retune_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_cached_sparsemax_tuning_can_be_captured() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 7;
        let input_host = sparsemax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let expected = expected_sparsemax(&input_host, columns);
        let options = TuneOptions::create(0, 1, 1);
        let config = SparsemaxConfig::create(rows, columns);
        let mut warmup_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;
        let mut graph_out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        // SAFETY: this draining scope retains every input and output through completion on all paths.
        stream.sync_scope(|_| {
            Ok(unsafe { sparsemax_f32(&stream, &mut warmup_out, &input, config.tune(options)) })
        })??;
        assert_sparsemax_output(&warmup_out, &expected, &stream)?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: capture records this access; the scoped replay retains every allocation until completion.
        unsafe { sparsemax_f32(&stream, &mut graph_out, &input, config.cached_tuning()) }?;
        let graph = stream.end_capture()?.instantiate()?;
        stream.sync_scope(|_| unsafe { graph.launch(&stream) })?;

        assert_sparsemax_output(&graph_out, &expected, &stream)
    }

    #[cfg(feature = "dtype_f32")]
    #[test]
    fn test_sparsemax_tune_and_retune_reject_capture() -> Result<()> {
        let Some(stream) = test_stream()? else {
            return Ok(());
        };
        let rows = 2;
        let columns = 8;
        let input_host = sparsemax_input(rows, columns);
        let input = DeviceMemory::from_slice(&stream.context(), &input_host)?;
        let options = TuneOptions::create(0, 1, 1);
        let config = SparsemaxConfig::create(rows, columns);
        let mut out = DeviceMemory::<f32>::zeroed(&stream.context(), input_host.len())?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: autotuning rejects capture before submitting any work.
        let error =
            unsafe { sparsemax_f32(&stream, &mut out, &input, config.tune(options)) }.unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "sparsemax_f32"
        ));

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        // SAFETY: autotuning rejects capture before submitting any work.
        let error = unsafe { sparsemax_f32(&stream, &mut out, &input, config.retune(options)) }
            .unwrap_err();
        drop(stream.end_capture());
        assert!(matches!(
            error,
            Error::AutotuneDuringCapture { op } if op == "sparsemax_f32"
        ));
        Ok(())
    }

    #[test]
    fn test_bf16_normalization_validation_accepts_affine_lengths() -> Result<()> {
        assert_eq!(
            validate_normalization_lengths(24, 24, Some(6), Some(6), 4, 6)?,
            24
        );
        Ok(())
    }

    #[test]
    fn test_bf16_normalization_validation_rejects_short_weight() {
        assert!(matches!(
            validate_normalization_lengths(24, 24, Some(5), None, 4, 6),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[test]
    fn test_bf16_normalization_validation_rejects_zero_cols() {
        assert!(matches!(
            validate_normalization_lengths(0, 0, None, None, 4, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn test_group_norm_validation_accepts_exact_lengths() -> Result<()> {
        let config = group_norm_validation_config();
        validate_group_norm(48, 48, 6, 6, config)?;
        Ok(())
    }

    #[test]
    fn test_group_norm_validation_rejects_uneven_groups() {
        let mut config = group_norm_validation_config();
        config.groups = 4;
        assert!(matches!(
            validate_group_norm(48, 48, 6, 6, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[test]
    fn test_group_norm_validation_rejects_short_weight() {
        let config = group_norm_validation_config();
        assert!(matches!(
            validate_group_norm(48, 48, 5, 6, config),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    fn group_norm_validation_config() -> GroupNormConfig {
        GroupNormConfig {
            batch: 2,
            channels: 6,
            groups: 2,
            spatial_len: 4,
            eps: 1e-5,
        }
    }
}
