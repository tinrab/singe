//! Inference-oriented fused operations that combine common model steps.
//!
//! Pure destinations and scratch accept pending storage. Dense transformations
//! produce their logical output elements, including separate residual and Q/K
//! outputs; configured stride gaps and unused capacity remain untouched.
//! KV-cache updates produce only the selected token rows in each head or
//! sequence layout. They do not initialize the rest of a cache.
//! Sinkhorn and cache compaction consume existing destination values and retain
//! initialized mutable access. Compaction and sliding-window append preserve
//! cache positions outside their written token ranges.

#[cfg(feature = "kernel_quantization")]
mod quantization;

#[cfg(feature = "kernel_quantization")]
pub use quantization::*;

#[cfg(feature = "dtype_f16")]
use std::fmt::{self, Formatter};
use std::{fmt::Debug, sync::Arc};

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
        interop::{input_pointer, write_only_pointer},
        positional,
        positional::QkRotaryEmbeddingConfig,
    },
    error::{Error, Result},
    utility::{
        checked_element_count, checked_rank4_len, ensure_len, ensure_rank2_reach,
        ensure_rank4_reach,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusedTileTuning {
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

impl FusedTileTuning {
    #[must_use]
    pub const fn tile(tile_width: usize) -> Self {
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
    pub const fn with_compile(mut self, compile: KernelCompileConfig) -> Self {
        self.compile = compile;
        self
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fused1dConfig {
    pub length: usize,
    pub tuning: TuningChoice<FusedTileTuning>,
}

impl Fused1dConfig {
    #[must_use]
    pub const fn create(length: usize) -> Self {
        Self {
            length,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<FusedTileTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: FusedTileTuning) -> Self {
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

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fused2dConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<FusedTileTuning>,
}

impl Fused2dConfig {
    #[must_use]
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: TuningChoice<FusedTileTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: FusedTileTuning) -> Self {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusedRmsNormConfig<T> {
    pub rows: usize,
    pub columns: usize,
    pub eps: T,
    pub weight_offset: T,
    pub tuning: TuningChoice<FusedTileTuning>,
}

impl<T: Copy> FusedRmsNormConfig<T> {
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
    pub const fn with_tuning(mut self, tuning: TuningChoice<FusedTileTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub const fn explicit_tuning(self, tuning: FusedTileTuning) -> Self {
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

#[derive(Debug, Clone, Copy)]
pub struct RopeQkCacheUpdateConfig {
    pub rotary: QkRotaryEmbeddingConfig,
    pub cache_max_seq: usize,
    pub cache_position_start: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BshdRopeQkCacheUpdateConfig {
    pub seq_len: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub cache_max_seq: usize,
    pub cache_position_start: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvCacheCompactConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub source_start: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct KvCacheSlidingWindowAppendConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub cache_start: usize,
    pub cache_len: usize,
    pub append_start: usize,
    pub window: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSlidingWindowAppendState {
    pub cache_start: usize,
    pub cache_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcApplyResidualConfig {
    pub batch: usize,
    pub n: usize,
    pub channels: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcGemmRmsScaleConfig {
    pub rows: usize,
    pub columns: usize,
    pub reduction: usize,
    pub n: usize,
    pub alpha_pre: f32,
    pub alpha_post: f32,
    pub alpha_res: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhcSplitGemmRmsScaleConfig {
    pub base: MhcGemmRmsScaleConfig,
    pub split_k: usize,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheSlidingWindowAppendPlan {
    pre_compact: Option<KvCacheCompactConfig>,
    append_position: usize,
    post_compact: Option<KvCacheCompactConfig>,
    state: KvCacheSlidingWindowAppendState,
}

macro_rules! binary_fused_fn {
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
            let len = validate_binary_fused_lengths(out.len(), lhs.len(), rhs.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
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

macro_rules! autotuned_binary_fused_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
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
            config: Fused1dConfig,
        ) -> Result<()> {
            fused_1d(
                stream,
                out,
                lhs,
                rhs,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }

        /// Times candidate tiles and enqueues the selected fused gate.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// These requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: Fused1dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_1d(
                stream,
                out,
                lhs,
                rhs,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
            )
        }
    };
}

macro_rules! unary_fused_fn {
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
        ) -> Result<()> {
            let len = ensure_unary_lengths(out.len(), input.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    len,
                )
            }
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! binary_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(lhs.len(), len)?;
            ensure_len(rhs.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    rows,
                    cols,
                )
            }
        }
    };
}

macro_rules! autotuned_binary_fused_2d_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d(
                stream,
                out,
                lhs,
                rhs,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d(
                stream,
                out,
                lhs,
                rhs,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }
    };
}

macro_rules! silu_and_mul_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            validate_silu_and_mul_packed_lengths(out.len(), input.len(), rows, hidden)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    rows,
                    hidden,
                )
            }
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! ternary_fused_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            let len = $crate::cuda::utility::ensure_binary_lengths(out.len(), a.len(), b.len())?;
            ensure_len(c.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(a),
                    input_pointer(b),
                    input_pointer(c),
                    len,
                )
            }
        }
    };
}

macro_rules! autotuned_ternary_fused_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused1dConfig,
        ) -> Result<()> {
            fused_1d_ternary(
                stream,
                out,
                a,
                b,
                c,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused1dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_1d_ternary(
                stream,
                out,
                a,
                b,
                c,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! ternary_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(a.len(), len)?;
            ensure_len(b.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(a),
                    input_pointer(b),
                    input_pointer(c),
                    rows,
                    cols,
                )
            }
        }
    };
}

macro_rules! autotuned_ternary_fused_2d_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d_ternary(
                stream,
                out,
                a,
                b,
                c,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d_ternary(
                stream,
                out,
                a,
                b,
                c,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }
    };
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! conditional_fused_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            let len = $crate::cuda::utility::ensure_binary_lengths(out.len(), a.len(), b.len())?;
            ensure_len(condition.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(condition),
                    input_pointer(a),
                    input_pointer(b),
                    input_pointer(c),
                    len,
                )
            }
        }
    };
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
macro_rules! autotuned_conditional_fused_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused1dConfig,
        ) -> Result<()> {
            fused_1d_conditional(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused1dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_1d_conditional(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }
    };
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! conditional_fused_2d_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(condition.len(), len)?;
            ensure_len(a.len(), len)?;
            ensure_len(b.len(), len)?;
            ensure_len(c.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(condition),
                    input_pointer(a),
                    input_pointer(b),
                    input_pointer(c),
                    rows,
                    cols,
                )
            }
        }
    };
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
macro_rules! autotuned_conditional_fused_2d_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d_conditional(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            condition: &impl DeviceInput<u8>,
            a: &impl DeviceInput<$ty>,
            b: &impl DeviceInput<$ty>,
            c: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d_conditional(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                    )
                },
            )
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! bias_fused_fn {
    ($description:literal, $name:ident, $ty:ty) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Input and column bias must be initialized and belong to the stream's context.
        /// Retain all ranges and exclusive output access until completion on all paths
        /// and throughout graph replay. Order previous accesses before the launch.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(bias.len(), cols)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    input_pointer(bias),
                    rows,
                    cols,
                )
            }
        }
    };
}

macro_rules! autotuned_bias_fused_fn {
    ($description:literal, $name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// Input and column bias must be initialized and belong to the stream's context.
        /// Retain all ranges and exclusive output access until completion on all paths
        /// and throughout graph replay. Order previous accesses before the launch.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d_bias(
                stream,
                out,
                input,
                bias,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }

        /// Tunes the bias epilogue and enqueues the selected candidate.
        ///
        /// # Safety
        ///
        /// The bias epilogue's input validity, ordering, and asynchronous resource
        /// lifetime requirements apply to every candidate launch.
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d_bias(
                stream,
                out,
                input,
                bias,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7| unsafe {
                    cutile::fused::$backend_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
                },
            )
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! affine_fused_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scale: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(scale.len(), cols)?;
            ensure_len(bias.len(), cols)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(scale),
                    input_pointer(bias),
                    rows,
                    cols,
                )
            }
        }
    };
}

macro_rules! autotuned_affine_fused_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scale: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d_column_ternary(
                stream,
                out,
                input,
                scale,
                bias,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            scale: &impl DeviceInput<$ty>,
            bias: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d_column_ternary(
                stream,
                out,
                input,
                scale,
                bias,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }
    };
}

#[allow(
    unused_macros,
    reason = "macros are used only by some feature combinations"
)]
macro_rules! column2_fused_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            ensure_len(lhs.len(), cols)?;
            ensure_len(rhs.len(), cols)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    rows,
                    cols,
                )
            }
        }
    };
}

macro_rules! autotuned_column2_fused_fn {
    ($name:ident, $tune_name:ident, $ty:ty, $dtype:literal, $backend:ident, $backend_tuned:ident) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
        ) -> Result<()> {
            fused_2d_column_ternary(
                stream,
                out,
                input,
                lhs,
                rhs,
                config,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
                    cutile::fused::$backend(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
                },
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }

        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $tune_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            config: Fused2dConfig,
            options: TuneOptions,
        ) -> Result<TuneReport<FusedTileTuning>> {
            tune_fused_2d_column_ternary(
                stream,
                out,
                input,
                lhs,
                rhs,
                config,
                options,
                $dtype,
                stringify!($name),
                // SAFETY: the caller upholds this function's safety contract.
                |arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8| unsafe {
                    cutile::fused::$backend_tuned(
                        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                    )
                },
            )
        }
    };
}

fn validate_kv_cache_compact(
    k_cache_len: usize,
    v_cache_len: usize,
    config: KvCacheCompactConfig,
) -> Result<()> {
    if config.heads == 0 || config.head_dim == 0 || config.max_seq == 0 || config.token_count == 0 {
        return Err(Error::InvalidLength);
    }
    let source_end = config
        .source_start
        .checked_add(config.token_count)
        .ok_or(Error::SizeOverflow)?;
    if source_end > config.max_seq {
        return Err(Error::InvalidLength);
    }
    let cache_len = checked_element_count(
        checked_element_count(config.heads, config.max_seq)?,
        config.head_dim,
    )?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    Ok(())
}

fn validate_rms_norm_add(
    out_len: usize,
    residual_out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(residual_out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(residual_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_rms_norm_add_wide(
    out_len: usize,
    residual_out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    partial_sum_len: usize,
    row_sum_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_rms_norm_add(
        out_len,
        residual_out_len,
        input_len,
        residual_len,
        weight_len,
        rows,
        cols,
    )?;
    let Some(plan) = cutile::fused::rms_norm_wide_plan(rows, cols)? else {
        return Ok(len);
    };
    ensure_len(partial_sum_len, plan.partial_len)?;
    ensure_len(row_sum_len, plan.row_sum_len)?;
    Ok(len)
}

fn validate_binary_fused_lengths(out_len: usize, lhs_len: usize, rhs_len: usize) -> Result<usize> {
    ensure_len(lhs_len, out_len)?;
    ensure_len(rhs_len, out_len)?;
    Ok(out_len)
}

fn validate_binary_fused_1d_lengths(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: Fused1dConfig,
) -> Result<usize> {
    ensure_len(out_len, config.length)?;
    ensure_len(lhs_len, config.length)?;
    ensure_len(rhs_len, config.length)?;
    Ok(config.length)
}

fn validate_ternary_fused_1d_lengths(
    out_len: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
    config: Fused1dConfig,
) -> Result<usize> {
    ensure_len(out_len, config.length)?;
    ensure_len(a_len, config.length)?;
    ensure_len(b_len, config.length)?;
    ensure_len(c_len, config.length)?;
    Ok(config.length)
}

fn validate_binary_fused_2d_lengths(
    out_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: Fused2dConfig,
) -> Result<usize> {
    let len = validate_matrix_shape(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(lhs_len, len)?;
    ensure_len(rhs_len, len)?;
    Ok(len)
}

fn validate_ternary_fused_2d_lengths(
    out_len: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
    config: Fused2dConfig,
) -> Result<usize> {
    let len = validate_matrix_shape(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(a_len, len)?;
    ensure_len(b_len, len)?;
    ensure_len(c_len, len)?;
    Ok(len)
}

fn validate_bias_fused_2d_lengths(
    out_len: usize,
    input_len: usize,
    bias_len: usize,
    config: Fused2dConfig,
) -> Result<usize> {
    let len = validate_matrix_shape(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(bias_len, config.columns)?;
    Ok(len)
}

fn validate_column_ternary_fused_2d_lengths(
    out_len: usize,
    input_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    config: Fused2dConfig,
) -> Result<usize> {
    let len = validate_matrix_shape(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(lhs_len, config.columns)?;
    ensure_len(rhs_len, config.columns)?;
    Ok(len)
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn validate_conditional_fused_1d_lengths(
    out_len: usize,
    condition_len: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
    config: Fused1dConfig,
) -> Result<usize> {
    ensure_len(out_len, config.length)?;
    ensure_len(condition_len, config.length)?;
    ensure_len(a_len, config.length)?;
    ensure_len(b_len, config.length)?;
    ensure_len(c_len, config.length)?;
    Ok(config.length)
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn validate_conditional_fused_2d_lengths(
    out_len: usize,
    condition_len: usize,
    a_len: usize,
    b_len: usize,
    c_len: usize,
    config: Fused2dConfig,
) -> Result<usize> {
    let len = validate_matrix_shape(config.rows, config.columns)?;
    ensure_len(out_len, len)?;
    ensure_len(condition_len, len)?;
    ensure_len(a_len, len)?;
    ensure_len(b_len, len)?;
    ensure_len(c_len, len)?;
    Ok(len)
}

fn launch_fused_1d_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused1dConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input_pointer(lhs),
        input_pointer(rhs),
        config.length,
    )
}

fn launch_fused_1d_ternary_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
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
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.length,
    )
}

fn launch_fused_1d_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused1dConfig,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
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
        input_pointer(lhs),
        input_pointer(rhs),
        config.length,
        tuning.tile_width,
        tuning.compile,
    )
}

fn launch_fused_1d_ternary_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    tuning: FusedTileTuning,
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
        usize,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.length,
        tuning.tile_width,
        tuning.compile,
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn launch_fused_1d_conditional_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(condition),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.length,
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn launch_fused_1d_conditional_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(condition),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.length,
        tuning.tile_width,
        tuning.compile,
    )
}

fn launch_fused_2d_default<T, F>(
    stream: &Stream,
    out: DevicePointer<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        out,
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
    )
}

fn launch_fused_2d_ternary_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
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
        usize,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.rows,
        config.columns,
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn launch_fused_2d_conditional_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(condition),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.rows,
        config.columns,
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn launch_fused_2d_conditional_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
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
        write_only_pointer(out),
        input_pointer(condition),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn launch_fused_2d_with_tuning<T, F>(
    stream: &Stream,
    out: DevicePointer<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
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
        out,
        input_pointer(lhs),
        input_pointer(rhs),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn launch_fused_2d_ternary_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    tuning: FusedTileTuning,
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
        usize,
        usize,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(a),
        input_pointer(b),
        input_pointer(c),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn fused_1d<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused1dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_binary_fused_1d_lengths(out.len(), lhs.len(), rhs.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_1d_default(stream, out, lhs, rhs, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_1d_with_tuning(
                stream,
                out,
                lhs,
                rhs,
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

    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_1d_with_tuning(stream, out, lhs, rhs, config, *tuning, launch_tuned)
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| launch_fused_1d_with_tuning(stream, out, lhs, rhs, config, tuning, launch_tuned),
    )
}

fn fused_1d_ternary<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_ternary_fused_1d_lengths(out.len(), a.len(), b.len(), c.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_1d_ternary_default(stream, out, a, b, c, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_1d_ternary_with_tuning(
                stream,
                out,
                a,
                b,
                c,
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

    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_1d_ternary_with_tuning(
                    stream,
                    out,
                    a,
                    b,
                    c,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_1d_ternary_with_tuning(stream, out, a, b, c, config, tuning, launch_tuned)
        },
    )
}

fn fused_2d<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_binary_fused_2d_lengths(out.len(), lhs.len(), rhs.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_2d_default(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                lhs,
                rhs,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                lhs,
                rhs,
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

    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_2d_with_tuning(
                    stream,
                    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                    unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                    lhs,
                    rhs,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                lhs,
                rhs,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

fn fused_2d_ternary<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_ternary_fused_2d_lengths(out.len(), a.len(), b.len(), c.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_2d_ternary_default(stream, out, a, b, c, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_2d_ternary_with_tuning(
                stream,
                out,
                a,
                b,
                c,
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

    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_2d_ternary_with_tuning(
                    stream,
                    out,
                    a,
                    b,
                    c,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_2d_ternary_with_tuning(stream, out, a, b, c, config, tuning, launch_tuned)
        },
    )
}

fn fused_2d_bias<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    bias: &impl DeviceInput<T>,
    config: Fused2dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_bias_fused_2d_lengths(out.len(), input.len(), bias.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_2d_default(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                input,
                bias,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                input,
                bias,
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

    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_2d_with_tuning(
                    stream,
                    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                    unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                    input,
                    bias,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                input,
                bias,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

fn fused_2d_column_ternary<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_column_ternary_fused_2d_lengths(out.len(), input.len(), lhs.len(), rhs.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_2d_ternary_default(
                stream,
                out,
                input,
                lhs,
                rhs,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_2d_ternary_with_tuning(
                stream,
                out,
                input,
                lhs,
                rhs,
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

    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_2d_ternary_with_tuning(
                    stream,
                    out,
                    input,
                    lhs,
                    rhs,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_2d_ternary_with_tuning(
                stream,
                out,
                input,
                lhs,
                rhs,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn fused_1d_conditional<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_conditional_fused_1d_lengths(
        out.len(),
        condition.len(),
        a.len(),
        b.len(),
        c.len(),
        config,
    )?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_1d_conditional_default(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_1d_conditional_with_tuning(
                stream,
                out,
                condition,
                a,
                b,
                c,
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

    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_1d_conditional_with_tuning(
                    stream,
                    out,
                    condition,
                    a,
                    b,
                    c,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_1d_conditional_with_tuning(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn fused_2d_conditional<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_conditional_fused_2d_lengths(
        out.len(),
        condition.len(),
        a.len(),
        b.len(),
        c.len(),
        config,
    )?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_fused_2d_conditional_default(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_fused_2d_conditional_with_tuning(
                stream,
                out,
                condition,
                a,
                b,
                c,
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

    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        fused_tile_candidates,
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_fused_2d_conditional_with_tuning(
                    stream,
                    out,
                    condition,
                    a,
                    b,
                    c,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_fused_2d_conditional_with_tuning(
                stream,
                out,
                condition,
                a,
                b,
                c,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

fn launch_rms_norm_add_default<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    residual_out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    residual: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        write_only_pointer(residual_out),
        input_pointer(input),
        input_pointer(residual),
        input_pointer(weight),
        config.rows,
        config.columns,
        config.eps,
        config.weight_offset,
    )
}

fn launch_rms_norm_add_with_tuning<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    residual_out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    residual: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        write_only_pointer(residual_out),
        input_pointer(input),
        input_pointer(residual),
        input_pointer(weight),
        config.rows,
        config.columns,
        config.eps,
        config.weight_offset,
        tuning.tile_width,
        tuning.compile,
    )
}

fn launch_rms_norm_silu_mul_default<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    up: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(up),
        config.rows,
        config.columns,
        config.eps,
        config.weight_offset,
    )
}

fn launch_rms_norm_silu_mul_with_tuning<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    up: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    tuning: FusedTileTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    F: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
        usize,
        KernelCompileConfig,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(up),
        config.rows,
        config.columns,
        config.eps,
        config.weight_offset,
        tuning.tile_width,
        tuning.compile,
    )
}

fn fused_rms_norm_add<T, E, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    residual_out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    residual: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            E,
            E,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_rms_norm_add(
        out.len(),
        residual_out.len(),
        input.len(),
        residual.len(),
        weight.len(),
        config.rows,
        config.columns,
    )?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_rms_norm_add_default(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_rms_norm_add_with_tuning(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
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

    let key = fused_rms_norm_tune_key(stream, op, dtype, config.rows, config.columns)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || rms_norm_fused_tile_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_rms_norm_add_with_tuning(
                    stream,
                    out,
                    residual_out,
                    input,
                    residual,
                    weight,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_rms_norm_add_with_tuning(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

fn fused_rms_norm_silu_mul<T, E, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    up: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    dtype: &'static str,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    E: Copy,
    FDefault: FnOnce(
        &Arc<CutileStream>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
        E,
        E,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            E,
            E,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_rms_norm_silu_mul(
        out.len(),
        input.len(),
        weight.len(),
        up.len(),
        config.rows,
        config.columns,
    )?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_rms_norm_silu_mul_default(
                stream,
                out,
                input,
                weight,
                up,
                config,
                launch_default,
            );
        }
        TuningChoice::Explicit(tuning) => {
            return launch_rms_norm_silu_mul_with_tuning(
                stream,
                out,
                input,
                weight,
                up,
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

    let key = fused_rms_norm_tune_key(stream, op, dtype, config.rows, config.columns)?;
    let tuning = resolve_fused_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || rms_norm_fused_tile_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_rms_norm_silu_mul_with_tuning(
                    stream,
                    out,
                    input,
                    weight,
                    up,
                    config,
                    *tuning,
                    launch_tuned,
                )
            })
        },
    )?;
    tuning.map_or_else(
        || unreachable!("default and explicit fused tuning return before cache resolution"),
        |tuning| {
            launch_rms_norm_silu_mul_with_tuning(
                stream,
                out,
                input,
                weight,
                up,
                config,
                tuning,
                launch_tuned,
            )
        },
    )
}

fn tune_fused_2d<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_binary_fused_2d_lengths(out.len(), lhs.len(), rhs.len(), config)?;
    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| {
            launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                lhs,
                rhs,
                config,
                *tuning,
                launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_2d_with_tuning(
        stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        lhs,
        rhs,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn tune_fused_2d_bias<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    bias: &impl DeviceInput<T>,
    config: Fused2dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_bias_fused_2d_lengths(out.len(), input.len(), bias.len(), config)?;
    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| {
            launch_fused_2d_with_tuning(
                stream,
                // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
                unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
                input,
                bias,
                config,
                *tuning,
                launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_2d_with_tuning(
        stream,
        // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
        unsafe { DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _) },
        input,
        bias,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn tune_fused_1d<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused1dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_binary_fused_1d_lengths(out.len(), lhs.len(), rhs.len(), config)?;
    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| launch_fused_1d_with_tuning(stream, out, lhs, rhs, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_1d_with_tuning(stream, out, lhs, rhs, config, report.best, launch)?;
    Ok(report)
}

fn tune_fused_1d_ternary<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_ternary_fused_1d_lengths(out.len(), a.len(), b.len(), c.len(), config)?;
    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| launch_fused_1d_ternary_with_tuning(stream, out, a, b, c, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_1d_ternary_with_tuning(stream, out, a, b, c, config, report.best, launch)?;
    Ok(report)
}

fn tune_fused_2d_column_ternary<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    lhs: &impl DeviceInput<T>,
    rhs: &impl DeviceInput<T>,
    config: Fused2dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_column_ternary_fused_2d_lengths(out.len(), input.len(), lhs.len(), rhs.len(), config)?;
    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| {
            launch_fused_2d_ternary_with_tuning(
                stream, out, input, lhs, rhs, config, *tuning, launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_2d_ternary_with_tuning(stream, out, input, lhs, rhs, config, report.best, launch)?;
    Ok(report)
}

fn tune_fused_2d_ternary<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_ternary_fused_2d_lengths(out.len(), a.len(), b.len(), c.len(), config)?;
    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| launch_fused_2d_ternary_with_tuning(stream, out, a, b, c, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_2d_ternary_with_tuning(stream, out, a, b, c, config, report.best, launch)?;
    Ok(report)
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn tune_fused_1d_conditional<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused1dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_conditional_fused_1d_lengths(
        out.len(),
        condition.len(),
        a.len(),
        b.len(),
        c.len(),
        config,
    )?;
    let key = fused_1d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| {
            launch_fused_1d_conditional_with_tuning(
                stream, out, condition, a, b, c, config, *tuning, launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_1d_conditional_with_tuning(
        stream,
        out,
        condition,
        a,
        b,
        c,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

#[cfg(all(
    feature = "dtype_u8",
    any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64")
))]
fn tune_fused_2d_conditional<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    condition: &impl DeviceInput<u8>,
    a: &impl DeviceInput<T>,
    b: &impl DeviceInput<T>,
    c: &impl DeviceInput<T>,
    config: Fused2dConfig,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_conditional_fused_2d_lengths(
        out.len(),
        condition.len(),
        a.len(),
        b.len(),
        c.len(),
        config,
    )?;
    let key = fused_2d_tune_key(stream, op, dtype, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        fused_tile_candidates(),
        |tuning| {
            launch_fused_2d_conditional_with_tuning(
                stream, out, condition, a, b, c, config, *tuning, launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_fused_2d_conditional_with_tuning(
        stream,
        out,
        condition,
        a,
        b,
        c,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn tune_fused_rms_norm_add<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    residual_out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    residual: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    E: Copy,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            E,
            E,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_rms_norm_add(
        out.len(),
        residual_out.len(),
        input.len(),
        residual.len(),
        weight.len(),
        config.rows,
        config.columns,
    )?;
    let key = fused_rms_norm_tune_key(stream, op, dtype, config.rows, config.columns)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        rms_norm_fused_tile_candidates(config.columns),
        |tuning| {
            launch_rms_norm_add_with_tuning(
                stream,
                out,
                residual_out,
                input,
                residual,
                weight,
                config,
                *tuning,
                launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_rms_norm_add_with_tuning(
        stream,
        out,
        residual_out,
        input,
        residual,
        weight,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn tune_fused_rms_norm_silu_mul<T, E, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<T>,
    weight: &impl DeviceInput<T>,
    up: &impl DeviceInput<T>,
    config: FusedRmsNormConfig<E>,
    options: TuneOptions,
    dtype: &'static str,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<FusedTileTuning>>
where
    T: DeviceRepr,
    E: Copy,
    F: Copy
        + Fn(
            &Arc<CutileStream>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            DevicePointer<T>,
            usize,
            usize,
            E,
            E,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_rms_norm_silu_mul(
        out.len(),
        input.len(),
        weight.len(),
        up.len(),
        config.rows,
        config.columns,
    )?;
    let key = fused_rms_norm_tune_key(stream, op, dtype, config.rows, config.columns)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        rms_norm_fused_tile_candidates(config.columns),
        |tuning| {
            launch_rms_norm_silu_mul_with_tuning(
                stream, out, input, weight, up, config, *tuning, launch,
            )
        },
    )?;
    autotune::cache().insert(key, report.best);
    launch_rms_norm_silu_mul_with_tuning(
        stream,
        out,
        input,
        weight,
        up,
        config,
        report.best,
        launch,
    )?;
    Ok(report)
}

fn fused_1d_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: Fused1dConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Fused1d {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            length: config.length,
        },
    ))
}

fn fused_rms_norm_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    rows: usize,
    columns: usize,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Fused2d {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows,
            columns,
        },
    ))
}

fn fused_2d_tune_key(
    stream: &Stream,
    op: &'static str,
    dtype: &'static str,
    config: Fused2dConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::Fused2d {
            dtype: dtype.to_owned(),
            op_kind: op.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn fused_tile_candidates() -> Vec<FusedTileTuning> {
    [128, 256, 512, 1024]
        .into_iter()
        .map(FusedTileTuning::tile)
        .collect()
}

fn rms_norm_fused_tile_candidates(columns: usize) -> Vec<FusedTileTuning> {
    [1, 8, 32, 64, 128, 256, 512, 1024]
        .into_iter()
        .filter(|tile_width| *tile_width >= columns)
        .map(FusedTileTuning::tile)
        .collect()
}

fn resolve_fused_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<FusedTileTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<FusedTileTuning>>
where
    C: FnOnce() -> Vec<FusedTileTuning>,
    F: FnOnce(TuneOptions, Vec<FusedTileTuning>) -> Result<TuneReport<FusedTileTuning>>,
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

fn validate_silu_and_mul_packed_lengths(
    out_len: usize,
    input_len: usize,
    rows: usize,
    hidden: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, hidden)?;
    let input_len_expected = checked_element_count(len, 2)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, input_len_expected)?;
    Ok(len)
}

fn validate_rms_norm_silu_mul(
    out_len: usize,
    input_len: usize,
    weight_len: usize,
    up_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(weight_len, cols)?;
    ensure_len(up_len, len)?;
    Ok(len)
}

fn validate_rms_norm_gated_silu(
    out_len: usize,
    input_len: usize,
    gate_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(gate_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_dual_rms_norm(
    q_out_len: usize,
    k_out_len: usize,
    q_len: usize,
    k_len: usize,
    q_weight_len: usize,
    k_weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(q_out_len, len)?;
    ensure_len(k_out_len, len)?;
    ensure_len(q_len, len)?;
    ensure_len(k_len, len)?;
    ensure_len(q_weight_len, cols)?;
    ensure_len(k_weight_len, cols)?;
    Ok(len)
}

fn validate_rms_norm_residual_add(
    out_len: usize,
    input_len: usize,
    residual_len: usize,
    weight_len: usize,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let len = validate_matrix_shape(rows, cols)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)?;
    ensure_len(residual_len, len)?;
    ensure_len(weight_len, cols)?;
    Ok(len)
}

fn validate_mhc_apply_residual(
    out_len: usize,
    x_len: usize,
    f_out_len: usize,
    y_len: usize,
    config: MhcApplyResidualConfig,
) -> Result<usize> {
    if config.batch == 0 || config.n == 0 || config.channels == 0 {
        return Err(Error::InvalidLength);
    }
    let token_len = checked_element_count(config.batch, config.n)?;
    let out_expected = checked_element_count(token_len, config.channels)?;
    let f_out_expected = checked_element_count(config.batch, config.channels)?;
    validate_mhc_y_buffer(y_len, config.batch, config.n)?;
    ensure_len(out_len, out_expected)?;
    ensure_len(x_len, out_expected)?;
    ensure_len(f_out_len, f_out_expected)?;
    Ok(out_expected)
}

fn validate_mhc_y_buffer(y_len: usize, batch: usize, n: usize) -> Result<usize> {
    if batch == 0 || n == 0 {
        return Err(Error::InvalidLength);
    }
    let y_row = checked_element_count(n, n.checked_add(2).ok_or(Error::SizeOverflow)?)?;
    let y_expected = checked_element_count(batch, y_row)?;
    ensure_len(y_len, y_expected)?;
    Ok(y_expected)
}

fn validate_mhc_gemm_rms_scale(
    y_len: usize,
    r_len: usize,
    x_len: usize,
    w_len: usize,
    bias_len: usize,
    config: MhcGemmRmsScaleConfig,
) -> Result<usize> {
    if config.rows == 0 || config.columns == 0 || config.reduction == 0 || config.n == 0 {
        return Err(Error::InvalidLength);
    }
    let min_columns = checked_element_count(2, config.n)?;
    if config.columns < min_columns {
        return Err(Error::LengthMismatch);
    }
    let y_expected = checked_element_count(config.rows, config.columns)?;
    let x_expected = checked_element_count(config.rows, config.reduction)?;
    let w_expected = checked_element_count(config.reduction, config.columns)?;
    ensure_len(y_len, y_expected)?;
    ensure_len(r_len, config.rows)?;
    ensure_len(x_len, x_expected)?;
    ensure_len(w_len, w_expected)?;
    ensure_len(bias_len, config.columns)?;
    Ok(y_expected)
}

fn validate_mhc_split_gemm_rms(
    y_acc_len: usize,
    r_acc_len: usize,
    x_len: usize,
    w_len: usize,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<usize> {
    if config.split_k == 0 {
        return Err(Error::InvalidLength);
    }
    validate_mhc_base_config(config.base)?;
    let base_len = checked_element_count(config.base.rows, config.base.columns)?;
    let x_expected = checked_element_count(config.base.rows, config.base.reduction)?;
    let w_expected = checked_element_count(config.base.reduction, config.base.columns)?;
    ensure_len(x_len, x_expected)?;
    ensure_len(w_len, w_expected)?;
    let temp_len = checked_element_count(base_len, config.split_k)?;
    ensure_len(y_acc_len, temp_len)?;
    ensure_len(r_acc_len, temp_len)?;
    Ok(temp_len)
}

fn validate_mhc_finalize_scale_bias_sigmoid(
    y_len: usize,
    r_len: usize,
    y_acc_len: usize,
    r_acc_len: usize,
    bias_len: usize,
    config: MhcSplitGemmRmsScaleConfig,
) -> Result<usize> {
    if config.split_k == 0 {
        return Err(Error::InvalidLength);
    }
    validate_mhc_base_config(config.base)?;
    let base_len = checked_element_count(config.base.rows, config.base.columns)?;
    let temp_len = checked_element_count(base_len, config.split_k)?;
    ensure_len(y_len, base_len)?;
    ensure_len(r_len, config.base.rows)?;
    ensure_len(y_acc_len, temp_len)?;
    ensure_len(r_acc_len, temp_len)?;
    ensure_len(bias_len, config.base.columns)?;
    Ok(base_len)
}

fn validate_mhc_base_config(config: MhcGemmRmsScaleConfig) -> Result<()> {
    if config.rows == 0 || config.columns == 0 || config.reduction == 0 || config.n == 0 {
        return Err(Error::InvalidLength);
    }
    let min_columns = checked_element_count(2, config.n)?;
    if config.columns < min_columns {
        return Err(Error::LengthMismatch);
    }
    Ok(())
}

fn validate_matrix_shape(rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidLength);
    }
    checked_element_count(rows, cols)
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
mod bias_affine;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
mod binary;
#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
mod cache;
#[cfg(any(
    all(feature = "dtype_f16", feature = "dtype_u8"),
    all(feature = "dtype_f32", feature = "dtype_u8"),
    all(feature = "dtype_f64", feature = "dtype_u8")
))]
mod conditional;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
mod normalization;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
mod ternary;

#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
pub use bias_affine::*;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
pub use binary::*;
#[cfg(any(feature = "dtype_f16", feature = "dtype_f32", feature = "dtype_f64"))]
pub use cache::*;
#[cfg(any(
    all(feature = "dtype_f16", feature = "dtype_u8"),
    all(feature = "dtype_f32", feature = "dtype_u8"),
    all(feature = "dtype_f64", feature = "dtype_u8")
))]
pub use conditional::*;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
pub use normalization::*;
#[cfg(any(
    feature = "dtype_bf16",
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_f64"
))]
pub use ternary::*;

#[cfg(test)]
mod tests;

#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
fn validate_kv_cache_sliding_window_append(
    k_cache_len: usize,
    v_cache_len: usize,
    new_k_len: usize,
    new_v_len: usize,
    seq_len: usize,
    config: KvCacheSlidingWindowAppendConfig,
) -> Result<KvCacheSlidingWindowAppendPlan> {
    if config.heads == 0
        || config.head_dim == 0
        || config.max_seq == 0
        || config.window == 0
        || config.cache_len > config.max_seq
        || config.window > config.max_seq
    {
        return Err(Error::InvalidLength);
    }
    let expected_append_start = config
        .cache_start
        .checked_add(config.cache_len)
        .ok_or(Error::SizeOverflow)?;
    if config.append_start != expected_append_start {
        return Err(Error::InvalidLength);
    }
    if seq_len > config.window {
        return Err(Error::LengthMismatch);
    }
    let cache_len = checked_element_count(
        checked_element_count(config.heads, config.max_seq)?,
        config.head_dim,
    )?;
    let new_len = checked_element_count(
        checked_element_count(seq_len, config.heads)?,
        config.head_dim,
    )?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    ensure_len(new_k_len, new_len)?;
    ensure_len(new_v_len, new_len)?;
    let appended_end = config
        .append_start
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    let total_len = config
        .cache_len
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    let retained_len = total_len.min(config.window);
    let new_cache_start = appended_end
        .checked_sub(retained_len)
        .ok_or(Error::SizeOverflow)?;
    let retained_existing_start = new_cache_start.max(config.cache_start);
    let retained_existing_len = config
        .append_start
        .checked_sub(retained_existing_start)
        .ok_or(Error::SizeOverflow)?;
    let pre_compact_source_start = retained_existing_start
        .checked_sub(config.cache_start)
        .ok_or(Error::SizeOverflow)?;
    let needs_pre_compact = total_len > config.max_seq && retained_existing_len > 0;
    let pre_compact = if needs_pre_compact {
        Some(KvCacheCompactConfig {
            heads: config.heads,
            head_dim: config.head_dim,
            max_seq: config.max_seq,
            source_start: pre_compact_source_start,
            token_count: retained_existing_len,
        })
    } else {
        None
    };
    let append_position = if retained_existing_len == 0 {
        0
    } else if needs_pre_compact {
        retained_existing_len
    } else {
        config.cache_len
    };
    let compacted_len = append_position
        .checked_add(seq_len)
        .ok_or(Error::SizeOverflow)?;
    if compacted_len > config.max_seq {
        return Err(Error::LengthMismatch);
    }
    let post_compact = if compacted_len > retained_len {
        Some(KvCacheCompactConfig {
            heads: config.heads,
            head_dim: config.head_dim,
            max_seq: config.max_seq,
            source_start: compacted_len - retained_len,
            token_count: retained_len,
        })
    } else {
        None
    };
    Ok(KvCacheSlidingWindowAppendPlan {
        pre_compact,
        append_position,
        post_compact,
        state: KvCacheSlidingWindowAppendState {
            cache_start: new_cache_start,
            cache_len: retained_len,
        },
    })
}

#[cfg(feature = "dtype_u32")]
macro_rules! kv_cache_update_dynpos_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            k_cache: &mut impl DeviceOutput<$ty>,
            v_cache: &mut impl DeviceOutput<$ty>,
            new_k: &impl DeviceInput<$ty>,
            new_v: &impl DeviceInput<$ty>,
            position_start: &impl DeviceInput<u32>,
            seq_len: usize,
            heads: usize,
            head_dim: usize,
            max_seq: usize,
        ) -> Result<()> {
            let new_len = checked_element_count(checked_element_count(seq_len, heads)?, head_dim)?;
            let cache_len =
                checked_element_count(checked_element_count(heads, max_seq)?, head_dim)?;
            ensure_len(new_k.len(), new_len)?;
            ensure_len(new_v.len(), new_len)?;
            ensure_len(k_cache.len(), cache_len)?;
            ensure_len(v_cache.len(), cache_len)?;
            ensure_len(position_start.len(), 1)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(k_cache),
                    write_only_pointer(v_cache),
                    input_pointer(new_k),
                    input_pointer(new_v),
                    input_pointer(position_start),
                    seq_len,
                    heads,
                    head_dim,
                    max_seq,
                )
            }
        }
    };
}

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f16, f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f32, f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_dynpos_f64, f64);

#[cfg(all(feature = "dtype_f16", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f16, f16);

#[cfg(all(feature = "dtype_f32", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f32, f32);

#[cfg(all(feature = "dtype_f64", feature = "dtype_u32"))]
kv_cache_update_dynpos_fn!(kv_cache_update_seq_dynpos_f64, f64);

macro_rules! rope_qk_fn {
    ($name:ident, $target:ident, $ty:ty) => {
        /// Writes query and key embeddings by rotating adjacent feature pairs.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Retain input allocations and exclusive, mutually disjoint destinations
        /// in the stream's context through completion and captured replay.
        /// Order prior accesses before launch and keep outputs disjoint from inputs.
        pub unsafe fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceOutput<$ty>,
            k_out: &mut impl DeviceOutput<$ty>,
            q_input: &impl DeviceInput<$ty>,
            k_input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { positional::$target(stream, q_out, k_out, q_input, k_input, cos, sin, config) }
        }
    };
}

#[cfg(feature = "dtype_f16")]
rope_qk_fn!(rope_qk_f16, rotary_embedding_qk_f16, f16);

#[cfg(feature = "dtype_f32")]
rope_qk_fn!(rope_qk_f32, rotary_embedding_qk_f32, f32);

#[cfg(feature = "dtype_f64")]
rope_qk_fn!(rope_qk_f64, rotary_embedding_qk_f64, f64);

macro_rules! rope_qk_cache_update_fn {
    ($name:ident, $ty:ty) => {
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $name(
            stream: &Stream,
            q_out: &mut impl DeviceOutput<$ty>,
            k_out: &mut impl DeviceOutput<$ty>,
            k_cache: &mut impl DeviceOutput<$ty>,
            q_input: &impl DeviceInput<$ty>,
            k_input: &impl DeviceInput<$ty>,
            cos: &impl DeviceInput<$ty>,
            sin: &impl DeviceInput<$ty>,
            config: RopeQkCacheUpdateConfig,
        ) -> Result<()> {
            validate_rope_qk_cache_update(
                q_out.len(),
                k_out.len(),
                k_cache.len(),
                q_input.len(),
                k_input.len(),
                cos.len(),
                sin.len(),
                config,
            )?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::fused::$name(
                    &stream,
                    write_only_pointer(q_out),
                    write_only_pointer(k_out),
                    write_only_pointer(k_cache),
                    input_pointer(q_input),
                    input_pointer(k_input),
                    input_pointer(cos),
                    input_pointer(sin),
                    config,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f16, f16);

#[cfg(feature = "dtype_f32")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f32, f32);

#[cfg(feature = "dtype_f64")]
rope_qk_cache_update_fn!(rope_qk_cache_update_f64, f64);

fn validate_rope_qk_cache_update(
    q_out_len: usize,
    k_out_len: usize,
    k_cache_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: RopeQkCacheUpdateConfig,
) -> Result<()> {
    let rotary = config.rotary;
    let q_output_len = checked_rank4_len(rotary.q_output_dimensions)?;
    let k_output_len = checked_rank4_len(rotary.k_output_dimensions)?;
    ensure_len(q_out_len, q_output_len)?;
    ensure_len(k_out_len, k_output_len)?;
    ensure_rank4_reach(
        q_input_len,
        rotary.q_output_dimensions,
        rotary.q_input_strides,
    )?;
    ensure_rank4_reach(
        k_input_len,
        rotary.k_output_dimensions,
        rotary.k_input_strides,
    )?;
    if rotary.q_output_dimensions[0] != rotary.k_output_dimensions[0]
        || rotary.q_output_dimensions[2] != rotary.k_output_dimensions[2]
        || rotary.q_output_dimensions[3] != rotary.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let cache_end = config
        .cache_position_start
        .checked_add(rotary.k_output_dimensions[2])
        .ok_or(Error::SizeOverflow)?;
    if cache_end > config.cache_max_seq {
        return Err(Error::LengthMismatch);
    }
    let cache_len = checked_rank4_len([
        rotary.k_output_dimensions[0],
        rotary.k_output_dimensions[1],
        config.cache_max_seq,
        rotary.k_output_dimensions[3],
    ])?;
    ensure_len(k_cache_len, cache_len)?;
    if rotary.rotary_pairs == 0 {
        return Ok(());
    }
    if rotary
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > rotary.q_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let trig_dimensions = [rotary.q_output_dimensions[2], rotary.rotary_pairs];
    ensure_rank2_reach(cos_len, trig_dimensions, rotary.cos_strides)?;
    ensure_rank2_reach(sin_len, trig_dimensions, rotary.sin_strides)
}

#[cfg(feature = "dtype_f16")]
fn validate_bshd_rope_qk_cache_update(
    q_out_len: usize,
    k_out_len: usize,
    k_cache_len: usize,
    v_cache_len: usize,
    q_input_len: usize,
    k_input_len: usize,
    v_input_len: usize,
    cos_len: usize,
    sin_len: usize,
    config: BshdRopeQkCacheUpdateConfig,
) -> Result<()> {
    if config.seq_len == 0
        || config.query_heads == 0
        || config.key_value_heads == 0
        || config.head_dim == 0
        || config.cache_max_seq == 0
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(Error::InvalidLength);
    }
    let cache_end = config
        .cache_position_start
        .checked_add(config.seq_len)
        .ok_or(Error::SizeOverflow)?;
    if cache_end > config.cache_max_seq {
        return Err(Error::LengthMismatch);
    }
    let q_len = checked_element_count(
        checked_element_count(config.seq_len, config.query_heads)?,
        config.head_dim,
    )?;
    let kv_len = checked_element_count(
        checked_element_count(config.seq_len, config.key_value_heads)?,
        config.head_dim,
    )?;
    let cache_len = checked_element_count(
        checked_element_count(config.key_value_heads, config.cache_max_seq)?,
        config.head_dim,
    )?;
    let trig_len = checked_element_count(config.seq_len, config.head_dim / 2)?;
    ensure_len(q_out_len, q_len)?;
    ensure_len(q_input_len, q_len)?;
    ensure_len(k_out_len, kv_len)?;
    ensure_len(k_input_len, kv_len)?;
    ensure_len(v_input_len, kv_len)?;
    ensure_len(k_cache_len, cache_len)?;
    ensure_len(v_cache_len, cache_len)?;
    ensure_len(cos_len, trig_len)?;
    ensure_len(sin_len, trig_len)
}

/// Single-batch Q/K normalization and half-split `RoPE` configuration.
/// Positions index cache rows, while `table_start` indexes supplied cosine/sine tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QkNormRopeConfig {
    pub seq_len: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub position_start: usize,
    pub table_start: usize,
}

/// Optional head-major KV-cache destinations for fused Q/K normalization and RoPE.
#[cfg(feature = "dtype_f16")]
pub struct QkNormRopeCache<'a> {
    pub key: &'a mut dyn DeviceOutput<f16>,
    pub value: &'a mut dyn DeviceOutput<f16>,
    pub capacity: usize,
}

#[cfg(feature = "dtype_f16")]
impl<'a> Debug for QkNormRopeCache<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("QkNormRopeCache").finish_non_exhaustive()
    }
}

/// Enqueues Q/K RMS normalization, RoPE, and optional KV-cache writes.
/// A device position overrides both start positions for graph replay.
#[cfg(feature = "dtype_f16")]
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
pub unsafe fn qk_norm_rope_f16(
    stream: &Stream,
    q_out: &mut impl DeviceOutput<f16>,
    k_out: &mut impl DeviceOutput<f16>,
    query: &impl DeviceInput<f16>,
    key: &impl DeviceInput<f16>,
    value: &impl DeviceInput<f16>,
    q_weight: &impl DeviceInput<f16>,
    k_weight: &impl DeviceInput<f16>,
    cos: &impl DeviceInput<f32>,
    sin: &impl DeviceInput<f32>,
    position: Option<&dyn DeviceInput<u32>>,
    cache: Option<QkNormRopeCache<'_>>,
    config: QkNormRopeConfig,
) -> Result<()> {
    let c = config;
    if c.head_dim == 0 || !c.head_dim.is_multiple_of(2) {
        return Err(Error::InvalidLength);
    }
    let q_len = checked_element_count(checked_element_count(c.seq_len, c.heads)?, c.head_dim)?;
    let kv_len = checked_element_count(checked_element_count(c.seq_len, c.kv_heads)?, c.head_dim)?;
    ensure_len(q_out.len(), q_len)?;
    ensure_len(query.len(), q_len)?;
    ensure_len(k_out.len(), kv_len)?;
    ensure_len(key.len(), kv_len)?;
    ensure_len(q_weight.len(), c.head_dim)?;
    ensure_len(k_weight.len(), c.head_dim)?;
    ensure_len(sin.len(), cos.len())?;
    if cos.len() % (c.head_dim / 2) != 0 {
        return Err(Error::InvalidLength);
    }
    if let Some(position) = position {
        ensure_len(position.len(), 1)?;
    }
    let raw_cache = if let Some(cache) = cache {
        ensure_len(value.len(), kv_len)?;
        let len = checked_element_count(
            checked_element_count(cache.capacity, c.kv_heads)?,
            c.head_dim,
        )?;
        ensure_len(cache.key.len(), len)?;
        ensure_len(cache.value.len(), len)?;
        Some((
            write_only_pointer(cache.key),
            write_only_pointer(cache.value),
            cache.capacity,
        ))
    } else {
        None
    };
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::transformer::qk_norm_rope_f16(
            &stream,
            write_only_pointer(q_out),
            write_only_pointer(k_out),
            input_pointer(query),
            input_pointer(key),
            input_pointer(value),
            input_pointer(q_weight),
            input_pointer(k_weight),
            input_pointer(cos),
            input_pointer(sin),
            cos.len() / (c.head_dim / 2),
            position.map(input_pointer),
            raw_cache,
            c,
        )
    }
}

/// Computes the greedy token without materializing vocabulary logits.
/// Scratch buffers hold one maximum and index per 64 vocabulary entries.
///
/// # Safety
///
/// Retain initialized weights and hidden values and exclusive destinations in
/// the stream's context through completion on every path and throughout replay.
/// Order prior accesses before launch and keep writable ranges disjoint.
#[cfg(all(
    feature = "dtype_f16",
    feature = "dtype_f32",
    feature = "dtype_i32",
    feature = "kernel_reduction"
))]
pub unsafe fn lm_head_argmax_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<i32>,
    block_max: &mut impl DeviceOutput<f32>,
    block_idx: &mut impl DeviceOutput<i32>,
    weights: &impl DeviceInput<f16>,
    hidden: &impl DeviceInput<f16>,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<()> {
    if vocab_size == 0 || hidden_size == 0 {
        return Err(Error::InvalidLength);
    }
    ensure_len(out.len(), 1)?;
    ensure_len(hidden.len(), hidden_size)?;
    ensure_len(
        weights.len(),
        checked_element_count(vocab_size, hidden_size)?,
    )?;
    let blocks = vocab_size.div_ceil(64);
    ensure_len(block_max.len(), blocks)?;
    ensure_len(block_idx.len(), blocks)?;
    let borrowed = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::transformer::lm_head_argmax_f16(
            &borrowed,
            DevicePointer::from_cu_deviceptr(block_max.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(block_idx.as_mut_ptr() as _),
            input_pointer(weights),
            input_pointer(hidden),
            vocab_size,
            hidden_size,
        )
    }?;
    // SAFETY: scratch reads follow their producer on this stream, and the caller
    // retains every range through completion without initialized scratch views.
    unsafe {
        cutile::reduction::reduce_block_argmax_f32(
            &borrowed,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(block_max.as_mut_ptr() as _),
            DevicePointer::from_cu_deviceptr(block_idx.as_mut_ptr() as _),
            blocks,
        )
    }
}
