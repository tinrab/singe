//! Boolean masks, causal masks, and mask composition.

#[cfg(feature = "cutile")]
use std::sync::Arc;

#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
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
    },
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskReductionConfig {
    pub rows: usize,
    pub columns: usize,
    pub tuning: TuningChoice<MaskReductionTuning>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskReductionTuning {
    pub tile_width: usize,
    pub compile: KernelCompileConfig,
}

impl MaskReductionConfig {
    pub const fn create(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            tuning: TuningChoice::Default,
        }
    }

    pub const fn with_tuning(mut self, tuning: TuningChoice<MaskReductionTuning>) -> Self {
        self.tuning = tuning;
        self
    }

    pub const fn explicit_tuning(self, tuning: MaskReductionTuning) -> Self {
        self.with_tuning(TuningChoice::Explicit(tuning))
    }

    pub const fn cached_tuning(self) -> Self {
        self.with_tuning(TuningChoice::Cached)
    }

    pub const fn tune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Tune(options))
    }

    pub const fn retune(self, options: TuneOptions) -> Self {
        self.with_tuning(TuningChoice::Retune(options))
    }
}

impl MaskReductionTuning {
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

    pub const fn compile(mut self, compile: KernelCompileConfig) -> Self {
        self.compile = compile;
        self
    }
}

/// Writes one for zero input bytes and zero for nonzero input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_not_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
) -> Result<()> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::mask::mask_not_u8(&stream, write_only_pointer(out), input_pointer(input), len)
    }
}

/// Writes the logical conjunction of corresponding nonzero input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_and_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    mask_binary(
        stream,
        out,
        lhs,
        rhs,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_and_u8(arg0, arg1, arg2, arg3, arg4)
        },
    )
}

/// Writes the logical disjunction of corresponding nonzero input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_or_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    mask_binary(
        stream,
        out,
        lhs,
        rhs,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_or_u8(arg0, arg1, arg2, arg3, arg4)
        },
    )
}

/// Writes the logical exclusive-or of corresponding nonzero input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_xor_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    mask_binary(
        stream,
        out,
        lhs,
        rhs,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_xor_u8(arg0, arg1, arg2, arg3, arg4)
        },
    )
}

/// Writes the logical negation of each input byte.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn logical_not_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { mask_not_u8(stream, out, input) }
}

/// Writes the logical conjunction of corresponding input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn logical_and_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { mask_and_u8(stream, out, lhs, rhs) }
}

/// Writes the logical disjunction of corresponding input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn logical_or_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { mask_or_u8(stream, out, lhs, rhs) }
}

/// Writes the logical exclusive-or of corresponding input bytes.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn logical_xor_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { mask_xor_u8(stream, out, lhs, rhs) }
}

/// Writes one output byte per row indicating whether any input byte is nonzero.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_any_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
) -> Result<()> {
    mask_reduction(
        stream,
        out,
        input,
        config,
        "mask_any_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_any_u8(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_any_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Measures row-reduction candidates and leaves the winning result in the output.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn tune_mask_any_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    options: TuneOptions,
) -> Result<TuneReport<MaskReductionTuning>> {
    tune_mask_reduction(
        stream,
        out,
        input,
        config,
        options,
        "mask_any_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_any_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Writes one output byte per row indicating whether every input byte is nonzero.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_all_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
) -> Result<()> {
    mask_reduction(
        stream,
        out,
        input,
        config,
        "mask_all_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_all_u8(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_all_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Measures row-reduction candidates and leaves the winning result in the output.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn tune_mask_all_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    options: TuneOptions,
) -> Result<TuneReport<MaskReductionTuning>> {
    tune_mask_reduction(
        stream,
        out,
        input,
        config,
        options,
        "mask_all_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_all_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Writes one u32 count of nonzero input bytes per row.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn mask_count_nonzero_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u32>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
) -> Result<()> {
    mask_reduction(
        stream,
        out,
        input,
        config,
        "mask_count_nonzero_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::mask_count_nonzero_u8(arg0, arg1, arg2, arg3, arg4)
        },
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_count_nonzero_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Measures row-reduction candidates and leaves the winning result in the output.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn tune_mask_count_nonzero_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u32>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    options: TuneOptions,
) -> Result<TuneReport<MaskReductionTuning>> {
    tune_mask_reduction(
        stream,
        out,
        input,
        config,
        options,
        "mask_count_nonzero_u8",
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5, arg6| unsafe {
            cutile::mask::mask_count_nonzero_u8_tuned(arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )
}

/// Writes a compact rows-by-cols lower-triangular mask at the given diagonal.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn tril_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_mask(
        stream,
        out,
        rows,
        cols,
        diagonal,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::tril_mask_u8(arg0, arg1, arg2, arg3, arg4)
        },
    )
}

/// Writes a compact rows-by-cols upper-triangular mask at the given diagonal.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn triu_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_mask(
        stream,
        out,
        rows,
        cols,
        diagonal,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4| unsafe {
            cutile::mask::triu_mask_u8(arg0, arg1, arg2, arg3, arg4)
        },
    )
}

/// Writes a compact rows-by-cols lower-triangular mask at the given diagonal.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn lower_triangular_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { tril_mask_u8(stream, out, rows, cols, diagonal) }
}

/// Writes a compact rows-by-cols upper-triangular mask at the given diagonal.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn upper_triangular_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { triu_mask_u8(stream, out, rows, cols, diagonal) }
}

/// Writes a compact rows-by-cols mask with ones on and below the diagonal.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn causal_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { cutile::mask::causal_mask_u8(&stream, write_only_pointer(out), rows, cols) }
}

macro_rules! causal_mask_fill_fn {
    ($name:ident, $ty:ty) => {
        /// Copies each batch of square scores, replacing entries above the diagonal with `fill_value`.
        ///
        /// # Safety
        ///
        /// Retain all allocations until completion on every path and throughout captured
        /// replay. Keep the output exclusive and disjoint from inputs, and order earlier
        /// accesses before launch. Inputs must be initialized when read. Output shape and
        /// storage must match the operation's layout on the stream's actual context.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            batch: usize,
            seq_len: usize,
            fill_value: $ty,
        ) -> Result<()> {
            validate_causal_score_mask(out.len(), input.len(), batch, seq_len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::mask::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    batch,
                    seq_len,
                    fill_value,
                )
            }
        }
    };
}

macro_rules! causal_mask_zero_fn {
    ($name:ident, $ty:ty) => {
        /// Copies each batch of square scores, replacing entries above the diagonal with zero.
        ///
        /// # Safety
        ///
        /// Retain all allocations until completion on every path and throughout captured
        /// replay. Keep the output exclusive and disjoint from inputs, and order earlier
        /// accesses before launch. Inputs must be initialized when read. Output shape and
        /// storage must match the operation's layout on the stream's actual context.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            batch: usize,
            seq_len: usize,
        ) -> Result<()> {
            validate_causal_score_mask(out.len(), input.len(), batch, seq_len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::mask::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    batch,
                    seq_len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
causal_mask_fill_fn!(causal_mask_fill_f32, f32);

#[cfg(feature = "dtype_f32")]
causal_mask_zero_fn!(causal_mask_zero_f32, f32);

#[cfg(feature = "dtype_f16")]
causal_mask_fill_fn!(causal_mask_fill_f16, f16);

#[cfg(feature = "dtype_f16")]
causal_mask_zero_fn!(causal_mask_zero_f16, f16);

#[cfg(feature = "dtype_bf16")]
causal_mask_fill_fn!(causal_mask_fill_bf16, bf16);

#[cfg(feature = "dtype_bf16")]
causal_mask_zero_fn!(causal_mask_zero_bf16, bf16);

/// Writes a compact rows-by-cols mask where each column is below its row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::mask::sequence_mask_u8(
            &stream,
            write_only_pointer(out),
            input_pointer(lengths),
            rows,
            cols,
        )
    }
}

/// Writes a lower-triangular mask intersected with each row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn tril_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_sequence_mask(
        stream,
        out,
        lengths,
        rows,
        cols,
        diagonal,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5| unsafe {
            cutile::mask::tril_sequence_mask_u8(arg0, arg1, arg2, arg3, arg4, arg5)
        },
    )
}

/// Writes an upper-triangular mask intersected with each row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn triu_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_sequence_mask(
        stream,
        out,
        lengths,
        rows,
        cols,
        diagonal,
        // SAFETY: the caller upholds this function's safety contract.
        |arg0, arg1, arg2, arg3, arg4, arg5| unsafe {
            cutile::mask::triu_sequence_mask_u8(arg0, arg1, arg2, arg3, arg4, arg5)
        },
    )
}

/// Writes a lower-triangular mask intersected with each row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn lower_triangular_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { tril_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal) }
}

/// Writes an upper-triangular mask intersected with each row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn upper_triangular_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    // SAFETY: the caller establishes the forwarded mask contract.
    unsafe { triu_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal) }
}

/// Writes a compact causal mask intersected with each row length.
///
/// # Safety
///
/// Retain all allocations until completion on every path and throughout captured
/// replay. Keep the output exclusive and disjoint from inputs, and order earlier
/// accesses before launch. Inputs must be initialized when read. Output shape and
/// storage must match the operation's layout on the stream's actual context.
pub unsafe fn causal_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::mask::causal_sequence_mask_u8(
            &stream,
            write_only_pointer(out),
            input_pointer(lengths),
            rows,
            cols,
        )
    }
}

fn validate_causal_score_mask(
    out_len: usize,
    input_len: usize,
    batch: usize,
    seq_len: usize,
) -> Result<()> {
    if seq_len == 0 {
        return Err(Error::InvalidLength);
    }
    let matrix_len = checked_element_count(seq_len, seq_len)?;
    let len = checked_element_count(batch, matrix_len)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)
}

fn mask_binary<F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lhs: &impl DeviceInput<u8>,
    rhs: &impl DeviceInput<u8>,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        DevicePointer<u8>,
        DevicePointer<u8>,
        usize,
    ) -> Result<()>,
{
    let len = out.len();
    ensure_len(lhs.len(), len)?;
    ensure_len(rhs.len(), len)?;
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_score_mask_validation_accepts_exact_lengths() -> Result<()> {
        validate_causal_score_mask(72, 72, 2, 6)
    }

    #[test]
    fn test_causal_score_mask_validation_rejects_short_output() {
        assert!(matches!(
            validate_causal_score_mask(71, 72, 2, 6),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[test]
    fn test_causal_score_mask_validation_rejects_short_input() {
        assert!(matches!(
            validate_causal_score_mask(72, 71, 2, 6),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[test]
    fn test_causal_score_mask_validation_rejects_zero_sequence() {
        assert!(matches!(
            validate_causal_score_mask(0, 0, 1, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn test_mask_reduction_config_builders_preserve_tuning_choice() {
        let tuning = MaskReductionTuning::tile_width(32);
        let config = MaskReductionConfig::create(3, 17);
        assert_eq!(
            config.explicit_tuning(tuning).tuning,
            TuningChoice::Explicit(tuning)
        );
        assert_eq!(config.cached_tuning().tuning, TuningChoice::Cached);
        assert!(matches!(
            config.tune(TuneOptions::default()).tuning,
            TuningChoice::Tune(_)
        ));
        assert!(matches!(
            config.retune(TuneOptions::default()).tuning,
            TuningChoice::Retune(_)
        ));
    }

    #[test]
    fn test_mask_reduction_candidates_include_launchable_widths() {
        let widths = mask_reduction_candidates(17)
            .into_iter()
            .map(|tuning| tuning.tile_width)
            .collect::<Vec<_>>();
        assert_eq!(widths, vec![32, 64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_mask_reduction_validation_accepts_exact_lengths() -> Result<()> {
        validate_mask_reduction_lengths(3, 51, MaskReductionConfig::create(3, 17))
    }

    #[test]
    fn test_mask_reduction_validation_rejects_short_input() {
        assert!(matches!(
            validate_mask_reduction_lengths(3, 50, MaskReductionConfig::create(3, 17)),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
    }

    #[test]
    fn test_mask_reduction_validation_rejects_zero_or_wide_columns() {
        assert!(matches!(
            validate_mask_reduction_lengths(0, 0, MaskReductionConfig::create(3, 0)),
            Err(Error::InvalidLength)
        ));
        assert!(matches!(
            validate_mask_reduction_lengths(3, 3075, MaskReductionConfig::create(3, 1025)),
            Err(Error::UnsupportedWidth { .. })
        ));
    }
}

fn mask_reduction_tune_key(
    stream: &Stream,
    op: &'static str,
    config: MaskReductionConfig,
) -> Result<autotune::TuneKey> {
    let device = DeviceFingerprint::create(stream.context().device())?;
    Ok(autotune::TuneKey::create(
        op,
        "cutile",
        device,
        TuneProblemKey::MaskReduction {
            dtype: "u8".to_owned(),
            op_kind: op.to_owned(),
            rows: config.rows,
            columns: config.columns,
        },
    ))
}

fn mask_reduction_candidates(columns: usize) -> Vec<MaskReductionTuning> {
    [1, 8, 32, 64, 128, 256, 512, 1024]
        .into_iter()
        .map(MaskReductionTuning::tile_width)
        .filter(|tuning| tuning.tile_width >= columns)
        .collect()
}

fn validate_mask_reduction_lengths(
    out_len: usize,
    input_len: usize,
    config: MaskReductionConfig,
) -> Result<()> {
    if config.columns == 0 {
        return Err(Error::InvalidLength);
    }
    if config.columns > 1024 {
        return Err(Error::UnsupportedWidth {
            op: "mask reduction".into(),
            width: config.columns,
        });
    }
    let len = checked_element_count(config.rows, config.columns)?;
    ensure_len(out_len, config.rows)?;
    ensure_len(input_len, len)
}

fn resolve_mask_reduction_tuning<C, F>(
    stream: &Stream,
    op: &'static str,
    key: autotune::TuneKey,
    choice: TuningChoice<MaskReductionTuning>,
    candidates: C,
    tune: F,
) -> Result<Option<MaskReductionTuning>>
where
    C: FnOnce() -> Vec<MaskReductionTuning>,
    F: FnOnce(TuneOptions, Vec<MaskReductionTuning>) -> Result<TuneReport<MaskReductionTuning>>,
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

fn launch_mask_reduction_default<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        usize,
        usize,
    ) -> Result<()>,
{
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(input),
        config.rows,
        config.columns,
    )
}

fn launch_mask_reduction_with_tuning<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    tuning: MaskReductionTuning,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<u8>,
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
        input_pointer(input),
        config.rows,
        config.columns,
        tuning.tile_width,
        tuning.compile,
    )
}

fn mask_reduction<T, FDefault, FTuned>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    op: &'static str,
    launch_default: FDefault,
    launch_tuned: FTuned,
) -> Result<()>
where
    T: DeviceRepr,
    FDefault: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<u8>,
        usize,
        usize,
    ) -> Result<()>,
    FTuned: Copy
        + Fn(
            &Arc<::cutile::cuda_core::Stream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_mask_reduction_lengths(out.len(), input.len(), config)?;
    match config.tuning {
        TuningChoice::Default => {
            return launch_mask_reduction_default(stream, out, input, config, launch_default);
        }
        TuningChoice::Explicit(tuning) => {
            return launch_mask_reduction_with_tuning(
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

    let key = mask_reduction_tune_key(stream, op, config)?;
    let tuning = resolve_mask_reduction_tuning(
        stream,
        op,
        key.clone(),
        config.tuning,
        || mask_reduction_candidates(config.columns),
        |options, candidates| {
            autotune::tune_candidates(stream, op, key, options, candidates, |tuning| {
                launch_mask_reduction_with_tuning(stream, out, input, config, *tuning, launch_tuned)
            })
        },
    )?;
    if let Some(tuning) = tuning {
        launch_mask_reduction_with_tuning(stream, out, input, config, tuning, launch_tuned)
    } else {
        unreachable!("default and explicit mask reduction tuning return before cache resolution")
    }
}

fn tune_mask_reduction<T, F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<T>,
    input: &impl DeviceInput<u8>,
    config: MaskReductionConfig,
    options: TuneOptions,
    op: &'static str,
    launch: F,
) -> Result<TuneReport<MaskReductionTuning>>
where
    T: DeviceRepr,
    F: Copy
        + Fn(
            &Arc<::cutile::cuda_core::Stream>,
            DevicePointer<T>,
            DevicePointer<u8>,
            usize,
            usize,
            usize,
            KernelCompileConfig,
        ) -> Result<()>,
{
    validate_mask_reduction_lengths(out.len(), input.len(), config)?;
    let key = mask_reduction_tune_key(stream, op, config)?;
    let report = autotune::tune_candidates(
        stream,
        op,
        key.clone(),
        options,
        mask_reduction_candidates(config.columns),
        |tuning| launch_mask_reduction_with_tuning(stream, out, input, config, *tuning, launch),
    )?;
    autotune::cache().insert(key, report.best);
    launch_mask_reduction_with_tuning(stream, out, input, config, report.best, launch)?;
    Ok(report)
}

fn triangular_mask<F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        usize,
        usize,
        isize,
    ) -> Result<()>,
{
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    let stream = stream.to_cuda_core()?;
    launch(&stream, write_only_pointer(out), rows, cols, diagonal)
}

fn triangular_sequence_mask<F>(
    stream: &Stream,
    out: &mut impl DeviceOutput<u8>,
    lengths: &impl DeviceInput<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        DevicePointer<u32>,
        usize,
        usize,
        isize,
    ) -> Result<()>,
{
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = stream.to_cuda_core()?;
    launch(
        &stream,
        write_only_pointer(out),
        input_pointer(lengths),
        rows,
        cols,
        diagonal,
    )
}
