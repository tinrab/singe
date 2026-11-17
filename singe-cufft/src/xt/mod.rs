//! Extended precision, distributed FFT storage and callbacks.

mod memory;
mod plan;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda::data_type::{DataType, DataTypeLike};
use singe_cufft_sys::xt as sys;

pub use self::{
    memory::{DistributedChunk, DistributedMemory},
    plan::{JitCallback, Plan},
};
use crate::{
    error::{Error, Result},
    utility::to_usize,
};

singe_core::native_enum! {
    /// Layout of distributed FFT storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SubFormat: sys::cufftXtSubFormat {
        /// Input in linear order across GPUs, the default.
        Input = CUFFT_XT_FORMAT_INPUT,
        /// Output in an order that depends on the transform, the default.
        Output = CUFFT_XT_FORMAT_OUTPUT,
        /// In-place data in input order, linear across GPUs, the default.
        InPlace = CUFFT_XT_FORMAT_INPLACE,
        /// Shuffled output order after the transform runs.
        InPlaceShuffled = CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
        /// Shuffled input order before a one-dimensional transform runs.
        Input1dShuffled = CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED,
        /// Distributed layout used only by cuFFTMp.
        DistributedInput = CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
        /// Distributed layout used only by cuFFTMp.
        DistributedOutput = CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT,
    }
}

singe_core::native_enum! {
    /// Policy controlling the amount of temporary work area.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum WorkAreaPolicy: sys::cufftXtWorkAreaPolicy {
        /// Maximum reduction in workspace.
        Minimal = CUFFT_WORKAREA_MINIMAL,
        /// Use the supplied per-device sizes as workspace limits.
        User = CUFFT_WORKAREA_USER,
        /// Default policy favoring performance.
        Performance = CUFFT_WORKAREA_PERFORMANCE,
    }
}

singe_core::native_enum! {
    /// Load or store callback element type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CallbackType: sys::cufftXtCallbackType {
        /// Load single-precision complex values.
        LoadComplex = CUFFT_CB_LD_COMPLEX,
        /// Load double-precision complex values.
        LoadComplexDouble = CUFFT_CB_LD_COMPLEX_DOUBLE,
        /// Load single-precision real values.
        LoadReal = CUFFT_CB_LD_REAL,
        /// Load double-precision real values.
        LoadRealDouble = CUFFT_CB_LD_REAL_DOUBLE,
        /// Store single-precision complex values.
        StoreComplex = CUFFT_CB_ST_COMPLEX,
        /// Store double-precision complex values.
        StoreComplexDouble = CUFFT_CB_ST_COMPLEX_DOUBLE,
        /// Store single-precision real values.
        StoreReal = CUFFT_CB_ST_REAL,
        /// Store double-precision real values.
        StoreRealDouble = CUFFT_CB_ST_REAL_DOUBLE,
    }
}

/// Factorization and index mapping of a distributed one-dimensional transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factors {
    /// Transform size.
    pub size: i64,
    /// Number of strings.
    pub string_count: i64,
    /// String length.
    pub string_length: i64,
    /// Substring length.
    pub substring_length: i64,
    /// First transform factor.
    pub factor1: i64,
    /// Second transform factor.
    pub factor2: i64,
    /// String index mask.
    pub string_mask: i64,
    /// Substring index mask.
    pub substring_mask: i64,
    /// First-factor index mask.
    pub factor1_mask: i64,
    /// Second-factor index mask.
    pub factor2_mask: i64,
    /// String index shift.
    pub string_shift: i32,
    /// Substring index shift.
    pub substring_shift: i32,
    /// First-factor index shift.
    pub factor1_shift: i32,
    /// Second-factor index shift.
    pub factor2_shift: i32,
}

/// Batched FFT shape and advanced input/output layouts.
/// All precisions must match; execution type must be complex.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Logical transform extents from outermost to innermost; one to three positive dimensions.
    pub dimensions: Vec<i64>,
    /// Positive number of transforms in the batch.
    pub batch: i64,
    /// Scalar or interleaved-complex representation of the input samples.
    pub input_type: DataType,
    /// Scalar or interleaved-complex representation of the output samples.
    pub output_type: DataType,
    /// Complex representation used for transform computation.
    pub execution_type: DataType,
    /// Physical input extents, one per dimension, or `None` for a contiguous native layout.
    pub input_embedding: Option<Vec<i64>>,
    /// Innermost input sample spacing in input elements; ignored without an input embedding.
    pub input_stride: i64,
    /// Spacing between input batches in input elements; ignored without an input embedding.
    pub input_distance: i64,
    /// Physical output extents, one per dimension, or `None` for a contiguous native layout.
    pub output_embedding: Option<Vec<i64>>,
    /// Innermost output sample spacing in output elements; ignored without an output embedding.
    pub output_stride: i64,
    /// Spacing between output batches in output elements; ignored without an output embedding.
    pub output_distance: i64,
}

impl Layout {
    /// Describes contiguous transforms; cuFFT ignores explicit strides when embeddings are absent.
    #[must_use]
    pub fn contiguous(
        dimensions: Vec<i64>,
        batch: i64,
        input_type: DataType,
        output_type: DataType,
        execution_type: DataType,
    ) -> Self {
        Self {
            dimensions,
            batch,
            input_type,
            output_type,
            execution_type,
            input_embedding: None,
            input_stride: 1,
            input_distance: 0,
            output_embedding: None,
            output_stride: 1,
            output_distance: 0,
        }
    }

    fn required_elements(&self, input: bool) -> Result<usize> {
        if self.dimensions.is_empty()
            || self.dimensions.len() > 3
            || self.batch <= 0
            || self.dimensions.iter().any(|&n| n <= 0)
        {
            return Err(Error::out_of_range("FFT dimensions"));
        }

        let (ty, embedding, stride, distance) = if input {
            (
                self.input_type,
                &self.input_embedding,
                self.input_stride,
                self.input_distance,
            )
        } else {
            (
                self.output_type,
                &self.output_embedding,
                self.output_stride,
                self.output_distance,
            )
        };
        let mut dimensions = self.dimensions.clone();
        if is_complex(ty) && (!is_complex(self.input_type) || !is_complex(self.output_type)) {
            let last = dimensions
                .last_mut()
                .expect("plans validate at least one dimension");
            *last = *last / 2 + 1;
        }

        let elements = if let Some(embedding) = embedding {
            if embedding.len() != dimensions.len() {
                return Err(Error::LengthMismatch {
                    name: "FFT embedding".into(),
                    expected: dimensions.len(),
                    actual: embedding.len(),
                });
            }

            if stride <= 0 || distance <= 0 || embedding.iter().zip(&dimensions).any(|(e, n)| e < n)
            {
                return Err(Error::out_of_range("FFT strides"));
            }

            let mut offset = 0i64;
            for (&extent, &embed) in dimensions.iter().zip(embedding) {
                offset = offset
                    .checked_mul(embed)
                    .and_then(|v| v.checked_add(extent - 1))
                    .ok_or_else(|| Error::out_of_range("FFT span"))?;
            }
            offset
                .checked_mul(stride)
                .and_then(|v| {
                    (self.batch - 1)
                        .checked_mul(distance)
                        .and_then(|b| v.checked_add(b))
                })
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| Error::out_of_range("FFT span"))?
        } else {
            dimensions.iter().try_fold(self.batch, |a, &b| {
                a.checked_mul(b)
                    .ok_or_else(|| Error::out_of_range("FFT span"))
            })?
        };
        to_usize(elements, "FFT elements")
    }
}

fn is_complex(ty: DataType) -> bool {
    matches!(
        ty,
        DataType::ComplexF16 | DataType::ComplexBf16 | DataType::ComplexF32 | DataType::ComplexF64
    )
}

fn check_buffer<T: DataTypeLike>(layout: &Layout, len: usize, input: bool) -> Result<()> {
    // Evaluates the element type's layout assertions at compile time.
    let () = T::LAYOUT;
    let expected_type = if input {
        layout.input_type
    } else {
        layout.output_type
    };
    if T::data_type() != expected_type {
        return Err(Error::DataTypeMismatch {
            name: "FFT buffer".into(),
            expected: expected_type,
            actual: T::data_type(),
        });
    }

    let required = layout.required_elements(input)?;
    if len < required {
        return Err(Error::BufferTooSmall {
            name: "FFT buffer".into(),
            required,
            actual: len,
        });
    }
    Ok(())
}

impl_enum_display!(SubFormat, {
    SubFormat::Input => "CUFFT_XT_FORMAT_INPUT",
    SubFormat::Output => "CUFFT_XT_FORMAT_OUTPUT",
    SubFormat::InPlace => "CUFFT_XT_FORMAT_INPLACE",
    SubFormat::InPlaceShuffled => "CUFFT_XT_FORMAT_INPLACE_SHUFFLED",
    SubFormat::Input1dShuffled => "CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED",
    SubFormat::DistributedInput => "CUFFT_XT_FORMAT_DISTRIBUTED_INPUT",
    SubFormat::DistributedOutput => "CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",
});

impl_enum_display!(WorkAreaPolicy, {
    WorkAreaPolicy::Minimal => "CUFFT_WORKAREA_MINIMAL",
    WorkAreaPolicy::User => "CUFFT_WORKAREA_USER",
    WorkAreaPolicy::Performance => "CUFFT_WORKAREA_PERFORMANCE",
});

impl_enum_display!(CallbackType, {
    CallbackType::LoadComplex => "CUFFT_CB_LD_COMPLEX",
    CallbackType::LoadComplexDouble => "CUFFT_CB_LD_COMPLEX_DOUBLE",
    CallbackType::LoadReal => "CUFFT_CB_LD_REAL",
    CallbackType::LoadRealDouble => "CUFFT_CB_LD_REAL_DOUBLE",
    CallbackType::StoreComplex => "CUFFT_CB_ST_COMPLEX",
    CallbackType::StoreComplexDouble => "CUFFT_CB_ST_COMPLEX_DOUBLE",
    CallbackType::StoreReal => "CUFFT_CB_ST_REAL",
    CallbackType::StoreRealDouble => "CUFFT_CB_ST_REAL_DOUBLE",
});
