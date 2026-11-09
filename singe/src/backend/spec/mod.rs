mod elementwise;
mod linear_algebra;
mod neural;
mod quantization;
mod reduction;
mod shape;

use std::borrow::Cow;

pub use self::reduction::empty_reduction_fill;
#[cfg(feature = "cuda")]
pub use self::{elementwise::binary_broadcast_problem, linear_algebra::ScaledMatmulProblem};
pub use self::{
    elementwise::{BinaryBroadcastProblem, clamp_problem, lerp_problem, where_elementwise_problem},
    linear_algebra::{
        MatmulScaling, contraction_problem, grouped_matmul_shape, matmul_problem,
        scaled_matmul_problem, takes_scaled_matmul,
    },
    neural::{
        ConvolutionWindow, attention_mask_shape, attention_problem, batch_norm_problem,
        check_paged_values, conv2d_problem, fuses_attention, normalization_problem,
        paged_attention_problem, paged_mla_attention_problem, pool2d_problem, repeat_kv_problem,
        rotary_embedding_problem, validate_conv2d_bias, validate_normalization_parameters,
    },
    quantization::{
        QuantizationGranularity, QuantizationProblem, quantization_problem, unpacked_shape,
        unpacked_value,
    },
    reduction::{ReductionProblem, reduction_problem, selects_topk, softmax_problem, topk_problem},
    shape::{
        IndexingProblem, MultiIndexProblem, concat_problem, embedding_problem, expand_problem,
        flatten_problem, gather_problem, index_select_problem, multi_index_problem, pad_problem,
    },
};
#[cfg(any(test, feature = "onnx"))]
pub use self::{
    linear_algebra::{gemm_output_shape, matmul_output_shape},
    neural::{conv2d_output_shape, pool2d_output_shape},
    shape::concat_output_shape,
};
use crate::{
    data_type::DataType,
    error::{Error, Result},
    utility::to_usize,
};

pub fn dimension_product_usize(
    name: impl Into<Cow<'static, str>>,
    dimensions: &[i64],
    allow_zero: bool,
) -> Result<usize> {
    let name = name.into();
    dimensions.iter().try_fold(1_usize, |acc, &dimension| {
        if !allow_zero && dimension <= 0 {
            return Err(Error::invalid_argument(name.clone()));
        }

        let dimension = to_usize(dimension, "backend spec dimension")?;
        acc.checked_mul(dimension)
            .ok_or_else(|| Error::out_of_range(name.clone()))
    })
}

pub fn validate_rank(
    diagnostic_prefix: &'static str,
    name: &str,
    dimensions: &[i64],
    expected_rank: usize,
    layout: &str,
) -> Result<()> {
    if dimensions.len() == expected_rank {
        return Ok(());
    }

    Err(Error::InvalidTensorRank {
        operation: diagnostic_prefix.to_string(),
        name: name.to_string().into(),
        expected_rank,
        actual_rank: dimensions.len(),
        layout: layout.to_string(),
        shape: dimensions.to_vec(),
    })
}

pub fn validate_positive(name: &'static str, value: i64) -> Result<()> {
    if value > 0 {
        return Ok(());
    }
    Err(Error::invalid_argument(name))
}

pub fn validate_non_negative(name: &'static str, value: i64) -> Result<()> {
    if value >= 0 {
        return Ok(());
    }
    Err(Error::out_of_range(name))
}

// Whether cuFFT transforms a real signal of `length` points of `data_type`: `F32` and `F64` signals long enough that the basis matmul's quadratic cost shows; shorter ones run as one matmul with the basis on tensor cores.
#[must_use]
pub fn transforms_with_cufft(length: i64, data_type: DataType) -> bool {
    length >= 512 && matches!(data_type, DataType::F32 | DataType::F64)
}

#[cfg(test)]
mod tests;
