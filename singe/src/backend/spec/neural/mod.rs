mod attention;
mod conv;
mod normalization;
mod pooling;

pub use self::{
    attention::{
        attention_mask_shape, attention_problem, check_paged_values, fuses_attention,
        paged_attention_problem, paged_mla_attention_problem, repeat_kv_problem,
        rotary_embedding_problem,
    },
    conv::{ConvolutionWindow, conv2d_problem, validate_conv2d_bias},
    normalization::{batch_norm_problem, normalization_problem, validate_normalization_parameters},
    pooling::pool2d_problem,
};
#[cfg(any(test, feature = "onnx"))]
pub use self::{conv::conv2d_output_shape, pooling::pool2d_output_shape};
use crate::{
    backend::spec::validate_rank,
    error::{Error, Result},
};

pub fn ensure_non_negative_dimensions(
    diagnostic_prefix: &'static str,
    dimensions: &[(&str, i64)],
) -> Result<()> {
    for &(name, dimension) in dimensions {
        if dimension < 0 {
            return Err(Error::invalid_argument(format!(
                "{diagnostic_prefix} {name}"
            )));
        }
    }
    Ok(())
}

pub fn ensure_positive_dimensions(
    diagnostic_prefix: &'static str,
    dimensions: &[(&str, i64)],
) -> Result<()> {
    for &(name, dimension) in dimensions {
        if dimension <= 0 {
            return Err(Error::invalid_argument(format!(
                "{diagnostic_prefix} {name}"
            )));
        }
    }
    Ok(())
}

pub fn fixed_attention_rank(
    diagnostic_prefix: &'static str,
    name: &str,
    dimensions: &[i64],
) -> Result<[i64; 4]> {
    fixed_rank_dimensions(diagnostic_prefix, name, dimensions, 4, "BHQD")
}

pub fn fixed_rank_dimensions<const RANK: usize>(
    diagnostic_prefix: &'static str,
    name: &str,
    dimensions: &[i64],
    expected_rank: usize,
    layout: &str,
) -> Result<[i64; RANK]> {
    validate_rank(diagnostic_prefix, name, dimensions, expected_rank, layout)?;
    dimensions.try_into().map_err(|_| Error::InvalidTensorRank {
        operation: diagnostic_prefix.to_string(),
        name: name.to_string().into(),
        expected_rank,
        actual_rank: dimensions.len(),
        layout: layout.to_string(),
        shape: dimensions.to_vec(),
    })
}
