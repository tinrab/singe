mod attention;
mod batch;
mod standard;

use num_traits::ToPrimitive;

pub use self::{
    attention::{
        paged_attention, paged_mla_attention, rotary_embedding, scaled_dot_product_attention,
    },
    batch::batch_norm,
    standard::{layer_norm, rms_norm},
};
use crate::{
    data_type::DataType,
    error::{Error, Result},
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

pub fn usize_to_f32(name: impl Into<String>, value: usize) -> Result<f32> {
    let name = name.into();
    value.to_f32().ok_or_else(|| Error::out_of_range(name))
}

pub fn validate_parameter(parameter: &Tensor, expected_count: i64) -> Result<()> {
    if parameter.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cpu normalization"));
    }

    let count = to_i64(
        parameter.shape().element_count(),
        "cpu normalization parameter element count",
    )?;
    if count != expected_count {
        return Err(Error::length_mismatch(
            "cpu normalization parameter elements",
            to_usize(
                expected_count,
                "cpu normalization expected parameter elements",
            )?,
            to_usize(count, "cpu normalization actual parameter elements")?,
        ));
    }
    Ok(())
}
