use crate::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::DataType,
    error::{Error, Result},
    frontend::graph::Graph,
    utility::to_usize,
};

/// Standard `ALiBi` slopes in the head order used by the C++ cuDNN frontend.
/// The returned FP32 values populate a `[1, heads, 1, 1]` slope tensor for
/// [`SdpaScoreModifiers::with_alibi_slopes`](crate::frontend::operation::SdpaScoreModifiers::with_alibi_slopes).
/// Non-power-of-two head counts append odd half-steps after the largest power-of-two prefix.
///
/// # Errors
///
/// Returns an error if the head count is nonpositive or cannot fit `usize`.
#[expect(
    clippy::cast_precision_loss,
    reason = "ALiBi slopes are computed in the f32 tensor precision"
)]
pub fn alibi_slopes(heads: i64) -> Result<Vec<f32>> {
    let heads = to_usize(heads, "alibi head count")?;
    if heads == 0 {
        return Err(Error::out_of_range("alibi head count"));
    }
    let prefix = 1_usize << heads.ilog2();
    Ok((0..heads)
        .map(|head| {
            let position = if head < prefix {
                (head + 1) as f32
            } else {
                (2 * (head - prefix) + 1) as f32 * 0.5
            };
            let exponent = position * -8.0 / prefix as f32;
            exponent.exp2()
        })
        .collect())
}

impl Graph {
    pub(crate) fn standard_alibi_slopes(&mut self, heads: i64) -> Result<TensorId> {
        let spec = TensorSpec::new(DataType::F32, Shape::contiguous([1, heads, 1, 1])?);
        let mut bytes = Vec::with_capacity(spec.storage_byte_len(None)?);
        for slope in alibi_slopes(heads)? {
            bytes.extend_from_slice(&slope.to_ne_bytes());
        }
        Ok(self.tensor(spec.with_workspace_data(bytes)?.with_name("alibi slopes")))
    }
}
