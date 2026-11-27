//! Scaling of vectors to unit length, as `torch.nn.functional.normalize` does, and layer norms over the channels of channels-first maps.

use singe::{
    Result as SingeResult, Tensor,
    nn::{LayerNorm, ModelState},
};

/// Returns `tensor` divided by its L2 norm along `dim`, the norm kept at least `1e-12`.
pub fn unit_vectors(tensor: &Tensor, dim: i64) -> SingeResult<Tensor> {
    let norm = tensor
        .mul(tensor)?
        .sum(&[dim], true)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    tensor.div(&norm)
}

/// Applies `norm` over the channels of `input`, `[B, C, ...]`, as a channels-first layer norm does.
pub fn channel_norm(input: &Tensor, norm: &LayerNorm, state: &ModelState) -> SingeResult<Tensor> {
    let rank = i64::try_from(input.shape().dimensions().len()).unwrap_or(i64::MAX);
    let last = (0..rank)
        .filter(|&axis| axis != 1)
        .chain([1])
        .collect::<Vec<_>>();
    let back = (0..rank)
        .map(|axis| match axis {
            0 => 0,
            1 => rank - 1,
            other => other - 1,
        })
        .collect::<Vec<_>>();
    input.permute(&last)?.apply(norm, state)?.permute(&back)
}
