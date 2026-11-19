// Scaling of vectors to unit length, as `torch.nn.functional.normalize` does.

use singe::{Result as SingeResult, Tensor};

// `tensor` divided by its L2 norm along `dim`, kept at least `1e-12`.
pub fn unit_vectors(tensor: &Tensor, dim: i64) -> SingeResult<Tensor> {
    let norm = tensor
        .mul(tensor)?
        .sum(&[dim], true)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    tensor.div(&norm)
}
