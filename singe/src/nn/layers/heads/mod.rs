// What turns a model's hidden states into a task's output, such as a sentence's embedding.

use crate::{
    data_type::DataType,
    error::Result,
    nn::{Layer, ModelState},
    tensor::Tensor,
};

// What a pooling head reads: hidden states `[B, T, W]`, and `padding`, `[B, T]` of `Bool`, true where a token is real.
#[derive(Debug, Clone, Copy)]
pub struct PoolingInput<'a> {
    pub hidden: &'a Tensor,
    pub padding: &'a Tensor,
}

// The mean of each sequence's real tokens, `[B, W]`, scaled to unit length if asked, as `sentence-transformers` pools and normalizes.
#[derive(Debug, Clone, Copy)]
pub struct MeanPooling {
    normalize: bool,
}

impl MeanPooling {
    #[must_use]
    pub const fn new(normalize: bool) -> Self {
        Self { normalize }
    }
}

// What a norm is kept from falling below before dividing by it, as `torch.nn.functional.normalize`'s default.
const NORM_FLOOR: f32 = 1e-12;

impl Layer<PoolingInput<'_>> for MeanPooling {
    fn forward(&self, input: &PoolingInput<'_>, _state: &ModelState) -> Result<Tensor> {
        let hidden = input.hidden;
        let weights = input.padding.cast(hidden.data_type())?.unsqueeze(-1)?;
        let total = hidden.mul(&weights)?.sum(&[1], false)?;
        // A sequence of no real tokens would divide by zero, so it counts at least one.
        let count = weights.sum(&[1], false)?.clamp(1.0, f32::MAX)?;
        let mean = total.div(&count)?;
        if !self.normalize {
            return Ok(mean);
        }

        let norm = mean
            .cast(DataType::F32)?
            .mul(&mean.cast(DataType::F32)?)?
            .sum(&[-1], true)?
            .sqrt()?
            .clamp(NORM_FLOOR, f32::MAX)?
            .cast(mean.data_type())?;
        mean.div(&norm)
    }
}
