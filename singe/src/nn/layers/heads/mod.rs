// What turns a model's hidden states into a task's output, such as a sentence's embedding.

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::Result,
    nn::{Layer, ModelState},
    shape::Shape,
    tensor::Tensor,
};

// What a pooling head reads: hidden states `[B, T, W]`, and `padding`, `[B, T]` of `Bool`, true where a token is real.
// Real tokens come first in each row, as `TokenBatch` pads.
#[derive(Debug, Clone, Copy)]
pub struct PoolingInput<'a> {
    pub hidden: &'a Tensor,
    pub padding: &'a Tensor,
}

// How each sequence's hidden states become one vector, as `sentence-transformers`' pooling modes name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingMode {
    // The first token's state.
    Cls,
    // The mean of the real tokens' states.
    Mean,
    // Each feature's largest value over the real tokens.
    Max,
    // The sum of the real tokens' states over the square root of their count.
    MeanSqrtLength,
    // The last real token's state.
    LastToken,
}

// Each sequence's vector, `[B, W]`, pooled by `mode` and scaled to unit length if asked, as `sentence-transformers` pools and normalizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pooling {
    pub mode: PoolingMode,
    pub normalize: bool,
}

impl Pooling {
    #[must_use]
    pub const fn new(mode: PoolingMode, normalize: bool) -> Self {
        Self { mode, normalize }
    }
}

// What a norm is kept from falling below before dividing by it, as `torch.nn.functional.normalize`'s default.
const NORM_FLOOR: f32 = 1e-12;

// What a padded token's features become before the maximum, as `sentence-transformers` fills them.
const PADDED_FEATURE: f64 = -1e9;

impl Layer<PoolingInput<'_>> for Pooling {
    fn forward(&self, input: &PoolingInput<'_>, _state: &ModelState) -> Result<Tensor> {
        let hidden = input.hidden;
        let weights = input.padding.cast(hidden.data_type())?.unsqueeze(-1)?;
        let pooled = match self.mode {
            PoolingMode::Cls => hidden.select(1, 0)?,
            PoolingMode::Mean | PoolingMode::MeanSqrtLength => {
                let total = hidden.mul(&weights)?.sum(&[1], false)?;
                // A sequence of no real tokens would divide by zero, so it counts at least one.
                let count = weights.sum(&[1], false)?.clamp(1.0, f32::MAX)?;
                if self.mode == PoolingMode::Mean {
                    total.div(&count)?
                } else {
                    total.div(&count.sqrt()?)?
                }
            }
            PoolingMode::Max => {
                // Padded tokens hold a value no real feature reaches, so the maximum takes real tokens alone.
                // It runs in `F32`, since that value overflows `F16` and would make real tokens' features NaN.
                let weights = weights.cast(DataType::F32)?;
                let padded = weights
                    .mul_scalar(-PADDED_FEATURE)?
                    .add_scalar(PADDED_FEATURE)?;
                hidden
                    .cast(DataType::F32)?
                    .mul(&weights)?
                    .add(&padded)?
                    .max_reduce(&[1], false)?
                    .cast(hidden.data_type())?
            }
            PoolingMode::LastToken => {
                let dimensions = hidden.shape().dimensions();
                let (rows, width) = (dimensions[0], dimensions[2]);
                let last = input
                    .padding
                    .cast(DataType::I64)?
                    .sum(&[1], false)?
                    .add_scalar(-1)?
                    .clamp(0.0, f32::MAX)?
                    .cast(DataType::I64)?
                    .reshape([rows, 1, 1])?
                    .expand(Shape::create([rows, 1, width])?)?;
                hidden.gather(1, &last)?.squeeze(1)?
            }
        };
        if !self.normalize {
            return Ok(pooled);
        }

        let norm = pooled
            .cast(DataType::F32)?
            .mul(&pooled.cast(DataType::F32)?)?
            .sum(&[-1], true)?
            .sqrt()?
            .clamp(NORM_FLOOR, f32::MAX)?
            .cast(pooled.data_type())?;
        pooled.div(&norm)
    }
}
