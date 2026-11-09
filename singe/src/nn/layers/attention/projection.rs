use crate::{
    error::Result,
    nn::{Layer, Linear, ModelState, RmsNorm},
    tensor::Tensor,
};

// A linear layer, a reshape from `[..., W]` to heads `[..., H, D]`, and an optional norm over each head.
#[derive(Debug, Clone)]
pub struct HeadProjection {
    linear: Linear,
    norm: Option<RmsNorm>,
    heads: i64,
    head_dim: i64,
}

impl HeadProjection {
    #[must_use]
    pub const fn new(linear: Linear, norm: Option<RmsNorm>, heads: i64, head_dim: i64) -> Self {
        Self {
            linear,
            norm,
            heads,
            head_dim,
        }
    }

    #[must_use]
    pub const fn heads(&self) -> i64 {
        self.heads
    }

    #[must_use]
    pub const fn head_dim(&self) -> i64 {
        self.head_dim
    }
}

impl Layer for HeadProjection {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let rank = input.shape().dimensions().len();
        let mut shape = (0..rank.saturating_sub(1))
            .map(|axis| input.dimension(axis))
            .collect::<Result<Vec<_>>>()?;
        shape.extend([self.heads.into(), self.head_dim.into()]);
        let projected = input.apply(&self.linear, state)?.reshape_with(&shape)?;
        match &self.norm {
            Some(norm) => norm.forward(&projected, state),
            None => Ok(projected),
        }
    }
}
