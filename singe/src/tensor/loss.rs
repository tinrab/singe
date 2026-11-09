use crate::{
    error::{Error, Result},
    tensor::{NormOrder, Tensor},
};

// How a loss combines the losses of its samples, as PyTorch's `reduction` argument does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LossReduction {
    None,
    #[default]
    Mean,
    Sum,
}

// Losses and similarities models compute at inference, composed from reductions, gathers, and element-wise operations.
impl Tensor {
    // The cross entropy of `self`, unnormalized logits with classes along dimension 1 (or 0 for a single sample), against `target`, the class index of each sample.
    // Each sample's loss is `logsumexp(logits) − logits[target]`, as `torch.nn.functional.cross_entropy` computes without weights, ignored indices, or label smoothing.
    pub fn cross_entropy(&self, target: &Self, reduction: LossReduction) -> Result<Self> {
        let classes = i64::from(self.shape().rank() > 1);
        if target.shape().rank() + 1 != self.shape().rank() {
            return Err(Error::invalid_argument(
                "cross entropy target must have one dimension fewer than the logits",
            ));
        }

        let chosen = self
            .gather(classes, &target.unsqueeze(classes)?)?
            .squeeze(classes)?;
        let losses = self.logsumexp(&[classes], false)?.sub(&chosen)?;
        losses.reduced_loss(reduction)
    }

    // The cosine similarity of `self` and `other` along `dim`, after broadcasting them together.
    // Each operand is divided by its norm, kept at least `eps`, before their product is summed, as `torch.nn.functional.cosine_similarity` does.
    pub fn cosine_similarity(&self, other: &Self, dim: i64, eps: f32) -> Result<Self> {
        let normalized = |value: &Self| -> Result<Self> {
            let norm = value
                .vector_norm(NormOrder::Finite(2.0), &[dim], true)?
                .clamp(eps, f32::INFINITY)?;
            value.div(&norm)
        };
        normalized(self)?
            .mul(&normalized(other)?)?
            .sum(&[dim], false)
    }

    fn reduced_loss(&self, reduction: LossReduction) -> Result<Self> {
        match reduction {
            LossReduction::None => Ok(self.clone()),
            LossReduction::Mean => self.mean(&[], false),
            LossReduction::Sum => self.sum(&[], false),
        }
    }
}
