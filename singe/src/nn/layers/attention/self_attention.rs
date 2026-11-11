use crate::{
    error::{Error, Result},
    nn::{
        DenseAttentionInput, GroupedQueryAttention, GroupedQueryLayers, HeadProjection, Layer,
        Linear, ModelState,
    },
    tensor::{AttentionOptions, Tensor},
};

/// Multi-head self attention without a cache or rotation: dense [`GroupedQueryAttention`] with as many key heads as query heads.
/// Providers choose implementations below this layer.
#[derive(Debug, Clone)]
pub struct SelfAttention {
    attention: GroupedQueryAttention,
}

impl SelfAttention {
    /// Stores query, key, value, and output projections for `heads` attention heads over `[batch, sequence, width]` input.
    ///
    /// No causal mask is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if `heads` is not positive or does not divide a nonzero `width`.
    pub fn create(
        query: Linear,
        key: Linear,
        value: Linear,
        output: Linear,
        heads: i64,
        width: i64,
    ) -> Result<Self> {
        if heads <= 0 || width <= 0 || width % heads != 0 {
            return Err(Error::invalid_argument("attention heads and width"));
        }

        let head_dim = width / heads;
        let head = |linear| HeadProjection::new(linear, heads, head_dim);
        Ok(Self {
            attention: GroupedQueryAttention::create(
                GroupedQueryLayers {
                    query: head(query),
                    key: head(key),
                    value: head(value),
                    output,
                },
                AttentionOptions::default(),
            )?,
        })
    }
}

impl Layer for SelfAttention {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        self.forward(
            &DenseAttentionInput {
                hidden: input,
                angles: None,
                key_lengths: None,
            },
            state,
        )
    }
}

impl Layer<DenseAttentionInput<'_>> for SelfAttention {
    fn forward(&self, input: &DenseAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        if input.hidden.shape().dimensions().len() != 3 {
            return Err(Error::invalid_argument(
                "attention requires [batch, sequence, width]",
            ));
        }
        self.attention.forward(input, state)
    }
}
