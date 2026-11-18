// Multi-head attention whose queries, keys, and values each project their own input, as DETR's, SAM's, BART's, T5's, and Mitra's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Linear, ModelState},
    tensor::AttentionOptions,
};

use crate::schema::Schema;

// The widths of an attention's projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpec {
    // The queries' input and the output's width.
    pub width: i64,
    // The keys' and values' input width.
    pub context: i64,
    // The width the heads split.
    pub inner: i64,
    pub heads: i64,
    pub bias: bool,
}

impl AttentionSpec {
    // Self-attention's widths: keys and values read inputs as wide as the queries', projected to as many features, with biases.
    #[must_use]
    pub const fn square(width: i64, heads: i64) -> Self {
        Self {
            width,
            context: width,
            inner: width,
            heads,
            bias: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl Attention {
    // Registers `query`, from `width` features to `inner`, `key` and `value`, from `context` to `inner`, and `output`, from `inner` back to `width`.
    pub fn create(schema: &mut Schema<'_>, spec: AttentionSpec) -> SingeResult<Self> {
        let AttentionSpec {
            width,
            context,
            inner,
            heads,
            bias,
        } = spec;
        Ok(Self {
            query: schema.linear("query", width, inner, bias)?,
            key: schema.linear("key", context, inner, bias)?,
            value: schema.linear("value", context, inner, bias)?,
            output: schema.linear("output", inner, width, bias)?,
            heads,
        })
    }

    // Queries from `query`, `[B, L, W]`, attending to keys from `key` and values from `value`, `[B, S, C]`, to `[B, L, W]`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.attend(query, key, value, None, AttentionOptions::default(), state)
    }

    // As `forward`, with `mask` added to the scores, `[B, H, L, S]` or broadcast to it, and attention under `options`.
    pub fn attend(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        options: AttentionOptions,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let heads = self.heads;
        let split = |tensor: Tensor| -> SingeResult<Tensor> {
            let &[batch, length, inner] = tensor.shape().dimensions() else {
                return Err(SingeError::invalid_argument("attention input"));
            };
            tensor
                .reshape([batch, length, heads, inner / heads])?
                .transpose(1, 2)
        };
        let attended = split(query.apply(&self.query, state)?)?.attend(
            &split(key.apply(&self.key, state)?)?,
            &split(value.apply(&self.value, state)?)?,
            mask,
            None,
            options,
        )?;
        let &[batch, _, length, head] = attended.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention output"));
        };
        attended
            .transpose(1, 2)?
            .reshape([batch, length, heads * head])?
            .apply(&self.output, state)
    }
}
