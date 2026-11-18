// Multi-head attention whose queries, keys, and values each project their own input, as DETR's and SAM's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Linear, ModelState},
};

use crate::schema::Schema;

#[derive(Debug, Clone)]
pub struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl Attention {
    // Registers `query`, `key`, and `value`, from `width` features to `inner`, and `output`, from `inner` back to `width`.
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        inner: i64,
        heads: i64,
    ) -> SingeResult<Self> {
        Ok(Self {
            query: schema.linear("query", width, inner, true)?,
            key: schema.linear("key", width, inner, true)?,
            value: schema.linear("value", width, inner, true)?,
            output: schema.linear("output", inner, width, true)?,
            heads,
        })
    }

    // Queries from `query`, `[B, L, W]`, attending to keys from `key` and values from `value`, `[B, S, W]`, to `[B, L, W]`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
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
        let attended = split(query.apply(&self.query, state)?)?.scaled_dot_product_attention(
            &split(key.apply(&self.key, state)?)?,
            &split(value.apply(&self.value, state)?)?,
            None,
            false,
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
