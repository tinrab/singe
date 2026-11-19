// Multi-head attention whose queries, keys, and values each project their own input, as DETR's, SAM's, BART's, T5's, and Mitra's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Linear, ModelState},
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Seq2SeqCache, StepRows},
    format::Rules,
    schema::Schema,
};

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
    pub qkv_bias: bool,
    pub output_bias: bool,
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
            qkv_bias: true,
            output_bias: true,
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
            qkv_bias,
            output_bias,
        } = spec;
        let attention = Self {
            query: schema.linear("query", width, inner, qkv_bias)?,
            key: schema.linear("key", context, inner, qkv_bias)?,
            value: schema.linear("value", context, inner, qkv_bias)?,
            output: schema.linear("output", inner, width, output_bias)?,
            heads,
        };
        schema.join_rows(&[&attention.query, &attention.key, &attention.value])?;
        Ok(attention)
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
        let heads =
            |input: &Tensor, linear: &Linear| split_heads(&input.apply(linear, state)?, self.heads);
        let attended = heads(query, &self.query)?.attend(
            &heads(key, &self.key)?,
            &heads(value, &self.value)?,
            mask,
            None,
            options,
        )?;
        merge_heads(&attended)?.apply(&self.output, state)
    }

    // The keys and values cross-attention reads from an encoder's states, `[F, C]`, `[F, H, D]` each.
    pub fn project_sources(
        &self,
        encoded: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        Ok((
            self.row_heads(encoded, &self.key, state)?,
            self.row_heads(encoded, &self.value, state)?,
        ))
    }

    // Self-attention of a decode step's rows, `[T, W]`: writes their keys and values to `cache`, then attends over their sequences' keys with `mask` added to the scores, `[T, H, 1, C]` or broadcast to it.
    pub fn decode(
        &self,
        hidden: &Tensor,
        rows: &StepRows,
        cache: &Seq2SeqCache,
        mask: &Tensor,
        options: AttentionOptions,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        cache.write(
            rows,
            &self.row_heads(hidden, &self.key, state)?,
            &self.row_heads(hidden, &self.value, state)?,
        )?;
        let (keys, values) = cache.read(rows)?;
        let attended = self
            .row_heads(hidden, &self.query, state)?
            .unsqueeze(2)?
            .attend(&keys, &values, Some(mask), None, options)?;
        self.merge_rows(&attended, state)
    }

    // Cross-attention of a decode step's rows, `[T, W]`, over their sequences' encoder states in `cache`.
    pub fn attend_sources(
        &self,
        hidden: &Tensor,
        rows: &StepRows,
        cache: &Seq2SeqCache,
        options: AttentionOptions,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let (keys, values, lengths) = cache.read_sources(rows)?;
        let attended = self
            .row_heads(hidden, &self.query, state)?
            .unsqueeze(2)?
            .attend_padded(&keys, &values, &lengths, options)?;
        self.merge_rows(&attended, state)
    }

    // `input`'s rows, `[N, C]`, through `linear`, split into heads, `[N, H, D]`.
    fn row_heads(
        &self,
        input: &Tensor,
        linear: &Linear,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let projected = input.apply(linear, state)?;
        let &[rows, inner] = projected.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention rows"));
        };
        projected.reshape([rows, self.heads, inner / self.heads])
    }

    // Each row's heads, `[T, H, 1, D]`, joined and through the output projection, `[T, W]`.
    fn merge_rows(&self, attended: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[rows, heads, _, dim] = attended.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attended rows"));
        };
        attended
            .reshape([rows, heads * dim])?
            .apply(&self.output, state)
    }
}

// A projection's `[B, L, inner]` split into `heads` heads, `[B, H, L, inner / H]`.
pub fn split_heads(tensor: &Tensor, heads: i64) -> SingeResult<Tensor> {
    let &[batch, length, inner] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("attention heads' input"));
    };
    tensor
        .reshape([batch, length, heads, inner / heads])?
        .transpose(1, 2)
}

pub fn merge_heads(tensor: &Tensor) -> SingeResult<Tensor> {
    let &[batch, heads, length, head] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("attention heads' output"));
    };
    tensor
        .transpose(1, 2)?
        .reshape([batch, length, heads * head])
}

#[must_use]
pub fn attention_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            ("out_proj", "output"),
        ],
    )
}
