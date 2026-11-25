//! Multi-head attention whose queries, keys, and values each project their own input, as DETR's, SAM's, BART's, T5's, and Mitra's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    graph::DimensionExpr,
    nn::{
        CrossAttention, GroupedQueryAttention, GroupedQueryLayers, HeadProjection, Linear,
        ModelState,
    },
    tensor::AttentionOptions,
};

use crate::{format::Rules, schema::Schema};

/// The widths of an attention's projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpec {
    /// The queries' input and the output's width.
    pub width: i64,
    /// The keys' and values' input width.
    pub context: i64,
    /// The width the heads split.
    pub inner: i64,
    pub heads: i64,
    pub qkv_bias: bool,
    pub output_bias: bool,
}

impl AttentionSpec {
    /// Returns self-attention's widths: keys and values read inputs as wide as the queries', projected to as many features, with biases.
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
    /// Registers `query`, from `width` features to `inner`, `key` and `value`, from `context` to `inner`, and `output`, from `inner` back to `width`.
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

    /// Attends with queries from `query`, `[B, L, W]`, to keys from `key` and values from `value`, `[B, S, C]`, and returns `[B, L, W]`.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.attend(query, key, value, None, AttentionOptions::default(), state)
    }

    /// Attends as [`Attention::forward`] does, with `mask` added to the scores, `[B, H, L, S]` or broadcast to it, and attention under `options`.
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
}

/// Registers the projections [`Attention::create`] does, as a decoder's causal self-attention over a paged cache, attending under `options`.
pub fn paged_self_attention(
    schema: &mut Schema<'_>,
    spec: AttentionSpec,
    options: AttentionOptions,
) -> SingeResult<GroupedQueryAttention> {
    let Attention {
        query,
        key,
        value,
        output,
        heads,
    } = Attention::create(schema, spec)?;
    let head = |linear| HeadProjection::new(linear, heads, spec.inner / heads);
    GroupedQueryAttention::create(
        GroupedQueryLayers {
            query: head(query),
            key: head(key),
            value: head(value),
            output,
        },
        AttentionOptions {
            causal: true,
            ..options
        },
    )
}

/// Registers the projections [`Attention::create`] does, as cross-attention over a decoder cache's pages of encoder states.
pub fn paged_cross_attention(
    schema: &mut Schema<'_>,
    spec: AttentionSpec,
) -> SingeResult<CrossAttention> {
    let Attention {
        query,
        key,
        value,
        output,
        heads,
    } = Attention::create(schema, spec)?;
    CrossAttention::create(query, key, value, output, heads, spec.inner)
}

/// Returns a projection's `[B, L, inner]` split into `heads` heads, `[B, H, L, inner / H]`.
pub fn split_heads(tensor: &Tensor, heads: i64) -> SingeResult<Tensor> {
    let &[_, _, inner] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("attention heads' input"));
    };
    tensor
        .reshape_with(&[
            tensor.dimension(0)?,
            tensor.dimension(1)?,
            DimensionExpr::constant(heads),
            DimensionExpr::constant(inner / heads),
        ])?
        .transpose(1, 2)
}

pub fn merge_heads(tensor: &Tensor) -> SingeResult<Tensor> {
    let &[_, heads, _, head] = tensor.shape().dimensions() else {
        return Err(SingeError::invalid_argument("attention heads' output"));
    };
    let joined = tensor.transpose(1, 2)?;
    joined.reshape_with(&[
        joined.dimension(0)?,
        joined.dimension(1)?,
        DimensionExpr::constant(heads * head),
    ])
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
