//! Multi-head attention whose queries, keys, and values each project their own input, as DETR's, SAM's, BART's, T5's, and Mitra's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
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

    #[must_use]
    pub const fn query(&self) -> &Linear {
        &self.query
    }

    #[must_use]
    pub const fn key(&self) -> &Linear {
        &self.key
    }

    #[must_use]
    pub const fn value(&self) -> &Linear {
        &self.value
    }

    #[must_use]
    pub const fn output(&self) -> &Linear {
        &self.output
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
    attention_rules_with_output(from, to, "out_proj")
}

/// Imports the attention `transformers` stores under `from` as [`attention_rules`] does, its output projection named `output`, such as `o_proj`.
#[must_use]
pub fn attention_rules_with_output(from: &str, to: &str, output: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            (output, "output"),
        ],
    )
}

/// Returns `x`, `[..., D]`, with each pair of adjacent features swapped and the first negated, `(-x₁, x₀)`: the second term of a rotary embedding that turns adjacent pairs.
pub fn swapped_pairs(x: &Tensor) -> SingeResult<Tensor> {
    let dimensions = x.shape().dimensions();
    let (&width, outer) = dimensions
        .split_last()
        .ok_or_else(|| SingeError::invalid_argument("rotated features"))?;
    let paired = x.reshape([outer, &[width / 2, 2]].concat())?;
    Tensor::stack(&[paired.select(-1, 1)?.neg()?, paired.select(-1, 0)?], -1)?.reshape(dimensions)
}

/// Axial rotary tables over a square grid of points, row by row, as SAM 2's memory attention and SAM 3's vision transformer turn their heads: the first half of a head's feature pairs turn by a point's column and the second half by its row, each pair at its own frequency.
#[derive(Debug, Clone)]
pub struct AxialRotary {
    cos: Tensor,
    sin: Tensor,
}

impl AxialRotary {
    /// Builds the tables for a `side × side` grid and heads of `head` features, each position multiplied by `scale` in `F32` first, in `like`'s type on its device.
    pub fn create(
        side: i64,
        head: i64,
        theta: f32,
        scale: f32,
        like: &Tensor,
    ) -> SingeResult<Self> {
        let quarter = head / 4;
        let frequencies = (0..quarter)
            .map(|pair| 1.0 / theta.powf((2 * pair) as f32 / (head / 2) as f32))
            .collect::<Vec<_>>();
        let mut angles = Vec::with_capacity((side * side * head) as usize);
        for row in 0..side {
            for column in 0..side {
                for feature in 0..head {
                    let pair = feature / 2;
                    let (position, frequency) = if pair < quarter {
                        (column, frequencies[pair as usize])
                    } else {
                        (row, frequencies[(pair - quarter) as usize])
                    };
                    angles.push(position as f32 * scale * frequency);
                }
            }
        }

        let shape = Shape::create([side * side, head])?;
        let table = |values: Vec<f32>| {
            Tensor::from_vec(like.device(), shape.clone(), values)?.cast(like.data_type())
        };
        Ok(Self {
            cos: table(angles.iter().map(|angle| angle.cos()).collect())?,
            sin: table(angles.iter().map(|angle| angle.sin()).collect())?,
        })
    }

    /// Returns the grid's points, the tables' first axis.
    #[must_use]
    pub fn points(&self) -> i64 {
        self.cos.shape().dimensions()[0]
    }

    /// Returns `x`, `[..., N, d]`, turned by the tables.
    pub fn rotated(&self, x: &Tensor) -> SingeResult<Tensor> {
        x.mul(&self.cos)?.add(&swapped_pairs(x)?.mul(&self.sin)?)
    }
}
