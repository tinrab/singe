// Multi-head attention whose queries, keys, and values each project their own input, as DETR's, SAM's, BART's, T5's, and Mitra's are, with `query`, `key`, `value`, and `output` projections.

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
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
        let context = AttentionCache {
            keys: self.heads(&key.apply(&self.key, state)?)?,
            values: self.heads(&value.apply(&self.value, state)?)?,
        };
        self.attend_cache(query, &context, mask, options, state)
    }

    // The keys and values `context`, `[B, S, C]`, projects to, which attention over it reads every step.
    pub fn project(&self, context: &Tensor, state: &ModelState) -> SingeResult<AttentionCache> {
        Ok(AttentionCache {
            keys: self.heads(&context.apply(&self.key, state)?)?,
            values: self.heads(&context.apply(&self.value, state)?)?,
        })
    }

    // `cache`, or nothing before the first step, followed by the keys and values of `input`'s tokens, `[B, N, C]`.
    pub fn extend(
        &self,
        cache: Option<&AttentionCache>,
        input: &Tensor,
        state: &ModelState,
    ) -> SingeResult<AttentionCache> {
        let added = self.project(input, state)?;
        let Some(cache) = cache else {
            return Ok(added);
        };
        Ok(AttentionCache {
            keys: Tensor::concat(&[cache.keys.clone(), added.keys], 2)?,
            values: Tensor::concat(&[cache.values.clone(), added.values], 2)?,
        })
    }

    // Queries from `query`, `[B, L, W]`, attending to the keys and values `cache` holds, to `[B, L, W]`.
    pub fn attend_cache(
        &self,
        query: &Tensor,
        cache: &AttentionCache,
        mask: Option<&Tensor>,
        options: AttentionOptions,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let attended = self.heads(&query.apply(&self.query, state)?)?.attend(
            &cache.keys,
            &cache.values,
            mask,
            None,
            options,
        )?;
        let &[batch, heads, length, head] = attended.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention output"));
        };
        attended
            .transpose(1, 2)?
            .reshape([batch, length, heads * head])?
            .apply(&self.output, state)
    }

    // A projection's `[B, L, inner]` split into heads, `[B, H, L, inner / H]`.
    fn heads(&self, tensor: &Tensor) -> SingeResult<Tensor> {
        let &[batch, length, inner] = tensor.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention input"));
        };
        tensor
            .reshape([batch, length, self.heads, inner / self.heads])?
            .transpose(1, 2)
    }
}

// The keys and values an attention reads, `[B, H, S, D]` each: a decoder's own, which each step extends, or those of the encoder's states, which stay.
#[derive(Debug, Clone)]
pub struct AttentionCache {
    keys: Tensor,
    values: Tensor,
}

impl AttentionCache {
    // The tokens whose keys and values it holds.
    #[must_use]
    pub fn length(&self) -> i64 {
        self.keys.shape().dimensions()[2]
    }
}

// The additive mask of `queries` queries after `first` earlier tokens over all `first + queries` keys, `[queries, first + queries]`: negative infinity where a key comes after its query, and zero elsewhere.
pub fn causal_mask(
    device: Device,
    data_type: DataType,
    first: i64,
    queries: i64,
) -> SingeResult<Tensor> {
    let keys = first + queries;
    let values = (0..queries)
        .flat_map(|query| {
            (0..keys).map(move |key| {
                if key > first + query {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create([queries, keys])?, values)?.cast(data_type)
}
