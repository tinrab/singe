use crate::{
    error::{Error, Result},
    nn::{
        AttentionInput, DecodingLayer, DenseAttentionInput, HeadProjection, KvCache, Layer, Linear,
        ModelState,
    },
    tensor::{AttentionOptions, Tensor},
};

// The layers of a [`GroupedQueryAttention`], by name, since nothing in their types tells a query projection from a key projection.
#[derive(Debug, Clone)]
pub struct GroupedQueryLayers {
    pub query: HeadProjection,
    pub key: HeadProjection,
    pub value: HeadProjection,
    pub output: Linear,
}

// Attention of `H` query heads over `G` key and value heads, reading and extending a [`KvCache`].
//
// A step's own keys join the cache first, so with a causal mask each query sees the keys up to its own position.
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    layers: GroupedQueryLayers,
    options: AttentionOptions,
}

impl GroupedQueryAttention {
    // Stores `layers`, whose key and value heads must divide the query heads and share their dimension.
    pub fn create(layers: GroupedQueryLayers, options: AttentionOptions) -> Result<Self> {
        let (query, key, value) = (&layers.query, &layers.key, &layers.value);
        if query.heads() <= 0
            || key.heads() <= 0
            || query.heads() % key.heads() != 0
            || key.heads() != value.heads()
            || query.head_dim() != key.head_dim()
            || key.head_dim() != value.head_dim()
        {
            return Err(Error::invalid_argument("grouped query attention heads"));
        }
        Ok(Self { layers, options })
    }
}

// Attends a whole batch at once, keeping nothing: `[B, T, W]` to `[B, T, W]`, with rotation if the angles are given.
impl Layer<DenseAttentionInput<'_>> for GroupedQueryAttention {
    fn forward(&self, input: &DenseAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        let layers = &self.layers;
        let hidden = input.hidden;
        let project = |layer: &HeadProjection, rotate: bool| -> Result<Tensor> {
            let heads = layer.forward(hidden, state)?;
            match input.angles {
                Some(angles) if rotate => angles.rotate(&heads),
                _ => Ok(heads),
            }?
            .permute(&[0, 2, 1, 3])
        };
        let (query, key, value) = (
            project(&layers.query, true)?,
            project(&layers.key, true)?,
            project(&layers.value, false)?,
        );
        // Each sequence's count of keys hides its padded keys from every query and head.
        let attended = input
            .key_lengths
            .map_or_else(
                || query.attend(&key, &value, None, None, self.options),
                |lengths| query.attend_padded(&key, &value, lengths, self.options),
            )?
            .permute(&[0, 2, 1, 3])?
            .reshape_with(&[
                hidden.dimension(0)?,
                hidden.dimension(1)?,
                (layers.query.heads() * layers.query.head_dim()).into(),
            ])?;
        attended.apply(&layers.output, state)
    }
}

impl DecodingLayer for GroupedQueryAttention {
    type Cache = KvCache;

    fn decode(&self, input: &AttentionInput<'_, KvCache>, state: &ModelState) -> Result<Tensor> {
        let layers = &self.layers;
        let hidden = input.hidden;
        let rotate = |heads: Tensor| match input.angles {
            Some(angles) => angles.rotate(&heads),
            None => Ok(heads),
        };
        let query = rotate(hidden.apply(&layers.query, state)?)?;
        let key = rotate(hidden.apply(&layers.key, state)?)?;
        let value = hidden.apply(&layers.value, state)?;
        input.cache.write(input.step, &key, &value)?;
        let (attended, _) = query.paged_attention(
            input.cache.keys(),
            input.cache.values(),
            input.step.paging(),
            self.options,
        )?;
        attended
            .reshape_with(&[
                hidden.dimension(0)?,
                (layers.query.heads() * layers.query.head_dim()).into(),
            ])?
            .apply(&layers.output, state)
    }
}
