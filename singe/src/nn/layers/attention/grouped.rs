use crate::{
    error::{Error, Result},
    graph::DimensionExpr,
    nn::{
        AttentionInput, DecodingLayer, DenseAttentionInput, HeadProjection, KvCache, Layer, Linear,
        ModelState, layers::linear::joined_forward,
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

impl GroupedQueryLayers {
    #[must_use]
    pub const fn siblings(&self) -> [&Linear; 3] {
        [self.query.linear(), self.key.linear(), self.value.linear()]
    }

    // Projects `hidden` to the query heads, the query's gate if it reads one, and the key and value heads, with one matmul when their weights are joined.
    fn project(
        &self,
        hidden: &Tensor,
        state: &ModelState,
    ) -> Result<(Tensor, Option<Tensor>, Tensor, Tensor)> {
        if let Some([query, key, value]) = joined_forward(&self.siblings(), hidden, state)? {
            let (query, gate) = self.query.split_heads(hidden, &query, state)?;
            let (key, _) = self.key.split_heads(hidden, &key, state)?;
            let (value, _) = self.value.split_heads(hidden, &value, state)?;
            return Ok((query, gate, key, value));
        }

        let (query, gate) = self.query.project(hidden, state)?;
        Ok((
            query,
            gate,
            hidden.apply(&self.key, state)?,
            hidden.apply(&self.value, state)?,
        ))
    }
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
    // Masks each query from the keys after it, as a causal text encoder attends.
    #[must_use]
    pub const fn causal(mut self) -> Self {
        self.options.causal = true;
        self
    }

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
        let rotate_heads = |heads: Tensor, rotate: bool| -> Result<Tensor> {
            match input.angles {
                Some(angles) if rotate => angles.rotate(&heads),
                _ => Ok(heads),
            }?
            .permute(&[0, 2, 1, 3])
        };
        let (query, gate, key, value) = layers.project(hidden, state)?;
        let (query, key, value) = (
            rotate_heads(query, true)?,
            rotate_heads(key, true)?,
            rotate_heads(value, false)?,
        );
        // Each sequence's count of keys hides its padded keys from every query and head.
        let attended = input
            .key_lengths
            .map_or_else(
                || query.attend(&key, &value, None, None, self.options),
                |lengths| query.attend_padded(&key, &value, lengths, self.options),
            )?
            .permute(&[0, 2, 1, 3])?;
        let shape = [
            hidden.dimension(0)?,
            hidden.dimension(1)?,
            (layers.query.heads() * layers.query.head_dim()).into(),
        ];
        gated(attended.reshape_with(&shape)?, gate, &shape)?.apply(&layers.output, state)
    }
}

// Multiplies `attended` by the sigmoid of `gate`, each reshaped to `shape`, when the query projection reads a gate.
fn gated(attended: Tensor, gate: Option<Tensor>, shape: &[DimensionExpr]) -> Result<Tensor> {
    match gate {
        Some(gate) => attended.mul(&gate.reshape_with(shape)?.sigmoid()?),
        None => Ok(attended),
    }
}

impl GroupedQueryAttention {
    // Decodes as `DecodingLayer::decode` does, adding `bias`, `[H, distances]`, to each score by its query's distance from its key, as T5's relative positions do.
    pub fn decode_biased(
        &self,
        input: &AttentionInput<'_, KvCache>,
        bias: &Tensor,
        state: &ModelState,
    ) -> Result<Tensor> {
        self.decode_paged(input, Some(bias), state)
    }

    fn decode_paged(
        &self,
        input: &AttentionInput<'_, KvCache>,
        bias: Option<&Tensor>,
        state: &ModelState,
    ) -> Result<Tensor> {
        let layers = &self.layers;
        let hidden = input.hidden;
        let rotate = |heads: Tensor| match input.angles {
            Some(angles) => angles.rotate(&heads),
            None => Ok(heads),
        };
        let (query, gate, key, value) = layers.project(hidden, state)?;
        let query = rotate(query)?;
        let key = rotate(key)?;
        input.cache.write(input.step, &key, &value)?;
        let (attended, _) = query.paged_attention(
            input.cache.keys(),
            input.cache.values(),
            input.step.paging(),
            bias,
            self.options,
        )?;
        let shape = [
            hidden.dimension(0)?,
            (layers.query.heads() * layers.query.head_dim()).into(),
        ];
        gated(attended.reshape_with(&shape)?, gate, &shape)?.apply(&layers.output, state)
    }
}

impl DecodingLayer for GroupedQueryAttention {
    type Cache = KvCache;

    fn decode(&self, input: &AttentionInput<'_, KvCache>, state: &ModelState) -> Result<Tensor> {
        self.decode_paged(input, None, state)
    }
}
