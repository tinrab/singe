use crate::{
    error::Result,
    nn::{
        Activation, AttentionInput, CrossAttention, CrossAttentionInput, DecodingLayer,
        DenseAttentionInput, EncoderDecoderCache, GroupedQueryAttention, Layer, LayerNorm,
        LayerScale, Linear, Mlp, ModelState, RmsNorm, SelfAttention,
    },
    tensor::Tensor,
};

// A block that norms before attention and before feed-forward, with a residual around each.
#[derive(Debug, Clone)]
pub struct PreNormBlock<A, F, N = RmsNorm> {
    attention_norm: N,
    attention: A,
    feed_forward_norm: N,
    feed_forward: F,
    output_norms: Option<(N, N)>,
    scales: Option<(LayerScale, LayerScale)>,
}

impl<A, F, N> PreNormBlock<A, F, N> {
    #[must_use]
    pub const fn new(
        attention_norm: N,
        attention: A,
        feed_forward_norm: N,
        feed_forward: F,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            feed_forward_norm,
            feed_forward,
            output_norms: None,
            scales: None,
        }
    }

    // Also norms what attention and feed-forward each return, before their residuals, as Gemma 2 does.
    #[must_use]
    pub fn with_output_norms(mut self, attention: N, feed_forward: N) -> Self {
        self.output_norms = Some((attention, feed_forward));
        self
    }

    // Also scales what attention and feed-forward each return, after any output norm and before their residuals, as DINOv2 does.
    #[must_use]
    pub fn with_layer_scales(mut self, attention: LayerScale, feed_forward: LayerScale) -> Self {
        self.scales = Some((attention, feed_forward));
        self
    }
}

impl<A, F: Layer, N: Layer> PreNormBlock<A, F, N> {
    fn run(
        &self,
        hidden: &Tensor,
        state: &ModelState,
        attend: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let output_norm = |index: usize, value: Tensor| {
            let value = match &self.output_norms {
                Some(norms) => {
                    (if index == 0 { &norms.0 } else { &norms.1 }).forward(&value, state)?
                }
                None => value,
            };
            match &self.scales {
                Some(scales) => {
                    (if index == 0 { &scales.0 } else { &scales.1 }).forward(&value, state)
                }
                None => Ok(value),
            }
        };
        let attended = attend(&hidden.apply(&self.attention_norm, state)?)?;
        let hidden = hidden.add(&output_norm(0, attended)?)?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?;
        hidden.add(&output_norm(1, fed)?)
    }
}

impl<A, F, N> DecodingLayer for PreNormBlock<A, F, N>
where
    A: DecodingLayer,
    F: Layer + 'static,
    N: Layer + 'static,
{
    type Cache = A::Cache;

    fn decode(&self, input: &AttentionInput<'_, A::Cache>, state: &ModelState) -> Result<Tensor> {
        self.run(input.hidden, state, |normed| {
            self.attention.decode(
                &AttentionInput {
                    hidden: normed,
                    ..*input
                },
                state,
            )
        })
    }
}

impl<A, F, N> Layer<DenseAttentionInput<'_>> for PreNormBlock<A, F, N>
where
    A: for<'a> Layer<DenseAttentionInput<'a>>,
    F: Layer,
    N: Layer,
{
    fn forward(&self, input: &DenseAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        self.run(input.hidden, state, |normed| {
            self.attention.forward(
                &DenseAttentionInput {
                    hidden: normed,
                    ..*input
                },
                state,
            )
        })
    }
}

// An encoder block that norms after each residual: attention, then a feed-forward of two projections around an activation, as BERT's layers are.
// It has no dropout or RNG.
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    attention: SelfAttention,
    norm1: LayerNorm,
    expand: Linear,
    contract: Linear,
    norm2: LayerNorm,
    activation: Activation,
}

impl EncoderBlock {
    // Composes attention, two post-residual layer norms, and the feed-forward's projections, with tanh-approximate GELU until `with_activation` names another.
    #[must_use]
    pub const fn new(
        attention: SelfAttention,
        norm1: LayerNorm,
        expand: Linear,
        contract: Linear,
        norm2: LayerNorm,
    ) -> Self {
        Self {
            attention,
            norm1,
            expand,
            contract,
            norm2,
            activation: Activation::Gelu,
        }
    }

    #[must_use]
    pub const fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl Layer for EncoderBlock {
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

impl Layer<DenseAttentionInput<'_>> for EncoderBlock {
    fn forward(&self, input: &DenseAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        let hidden = input
            .hidden
            .add(&self.attention.forward(input, state)?)?
            .apply(&self.norm1, state)?;
        let feedforward = hidden
            .apply(&self.expand, state)?
            .apply(&self.activation, state)?
            .apply(&self.contract, state)?;
        hidden.add(&feedforward)?.apply(&self.norm2, state)
    }
}

// A decoder block of an encoder-decoder that norms before each part, with a residual around each: causal self-attention over its cache, cross-attention to the encoder's outputs, then a feed-forward, as Whisper's and BART's decoders are.
#[derive(Debug, Clone)]
pub struct EncoderDecoderBlock {
    self_norm: LayerNorm,
    self_attention: GroupedQueryAttention,
    cross_norm: LayerNorm,
    cross_attention: CrossAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

// The parts of an [`EncoderDecoderBlock`], in the order they run.
#[derive(Debug, Clone)]
pub struct EncoderDecoderParts {
    pub self_norm: LayerNorm,
    pub self_attention: GroupedQueryAttention,
    pub cross_norm: LayerNorm,
    pub cross_attention: CrossAttention,
    pub feed_forward_norm: LayerNorm,
    pub feed_forward: Mlp,
}

impl EncoderDecoderBlock {
    #[must_use]
    pub fn new(parts: EncoderDecoderParts) -> Self {
        Self {
            self_norm: parts.self_norm,
            self_attention: parts.self_attention,
            cross_norm: parts.cross_norm,
            cross_attention: parts.cross_attention,
            feed_forward_norm: parts.feed_forward_norm,
            feed_forward: parts.feed_forward,
        }
    }

    // Writes one sequence's cross-attention keys and values to `cache`, from its encoder's outputs, `[F, W]`.
    pub fn write_sources(
        &self,
        encoded: &Tensor,
        sequence: usize,
        cache: &EncoderDecoderCache,
        state: &ModelState,
    ) -> Result<()> {
        let (keys, values) = self.cross_attention.sources(encoded, state)?;
        cache.write_cross(sequence, &keys, &values)
    }
}

impl DecodingLayer for EncoderDecoderBlock {
    type Cache = EncoderDecoderCache;

    fn decode(
        &self,
        input: &AttentionInput<'_, EncoderDecoderCache>,
        state: &ModelState,
    ) -> Result<Tensor> {
        let cache = input.cache.self_attention();
        let attended = self.self_attention.decode(
            &AttentionInput {
                hidden: &input.hidden.apply(&self.self_norm, state)?,
                step: input.step,
                angles: input.angles,
                cache: &cache,
            },
            state,
        )?;
        let hidden = input.hidden.add(&attended)?;
        let crossed = self.cross_attention.forward(
            &CrossAttentionInput {
                hidden: &hidden.apply(&self.cross_norm, state)?,
                step: input.step,
                keys: input.cache.cross_keys(),
                values: input.cache.cross_values(),
            },
            state,
        )?;
        let hidden = hidden.add(&crossed)?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?;
        hidden.add(&fed)
    }
}
