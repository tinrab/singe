use crate::{
    error::{Error, Result},
    nn::{
        AttentionInput, DecodingLayer, LatentKvCache, Layer, Linear, ModelState, RmsNorm, StateKey,
    },
    tensor::{AttentionOptions, Tensor},
};

// How a query reaches its heads: directly, or through a low-rank bottleneck.
#[derive(Debug, Clone)]
pub enum LatentQuery {
    Direct(Linear),
    LowRank {
        compress: Linear,
        norm: RmsNorm,
        expand: Linear,
    },
}

impl Layer for LatentQuery {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        match self {
            Self::Direct(projection) => projection.forward(input, state),
            Self::LowRank {
                compress,
                norm,
                expand,
            } => input
                .apply(compress, state)?
                .apply(norm, state)?
                .apply(expand, state),
        }
    }
}

// The layers and sizes of a [`MultiHeadLatentAttention`].
#[derive(Debug, Clone)]
pub struct MultiHeadLatentLayers {
    pub query: LatentQuery,
    // Compresses a token into its cached latent and positional parts.
    pub compress: Linear,
    pub latent_norm: RmsNorm,
    // The weight `[H * (N + V), C]` that would expand a latent into each head's key and value.
    pub expand: StateKey,
    pub output: Linear,
    pub heads: i64,
    pub key_dim: i64,
    pub rotary_dim: i64,
    pub value_dim: i64,
    pub latent_dim: i64,
}

// DeepSeek's multi-head latent attention, which caches a token's compressed latent instead of every head's key and value.
//
// A query multiplied by the key part of the expanding weight scores against latents directly, and the attended latent multiplied by the value part is the attended value.
// This equals expanding each cached latent first, and it is the form FlashInfer runs, since it reads one row per token.
#[derive(Debug, Clone)]
pub struct MultiHeadLatentAttention {
    layers: MultiHeadLatentLayers,
    options: AttentionOptions,
}

impl MultiHeadLatentAttention {
    // Stores `layers`, with `options.scale` set by the caller, since the operation's default takes the latent width rather than the head's.
    pub fn create(layers: MultiHeadLatentLayers, options: AttentionOptions) -> Result<Self> {
        if layers.heads <= 0
            || layers.key_dim <= 0
            || layers.rotary_dim <= 0
            || layers.value_dim <= 0
            || layers.latent_dim <= 0
        {
            return Err(Error::invalid_argument("multi-head latent attention sizes"));
        }

        if options.scale.is_none() {
            return Err(Error::invalid_argument(
                "multi-head latent attention needs an explicit scale",
            ));
        }
        Ok(Self { layers, options })
    }
}

impl DecodingLayer for MultiHeadLatentAttention {
    type Cache = LatentKvCache;

    fn decode(
        &self,
        input: &AttentionInput<'_, LatentKvCache>,
        state: &ModelState,
    ) -> Result<Tensor> {
        let layers = &self.layers;
        let hidden = input.hidden;
        let (heads, key_dim, rotary_dim, value_dim, latent_dim) = (
            layers.heads,
            layers.key_dim,
            layers.rotary_dim,
            layers.value_dim,
            layers.latent_dim,
        );
        let angles = input
            .angles
            .ok_or_else(|| Error::invalid_argument("latent attention needs rotary angles"))?;
        let tokens = hidden.dimension(0)?;

        let query = hidden.apply(&layers.query, state)?.reshape_with(&[
            tokens.clone(),
            heads.into(),
            (key_dim + rotary_dim).into(),
        ])?;
        let query_key = query.narrow(-1, 0, key_dim)?;
        let query_positional = angles.rotate(&query.narrow(-1, key_dim, rotary_dim)?)?;

        let compressed = hidden.apply(&layers.compress, state)?;
        let latent = compressed
            .narrow(-1, 0, latent_dim)?
            .apply(&layers.latent_norm, state)?;
        // One positional key serves every head.
        let positional = angles
            .rotate(
                &compressed
                    .narrow(-1, latent_dim, rotary_dim)?
                    .reshape_with(&[tokens.clone(), 1.into(), rotary_dim.into()])?,
            )?
            .reshape_with(&[tokens.clone(), rotary_dim.into()])?;
        input.cache.write(input.step, &latent, &positional)?;

        let expand =
            state
                .tensor(&layers.expand)?
                .reshape([heads, key_dim + value_dim, latent_dim])?;
        let absorb_key = expand.narrow(1, 0, key_dim)?;
        let absorb_value = expand.narrow(1, key_dim, value_dim)?;
        let absorbed = query_key
            .unsqueeze(2)?
            .matmul(&absorb_key.unsqueeze(0)?)?
            .reshape_with(&[tokens.clone(), heads.into(), latent_dim.into()])?;

        let (attended, _) = absorbed.paged_mla_attention(
            &query_positional,
            input.cache.latent(),
            input.cache.positional(),
            input.step.paging(),
            self.options,
        )?;
        let values = attended
            .unsqueeze(2)?
            .matmul(&absorb_value.transpose(-2, -1)?.unsqueeze(0)?)?
            .reshape_with(&[tokens, (heads * value_dim).into()])?;
        values.apply(&layers.output, state)
    }
}
