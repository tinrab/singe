// BART's encoder and decoder stacks, which TAPEX, MBart's decoder in Donut, and TrOCR's decoder share: token embeddings, scaled or not, and learned positions offset by two, normed, then blocks that norm after each residual, or before as MBart's do with a norm after the last.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Embedding, LayerNorm, Linear, ModelState, StateKey},
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Attention, AttentionSpec, causal_mask},
    format::Rules,
    head::DecoderCache,
    schema::Schema,
};

// The positions BART's learned position table skips before the first, as its padding once took them.
const POSITION_OFFSET: i64 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BartStackConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    // The positions the learned table covers.
    pub positions: i64,
    // Whether token embeddings are multiplied by `sqrt(width)`.
    pub scale_embedding: bool,
    pub embedding_norm: bool,
    // Whether blocks norm before each part, with a norm after the last block, as MBart's do, rather than after each residual.
    pub pre_norm: bool,
    // The width of the encoder states a decoder's cross-attention reads.
    pub cross_width: i64,
    pub eps: f32,
}

#[derive(Debug, Clone)]
struct Block {
    attention: Attention,
    attention_norm: LayerNorm,
    cross: Option<(Attention, LayerNorm)>,
    up: Linear,
    down: Linear,
    feed_forward_norm: LayerNorm,
}

// One encoder or decoder stack.
#[derive(Debug, Clone)]
pub struct BartStack {
    tokens: Embedding,
    // What token embeddings are multiplied by.
    embed_scale: f64,
    positions: StateKey,
    embedding_norm: Option<LayerNorm>,
    blocks: Vec<Block>,
    norm: Option<LayerNorm>,
    config: BartStackConfig,
}

impl BartStack {
    // Registers a stack in `schema`: `tokens.weight`, `positions.weight`, `embedding_norm`, `blocks.N` with `attention`, `attention_norm`, a decoder's `cross_attention` and `cross_norm`, `up`, `down`, and `feed_forward_norm`, and a pre-norm stack's `norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        decoder: bool,
        tokens: Option<&StateKey>,
    ) -> SingeResult<Self> {
        let width = config.width;
        #[expect(clippy::cast_precision_loss, reason = "model widths are small")]
        let embed_scale = if config.scale_embedding {
            (width as f64).sqrt()
        } else {
            1.0
        };
        let tokens = match tokens {
            Some(shared) => schema.tie("tokens.weight", shared)?,
            None => schema.parameter("tokens.weight", [config.vocab_size, width])?,
        };
        let positions = schema.parameter(
            "positions.weight",
            [config.positions + POSITION_OFFSET, width],
        )?;
        let embedding_norm = config
            .embedding_norm
            .then(|| schema.layer_norm("embedding_norm", width, config.eps))
            .transpose()?;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                let cross = decoder
                    .then(|| -> SingeResult<_> {
                        Ok((
                            Attention::create(
                                &mut schema.scope("cross_attention"),
                                AttentionSpec {
                                    context: config.cross_width,
                                    ..AttentionSpec::square(width, config.heads)
                                },
                            )?,
                            schema.layer_norm("cross_norm", width, config.eps)?,
                        ))
                    })
                    .transpose()?;
                Ok(Block {
                    attention: Attention::create(
                        &mut schema.scope("attention"),
                        AttentionSpec::square(width, config.heads),
                    )?,
                    attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
                    cross,
                    up: schema.linear("up", width, config.inner, true)?,
                    down: schema.linear("down", config.inner, width, true)?,
                    feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = config
            .pre_norm
            .then(|| schema.layer_norm("norm", width, config.eps))
            .transpose()?;
        Ok(Self {
            tokens: Embedding::new(tokens),
            embed_scale,
            positions,
            embedding_norm,
            blocks,
            norm,
            config: config.clone(),
        })
    }

    // Runs an encoder over `tokens`, `[1, T]`, returning each token's final state, `[1, T, W]`.
    pub fn forward(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.run(tokens, None, state)
    }

    // A decoder's state before its first token: each block's cross-attention keys and values over `encoded`.
    pub fn start(&self, encoded: &Tensor, state: &ModelState) -> SingeResult<DecoderCache> {
        let blocks = self
            .blocks
            .iter()
            .map(|block| {
                let (attention, _) = block
                    .cross
                    .as_ref()
                    .ok_or_else(|| SingeError::invalid_argument("an encoder stack decoding"))?;
                Ok((None, attention.project(encoded, state)?))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(DecoderCache { blocks, read: 0 })
    }

    // Runs a decoder over the tokens of `tokens` past those `cache` has read, adding them to it, and returns the last token's final state, `[W]`.
    pub fn last_state(
        &self,
        cache: &mut DecoderCache,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let device = state.device()?;
        let new = tokens
            .get(cache.read..)
            .filter(|new| !new.is_empty())
            .ok_or_else(|| SingeError::invalid_argument("a decoding step without a new token"))?;
        let count = i64::try_from(new.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let new = Tensor::from_vec(device, Shape::create([1, count])?, new.to_vec())?;
        let hidden = self.run(&new, Some(cache), state)?;
        cache.read = tokens.len();
        hidden.narrow(1, count - 1, 1)?.squeeze(1)?.squeeze(0)
    }

    // Runs the stack over `tokens`, `[1, T]`, an encoder's whole input, or a decoder's tokens after those `cache` has read, which it adds to `cache`.
    fn run(
        &self,
        tokens: &Tensor,
        mut cache: Option<&mut DecoderCache>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = tokens.shape().dimensions()[1];
        let first = cache.as_ref().map_or(Ok(0), |cache| {
            i64::try_from(cache.read).map_err(|_| SingeError::out_of_range("tokens"))
        })?;
        let positions = Tensor::from_vec(
            tokens.device(),
            Shape::create([length])?,
            (first + POSITION_OFFSET..first + length + POSITION_OFFSET).collect::<Vec<_>>(),
        )?;
        let table = state.tensor(&self.positions)?;
        let mut hidden = tokens
            .apply(&self.tokens, state)?
            .mul_scalar(self.embed_scale)?
            .add(&table.index_select(0, &positions)?.unsqueeze(0)?)?;
        if let Some(norm) = &self.embedding_norm {
            hidden = hidden.apply(norm, state)?;
        }

        // A part's input, normed before it when blocks norm first.
        let pre_norm = self.config.pre_norm;
        let input = |hidden: &Tensor, norm: &LayerNorm| {
            if pre_norm {
                hidden.apply(norm, state)
            } else {
                Ok(hidden.clone())
            }
        };
        // A part's output added to its residual, normed after the sum when blocks norm last.
        let residual = |hidden: &Tensor, output: &Tensor, norm: &LayerNorm| {
            let sum = hidden.add(output)?;
            if pre_norm {
                Ok(sum)
            } else {
                sum.apply(norm, state)
            }
        };
        for (index, block) in self.blocks.iter().enumerate() {
            let normed = input(&hidden, &block.attention_norm)?;
            let attended = match cache.as_deref_mut() {
                // A decoder's tokens see the keys up to their own: all of them for one token, under a causal mask for its first tokens, and under one past those read for several later.
                Some(cache) => {
                    let (own, _) = &mut cache.blocks[index];
                    let keys = block.attention.extend(own.as_ref(), &normed, state)?;
                    let mask = (first > 0 && length > 1)
                        .then(|| causal_mask(normed.device(), normed.data_type(), first, length))
                        .transpose()?;
                    let options = AttentionOptions {
                        causal: first == 0,
                        ..AttentionOptions::default()
                    };
                    let attended = block.attention.attend_cache(
                        &normed,
                        &keys,
                        mask.as_ref(),
                        options,
                        state,
                    )?;
                    *own = Some(keys);
                    attended
                }
                None => block.attention.forward(&normed, &normed, &normed, state)?,
            };
            hidden = residual(&hidden, &attended, &block.attention_norm)?;
            if let (Some((attention, norm)), Some(cache)) = (&block.cross, cache.as_deref()) {
                let normed = input(&hidden, norm)?;
                let attended = attention.attend_cache(
                    &normed,
                    &cache.blocks[index].1,
                    None,
                    AttentionOptions::default(),
                    state,
                )?;
                hidden = residual(&hidden, &attended, norm)?;
            }

            let normed = input(&hidden, &block.feed_forward_norm)?;
            let output = normed
                .apply(&block.up, state)?
                .apply(&self.config.activation, state)?
                .apply(&block.down, state)?;
            hidden = residual(&hidden, &output, &block.feed_forward_norm)?;
        }
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

// Imports a stack stored under `from`, as `transformers`' BART-family models name their parts, to the schema's names under `to`.
#[must_use]
pub fn stack_rules(from: &str, to: &str) -> Rules {
    let source = |name: &str| format!("{from}{name}");
    let target = |name: &str| format!("{to}{name}");
    let layer = |name: &str| format!("{from}layers.*.{name}");
    let block = |name: &str| format!("{to}blocks.*.{name}");
    let mut rules = Rules::new()
        .rename(&source("embed_tokens"), &target("tokens"))
        .rename(&source("embed_positions"), &target("positions"))
        .rename(&source("layernorm_embedding"), &target("embedding_norm"))
        .rename(&source("layer_norm"), &target("norm"));
    for (from, to) in [
        ("self_attn", "attention"),
        ("encoder_attn", "cross_attention"),
    ] {
        for (projection, name) in [
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            ("out_proj", "output"),
        ] {
            rules = rules.rename(
                &layer(&format!("{from}.{projection}")),
                &block(&format!("{to}.{name}")),
            );
        }
    }
    rules
        .rename(&layer("self_attn_layer_norm"), &block("attention_norm"))
        .rename(&layer("encoder_attn_layer_norm"), &block("cross_norm"))
        .rename(&layer("fc1"), &block("up"))
        .rename(&layer("fc2"), &block("down"))
        .rename(&layer("final_layer_norm"), &block("feed_forward_norm"))
}
