//! BART's encoder and decoder stacks, which TAPEX, `MBart`'s decoder in Donut, and `TrOCR`'s decoder share: token embeddings, scaled or not, and learned positions offset by two, normed where the model norms them, then blocks that norm after each residual, or before as `MBart`'s do with a norm after the last.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, AttentionInput, CrossAttention, CrossAttentionInput, DecodeStep, DecodingLayer,
        Embedding, EncoderDecoderCache, GroupedQueryAttention, KvCacheSpec, Layer, LayerNorm,
        Linear, Mlp, ModelState, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, attention_rules, paged_cross_attention, paged_self_attention,
    },
    format::{Rules, join},
    head::Seq2SeqDecoding,
    schema::Schema,
};

/// The positions BART's learned position table skips before the first, as its padding once took them.
const POSITION_OFFSET: i64 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BartStackConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    /// The positions the learned table covers.
    pub max_positions: i64,
    /// What token embeddings are multiplied by.
    pub embedding_scale: f64,
    pub embedding_norm: bool,
    /// Whether blocks norm before each part, with a norm after the last block, as `MBart`'s do, rather than after each residual.
    pub pre_norm: bool,
    /// The width of the encoder states a decoder's cross-attention reads.
    pub cross_width: i64,
    pub eps: f32,
}

#[must_use]
pub fn embedding_scale(scale_embedding: bool, width: i64) -> f64 {
    if scale_embedding {
        (width as f64).sqrt()
    } else {
        1.0
    }
}

/// A decoder block's attention: causal self-attention over its paged cache, then the norm and cross-attention that read the encoder's states.
#[derive(Debug, Clone)]
struct DecoderAttention {
    attention: GroupedQueryAttention,
    cross_norm: LayerNorm,
    cross: CrossAttention,
}

impl DecoderAttention {
    fn create(schema: &mut Schema<'_>, config: &BartStackConfig) -> SingeResult<Self> {
        let square = AttentionSpec::square(config.width, config.heads);
        Ok(Self {
            attention: paged_self_attention(
                &mut schema.scope("attention"),
                square,
                AttentionOptions::default(),
            )?,
            cross_norm: schema.layer_norm("cross_attention_norm", config.width, config.eps)?,
            cross: paged_cross_attention(
                &mut schema.scope("cross_attention"),
                AttentionSpec {
                    context: config.cross_width,
                    ..square
                },
            )?,
        })
    }
}

/// A block whose attention is `A`: an encoder's [`Attention`], or a decoder's [`DecoderAttention`].
#[derive(Debug, Clone)]
struct BartBlock<A> {
    attention_norm: LayerNorm,
    attention: A,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl<A> BartBlock<A> {
    fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            attention: attention(schema)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
        })
    }
}

/// The embeddings, blocks, and final norm of an encoder or decoder stack, whose blocks' attention is `A`.
#[derive(Debug, Clone)]
struct BartStack<A> {
    tokens: Embedding,
    positions: StateKey,
    embedding_norm: Option<LayerNorm>,
    blocks: Vec<BartBlock<A>>,
    norm: Option<LayerNorm>,
    config: BartStackConfig,
}

impl<A> BartStack<A> {
    /// Registers a stack in `schema`: `tokens.weight`, tied to `tokens` when given, `positions.weight`, `embedding_norm` when the config norms embeddings, `blocks.N` with `attention_norm`, what `attention` registers, `feed_forward_norm`, and `feed_forward`, and a pre-norm stack's `norm`.
    fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        tokens: Option<&StateKey>,
        attention: impl Fn(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let width = config.width;
        let tokens = match tokens {
            Some(shared) => schema.tie("tokens.weight", shared)?,
            None => schema.parameter("tokens.weight", [config.vocab_size, width])?,
        };
        let positions = schema.parameter(
            "positions.weight",
            [config.max_positions + POSITION_OFFSET, width],
        )?;
        let embedding_norm = config
            .embedding_norm
            .then(|| schema.layer_norm("embedding_norm", width, config.eps))
            .transpose()?;
        let blocks = (0..config.layers)
            .map(|index| {
                BartBlock::create(
                    &mut schema.scope(format!("blocks.{index}")),
                    config,
                    &attention,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = config
            .pre_norm
            .then(|| schema.layer_norm("norm", width, config.eps))
            .transpose()?;
        Ok(Self {
            tokens: Embedding::new(tokens),
            positions,
            embedding_norm,
            blocks,
            norm,
            config: config.clone(),
        })
    }

    /// Returns each token's embedding, scaled, with its position's added, and normed when the stack norms its embeddings.
    fn embed(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let embedded = tokens
            .apply(&self.tokens, state)?
            .mul_scalar(self.config.embedding_scale)?;
        self.position(&embedded, positions, state)
    }

    /// Returns embeddings with their positions' added, normed when the stack norms its embeddings.
    fn position(
        &self,
        embedded: &Tensor,
        positions: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let table =
            state
                .tensor(&self.positions)?
                .narrow(0, POSITION_OFFSET, self.config.max_positions)?;
        let hidden = embedded.add(&table.index_select(0, positions)?)?;
        match &self.embedding_norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }

    /// Runs one part of a block on `hidden`: its input normed first when blocks norm before each part, its output added to `hidden`, and the sum normed when blocks norm after each residual.
    fn part(
        &self,
        hidden: &Tensor,
        norm: &LayerNorm,
        state: &ModelState,
        part: impl FnOnce(&Tensor) -> SingeResult<Tensor>,
    ) -> SingeResult<Tensor> {
        if self.config.pre_norm {
            hidden.add(&part(&hidden.apply(norm, state)?)?)
        } else {
            hidden.add(&part(hidden)?)?.apply(norm, state)
        }
    }

    /// Runs a block's feed-forward part on `hidden`.
    fn feed_forward(
        &self,
        block: &BartBlock<A>,
        hidden: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.part(hidden, &block.feed_forward_norm, state, |normed| {
            normed.apply(&block.feed_forward, state)
        })
    }

    /// Returns the last block's output, normed when the stack has a norm after it.
    fn finish(&self, hidden: Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

/// A BART-family encoder stack.
#[derive(Debug, Clone)]
pub struct BartEncoder {
    stack: BartStack<Attention>,
}

impl BartEncoder {
    /// Registers an encoder in `schema`, with the names [`bart_stack_rules`] imports to, and its blocks' `attention`; its tokens are tied to `tokens` when given.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        tokens: Option<&StateKey>,
    ) -> SingeResult<Self> {
        let square = AttentionSpec::square(config.width, config.heads);
        Ok(Self {
            stack: BartStack::create(schema, config, tokens, |schema| {
                Attention::create(&mut schema.scope("attention"), square)
            })?,
        })
    }

    /// Runs the encoder over `tokens`, `[1, T]`, returning each token's final state, `[1, T, W]`.
    pub fn forward(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.forward_embedded(&self.embed_tokens(tokens, state)?, state)
    }

    /// Returns `tokens`' embeddings, `[1, T, W]`, scaled, before positions, which a model may splice other embeddings into before [`BartEncoder::forward_embedded`].
    pub fn embed_tokens(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        tokens
            .apply(&self.stack.tokens, state)?
            .mul_scalar(self.stack.config.embedding_scale)
    }

    /// Runs the encoder over embeddings, `[1, T, W]`, as `BartEncoder` reads `inputs_embeds`: positions added, normed, and through the blocks.
    pub fn forward_embedded(&self, embedded: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, length, _] = embedded.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a stack's embeddings"));
        };
        let positions = Tensor::from_vec(
            embedded.device(),
            Shape::create([length])?,
            (0..length).collect::<Vec<_>>(),
        )?;
        let stack = &self.stack;
        let mut hidden = stack.position(embedded, &positions, state)?;
        for block in &stack.blocks {
            hidden = stack.part(&hidden, &block.attention_norm, state, |normed| {
                block.attention.forward(normed, normed, normed, state)
            })?;
            hidden = stack.feed_forward(block, &hidden, state)?;
        }
        stack.finish(hidden, state)
    }
}

/// A BART-family decoder with its output: a decoder stack, the head over its last states, and the bias BART adds to the logits, `[1, V]`.
#[derive(Debug, Clone)]
pub struct BartDecoder {
    stack: BartStack<DecoderAttention>,
    head: Linear,
    logits_bias: Option<StateKey>,
}

impl BartDecoder {
    /// Registers a decoder in `schema`, with the names [`bart_stack_rules`] imports to, and its blocks' `attention`, `cross_attention_norm`, and `cross_attention`; its tokens are tied to `tokens` when given.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        tokens: Option<&StateKey>,
        head: Linear,
        logits_bias: Option<StateKey>,
    ) -> SingeResult<Self> {
        Ok(Self {
            stack: BartStack::create(schema, config, tokens, |schema| {
                DecoderAttention::create(schema, config)
            })?,
            head,
            logits_bias,
        })
    }
}

impl Seq2SeqDecoding for BartDecoder {
    fn cross_attentions(&self) -> Vec<&CrossAttention> {
        self.stack
            .blocks
            .iter()
            .map(|block| &block.attention.cross)
            .collect()
    }

    fn cache_spec(&self) -> KvCacheSpec {
        let config = &self.stack.config;
        KvCacheSpec {
            kv_heads: config.heads,
            head_dim: config.width / config.heads,
        }
    }

    fn forward(
        &self,
        step: &DecodeStep,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let stack = &self.stack;
        if caches.len() != stack.blocks.len() {
            return Err(SingeError::missing_input("a decoder block's cache"));
        }

        let mut hidden = stack.embed(step.tokens(), step.positions(), state)?;
        for (block, cache) in stack.blocks.iter().zip(caches) {
            let DecoderAttention {
                attention,
                cross_norm,
                cross,
            } = &block.attention;
            hidden = stack.part(&hidden, &block.attention_norm, state, |normed| {
                attention.decode(
                    &AttentionInput {
                        hidden: normed,
                        step,
                        angles: None,
                        cache: &cache.self_attention(),
                    },
                    state,
                )
            })?;
            hidden = stack.part(&hidden, cross_norm, state, |normed| {
                cross.forward(
                    &CrossAttentionInput {
                        hidden: normed,
                        step,
                        keys: cache.cross_keys(),
                        values: cache.cross_values(),
                        lengths: cache.source_lengths(),
                    },
                    state,
                )
            })?;
            hidden = stack.feed_forward(block, &hidden, state)?;
        }

        let logits = stack
            .finish(hidden, state)?
            .index_select(0, step.last_rows())?
            .apply(&self.head, state)?;
        match &self.logits_bias {
            Some(bias) => logits.add(state.tensor(bias)?),
            None => Ok(logits),
        }
    }
}

/// Imports a stack stored under `from`, as `transformers`' BART-family models name their parts, to the schema's names under `to`.
#[must_use]
pub fn bart_stack_rules(from: &str, to: &str) -> Rules {
    let (layers, blocks) = (join(from, "layers.*"), join(to, "blocks.*"));
    Rules::new()
        .rename_each(
            from,
            to,
            &[
                ("embed_tokens", "tokens"),
                ("embed_positions", "positions"),
                ("layernorm_embedding", "embedding_norm"),
                ("layer_norm", "norm"),
            ],
        )
        .then(attention_rules(
            &join(&layers, "self_attn"),
            &join(&blocks, "attention"),
        ))
        .then(attention_rules(
            &join(&layers, "encoder_attn"),
            &join(&blocks, "cross_attention"),
        ))
        .rename_each(
            &layers,
            &blocks,
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
}
