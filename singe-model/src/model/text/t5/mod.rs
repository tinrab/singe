//! T5: an encoder and a decoder of pre-norm blocks with RMS norms, whose attention scores are unscaled and whose self-attention scores are biased by buckets of each pair's relative position, learned once per stack.
//! A tied decoder's output reads the token embedding, after scaling the final states by `width^-0.5`.

mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::capture,
    nn::{
        Activation, AttentionInput, CrossAttention, CrossAttentionInput, DecodeStep, Embedding,
        EncoderDecoderCache, GatedMlp, GroupedQueryAttention, KvCacheSpec, Layer, Linear, Mlp,
        ModelState, RmsNorm, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Attention, AttentionSpec, paged_cross_attention, paged_self_attention},
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{Boxed, HeadLoader, Seq2SeqDecoding, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct T5Config {
    pub vocab_size: i64,
    pub width: i64,
    /// Each head's width, which `heads` times need not be `width`.
    pub head_dim: i64,
    pub heads: i64,
    pub inner: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub buckets: i64,
    pub max_distance: i64,
    pub eps: f32,
    pub activation: Activation,
    /// Whether the feed-forward layer gates its activation with a second projection, as T5 v1.1's does.
    pub gated: bool,
    /// Whether the output reads the token embedding, after scaling the decoder's final states by `width^-0.5`.
    pub tied: bool,
}

/// Which of T5's stacks a part belongs to: the encoder, whose position bias looks both ways, or the decoder, whose looks only back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackRole {
    Encoder,
    Decoder,
}

/// Returns the bucket of the relative position `relative`, `key - query`, as `_relative_position_bucket` sorts them.
/// An encoder gives each sign half the buckets, and a decoder gives all of them to keys at or before the query.
/// Within those, distances below half the buckets have a bucket each, and longer ones share buckets logarithmically up to `max_distance`.
fn bucket(relative: i64, role: StackRole, buckets: i64, max_distance: i64) -> i64 {
    let (buckets, mut bucket, distance) = match role {
        StackRole::Encoder => {
            let half = buckets / 2;
            (half, if relative > 0 { half } else { 0 }, relative.abs())
        }
        StackRole::Decoder => (buckets, 0, (-relative).max(0)),
    };
    let exact = buckets / 2;
    if distance < exact {
        bucket += distance;
    } else {
        let large = exact
            + ((distance as f32 / exact as f32).ln() / (max_distance as f32 / exact as f32).ln()
                * (buckets - exact) as f32) as i64;
        bucket += large.min(buckets - 1);
    }
    bucket
}

/// The attention options of T5's unscaled scores, its initialization having folded the scale into the queries.
const UNSCALED: AttentionOptions = AttentionOptions {
    scale: Some(1.0),
    causal: false,
    window: None,
    softcap: None,
};

#[derive(Debug, Clone)]
enum FeedForward {
    Plain(Mlp),
    Gated(GatedMlp),
}

impl Layer for FeedForward {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Plain(mlp) => mlp.forward(input, state),
            Self::Gated(mlp) => mlp.forward(input, state),
        }
    }
}

/// The projections of each attention in a block.
fn attention_spec(config: &T5Config) -> AttentionSpec {
    AttentionSpec {
        width: config.width,
        context: config.width,
        inner: config.head_dim * config.heads,
        heads: config.heads,
        qkv_bias: false,
        output_bias: false,
    }
}

/// A decoder block's attention: causal self-attention over its paged cache, then the norm and cross-attention that read the encoder's states.
#[derive(Debug, Clone)]
struct DecoderAttention {
    attention: GroupedQueryAttention,
    cross_norm: RmsNorm,
    cross: CrossAttention,
}

impl DecoderAttention {
    fn create(schema: &mut Schema<'_>, config: &T5Config) -> SingeResult<Self> {
        let spec = attention_spec(config);
        Ok(Self {
            attention: paged_self_attention(&mut schema.scope("attention"), spec, UNSCALED)?,
            cross_norm: schema.rms_norm("cross_attention_norm", config.width, config.eps)?,
            cross: paged_cross_attention(&mut schema.scope("cross_attention"), spec)?
                .with_scale(1.0),
        })
    }
}

/// A block whose attention is `A`: an encoder's [`Attention`], or a decoder's [`DecoderAttention`].
#[derive(Debug, Clone)]
struct Block<A> {
    attention_norm: RmsNorm,
    attention: A,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl<A> Block<A> {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let sizes = [width, config.inner, width];
        let feed_forward = if config.gated {
            FeedForward::Gated(schema.gated_mlp("feed_forward", sizes, config.activation, false)?)
        } else {
            FeedForward::Plain(schema.mlp("feed_forward", sizes, config.activation, false)?)
        };
        Ok(Self {
            attention_norm: schema.rms_norm("attention_norm", width, eps)?,
            attention: attention(schema)?,
            feed_forward_norm: schema.rms_norm("feed_forward_norm", width, eps)?,
            feed_forward,
        })
    }

    /// Returns `hidden` with the feed-forward layer's output on it, normed, added.
    fn add_feed_forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.feed_forward_norm, state)?;
        hidden.add(&normed.apply(&self.feed_forward, state)?)
    }
}

impl Block<Attention> {
    /// Runs an encoder's block over its whole input, `hidden`, with `bias` added to its attention's scores.
    fn forward(&self, hidden: &Tensor, bias: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended =
            self.attention
                .attend(&normed, &normed, &normed, Some(bias), UNSCALED, state)?;
        self.add_feed_forward(&hidden.add(&attended)?, state)
    }
}

impl Block<DecoderAttention> {
    /// Runs a decoder's block over a decode step's rows, `hidden`, which read and extend `cache`, with `bias`, `[heads, distances]`, added to each self-attention score by its query's distance from its key.
    fn decode(
        &self,
        hidden: &Tensor,
        step: &DecodeStep,
        bias: &Tensor,
        cache: &EncoderDecoderCache,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let DecoderAttention {
            attention,
            cross_norm,
            cross,
        } = &self.attention;
        let attended = attention.decode_biased(
            &AttentionInput {
                hidden: &hidden.apply(&self.attention_norm, state)?,
                step,
                angles: None,
                cache: &cache.self_attention(),
            },
            bias,
            state,
        )?;
        let hidden = hidden.add(&attended)?;
        let crossed = cross.forward(
            &CrossAttentionInput {
                hidden: &hidden.apply(cross_norm, state)?,
                step,
                keys: cache.cross_keys(),
                values: cache.cross_values(),
                lengths: cache.source_lengths(),
            },
            state,
        )?;
        self.add_feed_forward(&hidden.add(&crossed)?, state)
    }
}

/// An encoder or decoder stack, whose blocks' attention is `A`.
#[derive(Debug, Clone)]
struct Stack<A> {
    blocks: Vec<Block<A>>,
    /// Each relative position bucket's bias for each head, `[buckets, heads]`.
    position_bias: StateKey,
    norm: RmsNorm,
}

impl<A> Stack<A> {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        layers: i64,
        attention: impl Fn(&mut Schema<'_>) -> SingeResult<A>,
    ) -> SingeResult<Self> {
        let blocks = (0..layers)
            .map(|index| {
                Block::create(
                    &mut schema.scope(format!("blocks.{index}")),
                    config,
                    &attention,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            blocks,
            position_bias: schema.parameter("position_bias", [config.buckets, config.heads])?,
            norm: schema.rms_norm("norm", config.width, config.eps)?,
        })
    }
}

impl Stack<Attention> {
    /// Returns the bias of an encoder's attention among `length` tokens, `[1, heads, length, length]`.
    fn encoder_bias(
        &self,
        length: i64,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let table = state.tensor(&self.position_bias)?;
        let buckets = (0..length)
            .flat_map(|query| {
                (0..length).map(move |key| {
                    bucket(
                        key - query,
                        StackRole::Encoder,
                        config.buckets,
                        config.max_distance,
                    )
                })
            })
            .collect::<Vec<_>>();
        let index = Tensor::from_vec(device, Shape::create([length * length, 1])?, buckets)?
            .expand(Shape::create([length * length, config.heads])?)?;
        table
            .gather(0, &index)?
            .reshape([length, length, config.heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)
    }

    /// Runs the stack as an encoder over `hidden`, its whole input.
    fn forward(
        &self,
        hidden: &Tensor,
        config: &T5Config,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = sequence_length(hidden)?;
        let bias = self.encoder_bias(length, config, hidden.device(), state)?;
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &bias, state)?;
        }
        hidden.apply(&self.norm, state)
    }
}

impl Stack<DecoderAttention> {
    /// Returns each head's bias by a decoder query's distance from its key, `[heads, max_distance + 1]`.
    /// Every distance from `max_distance` on falls in the last bucket, so longer ones share the last column.
    fn distance_bias(
        &self,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let distances = config.max_distance + 1;
        let buckets = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([distances])?,
                (0..distances)
                    .map(|distance| {
                        bucket(
                            -distance,
                            StackRole::Decoder,
                            config.buckets,
                            config.max_distance,
                        )
                    })
                    .collect(),
            )
        })?;
        state
            .tensor(&self.position_bias)?
            .index_select(0, &buckets)?
            .transpose(0, 1)
    }
}

fn sequence_length(hidden: &Tensor) -> SingeResult<i64> {
    let &[_, length, _] = hidden.shape().dimensions() else {
        return Err(SingeError::invalid_argument("a T5 stack's input"));
    };
    Ok(length)
}

/// T5's decoder and its output, embedding tokens with the embedding the encoder shares.
#[derive(Debug, Clone)]
struct T5Decoder {
    tokens: Embedding,
    stack: Stack<DecoderAttention>,
    head: Linear,
    config: T5Config,
}

impl Seq2SeqDecoding for T5Decoder {
    fn cross_attentions(&self) -> Vec<&CrossAttention> {
        self.stack
            .blocks
            .iter()
            .map(|block| &block.attention.cross)
            .collect()
    }

    fn cache_spec(&self) -> KvCacheSpec {
        KvCacheSpec {
            kv_heads: self.config.heads,
            head_dim: self.config.head_dim,
        }
    }

    fn forward(
        &self,
        step: &DecodeStep,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        if caches.len() != self.stack.blocks.len() {
            return Err(SingeError::missing_input("a decoder block's cache"));
        }

        let bias = self
            .stack
            .distance_bias(&self.config, step.tokens().device(), state)?;
        let mut hidden = step.tokens().apply(&self.tokens, state)?;
        for (block, cache) in self.stack.blocks.iter().zip(caches) {
            hidden = block.decode(&hidden, step, &bias, cache, state)?;
        }

        let hidden = hidden
            .apply(&self.stack.norm, state)?
            .index_select(0, step.last_rows())?;
        let hidden = if self.config.tied {
            hidden.mul_scalar((self.config.width as f64).powf(-0.5))?
        } else {
            hidden
        };
        hidden.apply(&self.head, state)
    }
}

#[derive(Debug, Clone)]
pub struct T5Layers {
    tokens: Embedding,
    encoder: Stack<Attention>,
    decoder: Arc<dyn Seq2SeqDecoding>,
    config: T5Config,
}

impl T5Layers {
    pub fn create(schema: &mut Schema<'_>, config: &T5Config) -> SingeResult<Self> {
        let embedding = schema.parameter("tokens.weight", [config.vocab_size, config.width])?;
        let head = if config.tied {
            Linear::new(schema.tie("head.weight", &embedding)?, None)
        } else {
            schema.linear("head", config.width, config.vocab_size, false)?
        };
        let tokens = Embedding::new(embedding);
        Ok(Self {
            encoder: Stack::create(
                &mut schema.scope("encoder"),
                config,
                config.encoder_layers,
                |schema| Attention::create(&mut schema.scope("attention"), attention_spec(config)),
            )?,
            decoder: Arc::new(T5Decoder {
                tokens: tokens.clone(),
                stack: Stack::create(
                    &mut schema.scope("decoder"),
                    config,
                    config.decoder_layers,
                    |schema| DecoderAttention::create(schema, config),
                )?,
                head,
                config: config.clone(),
            }),
            tokens,
            config: config.clone(),
        })
    }
}

impl Seq2SeqLayers for T5Layers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Tokens(batch) = input else {
            return Err(SingeError::invalid_argument("T5 reads tokens"));
        };
        let hidden = batch.tokens().apply(&self.tokens, state)?;
        self.encoder.forward(&hidden, &self.config, state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "t5",
    formats: &[(Format::Transformers, &["t5"])],
    loaders: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: T5Config = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let model = Schema::model(loader.options().data_type, |schema| {
        T5Layers::create(schema, &config).map(|layers| Boxed::<dyn Seq2SeqLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.tied))?;
    Ok(Seq2SeqLm::new(model, generation))
}
