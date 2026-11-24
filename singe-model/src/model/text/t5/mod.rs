//! T5: an encoder and a decoder of pre-norm blocks with RMS norms, whose attention scores are unscaled and whose self-attention scores are biased by buckets of each pair's relative position, learned once per stack.
//! A tied decoder's output reads the token embedding, after scaling the final states by `width^-0.5`.

mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, DecodeStep, Embedding, GatedMlp, Layer, Linear, Mlp, ModelState, RmsNorm,
        StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, Seq2SeqCache, Seq2SeqCacheSpec, Seq2SeqTables, StackRole,
        StepRows,
    },
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

#[derive(Debug, Clone)]
struct Block {
    attention_norm: RmsNorm,
    attention: Attention,
    cross: Option<(RmsNorm, Attention)>,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl Block {
    fn create(schema: &mut Schema<'_>, config: &T5Config, role: StackRole) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let spec = AttentionSpec {
            width,
            context: width,
            inner: config.head_dim * config.heads,
            heads: config.heads,
            qkv_bias: false,
            output_bias: false,
        };
        let cross = match role {
            StackRole::Encoder => None,
            StackRole::Decoder => Some((
                schema.rms_norm("cross_attention_norm", width, eps)?,
                Attention::create(&mut schema.scope("cross_attention"), spec)?,
            )),
        };
        let sizes = [width, config.inner, width];
        let feed_forward = if config.gated {
            FeedForward::Gated(schema.gated_mlp("feed_forward", sizes, config.activation, false)?)
        } else {
            FeedForward::Plain(schema.mlp("feed_forward", sizes, config.activation, false)?)
        };
        Ok(Self {
            attention_norm: schema.rms_norm("attention_norm", width, eps)?,
            attention: Attention::create(&mut schema.scope("attention"), spec)?,
            cross,
            feed_forward_norm: schema.rms_norm("feed_forward_norm", width, eps)?,
            feed_forward,
        })
    }

    /// Runs the block over `hidden`, an encoder's whole input, or a decode step's rows, which read and extend the block's cache, with `bias` added to its self-attention's scores.
    fn forward(
        &self,
        hidden: &Tensor,
        bias: &Tensor,
        cached: Option<(&StepRows, &Seq2SeqCache)>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended = match cached {
            Some((rows, cache)) => self
                .attention
                .decode(&normed, rows, cache, bias, UNSCALED, state)?,
            None => {
                self.attention
                    .attend(&normed, &normed, &normed, Some(bias), UNSCALED, state)?
            }
        };
        let mut hidden = hidden.add(&attended)?;
        if let (Some((norm, attention)), Some((rows, cache))) = (&self.cross, cached) {
            let normed = hidden.apply(norm, state)?;
            hidden =
                hidden.add(&attention.attend_sources(&normed, rows, cache, UNSCALED, state)?)?;
        }

        let normed = hidden.apply(&self.feed_forward_norm, state)?;
        hidden.add(&normed.apply(&self.feed_forward, state)?)
    }
}

#[derive(Debug, Clone)]
struct Stack {
    blocks: Vec<Block>,
    /// Each relative position bucket's bias for each head, `[buckets, heads]`.
    position_bias: StateKey,
    norm: RmsNorm,
}

impl Stack {
    fn create(
        schema: &mut Schema<'_>,
        config: &T5Config,
        layers: i64,
        role: StackRole,
    ) -> SingeResult<Self> {
        let blocks = (0..layers)
            .map(|index| Block::create(&mut schema.scope(format!("blocks.{index}")), config, role))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            blocks,
            position_bias: schema.parameter("position_bias", [config.buckets, config.heads])?,
            norm: schema.rms_norm("norm", config.width, config.eps)?,
        })
    }

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

    /// Returns the bias of a decode step's rows' attention to every position of their sequences, `[T, heads, 1, C]`, with each row's mask added, which hides the positions after the row's own.
    fn step_bias(
        &self,
        rows: &StepRows,
        tables: &Seq2SeqTables,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let buckets = tables.buckets()?.index_select(0, rows.positions())?;
        let &[count, capacity] = buckets.shape().dimensions() else {
            return Err(SingeError::invalid_argument("relative position buckets"));
        };
        let table = state.tensor(&self.position_bias)?;
        let &[_, heads] = table.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a T5 position bias"));
        };
        table
            .index_select(0, &buckets.reshape([count * capacity])?)?
            .reshape([count, capacity, heads])?
            .permute(&[0, 2, 1])?
            .unsqueeze(2)?
            .add(rows.mask())
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
            hidden = block.forward(&hidden, &bias, None, state)?;
        }
        hidden.apply(&self.norm, state)
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
    stack: Stack,
    head: Linear,
    config: T5Config,
}

impl Seq2SeqDecoding for T5Decoder {
    fn cross_attentions(&self) -> Vec<&Attention> {
        self.stack
            .blocks
            .iter()
            .filter_map(|block| block.cross.as_ref().map(|(_, attention)| attention))
            .collect()
    }

    fn cache_spec(&self, sources: i64) -> Seq2SeqCacheSpec {
        Seq2SeqCacheSpec {
            heads: self.config.heads,
            head_dim: self.config.head_dim,
            sources,
        }
    }

    fn tables(
        &self,
        data_type: DataType,
        device: Device,
        capacity: i64,
    ) -> SingeResult<Seq2SeqTables> {
        let (buckets, max_distance) = (self.config.buckets, self.config.max_distance);
        Seq2SeqTables::create(
            data_type,
            device,
            capacity,
            Some(&|relative| bucket(relative, StackRole::Decoder, buckets, max_distance)),
        )
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Seq2SeqTables,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let rows = StepRows::create(step, tables)?;
        let bias = self.stack.step_bias(&rows, tables, state)?;
        let mut hidden = step.tokens().apply(&self.tokens, state)?;
        for (block, cache) in self.stack.blocks.iter().zip(caches) {
            hidden = block.forward(&hidden, &bias, Some((&rows, cache)), state)?;
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
    encoder: Stack,
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
                StackRole::Encoder,
            )?,
            decoder: Arc::new(T5Decoder {
                tokens: tokens.clone(),
                stack: Stack::create(
                    &mut schema.scope("decoder"),
                    config,
                    config.decoder_layers,
                    StackRole::Decoder,
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
