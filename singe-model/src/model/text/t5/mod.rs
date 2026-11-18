// T5: an encoder and a decoder of pre-norm blocks with RMS norms, whose self-attention scores are unscaled and biased by buckets of each pair's relative position, learned once per stack, and whose decoder's output, scaled when tied, reads the token embedding.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Embedding, Layer, Linear, Model, ModelState, RmsNorm, StateBuilder, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Attention, AttentionCache, AttentionSpec, causal_mask},
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{DecoderCache, HeadLoader, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct T5Config {
    pub vocab_size: i64,
    pub width: i64,
    // Each head's width, which `heads` times need not be `width`.
    pub head: i64,
    pub heads: i64,
    pub inner: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub buckets: i64,
    pub max_distance: i64,
    pub eps: f32,
    pub activation: Activation,
    // Whether the feed-forward gates its activation with a second projection, as T5 v1.1 does.
    pub gated: bool,
    // Whether the output reads the token embedding, scaled by `width^-0.5`.
    pub tied: bool,
}

// The bucket of each relative position `key - query`, as `_relative_position_bucket` sorts them: exact below half the buckets, logarithmic up to `max_distance`, and, for an encoder, signed.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "positions and buckets are small, and `transformers` truncates the logarithm the same way"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "`transformers` divides one logarithm by another"
)]
fn bucket(relative: i64, bidirectional: bool, buckets: i64, max_distance: i64) -> i64 {
    let (buckets, mut bucket, distance) = if bidirectional {
        let half = buckets / 2;
        (half, if relative > 0 { half } else { 0 }, relative.abs())
    } else {
        (buckets, 0, (-relative).max(0))
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

// T5's scores go unscaled, its initialization having folded the scale into the queries.
const UNSCALED: AttentionOptions = AttentionOptions {
    scale: Some(1.0),
    causal: false,
    window: None,
    softcap: None,
};

#[derive(Debug, Clone)]
struct FeedForward {
    up: Linear,
    gate: Option<Linear>,
    down: Linear,
    activation: Activation,
}

impl Layer for FeedForward {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.up, state)?
            .apply(&self.activation, state)?;
        if let Some(gate) = &self.gate {
            hidden = hidden.mul(&input.apply(gate, state)?)?;
        }
        hidden.apply(&self.down, state)
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
    // A block over `hidden`, an encoder's whole input, or a decoder's tokens after those `cache` holds the keys and values of, which it adds to `cache`.
    fn forward(
        &self,
        hidden: &Tensor,
        bias: &Tensor,
        mut cache: Option<&mut (Option<AttentionCache>, AttentionCache)>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended = match cache.as_deref_mut() {
            Some((own, _)) => {
                let keys = self.attention.extend(own.as_ref(), &normed, state)?;
                let attended =
                    self.attention
                        .attend_cache(&normed, &keys, Some(bias), UNSCALED, state)?;
                *own = Some(keys);
                attended
            }
            None => {
                self.attention
                    .attend(&normed, &normed, &normed, Some(bias), UNSCALED, state)?
            }
        };
        let mut hidden = hidden.add(&attended)?;
        if let (Some((norm, attention)), Some((_, encoded))) = (&self.cross, cache.as_deref()) {
            let normed = hidden.apply(norm, state)?;
            hidden =
                hidden.add(&attention.attend_cache(&normed, encoded, None, UNSCALED, state)?)?;
        }

        let normed = hidden.apply(&self.feed_forward_norm, state)?;
        hidden.add(&normed.apply(&self.feed_forward, state)?)
    }
}

#[derive(Debug, Clone)]
struct Stack {
    blocks: Vec<Block>,
    // Each relative position bucket's bias for each head, `[buckets, heads]`.
    position_bias: StateKey,
    norm: RmsNorm,
    // Whether queries see every key, as the encoder's do, or only those up to their own, as the decoder's.
    bidirectional: bool,
}

impl Stack {
    // The bias of the attention of `length` queries after `first` earlier tokens to every key up to them, `[1, heads, length, first + length]`, with keys after a decoder's query hidden.
    fn bias(
        &self,
        first: i64,
        length: i64,
        config: &T5Config,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let table = state.tensor(&self.position_bias)?;
        let keys = first + length;
        let buckets = (first..keys)
            .flat_map(|query| {
                (0..keys).map(move |key| {
                    bucket(
                        key - query,
                        self.bidirectional,
                        config.buckets,
                        config.max_distance,
                    )
                })
            })
            .collect::<Vec<_>>();
        let index = Tensor::from_vec(device, Shape::create([length * keys, 1])?, buckets)?
            .expand(Shape::create([length * keys, config.heads])?)?;
        let bias = table
            .gather(0, &index)?
            .reshape([length, keys, config.heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?;
        if self.bidirectional {
            return Ok(bias);
        }
        bias.add(&causal_mask(device, bias.data_type(), first, length)?)
    }

    // Runs an encoder over `hidden`, its whole input.
    fn forward(
        &self,
        hidden: &Tensor,
        config: &T5Config,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = hidden.shape().dimensions()[1];
        let bias = self.bias(0, length, config, hidden.device(), state)?;
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &bias, None, state)?;
        }
        hidden.apply(&self.norm, state)
    }

    // A decoder's state before its first token: each block's cross-attention keys and values over `encoded`.
    fn start(&self, encoded: &Tensor, state: &ModelState) -> SingeResult<DecoderCache> {
        let blocks =
            self.blocks
                .iter()
                .map(|block| {
                    let (_, attention) = block.cross.as_ref().ok_or_else(|| {
                        singe::Error::invalid_argument("an encoder stack decoding")
                    })?;
                    Ok((None, attention.project(encoded, state)?))
                })
                .collect::<SingeResult<Vec<_>>>()?;
        Ok(DecoderCache { blocks, read: 0 })
    }

    // Runs a decoder over `hidden`, the tokens after those `cache` has read, adding them to it.
    fn decode(
        &self,
        hidden: &Tensor,
        cache: &mut DecoderCache,
        config: &T5Config,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let first = i64::try_from(cache.read).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let length = hidden.shape().dimensions()[1];
        let bias = self.bias(first, length, config, hidden.device(), state)?;
        let mut hidden = hidden.clone();
        for (block, caches) in self.blocks.iter().zip(&mut cache.blocks) {
            hidden = block.forward(&hidden, &bias, Some(caches), state)?;
        }

        cache.read += usize::try_from(length).map_err(|_| singe::Error::out_of_range("tokens"))?;
        hidden.apply(&self.norm, state)
    }
}

#[derive(Debug, Clone)]
pub struct T5Layers {
    tokens: Embedding,
    encoder: Stack,
    decoder: Stack,
    head: Linear,
    config: T5Config,
}

impl Seq2SeqLayers for T5Layers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Tokens(batch) = input else {
            return Err(singe::Error::invalid_argument("T5 reads tokens"));
        };
        let hidden = batch.tokens().apply(&self.tokens, state)?;
        self.encoder.forward(&hidden, &self.config, state)
    }

    fn start(&self, encoded: &Tensor, state: &ModelState) -> SingeResult<DecoderCache> {
        self.decoder.start(encoded, state)
    }

    fn next_scores(
        &self,
        cache: &mut DecoderCache,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let new = tokens
            .get(cache.read..)
            .filter(|new| !new.is_empty())
            .ok_or_else(|| singe::Error::invalid_argument("a decoding step without a new token"))?;
        let count = i64::try_from(new.len()).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let new = Tensor::from_vec(state.device()?, Shape::create([1, count])?, new.to_vec())?;
        let hidden = self
            .decoder
            .decode(&new.apply(&self.tokens, state)?, cache, &self.config, state)?
            .narrow(1, count - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?;
        #[expect(clippy::cast_precision_loss, reason = "model widths are small")]
        let hidden = if self.config.tied {
            hidden.mul_scalar((self.config.width as f64).powf(-0.5))?
        } else {
            hidden
        };
        hidden.apply(&self.head, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "t5",
    formats: &[(Format::Transformers, &["t5"])],
    heads: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

fn stack(
    schema: &mut Schema<'_>,
    config: &T5Config,
    layers: i64,
    decoder: bool,
) -> SingeResult<Stack> {
    let (width, inner) = (config.width, config.head * config.heads);
    let attention = |schema: &mut Schema<'_>| {
        Attention::create(
            schema,
            AttentionSpec {
                width,
                context: width,
                inner,
                heads: config.heads,
                bias: false,
            },
        )
    };
    let blocks = (0..layers)
        .map(|index| {
            let mut schema = schema.scope(format!("blocks.{index}"));
            let cross = decoder
                .then(|| -> SingeResult<_> {
                    Ok((
                        schema.rms_norm("cross_norm", width, config.eps)?,
                        attention(&mut schema.scope("cross_attention"))?,
                    ))
                })
                .transpose()?;
            Ok(Block {
                attention_norm: schema.rms_norm("attention_norm", width, config.eps)?,
                attention: attention(&mut schema.scope("attention"))?,
                cross,
                feed_forward_norm: schema.rms_norm("feed_forward_norm", width, config.eps)?,
                feed_forward: FeedForward {
                    up: schema.linear("feed_forward.up", width, config.inner, false)?,
                    gate: config
                        .gated
                        .then(|| schema.linear("feed_forward.gate", width, config.inner, false))
                        .transpose()?,
                    down: schema.linear("feed_forward.down", config.inner, width, false)?,
                    activation: config.activation,
                },
            })
        })
        .collect::<SingeResult<Vec<_>>>()?;
    Ok(Stack {
        blocks,
        position_bias: schema.parameter("position_bias", [config.buckets, config.heads])?,
        norm: schema.rms_norm("norm", width, config.eps)?,
        bidirectional: !decoder,
    })
}

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: T5Config = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let embedding = schema.parameter("tokens.weight", [config.vocab_size, config.width])?;
        let head = if config.tied {
            Linear::new(schema.tie("head.weight", &embedding)?, None)
        } else {
            schema.linear("head", config.width, config.vocab_size, false)?
        };
        T5Layers {
            tokens: Embedding::new(embedding),
            encoder: stack(
                &mut schema.scope("encoder"),
                &config,
                config.encoder_layers,
                false,
            )?,
            decoder: stack(
                &mut schema.scope("decoder"),
                &config,
                config.decoder_layers,
                true,
            )?,
            head,
            config: config.clone(),
        }
    };
    let layers: Box<dyn Seq2SeqLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules(config.tied))?;
    loader.finish()?;
    Ok(Seq2SeqLm::new(model, generation))
}
