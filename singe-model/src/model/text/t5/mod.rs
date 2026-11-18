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
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{HeadLoader, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
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

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl Attention {
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        bias: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let split = |tensor: Tensor| -> SingeResult<Tensor> {
            let dimensions = tensor.shape().dimensions();
            let (batch, length, width) = (dimensions[0], dimensions[1], dimensions[2]);
            tensor
                .reshape([batch, length, self.heads, width / self.heads])?
                .transpose(1, 2)
        };
        let attended = split(hidden.apply(&self.query, state)?)?.attend(
            &split(context.apply(&self.key, state)?)?,
            &split(context.apply(&self.value, state)?)?,
            bias,
            None,
            AttentionOptions {
                scale: Some(1.0),
                ..AttentionOptions::default()
            },
        )?;
        let dimensions = attended.shape().dimensions();
        attended
            .transpose(1, 2)?
            .reshape([dimensions[0], dimensions[2], dimensions[1] * dimensions[3]])?
            .apply(&self.output, state)
    }
}

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
    fn forward(
        &self,
        hidden: &Tensor,
        bias: &Tensor,
        encoded: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.attention_norm, state)?;
        let mut hidden = hidden.add(&self.attention.forward(
            &normed,
            &normed,
            Some(bias),
            state,
        )?)?;
        if let (Some((norm, attention)), Some(encoded)) = (&self.cross, encoded) {
            let normed = hidden.apply(norm, state)?;
            hidden = hidden.add(&attention.forward(&normed, encoded, None, state)?)?;
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
    // The bias of each query's attention to each key, `[1, heads, Q, K]`, with keys after a decoder's query hidden.
    fn bias(
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
                        self.bidirectional,
                        config.buckets,
                        config.max_distance,
                    )
                })
            })
            .collect::<Vec<_>>();
        let index = Tensor::from_vec(device, Shape::create([length * length, 1])?, buckets)?
            .expand(Shape::create([length * length, config.heads])?)?;
        let bias = table
            .gather(0, &index)?
            .reshape([length, length, config.heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?;
        if self.bidirectional {
            return Ok(bias);
        }
        bias.add(&Tensor::attention_mask(
            device,
            bias.data_type(),
            length,
            length,
            true,
            None,
        )?)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoded: Option<&Tensor>,
        config: &T5Config,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = hidden.shape().dimensions()[1];
        let bias = self.bias(length, config, hidden.device(), state)?;
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &bias, encoded, state)?;
        }
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
        self.encoder.forward(&hidden, None, &self.config, state)
    }

    fn next_scores(
        &self,
        encoded: &Tensor,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let count =
            i64::try_from(tokens.len()).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let tokens = Tensor::from_vec(
            encoded.device(),
            Shape::create([1, count])?,
            tokens.to_vec(),
        )?;
        let hidden = self
            .decoder
            .forward(
                &tokens.apply(&self.tokens, state)?,
                Some(encoded),
                &self.config,
                state,
            )?
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
    let attention = |schema: &mut Schema<'_>| -> SingeResult<Attention> {
        Ok(Attention {
            query: schema.linear("query", width, inner, false)?,
            key: schema.linear("key", width, inner, false)?,
            value: schema.linear("value", width, inner, false)?,
            output: schema.linear("output", inner, width, false)?,
            heads: config.heads,
        })
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
