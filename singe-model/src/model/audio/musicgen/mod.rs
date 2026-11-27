//! `MusicGen`: T5's encoder reads the text, projected to the decoder's width; a pre-norm decoder reads each frame as the sum of one embedding per codebook and gives each codebook's logits from a head of its own; `EnCodec` decodes the codes it chose.
//! A frame's codes reach the decoder packed into one token, which it unpacks by shifts and masks, so the cached decoder that runs other encoder-decoders runs it unchanged.

mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::capture,
    nn::{
        Activation, AttentionInput, CrossAttention, CrossAttentionInput, DecodeStep, DecodingLayer,
        Embedding, EncoderDecoderCache, GroupedQueryAttention, KvCacheSpec, Layer, LayerNorm,
        Linear, Mlp, ModelState, StateKey, TokenBatch,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{AttentionSpec, paged_cross_attention, paged_self_attention},
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, MusicGeneration, MusicLayers, Seq2SeqDecoding, TextToMusic},
    model::{
        ModelEntry,
        audio::encodec::{EncodecConfig, EncodecLayers},
        text::t5::{T5Config, T5Encoder},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicgenDecoderConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub codebooks: i64,
    /// Each codebook's codes, which its head scores; its embeddings also hold the pad.
    pub vocab_size: i64,
    pub max_positions: i64,
    pub eps: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicgenConfig {
    pub text: T5Config,
    pub decoder: MusicgenDecoderConfig,
    pub codec: EncodecConfig,
    pub generation: MusicGeneration,
}

/// A decoder block: causal self-attention, cross-attention to the text, and a feed-forward, each normed before and added back.
#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    attention: GroupedQueryAttention,
    cross_norm: LayerNorm,
    cross: CrossAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl Block {
    fn create(schema: &mut Schema<'_>, config: &MusicgenDecoderConfig) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let spec = AttentionSpec {
            qkv_bias: false,
            output_bias: false,
            ..AttentionSpec::square(width, config.heads)
        };
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            attention: paged_self_attention(
                &mut schema.scope("attention"),
                spec,
                AttentionOptions::default(),
            )?,
            cross_norm: schema.layer_norm("cross_attention_norm", width, eps)?,
            cross: paged_cross_attention(&mut schema.scope("cross_attention"), spec)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                false,
            )?,
        })
    }
}

/// `MusicGen`'s decoder: from packed frames to each codebook's logits, `[S, K · V]`.
#[derive(Debug, Clone)]
struct MusicgenDecoder {
    tokens: Vec<Embedding>,
    positions: StateKey,
    blocks: Vec<Block>,
    norm: LayerNorm,
    heads: Vec<Linear>,
    config: MusicgenDecoderConfig,
    bits: u32,
}

impl MusicgenDecoder {
    /// Returns each frame's summed codebook embeddings, `[N, W]`, unpacking codebook `k`'s code from `bits` bits of each token, starting at bit `k · bits`.
    fn embed(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let device = tokens.device();
        let mask = capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::scalar(), vec![(1_i64 << self.bits) - 1])
        })?;
        let mut hidden: Option<Tensor> = None;
        for (embedding, codebook) in self.tokens.iter().zip(0_i64..) {
            let shift = i64::from(self.bits) * codebook;
            let shift = capture::literal(device, |device| {
                Tensor::from_vec(device, Shape::scalar(), vec![shift])
            })?;
            let codes = tokens.bitwise_right_shift(&shift)?.bitwise_and(&mask)?;
            let embedded = codes.apply(embedding, state)?;
            hidden = Some(match hidden {
                Some(hidden) => hidden.add(&embedded)?,
                None => embedded,
            });
        }
        hidden.ok_or_else(|| SingeError::invalid_argument("a MusicGen decoder without codebooks"))
    }
}

impl Seq2SeqDecoding for MusicgenDecoder {
    fn cross_attentions(&self) -> Vec<&CrossAttention> {
        self.blocks.iter().map(|block| &block.cross).collect()
    }

    fn cache_spec(&self) -> KvCacheSpec {
        KvCacheSpec {
            kv_heads: self.config.heads,
            head_dim: self.config.width / self.config.heads,
        }
    }

    fn forward(
        &self,
        step: &DecodeStep,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        if caches.len() != self.blocks.len() {
            return Err(SingeError::missing_input("a decoder block's cache"));
        }

        let positions = state
            .tensor(&self.positions)?
            .index_select(0, step.positions())?;
        let mut hidden = self.embed(step.tokens(), state)?.add(&positions)?;
        for (block, cache) in self.blocks.iter().zip(caches) {
            let attended = block.attention.decode(
                &AttentionInput {
                    hidden: &hidden.apply(&block.attention_norm, state)?,
                    step,
                    angles: None,
                    cache: &cache.self_attention(),
                },
                state,
            )?;
            hidden = hidden.add(&attended)?;
            let crossed = block.cross.forward(
                &CrossAttentionInput {
                    hidden: &hidden.apply(&block.cross_norm, state)?,
                    step,
                    keys: cache.cross_keys(),
                    values: cache.cross_values(),
                    lengths: cache.source_lengths(),
                },
                state,
            )?;
            hidden = hidden.add(&crossed)?;
            let fed = hidden
                .apply(&block.feed_forward_norm, state)?
                .apply(&block.feed_forward, state)?;
            hidden = hidden.add(&fed)?;
        }

        let last = hidden
            .apply(&self.norm, state)?
            .index_select(0, step.last_rows())?;
        let logits = self
            .heads
            .iter()
            .map(|head| last.apply(head, state))
            .collect::<SingeResult<Vec<_>>>()?;
        Tensor::concat(&logits, 1)
    }
}

/// `MusicGen`: T5's encoder, its projection to the decoder's width, the decoder, and `EnCodec`.
#[derive(Debug, Clone)]
pub struct MusicgenLayers {
    text: T5Encoder,
    projection: Linear,
    decoder: Arc<MusicgenDecoder>,
    codec: EncodecLayers,
}

impl MusicgenLayers {
    pub fn create(schema: &mut Schema<'_>, config: &MusicgenConfig) -> SingeResult<Self> {
        let text = &config.text;
        let decoder = &config.decoder;
        let mut text_schema = schema.scope("text_encoder");
        let tokens =
            Embedding::new(text_schema.parameter("tokens.weight", [text.vocab_size, text.width])?);
        let text_encoder = T5Encoder::create(&mut text_schema, text, tokens)?;
        let projection = schema.linear("enc_to_dec", text.width, decoder.width, true)?;
        let mut decoder_schema = schema.scope("decoder");
        let tokens = (0..decoder.codebooks)
            .map(|index| {
                decoder_schema
                    .parameter(
                        &format!("tokens.{index}.weight"),
                        [decoder.vocab_size + 1, decoder.width],
                    )
                    .map(Embedding::new)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let positions =
            decoder_schema.parameter("positions", [decoder.max_positions, decoder.width])?;
        let blocks = (0..decoder.layers)
            .map(|index| {
                Block::create(
                    &mut decoder_schema.scope(format!("blocks.{index}")),
                    decoder,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = decoder_schema.layer_norm("norm", decoder.width, decoder.eps)?;
        let heads = (0..decoder.codebooks)
            .map(|index| {
                decoder_schema.linear(
                    &format!("heads.{index}"),
                    decoder.width,
                    decoder.vocab_size,
                    false,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let codec = EncodecLayers::create(&mut schema.scope("audio_encoder"), &config.codec)?;
        Ok(Self {
            text: text_encoder,
            projection,
            decoder: Arc::new(MusicgenDecoder {
                tokens,
                positions,
                blocks,
                norm,
                heads,
                config: decoder.clone(),
                bits: config.generation.bits,
            }),
            codec,
        })
    }
}

impl MusicLayers for MusicgenLayers {
    fn encode(&self, tokens: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.text
            .forward(tokens.tokens(), state)?
            .apply(&self.projection, state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder) as Arc<dyn Seq2SeqDecoding>
    }

    fn decode_codes(&self, codes: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.codec.decode_codes(codes, state)?.cast(DataType::F32)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "musicgen",
    formats: &[(Format::Transformers, &["musicgen"])],
    loaders: &[HeadLoader::TextToMusic(load_music)],
};

pub fn load_music(loader: &mut Loader<'_>) -> Result<TextToMusic> {
    let config: MusicgenConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        MusicgenLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn MusicLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config))?;
    Ok(TextToMusic::new(model, config.generation))
}
