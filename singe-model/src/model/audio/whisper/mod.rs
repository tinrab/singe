//! Whisper: an encoder of log-mel features and a decoder that cross-attends to it, each with state of its own, `encoder` and `decoder`.

mod transformers;

use std::{collections::HashMap, slice};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, PromptChunks, generate_batch},
    graph::Graph,
    nn::{
        Activation, AttentionInput, CacheSize, Conv1d, Conv1dConfig, CrossAttention, DecodeCache,
        DecodeStep, DecoderModel, DecodingLayer, Embedding, EncoderDecoderBlock,
        EncoderDecoderCache, EncoderDecoderCacheSpec, EncoderDecoderParts, GroupedQueryAttention,
        GroupedQueryLayers, HeadProjection, Layer, LayerNorm, Linear, ModelState, StateKey,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    BatchOptions,
    assembly::{
        FinalNorm, PYTORCH_NORM_EPS, PreNormEncoder, PreNormEncoderConfig, causal, same_1d,
    },
    decoder_cache::cached_decoder,
    error::Result,
    format::Loader,
    head::{HeadLoader, LoopLimits, LoopSteps, SpeechToText, Transcription},
    model::{ModelEntry, audio::speech::SpeechModel},
    processor::text::TextTokenizer,
    schema::Schema,
    source::Format,
};

/// The slots of a cross-attention cache page.
const SOURCE_PAGE: i64 = 32;

/// Whisper's attention projection biases: on queries, values, and outputs, and none on keys.
const PROJECTION_BIASES: [bool; 4] = [true, false, true, true];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub vocab_size: i64,
    pub mels: i64,
    pub width: i64,
    pub encoder_layers: i64,
    pub encoder_heads: i64,
    pub encoder_inner: i64,
    pub decoder_layers: i64,
    pub decoder_heads: i64,
    pub decoder_inner: i64,
    /// The positions the encoder gives a clip, half its mel frames.
    pub source_positions: i64,
    /// The most tokens the decoder reads.
    pub target_positions: i64,
    pub activation: Activation,
}

impl WhisperConfig {
    /// Returns the mel frames the encoder reads, which its strided convolution halves.
    #[must_use]
    pub const fn frames(&self) -> i64 {
        2 * self.source_positions
    }

    const fn encoder(&self) -> PreNormEncoderConfig {
        PreNormEncoderConfig {
            width: self.width,
            heads: self.encoder_heads,
            inner: self.encoder_inner,
            activation: self.activation,
            eps: PYTORCH_NORM_EPS,
            projection_biases: PROJECTION_BIASES,
            causal: false,
            layer_scale: false,
        }
    }
}

/// Whisper's encoder: two convolutions with GELU, the second halving the frames, a position embedding added to each of the `F` rows, and pre-norm blocks of attention without a mask.
/// It turns log-mel features, `[B, mels, 2F]`, into `[B, F, D]`, and holds the log-mel front end that pads or cuts a clip to the samples those frames take.
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayers {
    features: LogMelSpectrogram,
    first: Conv1d,
    second: Conv1d,
    positions: StateKey,
    encoder: PreNormEncoder,
}

impl WhisperEncoderLayers {
    /// Creates the encoder's state under `encoder.`, so it shares no name with the decoder's in a graph of both: `convolutions.0` and `convolutions.1`, `embeddings.positions`, `blocks.N`, and `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &WhisperConfig) -> SingeResult<Self> {
        let mut schema = schema.scope("encoder");
        let width = config.width;
        let strided = |stride| Conv1dConfig {
            stride,
            ..same_1d(3, 1)
        };
        let options = LogMelOptions::whisper(config.mels);
        Ok(Self {
            // As many samples as the encoder reads frames, a hop each: 30 seconds for the published models.
            features: LogMelSpectrogram::create(LogMelOptions {
                samples: config.frames() * options.hop,
                ..options
            })?,
            first: schema.conv1d("convolutions.0", config.mels, width, 3, strided(1), true)?,
            second: schema.conv1d("convolutions.1", width, width, 3, strided(2), true)?,
            positions: schema.parameter(
                "embeddings.positions.weight",
                [config.source_positions, width],
            )?,
            encoder: PreNormEncoder::create(
                &mut schema,
                &config.encoder(),
                config.encoder_layers,
                FinalNorm::Included,
            )?,
        })
    }
}

impl Layer for WhisperEncoderLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = input
            .apply(&self.first, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.second, state)?
            .apply(&Activation::GeluExact, state)?
            .transpose(1, 2)?
            .add(state.tensor(&self.positions)?)?;
        self.encoder.encode(&hidden, None, state)
    }
}

/// Whisper's decoder: token and learned position embeddings, pre-norm blocks of causal self-attention, cross-attention to the encoder, and a feed-forward layer, then a final norm and a head tied to the token embeddings.
#[derive(Debug, Clone)]
pub struct WhisperDecoderLayers {
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<EncoderDecoderBlock>,
    norm: LayerNorm,
    head: Linear,
    cache: EncoderDecoderCacheSpec,
}

impl WhisperDecoderLayers {
    /// Creates the decoder's state under `decoder.`: `embeddings.tokens`, `embeddings.positions`, `blocks.N`, `norm`, and `head`, tied to the token embeddings.
    pub fn create(schema: &mut Schema<'_>, config: &WhisperConfig) -> SingeResult<Self> {
        let mut schema = schema.scope("decoder");
        let width = config.width;
        let tokens = schema.parameter("embeddings.tokens.weight", [config.vocab_size, width])?;
        let positions = schema.parameter(
            "embeddings.positions.weight",
            [config.target_positions, width],
        )?;
        let blocks = (0..config.decoder_layers)
            .map(|layer| Self::block(&mut schema.scope(format!("blocks.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            head: Linear::new(schema.tie("head.weight", &tokens)?, None),
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm: schema.layer_norm("norm", width, PYTORCH_NORM_EPS)?,
            cache: EncoderDecoderCacheSpec {
                heads: config.decoder_heads,
                head_dim: width / config.decoder_heads,
                sources: config.source_positions,
                source_page: SOURCE_PAGE,
            },
        })
    }

    fn block(schema: &mut Schema<'_>, config: &WhisperConfig) -> SingeResult<EncoderDecoderBlock> {
        let width = config.width;
        let heads = config.decoder_heads;
        let head_dim = width / heads;
        let [query, key, value, output] = schema
            .scope("attention")
            .attention_projections(width, PROJECTION_BIASES)?;
        let head = |linear| HeadProjection::new(linear, heads, head_dim);
        let self_attention = GroupedQueryAttention::create(
            GroupedQueryLayers {
                query: head(query),
                key: head(key),
                value: head(value),
                output,
            },
            causal(),
        )?;
        let [query, key, value, output] = schema
            .scope("cross_attention")
            .attention_projections(width, PROJECTION_BIASES)?;
        let cross_attention = CrossAttention::create(
            query,
            key,
            value,
            output,
            heads,
            width,
            config.source_positions,
        )?;
        Ok(EncoderDecoderBlock::new(EncoderDecoderParts {
            self_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            self_attention,
            cross_norm: schema.layer_norm("cross_attention_norm", width, PYTORCH_NORM_EPS)?,
            cross_attention,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.decoder_inner, width],
                config.activation,
                true,
            )?,
        }))
    }

    /// Writes one sequence's cross-attention keys and values to `caches`, from its encoder's outputs, `[F, D]`.
    pub fn write_sources(
        &self,
        encoded: &Tensor,
        sequence: usize,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<()> {
        for (block, cache) in self.blocks.iter().zip(caches) {
            block.write_sources(encoded, sequence, cache, state)?;
        }
        Ok(())
    }
}

impl DecoderModel for WhisperDecoderLayers {
    type Cache = EncoderDecoderCache;

    /// No tables: positions are looked up in the state, so nothing depends on the capacity.
    type Tables = ();

    fn tables(&self, _: DataType, _: Device, _: i64) -> SingeResult<()> {
        Ok(())
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<EncoderDecoderCache>> {
        self.blocks
            .iter()
            .map(|_| EncoderDecoderCache::create(device, data_type, size, self.cache))
            .collect()
    }

    fn forward(
        &self,
        step: &DecodeStep,
        (): &(),
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = step
            .tokens()
            .apply(&self.tokens, state)?
            .add(&step.positions().apply(&self.positions, state)?)?;
        for (block, cache) in self.blocks.iter().zip(caches) {
            hidden = block.decode(
                &AttentionInput {
                    hidden: &hidden,
                    step,
                    angles: None,
                    cache,
                },
                state,
            )?;
        }
        hidden
            .apply(&self.norm, state)?
            .index_select(0, step.last_rows())?
            .apply(&self.head, state)
    }
}

/// Whisper: its log-mel front end with its encoder, its decoder, and how its snapshot generates.
pub type Whisper = SpeechModel<WhisperConfig, WhisperEncoderLayers, WhisperDecoderLayers>;

impl Whisper {
    /// Returns how many samples the front end reads, to which a clip is padded with silence or cut.
    #[must_use]
    pub fn samples(&self) -> i64 {
        self.encoder().layer().features.options().samples
    }

    /// Returns the log-mel features the encoder reads, `[B, mels, frames]`, of 16 kHz samples, `[B, N]` or `[N]`.
    pub fn features(&self, samples: &Tensor) -> SingeResult<Tensor> {
        let encoder = self.encoder();
        encoder
            .layer()
            .features
            .forward(samples, encoder.state())?
            .cast(encoder.state().floating_type()?)
    }

    /// Encodes log-mel features, `[B, mels, frames]`, to `[B, F, D]`.
    pub fn encode(&self, features: &Tensor) -> SingeResult<Tensor> {
        self.encoder().forward(features)
    }

    /// Creates a decoder of `options.sequences` sequences, each of which reads an encoding written with [`Self::write_sources`] before its first step.
    pub fn create_decoder(
        &self,
        options: DecoderOptions,
    ) -> SingeResult<CachedDecoder<WhisperDecoderLayers>> {
        CachedDecoder::create(self.decoder(), options)
    }

    /// Writes one sequence's encoding, `[F, D]`, to `decoder`'s cross-attention caches.
    pub fn write_sources(
        &self,
        decoder: &CachedDecoder<WhisperDecoderLayers>,
        sequence: usize,
        encoded: &Tensor,
    ) -> SingeResult<()> {
        self.decoder().layer().write_sources(
            encoded,
            sequence,
            decoder.caches(),
            self.decoder().state(),
        )
    }

    /// Returns the step that encodes one clip of 16 kHz samples, `[N]` with `N` the front end's samples, a shorter clip padded with silence, and writes the cross-attention caches of `decoder`'s first sequence, and its example inputs.
    /// The step is a graph whose inputs are `samples` and the caches, named as the decoder's steps name them.
    /// Exported with the decoder's step as `encode` and `decode`, it lets a generation loop transcribe samples without Singe.
    pub fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<WhisperDecoderLayers>,
        samples: &Tensor,
    ) -> SingeResult<(Graph, HashMap<String, Tensor>)> {
        let layers = self.decoder().layer();
        decoder.encode_step_graph(
            self.encoder().state(),
            vec![("samples".to_owned(), samples.clone())],
            |inputs, caches| {
                let sources = self.encode(&self.features(&inputs[0])?)?.squeeze(0)?;
                layers.write_sources(&sources, 0, caches, self.decoder().state())?;
                Ok(vec![sources])
            },
        )
    }

    /// Generates for clips of 16 kHz samples, each `[N]`: encodes them in one batch, then generates after `prefix` in `options.rows` rows, each clip joining the running batch as a row frees, with its encoding written to its row first.
    /// Returns each clip's generated tokens, in order.
    pub fn generate(
        &self,
        clips: &[Tensor],
        prefix: &[i64],
        options: &BatchOptions,
    ) -> SingeResult<Vec<Vec<i64>>> {
        if clips.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); clips.len()]);
        }

        let features = clips
            .iter()
            .map(|clip| self.features(clip))
            .collect::<SingeResult<Vec<_>>>()?;
        let encoded = self.encode(&Tensor::concat(&features, 0)?)?;
        // Every sequence reads the same prefix, so one prompt step of its extra tokens is all a compiled decoder needs.
        let chunks = PromptChunks {
            smallest: prefix.len().saturating_sub(1),
            largest: prefix.len().saturating_sub(1),
        };
        let prompts = vec![prefix; clips.len()];
        let key = (
            options.rows,
            i64::try_from(prefix.len()).map_err(|_| SingeError::out_of_range("prefix"))?,
            options.compile,
        );
        self.decoders().run(
            key,
            || {
                cached_decoder(
                    self.decoder(),
                    DecoderOptions::new(options.rows, self.config().target_positions)
                        .with_prompt_chunks(chunks),
                    options.compile,
                )
            },
            |decoder| {
                generate_batch(
                    decoder,
                    &prompts,
                    &options.generation,
                    None,
                    |decoder, row, clip| {
                        let clip =
                            i64::try_from(clip).map_err(|_| SingeError::out_of_range("clips"))?;
                        self.write_sources(decoder, row, &encoded.narrow(0, clip, 1)?.squeeze(0)?)
                    },
                    |_, _| Ok(()),
                )
            },
        )
    }
}

impl SpeechToText for Whisper {
    fn sample_rate(&self) -> i64 {
        self.encoder().layer().features.options().sample_rate
    }

    /// Returns the encode and one-token steps of a loop that runs one sequence over Whisper's whole context, with no prompt step.
    /// The steps choose tokens with `generation`'s sampling and the snapshot's stop and suppressed tokens.
    fn loop_steps(
        &self,
        _limits: &LoopLimits,
        generation: &GenerationOptions,
    ) -> Result<LoopSteps> {
        let generation = GenerationOptions {
            sampling: generation.sampling,
            ..self.generation_options(1)
        };
        let decoder =
            self.create_decoder(DecoderOptions::new(1, self.config().target_positions))?;
        let samples = Tensor::zeroed_on(
            self.encoder().state().device()?,
            DataType::F32,
            Shape::create([self.samples()])?,
        )?;
        let (encode, _) = self.encode_step_graph(&decoder, &samples)?;
        LoopSteps::create(&decoder, Some(encode), &generation, None)
    }

    /// Transcribes one clip in `language`, a Whisper language code, or in English without one.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription> {
        let language = language.unwrap_or("en");
        let prefix = prompt(tokenizer, language, WhisperTask::Transcribe)?;
        let tokens = self
            .generate(
                slice::from_ref(samples),
                &prefix,
                &options.including_tokens_of(self.generation()),
            )?
            .pop()
            .ok_or_else(|| SingeError::missing_output("transcription"))?;
        Ok(Transcription {
            language: Some(language.to_owned()),
            text: tokenizer.decode(&tokens, true)?.trim().to_owned(),
        })
    }
}

/// What Whisper does with its audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

/// Returns the tokens a transcription without timestamps starts with: start of transcript, the language, the task, and no timestamps, as `transformers`' `generate` forces them.
/// It looks each token up by name in the snapshot's tokenizer.
///
/// # Errors
///
/// Returns an error if the tokenizer has no token of one of those names, such as for an unknown `language`.
pub fn prompt(tokenizer: &TextTokenizer, language: &str, task: WhisperTask) -> Result<Vec<i64>> {
    let task = match task {
        WhisperTask::Transcribe => "<|transcribe|>",
        WhisperTask::Translate => "<|translate|>",
    };
    [
        "<|startoftranscript|>",
        &format!("<|{language}|>"),
        task,
        "<|notimestamps|>",
    ]
    .into_iter()
    .map(|token| tokenizer.token_id(token))
    .collect()
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "whisper",
    formats: &[(Format::Transformers, &["whisper"])],
    loaders: &[HeadLoader::SpeechToText(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Whisper> {
    let config: WhisperConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        WhisperEncoderLayers::create(schema, &config)
    })?;
    let decoder = Schema::model(data_type, |schema| {
        WhisperDecoderLayers::create(schema, &config)
    })?;
    SpeechModel::load(
        loader,
        config,
        (&encoder, &transformers::encoder_rules()),
        (&decoder, &transformers::decoder_rules()),
    )
}
