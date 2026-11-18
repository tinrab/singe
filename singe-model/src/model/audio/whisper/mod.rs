// Whisper: an encoder of log-mel features and a decoder that cross-attends to it, each with state of its own, `encoder` and `decoder`.

mod transformers;

use std::{collections::HashMap, slice};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, PromptChunks, generate_batch},
    graph::Graph,
    nn::{
        Activation, AttentionInput, CacheSize, Conv1d, Conv1dConfig, CrossAttention, DecodeCache,
        DecodeStep, DecoderModel, DecodingLayer, DenseAttentionInput, Embedding,
        EncoderDecoderBlock, EncoderDecoderCache, EncoderDecoderCacheSpec, EncoderDecoderParts,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, Layer, LayerNorm, Linear, Mlp,
        Model, ModelState, StateBuilder, StateKey,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    BatchOptions,
    assembly::{PreNormEncoderBlock, PreNormEncoderSpec, causal, pre_norm_encoder_block},
    error::Result,
    format::{Loader, transformers::generation_options},
    head::{HeadLoader, LoopLimits, LoopSteps, SpeechToText, Transcription},
    model::ModelEntry,
    processing::text::TextTokenizer,
    schema::Schema,
    source::Format,
};

// The slots of a cross-attention cache page.
const SOURCE_PAGE: i64 = 32;

// `torch.nn.LayerNorm`'s default, which every Whisper norm uses.
const LAYER_NORM_EPS: f32 = 1e-5;

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
    // The positions the encoder gives a clip, half its mel frames.
    pub source_positions: i64,
    // The most tokens the decoder reads.
    pub target_positions: i64,
    pub activation: Activation,
}

impl WhisperConfig {
    // The mel frames the encoder reads, which its strided convolution halves.
    #[must_use]
    pub const fn frames(&self) -> i64 {
        2 * self.source_positions
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "whisper",
    formats: &[(Format::Transformers, &["whisper"])],
    heads: &[HeadLoader::SpeechToText(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

// Whisper's encoder: two convolutions with GELU, the second halving the frames, learned positions for every frame, and pre-norm blocks of attention without a mask.
// It turns log-mel features, `[B, mels, 2F]`, into `[B, F, D]`.
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayers {
    first: Conv1d,
    second: Conv1d,
    positions: StateKey,
    blocks: Vec<PreNormEncoderBlock>,
    norm: LayerNorm,
}

impl Layer for WhisperEncoderLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.first, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.second, state)?
            .apply(&Activation::GeluExact, state)?
            .transpose(1, 2)?
            .add(state.tensor(&self.positions)?)?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: None,
                },
                state,
            )?;
        }
        hidden.apply(&self.norm, state)
    }
}

// Whisper's decoder: token and learned position embeddings, pre-norm blocks of causal self-attention, cross-attention to the encoder, and a feed-forward, a final norm, and a head tied to the token embedding.
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
    // Writes one sequence's cross-attention keys and values to `caches`, from its encoder's outputs, `[F, D]`.
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

    // Positions are looked up in the state, so nothing depends on the capacity.
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

// Whisper: its log-mel front end, its encoder, and its decoder, each half with state of its own, and how its snapshot generates.
#[derive(Debug, Clone)]
pub struct Whisper {
    config: WhisperConfig,
    features: LogMelSpectrogram,
    encoder: Model<WhisperEncoderLayers>,
    decoder: Model<WhisperDecoderLayers>,
    // The snapshot's stop and suppressed tokens, generating nothing until a caller sets `max_new_tokens`.
    generation: GenerationOptions,
}

impl Whisper {
    #[must_use]
    pub const fn config(&self) -> &WhisperConfig {
        &self.config
    }

    // The samples the front end reads, to which a clip is padded with silence or cut.
    #[must_use]
    pub const fn samples(&self) -> i64 {
        self.features.options().samples
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<WhisperEncoderLayers> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<WhisperDecoderLayers> {
        &self.decoder
    }

    // How `transformers`' `generate` continues a prompt for this snapshot, greedily, up to `max_new_tokens`: the generation config's end of text stops a sequence, its `suppress_tokens` are never chosen, and its `begin_suppress_tokens` are not chosen first.
    #[must_use]
    pub fn generation_options(&self, max_new_tokens: usize) -> GenerationOptions {
        GenerationOptions {
            max_new_tokens,
            ..self.generation.clone()
        }
    }

    // Turns 16 kHz samples, `[B, N]` or `[N]`, into the log-mel features the encoder reads, `[B, mels, frames]`.
    pub fn features(&self, samples: &Tensor) -> SingeResult<Tensor> {
        self.features
            .forward(samples, self.encoder.state())?
            .cast(self.encoder.state().floating_type()?)
    }

    // Encodes log-mel features, `[B, mels, frames]`, to `[B, F, D]`.
    pub fn encode(&self, features: &Tensor) -> SingeResult<Tensor> {
        self.encoder.forward(features)
    }

    // A decoder of `options.sequences` sequences, each of which reads an encoding written with `write_sources` before its first step.
    pub fn create_decoder(
        &self,
        options: DecoderOptions,
    ) -> SingeResult<CachedDecoder<WhisperDecoderLayers>> {
        CachedDecoder::create(&self.decoder, options)
    }

    // Writes one sequence's encoding, `[F, D]`, to `decoder`'s cross-attention caches.
    pub fn write_sources(
        &self,
        decoder: &CachedDecoder<WhisperDecoderLayers>,
        sequence: usize,
        encoded: &Tensor,
    ) -> SingeResult<()> {
        self.decoder.layer().write_sources(
            encoded,
            sequence,
            decoder.caches(),
            self.decoder.state(),
        )
    }

    // The step that encodes one clip of 16 kHz samples, `[N]` with `N` the front end's samples, a shorter clip padded with silence, and writes the cross-attention caches of `decoder`'s first sequence, as a graph whose inputs are `samples` and the caches, named as the decoder's steps name them.
    // Exported with the decoder's step as `encode` and `decode`, it lets a generation loop transcribe samples with nothing of Singe.
    pub fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<WhisperDecoderLayers>,
        samples: &Tensor,
    ) -> SingeResult<(Graph, HashMap<String, Tensor>)> {
        let layers = self.decoder.layer();
        decoder.encode_step_graph(
            self.encoder.state(),
            vec![("samples".to_owned(), samples.clone())],
            |inputs, caches| {
                let sources = self
                    .encoder
                    .forward(&self.features(&inputs[0])?)?
                    .squeeze(0)?;
                layers.write_sources(&sources, 0, caches, self.decoder.state())?;
                Ok(vec![sources])
            },
        )
    }

    // Generates after `prefix` for one clip of 16 kHz samples, `[N]`, as `generate_clips` does for a batch of one.
    pub fn generate(
        &self,
        samples: &Tensor,
        prefix: &[i64],
        options: &BatchOptions,
    ) -> SingeResult<Vec<i64>> {
        self.generate_clips(slice::from_ref(samples), prefix, options)?
            .pop()
            .ok_or_else(|| SingeError::missing_output("transcription"))
    }

    // Generates for clips of 16 kHz samples, each `[N]`: encodes them in one batch, then generates after `prefix` in `options.rows` rows, each clip joining the running batch as a row frees, with its encoding written to its row first.
    // Returns each clip's generated tokens, in order.
    pub fn generate_clips(
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
        let mut decoder = self.create_decoder(
            DecoderOptions::new(options.rows, self.config.target_positions)
                .with_prompt_chunks(chunks),
        )?;
        if options.compile {
            decoder.compile()?;
        }

        let prompts = vec![prefix; clips.len()];
        generate_batch(
            &mut decoder,
            &prompts,
            &options.generation,
            None,
            |decoder, row, clip| {
                let clip = i64::try_from(clip).map_err(|_| SingeError::out_of_range("clips"))?;
                self.write_sources(decoder, row, &encoded.narrow(0, clip, 1)?.squeeze(0)?)
            },
            |_, _| Ok(()),
        )
    }
}

impl SpeechToText for Whisper {
    fn sample_rate(&self) -> i64 {
        self.features.options().sample_rate
    }

    // Whisper's loop runs one sequence over its whole context, prompted by its prefix, so it has no prompt step.
    fn loop_steps(
        &self,
        _limits: &LoopLimits,
        generation: &GenerationOptions,
    ) -> Result<LoopSteps> {
        let generation = GenerationOptions {
            sampling: generation.sampling,
            ..self.generation_options(1)
        };
        let decoder = self.create_decoder(DecoderOptions::new(1, self.config.target_positions))?;
        let samples = Tensor::zeroed_on(
            self.encoder.state().device()?,
            DataType::F32,
            Shape::create([self.samples()])?,
        )?;
        let (encode, _) = self.encode_step_graph(&decoder, &samples)?;
        LoopSteps::create(&decoder, Some(encode), &generation, None)
    }

    // Transcribes in `language`, a Whisper language code, English when it names none.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription> {
        let language = language.unwrap_or("en");
        let prefix = prompt(tokenizer, language, WhisperTask::Transcribe)?;
        let tokens = self.generate(samples, &prefix, &options.with_tokens_of(&self.generation))?;
        Ok(Transcription {
            language: Some(language.to_owned()),
            text: tokenizer.decode(&tokens, true)?.trim().to_owned(),
        })
    }
}

fn mlp(
    schema: &mut Schema<'_>,
    width: i64,
    inner: i64,
    activation: Activation,
) -> SingeResult<Mlp> {
    let mut schema = schema.scope("feed_forward");
    Ok(Mlp::new(
        schema.linear("up", width, inner, true)?,
        schema.linear("down", inner, width, true)?,
        activation,
    ))
}

// Whisper's attention projections: biases on queries, values, and outputs, and none on keys.
fn projections(schema: &mut Schema<'_>, width: i64) -> SingeResult<[Linear; 4]> {
    Ok([
        schema.linear("query", width, width, true)?,
        schema.linear("key", width, width, false)?,
        schema.linear("value", width, width, true)?,
        schema.linear("output", width, width, true)?,
    ])
}

fn decoder_block(
    schema: &mut Schema<'_>,
    config: &WhisperConfig,
) -> SingeResult<EncoderDecoderBlock> {
    let width = config.width;
    let heads = config.decoder_heads;
    let head_dim = width / heads;
    let [query, key, value, output] = projections(&mut schema.scope("attention"), width)?;
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
    let [query, key, value, output] = projections(&mut schema.scope("cross_attention"), width)?;
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
        self_norm: schema.layer_norm("attention_norm", width, LAYER_NORM_EPS)?,
        self_attention,
        cross_norm: schema.layer_norm("cross_attention_norm", width, LAYER_NORM_EPS)?,
        cross_attention,
        feed_forward_norm: schema.layer_norm("feed_forward_norm", width, LAYER_NORM_EPS)?,
        feed_forward: mlp(schema, width, config.decoder_inner, config.activation)?,
    }))
}

// The encoder's state: `convolutions.0` and `convolutions.1`, `embeddings.positions`, `blocks.N`, and `norm`.
fn encoder_schema(
    config: &WhisperConfig,
    data_type: DataType,
) -> Result<Model<WhisperEncoderLayers>> {
    let width = config.width;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut model = Schema::root(&mut builder, data_type);
        let convolution = |schema: &mut Schema<'_>, index: usize, input: i64, stride: i64| {
            let weight =
                schema.parameter(&format!("convolutions.{index}.weight"), [width, input, 3])?;
            let bias = schema.parameter(&format!("convolutions.{index}.bias"), [width])?;
            SingeResult::Ok(Conv1d::new(
                weight,
                Some(bias),
                Conv1dConfig {
                    stride,
                    padding: (1, 1),
                    ..Conv1dConfig::default()
                },
            ))
        };
        let first = convolution(&mut model, 0, config.mels, 1)?;
        let second = convolution(&mut model, 1, width, 2)?;
        let positions = model.parameter(
            "embeddings.positions.weight",
            [config.source_positions, width],
        )?;
        let spec = PreNormEncoderSpec {
            width,
            heads: config.encoder_heads,
            inner: config.encoder_inner,
            activation: config.activation,
            eps: LAYER_NORM_EPS,
            key_bias: false,
            causal: false,
        };
        let blocks = (0..config.encoder_layers)
            .map(|layer| pre_norm_encoder_block(&mut model.scope(format!("blocks.{layer}")), &spec))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("norm", width, LAYER_NORM_EPS)?;
        WhisperEncoderLayers {
            first,
            second,
            positions,
            blocks,
            norm,
        }
    };
    Ok(Model::new(layers, builder.finish()?))
}

// The decoder's state: `embeddings.tokens`, `embeddings.positions`, `blocks.N`, `norm`, and `head`, tied to the token embeddings.
fn decoder_schema(
    config: &WhisperConfig,
    data_type: DataType,
) -> Result<Model<WhisperDecoderLayers>> {
    let width = config.width;
    let mut builder = StateBuilder::new();
    let (tokens, positions, blocks, norm) = {
        let mut model = Schema::root(&mut builder, data_type);
        let tokens = model.parameter("embeddings.tokens.weight", [config.vocab_size, width])?;
        let positions = model.parameter(
            "embeddings.positions.weight",
            [config.target_positions, width],
        )?;
        let blocks = (0..config.decoder_layers)
            .map(|layer| decoder_block(&mut model.scope(format!("blocks.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("norm", width, LAYER_NORM_EPS)?;
        (tokens, positions, blocks, norm)
    };
    let head = builder.tie("head.weight", &tokens)?;
    Ok(Model::new(
        WhisperDecoderLayers {
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm,
            head: Linear::new(head, None),
            cache: EncoderDecoderCacheSpec {
                heads: config.decoder_heads,
                head_dim: width / config.decoder_heads,
                sources: config.source_positions,
                source_page: SOURCE_PAGE,
            },
        },
        builder.finish()?,
    ))
}

pub fn load(loader: &mut Loader<'_>) -> Result<Whisper> {
    let config: WhisperConfig = loader.config(transformers::config)?;
    let generation = generation_options(loader.source())?;
    let data_type = loader.options().data_type;
    let encoder = loader.load(
        "encoder",
        &encoder_schema(&config, data_type)?,
        &transformers::encoder_rules(),
    )?;
    let decoder = loader.load(
        "decoder",
        &decoder_schema(&config, data_type)?,
        &transformers::decoder_rules(),
    )?;
    loader.finish()?;
    Ok(Whisper {
        // As many samples as the encoder reads frames, a hop each: 30 seconds for the published models.
        features: LogMelSpectrogram::create(LogMelOptions {
            samples: config.frames() * LogMelOptions::whisper(config.mels).hop,
            ..LogMelOptions::whisper(config.mels)
        })?,
        encoder,
        decoder,
        generation,
        config,
    })
}

// What Whisper does with its audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

// The tokens a transcription without timestamps starts with: start of transcript, the language, the task, and no timestamps, as `transformers`' `generate` forces them, by name through the snapshot's tokenizer.
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
