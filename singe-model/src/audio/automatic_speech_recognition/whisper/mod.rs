use std::{collections::HashMap, iter, path::Path, slice};

use serde::Deserialize;
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Tensor,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, PromptChunks, RunningBatch},
    graph::Graph,
    nn::{
        Activation, AttentionInput, CacheSize, Conv1d, Conv1dConfig, CrossAttention, DecodeCache,
        DecodeStep, DecoderModel, DecodingLayer, DenseAttentionInput, Embedding,
        EncoderDecoderBlock, EncoderDecoderCache, EncoderDecoderCacheSpec, EncoderDecoderParts,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, Layer, LayerNorm, Linear, Mlp,
        Model, ModelState, PreNormBlock, SelfAttention, StateBuilder, StateKey, caches_from_inputs,
        named_cache_inputs,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    assembly::{causal, load_weights},
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
    tokenization::TextTokenizer,
};

// The slots of a cross-attention cache page.
const SOURCE_PAGE: i64 = 32;

// `torch.nn.LayerNorm`'s default, which every Whisper norm uses.
const LAYER_NORM_EPS: f32 = 1e-5;

// The fields of a Hugging Face Whisper `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WhisperConfig {
    pub vocab_size: i64,
    pub num_mel_bins: i64,
    pub d_model: i64,
    pub encoder_layers: i64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub decoder_layers: i64,
    pub decoder_attention_heads: i64,
    pub decoder_ffn_dim: i64,
    pub max_source_positions: i64,
    pub max_target_positions: i64,
    pub activation_function: Activation,
    #[serde(default)]
    pub scale_embedding: bool,
}

impl WhisperConfig {
    fn validate(&self) -> Result<()> {
        if self.scale_embedding {
            return Err(unsupported("scaled embeddings"));
        }
        Ok(())
    }

    // The mel frames the encoder reads, which its strided convolution halves.
    #[must_use]
    pub const fn frames(&self) -> i64 {
        2 * self.max_source_positions
    }
}

type EncoderBlock = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

// Whisper's encoder: two convolutions with GELU, the second halving the frames, learned positions for every frame, and pre-norm blocks of attention without a mask.
// It turns log-mel features, `[B, mels, 2F]`, into `[B, F, D]`.
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayers {
    first: Conv1d,
    second: Conv1d,
    positions: StateKey,
    blocks: Vec<EncoderBlock>,
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

// How `Whisper::transcribe_batch` generates: what to generate and when to stop, how many sequences run at once, and whether its steps are compiled.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    pub generation: GenerationOptions,
    pub rows: usize,
    pub compile: bool,
}

impl TranscribeOptions {
    // Generates eagerly, one sequence at a time.
    #[must_use]
    pub const fn new(generation: GenerationOptions) -> Self {
        Self {
            generation,
            rows: 1,
            compile: false,
        }
    }

    #[must_use]
    pub const fn with_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }

    #[must_use]
    pub const fn with_compile(mut self, compile: bool) -> Self {
        self.compile = compile;
        self
    }
}

// Whisper: its log-mel front end, its encoder, and its decoder, each half with state of its own.
#[derive(Debug, Clone)]
pub struct Whisper {
    config: WhisperConfig,
    features: LogMelSpectrogram,
    encoder: Model<WhisperEncoderLayers>,
    decoder: Model<WhisperDecoderLayers>,
}

impl Whisper {
    #[must_use]
    pub const fn config(&self) -> &WhisperConfig {
        &self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<WhisperEncoderLayers> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<WhisperDecoderLayers> {
        &self.decoder
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

    // The step that encodes one clip's features, `[1, mels, frames]`, and writes the cross-attention caches of `decoder`'s first sequence, as a graph whose inputs are `features` and the caches, named as the decoder's steps name them.
    // Exported with the decoder's step as `encode` and `decode`, it lets a generation loop transcribe with nothing of Singe.
    pub fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<WhisperDecoderLayers>,
        features: &Tensor,
    ) -> SingeResult<(Graph, HashMap<String, Tensor>)> {
        let (encoder, layers) = (&self.encoder, self.decoder.layer());
        let caches = named_cache_inputs(decoder.caches());
        let inputs = iter::once(("features".to_owned(), features.clone()))
            .chain(caches)
            .collect::<Vec<_>>();
        let bindings = encoder
            .state()
            .capture_bindings()?
            .merged(self.decoder.state().capture_bindings()?);
        let (graph, _) = Graph::capture_with_bindings(inputs.clone(), bindings, |tensors| {
            let (features, captured) = tensors.split_at(1);
            let caches = caches_from_inputs(decoder.caches(), captured)?;
            let sources = encoder.forward(&features[0])?.squeeze(0)?;
            layers.write_sources(&sources, 0, &caches, self.decoder.state())?;
            Ok(vec![sources])
        })?;
        Ok((graph, inputs.into_iter().collect()))
    }

    // Transcribes one clip of 16 kHz samples, `[N]`, as `transcribe_batch` does a batch of one.
    pub fn transcribe(
        &self,
        samples: &Tensor,
        prefix: &[i64],
        options: &TranscribeOptions,
    ) -> SingeResult<Vec<i64>> {
        self.transcribe_batch(slice::from_ref(samples), prefix, options)?
            .pop()
            .ok_or_else(|| SingeError::missing_output("transcription"))
    }

    // Transcribes clips of 16 kHz samples, each `[N]`: encodes them in one batch, then generates greedily after `prefix` in `options.rows` rows, each clip joining the running batch as a row frees, with its encoding written to its row first.
    // Returns each clip's generated tokens, in order.
    pub fn transcribe_batch(
        &self,
        clips: &[Tensor],
        prefix: &[i64],
        options: &TranscribeOptions,
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
            DecoderOptions::new(options.rows, self.config.max_target_positions)
                .with_prompt_chunks(chunks),
        )?;
        if options.compile {
            decoder.compile()?;
        }

        let mut tokens = vec![Vec::new(); clips.len()];
        let mut next = 0;
        let mut batch = RunningBatch::create(&mut decoder, options.generation.clone())?;
        loop {
            while next < clips.len() {
                let encoding = encoded
                    .narrow(
                        0,
                        i64::try_from(next).map_err(|_| SingeError::out_of_range("clips"))?,
                        1,
                    )?
                    .squeeze(0)?;
                let joined = batch.join_with(prefix, |decoder, row| {
                    self.write_sources(decoder, row, &encoding)
                })?;
                if joined.is_none() {
                    break;
                }

                next += 1;
            }

            if batch.is_empty() {
                return Ok(tokens);
            }

            for generated in batch.advance(None)? {
                tokens[generated.sequence].push(generated.token);
            }
        }
    }
}

fn mlp(
    schema: &mut Schema<'_>,
    width: i64,
    inner: i64,
    activation: Activation,
) -> SingeResult<Mlp> {
    Ok(Mlp::new(
        schema.linear("fc1", width, inner, true)?,
        schema.linear("fc2", inner, width, true)?,
        activation,
    ))
}

// Whisper's attention projections: biases on queries, values, and outputs, and none on keys.
fn projections(schema: &mut Schema<'_>, width: i64) -> SingeResult<[Linear; 4]> {
    Ok([
        schema.linear("q_proj", width, width, true)?,
        schema.linear("k_proj", width, width, false)?,
        schema.linear("v_proj", width, width, true)?,
        schema.linear("out_proj", width, width, true)?,
    ])
}

fn encoder_block(schema: &mut Schema<'_>, config: &WhisperConfig) -> SingeResult<EncoderBlock> {
    let width = config.d_model;
    let [query, key, value, output] = projections(&mut schema.scope("self_attn"), width)?;
    let attention = SelfAttention::create(
        query,
        key,
        value,
        output,
        config.encoder_attention_heads,
        width,
    )?;
    Ok(PreNormBlock::new(
        schema.layer_norm("self_attn_layer_norm", width, LAYER_NORM_EPS)?,
        attention,
        schema.layer_norm("final_layer_norm", width, LAYER_NORM_EPS)?,
        mlp(
            schema,
            width,
            config.encoder_ffn_dim,
            config.activation_function,
        )?,
    ))
}

fn decoder_block(
    schema: &mut Schema<'_>,
    config: &WhisperConfig,
) -> SingeResult<EncoderDecoderBlock> {
    let width = config.d_model;
    let heads = config.decoder_attention_heads;
    let head_dim = width / heads;
    let [query, key, value, output] = projections(&mut schema.scope("self_attn"), width)?;
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
    let [query, key, value, output] = projections(&mut schema.scope("encoder_attn"), width)?;
    let cross_attention = CrossAttention::create(
        query,
        key,
        value,
        output,
        heads,
        width,
        config.max_source_positions,
    )?;
    Ok(EncoderDecoderBlock::new(EncoderDecoderParts {
        self_norm: schema.layer_norm("self_attn_layer_norm", width, LAYER_NORM_EPS)?,
        self_attention,
        cross_norm: schema.layer_norm("encoder_attn_layer_norm", width, LAYER_NORM_EPS)?,
        cross_attention,
        feed_forward_norm: schema.layer_norm("final_layer_norm", width, LAYER_NORM_EPS)?,
        feed_forward: mlp(
            schema,
            width,
            config.decoder_ffn_dim,
            config.activation_function,
        )?,
    }))
}

fn encoder_schema(
    config: &WhisperConfig,
    data_type: DataType,
) -> Result<Model<WhisperEncoderLayers>> {
    let width = config.d_model;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut model = Schema::new(builder.scope("model.encoder"), data_type);
        let convolution = |schema: &mut Schema<'_>, name: &str, input: i64, stride: i64| {
            let weight = schema.parameter(&format!("{name}.weight"), [width, input, 3])?;
            let bias = schema.parameter(&format!("{name}.bias"), [width])?;
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
        let first = convolution(&mut model, "conv1", config.num_mel_bins, 1)?;
        let second = convolution(&mut model, "conv2", width, 2)?;
        let positions = model.parameter(
            "embed_positions.weight",
            [config.max_source_positions, width],
        )?;
        let blocks = (0..config.encoder_layers)
            .map(|layer| encoder_block(&mut model.scope(format!("layers.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("layer_norm", width, LAYER_NORM_EPS)?;
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

fn decoder_schema(
    config: &WhisperConfig,
    data_type: DataType,
) -> Result<Model<WhisperDecoderLayers>> {
    let width = config.d_model;
    let mut builder = StateBuilder::new();
    let (tokens, positions, blocks, norm) = {
        let mut model = Schema::new(builder.scope("model.decoder"), data_type);
        let tokens = model.parameter("embed_tokens.weight", [config.vocab_size, width])?;
        let positions = model.parameter(
            "embed_positions.weight",
            [config.max_target_positions, width],
        )?;
        let blocks = (0..config.decoder_layers)
            .map(|layer| decoder_block(&mut model.scope(format!("layers.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("layer_norm", width, LAYER_NORM_EPS)?;
        (tokens, positions, blocks, norm)
    };
    let head = builder.tie("proj_out.weight", &tokens)?;
    Ok(Model::new(
        WhisperDecoderLayers {
            tokens: Embedding::new(tokens),
            positions: Embedding::new(positions),
            blocks,
            norm,
            head: Linear::new(head, None),
            cache: EncoderDecoderCacheSpec {
                heads: config.decoder_attention_heads,
                head_dim: width / config.decoder_attention_heads,
                sources: config.max_source_positions,
                source_page: SOURCE_PAGE,
            },
        },
        builder.finish()?,
    ))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Whisper> {
    let snapshot = snapshot.as_ref();
    let config: WhisperConfig = Snapshot::new(snapshot).config()?;
    config.validate()?;
    let encoder_names = |name: &str| name.starts_with("model.encoder.").then(|| name.to_owned());
    let decoder_names = |name: &str| {
        (name.starts_with("model.decoder.") || name == "proj_out.weight").then(|| name.to_owned())
    };
    Ok(Whisper {
        features: LogMelSpectrogram::create(LogMelOptions::whisper(config.num_mel_bins))?,
        encoder: load_weights(
            &encoder_schema(&config, data_type)?,
            snapshot,
            device,
            Some(&encoder_names),
        )?,
        decoder: load_weights(
            &decoder_schema(&config, data_type)?,
            snapshot,
            device,
            Some(&decoder_names),
        )?,
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

// How `transformers`' `generate` continues a prompt for this snapshot, greedily: the generation config's end of text stops a sequence, its `suppress_tokens` are never chosen, and its `begin_suppress_tokens` are not chosen first.
pub fn generation_options(
    snapshot: impl AsRef<Path>,
    max_new_tokens: usize,
) -> Result<GenerationOptions> {
    #[derive(Deserialize)]
    struct Fields {
        #[serde(default)]
        suppress_tokens: Vec<i64>,
        #[serde(default)]
        begin_suppress_tokens: Vec<i64>,
    }

    let snapshot = Snapshot::new(snapshot);
    let fields: Fields = snapshot.json("generation_config.json")?;
    Ok(GenerationOptions {
        max_new_tokens,
        stop_tokens: snapshot.stop_tokens()?,
        sampling: None,
        suppressed_tokens: fields.suppress_tokens,
        first_suppressed_tokens: fields.begin_suppress_tokens,
    })
}
