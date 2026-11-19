// Qwen3-ASR: an audio encoder of log-mel features and a Qwen3 decoder whose prompt reads the encoder's rows, each with state of its own, `encoder` and `decoder`.

mod transformers;

use std::{collections::HashMap, iter, slice};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, generate_batch},
    graph::{Graph, capture},
    nn::{
        Activation, Conv2d, Conv2dConfig, DenseAttentionInput, GatedMlp, GroupedQueryAttention,
        Layer, LayerNorm, Linear, Model, ModelState, PreNormBlock, RmsNorm, SourcedCausalLm,
        StateBuilder, TensorGroup,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    BatchOptions,
    assembly::{
        CausalLmSpec, DecoderConfig, PreNormEncoderBlock, PreNormEncoderSpec, causal, causal_lm,
        grouped_attention, pre_norm_block, pre_norm_encoder_block,
    },
    error::Result,
    format::{Loader, transformers::generation_options},
    head::{HeadLoader, LoopLimits, LoopSteps, SpeechToText, Transcription},
    model::ModelEntry,
    processor::text::TextTokenizer,
    schema::Schema,
    source::Format,
};

// `torch.nn.LayerNorm`'s default, which every norm of the encoder uses.
const LAYER_NORM_EPS: f32 = 1e-5;

// The shortest clip the feature extractor reads, in samples, which it pads shorter clips to.
const MIN_SAMPLES: i64 = 8_000;

// The samples between mel frames.
const HOP: i64 = 160;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3AsrEncoderConfig {
    pub mels: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    // The channels of each convolution.
    pub channels: i64,
    // The mel frames of a chunk, which the convolutions read apart.
    pub chunk_frames: i64,
    // The mel frames of an attention window, a whole number of chunks.
    pub window_frames: i64,
    // The width of the projector's rows, which the decoder reads.
    pub output_width: i64,
    pub activation: Activation,
}

impl Qwen3AsrEncoderConfig {
    // The positions a chunk gives after the convolutions.
    #[must_use]
    pub const fn chunk_positions(&self) -> i64 {
        downsampled(self.chunk_frames)
    }

    // The positions an attention window holds after the convolutions: as many chunks' as fit its mel frames.
    #[must_use]
    pub const fn window_positions(&self) -> i64 {
        self.chunk_positions() * (self.window_frames / self.chunk_frames)
    }

    // The frequencies the convolutions leave of the mel bins.
    const fn frequencies(&self) -> i64 {
        downsampled(self.mels)
    }

    // The positions the encoder gives `frames` valid mel frames: each chunk's, after three convolutions of stride two.
    #[must_use]
    pub const fn positions(&self, frames: i64) -> i64 {
        let chunk = self.chunk_frames;
        frames / chunk * downsampled(chunk) + downsampled(frames % chunk)
    }
}

// What three 3 × 3 convolutions of stride two and padding one leave of `length`, and nothing of nothing.
const fn downsampled(length: i64) -> i64 {
    let mut length = length;
    let mut convolution = 0;
    while convolution < 3 {
        if length > 0 {
            length = (length - 1) / 2 + 1;
        }

        convolution += 1;
    }
    length
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3AsrConfig {
    pub encoder: Qwen3AsrEncoderConfig,
    pub decoder: DecoderConfig,
    // The prompt's placeholder token, in whose place the decoder reads the encoder's rows.
    pub audio_token: i64,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3_asr",
    formats: &[(Format::Transformers, &["qwen3_asr"])],
    heads: &[HeadLoader::SpeechToText(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

// Qwen3-ASR's audio encoder: each chunk of mel frames through three convolutions of stride two with GELU, a projection of channels and frequencies, and sinusoid positions that restart each chunk; then the valid positions through pre-norm blocks attending within windows, a norm, and a projector of two linears around GELU into the decoder's width.
#[derive(Debug, Clone)]
pub struct Qwen3AsrEncoderLayers {
    convolutions: [Conv2d; 3],
    projection: Linear,
    blocks: Vec<PreNormEncoderBlock>,
    norm: LayerNorm,
    projector: [Linear; 2],
    config: Qwen3AsrEncoderConfig,
    // The sinusoid positions of a chunk, `[positions, width]`, row-major.
    positions: Vec<f32>,
}

// The sinusoids of `length` positions over `channels` features, sines then cosines, with timescales up to 10000, as Qwen3-ASR's `SinusoidsPositionEmbedding` computes them.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "positions and channels are small, and the values lie in [-1, 1]"
)]
fn sinusoids(length: i64, channels: i64) -> Vec<f32> {
    let half = channels / 2;
    let increment = 10_000.0_f64.ln() / (half - 1) as f64;
    let mut table = Vec::with_capacity(usize::try_from(length * channels).unwrap_or(0));
    for position in 0..length {
        let angles = (0..half)
            .map(|index| position as f64 * (-increment * index as f64).exp())
            .collect::<Vec<_>>();
        table.extend(angles.iter().map(|angle| angle.sin() as f32));
        table.extend(angles.iter().map(|angle| angle.cos() as f32));
    }
    table
}

impl Qwen3AsrEncoderLayers {
    // Encodes one clip's log-mel features, `[1, mels, F]` with `F` whole chunks, of which the first `frames`, an `I64` `[1]`, are its own.
    // Returns a row per position of every chunk, `[F / chunk · positions, output]`, of which the first `count`, an `I64` `[1]`, are the rows the prompt's audio tokens read; the rest mean nothing.
    // The valid positions and each window's length are computed from `frames` in the graph, so one captured encoder serves every clip of up to `F` frames.
    #[expect(
        clippy::cast_precision_loss,
        reason = "frame and position counts are far below where `F64` rounds them"
    )]
    pub fn encode(
        &self,
        features: &Tensor,
        frames: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let config = &self.config;
        let &[1, mels, padded] = features.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR encodes one clip's features, [1, mels, frames]",
            ));
        };
        let chunk = config.chunk_frames;
        if padded % chunk != 0 {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR's features are whole chunks",
            ));
        }

        let chunks = padded / chunk;
        let mut hidden = features
            .reshape([mels, chunks, chunk])?
            .permute(&[1, 0, 2])?
            .unsqueeze(1)?;
        for convolution in &self.convolutions {
            hidden = hidden
                .apply(convolution, state)?
                .apply(&Activation::GeluExact, state)?;
        }

        let &[_, channels, frequencies, steps] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("Qwen3-ASR's convolutions"));
        };
        let width = config.width;
        let device = hidden.device();
        let positions = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([steps, width])?,
                self.positions.clone(),
            )
        })?
        .cast(hidden.data_type())?;
        let rows = chunks * steps;
        let hidden = hidden
            .permute(&[0, 3, 1, 2])?
            .contiguous()?
            .reshape([chunks, steps, channels * frequencies])?
            .apply(&self.projection, state)?
            .add(&positions)?
            .reshape([rows, width])?;

        // Every chunk before the last of the clip's is whole, so its valid positions come first: a whole chunk's for each whole chunk of frames, and an eighth of the rest, rounded up, after three convolutions of stride two.
        let frames = frames.cast(DataType::F32)?;
        let whole = frames.div_scalar(chunk as f64)?.floor()?;
        let count = whole.mul_scalar(steps as f64)?.add(
            &frames
                .sub(&whole.mul_scalar(chunk as f64)?)?
                .div_scalar(8.0)?
                .ceil()?,
        )?;

        // Each window attends within itself, as a batch whose windows count only the clip's own positions; a window past them counts one, whose rows mean nothing.
        let window = config.window_positions();
        let windows = (rows - 1) / window + 1;
        let starts = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([windows])?,
                (0..windows).map(|index| (index * window) as f32).collect(),
            )
        })?;
        let lengths = count
            .sub(&starts)?
            .clamp(1.0, window as f32)?
            .cast(DataType::I32)?;
        let mut hidden = hidden
            .pad(&[(0, windows * window - rows), (0, 0)], 0.0)?
            .reshape([windows, window, width])?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: Some(&lengths),
                },
                state,
            )?;
        }

        let [first, second] = &self.projector;
        let rows = hidden
            .reshape([windows * window, width])?
            .narrow(0, 0, rows)?
            .apply(&self.norm, state)?
            .apply(first, state)?
            .apply(&config.activation, state)?
            .apply(second, state)?;
        Ok((rows, count.cast(DataType::I64)?))
    }
}

// Qwen3-ASR's decoder: Qwen3, whose prompt reads the encoder's rows in place of its audio tokens.
pub type Qwen3AsrDecoderLayers =
    SourcedCausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>, RmsNorm>;

// Qwen3-ASR: a Whisper-style log-mel front end at its own clip length, the audio encoder, and a Qwen3 decoder whose prompt reads the encoder's rows, each half with state of its own, and how its snapshot generates.
#[derive(Debug, Clone)]
pub struct Qwen3Asr {
    config: Qwen3AsrConfig,
    encoder: Model<Qwen3AsrEncoderLayers>,
    decoder: Model<Qwen3AsrDecoderLayers>,
    // The snapshot's stop and suppressed tokens, generating nothing until a caller sets `max_new_tokens`.
    generation: GenerationOptions,
}

// Splits Qwen3-ASR's output, `language <name><asr_text>text`, into its language and text, as `transformers`' processor parses it.
#[must_use]
pub fn parse_transcription(output: &str) -> Transcription {
    let Some((prefix, text)) = output.split_once("<asr_text>") else {
        return Transcription {
            language: None,
            text: output.trim().to_owned(),
        };
    };
    let language = prefix
        .lines()
        .rev()
        .find_map(|line| line.trim().strip_prefix("language "))
        .map(str::trim)
        .filter(|language| !language.is_empty() && !language.eq_ignore_ascii_case("none"))
        .map(str::to_owned);
    Transcription {
        language,
        text: text.trim().to_owned(),
    }
}

impl Qwen3Asr {
    #[must_use]
    pub const fn config(&self) -> &Qwen3AsrConfig {
        &self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<Qwen3AsrEncoderLayers> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<Qwen3AsrDecoderLayers> {
        &self.decoder
    }

    // How `transformers`' `generate` continues a prompt for this snapshot, greedily, up to `max_new_tokens`: the generation config's end of text stops a sequence, and its suppressed tokens, if any, are not chosen.
    #[must_use]
    pub fn generation_options(&self, max_new_tokens: usize) -> GenerationOptions {
        GenerationOptions {
            max_new_tokens,
            ..self.generation.clone()
        }
    }

    // The samples of `chunks` chunks of mel frames, which a front end of that many chunks reads.
    #[must_use]
    pub const fn bucket_samples(&self, chunks: i64) -> i64 {
        chunks * self.config.encoder.chunk_frames * HOP
    }

    // The chunks of mel frames a clip of `samples` samples takes, at least one.
    #[must_use]
    pub const fn chunks_of(&self, samples: i64) -> i64 {
        let frames = if samples > MIN_SAMPLES {
            samples
        } else {
            MIN_SAMPLES
        } / HOP;
        let chunk = self.config.encoder.chunk_frames;
        let chunks = (frames + chunk - 1) / chunk;
        if chunks > 1 { chunks } else { 1 }
    }

    // Turns one clip whose first `count` of `samples` are its own, `[N]` with `N` whole chunks' samples and `count` an `I64` `[1]`, into its log-mel features, `[1, mels, F]`, and the count of its own frames, an `I64` `[1]`.
    // A clip shorter than half a second counts as half a second, its own samples then silence, as the feature extractor pads it; frames after the clip's own are zero.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the hop is a small count of samples"
    )]
    pub fn features(&self, samples: &Tensor, count: &Tensor) -> SingeResult<(Tensor, Tensor)> {
        let &[length] = samples.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR reads one clip's samples, [samples]",
            ));
        };
        if length % self.bucket_samples(1) != 0 {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR's front end reads whole chunks' samples",
            ));
        }

        let options = LogMelOptions {
            samples: length,
            ..LogMelOptions::whisper(self.config.encoder.mels)
        };
        let shortest = capture::literal(samples.device(), |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![MIN_SAMPLES])
        })?;
        let count = count.maximum(&shortest)?;
        let features = LogMelSpectrogram::create(options)?
            .counted(samples, &count)?
            .cast(self.encoder.state().floating_type()?)?;
        let frames = count
            .cast(DataType::F32)?
            .div_scalar(HOP as f64)?
            .floor()?
            .cast(DataType::I64)?;
        Ok((features, frames))
    }

    // Encodes one clip's features, `[1, mels, F]`, of which the first `frames`, an `I64` `[1]`, are its own, to rows its prompt's audio tokens read and their count, as `Qwen3AsrEncoderLayers::encode` does.
    pub fn encode(&self, features: &Tensor, frames: &Tensor) -> SingeResult<(Tensor, Tensor)> {
        self.encoder
            .layer()
            .encode(features, frames, self.encoder.state())
    }

    // Encodes one clip of 16 kHz samples, `[N]`, to the rows its prompt's audio tokens read, `[R, W]`, eagerly over as few chunks as it takes.
    pub fn encode_clip(&self, samples: &Tensor) -> SingeResult<Tensor> {
        let &[length] = samples.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR reads one clip's samples, [samples]",
            ));
        };
        let bucket = self.bucket_samples(self.chunks_of(length));
        let padded = samples
            .cast(DataType::F32)?
            .pad(&[(0, bucket - length)], 0.0)?;
        let count = Tensor::from_vec(samples.device(), Shape::create([1])?, vec![length])?;
        let (features, frames) = self.features(&padded, &count)?;
        let (rows, count) = self.encode(&features, &frames)?;
        let count = count.to_vec::<i64>()?[0];
        rows.narrow(0, 0, count)
    }

    // The step that encodes one clip of a bucket of `chunks` chunks into `decoder`'s first sequence's sources, as a graph whose inputs are `samples`, `[N]`, `source_length`, the clip's own samples, `source_position`, where its prompt's audio tokens start, both `I64` `[1]`, and the caches, named as the decoder's steps name them.
    // Its output, `source_rows`, is how many audio tokens the prompt holds.
    // Exported with the decoder's steps as `encode`, it lets a generation loop transcribe samples with nothing of Singe; the decoder's capacity must hold the prompt's start and every row of the bucket.
    pub fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<Qwen3AsrDecoderLayers>,
        chunks: i64,
    ) -> SingeResult<(Graph, HashMap<String, Tensor>)> {
        let device = decoder
            .caches()
            .first()
            .and_then(|cache| {
                cache
                    .named_tensors()
                    .first()
                    .map(|(_, tensor)| tensor.device())
            })
            .ok_or_else(|| SingeError::missing_input("a decoder's caches"))?;
        let samples = self.bucket_samples(chunks);
        let scalar = |value: i64| Tensor::from_vec(device, Shape::create([1])?, vec![value]);
        let inputs = vec![
            (
                "samples".to_owned(),
                Tensor::zeroed_on(device, DataType::F32, Shape::create([samples])?)?,
            ),
            ("source_length".to_owned(), scalar(samples)?),
            ("source_position".to_owned(), scalar(0)?),
        ];
        let (graph, inputs) =
            decoder.encode_step_graph(self.encoder.state(), inputs, |inputs, caches| {
                let [samples, length, position] = inputs else {
                    return Err(SingeError::missing_input("the encode step's inputs"));
                };
                let (features, frames) = self.features(samples, length)?;
                let (rows, count) = self.encode(&features, &frames)?;
                let offsets = capture::literal(device, |device| {
                    Tensor::arange(device, 0_i64, rows.shape().dimensions()[0], 1)
                })?;
                Qwen3AsrDecoderLayers::write_sources_at(caches, 0, &offsets.add(position)?, &rows)?;
                Ok(vec![count])
            })?;
        Ok((graph.with_output_names(["source_rows"])?, inputs))
    }

    // The prompt that asks for a transcription of audio of `audio_tokens` rows, as the snapshot's chat template writes it: the system's turn holds `system`, such as words the clip may say, the user's turn holds the audio, and the assistant's turn starts, given `language <name><asr_text>` when `language` forces one.
    pub fn prompt(
        &self,
        tokenizer: &TextTokenizer,
        audio_tokens: i64,
        language: Option<&str>,
        system: &str,
    ) -> Result<Vec<i64>> {
        let text = |text: &str| tokenizer.encode(text, false);
        let audio =
            usize::try_from(audio_tokens).map_err(|_| SingeError::out_of_range("audio tokens"))?;
        let mut prompt = text(&format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>"
        ))?;
        prompt.extend(iter::repeat_n(self.config.audio_token, audio));
        prompt.extend(text("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?);
        if let Some(language) = language {
            prompt.extend(text(&format!("language {language}<asr_text>"))?);
        }
        Ok(prompt)
    }

    // Generates for clips of 16 kHz samples, each `[N]`: encodes each, then generates after its prompt in `options.rows` rows, each clip joining the running batch as a row frees, with its encoder's rows written to its row's sources first.
    // Returns each clip's generated tokens, in order.
    pub fn generate_clips(
        &self,
        tokenizer: &TextTokenizer,
        clips: &[Tensor],
        language: Option<&str>,
        system: &str,
        options: &BatchOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if clips.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); clips.len()]);
        }

        let mut prompts = Vec::with_capacity(clips.len());
        for clip in clips {
            let encoded = self.encode_clip(clip)?;
            let prompt =
                self.prompt(tokenizer, encoded.shape().dimensions()[0], language, system)?;
            prompts.push((prompt, encoded));
        }
        self.generate(&prompts, options)
    }

    // Generates after each prompt in `options.rows` rows, each prompt joining the running batch as a row frees, with the encoder's rows its audio tokens read, `[N, W]`, written to its row's sources first.
    // Returns each prompt's generated tokens, in order.
    pub fn generate(
        &self,
        prompts: &[(Vec<i64>, Tensor)],
        options: &BatchOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if prompts.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); prompts.len()]);
        }

        let mut decoder = options.decoder(
            &self.decoder,
            prompts.iter().map(|(prompt, _)| prompt.len()),
        )?;

        let audio = self.config.audio_token;
        let tokens = prompts
            .iter()
            .map(|(prompt, _)| prompt.as_slice())
            .collect::<Vec<_>>();
        Ok(generate_batch(
            &mut decoder,
            &tokens,
            &options.generation,
            None,
            |decoder, row, index| {
                let (prompt, encoded) = &prompts[index];
                let first = prompt
                    .iter()
                    .position(|&token| token == audio)
                    .ok_or_else(|| SingeError::invalid_argument("a prompt without audio"))?;
                let first = i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
                Qwen3AsrDecoderLayers::write_sources(decoder.caches(), row, first, encoded)
            },
            |_, _| Ok(()),
        )?)
    }
}

impl SpeechToText for Qwen3Asr {
    fn sample_rate(&self) -> i64 {
        LogMelOptions::whisper(self.config.encoder.mels).sample_rate
    }

    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps> {
        let chunks = limits.audio_seconds.max(1);
        let rows = chunks * self.config.encoder.chunk_positions();
        // The encode step writes a row for every position of its chunks after the prompt's start, which the cache must hold beside the transcription.
        let capacity = limits.capacity.max(rows + 64);
        let decoder = CachedDecoder::create(self.decoder(), DecoderOptions::new(1, capacity))?;
        let (encode, _) = self.encode_step_graph(&decoder, chunks)?;
        LoopSteps::create(&decoder, Some(encode), generation, Some(limits.prompt_rows))
    }

    fn respond(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        instruction: &str,
        options: &BatchOptions,
    ) -> Result<String> {
        let tokens = self
            .generate_clips(
                tokenizer,
                slice::from_ref(samples),
                None,
                instruction,
                &options.with_tokens_of(&self.generation),
            )?
            .pop()
            .ok_or_else(|| SingeError::missing_output("response"))?;
        tokenizer.decode(&tokens, true)
    }

    // Names the language itself unless `language`, its full English name, forces one.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription> {
        let tokens = self
            .generate_clips(
                tokenizer,
                slice::from_ref(samples),
                language,
                "",
                &options.with_tokens_of(&self.generation),
            )?
            .pop()
            .ok_or_else(|| SingeError::missing_output("transcription"))?;
        let output = tokenizer.decode(&tokens, true)?;
        // A forced language is part of the prompt, so the output is the text alone.
        Ok(language.map_or_else(
            || parse_transcription(&output),
            |language| Transcription {
                language: Some(language.to_owned()),
                text: output.trim().to_owned(),
            },
        ))
    }
}

// The encoder's state, under `encoder.` so it shares no name with the decoder's in a graph of both: `convolutions.0` to `convolutions.2`, `convolutions.output`, which projects their channels and frequencies to the width, `blocks.N`, `norm`, and `projector.0` and `projector.1`.
fn encoder_schema(
    config: &Qwen3AsrEncoderConfig,
    data_type: DataType,
) -> Result<Model<Qwen3AsrEncoderLayers>> {
    let (width, channels) = (config.width, config.channels);
    let mut builder = StateBuilder::new();
    let layers = {
        let mut root = Schema::root(&mut builder, data_type);
        let mut model = root.scope("encoder");
        let mut convolutions = model.scope("convolutions");
        let convolution = |schema: &mut Schema<'_>, name: &str, input: i64| {
            schema.conv2d(
                name,
                input,
                channels,
                3,
                Conv2dConfig {
                    stride: (2, 2),
                    padding: (1, 1, 1, 1),
                    ..Conv2dConfig::default()
                },
                true,
            )
        };
        let strided = [
            convolution(&mut convolutions, "0", 1)?,
            convolution(&mut convolutions, "1", channels)?,
            convolution(&mut convolutions, "2", channels)?,
        ];
        let projection =
            convolutions.linear("output", channels * config.frequencies(), width, false)?;
        let spec = PreNormEncoderSpec {
            width,
            heads: config.heads,
            inner: config.inner,
            activation: config.activation,
            eps: LAYER_NORM_EPS,
            key_bias: true,
            causal: false,
        };
        let blocks = (0..config.layers)
            .map(|layer| pre_norm_encoder_block(&mut model.scope(format!("blocks.{layer}")), &spec))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = model.layer_norm("norm", width, LAYER_NORM_EPS)?;
        let mut projector = model.scope("projector");
        let projector = [
            projector.linear("0", width, width, true)?,
            projector.linear("1", width, config.output_width, true)?,
        ];
        Qwen3AsrEncoderLayers {
            convolutions: strided,
            projection,
            blocks,
            norm,
            projector,
            positions: sinusoids(config.chunk_positions(), width),
            config: config.clone(),
        }
    };
    Ok(Model::new(layers, builder.finish()?))
}

// The decoder's state, Qwen3's under `causal_lm`'s names.
fn decoder_schema(
    config: &Qwen3AsrConfig,
    data_type: DataType,
) -> Result<Model<Qwen3AsrDecoderLayers>> {
    let decoder = &config.decoder;
    let (layers, state) = causal_lm(
        decoder,
        data_type,
        CausalLmSpec::new(decoder.rotary(), |_| decoder.kv_cache()),
        |schema, _| {
            pre_norm_block(schema, decoder, |schema| {
                grouped_attention(schema, decoder, causal())
            })
        },
    )?;
    Ok(Model::new(
        SourcedCausalLm::new(layers, config.audio_token, decoder.width),
        state,
    ))
}

pub fn load(loader: &mut Loader<'_>) -> Result<Qwen3Asr> {
    let config: Qwen3AsrConfig = loader.config(transformers::config)?;
    let generation = generation_options(loader.source())?;
    let data_type = loader.options().data_type;
    let encoder = loader.load(
        "encoder",
        &encoder_schema(&config.encoder, data_type)?,
        &transformers::encoder_rules(),
    )?;
    let decoder = loader.load(
        "decoder",
        &decoder_schema(&config, data_type)?,
        &transformers::decoder_rules(),
    )?;
    loader.finish()?;
    Ok(Qwen3Asr {
        config,
        encoder,
        decoder,
        generation,
    })
}
