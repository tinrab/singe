//! Qwen3-ASR: an audio encoder of log-mel features and a Qwen3 decoder whose prompt reads the encoder's rows, each with state of its own, `encoder` and `decoder`.

mod transformers;

use std::{collections::HashMap, iter, slice};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, generate_batch},
    graph::{Graph, capture},
    nn::{
        Activation, Conv2d, GatedMlp, GroupedQueryAttention, Linear, Mlp, ModelState, PreNormBlock,
        RmsNorm, SourcedCausalLm, TensorGroup,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    BatchOptions,
    assembly::{
        DecoderConfig, FinalNorm, PYTORCH_NORM_EPS, PreNormEncoder, PreNormEncoderConfig,
        SinusoidArithmetic, SinusoidOrder, causal, grouped_decoder, same_2d, sinusoids,
    },
    error::Result,
    format::Loader,
    head::{HeadLoader, LoopLimits, LoopSteps, SpeechToText, Transcription},
    model::{ModelEntry, audio::speech::SpeechModel},
    processor::text::TextTokenizer,
    schema::Schema,
    source::Format,
};

/// The shortest clip the feature extractor reads, in samples, which it pads shorter clips to.
const MIN_SAMPLES: i64 = 8_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen3AsrEncoderConfig {
    pub mels: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    /// The channels of each convolution.
    pub channels: i64,
    /// The mel frames of a chunk, which the convolutions read apart.
    pub chunk_frames: i64,
    /// The mel frames of an attention window, a whole number of chunks.
    pub window_frames: i64,
    /// The width of the projector's rows, which the decoder reads.
    pub output_width: i64,
    pub activation: Activation,
}

impl Qwen3AsrEncoderConfig {
    /// Returns how many positions a chunk gives after the convolutions.
    #[must_use]
    pub const fn chunk_positions(&self) -> i64 {
        downsampled(self.chunk_frames)
    }

    /// Returns how many positions an attention window holds after the convolutions: those of as many chunks as fit its mel frames.
    #[must_use]
    pub const fn window_positions(&self) -> i64 {
        self.chunk_positions() * (self.window_frames / self.chunk_frames)
    }

    /// Returns how many frequencies the convolutions leave of the mel bins.
    const fn frequencies(&self) -> i64 {
        downsampled(self.mels)
    }

    /// Returns how many positions the encoder gives `frames` valid mel frames: each chunk's, after three convolutions of stride two.
    #[must_use]
    pub const fn positions(&self, frames: i64) -> i64 {
        let chunk = self.chunk_frames;
        frames / chunk * downsampled(chunk) + downsampled(frames % chunk)
    }

    #[must_use]
    pub const fn log_mel(&self) -> LogMelOptions {
        LogMelOptions::whisper(self.mels)
    }

    const fn encoder(&self) -> PreNormEncoderConfig {
        PreNormEncoderConfig {
            width: self.width,
            heads: self.heads,
            inner: self.inner,
            activation: self.activation,
            eps: PYTORCH_NORM_EPS,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        }
    }
}

/// Returns what three 3 × 3 convolutions of stride two and padding one leave of `length`, and zero of zero.
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
    /// The prompt's placeholder token, in whose place the decoder reads the encoder's rows.
    pub audio_token: i64,
}

/// Qwen3-ASR's audio encoder.
/// Each chunk of mel frames goes through three convolutions of stride two with GELU, a projection of channels and frequencies, and sinusoid positions that restart each chunk.
/// Then pre-norm blocks attend within windows of the clip's own positions, followed by a norm and a projector of two linears around the config's activation into the decoder's width.
#[derive(Debug, Clone)]
pub struct Qwen3AsrEncoderLayers {
    convolutions: [Conv2d; 3],
    projection: Linear,
    encoder: PreNormEncoder,
    projector: Mlp,
    config: Qwen3AsrEncoderConfig,
    /// The sinusoid positions of a chunk, `[positions, width]`, row-major.
    positions: Vec<f32>,
}

impl Qwen3AsrEncoderLayers {
    /// Creates the encoder's state under `encoder.`, so it shares no name with the decoder's in a graph of both: `convolutions.0` to `convolutions.2`, `convolutions.output`, which projects their channels and frequencies to the width, `blocks.N`, `norm`, and `projector`.
    pub fn create(schema: &mut Schema<'_>, config: &Qwen3AsrEncoderConfig) -> SingeResult<Self> {
        let mut schema = schema.scope("encoder");
        let (width, channels) = (config.width, config.channels);
        let mut strided =
            |name: &str, input: i64| schema.conv2d(name, input, channels, 3, same_2d(3, 2), true);
        let convolutions = [
            strided("convolutions.0", 1)?,
            strided("convolutions.1", channels)?,
            strided("convolutions.2", channels)?,
        ];
        Ok(Self {
            convolutions,
            projection: schema.linear(
                "convolutions.output",
                channels * config.frequencies(),
                width,
                false,
            )?,
            encoder: PreNormEncoder::create(
                &mut schema,
                &config.encoder(),
                config.layers,
                FinalNorm::Included,
            )?,
            projector: schema.mlp(
                "projector",
                [width, width, config.output_width],
                config.activation,
                true,
            )?,
            positions: positions(config.chunk_positions(), width)?,
            config: config.clone(),
        })
    }

    /// Encodes one clip's log-mel features, `[1, mels, F]` with `F` whole chunks, of which the first `frames`, an `I64` `[1]`, are its own.
    /// Returns a row per position of every chunk, `[F / chunk · positions, output]`, and `count`, an `I64` `[1]`: the first `count` rows are the ones the prompt's audio tokens read, and the rest mean nothing.
    /// The valid positions and each window's length are computed from `frames` in the graph, so one captured encoder serves every clip of up to `F` frames.
    ///
    /// # Errors
    ///
    /// Returns an error if `features` is not one clip's `[1, mels, F]`, or if `F` is not a whole number of chunks.
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
        let hidden = hidden
            .pad(&[(0, windows * window - rows), (0, 0)], 0.0)?
            .reshape([windows, window, width])?;
        let rows = self
            .encoder
            .encode(&hidden, Some(&lengths), state)?
            .reshape([windows * window, width])?
            .narrow(0, 0, rows)?
            .apply(&self.projector, state)?;
        Ok((rows, count.cast(DataType::I64)?))
    }
}

/// Qwen3-ASR's decoder: Qwen3, whose prompt reads the encoder's rows in place of its audio tokens.
pub type Qwen3AsrDecoderLayers =
    SourcedCausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>, RmsNorm>;

/// Qwen3-ASR: a Whisper-style log-mel front end at its own clip length, the audio encoder, a Qwen3 decoder whose prompt reads the encoder's rows, and how its snapshot generates.
pub type Qwen3Asr = SpeechModel<Qwen3AsrConfig, Qwen3AsrEncoderLayers, Qwen3AsrDecoderLayers>;

/// Splits Qwen3-ASR's output, `language <name><asr_text>text`, into its language and text.
/// The language is the last line before `<asr_text>` that starts with `language `, and an empty name or `None` gives no language.
/// An output without `<asr_text>` is all text.
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
    /// Returns how many samples `chunks` chunks of mel frames take, which a front end of that many chunks reads.
    #[must_use]
    pub const fn bucket_samples(&self, chunks: i64) -> i64 {
        let encoder = &self.config().encoder;
        chunks * encoder.chunk_frames * encoder.log_mel().hop
    }

    /// Returns how many chunks of mel frames a clip of `samples` samples takes, at least one, counting a clip shorter than `MIN_SAMPLES` as that long.
    #[must_use]
    pub const fn chunks_of(&self, samples: i64) -> i64 {
        let encoder = &self.config().encoder;
        let frames = if samples > MIN_SAMPLES {
            samples
        } else {
            MIN_SAMPLES
        } / encoder.log_mel().hop;
        let chunk = encoder.chunk_frames;
        let chunks = (frames + chunk - 1) / chunk;
        if chunks > 1 { chunks } else { 1 }
    }

    /// Returns the log-mel features, `[1, mels, F]`, of one clip whose first `count` of `samples` are its own, and the count of its own frames, an `I64` `[1]`.
    /// `samples` is `[N]` with `N` whole chunks' samples, and `count` is an `I64` `[1]`.
    /// A clip shorter than half a second counts as half a second, its own samples then silence, as the feature extractor pads it; frames after the clip's own are zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` is not one-dimensional, or if its length is not a whole number of chunks' samples.
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
            ..self.config().encoder.log_mel()
        };
        let shortest = capture::literal(samples.device(), |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![MIN_SAMPLES])
        })?;
        let count = count.maximum(&shortest)?;
        let features = LogMelSpectrogram::create(options)?
            .counted(samples, &count)?
            .cast(self.encoder().state().floating_type()?)?;
        let frames = count
            .cast(DataType::F32)?
            .div_scalar(options.hop as f64)?
            .floor()?
            .cast(DataType::I64)?;
        Ok((features, frames))
    }

    /// Encodes one clip's features, `[1, mels, F]`, of which the first `frames`, an `I64` `[1]`, are its own, to rows its prompt's audio tokens read and their count, as [`Qwen3AsrEncoderLayers::encode`] does.
    ///
    /// # Errors
    ///
    /// Returns an error if `features` is not one clip's `[1, mels, F]`, or if `F` is not a whole number of chunks.
    pub fn encode(&self, features: &Tensor, frames: &Tensor) -> SingeResult<(Tensor, Tensor)> {
        self.encoder()
            .layer()
            .encode(features, frames, self.encoder().state())
    }

    /// Encodes one clip of 16 kHz samples, `[N]`, to the rows its prompt's audio tokens read, `[R, W]`, eagerly over as few chunks as it takes.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` is not one-dimensional.
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
        let &[count] = count.to_vec::<i64>()?.as_slice() else {
            return Err(SingeError::invalid_argument("Qwen3-ASR's row count"));
        };
        rows.narrow(0, 0, count)
    }

    /// Returns the step that encodes one clip of a bucket of `chunks` chunks into `decoder`'s first sequence's sources, and its example inputs.
    /// The step is a graph whose inputs are `samples`, `[N]`, `source_length`, the clip's own samples, `source_position`, where its prompt's audio tokens start, both `I64` `[1]`, and the caches, named as the decoder's steps name them.
    /// Its output, `source_rows`, is how many audio tokens the prompt holds.
    /// Exported with the decoder's steps as `encode`, it lets a generation loop transcribe samples without Singe; the decoder's capacity must hold the prompt's start and every row of the bucket.
    ///
    /// # Errors
    ///
    /// Returns an error if `decoder` has no caches.
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
            decoder.encode_step_graph(self.encoder().state(), inputs, |inputs, caches| {
                let [samples, length, position] = inputs else {
                    return Err(SingeError::missing_input("the encode step's inputs"));
                };
                let (features, frames) = self.features(samples, length)?;
                let (rows, count) = self.encode(&features, &frames)?;
                let &[row_count, _] = rows.shape().dimensions() else {
                    return Err(SingeError::invalid_argument("Qwen3-ASR's encoded rows"));
                };
                let offsets =
                    capture::literal(device, |device| Tensor::arange(device, 0_i64, row_count, 1))?;
                Qwen3AsrDecoderLayers::write_sources_at(caches, 0, &offsets.add(position)?, &rows)?;
                Ok(vec![count])
            })?;
        Ok((graph.with_output_names(["source_rows"])?, inputs))
    }

    /// Returns the prompt that asks for a transcription of audio of `audio_tokens` rows, as the snapshot's chat template writes it.
    /// The system's turn holds `system`, such as words the clip may say, the user's turn holds the audio, and the assistant's turn starts, given `language <name><asr_text>` when `language` forces one.
    ///
    /// # Errors
    ///
    /// Returns an error if `audio_tokens` is negative.
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
        prompt.extend(iter::repeat_n(self.config().audio_token, audio));
        prompt.extend(text("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?);
        if let Some(language) = language {
            prompt.extend(text(&format!("language {language}<asr_text>"))?);
        }
        Ok(prompt)
    }

    /// Generates for clips of 16 kHz samples, each `[N]`: encodes each, then generates after its prompt in `options.rows` rows, each clip joining the running batch as a row frees, with its encoder's rows written to its row's sources first.
    /// Returns each clip's generated tokens, in order.
    pub fn generate(
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
            let &[rows, _] = encoded.shape().dimensions() else {
                return Err(SingeError::invalid_argument("Qwen3-ASR's encoded rows").into());
            };
            prompts.push((self.prompt(tokenizer, rows, language, system)?, encoded));
        }
        self.generate_prompts(&prompts, options)
    }

    /// Generates after each prompt in `options.rows` rows, each prompt joining the running batch as a row frees, with the encoder's rows its audio tokens read, `[N, W]`, written to its row's sources first.
    /// Returns each prompt's generated tokens, in order.
    fn generate_prompts(
        &self,
        prompts: &[(Vec<i64>, Tensor)],
        options: &BatchOptions,
    ) -> Result<Vec<Vec<i64>>> {
        options.run_decoder(
            self.decoders(),
            self.decoder(),
            options.decoder_options(prompts.iter().map(|(prompt, _)| prompt.len())),
            |decoder| {
                let audio = self.config().audio_token;
                let tokens = prompts
                    .iter()
                    .map(|(prompt, _)| prompt.as_slice())
                    .collect::<Vec<_>>();
                Ok(generate_batch(
                    decoder,
                    &tokens,
                    &options.generation,
                    None,
                    |decoder, row, index| {
                        let (prompt, encoded) = &prompts[index];
                        let first =
                            prompt
                                .iter()
                                .position(|&token| token == audio)
                                .ok_or_else(|| {
                                    SingeError::invalid_argument("a prompt without audio")
                                })?;
                        let first =
                            i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
                        Qwen3AsrDecoderLayers::write_sources(decoder.caches(), row, first, encoded)
                    },
                    |_, _| Ok(()),
                )?)
            },
        )
    }
}

impl SpeechToText for Qwen3Asr {
    fn sample_rate(&self) -> i64 {
        self.config().encoder.log_mel().sample_rate
    }

    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps> {
        let chunks = limits.audio_seconds.max(1);
        let rows = chunks * self.config().encoder.chunk_positions();
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
            .generate(
                tokenizer,
                slice::from_ref(samples),
                None,
                instruction,
                &options.including_tokens_of(self.generation()),
            )?
            .pop()
            .ok_or_else(|| SingeError::missing_output("response"))?;
        tokenizer.decode(&tokens, true)
    }

    /// Transcribes one clip, naming its language itself unless `language`, the language's full English name, forces one.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription> {
        let tokens = self
            .generate(
                tokenizer,
                slice::from_ref(samples),
                language,
                "",
                &options.including_tokens_of(self.generation()),
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

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3_asr",
    formats: &[(Format::Transformers, &["qwen3_asr"])],
    loaders: &[HeadLoader::SpeechToText(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Qwen3Asr> {
    let config: Qwen3AsrConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        Qwen3AsrEncoderLayers::create(schema, &config.encoder)
    })?;
    let decoder = Schema::model(data_type, |schema| {
        grouped_decoder(schema, &config.decoder, causal())
            .map(|layers| SourcedCausalLm::new(layers, config.audio_token, config.decoder.width))
    })?;
    SpeechModel::load(
        loader,
        config,
        (&encoder, &transformers::encoder_rules()),
        (&decoder, &transformers::decoder_rules()),
    )
}

/// Returns the sinusoids of `length` positions over `channels` features, sines then cosines, as Qwen3-ASR's `SinusoidsPositionEmbedding` computes them in `F64`.
fn positions(length: i64, channels: i64) -> SingeResult<Vec<f32>> {
    let width = usize::try_from(channels).map_err(|_| SingeError::out_of_range("channels"))?;
    Ok((0..length)
        .flat_map(|position| {
            sinusoids(
                position as f64,
                width,
                SinusoidOrder::SinesFirst,
                SinusoidArithmetic::Double,
            )
        })
        .collect())
}
