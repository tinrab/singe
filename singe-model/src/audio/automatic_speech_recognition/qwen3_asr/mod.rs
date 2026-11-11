use std::{iter, path::Path, slice};

use serde::Deserialize;
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, RunningBatch},
    nn::{
        Activation, Conv2d, Conv2dConfig, DenseAttentionInput, GatedMlp, GroupedQueryAttention,
        Layer, LayerNorm, Linear, Mlp, Model, ModelState, PreNormBlock, RmsNorm, SelfAttention,
        SourcedCausalLm, StateBuilder,
    },
    signal::{LogMelOptions, LogMelSpectrogram},
};

use crate::{
    assembly::{
        DecoderSpec, causal, causal_lm_in, grouped_attention, load_weights, pre_norm_block,
    },
    audio::automatic_speech_recognition::whisper::TranscribeOptions,
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
    text::text_generation::qwen3::Qwen3Config,
    tokenization::TextTokenizer,
};

// `torch.nn.LayerNorm`'s default, which every norm of the encoder uses.
const LAYER_NORM_EPS: f32 = 1e-5;

// The shortest clip the feature extractor reads, in samples, which it pads shorter clips to.
const MIN_SAMPLES: i64 = 8_000;

// The fields of the audio encoder's part of a Qwen3-ASR `config.json`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Qwen3AsrEncoderConfig {
    pub num_mel_bins: i64,
    pub d_model: i64,
    pub encoder_layers: i64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub downsample_hidden_size: i64,
    // The positions of one chunk after the convolutions.
    pub max_position_embeddings: i64,
    // Half the mel frames of a chunk.
    pub n_window: i64,
    // The mel frames of an attention window.
    pub n_window_infer: i64,
    pub output_dim: i64,
    pub activation_function: Activation,
    #[serde(default)]
    pub scale_embedding: bool,
}

impl Qwen3AsrEncoderConfig {
    // The mel frames of a chunk, which the convolutions read apart.
    #[must_use]
    pub const fn chunk(&self) -> i64 {
        2 * self.n_window
    }

    // The positions an attention window holds after the convolutions: as many chunks' as fit its mel frames.
    #[must_use]
    pub const fn window(&self) -> i64 {
        downsampled(self.chunk()) * (self.n_window_infer / self.chunk())
    }

    // The frequencies the convolutions leave of the mel bins.
    const fn frequencies(&self) -> i64 {
        downsampled(self.num_mel_bins)
    }

    // The positions the encoder gives `frames` valid mel frames: each chunk's, after three convolutions of stride two.
    #[must_use]
    pub const fn positions(&self, frames: i64) -> i64 {
        let chunk = self.chunk();
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

// The fields of a Hugging Face Qwen3-ASR `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3AsrConfig {
    pub audio_config: Qwen3AsrEncoderConfig,
    pub text_config: Qwen3Config,
    pub audio_token_id: i64,
}

impl Qwen3AsrConfig {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
        let audio = &self.audio_config;
        if audio.scale_embedding {
            return Err(unsupported("scaled audio embeddings"));
        }

        if audio.n_window <= 0 || audio.n_window_infer % audio.chunk() != 0 {
            return Err(unsupported(
                "attention windows that are not whole chunks of mel frames",
            ));
        }

        if downsampled(audio.chunk()) != audio.max_position_embeddings {
            return Err(unsupported("a chunk of other than its positions"));
        }
        self.text_config.validate()
    }
}

type EncoderBlock = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

// Qwen3-ASR's audio encoder: each chunk of mel frames through three convolutions of stride two with GELU, a projection of channels and frequencies, and sinusoid positions that restart each chunk; then the valid positions through pre-norm blocks attending within windows, a norm, and a projector of two linears around GELU into the decoder's width.
#[derive(Debug, Clone)]
pub struct Qwen3AsrEncoderLayers {
    convolutions: [Conv2d; 3],
    projection: Linear,
    blocks: Vec<EncoderBlock>,
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
    // Encodes one clip's log-mel features, `[1, mels, F]` with `F` whole chunks, of which the first `frames` are valid, to `[N, output]`, one row per audio token of its prompt.
    pub fn encode(
        &self,
        features: &Tensor,
        frames: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let config = &self.config;
        let &[1, mels, padded] = features.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR encodes one clip's features, [1, mels, frames]",
            ));
        };
        let chunk = config.chunk();
        if padded % chunk != 0 || frames <= 0 || frames > padded {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR's features are whole chunks, with valid frames among them",
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
        let width = config.d_model;
        let positions = Tensor::from_vec(
            hidden.device(),
            Shape::create([steps, width])?,
            self.positions.clone(),
        )?
        .cast(hidden.data_type())?;
        let valid = config.positions(frames);
        let hidden = hidden
            .permute(&[0, 3, 1, 2])?
            .contiguous()?
            .reshape([chunks, steps, channels * frequencies])?
            .apply(&self.projection, state)?
            .add(&positions)?
            .reshape([chunks * steps, width])?
            .narrow(0, 0, valid)?;

        // Each window attends within itself, as a batch whose last window counts only its own positions.
        let window = config.window();
        let windows = (valid - 1) / window + 1;
        let lengths = (0..windows)
            .map(|index| {
                i32::try_from((valid - index * window).min(window))
                    .map_err(|_| SingeError::out_of_range("attention window"))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let lengths = Tensor::from_vec(hidden.device(), Shape::create([windows])?, lengths)?;
        let mut hidden = hidden
            .pad(&[(0, windows * window - valid), (0, 0)], 0.0)?
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
        hidden
            .reshape([windows * window, width])?
            .narrow(0, 0, valid)?
            .apply(&self.norm, state)?
            .apply(first, state)?
            .apply(&config.activation_function, state)?
            .apply(second, state)
    }
}

// Qwen3-ASR's decoder: Qwen3, whose prompt reads the encoder's rows in place of its audio tokens.
pub type Qwen3AsrDecoderLayers =
    SourcedCausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>, RmsNorm>;

// Qwen3-ASR: a Whisper-style log-mel front end at its own clip length, the audio encoder, and a Qwen3 decoder whose prompt reads the encoder's rows, each half with state of its own.
#[derive(Debug, Clone)]
pub struct Qwen3Asr {
    config: Qwen3AsrConfig,
    encoder: Model<Qwen3AsrEncoderLayers>,
    decoder: Model<Qwen3AsrDecoderLayers>,
}

// What a transcription says: the language the model names, if any, and the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcription {
    pub language: Option<String>,
    pub text: String,
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

    // Turns one clip of 16 kHz samples, `[N]`, into its log-mel features, `[1, mels, F]` padded with zeros to whole chunks, and the count of its valid frames.
    // A clip shorter than half a second is padded with silence first, as the feature extractor pads it.
    pub fn features(&self, samples: &Tensor) -> SingeResult<(Tensor, i64)> {
        let &[length] = samples.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "Qwen3-ASR reads one clip's samples, [samples]",
            ));
        };
        let audio = &self.config.audio_config;
        let options = LogMelOptions {
            samples: length.max(MIN_SAMPLES),
            ..LogMelOptions::whisper(audio.num_mel_bins)
        };
        let frames = options.frames();
        let chunk = audio.chunk();
        let features = LogMelSpectrogram::create(options)?
            .forward(samples, self.encoder.state())?
            .pad(
                &[(0, 0), (0, 0), (0, (chunk - frames % chunk) % chunk)],
                0.0,
            )?
            .cast(self.encoder.state().floating_type()?)?;
        Ok((features, frames))
    }

    // Encodes one clip's features, `[1, mels, F]`, of which the first `frames` are valid, to the rows its prompt's audio tokens read, `[N, W]`.
    pub fn encode(&self, features: &Tensor, frames: i64) -> SingeResult<Tensor> {
        self.encoder
            .layer()
            .encode(features, frames, self.encoder.state())
    }

    // The prompt that asks for a transcription of audio of `audio_tokens` rows, as the snapshot's chat template writes it with no system text: the user's turn holds the audio, and the assistant's turn starts, given `language <name><asr_text>` when `language` forces one.
    pub fn prompt(
        &self,
        tokenizer: &TextTokenizer,
        audio_tokens: i64,
        language: Option<&str>,
    ) -> Result<Vec<i64>> {
        let text = |text: &str| tokenizer.encode(text, false);
        let audio =
            usize::try_from(audio_tokens).map_err(|_| SingeError::out_of_range("audio tokens"))?;
        let mut prompt = text("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
        prompt.extend(iter::repeat_n(self.config.audio_token_id, audio));
        prompt.extend(text("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?);
        if let Some(language) = language {
            prompt.extend(text(&format!("language {language}<asr_text>"))?);
        }
        Ok(prompt)
    }

    // Transcribes one clip of 16 kHz samples, `[N]`, as `transcribe_batch` does a batch of one.
    pub fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &TranscribeOptions,
    ) -> Result<Vec<i64>> {
        self.transcribe_batch(tokenizer, slice::from_ref(samples), language, options)?
            .pop()
            .ok_or_else(|| SingeError::missing_output("transcription").into())
    }

    // Transcribes clips of 16 kHz samples, each `[N]`: encodes each, then generates greedily after its prompt in `options.rows` rows, each clip joining the running batch as a row frees, with its encoder's rows written to its row's sources first.
    // Returns each clip's generated tokens, in order.
    pub fn transcribe_batch(
        &self,
        tokenizer: &TextTokenizer,
        clips: &[Tensor],
        language: Option<&str>,
        options: &TranscribeOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if clips.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); clips.len()]);
        }

        let mut prompts = Vec::with_capacity(clips.len());
        for clip in clips {
            let (features, frames) = self.features(clip)?;
            let encoded = self.encode(&features, frames)?;
            let prompt = self.prompt(tokenizer, encoded.shape().dimensions()[0], language)?;
            prompts.push((prompt, encoded));
        }
        self.generate(&prompts, options)
    }

    // Generates greedily after each prompt in `options.rows` rows, each prompt joining the running batch as a row frees, with the encoder's rows its audio tokens read, `[N, W]`, written to its row's sources first.
    // Returns each prompt's generated tokens, in order.
    pub fn generate(
        &self,
        prompts: &[(Vec<i64>, Tensor)],
        options: &TranscribeOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if prompts.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); prompts.len()]);
        }

        let capacity = prompts
            .iter()
            .map(|(prompt, _)| prompt.len() + options.generation.max_new_tokens)
            .max()
            .unwrap_or(0);
        let capacity =
            i64::try_from(capacity).map_err(|_| SingeError::out_of_range("transcription"))?;
        let mut decoder =
            CachedDecoder::create(&self.decoder, DecoderOptions::new(options.rows, capacity))?;
        if options.compile {
            decoder.compile()?;
        }

        let audio = self.config.audio_token_id;
        let mut tokens = vec![Vec::new(); prompts.len()];
        let mut next = 0;
        let mut batch = RunningBatch::create(&mut decoder, options.generation.clone())?;
        loop {
            while let Some((prompt, encoded)) = prompts.get(next) {
                let first = prompt
                    .iter()
                    .position(|&token| token == audio)
                    .ok_or_else(|| SingeError::invalid_argument("a prompt without audio"))?;
                let first = i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
                let joined = batch.join_with(prompt, |decoder, row| {
                    Qwen3AsrDecoderLayers::write_sources(decoder.caches(), row, first, encoded)
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

fn encoder_block(
    schema: &mut Schema<'_>,
    config: &Qwen3AsrEncoderConfig,
) -> SingeResult<EncoderBlock> {
    let width = config.d_model;
    let mut attention = schema.scope("self_attn");
    let attention = SelfAttention::create(
        attention.linear("q_proj", width, width, true)?,
        attention.linear("k_proj", width, width, true)?,
        attention.linear("v_proj", width, width, true)?,
        attention.linear("out_proj", width, width, true)?,
        config.encoder_attention_heads,
        width,
    )?;
    Ok(PreNormBlock::new(
        schema.layer_norm("self_attn_layer_norm", width, LAYER_NORM_EPS)?,
        attention,
        schema.layer_norm("final_layer_norm", width, LAYER_NORM_EPS)?,
        Mlp::new(
            schema.linear("fc1", width, config.encoder_ffn_dim, true)?,
            schema.linear("fc2", config.encoder_ffn_dim, width, true)?,
            config.activation_function,
        ),
    ))
}

fn encoder_schema(
    config: &Qwen3AsrEncoderConfig,
    data_type: DataType,
) -> Result<Model<Qwen3AsrEncoderLayers>> {
    let (width, hidden) = (config.d_model, config.downsample_hidden_size);
    let mut builder = StateBuilder::new();
    let layers = {
        let mut tower = Schema::new(builder.scope("model.audio_tower"), data_type);
        let convolution = |schema: &mut Schema<'_>, name: &str, input: i64| {
            schema.conv2d(
                name,
                input,
                hidden,
                3,
                Conv2dConfig {
                    stride: (2, 2),
                    padding: (1, 1, 1, 1),
                    ..Conv2dConfig::default()
                },
                true,
            )
        };
        let convolutions = [
            convolution(&mut tower, "conv2d1", 1)?,
            convolution(&mut tower, "conv2d2", hidden)?,
            convolution(&mut tower, "conv2d3", hidden)?,
        ];
        let projection = tower.linear("conv_out", hidden * config.frequencies(), width, false)?;
        let blocks = (0..config.encoder_layers)
            .map(|layer| encoder_block(&mut tower.scope(format!("layers.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = tower.layer_norm("ln_post", width, LAYER_NORM_EPS)?;
        let mut projector = Schema::new(builder.scope("model.multi_modal_projector"), data_type);
        let projector = [
            projector.linear("linear_1", width, width, true)?,
            projector.linear("linear_2", width, config.output_dim, true)?,
        ];
        Qwen3AsrEncoderLayers {
            convolutions,
            projection,
            blocks,
            norm,
            projector,
            positions: sinusoids(config.max_position_embeddings, width),
            config: config.clone(),
        }
    };
    Ok(Model::new(layers, builder.finish()?))
}

fn decoder_schema(
    config: &Qwen3AsrConfig,
    data_type: DataType,
) -> Result<Model<Qwen3AsrDecoderLayers>> {
    let spec = DecoderSpec {
        head_norms: true,
        ..config.text_config.decoder.spec(data_type, true)?
    };
    let (decoder, state) = causal_lm_in(
        "model.language_model",
        &spec,
        spec.rotary(),
        |_| spec.kv_cache(),
        None,
        |schema, _| {
            pre_norm_block(schema, &spec, |schema| {
                grouped_attention(schema, &spec, causal())
            })
        },
    )?;
    Ok(Model::new(
        SourcedCausalLm::new(decoder, config.audio_token_id, spec.width),
        state,
    ))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Qwen3Asr> {
    let snapshot = snapshot.as_ref();
    let config: Qwen3AsrConfig = Snapshot::new(snapshot).config()?;
    config.validate()?;
    let encoder_names = |name: &str| {
        (name.starts_with("model.audio_tower.") || name.starts_with("model.multi_modal_projector."))
            .then(|| name.to_owned())
    };
    let decoder_names = |name: &str| {
        (name.starts_with("model.language_model.") || name == "lm_head.weight")
            .then(|| name.to_owned())
    };
    Ok(Qwen3Asr {
        encoder: load_weights(
            &encoder_schema(&config.audio_config, data_type)?,
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
