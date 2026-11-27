//! CSM: a Llama backbone over text and audio frames, each frame the sum of its 32 codes' embeddings, that predicts a frame's first code, and a small depth decoder that predicts the rest from the backbone's last state, as `CsmForConditionalGeneration.generate` runs them greedily.
//! A frame enters the backbone as a source row at a placeholder token, and the backbone's head gives its last state beside its logits through an identity under `lm_head`.
//! The depth decoder runs each frame anew, densely over its at most 33 positions, and Mimi decodes the frames before the first that is all end codes.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, Decoder, DecoderOptions, PromptChunks},
    nn::{
        CausalLm, CausalLmParts, DenseAttentionInput, Embedding, GatedMlp, GroupedQueryAttention,
        Layer, Linear, Model, ModelState, PreNormBlock, RmsNorm, RotarySpec, SourcedCausalLm,
        StateKey,
    },
};

use crate::{
    assembly::{DecoderConfig, causal, grouped_attention, pre_norm_block},
    decoder_cache::{Decoders, bucketed_capacity, cached_decoder},
    error::{Error, Result},
    format::Loader,
    head::{
        AudioGenerator, HeadLoader, Sampler, Sampling, SamplingRequest, SpeechPrompt, TurnContent,
    },
    model::{
        ModelEntry,
        audio::mimi::{self, MimiConfig, MimiLayers},
    },
    processor::{audio::Audio, text::TextTokenizer},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsmConfig {
    pub backbone: DecoderConfig,
    pub depth: DecoderConfig,
    pub codec: MimiConfig,
    pub codebooks: i64,
    /// The code every codebook of a frame holds once the speech has ended.
    pub end_code: i64,
    /// The token whose embedding a frame's replaces.
    pub audio_token_id: i64,
    pub text_vocabulary: i64,
    /// The most frames a clip takes, from the generation config.
    pub frames: usize,
    /// How the backbone samples each frame's first code, from the generation config.
    pub sampling: Sampling,
    /// How the depth decoder samples each frame's other codes, from the generation config.
    pub depth_sampling: Sampling,
}

type Block = PreNormBlock<GroupedQueryAttention, GatedMlp>;

pub type CsmBackbone = SourcedCausalLm<Block>;

/// The depth decoder and the audio embeddings it shares with the backbone's frames.
#[derive(Debug)]
pub struct CsmDepth {
    /// Every codebook's code embeddings in turn, `[K · V, W]` for the backbone's width.
    audio: StateKey,
    projector: Linear,
    blocks: Vec<Block>,
    norm: RmsNorm,
    /// Each codebook's head after the first, `[K − 1, w, V]`.
    head: StateKey,
    rotary: RotarySpec,
    codebooks: i64,
    vocabulary: i64,
}

impl CsmDepth {
    fn create(schema: &mut Schema<'_>, config: &CsmConfig) -> SingeResult<Self> {
        let depth = &config.depth;
        let backbone = config.backbone.width;
        Ok(Self {
            audio: schema.parameter("audio", [config.codebooks * depth.vocab_size, backbone])?,
            projector: schema.linear("projector", backbone, depth.width, false)?,
            blocks: (0..depth.layers)
                .map(|layer| {
                    pre_norm_block(
                        &mut schema.scope(format!("blocks.{layer}")),
                        depth,
                        |schema| grouped_attention(schema, depth, causal()),
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            norm: schema.rms_norm("norm", depth.width, depth.eps)?,
            head: schema.parameter(
                "head",
                [config.codebooks - 1, depth.width, depth.vocab_size],
            )?,
            rotary: depth.rotary(),
            codebooks: config.codebooks,
            vocabulary: depth.vocab_size,
        })
    }

    /// Returns the embeddings of `codes`, one per codebook from the first, `[N, W]`.
    fn embed(&self, codes: &[i64], like: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let rows = codes
            .iter()
            .zip(0..)
            .map(|(&code, codebook)| code + codebook * self.vocabulary)
            .collect::<Vec<_>>();
        let count = i64::try_from(rows.len()).map_err(|_| SingeError::out_of_range("codes"))?;
        let indices = Tensor::from_vec(like.device(), Shape::create([count])?, rows)?;
        state.tensor(&self.audio)?.index_select(0, &indices)
    }

    /// Returns a frame's embedding for the backbone, `[1, W]`: the sum of its codes' embeddings.
    fn frame(&self, codes: &[i64], like: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.embed(codes, like, state)?.sum(&[0], true)
    }

    /// Returns a frame's codes after its first: each codebook's chosen by `sampler` from the depth decoder, which reads the backbone's last state, `[1, W]`, then the codes before it.
    fn codes(
        &self,
        last: &Tensor,
        first: i64,
        sampler: &mut Sampler,
        state: &ModelState,
    ) -> Result<Vec<i64>> {
        let mut codes = vec![first];
        for codebook in 1..self.codebooks {
            // Position 0 holds the backbone's state and position `p` the code of codebook `p − 1`.
            let inputs = Tensor::concat(&[last.clone(), self.embed(&codes, last, state)?], 0)?;
            let length = inputs.shape().dimensions()[0];
            let mut hidden = inputs.apply(&self.projector, state)?.unsqueeze(0)?;
            let positions = Tensor::range(last.device(), 0_i64, length)?;
            let angles = self.rotary.angles(&positions, hidden.data_type())?;
            for block in &self.blocks {
                hidden = block.forward(
                    &DenseAttentionInput {
                        hidden: &hidden,
                        angles: Some(&angles),
                        key_lengths: None,
                        mask: None,
                    },
                    state,
                )?;
            }

            let logits = hidden
                .narrow(1, length - 1, 1)?
                .squeeze(0)?
                .apply(&self.norm, state)?
                .matmul(&state.tensor(&self.head)?.select(0, codebook - 1)?)?
                .cast(DataType::F32)?
                .to_vec::<f32>()?;
            codes.push(chosen(&logits, sampler)?);
        }
        Ok(codes)
    }
}

/// Returns the code `sampler` chooses from `logits`.
fn chosen(logits: &[f32], sampler: &mut Sampler) -> Result<i64> {
    i64::try_from(sampler.choose(logits)?).map_err(|_| Error::unsupported("a code past `i64`"))
}

/// A prompt's tokens, and the codes of each frame at its placeholders, by position.
struct Prompt {
    tokens: Vec<i64>,
    frames: Vec<(usize, Vec<i64>)>,
}

/// CSM speaking text: its tokenizer, the backbone, the depth decoder, and Mimi.
#[derive(Debug)]
pub struct CsmSpeaker {
    config: CsmConfig,
    tokenizer: TextTokenizer,
    backbone: Model<CsmBackbone>,
    depth: Model<CsmDepth>,
    codec: Model<MimiLayers>,
    /// The backbone's cached decoders, by capacity.
    decoders: Decoders<i64, CachedDecoder<CsmBackbone>>,
}

impl CsmSpeaker {
    /// Returns each frame's codes, as `sampling` asks over the generation config's two samplings, until a frame's codes before the last are all end codes or `frames` frames, after `prompt`'s tokens, whose placeholders hold its frames.
    fn frames(
        &self,
        prompt: &Prompt,
        frames: usize,
        sampling: &SamplingRequest,
    ) -> Result<Vec<Vec<i64>>> {
        let tokens = &prompt.tokens;
        let config = &self.config;
        let vocabulary = config.depth.vocab_size;
        let width = config.backbone.width;
        let (depth, state) = (self.depth.layer(), self.depth.state());
        let capacity = bucketed_capacity(tokens.len() + frames);
        let mut sampler = sampling.sampler(config.sampling);
        // The depth decoder draws from its own sequence of the seed's draws.
        let mut depth_sampler = SamplingRequest {
            seed: sampling.seed.wrapping_add(1),
            ..*sampling
        }
        .sampler(config.depth_sampling);
        self.decoders.run(
            capacity,
            || {
                let options =
                    DecoderOptions::new(1, capacity).with_prompt_chunks(PromptChunks::NONE);
                Ok::<_, Error>(cached_decoder(&self.backbone, options, false)?)
            },
            |decoder| {
                let like = Tensor::zeroed_on(
                    state.device()?,
                    state.floating_type()?,
                    Shape::create([1, width])?,
                )?;
                for (position, codes) in &prompt.frames {
                    let row = depth.frame(codes, &like, state)?;
                    let position = i64::try_from(*position)
                        .map_err(|_| SingeError::out_of_range("prompt position"))?;
                    CsmBackbone::write_sources(decoder.caches(), 0, position, &row)?;
                }

                let mut output = decoder.step(&[tokens.as_slice()], None)?;
                let mut position =
                    i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("prompt"))?;
                let mut generated = Vec::new();
                loop {
                    let scores = output
                        .narrow(1, 0, vocabulary)?
                        .cast(DataType::F32)?
                        .contiguous()?
                        .to_vec::<f32>()?;
                    let last = output.narrow(1, vocabulary, width)?.contiguous()?;
                    let first = chosen(&scores, &mut sampler)?;
                    let codes = depth.codes(&last, first, &mut depth_sampler, state)?;
                    let ended = codes[..codes.len() - 1]
                        .iter()
                        .all(|&code| code == config.end_code);
                    generated.push(codes);
                    if ended || generated.len() == frames {
                        break;
                    }

                    let row = depth.frame(&generated[generated.len() - 1], &last, state)?;
                    CsmBackbone::write_sources(decoder.caches(), 0, position, &row)?;
                    output = decoder.step(&[&[config.audio_token_id]], None)?;
                    position += 1;
                }
                Ok(generated)
            },
        )
    }

    /// Returns the prompt CSM's chat template writes: each turn `<bos>[speaker]text<eos>`, followed where it has audio by a placeholder per Mimi frame of the clip and one for the frame of end codes, as `CsmForConditionalGeneration` embeds `<|AUDIO|>` and `<|audio_eos|>`.
    /// Text alone is speaker 0's turn unless it names a speaker.
    fn prompt(&self, prompt: &SpeechPrompt) -> Result<Prompt> {
        let config = &self.config;
        let turns = match prompt {
            SpeechPrompt::Text(text) => {
                let text = if text.starts_with('[') {
                    text.clone()
                } else {
                    format!("[0]{text}")
                };
                return Ok(Prompt {
                    tokens: self.tokenizer.encode(&text, true)?,
                    frames: Vec::new(),
                });
            }
            SpeechPrompt::Conversation(turns) => turns,
        };
        let mut tokens = Vec::new();
        let mut frames = Vec::new();
        let codebooks = usize::try_from(config.codebooks)
            .map_err(|_| Error::invalid("a negative codebook count"))?;
        for (index, turn) in turns.iter().enumerate() {
            let (mut texts, mut clips) = (Vec::new(), Vec::new());
            for content in &turn.content {
                match content {
                    TurnContent::Text { text } => texts.push(text),
                    TurnContent::Audio { audio } => clips.push(audio),
                }
            }

            let is_last = index + 1 == turns.len();
            if texts.len() != 1 || clips.len() > 1 || (!is_last && clips.is_empty()) {
                return Err(Error::invalid(
                    "a CSM turn other than one text and, before the last turn, one clip",
                ));
            }

            if turn.role.parse::<u32>().is_err() {
                return Err(Error::invalid("a CSM speaker other than an integer"));
            }

            tokens.extend(
                self.tokenizer
                    .encode(&format!("[{}]{}", turn.role, texts[0]), true)?,
            );
            if let Some(clip) = clips.first() {
                for codes in self.clip_frames(clip)? {
                    frames.push((tokens.len(), codes));
                    tokens.push(config.audio_token_id);
                }

                frames.push((tokens.len(), vec![config.end_code; codebooks]));
                tokens.push(config.audio_token_id);
            }
        }
        Ok(Prompt { tokens, frames })
    }

    /// Returns each Mimi frame's codes of `clip`, at the codec's rate.
    fn clip_frames(&self, clip: &Audio) -> Result<Vec<Vec<i64>>> {
        let rate = u32::try_from(self.config.codec.sample_rate)
            .map_err(|_| Error::invalid("a codec sample rate out of range"))?;
        let (layers, state) = (self.codec.layer(), self.codec.state());
        let samples = clip
            .resampled(rate)
            .to_tensor(state.device()?)?
            .cast(state.floating_type()?)?
            .unsqueeze(0)?;
        let codes = layers.encode_codes(&samples, state)?;
        let &[_, codebooks, count] = codes.shape().dimensions() else {
            return Err(Error::invalid("Mimi codes, [1, K, T]"));
        };
        let values = codes.contiguous()?.to_vec::<i64>()?;
        let (codebooks, count) = (
            usize::try_from(codebooks).map_err(|_| Error::invalid("a codebook count"))?,
            usize::try_from(count).map_err(|_| Error::invalid("a frame count"))?,
        );
        Ok((0..count)
            .map(|frame| {
                (0..codebooks)
                    .map(|codebook| values[codebook * count + frame])
                    .collect()
            })
            .collect())
    }
}

impl AudioGenerator for CsmSpeaker {
    fn generate_audio(
        &self,
        prompt: &SpeechPrompt,
        max_new_tokens: Option<usize>,
        sampling: &SamplingRequest,
    ) -> Result<Vec<f32>> {
        let config = &self.config;
        let prompt = self.prompt(prompt)?;
        let mut frames = self.frames(&prompt, max_new_tokens.unwrap_or(config.frames), sampling)?;
        // `generate` decodes the frames before the first whose codes are all end codes.
        if let Some(end) = frames
            .iter()
            .position(|codes| codes.iter().all(|&code| code == config.end_code))
        {
            frames.truncate(end);
        }

        let count = i64::try_from(frames.len()).map_err(|_| SingeError::out_of_range("frames"))?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let codebooks = usize::try_from(config.codebooks)
            .map_err(|_| Error::invalid("a negative codebook count"))?;
        let codes = (0..codebooks)
            .flat_map(|codebook| frames.iter().map(move |frame| frame[codebook]))
            .collect::<Vec<_>>();
        let device = self.codec.state().device()?;
        let codes = Tensor::from_vec(device, Shape::create([1, config.codebooks, count])?, codes)?;
        Ok(self
            .codec
            .layer()
            .decode_codes(&codes, self.codec.state())?
            .cast(DataType::F32)?
            .contiguous()?
            .to_vec::<f32>()?)
    }

    fn sample_rate(&self) -> i64 {
        self.config.codec.sample_rate
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "csm",
    formats: &[(Format::Transformers, &["csm"])],
    loaders: &[HeadLoader::AudioGeneration(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

/// Loads the backbone, as part `backbone`, the depth decoder with the audio embeddings, as part `depth`, and Mimi, as part `codec`.
pub fn load(loader: &mut Loader<'_>) -> Result<CsmSpeaker> {
    let config: CsmConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let backbone_config = &config.backbone;
    let backbone = Schema::model(data_type, |schema| {
        let width = backbone_config.width;
        let table =
            schema.parameter("embeddings.tokens.weight", [config.text_vocabulary, width])?;
        let blocks = (0..backbone_config.layers)
            .map(|layer| {
                pre_norm_block(
                    &mut schema.scope(format!("blocks.{layer}")),
                    backbone_config,
                    |schema| grouped_attention(schema, backbone_config, causal()),
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = schema.rms_norm("norm", width, backbone_config.eps)?;
        let head = schema.parameter("head.weight", [backbone_config.vocab_size + width, width])?;
        let caches = (0..blocks.len())
            .map(|_| backbone_config.kv_cache())
            .collect();
        Ok::<_, SingeError>(SourcedCausalLm::new(
            CausalLm::new(CausalLmParts {
                embedding: Embedding::new(table),
                embedding_scale: None,
                logit_scale: None,
                logit_cap: None,
                rotary: Some(backbone_config.rotary()),
                blocks,
                norm,
                head: Linear::new(head, None),
                caches,
            }),
            config.audio_token_id,
            width,
        ))
    })?;
    let backbone = loader.load("backbone", &backbone, &transformers::backbone_rules())?;
    let depth = Schema::model(data_type, |schema| CsmDepth::create(schema, &config))?;
    let depth = loader.load("depth", &depth, &transformers::depth_rules())?;
    let codec = Schema::model(data_type, |schema| {
        MimiLayers::create(&mut schema.scope("codec_model"), &config.codec)
    })?;
    let codec = loader.load(
        "codec",
        &codec,
        &mimi::transformers::rules(&config.codec, "codec_model"),
    )?;
    loader.finish()?;
    Ok(CsmSpeaker {
        tokenizer: TextTokenizer::load(loader.source())?,
        config,
        backbone,
        depth,
        codec,
        decoders: Decoders::default(),
    })
}
