//! Janus: a `SigLIP`-style vision tower whose aligned patches a Llama decoder reads in place of its image tokens, and the same decoder generating an image as a grid of VQ codes.
//! An image's codes are generated under classifier-free guidance, greedily unless sampling is asked for: a second row reads the prompt with every token but the first and the image start padded, and each code is chosen from the unguided row's logits moved `guidance_scale` times their distance toward the guided row's. A code enters the decoder through its own embedding and aligner, and the decoder's last state leaves through the generation head; the VQ decoder turns the grid into pixels.

mod transformers;
pub mod vq;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{
        CachedDecoder, Decoder, DecoderOptions, GenerationOptions, PromptChunking, generate_batch,
    },
    graph::capture,
    nn::{
        Activation, CacheSize, CausalLm, CausalLmParts, DecodeStep, DecoderModel, Embedding,
        GatedMlp, GroupedQueryAttention, KvCache, Layer, Linear, Model, ModelState, PatchTokens,
        PreNormBlock, RmsNorm, RotaryTable, SourcedCausalLm,
    },
};

use self::vq::{JanusVqConfig, VqDecoder};
use crate::{
    BatchOptions,
    assembly::{
        DecoderConfig, FinalNorm, LinearStack, PreNormEncoder, VisionTowerConfig, causal,
        grouped_attention, grouped_decoder, patch_embedding, pre_norm_block,
    },
    decoder_cache::{DecoderCache, DecoderKey, bucketed_capacity, cached_decoder},
    error::{Error, Result},
    format::Loader,
    head::{
        HeadLoader, ImageGenerator, ImageTextToText, LoopLimits, LoopStepSet, Sampler, Sampling,
        SamplingRequest,
    },
    model::ModelEntry,
    processor::{
        chat::{ChatTemplate, Content, Message, Part},
        image::{Image, ImagePreprocessor, Normalization},
        text::TextTokenizer,
    },
    schema::Schema,
    source::Format,
};

/// What `JanusProcessor` puts before a prompt for text when its config asks for the default system prompt.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JanusVisionConfig {
    pub tower: VisionTowerConfig,
    /// The aligner's linear layers.
    pub aligner_depth: i64,
    /// The width of the aligner's hidden layers.
    pub aligner_width: i64,
    pub activation: Activation,
}

/// How Janus generates an image's codes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JanusGenerationConfig {
    /// The width of the generation aligner's and head's hidden layers.
    pub projection: i64,
    pub aligner_depth: i64,
    pub activation: Activation,
    /// The codes an image takes.
    pub tokens: usize,
    pub guidance_scale: f32,
    /// How the generation config samples each code.
    pub sampling: Sampling,
    pub bos_token_id: i64,
    pub pad_token_id: i64,
}

/// How `JanusProcessor` writes a prompt for text: an optional system prompt first, and each image placeholder expanded to its image tokens between the image's start and end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JanusPrompt {
    pub system_prompt: bool,
    pub image_tokens: usize,
    pub image_token: String,
    pub boi_token: String,
    pub eoi_token: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JanusConfig {
    pub vision: JanusVisionConfig,
    pub text: DecoderConfig,
    pub vq: JanusVqConfig,
    pub generation: JanusGenerationConfig,
    pub image_token_id: i64,
    pub prompt: JanusPrompt,
}

/// The vision tower and aligner: from images, `[N, C, S, S]`, to the rows the decoder reads in place of their image tokens, `[N, (S / P)², W]`.
#[derive(Debug, Clone)]
pub struct JanusVision {
    embeddings: PatchTokens,
    encoder: PreNormEncoder,
    aligner: LinearStack,
}

impl JanusVision {
    /// Registers `embeddings` (`patches`, `positions`), `blocks`, `norm`, and `aligner`, which projects to `out_width`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &JanusVisionConfig,
        out_width: i64,
    ) -> SingeResult<Self> {
        let tower = &config.tower;
        let width = tower.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patches =
                patch_embedding(&mut schema, tower.channels, width, tower.patch_size, true)?;
            PatchTokens::new(patches)
                .with_positions(schema.parameter("positions", [1, tower.patches(), width])?)
        };
        let encoder =
            PreNormEncoder::create(schema, &tower.encoder, tower.layers, FinalNorm::Included)?;
        let aligner = LinearStack::create(
            &mut schema.scope("aligner"),
            [width, config.aligner_width, out_width],
            config.aligner_depth,
            config.activation,
        )?;
        Ok(Self {
            embeddings,
            encoder,
            aligner,
        })
    }
}

impl Layer for JanusVision {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(&input.apply(&self.embeddings, state)?, state)?
            .apply(&self.aligner, state)
    }
}

pub type JanusDecoder = SourcedCausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>>;

/// Janus reading images: the processor's transforms and prompt settings, the vision tower, and the decoder, each with state of its own.
#[derive(Debug, Clone)]
pub struct Janus {
    config: JanusConfig,
    vision: Model<JanusVision>,
    decoder: Model<JanusDecoder>,
    processor: ImagePreprocessor,
    device: Device,
    decoders: DecoderCache<DecoderKey, CachedDecoder<JanusDecoder>>,
}

impl Janus {
    /// Returns `prompt` as `JanusProcessor` writes it for text from `images` images.
    fn expanded(&self, prompt: &str, images: usize) -> Result<String> {
        let settings = &self.config.prompt;
        let placeholders = prompt.matches(settings.image_token.as_str()).count();
        if placeholders != images {
            return Err(Error::invalid(format!(
                "a prompt of {placeholders} image placeholders for {images} images"
            )));
        }

        let expansion = format!(
            "{}{}{}",
            settings.boi_token,
            settings.image_token.repeat(settings.image_tokens),
            settings.eoi_token
        );
        let prompt = prompt.replace(&settings.image_token, &expansion);
        Ok(if settings.system_prompt {
            format!("{DEFAULT_SYSTEM_PROMPT}{prompt}")
        } else {
            prompt
        })
    }

    /// Generates after `tokens`, writing `rows`, `[N, W]`, in place of its image tokens in order.
    fn generate_tokens(
        &self,
        tokens: &[i64],
        rows: &Tensor,
        options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        let image = self.config.image_token_id;
        let places = tokens
            .iter()
            .enumerate()
            .filter(|&(_, &token)| token == image)
            .map(|(place, _)| i64::try_from(place).map_err(|_| SingeError::out_of_range("prompt")))
            .collect::<SingeResult<Vec<_>>>()?;
        let &[count, _] = rows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("an image's rows").into());
        };
        if i64::try_from(places.len()).ok() != Some(count) {
            return Err(SingeError::invalid_argument(format!(
                "a prompt of {} image tokens for {count} image rows",
                places.len()
            ))
            .into());
        }

        let places = Tensor::from_vec(self.device, Shape::create([count])?, places)?;
        let mut generated = options.run_decoder(
            &self.decoders,
            &self.decoder,
            options.decoder_options(iter::once(tokens.len())),
            |decoder| {
                generate_batch(
                    decoder,
                    &[tokens],
                    &options.generation,
                    None,
                    |decoder, row, _| {
                        JanusDecoder::write_sources_at(decoder.caches(), row, &places, rows)
                    },
                    |_, _| Ok(()),
                )
            },
        )?;
        Ok(generated.pop().unwrap_or_default())
    }
}

impl ImageTextToText for Janus {
    fn loop_steps(
        &self,
        limits: &LoopLimits,
        generation: &GenerationOptions,
    ) -> Result<LoopStepSet> {
        LoopStepSet::one_sequence(&self.decoder, limits, generation, |_| Ok(None))
    }

    fn image_token(&self) -> i64 {
        self.config.image_token_id
    }

    fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Image],
        options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        let text = self.expanded(prompt, images.len())?;
        let data_type = self.vision.state().floating_type()?;
        let pixels = self
            .processor
            .preprocess_batch(images, self.device, data_type)?;
        let rows = self.vision.forward(&pixels)?;
        let &[count, per_image, width] = rows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the vision tower's rows").into());
        };
        let rows = rows.reshape([count * per_image, width])?;
        self.generate_tokens(&tokenizer.encode(&text, true)?, &rows, options)
    }
}

/// The decoder's final norm followed by the generation head's first projection and activation, so that the decoder's head is the generation head's last layer.
#[derive(Debug, Clone)]
pub struct GenerationNorm {
    norm: RmsNorm,
    projection: Linear,
    activation: Activation,
}

impl Layer for GenerationNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.norm, state)?
            .apply(&self.projection, state)?
            .apply(&self.activation, state)
    }
}

type ImageLm = CausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>, GenerationNorm>;

/// The decoder generating an image's codes: a prompt's text tokens embedded as text, and code `c`, fed as token `V + c` for a text vocabulary of `V`, through the generation embedding and aligner.
#[derive(Debug)]
pub struct JanusImageDecoder {
    lm: ImageLm,
    text: Embedding,
    codes: Embedding,
    aligner: LinearStack,
    vocabulary: i64,
}

impl JanusImageDecoder {
    pub fn create(schema: &mut Schema<'_>, config: &JanusConfig) -> SingeResult<Self> {
        let text = &config.text;
        let generation = &config.generation;
        let table = schema.parameter("embeddings.tokens.weight", [text.vocab_size, text.width])?;
        let blocks = (0..text.layers)
            .map(|layer| {
                pre_norm_block(
                    &mut schema.scope(format!("blocks.{layer}")),
                    text,
                    |schema| grouped_attention(schema, text, causal()),
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = GenerationNorm {
            norm: schema.rms_norm("norm.norm", text.width, text.eps)?,
            projection: schema.linear(
                "norm.projection",
                text.width,
                generation.projection,
                true,
            )?,
            activation: generation.activation,
        };
        let head = schema.linear("head", generation.projection, config.vq.codebook, true)?;
        let codes = schema.parameter(
            "generation_embeddings.weight",
            [config.vq.codebook, config.vq.entry],
        )?;
        let aligner = LinearStack::create(
            &mut schema.scope("generation_aligner"),
            [config.vq.entry, generation.projection, text.width],
            generation.aligner_depth,
            generation.activation,
        )?;
        let caches = (0..blocks.len()).map(|_| text.kv_cache()).collect();
        Ok(Self {
            lm: CausalLm::new(CausalLmParts {
                embedding: Embedding::new(table.clone()),
                embedding_scale: None,
                logit_scale: None,
                logit_cap: None,
                rotary: Some(text.rotary()),
                blocks,
                norm,
                head,
                caches,
            }),
            text: Embedding::new(table),
            codes: Embedding::new(codes),
            aligner,
            vocabulary: text.vocab_size,
        })
    }

    /// Embeds a step's tokens, `[T]`: text tokens through the text embedding, codes through the code embedding and aligner.
    fn embed(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let vocabulary = capture::scalar(tokens.device(), self.vocabulary)?;
        let is_code = tokens.greater_equal(&vocabulary)?;
        let text = tokens
            .minimum(&capture::scalar(tokens.device(), self.vocabulary - 1)?)?
            .apply(&self.text, state)?;
        let codes = tokens
            .sub(&vocabulary)?
            .maximum(&capture::scalar(tokens.device(), 0_i64)?)?
            .apply(&self.codes, state)?
            .apply(&self.aligner, state)?;
        Tensor::where_select(&is_code.unsqueeze(-1)?, &codes, &text)
    }
}

impl DecoderModel for JanusImageDecoder {
    type Cache = KvCache;

    type Tables = Option<RotaryTable>;

    fn tables(
        &self,
        data_type: DataType,
        device: Device,
        capacity: i64,
    ) -> SingeResult<Option<RotaryTable>> {
        self.lm.tables(data_type, device, capacity)
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<KvCache>> {
        self.lm.caches(data_type, device, size)
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Option<RotaryTable>,
        caches: &[KvCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self.embed(step.tokens(), state)?;
        self.lm
            .forward_embedded(step, hidden, tables, caches, state)
    }
}

/// Janus generating images: the tokenizer and chat template a prompt goes through, the decoder over codes, and the VQ decoder.
#[derive(Debug)]
pub struct JanusImageGenerator {
    config: JanusConfig,
    tokenizer: TextTokenizer,
    template: ChatTemplate,
    decoder: Model<JanusImageDecoder>,
    vq: Model<VqDecoder>,
    /// The normalization the processor gives pixels, which post-processing undoes.
    normalization: Normalization,
    /// Kept by their capacity.
    decoders: DecoderCache<i64, CachedDecoder<JanusImageDecoder>>,
}

impl JanusImageGenerator {
    /// Returns the guided and unguided rows' prompts: the user's prompt through the chat template with the image start after it, and the same with every token but the first and the image start padded.
    fn prompts(&self, prompt: &str) -> Result<[Vec<i64>; 2]> {
        let rendered = self.template.render(
            &[Message::new(
                "user",
                Content::Parts(vec![Part::Text {
                    text: prompt.to_owned(),
                }]),
            )],
            true,
            &serde_json::Map::new(),
        )?;
        let start = self.tokenizer.token_id(&self.config.prompt.boi_token)?;
        let guided = self
            .tokenizer
            .encode(&format!("{rendered}{}", self.config.prompt.boi_token), true)?;
        let generation = &self.config.generation;
        let unguided = guided
            .iter()
            .map(|&token| {
                if token == generation.bos_token_id || token == start {
                    token
                } else {
                    generation.pad_token_id
                }
            })
            .collect();
        Ok([guided, unguided])
    }

    /// Returns the code `sampler` chooses from `scores`' two rows: the unguided row's logits moved `scale` times their distance toward the guided row's, as `ClassifierFreeGuidanceLogitsProcessor` moves them.
    fn chosen(scores: &Tensor, scale: f32, sampler: &mut Sampler) -> Result<i64> {
        let scores = scores.cast(DataType::F32)?.to_vec::<f32>()?;
        let (guided, unguided) = scores.split_at(scores.len() / 2);
        let logits = guided
            .iter()
            .zip(unguided)
            .map(|(&guided, &unguided)| unguided + (guided - unguided) * scale)
            .collect::<Vec<_>>();
        i64::try_from(sampler.choose(&logits)?).map_err(|_| Error::unsupported("a code past `i64`"))
    }

    /// Returns the image of decoded `pixels`, `[1, 3, H, W]` in `[-1, 1]`, as `JanusImageProcessor.postprocess` gives it: the normalization undone, scaled to 255, clipped, and truncated.
    fn image(&self, pixels: &Tensor) -> Result<Image> {
        let &[_, _, height, width] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("decoded pixels, [1, 3, H, W]").into());
        };
        let values = pixels
            .cast(DataType::F32)?
            .squeeze(0)?
            .permute(&[1, 2, 0])?
            .contiguous()?
            .to_vec::<f32>()?;
        let Normalization { mean, std, .. } = self.normalization;
        let pixels = values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let channel = index % 3;
                let restored = 255.0 * (f64::from(value) * std[channel] + mean[channel]);
                restored.clamp(0.0, 255.0) as u8
            })
            .collect();
        Ok(Image {
            width: u32::try_from(width).map_err(|_| SingeError::out_of_range("image width"))?,
            height: u32::try_from(height).map_err(|_| SingeError::out_of_range("image height"))?,
            pixels,
        })
    }
}

impl ImageGenerator for JanusImageGenerator {
    fn generate_image(
        &self,
        prompt: &str,
        guidance_scale: Option<f32>,
        sampling: &SamplingRequest,
    ) -> Result<Image> {
        let generation = &self.config.generation;
        let scale = guidance_scale.unwrap_or(generation.guidance_scale);
        let mut sampler = sampling.sampler(generation.sampling);
        let [guided, unguided] = self.prompts(prompt)?;
        let capacity = bucketed_capacity(guided.len() + generation.tokens);
        let vocabulary = self.config.text.vocab_size;
        let codes = self.decoders.run(
            capacity,
            || {
                let options =
                    DecoderOptions::new(2, capacity).with_prompt_chunks(PromptChunking::NONE);
                Ok::<_, Error>(cached_decoder(&self.decoder, options, false)?)
            },
            |decoder| {
                let mut scores = decoder.step(&[guided.as_slice(), unguided.as_slice()], None)?;
                let mut codes = Vec::with_capacity(generation.tokens);
                for index in 0..generation.tokens {
                    let code = Self::chosen(&scores, scale, &mut sampler)?;
                    codes.push(code);
                    if index + 1 < generation.tokens {
                        let token = [vocabulary + code];
                        scores = decoder.step(&[&token[..], &token[..]], None)?;
                    }
                }
                Ok(codes)
            },
        )?;

        let device = self.vq.state().device()?;
        let count = i64::try_from(codes.len()).map_err(|_| SingeError::out_of_range("codes"))?;
        let tokens = Tensor::from_vec(device, Shape::create([1, count])?, codes)?;
        let pixels = self.vq.layer().decode(&tokens, self.vq.state())?;
        self.image(&pixels)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "janus",
    formats: &[(Format::Transformers, &["janus"])],
    loaders: &[
        HeadLoader::ImageTextToText(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::ImageGeneration(|loader| Ok(Box::new(load_generator(loader)?))),
    ],
};

/// Loads the vision tower and aligner, as part `vision`, and the decoder reading its rows, as part `decoder`.
pub fn load(loader: &mut Loader<'_>) -> Result<Janus> {
    let config: JanusConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let vision = Schema::model(data_type, |schema| {
        JanusVision::create(schema, &config.vision, config.text.width)
    })?;
    let vision = loader.load("vision", &vision, &transformers::vision_rules(&config))?;
    let decoder = Schema::model(data_type, |schema| {
        grouped_decoder(schema, &config.text, causal())
            .map(|text| SourcedCausalLm::new(text, config.image_token_id, config.text.width))
    })?;
    let decoder = loader.load("decoder", &decoder, &transformers::decoder_rules())?;
    loader.finish()?;
    let processor = ImagePreprocessor::load(loader.source())?;
    Ok(Janus {
        config,
        vision,
        decoder,
        processor,
        device: loader.options().device,
        decoders: DecoderCache::default(),
    })
}

/// Loads the decoder over codes, as part `generation`, and the VQ decoder, as part `vq`.
pub fn load_generator(loader: &mut Loader<'_>) -> Result<JanusImageGenerator> {
    let config: JanusConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let decoder = Schema::model(data_type, |schema| {
        JanusImageDecoder::create(schema, &config)
    })?;
    let decoder = loader.load(
        "generation",
        &decoder,
        &transformers::generation_rules(&config),
    )?;
    let vq = Schema::model(data_type, |schema| VqDecoder::create(schema, &config.vq))?;
    let vq = loader.load("vq", &vq, &transformers::vq_rules())?;
    loader.finish()?;
    let source = loader.source();
    let normalization = ImagePreprocessor::load(source)?
        .normalization
        .ok_or_else(|| Error::unsupported("a Janus processor that does not normalize"))?;
    Ok(JanusImageGenerator {
        normalization,
        tokenizer: TextTokenizer::load(source)?,
        template: ChatTemplate::load(source)?
            .ok_or_else(|| Error::unsupported("a Janus snapshot without a chat template"))?,
        config,
        decoder,
        vq,
        decoders: DecoderCache::default(),
    })
}
