//! `SmolVLM`: a `SigLIP` vision tower whose patches a pixel shuffle folds into fewer, wider rows and a linear connector projects to the decoder's width, and a Llama decoder whose prompts read those rows in place of their image tokens.
//!
//! An image is cut into tiles the tower reads at its own size, plus the whole image shrunk to one tile, and a clip's sampled frames each become one tile.

mod processor;
pub(crate) mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, GenerationOptions, generate_batch},
    nn::{
        GatedMlp, GroupedQueryAttention, Layer, Linear, Model, ModelState, PatchTokens,
        PreNormBlock, SourcedCausalLm,
    },
};

pub use self::processor::{SmolVlmProcessor, VideoSampling};
use crate::{
    BatchOptions,
    assembly::{
        DecoderConfig, FinalNorm, PreNormEncoder, VisionTowerConfig, causal, grouped_decoder,
        patch_embedding,
    },
    decoder_cache::{DecoderKey, Decoders},
    error::Result,
    format::Loader,
    head::{HeadLoader, ImageTextToText, LoopLimits, LoopSteps},
    model::ModelEntry,
    processor::{image::Image, text::TextTokenizer, video::Video},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmolVlmVisionConfig {
    pub tower: VisionTowerConfig,
    /// The side of each square of patches the pixel shuffle folds into one row.
    pub scale: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmolVlmConfig {
    pub vision: SmolVlmVisionConfig,
    pub text: DecoderConfig,
    pub image_token_id: i64,
    pub processor: SmolVlmProcessor,
}

/// The vision tower and connector: from tiles, `[N, C, S, S]`, to the rows the decoder reads in place of their image tokens, `[N, (S / P)² / scale², W]`.
#[derive(Debug, Clone)]
pub struct SmolVlmVision {
    embeddings: PatchTokens,
    encoder: PreNormEncoder,
    connector: Linear,
    /// The patches along a tile's side.
    side: i64,
    scale: i64,
}

impl SmolVlmVision {
    /// Registers the tower in `schema`: `embeddings` (`patches`, `positions`), `blocks`, `norm`, and `connector`, which projects to `out_width`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &SmolVlmVisionConfig,
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
        let folded = width * config.scale * config.scale;
        Ok(Self {
            embeddings,
            encoder,
            connector: schema.linear("connector", folded, out_width, false)?,
            side: tower.side(),
            scale: config.scale,
        })
    }
}

impl Layer for SmolVlmVision {
    /// Returns each tile's rows, `[N, (S / P)² / scale², W]`.
    ///
    /// Every tile fills the tower's size, so each patch takes the learned position of its own place, as `transformers`' bucketed positions give a tile without padding.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .encoder
            .forward(&input.apply(&self.embeddings, state)?, state)?;
        let &[tiles, _, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the vision tower's states"));
        };
        let (side, scale) = (self.side, self.scale);
        // `transformers`' `pixel_shuffle`: each `scale × scale` square of patches becomes one row of their features side by side.
        hidden
            .reshape([tiles, side, side / scale, width * scale])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([tiles, side / scale, side / scale, width * scale * scale])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([tiles, side * side / (scale * scale), width * scale * scale])?
            .apply(&self.connector, state)
    }
}

pub type SmolVlmDecoder = SourcedCausalLm<PreNormBlock<GroupedQueryAttention, GatedMlp>>;

/// `SmolVLM` reading images and clips: the processor's settings, the vision tower, and the decoder, each with state of its own.
#[derive(Debug, Clone)]
pub struct SmolVlm {
    config: SmolVlmConfig,
    vision: Model<SmolVlmVision>,
    decoder: Model<SmolVlmDecoder>,
    device: Device,
    decoders: Decoders<DecoderKey, CachedDecoder<SmolVlmDecoder>>,
}

impl SmolVlm {
    /// Encodes tiles, `[N, C, S, S]` normalized, into their image tokens' rows, `[N · R, W]` for `R` rows a tile.
    pub fn encode(&self, tiles: &Tensor) -> Result<Tensor> {
        let rows = self.vision.forward(tiles)?;
        let &[count, per_tile, width] = rows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the vision tower's rows").into());
        };
        Ok(rows.reshape([count * per_tile, width])?)
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
                        SmolVlmDecoder::write_sources_at(decoder.caches(), row, &places, rows)
                    },
                    |_, _| Ok(()),
                )
            },
        )?;
        Ok(generated.pop().unwrap_or_default())
    }

    fn vision_type(&self) -> Result<DataType> {
        Ok(self.vision.state().floating_type()?)
    }
}

impl ImageTextToText for SmolVlm {
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps> {
        LoopSteps::one_sequence(&self.decoder, limits, generation, |_| Ok(None))
    }

    fn image_token(&self) -> i64 {
        self.config.image_token_id
    }

    /// Generates after `prompt`, each image cut into tiles and its placeholder in the prompt replaced by the tiles' image tokens between their markers, as `SmolVLMProcessor` expands it before tokenizing.
    fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Image],
        options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        let processor = &self.config.processor;
        let mut tiles = Vec::new();
        let mut expansions = Vec::new();
        for image in images {
            let (cut, grid) = processor.tiles(image)?;
            expansions.push(processor.image_prompt(grid));
            tiles.extend(cut);
        }

        let text = processor.expand(prompt, &processor.image_token, &expansions)?;
        let pixels = processor.pixels(&tiles, self.device, self.vision_type()?)?;
        let rows = self.encode(&pixels)?;
        self.generate_tokens(&tokenizer.encode(&text, true)?, &rows, options)
    }

    fn video_template(&self) -> Option<&str> {
        Some(processor::VIDEO_CHAT_TEMPLATE)
    }

    /// Generates after `prompt`, the clip's frames sampled and each read as one tile, and its placeholder replaced by an introduction and each frame's timestamp and image tokens, as `SmolVLMProcessor` expands it.
    fn generate_video(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        video: &Video,
        fps: f64,
        options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        let processor = &self.config.processor;
        let (frames, expansion) = processor.frames(video, fps)?;
        let text = processor.expand(prompt, processor::VIDEO_TOKEN, &[expansion])?;
        let pixels = processor.pixels(&frames, self.device, self.vision_type()?)?;
        let rows = self.encode(&pixels)?;
        self.generate_tokens(&tokenizer.encode(&text, true)?, &rows, options)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "smolvlm",
    formats: &[(Format::Transformers, &["smolvlm"])],
    loaders: &[HeadLoader::ImageTextToText(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

/// Loads the vision tower, as part `vision`, and the decoder reading its rows, as part `decoder`.
pub fn load(loader: &mut Loader<'_>) -> Result<SmolVlm> {
    let config: SmolVlmConfig = loader.config(transformers::config)?;
    let data_type = loader.options().data_type;
    let vision = Schema::model(data_type, |schema| {
        SmolVlmVision::create(schema, &config.vision, config.text.width)
    })?;
    let vision = loader.load(
        "vision",
        &vision,
        &transformers::vision_rules(
            "model.vision_model",
            "model.connector.modality_projection.proj",
        ),
    )?;
    let decoder = Schema::model(data_type, |schema| {
        grouped_decoder(schema, &config.text, causal())
            .map(|text| SourcedCausalLm::new(text, config.image_token_id, config.text.width))
    })?;
    let decoder = loader.load("decoder", &decoder, &transformers::decoder_rules())?;
    loader.finish()?;
    Ok(SmolVlm {
        config,
        vision,
        decoder,
        device: loader.options().device,
        decoders: Decoders::default(),
    })
}
