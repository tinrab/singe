//! Qwen3.5: a text decoder of gated full attention and gated delta nets, with dense or mixture-of-experts feed-forward layers and a multi-token-prediction draft layer, and in a model that reads images, a vision encoder whose rows its prompts read in place of their image tokens.

mod processor;
mod text;
mod transformers;
mod vision;

use serde::{Deserialize, Serialize};
use singe::nn::{Activation, GatedDeltaNetSpec, SourcedCausalLm};

pub use self::{
    processor::{ImageSizeRange, Qwen35ImageProcessor},
    text::{Qwen35, Qwen35Block, Qwen35Layer, Qwen35Layers},
    vision::{Qwen35ImageDecoder, Qwen35VisionEncoder, Qwen35VisionLanguage},
};
use crate::{
    assembly::{DecoderConfig, ExpertsConfig},
    decoder_cache::DecoderCache,
    error::{Error, Result},
    format::Loader,
    head::HeadLoader,
    model::{
        ModelEntry,
        text::qwen3_5::transformers::{Vision, decoder_rules, vision_rules},
    },
    schema::Schema,
    source::Format,
};

/// What mixes a block's tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35Mixer {
    /// Gated full attention over a cache of keys and values.
    Attention,
    /// A gated delta net, linear attention over a state per sequence.
    DeltaNet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Draft {
    Omitted,
    Included,
}

/// The settings of Qwen3.5's text decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35TextConfig {
    pub decoder: DecoderConfig,
    /// Each block's mixer, in order.
    pub mixers: Vec<Qwen35Mixer>,
    pub delta_net: GatedDeltaNetSpec,
    /// A mixture of experts in place of every block's dense feed-forward layer.
    pub experts: Option<ExpertsConfig>,
}

impl Qwen35TextConfig {
    /// Returns whether block `index` is full attention rather than a gated delta net.
    fn is_full_attention(&self, index: usize) -> bool {
        self.mixers.get(index) == Some(&Qwen35Mixer::Attention)
    }
}

/// The settings of Qwen3.5's vision encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen35VisionConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    /// The rows of the square table of learned positions.
    pub max_positions: i64,
    /// The width of the rows the encoder gives the decoder, one per merged block of patches.
    pub out_width: i64,
    pub patch_size: i64,
    /// The patches along each side of a merged block.
    pub merge_size: i64,
    /// The frames each patch holds.
    pub frames: i64,
    pub channels: i64,
}

impl Qwen35VisionConfig {
    /// Returns the features of one patch: every channel of every frame of its pixels.
    const fn patch_features(&self) -> i64 {
        self.channels * self.frames * self.patch_size * self.patch_size
    }

    /// Returns the side every resized image's sides are a multiple of.
    const fn size_factor(&self) -> i64 {
        self.patch_size * self.merge_size
    }

    /// Returns the side of the square table of learned positions, or an error if the table is not square.
    fn position_side(&self) -> Result<i64> {
        let side = (1..=self.max_positions)
            .find(|side| side * side >= self.max_positions)
            .unwrap_or(0);
        if side * side == self.max_positions {
            Ok(side)
        } else {
            Err(Error::unsupported(
                "a table of learned positions that is not square",
            ))
        }
    }
}

/// What a Qwen3.5 that reads images adds to its text decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35VisionLanguageConfig {
    pub encoder: Qwen35VisionConfig,
    /// The token a prompt holds in place of each of an image's rows.
    pub image_token_id: i64,
    /// The rotary frequencies each of `M-RoPE`'s axes takes, time, height, and width, interleaved.
    pub position_sections: [i64; 3],
    pub processor: Qwen35ImageProcessor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35Config {
    pub text: Qwen35TextConfig,
    /// What the model adds to read images, absent in a model that reads only text.
    pub vision: Option<Qwen35VisionLanguageConfig>,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3_5",
    formats: &[(
        Format::Transformers,
        &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"],
    )],
    loaders: &[
        HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::ImageTextToText(|loader| Ok(Box::new(load_vision_language(loader)?))),
    ],
};

/// Loads the text decoder, as part `model`, without its draft layer.
pub fn load(loader: &mut Loader<'_>) -> Result<Qwen35> {
    load_decoder(loader, Draft::Omitted)
}

/// Loads the text decoder, as part `model`, with its multi-token-prediction draft layer, which speculative decoding drafts with.
///
/// # Errors
///
/// Returns an error if the model has experts, whose draft layer this does not support, or if its config or checkpoint does not load.
pub fn load_drafting(loader: &mut Loader<'_>) -> Result<Qwen35> {
    load_decoder(loader, Draft::Included)
}

fn load_decoder(loader: &mut Loader<'_>, draft: Draft) -> Result<Qwen35> {
    let config: Qwen35Config = loader.config(transformers::config)?;
    if draft == Draft::Included && config.text.experts.is_some() {
        return Err(Error::unsupported("a draft layer of experts"));
    }

    let model = Schema::model(loader.options().data_type, |schema| {
        text::decoder(schema, &config.text, draft)
    })?;
    let rules = decoder_rules(&config, draft, Vision::Skipped);
    loader.load_whole(&model, &rules)
}

/// Loads the vision encoder, as part `vision`, and the text decoder reading its rows, as part `decoder`.
///
/// # Errors
///
/// Returns an error if the model reads only text, or if its config or checkpoint does not load.
pub fn load_vision_language(loader: &mut Loader<'_>) -> Result<Qwen35VisionLanguage> {
    let config: Qwen35Config = loader.config(transformers::config)?;
    let images = config
        .vision
        .clone()
        .ok_or_else(|| Error::unsupported("reading images with a model that reads only text"))?;
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        Qwen35VisionEncoder::create(schema, &images.encoder)
    })?;
    let encoder = loader.load("vision", &encoder, &vision_rules())?;
    let decoder = Schema::model(data_type, |schema| {
        text::decoder(schema, &config.text, Draft::Omitted).map(|layers| {
            SourcedCausalLm::new(layers, images.image_token_id, images.encoder.out_width)
                .with_position_sections(images.position_sections)
        })
    })?;
    let decoder = loader.load(
        "decoder",
        &decoder,
        &decoder_rules(&config, Draft::Omitted, Vision::Loaded),
    )?;
    loader.finish()?;
    Ok(Qwen35VisionLanguage {
        config: images,
        encoder,
        decoder,
        device: loader.options().device,
        decoders: DecoderCache::default(),
    })
}
