// Qwen3.5: a text decoder of gated full attention and gated delta nets, with dense or mixture-of-experts feed-forward layers and a multi-token-prediction draft layer, and in a model that reads images, a vision encoder whose rows its prompts read in place of their image tokens.

mod text;
mod transformers;
mod vision;

use serde::{Deserialize, Serialize};
use singe::nn::{Activation, GatedDeltaNetSpec, Model, SourcedCausalLm};

pub use self::{
    text::{Qwen35, Qwen35Block, Qwen35Layer, Qwen35Layers},
    vision::{
        ImagePatches, ImagePrompt, Qwen35ImageDecoder, Qwen35VisionEncoder, Qwen35VisionLanguage,
        smart_resize,
    },
};
use crate::{
    assembly::DecoderConfig,
    error::{Error, Result},
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    source::Format,
};

// What mixes a block's tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35Mixer {
    // Gated full attention over a cache of keys and values.
    Attention,
    // A gated delta net, linear attention over a state per sequence.
    DeltaNet,
}

// A mixture of experts in place of every block's dense feed-forward layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen35ExpertsConfig {
    pub experts: i64,
    // The experts the router chooses for each token.
    pub top_k: i64,
    // Each expert's inner width.
    pub inner: i64,
    // The shared expert's inner width.
    pub shared_inner: i64,
}

// Qwen3.5's text decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35TextConfig {
    pub decoder: DecoderConfig,
    // Each block's mixer, in order.
    pub mixers: Vec<Qwen35Mixer>,
    pub delta_net: GatedDeltaNetSpec,
    pub experts: Option<Qwen35ExpertsConfig>,
}

impl Qwen35TextConfig {
    // Whether block `index` is full attention rather than a gated delta net.
    fn is_full_attention(&self, index: usize) -> bool {
        self.mixers.get(index) == Some(&Qwen35Mixer::Attention)
    }
}

// Qwen3.5's vision encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qwen35VisionConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    // The rows of the square table of learned positions.
    pub positions: i64,
    // The width of the rows the encoder gives the decoder, one per merged block of patches.
    pub out_width: i64,
    pub patch_size: i64,
    // The patches along each side of a merged block.
    pub merge_size: i64,
    // The frames each patch holds.
    pub frames: i64,
    pub channels: i64,
}

impl Qwen35VisionConfig {
    // The features of one patch: every channel of every frame of its pixels.
    const fn patch_features(&self) -> i64 {
        self.channels * self.frames * self.patch_size * self.patch_size
    }

    // The side of the square table of learned positions.
    fn position_side(&self) -> Result<i64> {
        let side = (1..=self.positions)
            .find(|side| side * side >= self.positions)
            .unwrap_or(0);
        if side * side == self.positions {
            Ok(side)
        } else {
            Err(Error::unsupported(
                "a table of learned positions that is not square",
            ))
        }
    }
}

// The fewest and most pixels an image is resized into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSizeBounds {
    pub min_pixels: i64,
    pub max_pixels: i64,
}

// How the image front end resizes and normalizes an image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35ImageProcessorConfig {
    pub bounds: ImageSizeBounds,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

// What a Qwen3.5 that reads images adds to its text decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35VisionLanguageConfig {
    pub encoder: Qwen35VisionConfig,
    // The token a prompt holds in place of each of an image's rows.
    pub image_token_id: i64,
    // The rotary frequencies each of M-RoPE's axes takes, time, height, and width, interleaved.
    pub position_sections: [i64; 3],
    pub processor: Qwen35ImageProcessorConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35Config {
    pub text: Qwen35TextConfig,
    // Absent in a model that reads only text.
    pub vision: Option<Qwen35VisionLanguageConfig>,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3_5",
    formats: &[(
        Format::Transformers,
        &["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"],
    )],
    heads: &[
        HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::ImageTextToText(|loader| Ok(Box::new(load_vision_language(loader)?))),
    ],
};

// Loads the text decoder, as part `decoder`, without its draft layer.
pub fn load(loader: &mut Loader<'_>) -> Result<Qwen35> {
    load_decoder(loader, false)
}

// Loads the text decoder with its multi-token-prediction draft layer, which speculative decoding drafts with.
pub fn load_drafting(loader: &mut Loader<'_>) -> Result<Qwen35> {
    load_decoder(loader, true)
}

fn load_decoder(loader: &mut Loader<'_>, drafting: bool) -> Result<Qwen35> {
    let config: Qwen35Config = loader.config(transformers::config)?;
    if drafting && config.text.experts.is_some() {
        return Err(Error::unsupported("a draft layer of experts"));
    }

    let (layers, state) = text::decoder(&config.text, loader.options().data_type, drafting)?;
    let rules = transformers::decoder_rules(&config, drafting, false);
    let model = loader.load("decoder", &Model::new(layers, state), &rules)?;
    loader.finish()?;
    Ok(model)
}

// Loads the vision encoder, as part `vision`, and the text decoder reading its rows, as part `decoder`.
pub fn load_vision_language(loader: &mut Loader<'_>) -> Result<Qwen35VisionLanguage> {
    let config: Qwen35Config = loader.config(transformers::config)?;
    let images = config
        .vision
        .clone()
        .ok_or_else(|| Error::unsupported("reading images with a model that reads only text"))?;
    let data_type = loader.options().data_type;
    let encoder = loader.load(
        "vision",
        &vision::encoder(&images.encoder, data_type)?,
        &transformers::vision_rules(),
    )?;
    let (layers, state) = text::decoder(&config.text, data_type, false)?;
    let schema = Model::new(
        SourcedCausalLm::new(layers, images.image_token_id, images.encoder.out_width)
            .with_position_sections(images.position_sections),
        state,
    );
    let decoder = loader.load(
        "decoder",
        &schema,
        &transformers::decoder_rules(&config, false, true),
    )?;
    loader.finish()?;
    Ok(Qwen35VisionLanguage {
        config: images,
        encoder,
        decoder,
        device: loader.options().device,
    })
}
