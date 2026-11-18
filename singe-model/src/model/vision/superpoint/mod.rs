// SuperPoint: a VGG-style encoder over one channel at an eighth of the image, a head scoring each pixel as a keypoint, and a head describing each point of the map, which the keypoints sample.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Model, ModelState, StateBuilder},
};

use super::same;
use crate::{
    error::Result,
    format::Loader,
    head::{HeadLoader, KeypointConfig, KeypointDetector, KeypointLayers},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperPointConfig {
    // Each block's channels; every block but the last halves the map.
    pub encoder: Vec<i64>,
    pub decoder: i64,
    pub descriptor: i64,
    pub keypoints: KeypointConfig,
}

// A cell's 64 pixels and one class for no keypoint.
const CELL_CLASSES: i64 = 65;

#[derive(Debug, Clone)]
struct Block {
    first: Conv2d,
    second: Conv2d,
    pool: bool,
}

#[derive(Debug, Clone)]
pub struct SuperPointLayers {
    blocks: Vec<Block>,
    score_hidden: Conv2d,
    score: Conv2d,
    descriptor_hidden: Conv2d,
    descriptor: Conv2d,
}

impl KeypointLayers for SuperPointLayers {
    // Each pixel's keypoint probability, `[B, H, W]`, and each map point's unit descriptor, `[B, D, H / 8, W / 8]`, from images whose first channel the model reads.
    fn maps(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<(Tensor, Tensor)> {
        let mut hidden = pixels.narrow(1, 0, 1)?;
        for block in &self.blocks {
            hidden = hidden
                .apply(&block.first, state)?
                .apply(&Activation::Relu, state)?
                .apply(&block.second, state)?
                .apply(&Activation::Relu, state)?;
            if block.pool {
                hidden = hidden.max_pool2d((2, 2), (2, 2), (0, 0, 0, 0), (1, 1), false)?;
            }
        }

        let scores = hidden
            .apply(&self.score_hidden, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.score, state)?
            .softmax(1)?;
        let dimensions = scores.shape().dimensions();
        let (batch, rows, columns) = (dimensions[0], dimensions[2], dimensions[3]);
        // Each cell's 64 probabilities, but no keypoint's, laid out as its 8 × 8 pixels.
        let scores = scores
            .narrow(1, 0, CELL_CLASSES - 1)?
            .permute(&[0, 2, 3, 1])?
            .reshape([batch, rows, columns, 8, 8])?
            .permute(&[0, 1, 3, 2, 4])?
            .reshape([batch, rows * 8, columns * 8])?;

        let descriptors = hidden
            .apply(&self.descriptor_hidden, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.descriptor, state)?;
        let norms = descriptors
            .mul(&descriptors)?
            .sum(&[1], true)?
            .sqrt()?
            .clamp(1e-12, f32::MAX)?;
        Ok((scores, descriptors.div(&norms)?))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "superpoint",
    formats: &[(Format::Transformers, &["superpoint"])],
    heads: &[HeadLoader::KeypointDetection(load_detector)],
};

// Registers SuperPoint in `schema`: `blocks` (`first`, `second`), `score_hidden`, `score`, `descriptor_hidden`, and `descriptor`.
pub fn layers(schema: &mut Schema<'_>, config: &SuperPointConfig) -> SingeResult<SuperPointLayers> {
    let mut input = 1;
    let mut blocks = Vec::new();
    for (index, &channels) in config.encoder.iter().enumerate() {
        let mut schema = schema.scope(format!("blocks.{index}"));
        blocks.push(Block {
            first: schema.conv2d("first", input, channels, 3, same(3), true)?,
            second: schema.conv2d("second", channels, channels, 3, same(3), true)?,
            pool: index + 1 < config.encoder.len(),
        });
        input = channels;
    }
    Ok(SuperPointLayers {
        blocks,
        score_hidden: schema.conv2d("score_hidden", input, config.decoder, 3, same(3), true)?,
        score: schema.conv2d("score", config.decoder, CELL_CLASSES, 1, same(1), true)?,
        descriptor_hidden: schema.conv2d(
            "descriptor_hidden",
            input,
            config.decoder,
            3,
            same(3),
            true,
        )?,
        descriptor: schema.conv2d(
            "descriptor",
            config.decoder,
            config.descriptor,
            1,
            same(1),
            true,
        )?,
    })
}

pub fn load_detector(loader: &mut Loader<'_>) -> Result<KeypointDetector> {
    let config: SuperPointConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let layers: Box<dyn KeypointLayers> = Box::new(layers(
        &mut Schema::root(&mut builder, loader.options().data_type),
        &config,
    )?);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(KeypointDetector::new(model, config.keypoints))
}
