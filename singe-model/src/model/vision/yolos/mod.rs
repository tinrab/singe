// YOLOS: a vision transformer whose detection tokens, after its patches, each find an object, read by a class head and a box head of three linear layers each.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2dConfig, Layer, Model, ModelState, PatchEmbedding, StateBuilder, StateKey,
    },
    tensor::InterpolateMode,
};

use crate::{
    assembly::{Mlp, PreNormEncoder, PreNormEncoderSpec},
    error::Result,
    format::Loader,
    head::{ClassifierConfig, HeadLoader, ObjectDetector, PixelLayers},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YolosConfig {
    pub encoder: PreNormEncoderSpec,
    pub layers: i64,
    // The image height and width the positions are learned for.
    pub image_size: [i64; 2],
    pub patch_size: i64,
    pub channels: i64,
    pub detection_tokens: i64,
    // The labels, before the last score, which stands for no object.
    pub classifier: ClassifierConfig,
}

// The class token, the patches, and the detection tokens, with learned positions whose patch part is resized bicubically to the image's grid, as `InterpolateInitialPositionEmbeddings` resizes it.
#[derive(Debug, Clone)]
struct YolosTokens {
    patches: PatchEmbedding,
    class: StateKey,
    detection: StateKey,
    positions: StateKey,
    // The patch grid the positions are learned for.
    grid: [i64; 2],
    patch_size: i64,
}

impl Layer for YolosTokens {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let &[batch, _, width] = patches.shape().dimensions() else {
            return Err(SingeError::invalid_argument("patch tokens"));
        };
        let expand = |key: &StateKey| -> SingeResult<Tensor> {
            let tensor = state.tensor(key)?;
            let count = tensor.shape().dimensions()[1];
            tensor.expand(Shape::create([batch, count, width])?)
        };
        let detection = expand(&self.detection)?;
        let tokens = Tensor::concat(&[expand(&self.class)?, patches, detection.clone()], 1)?;

        let positions = state.tensor(&self.positions)?;
        let count = positions.shape().dimensions()[1];
        let detections = detection.shape().dimensions()[1];
        let [rows, columns] = self.grid;
        let dimensions = input.shape().dimensions();
        let size = (
            dimensions[2] / self.patch_size,
            dimensions[3] / self.patch_size,
        );
        let patch_positions = positions
            .narrow(1, 1, count - 1 - detections)?
            .transpose(1, 2)?
            .reshape([1, width, rows, columns])?
            .interpolate(size, InterpolateMode::Bicubic)?
            .flatten(2, 3)?
            .transpose(1, 2)?;
        let positions = Tensor::concat(
            &[
                positions.narrow(1, 0, 1)?,
                patch_positions,
                positions.narrow(1, count - detections, detections)?,
            ],
            1,
        )?;
        tokens.add(&positions)
    }
}

// YOLOS and its heads: from images to each detection token's label scores and box, `[B, Q, labels + 1 + 4]`.
#[derive(Debug, Clone)]
pub struct YolosDetectorLayers {
    embeddings: YolosTokens,
    encoder: PreNormEncoder,
    classes: Mlp,
    boxes: Mlp,
    detection_tokens: i64,
}

impl Layer for YolosDetectorLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .encoder
            .forward(&input.apply(&self.embeddings, state)?, state)?;
        let count = hidden.shape().dimensions()[1];
        let detections = hidden.narrow(1, count - self.detection_tokens, self.detection_tokens)?;
        let scores = detections.apply(&self.classes, state)?;
        let boxes = detections
            .apply(&self.boxes, state)?
            .apply(&Activation::Sigmoid, state)?;
        Tensor::concat(&[scores, boxes], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "yolos",
    formats: &[(Format::Transformers, &["yolos"])],
    heads: &[HeadLoader::ObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<ObjectDetector> {
    let config: YolosConfig = loader.config(transformers::config)?;
    let width = config.encoder.width;
    let [height, image_width] = config.image_size;
    let grid = [height / config.patch_size, image_width / config.patch_size];
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            YolosTokens {
                patches: PatchEmbedding::new(schema.conv2d(
                    "patches",
                    config.channels,
                    width,
                    config.patch_size,
                    Conv2dConfig {
                        stride: (config.patch_size, config.patch_size),
                        ..Conv2dConfig::default()
                    },
                    true,
                )?),
                class: schema.parameter("class", [1, 1, width])?,
                detection: schema.parameter("detection", [1, config.detection_tokens, width])?,
                positions: schema.parameter(
                    "positions",
                    [1, 1 + grid[0] * grid[1] + config.detection_tokens, width],
                )?,
                grid,
                patch_size: config.patch_size,
            }
        };
        let encoder = PreNormEncoder::create(&mut schema, &config.encoder, config.layers, true)?;
        let labels = i64::try_from(config.classifier.labels.len())
            .map_err(|_| SingeError::out_of_range("labels"))?;
        YolosDetectorLayers {
            embeddings,
            encoder,
            classes: Mlp::create(
                &mut schema.scope("classes.layers"),
                width,
                width,
                labels + 1,
                3,
            )?,
            boxes: Mlp::create(&mut schema.scope("boxes.layers"), width, width, 4, 3)?,
            detection_tokens: config.detection_tokens,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(ObjectDetector::new(model, config.classifier))
}
