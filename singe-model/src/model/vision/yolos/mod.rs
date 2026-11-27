//! YOLOS: a vision transformer whose detection tokens, after its patches, each find an object, read by a class head and a box head of three linear layers each.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Layer, ModelState, PatchEmbedding, StateKey},
    shape::spatial::{Spatial2d, Spatial2dI64},
    tensor::InterpolateMode,
};

use crate::{
    assembly::{FinalNorm, LinearStack, PreNormEncoder, PreNormEncoderConfig, patch_embedding},
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, DetectionScoring, HeadLoader, ObjectDetector},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YolosConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    /// The image height and width the positions are learned for.
    pub image_size: Spatial2dI64,
    pub patch_size: i64,
    pub channels: i64,
    pub detection_tokens: i64,
    /// The labels, before the last score, which stands for no object.
    pub classifier: ClassifierConfig,
}

/// The class token, the patches, and the detection tokens, with learned positions whose patch part is resized bicubically to the image's grid, as `InterpolateInitialPositionEmbeddings` resizes it.
#[derive(Debug, Clone)]
struct YolosTokens {
    patches: PatchEmbedding,
    class: StateKey,
    detection: StateKey,
    positions: StateKey,
    /// The patch grid the positions are learned for.
    grid: Spatial2dI64,
    patch_size: i64,
}

impl YolosTokens {
    fn create(schema: &mut Schema<'_>, config: &YolosConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let grid = config.image_size.map(|side| side / config.patch_size);
        Ok(Self {
            patches: patch_embedding(schema, config.channels, width, config.patch_size, true)?,
            class: schema.parameter("class", [1, 1, width])?,
            detection: schema.parameter("detection", [1, config.detection_tokens, width])?,
            positions: schema.parameter(
                "positions",
                [
                    1,
                    1 + grid.height * grid.width + config.detection_tokens,
                    width,
                ],
            )?,
            grid,
            patch_size: config.patch_size,
        })
    }
}

impl Layer for YolosTokens {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let &[_, _, width] = patches.shape().dimensions() else {
            return Err(SingeError::invalid_argument("patch tokens"));
        };
        let expand = |key: &StateKey| -> SingeResult<Tensor> {
            let tensor = state.tensor(key)?;
            let &[_, count, _] = tensor.shape().dimensions() else {
                return Err(SingeError::invalid_argument("learned tokens"));
            };
            tensor.expand_with(&[patches.dimension(0)?, count.into(), width.into()])
        };
        let detection = expand(&self.detection)?;
        let &[_, detections, _] = detection.shape().dimensions() else {
            return Err(SingeError::invalid_argument("detection tokens"));
        };
        let tokens = Tensor::concat(&[expand(&self.class)?, patches, detection], 1)?;

        let positions = state.tensor(&self.positions)?;
        let &[_, count, _] = positions.shape().dimensions() else {
            return Err(SingeError::invalid_argument("positions"));
        };
        if input.shape().rank() != 4 {
            return Err(SingeError::invalid_argument("images"));
        }

        let Spatial2d {
            height: rows,
            width: columns,
        } = self.grid;
        let size = Spatial2d::new(
            input.dimension(2)? / self.patch_size,
            input.dimension(3)? / self.patch_size,
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

/// YOLOS and its heads: from images to each detection token's label scores and box, `[B, Q, labels + 1 + 4]`.
#[derive(Debug, Clone)]
pub struct YolosDetectorLayers {
    embeddings: YolosTokens,
    encoder: PreNormEncoder,
    classes: LinearStack,
    boxes: LinearStack,
    detection_tokens: i64,
}

impl YolosDetectorLayers {
    pub fn create(schema: &mut Schema<'_>, config: &YolosConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = YolosTokens::create(&mut schema.scope("embeddings"), config)?;
        let encoder =
            PreNormEncoder::create(schema, &config.encoder, config.layers, FinalNorm::Included)?;
        let labels = config.classifier.label_count()?;
        Ok(Self {
            embeddings,
            encoder,
            classes: LinearStack::create(
                &mut schema.scope("classes"),
                [width, width, labels + 1],
                3,
                Activation::Relu,
            )?,
            boxes: LinearStack::create(
                &mut schema.scope("boxes"),
                [width, width, 4],
                3,
                Activation::Relu,
            )?,
            detection_tokens: config.detection_tokens,
        })
    }
}

impl Layer for YolosDetectorLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .encoder
            .forward(&input.apply(&self.embeddings, state)?, state)?;
        if hidden.shape().rank() != 3 {
            return Err(SingeError::invalid_argument("encoder states"));
        }

        let detections = hidden.narrow_with(
            1,
            hidden.dimension(1)? - self.detection_tokens,
            self.detection_tokens,
        )?;
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
    loaders: &[HeadLoader::ObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<ObjectDetector> {
    let config: YolosConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        YolosDetectorLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ObjectDetector::new(
        model,
        config.classifier,
        DetectionScoring::NoObject,
    ))
}
