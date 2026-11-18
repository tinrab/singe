// Heads over images and clips: classification of an image or a clip, object detection, and the contrastive image-text models zero-shot classification and detection run.
// A model builds its layers to these outputs itself, since vision models differ in what their heads read: a class token, a mean of the patches, or detection tokens.

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState, StateKey, TokenBatch},
};

use super::ClassifierConfig;
use crate::error::{Error, Result};

// A model's layers from pixels, or from a clip's frames, to one tensor of outputs, which its head reads.
#[derive(Debug)]
pub struct PixelLayers(pub Box<dyn Layer>);

impl Layer for PixelLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.0.forward(input, state)
    }
}

// A text tower's layers from a batch of token sequences to each sequence's embedding, `[N, D]`.
#[derive(Debug)]
pub struct TextEmbeddingLayers(pub Box<dyn Layer<TokenBatch>>);

impl Layer<TokenBatch> for TextEmbeddingLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.0.forward(input, state)
    }
}

// The rows of a `[B, …, N]` tensor's last axis, as host vectors.
fn rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let dimensions = tensor.shape().dimensions();
    let width = usize::try_from(dimensions[dimensions.len() - 1])
        .map_err(|_| SingeError::out_of_range("output width"))?;
    let values = tensor.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok(values.chunks(width).map(<[f32]>::to_vec).collect())
}

// A classifier of images or clips: its layers to each input's scores, `[B, labels]`, and its labels.
#[derive(Debug)]
pub struct Classifier {
    model: Model<PixelLayers>,
    config: ClassifierConfig,
}

impl Classifier {
    #[must_use]
    pub const fn new(model: Model<PixelLayers>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<PixelLayers> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    // Returns each input's scores, one per label, before any score function.
    pub fn scores(&self, pixels: &Tensor) -> Result<Vec<Vec<f32>>> {
        rows(&self.model.forward(pixels)?)
    }
}

// An image classifier, as `*ForImageClassification` heads are.
#[derive(Debug)]
pub struct ImageClassifier(pub Classifier);

// A clip classifier, as `*ForVideoClassification` heads are; its input is `[B, T, C, H, W]`.
#[derive(Debug)]
pub struct VideoClassifier(pub Classifier);

// One query of a detector: each label's score, the last for no object, and its box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub scores: Vec<f32>,
    pub center_box: [f32; 4],
}

// An object detector, as `*ForObjectDetection` heads are: its layers give each query's label scores, the last for no object, then its box, `[B, Q, labels + 1 + 4]`.
#[derive(Debug)]
pub struct ObjectDetector {
    model: Model<PixelLayers>,
    config: ClassifierConfig,
}

impl ObjectDetector {
    #[must_use]
    pub const fn new(model: Model<PixelLayers>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<PixelLayers> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    // Returns each image's queries.
    pub fn detect(&self, pixels: &Tensor) -> Result<Vec<Vec<Detection>>> {
        let output = self.model.forward(pixels)?;
        let queries = usize::try_from(output.shape().dimensions()[1])
            .map_err(|_| SingeError::out_of_range("queries"))?;
        let labels = self.config.labels.len() + 1;
        let detections = rows(&output)?
            .into_iter()
            .map(|row| Detection {
                scores: row[..labels].to_vec(),
                center_box: [
                    row[labels],
                    row[labels + 1],
                    row[labels + 2],
                    row[labels + 3],
                ],
            })
            .collect::<Vec<_>>();
        Ok(detections
            .chunks(queries)
            .map(<[Detection]>::to_vec)
            .collect())
    }
}

// A contrastive model, as CLIP's and CLAP's `*Model` are: a tower of images or clips and a tower of texts to embeddings of one space, and the scale of their cosines.
#[derive(Debug)]
pub struct ContrastiveEncoder {
    inputs: Model<PixelLayers>,
    texts: Model<TextEmbeddingLayers>,
    // The learned log of the cosines' scale.
    logit_scale: StateKey,
}

impl ContrastiveEncoder {
    #[must_use]
    pub const fn new(
        inputs: Model<PixelLayers>,
        texts: Model<TextEmbeddingLayers>,
        logit_scale: StateKey,
    ) -> Self {
        Self {
            inputs,
            texts,
            logit_scale,
        }
    }

    #[must_use]
    pub const fn inputs(&self) -> &Model<PixelLayers> {
        &self.inputs
    }

    #[must_use]
    pub const fn texts(&self) -> &Model<TextEmbeddingLayers> {
        &self.texts
    }

    // Returns each input's scaled cosine with each text, `[B][N]`, as `logits_per_image` and `logits_per_audio` are.
    pub fn logits(&self, inputs: &Tensor, texts: &TokenBatch) -> Result<Vec<Vec<f32>>> {
        let normalized = |embeddings: Tensor| -> SingeResult<Tensor> {
            let embeddings = embeddings.cast(DataType::F32)?;
            let norms = embeddings.mul(&embeddings)?.sum(&[1], true)?.sqrt()?;
            embeddings.div(&norms)
        };
        let images = normalized(self.inputs.forward(inputs)?)?;
        let texts = normalized(self.texts.forward_group(texts)?)?;
        let scale = self
            .inputs
            .state()
            .tensor(&self.logit_scale)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?
            .first()
            .copied()
            .ok_or_else(|| Error::unsupported("a model without a logit scale"))?
            .exp();
        rows(
            &images
                .matmul(&texts.transpose(0, 1)?)?
                .mul_scalar(f64::from(scale))?,
        )
    }
}

// An open-vocabulary detector, as OWL-ViT's `*ForObjectDetection` is: an image tower to each patch's class embedding, box, logit shift, and logit scale, `[B, P, D + 4 + 2]`, and a text tower to each query's embedding, `[N, D]`.
#[derive(Debug)]
pub struct OpenVocabularyDetector {
    images: Model<PixelLayers>,
    texts: Model<TextEmbeddingLayers>,
}

// One patch of an open-vocabulary detector: its score for each query, before the sigmoid, and its box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryDetection {
    pub logits: Vec<f32>,
    pub center_box: [f32; 4],
}

impl OpenVocabularyDetector {
    #[must_use]
    pub const fn new(images: Model<PixelLayers>, texts: Model<TextEmbeddingLayers>) -> Self {
        Self { images, texts }
    }

    #[must_use]
    pub const fn images(&self) -> &Model<PixelLayers> {
        &self.images
    }

    // Returns each image's patches, scored against each text, as OWL-ViT's class head scores them: cosines shifted, then scaled.
    pub fn detect(&self, pixels: &Tensor, texts: &TokenBatch) -> Result<Vec<Vec<QueryDetection>>> {
        let output = self.images.forward(pixels)?;
        let patches = usize::try_from(output.shape().dimensions()[1])
            .map_err(|_| SingeError::out_of_range("patches"))?;
        let queries = rows(&self.texts.forward_group(texts)?)?;
        let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let queries = queries
            .iter()
            .map(|query| {
                let norm = norm(query) + 1e-6;
                query.iter().map(|value| value / norm).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let width = queries.first().map_or(0, Vec::len);
        let detections = rows(&output)?
            .into_iter()
            .map(|row| {
                let (class, rest) = row.split_at(width);
                let norm = norm(class) + 1e-6;
                let (shift, scale) = (rest[4], rest[5]);
                // ELU plus one.
                let scale = if scale > 0.0 {
                    scale + 1.0
                } else {
                    scale.exp()
                };
                QueryDetection {
                    logits: queries
                        .iter()
                        .map(|query| {
                            let cosine = class
                                .iter()
                                .zip(query)
                                .map(|(class, query)| class / norm * query)
                                .sum::<f32>();
                            (cosine + shift) * scale
                        })
                        .collect(),
                    center_box: [rest[0], rest[1], rest[2], rest[3]],
                }
            })
            .collect::<Vec<_>>();
        Ok(detections
            .chunks(patches)
            .map(<[QueryDetection]>::to_vec)
            .collect())
    }
}

// A depth estimator, as `*ForDepthEstimation` heads are: its layers give each pixel's depth at the size the model reads, `[B, H, W]`.
#[derive(Debug)]
pub struct DepthEstimator {
    model: Model<PixelLayers>,
}

impl DepthEstimator {
    #[must_use]
    pub const fn new(model: Model<PixelLayers>) -> Self {
        Self { model }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<PixelLayers> {
        &self.model
    }

    pub fn depth(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(pixels)?)
    }
}

// A semantic segmenter, as `*ForSemanticSegmentation` heads are: its layers give each label's score at each point of a map, `[B, labels, h, w]`, which the pipeline resizes to the image.
#[derive(Debug)]
pub struct SemanticSegmenter {
    model: Model<PixelLayers>,
    config: ClassifierConfig,
}

impl SemanticSegmenter {
    #[must_use]
    pub const fn new(model: Model<PixelLayers>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<PixelLayers> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    pub fn scores(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(pixels)?)
    }
}

// A super-resolution model, as `*ForImageSuperResolution` heads are: its layers take images in [0, 1] whose sides are whole windows and give them larger, `[B, 3, H · s, W · s]`.
#[derive(Debug)]
pub struct ImageUpscaler {
    model: Model<PixelLayers>,
    window: i64,
}

impl ImageUpscaler {
    #[must_use]
    pub const fn new(model: Model<PixelLayers>, window: i64) -> Self {
        Self { model, window }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<PixelLayers> {
        &self.model
    }

    // The side its inputs' sides are multiples of.
    #[must_use]
    pub const fn window(&self) -> i64 {
        self.window
    }

    pub fn upscale(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(pixels)?)
    }
}
