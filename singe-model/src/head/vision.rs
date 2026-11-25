//! Heads over images and clips: classification of an image or a clip, object detection, the contrastive models zero-shot classification and detection run, depth estimation, semantic segmentation, and super-resolution.
//! A model builds its layers to these outputs itself, since vision models differ in what their heads read: a class token, a mean of the patches, or detection tokens.

use std::{ops::Deref, sync::Arc};

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, Model, StateKey, TokenBatch},
};

use crate::{
    error::{Error, Result},
    function::{
        TensorFunction,
        forward::{External, ExternalHead, Forward},
    },
    head::{Boxed, ClassifierConfig},
    host,
    source::Source,
};

/// Returns the rows of a `[B, …, N]` tensor's last axis, as host vectors.
fn rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let Some(&width) = tensor.shape().dimensions().last() else {
        return Err(SingeError::invalid_argument("output, [B, …, N]").into());
    };
    let width = usize::try_from(width).map_err(|_| SingeError::out_of_range("output width"))?;
    let values = tensor.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok(host::rows(&values, width))
}

/// A classifier of images or clips: its layers to each input's scores, `[B, labels]`, and its labels.
#[derive(Debug)]
pub struct Classifier {
    model: Forward<Boxed<dyn Layer>>,
    config: ClassifierConfig,
}

impl Classifier {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, config: ClassifierConfig) -> Self {
        Self {
            model: Forward::Native(model),
            config,
        }
    }

    #[must_use]
    pub const fn external(model: External, config: ClassifierConfig) -> Self {
        Self {
            model: Forward::External(model),
            config,
        }
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
    }

    pub fn upload<const N: usize>(&self, dimensions: [i64; N], values: Vec<f32>) -> Result<Tensor> {
        self.model.upload(dimensions, values)
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Returns each input's scores, one per label, before any score function.
    pub fn scores(&self, pixels: &Tensor) -> Result<Vec<Vec<f32>>> {
        rows(&self.model.forward(pixels)?)
    }
}

/// An image classifier, as `*ForImageClassification` heads are.
#[derive(Debug)]
pub struct ImageClassifier(pub Classifier);

impl Deref for ImageClassifier {
    type Target = Classifier;

    fn deref(&self) -> &Classifier {
        &self.0
    }
}

impl ExternalHead for ImageClassifier {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let (model, config) = External::classifier(source, function, 1)?;
        Ok(Self(Classifier::external(model, config)))
    }
}

/// A clip classifier, as `*ForVideoClassification` heads are; its input is `[B, T, C, H, W]`.
#[derive(Debug)]
pub struct VideoClassifier(pub Classifier);

impl Deref for VideoClassifier {
    type Target = Classifier;

    fn deref(&self) -> &Classifier {
        &self.0
    }
}

/// One query of a detector: each label's score, the last for no object, and its box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub scores: Vec<f32>,
    pub center_box: [f32; 4],
}

/// An object detector, as `*ForObjectDetection` heads are: its layers give each query's label scores, the last for no object, then its box, `[B, Q, labels + 1 + 4]`.
#[derive(Debug)]
pub struct ObjectDetector {
    model: Model<Boxed<dyn Layer>>,
    config: ClassifierConfig,
}

impl ObjectDetector {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, config: ClassifierConfig) -> Self {
        Self { model, config }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn Layer>> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Returns each image's detections, one per query.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's output is not `[B, Q, N]`, or if running the model fails.
    pub fn detect(&self, pixels: &Tensor) -> Result<Vec<Vec<Detection>>> {
        let output = self.model.forward(pixels)?;
        let &[_, queries, _] = output.shape().dimensions() else {
            return Err(SingeError::invalid_argument("detections, [B, Q, N]").into());
        };
        let queries = usize::try_from(queries).map_err(|_| SingeError::out_of_range("queries"))?;
        let labels = self.config.labels().len() + 1;
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
        Ok(host::rows(&detections, queries))
    }
}

/// A contrastive model, as CLIP's and CLAP's `*Model` are: a tower of images or clips and a tower of texts to embeddings of one space, and the scale of their cosines.
#[derive(Debug)]
pub struct ContrastiveEncoder {
    inputs: Model<Boxed<dyn Layer>>,
    texts: Model<Boxed<dyn Layer<TokenBatch>>>,
    /// The learned log of the cosines' scale.
    logit_scale: StateKey,
}

impl ContrastiveEncoder {
    #[must_use]
    pub const fn new(
        inputs: Model<Boxed<dyn Layer>>,
        texts: Model<Boxed<dyn Layer<TokenBatch>>>,
        logit_scale: StateKey,
    ) -> Self {
        Self {
            inputs,
            texts,
            logit_scale,
        }
    }

    #[must_use]
    pub const fn inputs(&self) -> &Model<Boxed<dyn Layer>> {
        &self.inputs
    }

    #[must_use]
    pub const fn texts(&self) -> &Model<Boxed<dyn Layer<TokenBatch>>> {
        &self.texts
    }

    /// Returns each input's scaled cosine with each text, `[B][N]`, as `logits_per_image` and `logits_per_audio` are.
    ///
    /// # Errors
    ///
    /// Returns an error if the logit scale holds no value, or if running either tower fails.
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

/// An open-vocabulary detector, as OWL-ViT's `*ForObjectDetection` is: an image tower to each patch's class embedding, box, logit shift, and logit scale, `[B, P, D + 4 + 2]`, and a text tower to each query's embedding, `[N, D]`.
#[derive(Debug)]
pub struct OpenVocabularyDetector {
    images: Model<Boxed<dyn Layer>>,
    texts: Model<Boxed<dyn Layer<TokenBatch>>>,
}

/// One patch of an open-vocabulary detector: its score for each query, before the sigmoid, and its box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryDetection {
    pub logits: Vec<f32>,
    pub center_box: [f32; 4],
}

impl OpenVocabularyDetector {
    #[must_use]
    pub const fn new(
        images: Model<Boxed<dyn Layer>>,
        texts: Model<Boxed<dyn Layer<TokenBatch>>>,
    ) -> Self {
        Self { images, texts }
    }

    #[must_use]
    pub const fn images(&self) -> &Model<Boxed<dyn Layer>> {
        &self.images
    }

    /// Returns each image's patches, scored against each text, as OWL-ViT's class head scores them: cosines shifted, then scaled.
    ///
    /// # Errors
    ///
    /// Returns an error if the image tower's output is not `[B, P, N]`, or if running either tower fails.
    pub fn detect(&self, pixels: &Tensor, texts: &TokenBatch) -> Result<Vec<Vec<QueryDetection>>> {
        let output = self.images.forward(pixels)?;
        let &[_, patches, _] = output.shape().dimensions() else {
            return Err(SingeError::invalid_argument("patch detections, [B, P, N]").into());
        };
        let patches = usize::try_from(patches).map_err(|_| SingeError::out_of_range("patches"))?;
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
        Ok(host::rows(&detections, patches))
    }
}

/// A depth estimator, as `*ForDepthEstimation` heads are: its layers give each pixel's depth at the size the model reads, `[B, H, W]`.
#[derive(Debug)]
pub struct DepthEstimator {
    model: Model<Boxed<dyn Layer>>,
}

impl DepthEstimator {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>) -> Self {
        Self { model }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn Layer>> {
        &self.model
    }

    pub fn depth(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(pixels)?)
    }
}

/// A semantic segmenter, as `*ForSemanticSegmentation` heads are: its layers give each label's score at each point of a map, `[B, labels, h, w]`, which the pipeline resizes to the image.
#[derive(Debug)]
pub struct SemanticSegmenter {
    model: Forward<Boxed<dyn Layer>>,
    config: ClassifierConfig,
}

impl SemanticSegmenter {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, config: ClassifierConfig) -> Self {
        Self {
            model: Forward::Native(model),
            config,
        }
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    pub fn scores(&self, pixels: &Tensor) -> Result<Tensor> {
        self.model.forward(pixels)
    }
}

impl ExternalHead for SemanticSegmenter {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let (model, config) = External::classifier(source, function, 1)?;
        Ok(Self {
            model: Forward::External(model),
            config,
        })
    }
}

/// A super-resolution model, as `*ForImageSuperResolution` heads are: its layers take images in `[0, 1]` whose sides are whole windows and give them larger, `[B, 3, H · s, W · s]`.
#[derive(Debug)]
pub struct ImageUpscaler {
    model: Model<Boxed<dyn Layer>>,
    window: i64,
}

impl ImageUpscaler {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>, window: i64) -> Self {
        Self { model, window }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn Layer>> {
        &self.model
    }

    /// Returns the side its inputs' sides are multiples of.
    #[must_use]
    pub const fn window(&self) -> i64 {
        self.window
    }

    pub fn upscale(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(pixels)?)
    }
}
