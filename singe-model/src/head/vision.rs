//! Heads over images and clips: classification of an image or a clip, object detection, the contrastive models zero-shot classification and detection run, depth estimation, semantic segmentation, and super-resolution.
//! A model builds its layers to these outputs itself, since vision models differ in what their heads read: a class token, a mean of the patches, or detection tokens.

use std::{ops::Deref, sync::Arc};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Model, StateKey, TokenBatch},
};

use crate::{
    error::{Error, Result},
    format::transformers::{classifier_config, detection_scoring},
    function::{
        TensorFunction,
        forward::{External, ExternalHead, Forward},
    },
    head::{Boxed, ClassifierConfig, ScoreFunction},
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

/// A clip encoder, as V-JEPA 2's is: from clips, `[B, T, C, H, W]`, to each token's state, `[B, N, D]`.
#[derive(Debug)]
pub struct VideoEncoder(pub Model<Boxed<dyn Layer>>);

/// How a detector's label scores become detections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionScoring {
    /// A softmax over each query's labels and a last score for no object, keeping each query's best label, as DETR and YOLOS score.
    NoObject,
    /// The sigmoid of each query's score for each label, keeping the best `k` of them over all queries, best first, as Deformable DETR scores.
    SigmoidTopK(usize),
}

impl DetectionScoring {
    /// Returns the scores each query has for `labels` labels.
    #[must_use]
    pub const fn columns(self, labels: usize) -> usize {
        match self {
            Self::NoObject => labels + 1,
            Self::SigmoidTopK(_) => labels,
        }
    }
}

/// One query of a detector: each label's score, then no object's under [`DetectionScoring::NoObject`], and its box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub scores: Vec<f32>,
    pub center_box: [f32; 4],
}

/// An object detector, as `*ForObjectDetection` heads are: its layers give each query's scores, as many as its scoring has columns, then its box, `[B, Q, scores + 4]`.
#[derive(Debug)]
pub struct ObjectDetector {
    model: Forward<Boxed<dyn Layer>>,
    config: ClassifierConfig,
    scoring: DetectionScoring,
}

impl ObjectDetector {
    #[must_use]
    pub const fn new(
        model: Model<Boxed<dyn Layer>>,
        config: ClassifierConfig,
        scoring: DetectionScoring,
    ) -> Self {
        Self {
            model: Forward::Native(model),
            config,
            scoring,
        }
    }

    #[must_use]
    pub const fn scoring(&self) -> DetectionScoring {
        self.scoring
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
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
        let output = match &self.model {
            Forward::Native(model) => model.forward(pixels)?,
            // `optimum` names the label scores and the boxes as two outputs.
            Forward::External(model) => Tensor::concat(&model.single_outputs(pixels)?, 2)?,
        };
        let &[_, queries, _] = output.shape().dimensions() else {
            return Err(SingeError::invalid_argument("detections, [B, Q, N]").into());
        };
        let queries = usize::try_from(queries).map_err(|_| SingeError::out_of_range("queries"))?;
        let labels = self.scoring.columns(self.config.labels().len());
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

impl ExternalHead for ObjectDetector {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let config = classifier_config(source)?;
        let scoring = detection_scoring(source)?;
        let model = External::create(function, &["logits", "pred_boxes"])?;
        let scores = model
            .output("logits")
            .and_then(|spec| spec.shape.get(2).copied());
        let columns = scoring.columns(config.labels().len());
        if scores.and_then(|scores| usize::try_from(scores).ok()) != Some(columns) {
            return Err(Error::invalid(format!(
                "output logits of {scores:?} scores per query, where the config's labels and scoring give {columns}"
            )));
        }

        Ok(Self {
            model: Forward::External(model),
            config,
            scoring,
        })
    }
}

/// A contrastive model, as CLIP's and CLAP's `*Model` are: a tower of images or clips and a tower of texts to embeddings of one space, and the scale of their cosines.
#[derive(Debug)]
pub struct ContrastiveEncoder {
    inputs: Model<Boxed<dyn Layer>>,
    texts: Model<Boxed<dyn Layer<TokenBatch>>>,
    /// The learned log of the cosines' scale.
    logit_scale: StateKey,
    /// What `SigLIP` adds to the scaled cosines, whose sigmoid then scores each text alone.
    logit_bias: Option<StateKey>,
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
            logit_bias: None,
        }
    }

    /// Adds `logit_bias` to the scaled cosines and scores them by sigmoid, as `SigLIP` does.
    #[must_use]
    pub fn with_sigmoid_scores(mut self, logit_bias: StateKey) -> Self {
        self.logit_bias = Some(logit_bias);
        self
    }

    /// Returns how the logits become scores: a sigmoid per text with a logit bias, and a softmax over the texts without.
    #[must_use]
    pub const fn score_function(&self) -> ScoreFunction {
        if self.logit_bias.is_some() {
            ScoreFunction::Sigmoid
        } else {
            ScoreFunction::Softmax
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
        let scalar = |key: &StateKey| -> Result<f32> {
            self.inputs
                .state()
                .tensor(key)?
                .cast(DataType::F32)?
                .to_vec::<f32>()?
                .first()
                .copied()
                .ok_or_else(|| Error::unsupported("a logit scale or bias without a value"))
        };
        let scale = scalar(&self.logit_scale)?.exp();
        let scaled = images
            .matmul(&texts.transpose(0, 1)?)?
            .mul_scalar(f64::from(scale))?;
        match &self.logit_bias {
            Some(bias) => rows(&scaled.add_scalar(f64::from(scalar(bias)?))?),
            None => rows(&scaled),
        }
    }
}

/// What a grounded detector reads: images and one text's tokens, with the positions and the mask the text's phrases give.
#[derive(Debug, Clone)]
pub struct GroundingInput {
    /// `[B, 3, H, W]`.
    pub pixels: Tensor,
    /// `[B, T]` in `I64`.
    pub tokens: Tensor,
    /// Each token's position within its phrase, `[B, T]` in `I64`.
    pub positions: Tensor,
    /// `[B, 1, T, T]`: zero where a token reads another, `F32`'s lowest value elsewhere.
    pub mask: Tensor,
}

/// An open-vocabulary detector, as `*ForObjectDetection` heads of the zero-shot task are.
#[derive(Debug)]
pub enum OpenVocabularyDetector {
    /// OWL-ViT's: an image tower to each patch's class embedding, box, logit shift, and logit scale, `[B, P, D + 4 + 2]`, and a text tower to each query's embedding, `[N, D]`.
    Patches {
        images: Model<Boxed<dyn Layer>>,
        texts: Model<Boxed<dyn Layer<TokenBatch>>>,
    },
    /// Grounding DINO's: a model from images and one text to each query's logit for each of the text's tokens, then its box, `[B, Q, T + 4]`, run once per label.
    /// `special` are the tokens that end phrases.
    Grounded {
        model: Model<Boxed<dyn Layer<GroundingInput>>>,
        special: Vec<i64>,
    },
}

/// A box an open-vocabulary detector gives a label: the label's index, the box's logit before the sigmoid, and the box, normalized `[cx, cy, w, h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelDetection {
    pub label: usize,
    pub logit: f32,
    pub center_box: [f32; 4],
}

impl OpenVocabularyDetector {
    /// Returns the first image's boxes for each label of `texts`, label by label, as OWL-ViT's class head scores its patches: each patch's cosine with each text, shifted, then scaled.
    ///
    /// # Errors
    ///
    /// Returns an error if the detector is grounded, if the image tower's output is not `[B, P, N]`, or if running either tower fails.
    pub fn detect_patches(
        &self,
        pixels: &Tensor,
        texts: &TokenBatch,
    ) -> Result<Vec<LabelDetection>> {
        let Self::Patches {
            images,
            texts: tower,
        } = self
        else {
            return Err(Error::unsupported("a grounded detector's patches"));
        };
        let output = images.forward(pixels)?;
        let &[_, patches, _] = output.shape().dimensions() else {
            return Err(SingeError::invalid_argument("patch detections, [B, P, N]").into());
        };
        let patches = usize::try_from(patches).map_err(|_| SingeError::out_of_range("patches"))?;
        let queries = rows(&tower.forward_group(texts)?)?;
        let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let queries = queries
            .iter()
            .map(|query| {
                let norm = norm(query) + 1e-6;
                query.iter().map(|value| value / norm).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let width = queries.first().map_or(0, Vec::len);
        let mut detections = Vec::new();
        for row in rows(&output)?.into_iter().take(patches) {
            let (class, rest) = row.split_at(width);
            let norm = norm(class) + 1e-6;
            let (shift, scale) = (rest[4], rest[5]);
            // ELU plus one.
            let scale = if scale > 0.0 {
                scale + 1.0
            } else {
                scale.exp()
            };
            for (label, query) in queries.iter().enumerate() {
                let cosine = class
                    .iter()
                    .zip(query)
                    .map(|(class, query)| class / norm * query)
                    .sum::<f32>();
                detections.push(LabelDetection {
                    label,
                    logit: (cosine + shift) * scale,
                    center_box: [rest[0], rest[1], rest[2], rest[3]],
                });
            }
        }

        detections.sort_by_key(|detection| detection.label);
        Ok(detections)
    }

    /// Returns the first image's boxes for label `label`, whose text is `tokens` with its special tokens, as Grounding DINO scores its queries: each query's best logit over the text's tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the detector is not grounded, if the model's output is not `[B, Q, T + 4]`, or if running it fails.
    pub fn detect_grounded(
        &self,
        pixels: &Tensor,
        tokens: &[i64],
        label: usize,
    ) -> Result<Vec<LabelDetection>> {
        let Self::Grounded { model, special } = self else {
            return Err(Error::unsupported("a patch detector's grounded queries"));
        };
        let device = pixels.device();
        let count = i64::try_from(tokens.len()).map_err(|_| SingeError::out_of_range("tokens"))?;
        let (positions, mask) = phrases(tokens, special);
        let input = GroundingInput {
            pixels: pixels.clone(),
            tokens: Tensor::from_vec(device, Shape::create([1, count])?, tokens.to_vec())?,
            positions: Tensor::from_vec(device, Shape::create([1, count])?, positions)?,
            mask: Tensor::from_vec(device, Shape::create([1, 1, count, count])?, mask)?,
        };
        let output = model.layer().forward(&input, model.state())?;
        let width = tokens.len();
        Ok(rows(&output)?
            .into_iter()
            .map(|row| LabelDetection {
                label,
                logit: row[..width]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
                center_box: [row[width], row[width + 1], row[width + 2], row[width + 3]],
            })
            .collect())
    }
}

/// Returns each token's position within its phrase and the mask of which tokens read which, `[T, T]` with zero where one reads another and `F32`'s lowest value elsewhere, as `generate_masks_with_special_tokens_and_transfer_map` makes them.
///
/// A phrase is the tokens after a special token up to and including the next one; its tokens read each other and count their positions from zero.
/// Every token reads itself, and a phrase that ends at the first token, at the last, or past it is no phrase, so its tokens read only themselves at position zero.
fn phrases(tokens: &[i64], special: &[i64]) -> (Vec<i64>, Vec<f32>) {
    let length = tokens.len();
    let is_special = |index: usize| special.contains(&tokens[index]);
    let mut next = vec![length; length];
    for index in (0..length).rev() {
        next[index] = if is_special(index) {
            index
        } else if index + 1 < length {
            next[index + 1]
        } else {
            length
        };
    }

    let mut previous = Vec::with_capacity(length);
    let mut last: i64 = -1;
    for index in 0..length {
        previous.push(last);
        if is_special(index) {
            last = i64::try_from(index).unwrap_or(i64::MAX);
        }
    }

    let is_phrase =
        |index: usize| next[index] != 0 && next[index] + 1 != length && next[index] != length;
    let positions = (0..length)
        .map(|index| {
            if is_phrase(index) {
                (i64::try_from(index).unwrap_or(i64::MAX) - previous[index] - 1).max(0)
            } else {
                0
            }
        })
        .collect();
    let mask = (0..length)
        .flat_map(|row| (0..length).map(move |column| (row, column)))
        .map(|(row, column)| {
            if row == column || (next[row] == next[column] && is_phrase(column)) {
                0.0
            } else {
                f32::MIN
            }
        })
        .collect();
    (positions, mask)
}

/// A depth estimator, as `*ForDepthEstimation` heads are: its layers give each pixel's depth at the size the model reads, `[B, H, W]`.
#[derive(Debug)]
pub struct DepthEstimator {
    model: Forward<Boxed<dyn Layer>>,
}

impl DepthEstimator {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn Layer>>) -> Self {
        Self {
            model: Forward::Native(model),
        }
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
    }

    pub fn depth(&self, pixels: &Tensor) -> Result<Tensor> {
        self.model.forward(pixels)
    }
}

impl ExternalHead for DepthEstimator {
    fn external(_source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        Ok(Self {
            model: Forward::External(External::create(function, &["predicted_depth"])?),
        })
    }
}

/// How an image segmenter's layers lay out what they give.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmenterLayout {
    /// Each label's score at each point of a map, `[B, labels, h, w]`, as `*ForSemanticSegmentation` heads give them.
    Semantic,
    /// Each query's label scores, the last for no object, then its mask's logits at each point of a map `stride` times smaller than the image, `[B, Q, labels + 1 + h · w]`, as `*ForUniversalSegmentation` heads give them.
    Universal { stride: i64 },
}

/// The queries of a universal segmenter: each one's label scores, the last for no object, `[B, Q, labels + 1]`, and its mask's logits, `[B, Q, h, w]`.
#[derive(Debug, Clone)]
pub struct SegmentQueries {
    pub classes: Tensor,
    pub masks: Tensor,
}

/// An image segmenter, as `*ForSemanticSegmentation` and `*ForUniversalSegmentation` heads are, whose layers give what its layout says.
#[derive(Debug)]
pub struct ImageSegmenter {
    model: Forward<Boxed<dyn Layer>>,
    config: ClassifierConfig,
    layout: SegmenterLayout,
}

impl ImageSegmenter {
    #[must_use]
    pub const fn new(
        model: Model<Boxed<dyn Layer>>,
        config: ClassifierConfig,
        layout: SegmenterLayout,
    ) -> Self {
        Self {
            model: Forward::Native(model),
            config,
            layout,
        }
    }

    pub fn get_model(&self) -> Result<&Model<Boxed<dyn Layer>>> {
        self.model.get_native()
    }

    #[must_use]
    pub const fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    #[must_use]
    pub const fn layout(&self) -> SegmenterLayout {
        self.layout
    }

    /// Returns a semantic segmenter's scores, `[B, labels, h, w]`.
    pub fn scores(&self, pixels: &Tensor) -> Result<Tensor> {
        self.model.forward(pixels)
    }

    /// Returns a universal segmenter's queries for `pixels`, `[B, 3, H, W]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the segmenter is semantic, if its output is not `[B, Q, N]`, or if running the model fails.
    pub fn queries(&self, pixels: &Tensor) -> Result<SegmentQueries> {
        let SegmenterLayout::Universal { stride } = self.layout else {
            return Err(Error::unsupported("queries of a semantic segmenter"));
        };
        match &self.model {
            Forward::Native(model) => {
                let output = model.forward(pixels)?;
                let &[batch, queries, _] = output.shape().dimensions() else {
                    return Err(SingeError::invalid_argument("segment queries, [B, Q, N]").into());
                };
                let &[_, _, height, width] = pixels.shape().dimensions() else {
                    return Err(SingeError::invalid_argument("pixels, [B, 3, H, W]").into());
                };
                let scores = self.config.label_count()? + 1;
                let (height, width) = (height / stride, width / stride);
                Ok(SegmentQueries {
                    classes: output.narrow(2, 0, scores)?,
                    masks: output
                        .narrow(2, scores, height * width)?
                        .reshape([batch, queries, height, width])?,
                })
            }
            Forward::External(model) => {
                let [classes, masks] = <[Tensor; 2]>::try_from(model.single_outputs(pixels)?)
                    .map_err(|_| Error::invalid("a universal segmenter without two outputs"))?;
                Ok(SegmentQueries { classes, masks })
            }
        }
    }
}

impl ExternalHead for ImageSegmenter {
    fn external(source: &Source, function: Arc<dyn TensorFunction>) -> Result<Self> {
        let universal = source
            .classes()?
            .iter()
            .any(|class| class.ends_with("ForUniversalSegmentation"));
        if universal {
            let model =
                External::create(function, &["class_queries_logits", "masks_queries_logits"])?;
            return Ok(Self {
                model: Forward::External(model),
                config: classifier_config(source)?,
                // The file gives the masks as an output of their own, so their stride is never read.
                layout: SegmenterLayout::Universal { stride: 1 },
            });
        }

        let (model, config) = External::classifier(source, function, 1)?;
        Ok(Self {
            model: Forward::External(model),
            config,
            layout: SegmenterLayout::Semantic,
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
