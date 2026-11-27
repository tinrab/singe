//! `object-detection`: an image to the objects found in it, each a label, a score, and a box in the image's pixels, as `transformers`' pipeline post-processes a DETR-style detector.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::{DetectionScoring, ObjectDetector, ScoreFunction, argmax},
    model,
    processor::image::{Image, ImagePreprocessor},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct ObjectDetectionPipeline {
    model: ObjectDetector,
    processor: ImagePreprocessor,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct ObjectDetectionParameters {
    /// The score an object must exceed to be kept, 0.5 by default.
    pub threshold: f32,
}

impl Default for ObjectDetectionParameters {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

/// A box's corners in an image's pixels, truncated as `transformers` truncates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BoundingBox {
    pub xmin: i64,
    pub ymin: i64,
    pub xmax: i64,
    pub ymax: i64,
}

impl BoundingBox {
    /// Returns the corners of a normalized `[cx, cy, w, h]` box in an image of `width` and `height` pixels.
    #[must_use]
    pub fn from_center([cx, cy, w, h]: [f32; 4], width: u32, height: u32) -> Self {
        let (width, height) = (width as f32, height as f32);
        Self {
            xmin: ((cx - 0.5 * w) * width) as i64,
            ymin: ((cy - 0.5 * h) * height) as i64,
            xmax: ((cx + 0.5 * w) * width) as i64,
            ymax: ((cy + 0.5 * h) * height) as i64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetectedObject {
    pub score: f32,
    pub label: String,
    #[serde(rename = "box")]
    pub bounds: BoundingBox,
}

impl Pipeline for ObjectDetectionPipeline {
    type Input = Image;

    type Parameters = ObjectDetectionParameters;

    type Output = Vec<DetectedObject>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load_served(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            options,
        })
    }

    /// Detects the objects in `image`, as the model's scoring picks them, and keeps those scoring above the threshold.
    /// Under [`DetectionScoring::NoObject`] each query's labels go through a softmax with the no-object score, and a query is kept with its best real label, in query order.
    /// Under [`DetectionScoring::SigmoidTopK`] every query's label scores go through a sigmoid, and the best `k` pairs of a query and a label are kept, best first.
    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let labels = self.model.config().labels();
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let queries = &self.model.detect(&pixels)?[0];
        let detected = |query: usize, label: usize, score: f32| DetectedObject {
            score,
            label: labels[label].clone(),
            bounds: BoundingBox::from_center(queries[query].center_box, image.width, image.height),
        };
        Ok(match self.model.scoring() {
            DetectionScoring::NoObject => queries
                .iter()
                .enumerate()
                .filter_map(|(index, query)| {
                    let scores = ScoreFunction::Softmax.apply(&query.scores);
                    let (label, score) =
                        argmax(scores[..labels.len()].iter().copied().enumerate())?;
                    (score > parameters.threshold).then(|| detected(index, label, score))
                })
                .collect(),
            DetectionScoring::SigmoidTopK(k) => {
                let mut ranked = queries
                    .iter()
                    .enumerate()
                    .flat_map(|(index, query)| {
                        ScoreFunction::Sigmoid
                            .apply(&query.scores)
                            .into_iter()
                            .enumerate()
                            .map(move |(label, score)| (index, label, score))
                    })
                    .collect::<Vec<_>>();
                ranked.sort_by(|left, right| right.2.total_cmp(&left.2));
                ranked
                    .into_iter()
                    .take(k)
                    .filter(|&(_, _, score)| score > parameters.threshold)
                    .map(|(index, label, score)| detected(index, label, score))
                    .collect()
            }
        })
    }
}
