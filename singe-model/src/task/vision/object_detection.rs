// `object-detection`: an image to the objects found in it, each a label, a score, and a box in the image's pixels, as `transformers`' pipeline post-processes a DETR-style detector.

use std::slice;

use serde::{Deserialize, Serialize};
use singe::DataType;

use crate::{
    error::Result,
    head::{ObjectDetector, ScoreFunction, argmax},
    model,
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct ObjectDetectionPipeline {
    model: ObjectDetector,
    processor: ImageTransforms,
    data_type: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct ObjectDetectionParameters {
    // The least score an object keeps.
    pub threshold: f32,
}

impl Default for ObjectDetectionParameters {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

// A box's corners in an image's pixels, truncated as `transformers` truncates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BoundingBox {
    pub xmin: i64,
    pub ymin: i64,
    pub xmax: i64,
    pub ymax: i64,
}

impl BoundingBox {
    // The corners of a normalized `[cx, cy, w, h]` box in an image of `width` and `height` pixels.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`transformers` truncates box corners to integers"
    )]
    #[expect(
        clippy::suboptimal_flops,
        reason = "rounds each product as the reference does, which the verifications compare with"
    )]
    pub fn from_center([cx, cy, w, h]: [f32; 4], width: u32, height: u32) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "image sides are far below 2^24")]
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
    const TASK: Task = Task::ObjectDetection;

    type Input = Image;

    type Parameters = ObjectDetectionParameters;

    type Output = Vec<DetectedObject>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    // Each query's labels go through a softmax with the no-object score, whose best real label it keeps above the threshold.
    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.model().state().device()?;
        let labels = &self.model.config().labels;
        inputs
            .iter()
            .map(|image| {
                let pixels =
                    self.processor
                        .preprocess(slice::from_ref(image), device, self.data_type)?;
                Ok(self.model.detect(&pixels)?[0]
                    .iter()
                    .filter_map(|query| {
                        let scores = ScoreFunction::Softmax.apply(&query.scores);
                        let (label, score) = argmax(&scores[..labels.len()])?;
                        (score > parameters.threshold).then(|| DetectedObject {
                            score,
                            label: labels[label].clone(),
                            bounds: BoundingBox::from_center(
                                query.center_box,
                                image.width,
                                image.height,
                            ),
                        })
                    })
                    .collect())
            })
            .collect()
    }
}
