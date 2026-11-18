// `keypoint-detection`: an image to its keypoints, their scores, and their descriptors, at the image's own pixels, as `transformers`' `post_process_keypoint_detection` gives them; `transformers` has no pipeline for it.

use std::slice;

use serde::Serialize;
use singe::DataType;

use crate::{
    error::Result,
    head::{KeypointDetector, Keypoints},
    model,
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct KeypointDetectionPipeline {
    model: KeypointDetector,
    processor: ImageTransforms,
    data_type: DataType,
}

impl KeypointDetectionPipeline {
    // Detects `image`'s keypoints at the pixels the model reads, whose size it returns too, `(height, width)`.
    fn detect(&self, image: &Image) -> Result<(Keypoints, (i64, i64))> {
        let device = self.model.model().state().device()?;
        let pixels = self
            .processor
            .preprocess(slice::from_ref(image), device, self.data_type)?;
        let dimensions = pixels.shape().dimensions();
        let size = (dimensions[2], dimensions[3]);
        let keypoints = self.model.detect(&pixels)?.remove(0);
        Ok((keypoints, size))
    }
}

// An image's keypoints at its own whole pixels, `[x, y]`, their scores, and their descriptors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetectedKeypoints {
    pub keypoints: Vec<[i32; 2]>,
    pub scores: Vec<f32>,
    pub descriptors: Vec<Vec<f32>>,
}

// Moves a keypoint at the pixels of an image of `from` pixels, `(height, width)`, to `image`'s, `[x, y]`, through coordinates relative to each side in `F32`, truncated to whole pixels, as `post_process_keypoint_detection` and `post_process_keypoint_matching` compute them.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "image sides are far below 2^24, and `transformers` truncates to `int32`"
)]
pub(crate) fn rescaled_point([x, y]: [f32; 2], from: (i64, i64), image: &Image) -> [i32; 2] {
    let (height, width) = (from.0 as f32, from.1 as f32);
    [
        (x / width * image.width as f32) as i32,
        (y / height * image.height as f32) as i32,
    ]
}

// Moves keypoints at the pixels of an image of `from` pixels to `image`'s, as `rescaled_point` moves each.
fn rescaled(keypoints: Keypoints, from: (i64, i64), image: &Image) -> DetectedKeypoints {
    DetectedKeypoints {
        keypoints: keypoints
            .keypoints
            .iter()
            .map(|&point| rescaled_point(point, from, image))
            .collect(),
        scores: keypoints.scores,
        descriptors: keypoints.descriptors,
    }
}

impl Pipeline for KeypointDetectionPipeline {
    const TASK: Task = Task::KeypointDetection;

    type Input = Image;

    type Parameters = ();

    type Output = DetectedKeypoints;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|image| {
                let (keypoints, size) = self.detect(image)?;
                Ok(rescaled(keypoints, size, image))
            })
            .collect()
    }
}
