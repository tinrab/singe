//! `keypoint-detection`: an image to its keypoints, their scores, and their descriptors, at the image's own pixels, as `transformers`' `post_process_keypoint_detection` gives them; `transformers` has no pipeline for it.

use serde::Serialize;
use singe::{Error as SingeError, shape::spatial::Spatial2d};

use crate::{
    error::Result,
    head::KeypointDetector,
    model,
    processor::image::{Image, ImagePreprocessor},
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::rescaled_point},
};

#[derive(Debug)]
pub struct KeypointDetectionPipeline {
    model: KeypointDetector,
    processor: ImagePreprocessor,
    options: LoadOptions,
}

/// An image's keypoints at its own whole pixels, `[x, y]`, their scores, and their descriptors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KeypointDetectionOutput {
    pub keypoints: Vec<[i32; 2]>,
    pub scores: Vec<f32>,
    pub descriptors: Vec<Vec<f32>>,
}

impl Pipeline for KeypointDetectionPipeline {
    type Input = Image;

    type Parameters = ();

    type Output = KeypointDetectionOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let &[_, _, height, width] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("images, [B, C, H, W]").into());
        };
        let keypoints = self.model.detect(&pixels)?.remove(0);
        Ok(KeypointDetectionOutput {
            keypoints: keypoints
                .keypoints
                .iter()
                .map(|&point| rescaled_point(point, Spatial2d::new(height, width), image))
                .collect(),
            scores: keypoints.scores,
            descriptors: keypoints.descriptors,
        })
    }
}
