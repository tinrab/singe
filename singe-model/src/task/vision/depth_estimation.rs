//! `depth-estimation`: an image to its depth as a grayscale image of the image's size, as `transformers`' pipeline gives its `depth`: the model's depth resized to the image, then scaled from its least value, black, to its greatest, white.
//! The pipeline does not return `transformers`' `predicted_depth`.

use serde::Serialize;
use singe::{DataType, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    head::DepthEstimator,
    model,
    processor::image::{Image, ImagePreprocessor},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct DepthEstimationPipeline {
    model: DepthEstimator,
    processor: ImagePreprocessor,
    options: LoadOptions,
}

#[derive(Debug, Clone, Serialize)]
pub struct DepthEstimate {
    /// The depth scaled from its least to its greatest value onto bytes.
    pub depth: Image,
}

impl Pipeline for DepthEstimationPipeline {
    type Input = Image;

    type Parameters = ();

    type Output = DepthEstimate;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load_served(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        // Resized bicubically to the image, as `post_process_depth_estimation` resizes it.
        let values = self
            .model
            .depth(&pixels)?
            .cast(DataType::F32)?
            .unsqueeze(1)?
            .interpolate(
                (i64::from(image.height), i64::from(image.width)),
                InterpolateMode::Bicubic,
            )?
            .squeeze(1)?
            .squeeze(0)?
            .contiguous()?
            .to_vec::<f32>()?;
        if values.is_empty() {
            return Err(Error::invalid("an image without pixels"));
        }

        let least = values.iter().copied().fold(f32::INFINITY, f32::min);
        let greatest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bytes = values
            .iter()
            .map(|&value| ((value - least) / (greatest - least) * 255.0) as u8);
        Ok(DepthEstimate {
            depth: Image::grayscale(image.width, image.height, bytes),
        })
    }
}
