// `depth-estimation`: an image to its depth at each pixel, resized to the image, and that depth as a grayscale image from nearest to farthest, as `transformers`' pipeline gives them.

use std::slice;

use serde::Serialize;
use singe::{DataType, Tensor, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    head::DepthEstimator,
    model,
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct DepthEstimationPipeline {
    model: DepthEstimator,
    processor: ImageTransforms,
    data_type: DataType,
}

#[derive(Debug, Clone, Serialize)]
pub struct DepthEstimate {
    // Each pixel's depth at the image's size, `[H, W]` of `F32`.
    #[serde(skip)]
    pub predicted_depth: Tensor,
    // The depth scaled from its least to its greatest value onto bytes.
    pub depth: Image,
}

impl Pipeline for DepthEstimationPipeline {
    const TASK: Task = Task::DepthEstimation;

    type Input = Image;

    type Parameters = ();

    type Output = DepthEstimate;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        let device = self.model.model().state().device()?;
        inputs
            .iter()
            .map(|image| {
                let pixels =
                    self.processor
                        .preprocess(slice::from_ref(image), device, self.data_type)?;
                // Resized bicubically to the image, as `post_process_depth_estimation` resizes it.
                let predicted_depth = self
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
                    .contiguous()?;
                let values = predicted_depth.to_vec::<f32>()?;
                if values.is_empty() {
                    return Err(Error::Media("an empty depth map".to_owned()));
                }

                let least = values.iter().copied().fold(f32::INFINITY, f32::min);
                let greatest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "values in [0, 255] truncate to bytes, as NumPy's `astype(\"uint8\")` does"
                )]
                let pixels = values
                    .iter()
                    .flat_map(|&value| {
                        let byte = ((value - least) / (greatest - least) * 255.0) as u8;
                        [byte; 3]
                    })
                    .collect();
                Ok(DepthEstimate {
                    predicted_depth,
                    depth: Image {
                        width: image.width,
                        height: image.height,
                        pixels,
                    },
                })
            })
            .collect()
    }
}
