//! `image-segmentation`: an image to labeled masks of its regions, here a semantic segmenter's: each pixel takes its best label, after the scores are resized to the image, and each label present gets a mask, as `transformers`' pipeline gives them.

use std::collections::BTreeSet;

use serde::Serialize;
use singe::{DataType, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    head::SemanticSegmenter,
    model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, vision::pixels::mask},
};

#[derive(Debug)]
pub struct ImageSegmentationPipeline {
    model: SemanticSegmenter,
    processor: ImageTransforms,
    options: LoadOptions,
}

/// A region: its label, a score when the model gives one, and its mask, white where the region is.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Segment {
    pub score: Option<f32>,
    pub label: String,
    pub mask: Image,
}

impl Pipeline for ImageSegmentationPipeline {
    type Input = Image;

    type Parameters = ();

    type Output = Vec<Segment>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let labels = self.model.config().labels();
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        // Resized bilinearly to the image, as `post_process_semantic_segmentation` resizes the scores.
        let (_, best) = self
            .model
            .scores(&pixels)?
            .cast(DataType::F32)?
            .interpolate(
                (i64::from(image.height), i64::from(image.width)),
                InterpolateMode::Bilinear,
            )?
            .topk(1, 1, true, false)?;
        let best = best.contiguous()?.to_vec::<i64>()?;
        let present = best.iter().copied().collect::<BTreeSet<_>>();
        present
            .into_iter()
            .map(|id| {
                let label = usize::try_from(id)
                    .ok()
                    .and_then(|id| labels.get(id))
                    .ok_or_else(|| {
                        Error::unsupported(format!("class {id} beyond the config's labels"))
                    })?;
                Ok(Segment {
                    score: None,
                    label: label.clone(),
                    mask: mask(
                        image.width,
                        image.height,
                        best.iter().map(|&best| best == id),
                    ),
                })
            })
            .collect()
    }
}
