//! `zero-shot-object-detection`: an image and candidate labels to the objects found of each label, with scores and boxes in the image's pixels, best first, as `transformers`' pipeline post-processes OWL-ViT: each patch's sigmoid score for a label, kept above a threshold.

use serde::Deserialize;

use crate::{
    error::Result,
    head::{OpenVocabularyDetector, ScoreFunction},
    host, model,
    processor::{image::ImageTransforms, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        inputs::ImageCandidates,
        vision::object_detection::{BoundingBox, DetectedObject},
    },
};

#[derive(Debug)]
pub struct ZeroShotObjectDetectionPipeline {
    model: OpenVocabularyDetector,
    processor: ImageTransforms,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct ZeroShotObjectDetectionParameters {
    /// The score an object must exceed to be kept, 0.1 by default.
    pub threshold: f32,
    /// The most objects to keep, best first, or every one for `None`.
    pub top_k: Option<usize>,
}

impl Default for ZeroShotObjectDetectionParameters {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            top_k: None,
        }
    }
}

impl Pipeline for ZeroShotObjectDetectionPipeline {
    type Input = ImageCandidates;

    type Parameters = ZeroShotObjectDetectionParameters;

    type Output = Vec<DetectedObject>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let LoadOptions {
            device, data_type, ..
        } = self.options;
        let texts = input
            .candidate_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
        let pixels = self.processor.preprocess(&input.image, device, data_type)?;
        // Each label in turn, as the pipeline runs one per pass, then every object best first.
        let patches = &self.model.detect(&pixels, &tokens)?[0];
        let scores = patches
            .iter()
            .map(|patch| ScoreFunction::Sigmoid.apply(&patch.logits))
            .collect::<Vec<_>>();
        let mut objects = input
            .candidate_labels
            .iter()
            .enumerate()
            .flat_map(|(label, name)| {
                patches
                    .iter()
                    .zip(&scores)
                    .filter_map(move |(patch, scores)| {
                        let score = scores[label];
                        (score > parameters.threshold).then(|| DetectedObject {
                            score,
                            label: name.clone(),
                            bounds: BoundingBox::from_center(
                                patch.center_box,
                                input.image.width,
                                input.image.height,
                            ),
                        })
                    })
            })
            .collect::<Vec<_>>();
        host::best_first(
            &mut objects,
            parameters.top_k.unwrap_or(usize::MAX),
            |object| object.score,
        );
        Ok(objects)
    }
}
