//! `zero-shot-object-detection`: an image and candidate labels to the objects found of each label, with scores and boxes in the image's pixels, best first, as `transformers`' pipeline post-processes them.
//! OWL-ViT scores each patch for each label; Grounding DINO runs once per label and scores each query by its best token; either score goes through a sigmoid and is kept above a threshold.

use serde::Deserialize;

use crate::{
    error::Result,
    head::{OpenVocabularyDetector, ScoreFunction},
    host, model,
    processor::{image::ImagePreprocessor, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        inputs::ZeroShotImageInput,
        vision::object_detection::{BoundingBox, DetectedObject},
    },
};

#[derive(Debug)]
pub struct ZeroShotObjectDetectionPipeline {
    model: OpenVocabularyDetector,
    processor: ImagePreprocessor,
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
    type Input = ZeroShotImageInput;

    type Parameters = ZeroShotObjectDetectionParameters;

    type Output = Vec<DetectedObject>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let LoadOptions {
            device, data_type, ..
        } = self.options;
        let pixels = self.processor.preprocess(&input.image, device, data_type)?;
        // Each label in turn, as the pipeline runs one per pass, then every object best first.
        let detections = match &self.model {
            OpenVocabularyDetector::Patches { .. } => {
                let texts = input
                    .candidate_labels
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
                self.model.detect_patches(&pixels, &tokens)?
            }
            OpenVocabularyDetector::Grounded { .. } => {
                let mut detections = Vec::new();
                for (label, text) in input.candidate_labels.iter().enumerate() {
                    let tokens = self.tokenizer.encode(text, true)?;
                    detections.extend(self.model.detect_grounded(&pixels, &tokens, label)?);
                }
                detections
            }
        };
        let mut objects = detections
            .into_iter()
            .filter_map(|detection| {
                let score = ScoreFunction::Sigmoid.apply(&[detection.logit])[0];
                (score > parameters.threshold).then(|| DetectedObject {
                    score,
                    label: input.candidate_labels[detection.label].clone(),
                    bounds: BoundingBox::from_center(
                        detection.center_box,
                        input.image.width,
                        input.image.height,
                    ),
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
