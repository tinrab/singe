// `zero-shot-object-detection`: an image and candidate labels to the objects found of each label, with scores and boxes in the image's pixels, best first, as `transformers`' pipeline post-processes OWL-ViT: each patch's sigmoid score for a label, kept above a threshold.

use std::slice;

use serde::Deserialize;
use singe::DataType;

use crate::{
    error::Result,
    head::{OpenVocabularyDetector, ScoreFunction},
    model,
    processing::{
        image::{Image, ImageTransforms},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::object_detection::{BoundingBox, DetectedObject},
    },
};

#[derive(Debug)]
pub struct ZeroShotObjectDetectionPipeline {
    model: OpenVocabularyDetector,
    processor: ImageTransforms,
    tokenizer: TextTokenizer,
    data_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ZeroShotObjectInput {
    pub image: Image,
    pub candidate_labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default)]
pub struct ZeroShotObjectDetectionParameters {
    // The least score an object keeps.
    pub threshold: f32,
    // Keeps the best this many objects, or all when absent.
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
    const TASK: Task = Task::ZeroShotObjectDetection;

    type Input = ZeroShotObjectInput;

    type Parameters = ZeroShotObjectDetectionParameters;

    type Output = Vec<DetectedObject>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            data_type: options.data_type,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.images().state().device()?;
        inputs
            .iter()
            .map(|input| {
                let texts = input
                    .candidate_labels
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
                let pixels = self.processor.preprocess(
                    slice::from_ref(&input.image),
                    device,
                    self.data_type,
                )?;
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
                objects.sort_by(|left, right| right.score.total_cmp(&left.score));
                objects.truncate(parameters.top_k.unwrap_or(usize::MAX));
                Ok(objects)
            })
            .collect()
    }
}
