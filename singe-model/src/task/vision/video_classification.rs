// `video-classification`: a clip to its labels' scores, best first, from frames sampled evenly from its start, as `transformers`' pipeline samples them.

use serde::Deserialize;
use singe::DataType;

use crate::{
    error::Result,
    head::{ScoreFunction, VideoClassifier},
    model,
    processing::{image::ImageTransforms, video::Video},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::image_classification::{LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct VideoClassificationPipeline {
    model: VideoClassifier,
    processor: ImageTransforms,
    data_type: DataType,
    // The frames the model reads, as its config's `num_frames` gives them.
    frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct VideoClassificationParameters {
    pub top_k: usize,
    pub function_to_apply: Option<ScoreFunction>,
    // The frames to sample, or the model's own count when absent.
    pub num_frames: Option<usize>,
    // The frames between two samples.
    pub frame_sampling_rate: usize,
}

impl Default for VideoClassificationParameters {
    fn default() -> Self {
        Self {
            top_k: 5,
            function_to_apply: None,
            num_frames: None,
            frame_sampling_rate: 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct FrameCount {
    num_frames: usize,
}

impl Pipeline for VideoClassificationPipeline {
    const TASK: Task = Task::VideoClassification;

    type Input = Video;

    type Parameters = VideoClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
            frames: source.json::<FrameCount>("config.json")?.num_frames,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let classifier = &self.model.0;
        let device = classifier.model().state().device()?;
        let config = classifier.config();
        let function = parameters.function_to_apply.unwrap_or(config.function);
        let count = parameters.num_frames.unwrap_or(self.frames);
        inputs
            .iter()
            .map(|video| {
                let frames = video
                    .sample(count, parameters.frame_sampling_rate)?
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                // `[T, C, H, W]` of processed frames, as a batch of one clip.
                let pixels = self
                    .processor
                    .preprocess(&frames, device, self.data_type)?
                    .unsqueeze(0)?;
                let scores = classifier.scores(&pixels)?;
                Ok(ranked(
                    &scores[0],
                    &config.labels,
                    function,
                    parameters.top_k,
                ))
            })
            .collect()
    }
}
