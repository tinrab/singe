//! `video-classification`: a clip to its labels' scores, best first, from frames sampled evenly from its start, as `transformers`' pipeline samples them.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    head::VideoClassifier,
    model,
    processor::{image::ImagePreprocessor, video::Video},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        ranking::{ClassificationParameters, LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct VideoClassificationPipeline {
    model: VideoClassifier,
    processor: ImagePreprocessor,
    options: LoadOptions,
    /// The frames the model reads, as its config's `num_frames` gives them.
    frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct VideoClassificationParameters {
    #[serde(flatten)]
    pub classification: ClassificationParameters,
    /// The frames to sample, or the model's own count when `None`.
    pub num_frames: Option<usize>,
    /// The multiple of `num_frames` the samples spread evenly over, the first `num_frames × frame_sampling_rate` frames, 1 by default.
    pub frame_sampling_rate: usize,
}

impl Default for VideoClassificationParameters {
    fn default() -> Self {
        Self {
            classification: ClassificationParameters::default(),
            num_frames: None,
            frame_sampling_rate: 1,
        }
    }
}

/// The frames a model reads: its config's `num_frames`, or `frames_per_clip` where it names none, as V-JEPA 2's does.
#[derive(Debug, Deserialize)]
struct FrameCount {
    num_frames: Option<usize>,
    frames_per_clip: Option<usize>,
}

impl Pipeline for VideoClassificationPipeline {
    type Input = Video;

    type Parameters = VideoClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            options,
            frames: {
                let count = source.json::<FrameCount>("config.json")?;
                count.num_frames.or(count.frames_per_clip).ok_or_else(|| {
                    Error::unsupported("a video model whose config names no frame count")
                })?
            },
        })
    }

    fn run_one(&self, video: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let config = self.model.config();
        let frames = video
            .sample(
                parameters.num_frames.unwrap_or(self.frames),
                parameters.frame_sampling_rate,
            )?
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        // `[T, C, H, W]` of processed frames, as a batch of one clip.
        let pixels = self
            .processor
            .preprocess_batch(&frames, self.options.device, self.options.data_type)?
            .unsqueeze(0)?;
        let scores = self.model.scores(&pixels)?;
        let classification = parameters.classification;
        Ok(ranked(
            &scores[0],
            config.labels(),
            classification
                .function_to_apply
                .unwrap_or_else(|| config.function()),
            classification.top_k,
        ))
    }
}
