//! `video-feature-extraction`: a clip to its encoder's token states, or their mean, as `VJEPA2Model.get_vision_features` gives them; `transformers` has no pipeline for it.

use serde::Deserialize;
use singe::DataType;

use crate::{
    error::{Error, Result},
    head::VideoEncoder,
    host, model,
    processor::{image::ImagePreprocessor, video::Video},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct VideoFeatureExtractionPipeline {
    model: VideoEncoder,
    processor: ImagePreprocessor,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct VideoFeatureExtractionParameters {
    /// The frames to sample evenly, or every frame of the clip when `None`.
    pub num_frames: Option<usize>,
    /// The multiple of `num_frames` the samples spread evenly over, 1 by default.
    pub frame_sampling_rate: usize,
    /// Whether to return the mean of the tokens' states rather than each token's.
    pub pool: bool,
}

impl Default for VideoFeatureExtractionParameters {
    fn default() -> Self {
        Self {
            num_frames: None,
            frame_sampling_rate: 1,
            pool: false,
        }
    }
}

impl Pipeline for VideoFeatureExtractionPipeline {
    type Input = Video;

    type Parameters = VideoFeatureExtractionParameters;

    /// Each token's state, or one row of their mean when pooled.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImagePreprocessor::load(source)?,
            options,
        })
    }

    fn run_one(&self, video: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let frames = match parameters.num_frames {
            Some(count) => video
                .sample(count, parameters.frame_sampling_rate)?
                .into_iter()
                .cloned()
                .collect(),
            None => video.frames.clone(),
        };
        // `[T, C, H, W]` of processed frames, as a batch of one clip.
        let pixels = self
            .processor
            .preprocess_batch(&frames, self.options.device, self.options.data_type)?
            .unsqueeze(0)?;
        let mut states = self.model.0.forward(&pixels)?.squeeze(0)?;
        if parameters.pool {
            states = states.mean(&[0], true)?;
        }

        let &[_, width] = states.shape().dimensions() else {
            return Err(Error::invalid("a video encoder's states, [N, D]"));
        };
        let width = usize::try_from(width).map_err(|_| Error::invalid("a negative width"))?;
        Ok(host::rows(
            &states.cast(DataType::F32)?.contiguous()?.to_vec::<f32>()?,
            width,
        ))
    }
}
