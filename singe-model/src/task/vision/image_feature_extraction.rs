// `image-feature-extraction`: an image to each token's last hidden state, or to one pooled vector.

use std::slice;

use serde::Deserialize;
use singe::DataType;

use crate::{
    error::{Error, Result},
    head::ImageEncoder,
    model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct ImageFeatureExtractionPipeline {
    model: ImageEncoder,
    processor: ImageTransforms,
    data_type: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct ImageFeatureExtractionParameters {
    // Returns the model's pooled vector, as `pool=True` does, rather than every token's state.
    pub pool: bool,
}

impl Pipeline for ImageFeatureExtractionPipeline {
    const TASK: Task = Task::ImageFeatureExtraction;

    type Input = Image;

    type Parameters = ImageFeatureExtractionParameters;

    // Each token's state, `[T, D]`, or the pooled vector as one row.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.state().device()?;
        inputs
            .iter()
            .map(|image| {
                let pixels =
                    self.processor
                        .preprocess(slice::from_ref(image), device, self.data_type)?;
                let hidden = self.model.forward(&pixels)?;
                let features = if parameters.pool {
                    self.model.layer().pooled(&hidden)?
                } else {
                    hidden.squeeze(0)?
                };
                let dimensions = features.shape().dimensions();
                let width = usize::try_from(dimensions[dimensions.len() - 1])
                    .map_err(|_| Error::unsupported("this width"))?;
                let values = features.cast(DataType::F32)?.to_vec::<f32>()?;
                Ok(values.chunks(width).map(<[f32]>::to_vec).collect())
            })
            .collect()
    }
}
