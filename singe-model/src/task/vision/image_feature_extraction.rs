// `image-feature-extraction`: an image to each token's last hidden state, or to one pooled vector.

use serde::Deserialize;
use singe::{DataType, Error as SingeError};

use crate::{
    error::Result,
    head::ImageEncoder,
    host, model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct ImageFeatureExtractionPipeline {
    model: ImageEncoder,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct ImageFeatureExtractionParameters {
    // Returns the model's pooled vector, as `pool=True` does, rather than every token's state.
    pub pool: bool,
}

impl Pipeline for ImageFeatureExtractionPipeline {
    type Input = Image;

    type Parameters = ImageFeatureExtractionParameters;

    // Each token's state, `[T, D]`, or the pooled vector as one row.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let hidden = self.model.forward(&pixels)?;
        let features = if parameters.pool {
            self.model.layer().pooled(&hidden)?
        } else {
            hidden.squeeze(0)?
        };
        let &[_, width] = features.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image features, [T, D]").into());
        };
        let width =
            usize::try_from(width).map_err(|_| SingeError::out_of_range("feature width"))?;
        let values = features.cast(DataType::F32)?.to_vec::<f32>()?;
        Ok(host::rows(&values, width))
    }
}
