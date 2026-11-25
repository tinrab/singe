//! `image-feature-extraction`: an image to each token's last hidden state, or to one pooled vector.

use std::sync::Arc;

use serde::Deserialize;
use singe::{DataType, Error as SingeError};

use crate::{
    error::{Error, Result},
    function::forward::{External, Forward},
    head::{Boxed, ImageEncoder, PooledLayer},
    host, model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct ImageFeatureExtractionPipeline {
    model: Forward<Boxed<dyn PooledLayer>>,
    pooled: Option<External>,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct ImageFeatureExtractionParameters {
    /// Whether to return the model's pooled vector, as `pool=True` does, rather than every token's state.
    pub pool: bool,
}

impl Pipeline for ImageFeatureExtractionPipeline {
    type Input = Image;

    type Parameters = ImageFeatureExtractionParameters;

    /// Each token's state, `[T, D]`, or the pooled vector as one row.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let (model, pooled) = if let Some(function) = source.function() {
            let hidden = External::create(Arc::clone(function), &["last_hidden_state"])?;
            let pooled = hidden
                .output("pooler_output")
                .is_some()
                .then(|| External::create(Arc::clone(function), &["pooler_output"]))
                .transpose()?;
            (Forward::External(hidden), pooled)
        } else {
            let model: ImageEncoder = model::load(source, options)?;
            (Forward::Native(model), None)
        };
        Ok(Self {
            model,
            pooled,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let features = match (&self.model, &self.pooled, parameters.pool) {
            (Forward::Native(model), _, true) => model.layer().pooled(&model.forward(&pixels)?)?,
            (Forward::External(_), Some(pooled), true) => pooled.single(&pixels)?,
            (Forward::External(_), None, true) => {
                return Err(Error::unsupported(
                    "pooling with a model without the output pooler_output",
                ));
            }
            (model, _, false) => model.forward(&pixels)?.squeeze(0)?,
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
