//! `ConvNeXt` and `ConvNeXt` V2 classifiers: the last stage's map averaged, layer-normed, and read by a linear classifier.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, LayerNorm, Linear, ModelState},
};

use crate::{
    assembly::{ConvNext, ConvNextConfig},
    error::Result,
    format::Loader,
    head::{Boxed, Classifier, ClassifierConfig, HeadLoader, ImageClassifier},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvNextClassifierConfig {
    pub backbone: ConvNextConfig,
    /// The epsilon of the pooled map's layer norm, which the config's `layer_norm_eps` sets.
    pub eps: f32,
    pub classifier: ClassifierConfig,
}

/// A `ConvNeXt` and its classifier: from images to each label's score, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct ConvNextClassifierLayers {
    backbone: ConvNext,
    norm: LayerNorm,
    classifier: Linear,
}

impl ConvNextClassifierLayers {
    pub fn create(schema: &mut Schema<'_>, config: &ConvNextClassifierConfig) -> SingeResult<Self> {
        let width = config.backbone.widths.last().copied().unwrap_or_default();
        Ok(Self {
            backbone: ConvNext::create(&mut schema.scope("backbone"), &config.backbone)?,
            norm: schema.layer_norm("norm", width, config.eps)?,
            classifier: schema.linear(
                "classifier",
                width,
                config.classifier.label_count()?,
                true,
            )?,
        })
    }
}

impl Layer for ConvNextClassifierLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let last = self
            .backbone
            .stages(input, state)?
            .pop()
            .ok_or_else(|| SingeError::invalid_argument("a ConvNeXt without stages"))?;
        last.mean(&[2, 3], false)?
            .apply(&self.norm, state)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "convnext",
    formats: &[(Format::Transformers, &["convnext", "convnextv2"])],
    loaders: &[HeadLoader::ImageClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<ImageClassifier> {
    let config: ConvNextClassifierConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        ConvNextClassifierLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.backbone.scaling))?;
    Ok(ImageClassifier(Classifier::new(model, config.classifier)))
}
