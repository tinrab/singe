// ViT: a vision transformer of a class token and patches at a fixed image size, pre-norm blocks, and a final norm, whose class token an image classifier reads.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Linear, ModelState, PatchTokens},
};

use crate::{
    assembly::{FinalNorm, PreNormEncoder, VisionTowerConfig, patch_embedding},
    error::Result,
    format::Loader,
    head::{Boxed, Classifier, ClassifierConfig, HeadLoader, ImageClassifier},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitConfig {
    pub tower: VisionTowerConfig,
    pub classifier: ClassifierConfig,
}

// ViT and its classifier: from images `[B, C, H, W]` to each image's scores, `[B, labels]`, read from the class token's final state.
#[derive(Debug, Clone)]
pub struct VitClassifierLayers {
    embeddings: PatchTokens,
    encoder: PreNormEncoder,
    classifier: Linear,
}

impl VitClassifierLayers {
    pub fn create(schema: &mut Schema<'_>, config: &VitConfig) -> SingeResult<Self> {
        let tower = &config.tower;
        let width = tower.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patches =
                patch_embedding(&mut schema, tower.channels, width, tower.patch_size, true)?;
            PatchTokens::new(patches)
                .with_leading_token(schema.parameter("class", [1, 1, width])?)
                .with_positions(schema.parameter("positions", [1, tower.patches() + 1, width])?)
        };
        let encoder =
            PreNormEncoder::create(schema, &tower.encoder, tower.layers, FinalNorm::Included)?;
        let labels = config.classifier.label_count()?;
        let classifier = schema.linear("classifier.output", width, labels, true)?;
        Ok(Self {
            embeddings,
            encoder,
            classifier,
        })
    }
}

impl Layer for VitClassifierLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .encoder
            .forward(&input.apply(&self.embeddings, state)?, state)?;
        hidden.select(1, 0)?.apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "vit",
    formats: &[(Format::Transformers, &["vit"])],
    loaders: &[HeadLoader::ImageClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<ImageClassifier> {
    let config: VitConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        VitClassifierLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ImageClassifier(Classifier::new(model, config.classifier)))
}
