// ViT: a vision transformer of a class token and patches at a fixed image size, pre-norm blocks, and a final norm, whose class token an image classifier reads.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Conv2dConfig, Layer, Linear, Model, ModelState, PatchEmbedding, PatchTokens, StateBuilder,
    },
};

use crate::{
    assembly::{PreNormEncoder, PreNormEncoderSpec},
    error::Result,
    format::Loader,
    head::{Classifier, ClassifierConfig, HeadLoader, ImageClassifier, PixelLayers},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitConfig {
    pub encoder: PreNormEncoderSpec,
    pub layers: i64,
    // The square image side the positions are learned for, a whole number of patches.
    pub image_size: i64,
    pub patch_size: i64,
    pub channels: i64,
    pub classifier: ClassifierConfig,
}

// ViT's tokens: its class token and patches, with learned positions, as `embeddings`.
pub fn patch_tokens(schema: &mut Schema<'_>, config: &VitConfig) -> SingeResult<PatchTokens> {
    let width = config.encoder.width;
    let mut schema = schema.scope("embeddings");
    let patches = schema.conv2d(
        "patches",
        config.channels,
        width,
        config.patch_size,
        Conv2dConfig {
            stride: (config.patch_size, config.patch_size),
            ..Conv2dConfig::default()
        },
        true,
    )?;
    let side = config.image_size / config.patch_size;
    Ok(PatchTokens::new(PatchEmbedding::new(patches))
        .with_class_token(schema.parameter("class", [1, 1, width])?)
        .with_positions(schema.parameter("positions", [1, side * side + 1, width])?))
}

// ViT and its classifier: from images `[B, C, H, W]` to each image's scores, `[B, labels]`, read from the class token's final state.
#[derive(Debug, Clone)]
pub struct VitClassifierLayers {
    embeddings: PatchTokens,
    encoder: PreNormEncoder,
    classifier: Linear,
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
    heads: &[HeadLoader::ImageClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<ImageClassifier> {
    let config: VitConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let embeddings = patch_tokens(&mut schema, &config)?;
        let encoder = PreNormEncoder::create(&mut schema, &config.encoder, config.layers, true)?;
        let labels = i64::try_from(config.classifier.labels.len())
            .map_err(|_| singe::Error::out_of_range("labels"))?;
        let classifier = schema.linear("classifier.output", config.encoder.width, labels, true)?;
        VitClassifierLayers {
            embeddings,
            encoder,
            classifier,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(ImageClassifier(Classifier::new(model, config.classifier)))
}
