//! The Audio Spectrogram Transformer: a `ViT` over a clip's log mel filter banks, whose overlapping patches follow a class and a distillation token, the two tokens' mean classified.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Conv2dConfig, Layer, LayerNorm, Linear, ModelState, PatchEmbedding, PatchTokens},
    shape::spatial::{Spatial2d, Spatial2dI64},
};

use crate::{
    assembly::{FinalNorm, PreNormEncoder, PreNormEncoderConfig},
    error::Result,
    format::Loader,
    head::{AudioClassifier, Boxed, Classifier, ClassifierConfig, HeadLoader},
    model::ModelEntry,
    processor::fbank::FilterBankOptions,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AstConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    pub patch_size: i64,
    pub frequency_stride: i64,
    pub time_stride: i64,
    pub mels: i64,
    pub frames: i64,
    pub classifier: ClassifierConfig,
    pub features: FilterBankOptions,
}

impl AstConfig {
    /// Returns how many patches fit along frequency and along time.
    const fn grid(&self) -> Spatial2dI64 {
        Spatial2d::new(
            (self.mels - self.patch_size) / self.frequency_stride + 1,
            (self.frames - self.patch_size) / self.time_stride + 1,
        )
    }
}

/// The layers that score filter banks, `[B, frames, mels]`, for each label, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct AstLayers {
    tokens: PatchTokens,
    encoder: PreNormEncoder,
    head_norm: LayerNorm,
    head: Linear,
}

impl AstLayers {
    pub fn create(schema: &mut Schema<'_>, config: &AstConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let grid = config.grid();
        let tokens = {
            let mut embeddings = schema.scope("embeddings");
            let patches = PatchEmbedding::new(embeddings.conv2d(
                "patches",
                1,
                width,
                config.patch_size,
                Conv2dConfig {
                    stride: Spatial2d::new(config.frequency_stride, config.time_stride),
                    ..Conv2dConfig::default()
                },
                true,
            )?);
            PatchTokens::new(patches)
                .with_leading_token(embeddings.parameter("class", [1, 1, width])?)
                .with_leading_token(embeddings.parameter("distillation", [1, 1, width])?)
                .with_positions(
                    embeddings.parameter("positions", [1, grid.height * grid.width + 2, width])?,
                )
        };
        Ok(Self {
            tokens,
            encoder: PreNormEncoder::create(
                schema,
                &config.encoder,
                config.layers,
                FinalNorm::Included,
            )?,
            head_norm: schema.layer_norm("head_norm", width, config.encoder.eps)?,
            head: schema.linear("head", width, config.classifier.label_count()?, true)?,
        })
    }
}

impl Layer for AstLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Frequency as height and time as width.
        let hidden = input
            .unsqueeze(1)?
            .transpose(2, 3)?
            .apply(&self.tokens, state)?
            .apply(&self.encoder, state)?;
        hidden
            .narrow(1, 0, 1)?
            .add(&hidden.narrow(1, 1, 1)?)?
            .mul_scalar(0.5)?
            .squeeze(1)?
            .apply(&self.head_norm, state)?
            .apply(&self.head, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "audio_spectrogram_transformer",
    formats: &[(Format::Transformers, &["audio-spectrogram-transformer"])],
    loaders: &[HeadLoader::AudioClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<AudioClassifier> {
    let config: AstConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        AstLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    Ok(AudioClassifier {
        classifier: Classifier::new(
            loader.load_whole(&model, &transformers::rules())?,
            config.classifier,
        ),
        features: config.features,
    })
}
