// The Audio Spectrogram Transformer: a ViT over a clip's log mel filter banks, whose overlapping patches follow a class and a distillation token, the two tokens' mean classified.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{
        Conv2dConfig, Layer, LayerNorm, Linear, Model, ModelState, PatchEmbedding, StateBuilder,
        StateKey,
    },
};

use crate::{
    assembly::{PreNormEncoder, PreNormEncoderSpec},
    error::Result,
    format::Loader,
    head::{
        AudioClassifier, Classifier, ClassifierConfig, FilterBankOptions, HeadLoader, PixelLayers,
    },
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AstConfig {
    pub encoder: PreNormEncoderSpec,
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
    // The patches along frequency and time.
    const fn grid(&self) -> (i64, i64) {
        (
            (self.mels - self.patch_size) / self.frequency_stride + 1,
            (self.frames - self.patch_size) / self.time_stride + 1,
        )
    }
}

// From filter banks `[B, frames, mels]` to each label's score, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct AstLayers {
    patches: PatchEmbedding,
    class: StateKey,
    distillation: StateKey,
    positions: StateKey,
    encoder: PreNormEncoder,
    head_norm: LayerNorm,
    head: Linear,
}

impl Layer for AstLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Frequency as height and time as width.
        let patches = input
            .unsqueeze(1)?
            .transpose(2, 3)?
            .apply(&self.patches, state)?;
        let dimensions = patches.shape().dimensions();
        let (batch, width) = (dimensions[0], dimensions[2]);
        let token = |key: &StateKey| state.tensor(key)?.expand(Shape::create([batch, 1, width])?);
        let hidden = Tensor::concat(
            &[token(&self.class)?, token(&self.distillation)?, patches],
            1,
        )?
        .add(state.tensor(&self.positions)?)?;
        let hidden = self.encoder.forward(&hidden, state)?;
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
    heads: &[HeadLoader::AudioClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<AudioClassifier> {
    let config: AstConfig = loader.config(transformers::config)?;
    let width = config.encoder.width;
    let (rows, columns) = config.grid();
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let labels = i64::try_from(config.classifier.labels.len())
            .map_err(|_| singe::Error::out_of_range("labels"))?;
        AstLayers {
            patches: PatchEmbedding::new(schema.conv2d(
                "embeddings.patches",
                1,
                width,
                config.patch_size,
                Conv2dConfig {
                    stride: (config.frequency_stride, config.time_stride),
                    ..Conv2dConfig::default()
                },
                true,
            )?),
            class: schema.parameter("embeddings.class", [1, 1, width])?,
            distillation: schema.parameter("embeddings.distillation", [1, 1, width])?,
            positions: schema.parameter("embeddings.positions", [1, rows * columns + 2, width])?,
            encoder: PreNormEncoder::create(&mut schema, &config.encoder, config.layers, true)?,
            head_norm: schema.layer_norm("head_norm", width, config.encoder.eps)?,
            head: schema.linear("head", width, labels, true)?,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(AudioClassifier {
        classifier: Classifier::new(model, config.classifier),
        features: config.features,
    })
}
