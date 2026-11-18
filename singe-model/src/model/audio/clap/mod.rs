// CLAP: HTSAT, a Swin over a clip's mel spectrogram folded into a square image, and RoBERTa, each projected into one space where a clip and a text compare by cosine.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, BatchNorm2d, Layer, LayerNorm, Linear, Model, ModelState, StateBuilder,
        TokenBatch,
    },
    tensor::{InterpolateMode, Interpolation},
};

use crate::{
    assembly::{PostNormEncoder, PostNormEncoderConfig, Swin, SwinConfig},
    error::Result,
    format::Loader,
    head::{
        ContrastiveEncoder, HeadLoader, PixelLayers, SpectrogramOptions, TextEmbeddingLayers,
        ZeroShotAudioClassifier,
    },
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClapConfig {
    pub audio: SwinConfig,
    // The side of the square image a spectrogram folds into.
    pub spec_size: i64,
    pub text: PostNormEncoderConfig,
    pub projection: i64,
    pub projection_activation: Activation,
    pub features: SpectrogramOptions,
}

// A projection: two linear layers with an activation between them.
#[derive(Debug, Clone)]
struct Projection {
    first: Linear,
    second: Linear,
    activation: Activation,
}

impl Layer for Projection {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.first, state)?
            .apply(&self.activation, state)?
            .apply(&self.second, state)
    }
}

// HTSAT's spectrograms `[B, 1, frames, mels]` to their projected embeddings, `[B, P]`.
#[derive(Debug, Clone)]
struct ClapAudioLayers {
    norm: BatchNorm2d,
    swin: Swin,
    final_norm: LayerNorm,
    projection: Projection,
    spec_size: i64,
}

impl Layer for ClapAudioLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Each mel bin normalized by its running statistics.
        let normalized = input
            .transpose(1, 3)?
            .apply(&self.norm, state)?
            .transpose(1, 3)?;
        let dimensions = normalized.shape().dimensions();
        let (batch, frames, mels) = (dimensions[0], dimensions[2], dimensions[3]);
        let ratio = self.spec_size / mels;
        let width = self.spec_size * ratio;
        // Stretched to the image's width in frames, then folded: each quarter of the time becomes a band of the image's height, as `reshape_mel2img` does.
        let stretched = if frames < width {
            normalized.interpolate(
                (width, mels),
                Interpolation::new(InterpolateMode::Bicubic).with_align_corners(true),
            )?
        } else {
            normalized
        };
        let image = stretched
            .reshape([batch, ratio, width / ratio, mels])?
            .permute(&[0, 1, 3, 2])?
            .reshape([batch, 1, mels * ratio, width / ratio])?;
        let pooled = image
            .apply(&self.swin, state)?
            .apply(&self.final_norm, state)?
            .mean(&[1], false)?;
        pooled.apply(&self.projection, state)
    }
}

// RoBERTa's texts to their projected embeddings, `[N, P]`, from its pooler over the first token.
#[derive(Debug, Clone)]
struct ClapTextLayers {
    encoder: PostNormEncoder,
    pooler: Linear,
    projection: Projection,
}

impl Layer<TokenBatch> for ClapTextLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(input, state)?
            .select(1, 0)?
            .apply(&self.pooler, state)?
            .apply(&Activation::Tanh, state)?
            .apply(&self.projection, state)
    }
}

#[derive(Debug, Clone)]
struct ClapLayers {
    audio: ClapAudioLayers,
    text: ClapTextLayers,
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "clap",
    formats: &[(Format::Transformers, &["clap"])],
    heads: &[HeadLoader::ZeroShotAudioClassification(load_contrastive)],
};

fn projection(
    schema: &mut Schema<'_>,
    input: i64,
    width: i64,
    activation: Activation,
) -> SingeResult<Projection> {
    Ok(Projection {
        first: schema.linear("first", input, width, true)?,
        second: schema.linear("second", width, width, true)?,
        activation,
    })
}

pub fn load_contrastive(loader: &mut Loader<'_>) -> Result<ZeroShotAudioClassifier> {
    let config: ClapConfig = loader.config(transformers::config)?;
    let audio_width = config.audio.width << (config.audio.stages.len().saturating_sub(1));
    let mels =
        i64::try_from(config.features.mels).map_err(|_| singe::Error::out_of_range("mels"))?;
    let mut builder = StateBuilder::new();
    let (layers, logit_scale) = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let audio = ClapAudioLayers {
            norm: schema.batch_norm2d("audio.norm", mels, 1e-5)?,
            swin: Swin::create(&mut schema.scope("audio"), &config.audio)?,
            // HTSAT's final norm keeps `nn.LayerNorm`'s default epsilon instead of `layer_norm_eps`.
            final_norm: schema.layer_norm("audio.final_norm", audio_width, 1e-5)?,
            projection: projection(
                &mut schema.scope("audio_projection"),
                audio_width,
                config.projection,
                config.projection_activation,
            )?,
            spec_size: config.spec_size,
        };
        let text_width = config.text.width;
        let text = ClapTextLayers {
            encoder: PostNormEncoder::create(&mut schema.scope("text"), &config.text)?,
            pooler: schema.linear("text.pooler", text_width, text_width, true)?,
            projection: projection(
                &mut schema.scope("text_projection"),
                text_width,
                config.projection,
                config.projection_activation,
            )?,
        };
        let logit_scale = schema.parameter("logit_scale", [])?;
        (ClapLayers { audio, text }, logit_scale)
    };
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    let state = model.state().clone();
    Ok(ZeroShotAudioClassifier {
        encoder: ContrastiveEncoder::new(
            Model::new(
                PixelLayers(Box::new(model.layer().audio.clone())),
                state.clone(),
            ),
            Model::new(
                TextEmbeddingLayers(Box::new(model.layer().text.clone())),
                state,
            ),
            logit_scale,
        ),
        spectrogram: config.features,
    })
}
