//! CLAP: HTSAT, a Swin over a clip's mel spectrogram folded into a square image, and `RoBERTa`, each projected into one space where a clip and a text compare by cosine.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Activation, BatchNorm, Layer, LayerNorm, Linear, Mlp, Model, ModelState, StateKey,
        TokenBatch,
    },
    tensor::{InterpolateMode, Interpolation},
};

use crate::{
    assembly::{PYTORCH_NORM_EPS, PostNormEncoder, PostNormEncoderConfig, Swin, SwinConfig},
    error::Result,
    format::Loader,
    head::{Boxed, ContrastiveEncoder, HeadLoader, ZeroShotAudioClassifier},
    model::ModelEntry,
    processor::fbank::SpectrogramOptions,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClapConfig {
    pub audio: SwinConfig,
    /// The side of the square image a spectrogram folds into.
    pub spec_size: i64,
    pub text: PostNormEncoderConfig,
    pub projection: i64,
    pub projection_activation: Activation,
    pub features: SpectrogramOptions,
}

/// The audio tower, HTSAT, from mel spectrograms, `[B, 1, frames, mels]`, to their projected embeddings, `[B, P]`.
#[derive(Debug, Clone)]
struct ClapAudioLayers {
    norm: BatchNorm,
    swin: Swin,
    final_norm: LayerNorm,
    projection: Mlp,
    spec_size: i64,
}

impl ClapAudioLayers {
    fn create(schema: &mut Schema<'_>, config: &ClapConfig) -> SingeResult<Self> {
        let width = config.audio.out_width();
        let mels =
            i64::try_from(config.features.mels).map_err(|_| SingeError::out_of_range("mels"))?;
        Ok(Self {
            norm: schema.batch_norm("audio.norm", mels, PYTORCH_NORM_EPS)?,
            swin: Swin::create(&mut schema.scope("audio"), &config.audio)?,
            // HTSAT's final norm keeps `nn.LayerNorm`'s default epsilon instead of `layer_norm_eps`.
            final_norm: schema.layer_norm("audio.final_norm", width, PYTORCH_NORM_EPS)?,
            projection: schema.mlp(
                "audio_projection",
                [width, config.projection, config.projection],
                config.projection_activation,
                true,
            )?,
            spec_size: config.spec_size,
        })
    }
}

impl Layer for ClapAudioLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Each mel bin normalized by its running statistics.
        let normalized = input
            .transpose(1, 3)?
            .apply(&self.norm, state)?
            .transpose(1, 3)?;
        let &[batch, _, frames, mels] = normalized.shape().dimensions() else {
            return Err(SingeError::invalid_argument("CLAP spectrograms"));
        };
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
        stretched
            .reshape([batch, ratio, width / ratio, mels])?
            .permute(&[0, 1, 3, 2])?
            .reshape([batch, 1, mels * ratio, width / ratio])?
            .apply(&self.swin, state)?
            .apply(&self.final_norm, state)?
            .mean(&[1], false)?
            .apply(&self.projection, state)
    }
}

/// The text tower, `RoBERTa` and its pooler over the first token, from token batches to their projected embeddings, `[N, P]`.
#[derive(Debug, Clone)]
struct ClapTextLayers {
    encoder: PostNormEncoder,
    pooler: Linear,
    projection: Mlp,
}

impl ClapTextLayers {
    fn create(schema: &mut Schema<'_>, config: &ClapConfig) -> SingeResult<Self> {
        let width = config.text.width;
        Ok(Self {
            encoder: PostNormEncoder::create(&mut schema.scope("text"), &config.text)?,
            pooler: schema.linear("text.pooler", width, width, true)?,
            projection: schema.mlp(
                "text_projection",
                [width, config.projection, config.projection],
                config.projection_activation,
                true,
            )?,
        })
    }
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
    logit_scale: StateKey,
}

impl ClapLayers {
    fn create(schema: &mut Schema<'_>, config: &ClapConfig) -> SingeResult<Self> {
        Ok(Self {
            audio: ClapAudioLayers::create(schema, config)?,
            text: ClapTextLayers::create(schema, config)?,
            logit_scale: schema.parameter("logit_scale", [])?,
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "clap",
    formats: &[(Format::Transformers, &["clap"])],
    loaders: &[HeadLoader::ZeroShotAudioClassification(load_contrastive)],
};

pub fn load_contrastive(loader: &mut Loader<'_>) -> Result<ZeroShotAudioClassifier> {
    let config: ClapConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        ClapLayers::create(schema, &config)
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    let ClapLayers {
        audio,
        text,
        logit_scale,
    } = model.layer().clone();
    let state = model.state().clone();
    Ok(ZeroShotAudioClassifier {
        encoder: ContrastiveEncoder::new(
            Model::new(Boxed::<dyn Layer>(Box::new(audio)), state.clone()),
            Model::new(Boxed::<dyn Layer<TokenBatch>>(Box::new(text)), state),
            logit_scale,
        ),
        spectrogram: config.features,
    })
}
