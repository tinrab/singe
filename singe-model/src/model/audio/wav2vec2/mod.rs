//! `Wav2Vec2`: raw samples through strided convolutions into frames, projected, given positions by a grouped convolution, then post-norm blocks, whose states a CTC head reads as characters.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv1d, Conv1dConfig, DenseAttentionInput, EncoderBlock, GroupNorm, Layer,
        LayerNorm, Linear, ModelState,
    },
};

use crate::{
    assembly::{PYTORCH_NORM_EPS, PostNormBlockConfig, post_norm_encoder_block},
    error::Result,
    format::Loader,
    head::{Boxed, CtcInput, CtcRecognizer, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureConvolution {
    pub channels: i64,
    pub kernel: i64,
    pub stride: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wav2Vec2Config {
    pub convolutions: Vec<FeatureConvolution>,
    pub convolution_bias: bool,
    pub feature_activation: Activation,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub position_kernel: i64,
    pub position_groups: i64,
    pub vocab_size: i64,
    pub sample_rate: i64,
    /// Whether a clip is normalized to zero mean and unit variance first.
    pub normalize: bool,
}

#[derive(Debug, Clone)]
pub struct Wav2Vec2Layers {
    convolutions: Vec<Conv1d>,
    /// The first convolution's norm, of each channel over time.
    first_norm: GroupNorm,
    projection_norm: LayerNorm,
    projection: Linear,
    positions: Conv1d,
    norm: LayerNorm,
    blocks: Vec<EncoderBlock>,
    head: Linear,
    feature_activation: Activation,
}

impl Wav2Vec2Layers {
    pub fn create(schema: &mut Schema<'_>, config: &Wav2Vec2Config) -> SingeResult<Self> {
        let width = config.width;
        let Some(first) = config.convolutions.first() else {
            return Err(SingeError::invalid_argument(
                "a Wav2Vec2 without feature convolutions",
            ));
        };
        let mut input = 1;
        let mut convolutions = Vec::with_capacity(config.convolutions.len());
        for (index, convolution) in config.convolutions.iter().enumerate() {
            convolutions.push(schema.conv1d(
                &format!("convolutions.{index}"),
                input,
                convolution.channels,
                convolution.kernel,
                Conv1dConfig {
                    stride: convolution.stride,
                    ..Conv1dConfig::default()
                },
                config.convolution_bias,
            )?);
            input = convolution.channels;
        }

        // An even kernel would give one frame too many, which `Wav2Vec2SamePadLayer` drops; padding one less after gives the frames alone, so the layer reads no frame count.
        let half = config.position_kernel / 2;
        let after = half - (1 - config.position_kernel % 2);
        let block = PostNormBlockConfig {
            width,
            heads: config.heads,
            inner: config.inner,
            activation: config.activation,
            eps: config.eps,
        };
        Ok(Self {
            convolutions,
            first_norm: schema.group_norm(
                "first_norm",
                first.channels,
                first.channels,
                PYTORCH_NORM_EPS,
            )?,
            projection_norm: schema.layer_norm("projection_norm", input, config.eps)?,
            projection: schema.linear("projection", input, width, true)?,
            positions: schema.conv1d(
                "positions",
                width,
                width,
                config.position_kernel,
                Conv1dConfig {
                    padding: (half, after),
                    groups: config.position_groups,
                    ..Conv1dConfig::default()
                },
                true,
            )?,
            norm: schema.layer_norm("norm", width, config.eps)?,
            blocks: (0..config.layers)
                .map(|index| {
                    post_norm_encoder_block(&mut schema.scope(format!("blocks.{index}")), &block)
                })
                .collect::<SingeResult<Vec<_>>>()?,
            head: schema.linear("head", width, config.vocab_size, true)?,
            feature_activation: config.feature_activation,
        })
    }
}

impl Layer for Wav2Vec2Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let activation = self.feature_activation;
        let mut hidden = input.unsqueeze(1)?;
        for (index, convolution) in self.convolutions.iter().enumerate() {
            hidden = hidden.apply(convolution, state)?;
            if index == 0 {
                hidden = hidden.apply(&self.first_norm, state)?;
            }

            hidden = hidden.apply(&activation, state)?;
        }

        let hidden = hidden
            .transpose(1, 2)?
            .apply(&self.projection_norm, state)?
            .apply(&self.projection, state)?;
        let positions = hidden
            .transpose(1, 2)?
            .apply(&self.positions, state)?
            .apply(&activation, state)?
            .transpose(1, 2)?;
        let mut hidden = hidden.add(&positions)?.apply(&self.norm, state)?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: None,
                },
                state,
            )?;
        }
        hidden.apply(&self.head, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "wav2vec2",
    formats: &[(Format::Transformers, &["wav2vec2"])],
    loaders: &[HeadLoader::Ctc(load_ctc)],
};

pub fn load_ctc(loader: &mut Loader<'_>) -> Result<CtcRecognizer> {
    let config: Wav2Vec2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Wav2Vec2Layers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    Ok(CtcRecognizer::new(
        loader.load_whole(&model, &transformers::rules())?,
        config.sample_rate,
        CtcInput::Samples {
            normalize: config.normalize,
        },
    ))
}
