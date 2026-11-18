// Wav2Vec2: raw samples through strided convolutions into frames, projected, given positions by a grouped convolution, then post-norm blocks, whose states a CTC head reads as characters.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, Conv1d, Conv1dConfig, DenseAttentionInput, EncoderBlock, GroupNorm, Layer,
        LayerNorm, Linear, Model, ModelState, StateBuilder,
    },
};

use crate::{
    assembly::encoder_block,
    error::Result,
    format::Loader,
    head::{AudioLayers, CtcRecognizer, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wav2Vec2Config {
    // Each convolution's channels, kernel, and stride.
    pub convolutions: Vec<(i64, i64, i64)>,
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
    // Whether a clip is normalized to zero mean and unit variance first.
    pub normalize: bool,
}

#[derive(Debug, Clone)]
pub struct Wav2Vec2Layers {
    convolutions: Vec<Conv1d>,
    // The first convolution's norm, of each channel over time.
    first_norm: GroupNorm,
    projection_norm: LayerNorm,
    projection: Linear,
    positions: Conv1d,
    norm: LayerNorm,
    blocks: Vec<EncoderBlock>,
    head: Linear,
    config: Wav2Vec2Config,
}

impl Layer for Wav2Vec2Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let activation = self.config.feature_activation;
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
        // An even kernel leaves one frame too many, which `Wav2Vec2SamePadLayer` drops.
        let frames = hidden.shape().dimensions()[1];
        let positions = hidden
            .transpose(1, 2)?
            .apply(&self.positions, state)?
            .narrow(2, 0, frames)?
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
    heads: &[HeadLoader::Ctc(load_ctc)],
};

pub fn load_ctc(loader: &mut Loader<'_>) -> Result<CtcRecognizer> {
    let config: Wav2Vec2Config = loader.config(transformers::config)?;
    let width = config.width;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let mut input = 1;
        let mut convolutions = Vec::new();
        for (index, &(channels, kernel, stride)) in config.convolutions.iter().enumerate() {
            convolutions.push(schema.conv1d(
                &format!("convolutions.{index}"),
                input,
                channels,
                kernel,
                Conv1dConfig {
                    stride,
                    ..Conv1dConfig::default()
                },
                config.convolution_bias,
            )?);
            input = channels;
        }

        let first = config
            .convolutions
            .first()
            .map_or(0, |&(channels, _, _)| channels);
        let half = config.position_kernel / 2;
        Wav2Vec2Layers {
            convolutions,
            first_norm: schema.group_norm("first_norm", first, first, 1e-5)?,
            projection_norm: schema.layer_norm("projection_norm", input, config.eps)?,
            projection: schema.linear("projection", input, width, true)?,
            positions: schema.conv1d(
                "positions",
                width,
                width,
                config.position_kernel,
                Conv1dConfig {
                    padding: (half, half),
                    groups: config.position_groups,
                    ..Conv1dConfig::default()
                },
                true,
            )?,
            norm: schema.layer_norm("norm", width, config.eps)?,
            blocks: (0..config.layers)
                .map(|index| {
                    encoder_block(
                        &mut schema.scope(format!("blocks.{index}")),
                        width,
                        config.heads,
                        config.inner,
                        config.activation,
                        config.eps,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            head: schema.linear("head", width, config.vocab_size, true)?,
            config: config.clone(),
        }
    };
    let schema = Model::new(AudioLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(CtcRecognizer::new(
        model,
        config.sample_rate,
        config.normalize,
    ))
}
