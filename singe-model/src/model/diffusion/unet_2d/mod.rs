// `diffusers`' `UNet2DModel`: an unconditional denoiser of residual blocks, with self-attention in the blocks that ask for it, down to a middle block and back up, each up block reading the down blocks' outputs in reverse.
//
// Its state is `input`, `time` (`input`, `output`), `down.N` and `up.N` (`resnets.M`, `attentions.M` where the block attends, and `downsample` or `upsample` but in the last), `middle` (`resnets.0`, `attentions.0`, `resnets.1`), `norm`, and `output`.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, GroupNorm, Model, ModelState, StateBuilder},
};

use crate::{
    assembly::{
        Norms, ResnetBlock, SpatialAttention, TimeEmbedding, UpsampleBlock, conv3x3, downsample,
        timestep_input,
    },
    error::Result,
    format::Loader,
    head::{Denoiser, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unet2dConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub sample_size: i64,
    pub widths: Vec<i64>,
    // Whether each down block, and each up block, attends.
    pub down_attention: Vec<bool>,
    pub up_attention: Vec<bool>,
    pub layers_per_block: usize,
    pub groups: i64,
    pub eps: f32,
    // Each attention's head width, or one head as wide as its block when absent.
    pub head_width: Option<i64>,
    pub downsample_padding: i64,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f32,
    pub middle_scale: f64,
}

#[derive(Debug, Clone)]
struct Stage {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<SpatialAttention>,
    downsample: Option<Conv2d>,
    upsample: Option<UpsampleBlock>,
}

#[derive(Debug, Clone)]
pub struct Unet2dLayers {
    config: Unet2dConfig,
    input: Conv2d,
    time: TimeEmbedding,
    down: Vec<Stage>,
    middle: Stage,
    up: Vec<Stage>,
    norm: GroupNorm,
    output: Conv2d,
}

impl Unet2dLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Unet2dConfig) -> SingeResult<Self> {
        let norms = Norms {
            groups: config.groups,
            eps: config.eps,
        };
        let widths = &config.widths;
        let time_width = widths[0] * 4;
        let heads = |channels: i64| config.head_width.map_or(1, |width| channels / width);
        let last = widths.len() - 1;
        let down = widths
            .iter()
            .enumerate()
            .map(|(index, &output)| {
                let mut stage = schema.scope(format!("down.{index}"));
                let input = if index == 0 {
                    widths[0]
                } else {
                    widths[index - 1]
                };
                let resnets = (0..config.layers_per_block)
                    .map(|layer| {
                        ResnetBlock::create(
                            &mut stage.scope(format!("resnets.{layer}")),
                            [if layer == 0 { input } else { output }, output],
                            Some(time_width),
                            norms,
                            1.0,
                        )
                    })
                    .collect::<SingeResult<Vec<_>>>()?;
                let attentions = if config.down_attention[index] {
                    (0..config.layers_per_block)
                        .map(|layer| {
                            SpatialAttention::create(
                                &mut stage.scope(format!("attentions.{layer}")),
                                output,
                                heads(output),
                                norms,
                                1.0,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?
                } else {
                    Vec::new()
                };
                let downsample = (index < last)
                    .then(|| {
                        downsample(&mut stage, "downsample", output, config.downsample_padding)
                    })
                    .transpose()?;
                Ok(Stage {
                    resnets,
                    attentions,
                    downsample,
                    upsample: None,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let middle = {
            let mut stage = schema.scope("middle");
            let width = widths[last];
            let block = |stage: &mut Schema<'_>, layer: usize| {
                ResnetBlock::create(
                    &mut stage.scope(format!("resnets.{layer}")),
                    [width, width],
                    Some(time_width),
                    norms,
                    config.middle_scale,
                )
            };
            let first = block(&mut stage, 0)?;
            let attention = SpatialAttention::create(
                &mut stage.scope("attentions.0"),
                width,
                config.head_width.map_or(1, |head| width / head),
                norms,
                config.middle_scale,
            )?;
            Stage {
                resnets: vec![first, block(&mut stage, 1)?],
                attentions: vec![attention],
                downsample: None,
                upsample: None,
            }
        };
        let reversed = widths.iter().rev().copied().collect::<Vec<_>>();
        let up = (0..widths.len())
            .map(|index| {
                let mut stage = schema.scope(format!("up.{index}"));
                let previous = reversed[index.saturating_sub(1)];
                let output = reversed[index];
                let input = reversed[(index + 1).min(last)];
                let layers = config.layers_per_block + 1;
                let resnets = (0..layers)
                    .map(|layer| {
                        let skip = if layer == layers - 1 { input } else { output };
                        let from = if layer == 0 { previous } else { output };
                        ResnetBlock::create(
                            &mut stage.scope(format!("resnets.{layer}")),
                            [from + skip, output],
                            Some(time_width),
                            norms,
                            1.0,
                        )
                    })
                    .collect::<SingeResult<Vec<_>>>()?;
                let attentions = if config.up_attention[index] {
                    (0..layers)
                        .map(|layer| {
                            SpatialAttention::create(
                                &mut stage.scope(format!("attentions.{layer}")),
                                output,
                                heads(output),
                                norms,
                                1.0,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()?
                } else {
                    Vec::new()
                };
                let upsample = (index < last)
                    .then(|| UpsampleBlock::create(&mut stage, "upsample", output))
                    .transpose()?;
                Ok(Stage {
                    resnets,
                    attentions,
                    downsample: None,
                    upsample,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            input: conv3x3(schema, "input", config.in_channels, widths[0])?,
            time: TimeEmbedding::create(schema, "time", widths[0], time_width)?,
            down,
            middle,
            up,
            norm: schema.group_norm("norm", widths[0], config.groups, config.eps)?,
            output: conv3x3(schema, "output", widths[0], config.out_channels)?,
        })
    }

    // The predicted noise of `sample`, `[B, C, H, W]`, at the timestep whose features `time` holds, `[1, F]`.
    fn predict(&self, sample: &Tensor, time: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let time = time.apply(&self.time, state)?;
        let mut hidden = sample.apply(&self.input, state)?;
        let mut skips = vec![hidden.clone()];
        for stage in &self.down {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                hidden = resnet.forward(&hidden, Some(&time), state)?;
                if let Some(attention) = stage.attentions.get(index) {
                    hidden = hidden.apply(attention, state)?;
                }

                skips.push(hidden.clone());
            }
            if let Some(downsample) = &stage.downsample {
                hidden = hidden.apply(downsample, state)?;
                skips.push(hidden.clone());
            }
        }

        let middle = &self.middle;
        hidden = middle.resnets[0].forward(&hidden, Some(&time), state)?;
        hidden = hidden.apply(&middle.attentions[0], state)?;
        hidden = middle.resnets[1].forward(&hidden, Some(&time), state)?;
        for stage in &self.up {
            for (index, resnet) in stage.resnets.iter().enumerate() {
                let skip = skips
                    .pop()
                    .ok_or_else(|| singe::Error::missing_input("a skip connection"))?;
                hidden =
                    resnet.forward(&Tensor::concat(&[hidden, skip], 1)?, Some(&time), state)?;
                if let Some(attention) = stage.attentions.get(index) {
                    hidden = hidden.apply(attention, state)?;
                }
            }
            if let Some(upsample) = &stage.upsample {
                // The next skip's size, which doubling misses after an odd size was halved.
                let size = skips
                    .last()
                    .map(|skip| skip.shape().dimensions()[2..].to_vec())
                    .ok_or_else(|| singe::Error::missing_input("a skip connection"))?;
                hidden = upsample.forward_to(&hidden, size[0], size[1], state)?;
            }
        }
        hidden
            .apply(&self.norm, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)
    }
}

pub type Unet2d = Model<Unet2dLayers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "unet_2d",
    formats: &[(Format::Diffusers, &["UNet2DModel"])],
    heads: &[HeadLoader::Denoiser(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Unet2d> {
    let config: Unet2dConfig = loader.config(diffusers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        Unet2dLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "unet",
        &Model::new(layers, builder.finish()?),
        &diffusers::rules(),
    )?;
    loader.finish()?;
    Ok(model)
}

impl Denoiser for Unet2d {
    fn sample_shape(&self) -> [i64; 3] {
        let config = &self.layer().config;
        [config.in_channels, config.sample_size, config.sample_size]
    }

    #[expect(clippy::cast_precision_loss, reason = "timesteps are small")]
    fn predict(&self, sample: &Tensor, timestep: i64) -> Result<Tensor> {
        let config = &self.layer().config;
        let time = timestep_input(
            timestep as f32,
            config.widths[0],
            config.flip_sin_to_cos,
            config.freq_shift,
            self.state(),
        )?;
        let sample = sample.cast(self.state().floating_type()?)?;
        Ok(self
            .layer()
            .predict(&sample, &time, self.state())?
            .cast(DataType::F32)?)
    }
}
