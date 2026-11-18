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
        Norms, SpatialAttention, TimeEmbedding, UnetLayout, UnetSite, UnetStages, conv3x3,
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
pub struct Unet2dLayers {
    config: Unet2dConfig,
    input: Conv2d,
    time: TimeEmbedding,
    stages: UnetStages<SpatialAttention>,
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
        let layout = UnetLayout {
            widths,
            layers_per_block: config.layers_per_block,
            down_attention: &config.down_attention,
            up_attention: &config.up_attention,
            time_width,
            norms,
            middle_scale: config.middle_scale,
            downsample_padding: config.downsample_padding,
        };
        let stages = UnetStages::create(schema, &layout, |schema, site, width| {
            let heads = config.head_width.map_or(1, |head| width / head);
            // The middle block's attention divides its sum as its residual blocks do.
            let rescale = if site == UnetSite::Middle {
                config.middle_scale
            } else {
                1.0
            };
            SpatialAttention::create(schema, width, heads, norms, rescale)
        })?;
        Ok(Self {
            config: config.clone(),
            input: conv3x3(schema, "input", config.in_channels, widths[0])?,
            time: TimeEmbedding::create(schema, "time", widths[0], time_width)?,
            stages,
            norm: schema.group_norm("norm", widths[0], config.groups, config.eps)?,
            output: conv3x3(schema, "output", widths[0], config.out_channels)?,
        })
    }

    // The predicted noise of `sample`, `[B, C, H, W]`, at the timestep whose features `time` holds, `[1, F]`.
    fn predict(&self, sample: &Tensor, time: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let time = time.apply(&self.time, state)?;
        self.stages
            .forward(
                sample.apply(&self.input, state)?,
                &time,
                state,
                |hidden, attention, _, _| match attention {
                    Some(attention) => hidden.apply(attention, state),
                    None => Ok(hidden),
                },
            )?
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
