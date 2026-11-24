//! `diffusers`' `UNet2DModel`: an unconditional denoiser of residual blocks, with self-attention in the blocks that ask for it, down to a middle block and back up, each up block reading the down blocks' outputs in reverse.
//!
//! Its state is `input`, `time` (`up`, `down`), `down.N` and `up.N` (`resnets.M`, `attentions.M` where the block attends, and `downsample` or `upsample` but in the last), `middle` (`resnets.0`, `attentions.0`, `resnets.1`), `norm`, and `output`.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Model, ModelState},
};

use crate::{
    assembly::{SpatialAttention, Unet, UnetConfig, UnetSite},
    error::Result,
    format::Loader,
    head::{Denoiser, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unet2dConfig {
    #[serde(flatten)]
    pub unet: UnetConfig,
    /// Each attention's head width, or one head as wide as its block when absent.
    pub head_width: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct Unet2dLayers {
    config: Unet2dConfig,
    unet: Unet<SpatialAttention>,
}

impl Unet2dLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Unet2dConfig) -> SingeResult<Self> {
        let unet = &config.unet;
        let norms = unet.norms();
        Ok(Self {
            config: config.clone(),
            unet: Unet::create(schema, unet, |schema, site, width| {
                let heads = config.head_width.map_or(1, |head| width / head);
                // The middle block's attention divides its sum as its residual blocks do.
                let rescale = if site == UnetSite::Middle {
                    unet.middle_scale
                } else {
                    1.0
                };
                SpatialAttention::create(schema, width, heads, norms, rescale)
            })?,
        })
    }

    /// Returns the predicted noise of `sample`, `[B, C, H, W]`, at the timestep whose features `time` holds, `[1, F]`.
    fn predict(&self, sample: &Tensor, time: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.unet.denoise(
            self.unet.embed(sample, state)?,
            time,
            state,
            |hidden, attention, _, _| match attention {
                Some(attention) => hidden.apply(attention, state),
                None => Ok(hidden),
            },
        )
    }
}

pub type Unet2d = Model<Unet2dLayers>;

impl Denoiser for Unet2d {
    fn sample_shape(&self) -> [i64; 3] {
        let config = &self.layer().config.unet;
        [config.in_channels, config.sample_size, config.sample_size]
    }

    fn predict(&self, sample: &Tensor, timestep: f32) -> Result<Tensor> {
        let time = self
            .layer()
            .config
            .unet
            .time_input(timestep, self.state())?;
        let sample = sample.cast(self.state().floating_type()?)?;
        Ok(self
            .layer()
            .predict(&sample, &time, self.state())?
            .cast(DataType::F32)?)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "unet_2d",
    formats: &[(Format::Diffusers, &["UNet2DModel"])],
    loaders: &[HeadLoader::Denoiser(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Unet2d> {
    let config: Unet2dConfig = loader.config(diffusers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Unet2dLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &diffusers::rules())
}
