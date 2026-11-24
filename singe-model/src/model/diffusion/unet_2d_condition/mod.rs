//! `diffusers`' `UNet2DConditionModel`, Stable Diffusion's denoiser: residual blocks reading the timestep, and in the blocks that ask for it transformers whose blocks attend to the image's positions and then to the text's states, down to a middle block and back up.
//!
//! Its state is that of `UNet2DModel` (`input`, `time`, `down.N`, `middle`, `up.N`, `norm`, `output`), with transformers at `attentions.M` (`norm`, `proj_in`, `blocks.K`, `proj_out`), each block's `attention_norm`, `attention`, `cross_attention_norm`, `cross_attention`, `feed_forward_norm`, and `feed_forward` (`gate`, `up`, `down`).

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, GatedMlp, GroupNorm, LayerNorm, Linear, Model,
        ModelState, StateKey,
    },
};

use crate::{
    assembly::{
        Attention, AttentionSpec, PYTORCH_NORM_EPS, Unet, UnetConfig, UnetSite, same_2d, spatial,
        to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{ConditionalDenoiser, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::{Format, LoadOptions, Source},
};

const TRANSFORMER_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unet2dConditionConfig {
    #[serde(flatten)]
    pub unet: UnetConfig,
    /// Each down block's heads; the up blocks take them reversed, and the middle the last.
    pub heads: Vec<i64>,
    pub context_width: i64,
    pub linear_projection: bool,
}

/// `BasicTransformerBlock`: self-attention, attention to the context, and a GEGLU feed-forward layer, each after a layer norm and added to its input.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: Attention,
    cross_attention_norm: LayerNorm,
    cross_attention: Attention,
    feed_forward_norm: LayerNorm,
    feed_forward: GatedMlp,
}

impl TransformerBlock {
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        context: i64,
        heads: i64,
    ) -> SingeResult<Self> {
        let inner = 4 * width;
        let spec = |context| AttentionSpec {
            width,
            context,
            inner: width,
            heads,
            qkv_bias: false,
            output_bias: true,
        };
        let mut scope = schema.scope("feed_forward");
        let feed_forward = GatedMlp::new(
            scope.linear("gate", width, inner, true)?,
            scope.linear("up", width, inner, true)?,
            scope.linear("down", inner, width, true)?,
            Activation::GeluExact,
        );
        scope.join_rows(&feed_forward.siblings())?;
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            attention: Attention::create(&mut schema.scope("attention"), spec(width))?,
            cross_attention_norm: schema.layer_norm(
                "cross_attention_norm",
                width,
                PYTORCH_NORM_EPS,
            )?,
            cross_attention: Attention::create(
                &mut schema.scope("cross_attention"),
                spec(context),
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward,
        })
    }

    /// Runs `hidden`, `[B, L, W]`, through the block, with `positions` added to each attention's normed input where given, its second attention attending to `context`, `[B, M, C]`, or, without one, to its own normed input.
    pub fn forward(
        &self,
        hidden: &Tensor,
        positions: Option<&Tensor>,
        context: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let positioned = |normed: Tensor| match positions {
            Some(positions) => normed.add(positions),
            None => Ok(normed),
        };
        let normed = positioned(hidden.apply(&self.attention_norm, state)?)?;
        let hidden = hidden.add(&self.attention.forward(&normed, &normed, &normed, state)?)?;
        let normed = positioned(hidden.apply(&self.cross_attention_norm, state)?)?;
        let context = context.unwrap_or(&normed);
        let hidden = hidden.add(
            &self
                .cross_attention
                .forward(&normed, context, context, state)?,
        )?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?;
        hidden.add(&fed)
    }
}

#[derive(Debug, Clone)]
enum Projection {
    Linear(Linear),
    Convolution(Conv2d),
}

/// `Transformer2DModel` over an image's positions: group-normed, projected in, its blocks, projected out, and added to its input.
#[derive(Debug, Clone)]
pub struct Transformer2d {
    norm: GroupNorm,
    proj_in: Projection,
    blocks: Vec<TransformerBlock>,
    proj_out: Projection,
}

impl Transformer2d {
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        context: i64,
        heads: i64,
        linear: bool,
        groups: i64,
    ) -> SingeResult<Self> {
        let projection = |schema: &mut Schema<'_>, name: &str| -> SingeResult<Projection> {
            Ok(if linear {
                Projection::Linear(schema.linear(name, width, width, true)?)
            } else {
                Projection::Convolution(schema.conv2d(
                    name,
                    width,
                    width,
                    1,
                    Conv2dConfig::default(),
                    true,
                )?)
            })
        };
        Ok(Self {
            norm: schema.group_norm("norm", width, groups, TRANSFORMER_NORM_EPS)?,
            proj_in: projection(schema, "proj_in")?,
            blocks: vec![TransformerBlock::create(
                &mut schema.scope("blocks.0"),
                width,
                context,
                heads,
            )?],
            proj_out: projection(schema, "proj_out")?,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        context: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let size = spatial(input)?;
        let normed = input.apply(&self.norm, state)?;
        let mut hidden = match &self.proj_in {
            Projection::Linear(linear) => to_tokens(&normed)?.apply(linear, state)?,
            Projection::Convolution(conv) => to_tokens(&normed.apply(conv, state)?)?,
        };
        for block in &self.blocks {
            hidden = block.forward(&hidden, None, Some(context), state)?;
        }

        let output = match &self.proj_out {
            Projection::Linear(linear) => to_map(&hidden.apply(linear, state)?, size)?,
            Projection::Convolution(conv) => to_map(&hidden, size)?.apply(conv, state)?,
        };
        output.add(input)
    }
}

#[derive(Debug, Clone)]
pub struct Unet2dConditionLayers {
    config: Unet2dConditionConfig,
    unet: Unet<Transformer2d>,
}

impl Unet2dConditionLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Unet2dConditionConfig) -> SingeResult<Self> {
        let last = config.unet.widths.len() - 1;
        Ok(Self {
            config: config.clone(),
            unet: Unet::create(schema, &config.unet, |schema, site, width| {
                let heads = match site {
                    UnetSite::Down(index) => config.heads[index],
                    UnetSite::Middle => config.heads[last],
                    UnetSite::Up(index) => config.heads[last - index],
                };
                Transformer2d::create(
                    schema,
                    width,
                    config.context_width,
                    heads,
                    config.linear_projection,
                    config.unet.groups,
                )
            })?,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Unet2dConditionConfig {
        &self.config
    }

    /// Returns the predicted noise of `sample`, `[B, C, H, W]`, at the timestep whose features `time` holds, `[1, F]`, conditioned on `context`, `[B, T, D]`.
    ///
    /// With `motion`, whose clips' frames make up the batch, its temporal transformers run after each block that has one.
    pub fn predict(
        &self,
        sample: &Tensor,
        time: &Tensor,
        context: &Tensor,
        motion: Option<&Motion<'_>>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let input =
            match motion.and_then(|motion| Some((motion.modules.input.as_ref()?, motion.state))) {
                // The UNet's channels through its own convolution, and those past them through the adapter's.
                Some((input, motion_state)) => {
                    let &[_, channels, _, _] = sample.shape().dimensions() else {
                        return Err(SingeError::invalid_argument("UNet sample"));
                    };
                    let latents = self.config.unet.in_channels;
                    self.unet
                        .embed(&sample.narrow(1, 0, latents)?, state)?
                        .add(
                            &sample
                                .narrow(1, latents, channels - latents)?
                                .apply(input, motion_state)?,
                        )?
                }
                None => self.unet.embed(sample, state)?,
            };
        self.unet
            .denoise(input, time, state, |hidden, attention, site, index| {
                let hidden = match attention {
                    Some(attention) => attention.forward(&hidden, context, state)?,
                    None => hidden,
                };
                let Some(motion) = motion else {
                    return Ok(hidden);
                };
                let modules = motion.modules;
                let module = match site {
                    UnetSite::Down(number) => {
                        modules.down.get(number).and_then(|stage| stage.get(index))
                    }
                    UnetSite::Middle => modules.middle.first(),
                    UnetSite::Up(number) => {
                        modules.up.get(number).and_then(|stage| stage.get(index))
                    }
                };
                match module {
                    Some(module) => module.forward(&hidden, motion.frames, motion.state),
                    None => Ok(hidden),
                }
            })
    }
}

pub type Unet2dCondition = Model<Unet2dConditionLayers>;

impl ConditionalDenoiser for Unet2dCondition {
    fn in_channels(&self) -> i64 {
        self.layer().config.unet.in_channels
    }

    fn predict(&self, sample: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let data_type = self.state().floating_type()?;
        let time = self
            .layer()
            .config
            .unet
            .time_input(timestep, self.state())?;
        Ok(self
            .layer()
            .predict(
                &sample.cast(data_type)?,
                &time,
                &context.cast(data_type)?,
                None,
                self.state(),
            )?
            .cast(DataType::F32)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionConfig {
    pub widths: Vec<i64>,
    pub layers_per_block: usize,
    pub heads: i64,
    /// The most frames the stored positions cover.
    pub max_positions: i64,
    pub middle: bool,
    pub groups: i64,
    /// The input channels of an adapter with its own input convolution, as PIA's reads its image condition beside the latents.
    ///
    /// The U-Net's input convolution reads the latents' channels, and the adapter's the channels past them.
    pub in_channels: Option<i64>,
}

/// `BasicTransformerBlock` over frames, whose second attention attends to the frames themselves, each attention reading its input with the frames' sinusoidal positions added.
#[derive(Debug, Clone)]
struct TemporalBlock {
    block: TransformerBlock,
    positions: StateKey,
}

/// `AnimateDiffTransformer3D`: each position of a clip's frames attending across the frames, group-normed over the whole clip, and added to its input.
#[derive(Debug, Clone)]
pub struct TemporalTransformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<TemporalBlock>,
    proj_out: Linear,
}

impl TemporalTransformer {
    fn create(schema: &mut Schema<'_>, width: i64, config: &MotionConfig) -> SingeResult<Self> {
        let mut scope = schema.scope("blocks.0");
        let block = TemporalBlock {
            block: TransformerBlock::create(&mut scope, width, width, config.heads)?,
            positions: scope.parameter("positions", [1, config.max_positions, width])?,
        };
        Ok(Self {
            norm: schema.group_norm("norm", width, config.groups, TRANSFORMER_NORM_EPS)?,
            proj_in: schema.linear("proj_in", width, width, true)?,
            blocks: vec![block],
            proj_out: schema.linear("proj_out", width, width, true)?,
        })
    }

    /// Runs `input`, `[B · F, C, H, W]` with each clip's `frames` together, through the transformer.
    fn forward(&self, input: &Tensor, frames: i64, state: &ModelState) -> SingeResult<Tensor> {
        let &[rows, channels, height, width] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("temporal transformer input"));
        };
        let clips = rows / frames;
        // The norm's statistics span each clip's frames, as a 5-D group norm's do.
        let normed = input
            .reshape([clips, frames, channels, height, width])?
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?
            .reshape([clips, channels, frames * height, width])?
            .apply(&self.norm, state)?
            .reshape([clips, channels, frames, height, width])?;
        let mut hidden = normed
            .permute(&[0, 3, 4, 2, 1])?
            .contiguous()?
            .reshape([clips * height * width, frames, channels])?
            .apply(&self.proj_in, state)?;
        for TemporalBlock { block, positions } in &self.blocks {
            let positions = state.tensor(positions)?.narrow(1, 0, frames)?;
            hidden = block.forward(&hidden, Some(&positions), None, state)?;
        }
        hidden
            .apply(&self.proj_out, state)?
            .reshape([clips, height, width, frames, channels])?
            .permute(&[0, 3, 4, 1, 2])?
            .contiguous()?
            .reshape([rows, channels, height, width])?
            .add(input)
    }
}

/// A `MotionAdapter`'s temporal transformers, one after each of the U-Net's down and up blocks' layers and, where it has one, after the middle's attention.
#[derive(Debug, Clone)]
pub struct MotionModules {
    input: Option<Conv2d>,
    down: Vec<Vec<TemporalTransformer>>,
    middle: Vec<TemporalTransformer>,
    up: Vec<Vec<TemporalTransformer>>,
}

impl MotionModules {
    /// Creates the modules for a U-Net whose own input convolution reads `latents` channels.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &MotionConfig,
        latents: i64,
    ) -> SingeResult<Self> {
        let stages = |schema: &mut Schema<'_>, name: &str, widths: &[i64], layers: usize| {
            widths
                .iter()
                .enumerate()
                .map(|(index, &width)| {
                    (0..layers)
                        .map(|layer| {
                            TemporalTransformer::create(
                                &mut schema.scope(format!("{name}.{index}.motions.{layer}")),
                                width,
                                config,
                            )
                        })
                        .collect::<SingeResult<Vec<_>>>()
                })
                .collect::<SingeResult<Vec<_>>>()
        };
        let reversed = config.widths.iter().rev().copied().collect::<Vec<_>>();
        let (first, last) = (config.widths[0], reversed[0]);
        Ok(Self {
            // The adapter's weights for the channels past the latents', which `from_unet2d` splices after the UNet's own, with the UNet's bias.
            input: config
                .in_channels
                .map(|channels| {
                    schema.conv2d("input", channels - latents, first, 3, same_2d(3, 1), false)
                })
                .transpose()?,
            down: stages(schema, "down", &config.widths, config.layers_per_block)?,
            middle: if config.middle {
                vec![TemporalTransformer::create(
                    &mut schema.scope("middle.motions.0"),
                    last,
                    config,
                )?]
            } else {
                Vec::new()
            },
            up: stages(schema, "up", &reversed, config.layers_per_block + 1)?,
        })
    }
}

/// The motion modules a U-Net reads, their state, and the frames of each clip in its batch.
#[derive(Debug, Clone, Copy)]
pub struct Motion<'a> {
    pub modules: &'a MotionModules,
    pub state: &'a ModelState,
    pub frames: i64,
}

/// A U-Net with a motion adapter, as `UNetMotionModel` is, each with its own state.
#[derive(Debug)]
pub struct UnetMotion {
    pub unet: Unet2dCondition,
    pub motion: Model<MotionModules>,
}

impl UnetMotion {
    /// Returns the predicted noise of clips' latents, `[B, C, F, H, W]`, at `timestep`, each clip conditioned on its row of `context`, `[B, T, D]`, in the model's type, as `diffusers` keeps a pipeline's latents in it.
    pub fn predict(&self, sample: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let &[clips, channels, frames, height, width] = sample.shape().dimensions() else {
            return Err(SingeError::invalid_argument("clip latents").into());
        };
        let &[_, tokens, features] = context.shape().dimensions() else {
            return Err(SingeError::invalid_argument("clip context").into());
        };
        let unet = self.unet.layer();
        let data_type = self.unet.state().floating_type()?;
        let time = unet.config.unet.time_input(timestep, self.unet.state())?;
        let frames_first = sample
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?
            .reshape([clips * frames, channels, height, width])?
            .cast(data_type)?;
        // Each clip's text states, once for each of its frames.
        let repeated = context
            .cast(data_type)?
            .unsqueeze(1)?
            .expand(Shape::create([clips, frames, tokens, features])?)?
            .contiguous()?
            .reshape([clips * frames, tokens, features])?;
        let motion = Motion {
            modules: self.motion.layer(),
            state: self.motion.state(),
            frames,
        };
        Ok(unet
            .predict(
                &frames_first,
                &time,
                &repeated,
                Some(&motion),
                self.unet.state(),
            )?
            .reshape([clips, frames, -1, height, width])?
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "unet_2d_condition",
    formats: &[(Format::Diffusers, &["UNet2DConditionModel"])],
    loaders: &[HeadLoader::ConditionalDenoiser(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Unet2dCondition> {
    let config: Unet2dConditionConfig = loader.config(diffusers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Unet2dConditionLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &diffusers::rules())
}

/// Loads the U-Net from `unet` and its motion adapter from `adapter`.
pub fn load_motion(unet: &Source, adapter: &Source, options: LoadOptions) -> Result<UnetMotion> {
    let unet = load(&mut Loader::new(unet, options))?;
    let latents = unet.layer().config.unet.in_channels;
    let mut loader = Loader::new(adapter, options);
    let config: MotionConfig = loader.config(diffusers::motion_config)?;
    let motion = Schema::model(options.data_type, |schema| {
        MotionModules::create(schema, &config, latents)
    })?;
    let motion = loader.load_whole(&motion, &diffusers::motion_rules(latents))?;
    Ok(UnetMotion { unet, motion })
}
