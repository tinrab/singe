// `diffusers`' `UNet2DConditionModel`, Stable Diffusion's denoiser: residual blocks reading the timestep, and in the blocks that ask for it transformers whose blocks attend to the image's positions and then to the text's states, down to a middle block and back up.
//
// Its state is that of `UNet2DModel` (`input`, `time`, `down.N`, `middle`, `up.N`, `norm`, `output`), with transformers at `attentions.M` (`norm`, `proj_in`, `blocks.K`, `proj_out`), each block's `norm1`, `attn1`, `norm2`, `attn2`, `norm3`, and `feed_forward` (`gate`, `up`, `down`).

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, GatedMlp, GroupNorm, LayerNorm, Linear, Model,
        ModelState, StateBuilder, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Norms, TimeEmbedding, UnetLayout, UnetSite, UnetStages, conv3x3, timestep_input},
    error::Result,
    format::Loader,
    head::{ConditionalDenoiser, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::{Format, LoadOptions, Source},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unet2dConditionConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub sample_size: i64,
    pub widths: Vec<i64>,
    // Whether each down block, and each up block, has transformers.
    pub down_attention: Vec<bool>,
    pub up_attention: Vec<bool>,
    // Each down block's heads; the up blocks take them reversed, and the middle the last.
    pub heads: Vec<i64>,
    pub layers_per_block: usize,
    pub context_width: i64,
    pub linear_projection: bool,
    pub groups: i64,
    pub eps: f32,
    pub downsample_padding: i64,
    pub flip_sin_to_cos: bool,
    pub freq_shift: f32,
    // What the middle block's residual blocks divide their sums by.
    pub middle_scale: f64,
}

#[derive(Debug, Clone)]
pub struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl Attention {
    pub fn create(
        schema: &mut Schema<'_>,
        width: i64,
        context: i64,
        heads: i64,
    ) -> SingeResult<Self> {
        Ok(Self {
            query: schema.linear("query", width, width, false)?,
            key: schema.linear("key", context, width, false)?,
            value: schema.linear("value", context, width, false)?,
            output: schema.linear("output", width, width, true)?,
            heads,
        })
    }

    // `hidden`, `[B, L, W]`, attending to `context`, `[B, M, C]`, each sequence alone.
    pub fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let width = dimensions[2];
        let heads = |input: &Tensor, projection: &Linear| -> SingeResult<Tensor> {
            let dimensions = input.shape().dimensions();
            input
                .apply(projection, state)?
                .reshape([dimensions[0], dimensions[1], self.heads, width / self.heads])?
                .permute(&[0, 2, 1, 3])
        };
        heads(hidden, &self.query)?
            .attend(
                &heads(context, &self.key)?,
                &heads(context, &self.value)?,
                None,
                None,
                AttentionOptions::default(),
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([dimensions[0], dimensions[1], width])?
            .apply(&self.output, state)
    }
}

// `BasicTransformerBlock`: self-attention, attention to the context, and a GEGLU feed-forward layer, each after a layer norm and added to its input.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: Attention,
    norm3: LayerNorm,
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
        let mut feed_forward = schema.scope("feed_forward");
        let feed_forward = GatedMlp::new(
            feed_forward.linear("gate", width, inner, true)?,
            feed_forward.linear("up", width, inner, true)?,
            feed_forward.linear("down", inner, width, true)?,
            Activation::GeluExact,
        );
        Ok(Self {
            norm1: schema.layer_norm("norm1", width, 1e-5)?,
            attn1: Attention::create(&mut schema.scope("attn1"), width, width, heads)?,
            norm2: schema.layer_norm("norm2", width, 1e-5)?,
            attn2: Attention::create(&mut schema.scope("attn2"), width, context, heads)?,
            norm3: schema.layer_norm("norm3", width, 1e-5)?,
            feed_forward,
        })
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = hidden.apply(&self.norm1, state)?;
        let hidden = hidden.add(&self.attn1.forward(&normed, &normed, state)?)?;
        let normed = hidden.apply(&self.norm2, state)?;
        let hidden = hidden.add(&self.attn2.forward(&normed, context, state)?)?;
        let fed = hidden
            .apply(&self.norm3, state)?
            .apply(&self.feed_forward, state)?;
        hidden.add(&fed)
    }
}

#[derive(Debug, Clone)]
enum Projection {
    Linear(Linear),
    Convolution(Conv2d),
}

// `Transformer2DModel` over an image's positions: group-normed, projected in, its blocks, projected out, and added to its input.
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
            norm: schema.group_norm("norm", width, groups, 1e-6)?,
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
        let dimensions = input.shape().dimensions();
        let (batch, width, height, columns) =
            (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
        let flatten = |image: &Tensor| -> SingeResult<Tensor> {
            image
                .permute(&[0, 2, 3, 1])?
                .reshape([batch, height * columns, width])
        };
        let unflatten = |sequence: &Tensor| -> SingeResult<Tensor> {
            sequence
                .reshape([batch, height, columns, width])?
                .permute(&[0, 3, 1, 2])?
                .contiguous()
        };
        let normed = input.apply(&self.norm, state)?;
        let mut hidden = match &self.proj_in {
            Projection::Linear(linear) => flatten(&normed)?.apply(linear, state)?,
            Projection::Convolution(conv) => flatten(&normed.apply(conv, state)?)?,
        };
        for block in &self.blocks {
            hidden = block.forward(&hidden, context, state)?;
        }

        let output = match &self.proj_out {
            Projection::Linear(linear) => unflatten(&hidden.apply(linear, state)?)?,
            Projection::Convolution(conv) => unflatten(&hidden)?.apply(conv, state)?,
        };
        output.add(input)
    }
}

#[derive(Debug, Clone)]
pub struct Unet2dConditionLayers {
    config: Unet2dConditionConfig,
    input: Conv2d,
    time: TimeEmbedding,
    stages: UnetStages<Transformer2d>,
    norm: GroupNorm,
    output: Conv2d,
}

impl Unet2dConditionLayers {
    #[must_use]
    pub const fn config(&self) -> &Unet2dConditionConfig {
        &self.config
    }

    pub fn create(schema: &mut Schema<'_>, config: &Unet2dConditionConfig) -> SingeResult<Self> {
        let norms = Norms {
            groups: config.groups,
            eps: config.eps,
        };
        let widths = &config.widths;
        let time_width = widths[0] * 4;
        let last = widths.len() - 1;
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
                config.groups,
            )
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

    // The predicted noise of `sample`, `[B, C, H, W]`, at the timestep whose features `time` holds, `[1, F]`, conditioned on `context`, `[B, T, D]`, with `motion`'s temporal transformers after each block that has one, where frames of clips make up the batch.
    pub fn predict(
        &self,
        sample: &Tensor,
        time: &Tensor,
        context: &Tensor,
        motion: Option<&Motion<'_>>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let time = time.apply(&self.time, state)?;
        let input = match motion
            .and_then(|motion| Some((motion.modules.input.as_ref()?, motion.state)))
        {
            // The UNet's channels through its own convolution, and those past them through the adapter's.
            Some((input, motion_state)) => {
                let (channels, latents) = (sample.shape().dimensions()[1], self.config.in_channels);
                sample
                    .narrow(1, 0, latents)?
                    .apply(&self.input, state)?
                    .add(
                        &sample
                            .narrow(1, latents, channels - latents)?
                            .apply(input, motion_state)?,
                    )?
            }
            None => sample.apply(&self.input, state)?,
        };
        self.stages
            .forward(input, &time, state, |hidden, attention, site, index| {
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
            })?
            .apply(&self.norm, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)
    }
}

pub type Unet2dCondition = Model<Unet2dConditionLayers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "unet_2d_condition",
    formats: &[(Format::Diffusers, &["UNet2DConditionModel"])],
    heads: &[HeadLoader::ConditionalDenoiser(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Unet2dCondition> {
    let config: Unet2dConditionConfig = loader.config(diffusers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        Unet2dConditionLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "unet",
        &Model::new(layers, builder.finish()?),
        &diffusers::rules(),
    )?;
    loader.finish()?;
    Ok(model)
}

impl ConditionalDenoiser for Unet2dCondition {
    fn in_channels(&self) -> i64 {
        self.layer().config.in_channels
    }

    fn predict(&self, sample: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let config = &self.layer().config;
        let data_type = self.state().floating_type()?;
        let time = timestep_input(
            timestep,
            config.widths[0],
            config.flip_sin_to_cos,
            config.freq_shift,
            self.state(),
        )?;
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
    // The most frames the learned positions cover.
    pub positions: i64,
    pub middle: bool,
    pub groups: i64,
    // The input channels of the adapter's own input convolution, which replaces the UNet's, as PIA's reads its image condition.
    pub in_channels: Option<i64>,
}

// `BasicTransformerBlock` over frames: a transformer block whose second attention attends to the frames themselves, each attention reading its input with the frames' sinusoidal positions added.
#[derive(Debug, Clone)]
struct TemporalBlock {
    block: TransformerBlock,
    positions: StateKey,
}

impl TemporalBlock {
    fn forward(&self, hidden: &Tensor, frames: i64, state: &ModelState) -> SingeResult<Tensor> {
        let block = &self.block;
        let positions = state.tensor(&self.positions)?.narrow(1, 0, frames)?;
        let normed = hidden.apply(&block.norm1, state)?.add(&positions)?;
        let hidden = hidden.add(&block.attn1.forward(&normed, &normed, state)?)?;
        let normed = hidden.apply(&block.norm2, state)?.add(&positions)?;
        let hidden = hidden.add(&block.attn2.forward(&normed, &normed, state)?)?;
        let fed = hidden
            .apply(&block.norm3, state)?
            .apply(&block.feed_forward, state)?;
        hidden.add(&fed)
    }
}

// `AnimateDiffTransformer3D`: each position of a clip's frames attending across the frames, group-normed over the whole clip, and added to its input.
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
            positions: scope.parameter("positions", [1, config.positions, width])?,
        };
        Ok(Self {
            norm: schema.group_norm("norm", width, config.groups, 1e-6)?,
            proj_in: schema.linear("proj_in", width, width, true)?,
            blocks: vec![block],
            proj_out: schema.linear("proj_out", width, width, true)?,
        })
    }

    // `input`, `[B · F, C, H, W]` with each clip's frames together, through the transformer.
    fn forward(&self, input: &Tensor, frames: i64, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = input.shape().dimensions();
        let (rows, channels, height, width) =
            (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
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
        for block in &self.blocks {
            hidden = block.forward(&hidden, frames, state)?;
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

// A `MotionAdapter`'s temporal transformers, one after each of the UNet's down and up blocks' layers and, where it has one, after the middle's attention.
#[derive(Debug, Clone)]
pub struct MotionModules {
    input: Option<Conv2d>,
    down: Vec<Vec<TemporalTransformer>>,
    middle: Vec<TemporalTransformer>,
    up: Vec<Vec<TemporalTransformer>>,
}

impl MotionModules {
    // The modules for a UNet whose own input convolution reads `latents` channels.
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
        let last = config.widths[config.widths.len() - 1];
        Ok(Self {
            // The adapter's weights for the channels past the latents', which `from_unet2d` splices after the UNet's own, with the UNet's bias.
            input: config
                .in_channels
                .map(|channels| {
                    schema.conv2d(
                        "input",
                        channels - latents,
                        config.widths[0],
                        3,
                        Conv2dConfig {
                            padding: (1, 1, 1, 1),
                            ..Conv2dConfig::default()
                        },
                        false,
                    )
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

// The motion modules a UNet reads, their state, and the frames of each clip in its batch.
#[derive(Debug, Clone, Copy)]
pub struct Motion<'a> {
    pub modules: &'a MotionModules,
    pub state: &'a ModelState,
    pub frames: i64,
}

// A UNet with a motion adapter, as `UNetMotionModel` is, each with its own state.
#[derive(Debug)]
pub struct UnetMotion {
    pub unet: Unet2dCondition,
    pub motion: Model<MotionModules>,
}

// Loads the UNet from `unet` and its motion adapter from `adapter`.
pub fn load_motion(unet: &Source, adapter: &Source, options: LoadOptions) -> Result<UnetMotion> {
    let unet = load(&mut Loader::new(unet, options))?;
    let latents = unet.layer().config.in_channels;
    let mut loader = Loader::new(adapter, options);
    let config: MotionConfig = loader.config(diffusers::motion_config)?;
    let mut builder = StateBuilder::new();
    let modules = {
        let mut schema = Schema::root(&mut builder, options.data_type);
        MotionModules::create(&mut schema, &config, latents)?
    };
    let motion = loader.load(
        "motion",
        &Model::new(modules, builder.finish()?),
        &diffusers::motion_rules(latents),
    )?;
    loader.finish()?;
    Ok(UnetMotion { unet, motion })
}

impl UnetMotion {
    // The predicted noise of clips' latents, `[B, C, F, H, W]`, at `timestep`, each clip conditioned on its row of `context`, `[B, T, D]`, in the model's type, as `diffusers` keeps a pipeline's latents in it.
    pub fn predict(&self, sample: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor> {
        let dimensions = sample.shape().dimensions();
        let (clips, channels, frames, height, width) = (
            dimensions[0],
            dimensions[1],
            dimensions[2],
            dimensions[3],
            dimensions[4],
        );
        let config = &self.unet.layer().config;
        let data_type = self.unet.state().floating_type()?;
        let time = timestep_input(
            timestep,
            config.widths[0],
            config.flip_sin_to_cos,
            config.freq_shift,
            self.unet.state(),
        )?;
        let frames_first = sample
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?
            .reshape([clips * frames, channels, height, width])?
            .cast(data_type)?;
        // Each clip's text states, once for each of its frames.
        let context_dimensions = context.shape().dimensions();
        let repeated = context
            .cast(data_type)?
            .unsqueeze(1)?
            .expand(Shape::create([
                clips,
                frames,
                context_dimensions[1],
                context_dimensions[2],
            ])?)?
            .contiguous()?
            .reshape([clips * frames, context_dimensions[1], context_dimensions[2]])?;
        let motion = Motion {
            modules: self.motion.layer(),
            state: self.motion.state(),
            frames,
        };
        Ok(self
            .unet
            .layer()
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
