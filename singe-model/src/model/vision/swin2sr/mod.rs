// Swin2SR: a convolution into features, stages of Swin V2 blocks attending within windows, shifted every other block, by scaled cosine similarity and a learned bias of relative positions, then a convolution and pixel shuffles up to a larger image.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Layer, LayerNorm, Linear, Model, ModelState, StateBuilder, StateKey},
};

use super::same;
use crate::{
    assembly::{merge_windows, partition_windows, roll, shift_mask, to_map, to_tokens},
    error::{Error, Result},
    format::Loader,
    head::{HeadLoader, ImageUpscaler, PixelLayers},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// The mean color DIV2K's images have, which the model subtracts from three-channel images.
const MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Swin2SrConfig {
    pub width: i64,
    // Each stage's blocks and heads.
    pub stages: Vec<(i64, i64)>,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub window: i64,
    pub channels: i64,
    pub upscale: i64,
    // The upsampler's features.
    pub upsampler: i64,
}

// The log-spaced relative coordinates the position bias network reads, `[(2w - 1)², 2]`, and each pair of a window's points' index among them, `[w⁴]`, as `create_coords_table_and_index` makes them.
#[expect(clippy::cast_precision_loss, reason = "window sides are small")]
fn relative_positions(window: i64) -> (Vec<f32>, Vec<i64>) {
    let span = window - 1;
    let scaled = |offset: i64| {
        let value = offset as f32 / span as f32 * 8.0;
        value.signum() * (value.abs() + 1.0).log2() / 8_f32.log2()
    };
    let table = (-span..=span)
        .flat_map(|row| (-span..=span).flat_map(move |column| [scaled(row), scaled(column)]))
        .collect();
    let points = (0..window)
        .flat_map(|row| (0..window).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    let index = points
        .iter()
        .flat_map(|&(row, column)| {
            points.iter().map(move |&(other_row, other_column)| {
                (row - other_row + span) * (2 * window - 1) + (column - other_column + span)
            })
        })
        .collect();
    (table, index)
}

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    logit_scale: StateKey,
    bias_hidden: Linear,
    bias_out: Linear,
    heads: i64,
}

#[derive(Debug, Clone)]
struct Block {
    attention: Attention,
    attention_norm: LayerNorm,
    up: Linear,
    down: Linear,
    feed_forward_norm: LayerNorm,
    activation: Activation,
    shift: i64,
}

#[derive(Debug, Clone)]
struct Stage {
    blocks: Vec<Block>,
    convolution: Conv2d,
    projection: Conv2d,
}

// Swin2SR and its upsampler: from images in [0, 1] whose sides are whole windows to images `upscale` times larger.
#[derive(Debug, Clone)]
pub struct Swin2SrLayers {
    first: Conv2d,
    projection: Conv2d,
    projection_norm: LayerNorm,
    stages: Vec<Stage>,
    norm: LayerNorm,
    after_body: Conv2d,
    before_upsample: Conv2d,
    upsample: Vec<Conv2d>,
    last: Conv2d,
    window: i64,
}

fn mean(device: Device) -> SingeResult<Tensor> {
    Tensor::from_vec(device, Shape::create([1, 3, 1, 1])?, MEAN.to_vec())
}

impl Swin2SrLayers {
    fn block(
        &self,
        block: &Block,
        hidden: &Tensor,
        (height, width): (i64, i64),
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let window = self.window;
        let dimensions = hidden.shape().dimensions();
        let (batch, channels) = (dimensions[0], dimensions[2]);
        let attention = &block.attention;
        let heads = attention.heads;
        let head = channels / heads;
        let device = hidden.device();

        let shifted = roll(
            &hidden.reshape([batch, height, width, channels])?,
            -block.shift,
        )?;
        let windows = partition_windows(&shifted, window)?;
        let count = windows.shape().dimensions()[0];
        let area = window * window;
        let split = |tensor: Tensor| tensor.reshape([count, area, heads, head])?.transpose(1, 2);
        let unit = |tensor: Tensor| -> SingeResult<Tensor> {
            let norm = tensor
                .mul(&tensor)?
                .sum(&[-1], true)?
                .sqrt()?
                .clamp(1e-12, f32::MAX)?;
            tensor.div(&norm)
        };
        let query = unit(split(windows.apply(&attention.query, state)?)?)?;
        let key = unit(split(windows.apply(&attention.key, state)?)?)?;
        let value = split(windows.apply(&attention.value, state)?)?;
        let scale = state
            .tensor(&attention.logit_scale)?
            .clamp(f32::MIN, 100_f32.ln())?
            .exp()?
            .unsqueeze(0)?;
        let mut scores = query.matmul(&key.transpose(2, 3)?)?.mul(&scale)?;

        // The bias of each pair of points' relative position, from a small network over log-spaced coordinates.
        let (table, index) = relative_positions(window);
        let entries = (2 * window - 1) * (2 * window - 1);
        let table = Tensor::from_vec(device, Shape::create([entries, 2])?, table)?
            .cast(hidden.data_type())?;
        let biases = table
            .apply(&attention.bias_hidden, state)?
            .apply(&Activation::Relu, state)?
            .apply(&attention.bias_out, state)?;
        let index = Tensor::from_vec(device, Shape::create([area * area, 1])?, index)?
            .expand(Shape::create([area * area, heads])?)?;
        let bias = biases
            .gather(0, &index)?
            .reshape([area, area, heads])?
            .permute(&[2, 0, 1])?
            .apply(&Activation::Sigmoid, state)?
            .mul_scalar(16.0)?
            .unsqueeze(0)?;
        scores = scores.add(&bias)?;
        if block.shift > 0 {
            // -100 twice, as `Swin2SRSelfAttention` adds the mask twice.
            let windows_per_image = count / batch;
            let mask = Tensor::from_vec(
                device,
                Shape::create([1, windows_per_image, 1, area, area])?,
                shift_mask(height, width, window, block.shift, -200.0),
            )?
            .cast(hidden.data_type())?;
            scores = scores
                .reshape([batch, windows_per_image, heads, area, area])?
                .add(&mask)?
                .reshape([count, heads, area, area])?;
        }

        let attended = scores
            .softmax(-1)?
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape([count, area, channels])?
            .apply(&attention.output, state)?;
        let attended = roll(
            &merge_windows(&attended, window, (batch, height, width))?,
            block.shift,
        )?
        .reshape([batch, height * width, channels])?;

        let hidden = attended.apply(&block.attention_norm, state)?.add(hidden)?;
        hidden
            .apply(&block.up, state)?
            .apply(&block.activation, state)?
            .apply(&block.down, state)?
            .apply(&block.feed_forward_norm, state)?
            .add(&hidden)
    }
}

impl Layer for Swin2SrLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = input.shape().dimensions();
        let size = (dimensions[2], dimensions[3]);
        if size.0 % self.window != 0 || size.1 % self.window != 0 {
            return Err(singe::Error::invalid_argument(
                "an image whose sides are not whole windows",
            ));
        }

        let mean = mean(input.device())?.cast(input.data_type())?;
        let features = input.sub(&mean)?.apply(&self.first, state)?;
        let mut hidden = to_tokens(&features.apply(&self.projection, state)?)?
            .apply(&self.projection_norm, state)?;
        for stage in &self.stages {
            let residual = hidden.clone();
            for block in &stage.blocks {
                hidden = self.block(block, &hidden, size, state)?;
            }

            hidden = to_tokens(
                &to_map(&hidden, size)?
                    .apply(&stage.convolution, state)?
                    .apply(&stage.projection, state)?,
            )?
            .add(&residual)?;
        }

        let body = to_map(&hidden.apply(&self.norm, state)?, size)?
            .apply(&self.after_body, state)?
            .add(&features)?;

        let mut upsampled = body.apply(&self.before_upsample, state)?.leaky_relu(0.01)?;
        for convolution in &self.upsample {
            upsampled = pixel_shuffle(&upsampled.apply(convolution, state)?, 2)?;
        }
        upsampled.apply(&self.last, state)?.add(&mean)
    }
}

// `[B, C · r², H, W]` as `[B, C, H · r, W · r]`, as `nn.PixelShuffle(r)` rearranges it.
fn pixel_shuffle(input: &Tensor, factor: i64) -> SingeResult<Tensor> {
    let dimensions = input.shape().dimensions();
    let (batch, channels, height, width) = (
        dimensions[0],
        dimensions[1] / (factor * factor),
        dimensions[2],
        dimensions[3],
    );
    input
        .reshape([batch, channels, factor, factor, height, width])?
        .permute(&[0, 1, 4, 2, 5, 3])?
        .reshape([batch, channels, height * factor, width * factor])
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "swin2sr",
    formats: &[(Format::Transformers, &["swin2sr"])],
    heads: &[HeadLoader::SuperResolution(load_upscaler)],
};

pub fn load_upscaler(loader: &mut Loader<'_>) -> Result<ImageUpscaler> {
    let config: Swin2SrConfig = loader.config(transformers::config)?;
    if config.channels != 3 {
        return Err(Error::unsupported("a Swin2SR of other than three channels"));
    }

    let width = config.width;
    let eps = config.eps;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(index, &(depth, heads))| {
                let mut schema = schema.scope(format!("stages.{index}"));
                let blocks = (0..depth)
                    .map(|block| {
                        let mut schema = schema.scope(format!("blocks.{block}"));
                        let attention = {
                            let mut schema = schema.scope("attention");
                            Attention {
                                query: schema.linear("query", width, width, true)?,
                                key: schema.linear("key", width, width, false)?,
                                value: schema.linear("value", width, width, true)?,
                                output: schema.linear("output", width, width, true)?,
                                logit_scale: schema.parameter("logit_scale", [heads, 1, 1])?,
                                bias_hidden: schema.linear("bias_hidden", 2, 512, true)?,
                                bias_out: schema.linear("bias_out", 512, heads, false)?,
                                heads,
                            }
                        };
                        Ok(Block {
                            attention,
                            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
                            up: schema.linear("up", width, config.inner, true)?,
                            down: schema.linear("down", config.inner, width, true)?,
                            feed_forward_norm: schema.layer_norm(
                                "feed_forward_norm",
                                width,
                                eps,
                            )?,
                            activation: config.activation,
                            shift: if block % 2 == 0 { 0 } else { config.window / 2 },
                        })
                    })
                    .collect::<SingeResult<Vec<_>>>()?;
                Ok(Stage {
                    blocks,
                    convolution: schema.conv2d("convolution", width, width, 3, same(3), true)?,
                    projection: schema.conv2d("projection", width, width, 1, same(1), true)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let features = config.upsampler;
        let shuffles = config.upscale.trailing_zeros();
        Swin2SrLayers {
            first: schema.conv2d("first", config.channels, width, 3, same(3), true)?,
            projection: schema.conv2d("projection", width, width, 1, same(1), true)?,
            projection_norm: schema.layer_norm("projection_norm", width, eps)?,
            stages,
            norm: schema.layer_norm("norm", width, eps)?,
            after_body: schema.conv2d("after_body", width, width, 3, same(3), true)?,
            before_upsample: schema.conv2d("before_upsample", width, features, 3, same(3), true)?,
            upsample: (0..shuffles)
                .map(|index| {
                    schema.conv2d(
                        &format!("upsample.{index}"),
                        features,
                        4 * features,
                        3,
                        same(3),
                        true,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            last: schema.conv2d("last", features, config.channels, 3, same(3), true)?,
            window: config.window,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(ImageUpscaler::new(model, config.window))
}
