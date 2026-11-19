// Swin2SR: a convolution into features, stages of Swin V2 blocks attending within windows, shifted every other block, by scaled cosine similarity and a learned bias of relative positions, then a convolution and pixel shuffles up to a larger image.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
};

use crate::{
    assembly::{
        merge_heads, merge_windows, partition_windows, relative_index, same_2d, shift_mask,
        spatial, split_heads, to_map, to_tokens, unit_vectors,
    },
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, ImageUpscaler},
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

// A window's side, the log-spaced relative coordinates the position bias network reads, `[(2w - 1)², 2]`, and each pair of a window's points' index among them, `[w⁴]`, as `create_coords_table_and_index` makes them.
#[derive(Debug, Clone)]
struct Windows {
    window: i64,
    coordinates: Vec<f32>,
    index: Vec<i64>,
}

impl Windows {
    fn create(window: i64) -> Self {
        let span = window - 1;
        let scaled = |offset: i64| {
            let value = offset as f32 / span as f32 * 8.0;
            value.signum() * (value.abs() + 1.0).log2() / 8_f32.log2()
        };
        let coordinates = (-span..=span)
            .flat_map(|row| (-span..=span).flat_map(move |column| [scaled(row), scaled(column)]))
            .collect();
        Self {
            window,
            coordinates,
            index: relative_index(window),
        }
    }
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

impl Attention {
    fn create(schema: &mut Schema<'_>, width: i64, heads: i64) -> SingeResult<Self> {
        let attention = Self {
            query: schema.linear("query", width, width, true)?,
            key: schema.linear("key", width, width, false)?,
            value: schema.linear("value", width, width, true)?,
            output: schema.linear("output", width, width, true)?,
            logit_scale: schema.parameter("logit_scale", [heads, 1, 1])?,
            bias_hidden: schema.linear("bias_hidden", 2, 512, true)?,
            bias_out: schema.linear("bias_out", 512, heads, false)?,
            heads,
        };
        schema.join_rows(&[&attention.query, &attention.key, &attention.value])?;
        Ok(attention)
    }

    fn forward(
        &self,
        windows: &Tensor,
        geometry: &Windows,
        mask: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[count, area, _] = windows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention windows"));
        };
        let heads = self.heads;
        let device = windows.device();
        let query = unit_vectors(
            &split_heads(&windows.apply(&self.query, state)?, heads)?,
            -1,
        )?;
        let key = unit_vectors(&split_heads(&windows.apply(&self.key, state)?, heads)?, -1)?;
        let value = split_heads(&windows.apply(&self.value, state)?, heads)?;
        let scale = state
            .tensor(&self.logit_scale)?
            .clamp(f32::MIN, 100_f32.ln())?
            .exp()?
            .unsqueeze(0)?;
        let mut scores = query.matmul(&key.transpose(2, 3)?)?.mul(&scale)?;

        // The bias of each pair of points' relative position, from a small network over log-spaced coordinates.
        let window = geometry.window;
        let entries = (2 * window - 1) * (2 * window - 1);
        let coordinates = Tensor::from_vec(
            device,
            Shape::create([entries, 2])?,
            geometry.coordinates.clone(),
        )?
        .cast(windows.data_type())?;
        let biases = coordinates
            .apply(&self.bias_hidden, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.bias_out, state)?;
        let index = Tensor::from_vec(
            device,
            Shape::create([area * area, 1])?,
            geometry.index.clone(),
        )?
        .expand(Shape::create([area * area, heads])?)?;
        let bias = biases
            .gather(0, &index)?
            .reshape([area, area, heads])?
            .permute(&[2, 0, 1])?
            .apply(&Activation::Sigmoid, state)?
            .mul_scalar(16.0)?
            .unsqueeze(0)?;
        scores = scores.add(&bias)?;
        if let Some(mask) = mask {
            let &[_, windows_per_image, ..] = mask.shape().dimensions() else {
                return Err(SingeError::invalid_argument("shifted window mask"));
            };
            scores = scores
                .reshape([
                    count / windows_per_image,
                    windows_per_image,
                    heads,
                    area,
                    area,
                ])?
                .add(mask)?
                .reshape([count, heads, area, area])?;
        }

        merge_heads(&scores.softmax(-1)?.matmul(&value)?)?.apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
    shift: i64,
}

impl Block {
    fn create(
        schema: &mut Schema<'_>,
        config: &Swin2SrConfig,
        heads: i64,
        shift: i64,
    ) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention: Attention::create(&mut schema.scope("attention"), width, heads)?,
            attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
            shift,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        (height, width): (i64, i64),
        geometry: &Windows,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let window = geometry.window;
        let &[batch, _, channels] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("Swin2SR block input"));
        };
        let shifted = hidden
            .reshape([batch, height, width, channels])?
            .roll(&[-self.shift, -self.shift], &[1, 2])?;
        let mask = if self.shift > 0 {
            // -100 twice, as `Swin2SRSelfAttention` adds the mask twice.
            let windows_per_image = (height / window) * (width / window);
            let area = window * window;
            Some(
                Tensor::from_vec(
                    hidden.device(),
                    Shape::create([1, windows_per_image, 1, area, area])?,
                    shift_mask(height, width, window, self.shift, -200.0),
                )?
                .cast(hidden.data_type())?,
            )
        } else {
            None
        };
        let attended = self.attention.forward(
            &partition_windows(&shifted, window)?,
            geometry,
            mask.as_ref(),
            state,
        )?;
        let attended = merge_windows(&attended, window, (batch, height, width))?
            .roll(&[self.shift, self.shift], &[1, 2])?
            .reshape([batch, height * width, channels])?;

        let hidden = attended.apply(&self.attention_norm, state)?.add(hidden)?;
        hidden
            .apply(&self.feed_forward, state)?
            .apply(&self.feed_forward_norm, state)?
            .add(&hidden)
    }
}

#[derive(Debug, Clone)]
struct Stage {
    blocks: Vec<Block>,
    convolution: Conv2d,
    projection: Conv2d,
}

impl Stage {
    fn create(
        schema: &mut Schema<'_>,
        config: &Swin2SrConfig,
        (depth, heads): (i64, i64),
    ) -> SingeResult<Self> {
        let width = config.width;
        let blocks = (0..depth)
            .map(|block| {
                let shift = if block % 2 == 0 { 0 } else { config.window / 2 };
                Block::create(
                    &mut schema.scope(format!("blocks.{block}")),
                    config,
                    heads,
                    shift,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            blocks,
            convolution: schema.conv2d("convolution", width, width, 3, same_2d(3, 1), true)?,
            projection: schema.conv2d("projection", width, width, 1, same_2d(1, 1), true)?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        size: (i64, i64),
        geometry: &Windows,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, size, geometry, state)?;
        }
        to_tokens(
            &to_map(&hidden, size)?
                .apply(&self.convolution, state)?
                .apply(&self.projection, state)?,
        )?
        .add(input)
    }
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
    windows: Windows,
}

impl Swin2SrLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Swin2SrConfig) -> SingeResult<Self> {
        let width = config.width;
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(index, &stage)| {
                Stage::create(&mut schema.scope(format!("stages.{index}")), config, stage)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let features = config.upsampler;
        let shuffles = config.upscale.trailing_zeros();
        let upsample = (0..shuffles)
            .map(|index| {
                schema.conv2d(
                    &format!("upsample.{index}"),
                    features,
                    4 * features,
                    3,
                    same_2d(3, 1),
                    true,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            first: schema.conv2d("first", config.channels, width, 3, same_2d(3, 1), true)?,
            projection: schema.conv2d("projection", width, width, 1, same_2d(1, 1), true)?,
            projection_norm: schema.layer_norm("projection_norm", width, config.eps)?,
            stages,
            norm: schema.layer_norm("norm", width, config.eps)?,
            after_body: schema.conv2d("after_body", width, width, 3, same_2d(3, 1), true)?,
            before_upsample: schema.conv2d(
                "before_upsample",
                width,
                features,
                3,
                same_2d(3, 1),
                true,
            )?,
            upsample,
            last: schema.conv2d("last", features, config.channels, 3, same_2d(3, 1), true)?,
            windows: Windows::create(config.window),
        })
    }
}

impl Layer for Swin2SrLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let window = self.windows.window;
        let size = spatial(input)?;
        if size.0 % window != 0 || size.1 % window != 0 {
            return Err(SingeError::invalid_argument(
                "an image whose sides are not whole windows",
            ));
        }

        let mean = mean(input.device())?.cast(input.data_type())?;
        let features = input.sub(&mean)?.apply(&self.first, state)?;
        let mut hidden = to_tokens(&features.apply(&self.projection, state)?)?
            .apply(&self.projection_norm, state)?;
        for stage in &self.stages {
            hidden = stage.forward(&hidden, size, &self.windows, state)?;
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

fn mean(device: Device) -> SingeResult<Tensor> {
    Tensor::from_vec(device, Shape::create([1, 3, 1, 1])?, MEAN.to_vec())
}

// `[B, C · r², H, W]` as `[B, C, H · r, W · r]`, as `nn.PixelShuffle(r)` rearranges it.
fn pixel_shuffle(input: &Tensor, factor: i64) -> SingeResult<Tensor> {
    let &[batch, channels, height, width] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument("pixel shuffle input"));
    };
    let channels = channels / (factor * factor);
    input
        .reshape([batch, channels, factor, factor, height, width])?
        .permute(&[0, 1, 4, 2, 5, 3])?
        .reshape([batch, channels, height * factor, width * factor])
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "swin2sr",
    formats: &[(Format::Transformers, &["swin2sr"])],
    loaders: &[HeadLoader::SuperResolution(load_upscaler)],
};

pub fn load_upscaler(loader: &mut Loader<'_>) -> Result<ImageUpscaler> {
    let config: Swin2SrConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Swin2SrLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ImageUpscaler::new(model, config.window))
}
