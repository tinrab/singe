// Depth Anything: DINOv2's features after four blocks, reassembled into maps at four scales and fused coarsest first by DPT's neck, then a convolutional head that predicts each pixel's relative depth.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTransposeConfig, Layer, Model,
        ModelState, StateBuilder,
    },
    tensor::{InterpolateMode, Interpolation},
};

use crate::{
    error::Result,
    format::Loader,
    head::{DepthEstimator, HeadLoader, PixelLayers},
    model::{
        ModelEntry,
        vision::dinov2::{Dinov2Config, Dinov2Layers},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthAnythingConfig {
    pub backbone: Dinov2Config,
    // The blocks, counted from one, whose states the neck reads.
    pub stages: Vec<usize>,
    // Each stage's channels in the neck, and the factor its map is resized by: up by a transposed convolution above one, down by a strided one below.
    pub neck: Vec<(i64, f64)>,
    pub fusion: i64,
    pub head: i64,
    // Whether the head ends in a sigmoid, for metric depth, rather than a ReLU, for relative depth; either is scaled to `max_depth`.
    pub metric: bool,
    pub max_depth: f64,
}

#[derive(Debug, Clone)]
enum Resize {
    Up(ConvTranspose2d),
    Same,
    Down(Conv2d),
}

#[derive(Debug, Clone)]
struct Reassemble {
    projection: Conv2d,
    resize: Resize,
}

// A residual unit that activates before each of its two convolutions.
#[derive(Debug, Clone)]
struct Residual {
    first: Conv2d,
    second: Conv2d,
}

impl Layer for Residual {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&Activation::Relu, state)?
            .apply(&self.first, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.second, state)?
            .add(input)
    }
}

#[derive(Debug, Clone)]
struct Fusion {
    residual: Residual,
    refine: Residual,
    projection: Conv2d,
}

fn bilinear(input: &Tensor, size: (i64, i64), align_corners: bool) -> SingeResult<Tensor> {
    input.interpolate(
        size,
        Interpolation::new(InterpolateMode::Bilinear).with_align_corners(align_corners),
    )
}

fn spatial(tensor: &Tensor) -> (i64, i64) {
    let dimensions = tensor.shape().dimensions();
    (dimensions[2], dimensions[3])
}

impl Fusion {
    // Adds `skip`, refined, to `hidden`, refines the sum, and resizes it to `size`, or doubles it.
    fn forward(
        &self,
        hidden: &Tensor,
        skip: Option<&Tensor>,
        size: Option<(i64, i64)>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = hidden.clone();
        if let Some(skip) = skip {
            let skip = if spatial(skip) == spatial(&hidden) {
                skip.clone()
            } else {
                bilinear(skip, spatial(&hidden), false)?
            };
            hidden = hidden.add(&skip.apply(&self.residual, state)?)?;
        }

        let hidden = hidden.apply(&self.refine, state)?;
        let (height, width) = spatial(&hidden);
        let size = size.unwrap_or((height * 2, width * 2));
        bilinear(&hidden, size, true)?.apply(&self.projection, state)
    }
}

// Depth Anything: from images to each pixel's depth at the size the patches cover, `[B, H, W]`.
#[derive(Debug, Clone)]
pub struct DepthAnythingLayers {
    backbone: Dinov2Layers,
    stages: Vec<usize>,
    patch_size: i64,
    reassemble: Vec<Reassemble>,
    convolutions: Vec<Conv2d>,
    fusions: Vec<Fusion>,
    head: [Conv2d; 3],
    metric: bool,
    max_depth: f64,
}

impl Layer for DepthAnythingLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let (height, width) = spatial(input);
        let (rows, columns) = (height / self.patch_size, width / self.patch_size);
        let features = self.backbone.features(input, &self.stages, state)?;
        let maps = features
            .iter()
            .zip(&self.reassemble)
            .zip(&self.convolutions)
            .map(|((hidden, reassemble), convolution)| {
                let &[batch, tokens, channels] = hidden.shape().dimensions() else {
                    return Err(SingeError::invalid_argument("backbone features"));
                };
                let map = hidden
                    .narrow(1, 1, tokens - 1)?
                    .reshape([batch, rows, columns, channels])?
                    .permute(&[0, 3, 1, 2])?
                    .apply(&reassemble.projection, state)?;
                let map = match &reassemble.resize {
                    Resize::Up(layer) => map.apply(layer, state)?,
                    Resize::Same => map,
                    Resize::Down(layer) => map.apply(layer, state)?,
                };
                map.apply(convolution, state)
            })
            .collect::<SingeResult<Vec<_>>>()?;

        // Fused from the coarsest map, each fusion resized to the next finer map, and the last doubled.
        let mut fused: Option<Tensor> = None;
        for (index, (map, fusion)) in maps.iter().rev().zip(&self.fusions).enumerate() {
            let size = maps
                .len()
                .checked_sub(index + 2)
                .map(|next| spatial(&maps[next]));
            fused = Some(match &fused {
                None => fusion.forward(map, None, size, state)?,
                Some(hidden) => fusion.forward(hidden, Some(map), size, state)?,
            });
        }

        let fused = fused.ok_or_else(|| SingeError::invalid_argument("no backbone features"))?;

        let [first, second, third] = &self.head;
        let depth = bilinear(
            &fused.apply(first, state)?,
            (rows * self.patch_size, columns * self.patch_size),
            true,
        )?
        .apply(second, state)?
        .apply(&Activation::Relu, state)?
        .apply(third, state)?;
        let activation = if self.metric {
            Activation::Sigmoid
        } else {
            Activation::Relu
        };
        depth
            .apply(&activation, state)?
            .mul_scalar(self.max_depth)?
            .squeeze(1)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "depth_anything",
    formats: &[(Format::Transformers, &["depth_anything"])],
    heads: &[HeadLoader::DepthEstimation(load_estimator)],
};

fn padded(padding: i64) -> Conv2dConfig {
    Conv2dConfig {
        padding: (padding, padding, padding, padding),
        ..Conv2dConfig::default()
    }
}

fn residual(schema: &mut Schema<'_>, width: i64) -> SingeResult<Residual> {
    Ok(Residual {
        first: schema.conv2d("first", width, width, 3, padded(1), true)?,
        second: schema.conv2d("second", width, width, 3, padded(1), true)?,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "resize factors are small whole numbers or their inverses"
)]
fn reassemble(
    schema: &mut Schema<'_>,
    width: i64,
    channels: i64,
    factor: f64,
) -> SingeResult<Reassemble> {
    let projection = schema.conv2d("projection", width, channels, 1, padded(0), true)?;
    let resize = if factor > 1.0 {
        let factor = factor.round() as i64;
        Resize::Up(schema.conv_transpose2d(
            "resize",
            channels,
            channels,
            factor,
            ConvTransposeConfig {
                stride: [factor, factor],
                ..ConvTransposeConfig::default()
            },
            true,
        )?)
    } else if factor < 1.0 {
        let stride = (1.0 / factor).round() as i64;
        Resize::Down(schema.conv2d(
            "resize",
            channels,
            channels,
            3,
            Conv2dConfig {
                stride: (stride, stride),
                padding: (1, 1, 1, 1),
                ..Conv2dConfig::default()
            },
            true,
        )?)
    } else {
        Resize::Same
    };
    Ok(Reassemble { projection, resize })
}

pub fn load_estimator(loader: &mut Loader<'_>) -> Result<DepthEstimator> {
    let config: DepthAnythingConfig = loader.config(transformers::config)?;
    let width = config.backbone.width;
    let fusion = config.fusion;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let backbone = Dinov2Layers::create(&mut schema.scope("backbone"), &config.backbone)?;
        let mut reassembled = Vec::new();
        let mut convolutions = Vec::new();
        let mut fusions = Vec::new();
        for (index, &(channels, factor)) in config.neck.iter().enumerate() {
            reassembled.push(reassemble(
                &mut schema.scope(format!("reassemble.{index}")),
                width,
                channels,
                factor,
            )?);
            convolutions.push(schema.conv2d(
                &format!("convolutions.{index}"),
                channels,
                fusion,
                3,
                padded(1),
                false,
            )?);
            let mut scope = schema.scope(format!("fusions.{index}"));
            fusions.push(Fusion {
                residual: residual(&mut scope.scope("residual"), fusion)?,
                refine: residual(&mut scope.scope("refine"), fusion)?,
                projection: scope.conv2d("projection", fusion, fusion, 1, padded(0), true)?,
            });
        }

        let head = {
            let mut schema = schema.scope("head");
            [
                schema.conv2d("0", fusion, fusion / 2, 3, padded(1), true)?,
                schema.conv2d("1", fusion / 2, config.head, 3, padded(1), true)?,
                schema.conv2d("2", config.head, 1, 1, padded(0), true)?,
            ]
        };
        DepthAnythingLayers {
            backbone,
            stages: config.stages.clone(),
            patch_size: config.backbone.patch_size,
            reassemble: reassembled,
            convolutions,
            fusions,
            head,
            metric: config.metric,
            max_depth: config.max_depth,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(DepthEstimator::new(model))
}
