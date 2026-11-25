//! Depth Anything: `DINOv2`'s features after the configured blocks, each reassembled into a map at its own scale and fused coarsest first by DPT's neck, then a convolutional head that predicts each pixel's relative or metric depth.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, ConvTranspose2d, ConvTranspose2dConfig, Layer, ModelState},
    shape::spatial::{Spatial2d, Spatial2dExpr},
    tensor::{InterpolateMode, Interpolation},
};

use crate::{
    assembly::{VisionTowerConfig, same_2d, spatial, spatial_dimensions, to_map},
    error::Result,
    format::Loader,
    head::{Boxed, DepthEstimator, HeadLoader},
    model::{ModelEntry, vision::dinov2::Dinov2Layers},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthAnythingConfig {
    pub backbone: VisionTowerConfig,
    /// The blocks, counted from one, whose states the neck reads.
    pub stages: Vec<usize>,
    /// Each stage's channels in the neck, and the factor its map is resized by: up by a transposed convolution above one, down by a strided one below.
    pub neck: Vec<(i64, f64)>,
    pub fusion: i64,
    pub head: i64,
    /// Whether the head ends in a sigmoid, for metric depth, rather than a `ReLU`, for relative depth; either is multiplied by `max_depth`.
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

impl Reassemble {
    fn create(
        schema: &mut Schema<'_>,
        width: i64,
        channels: i64,
        factor: f64,
    ) -> SingeResult<Self> {
        let projection = schema.conv2d("projection", width, channels, 1, same_2d(1, 1), true)?;
        let resize = if factor > 1.0 {
            let factor = factor.round() as i64;
            Resize::Up(schema.conv_transpose2d(
                "resize",
                channels,
                channels,
                factor,
                ConvTranspose2dConfig {
                    stride: Spatial2d::square(factor),
                    ..ConvTranspose2dConfig::default()
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
                same_2d(3, stride),
                true,
            )?)
        } else {
            Resize::Same
        };
        Ok(Self { projection, resize })
    }
}

impl Layer for Reassemble {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let map = input.apply(&self.projection, state)?;
        match &self.resize {
            Resize::Up(layer) => map.apply(layer, state),
            Resize::Same => Ok(map),
            Resize::Down(layer) => map.apply(layer, state),
        }
    }
}

/// A residual unit that activates before each of its two convolutions.
#[derive(Debug, Clone)]
struct Residual {
    first: Conv2d,
    second: Conv2d,
}

impl Residual {
    fn create(schema: &mut Schema<'_>, width: i64) -> SingeResult<Self> {
        Ok(Self {
            first: schema.conv2d("first", width, width, 3, same_2d(3, 1), true)?,
            second: schema.conv2d("second", width, width, 3, same_2d(3, 1), true)?,
        })
    }
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

impl Fusion {
    fn create(schema: &mut Schema<'_>, width: i64) -> SingeResult<Self> {
        Ok(Self {
            residual: Residual::create(&mut schema.scope("residual"), width)?,
            refine: Residual::create(&mut schema.scope("refine"), width)?,
            projection: schema.conv2d("projection", width, width, 1, same_2d(1, 1), true)?,
        })
    }

    /// Adds `skip`, refined, to `hidden`, refines the sum, resizes it to `size`, or to twice its size when `size` is `None`, and projects it.
    fn forward(
        &self,
        hidden: &Tensor,
        skip: Option<&Tensor>,
        size: Option<Spatial2dExpr>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = hidden.clone();
        if let Some(skip) = skip {
            // Each fusion before resizes to the size of the map it adds next, so the sizes differ only in how the program was built.
            let skip = if spatial(skip)? == spatial(&hidden)? {
                skip.clone()
            } else {
                bilinear(skip, spatial_dimensions(&hidden)?, false)?
            };
            hidden = hidden.add(&skip.apply(&self.residual, state)?)?;
        }

        let hidden = hidden.apply(&self.refine, state)?;
        let size = match size {
            Some(size) => size,
            None => spatial_dimensions(&hidden)?.map(|side| side * 2),
        };
        bilinear(&hidden, size, true)?.apply(&self.projection, state)
    }
}

fn bilinear(input: &Tensor, size: Spatial2dExpr, align_corners: bool) -> SingeResult<Tensor> {
    input.interpolate(
        size,
        Interpolation::new(InterpolateMode::Bilinear).with_align_corners(align_corners),
    )
}

/// Depth Anything: from images to each pixel's depth at the size the patches cover, `[B, H, W]`.
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

impl DepthAnythingLayers {
    pub fn create(schema: &mut Schema<'_>, config: &DepthAnythingConfig) -> SingeResult<Self> {
        let width = config.backbone.encoder.width;
        let fusion = config.fusion;
        let backbone = Dinov2Layers::create(&mut schema.scope("backbone"), &config.backbone)?;
        let mut reassemble = Vec::new();
        let mut convolutions = Vec::new();
        let mut fusions = Vec::new();
        for (index, &(channels, factor)) in config.neck.iter().enumerate() {
            reassemble.push(Reassemble::create(
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
                same_2d(3, 1),
                false,
            )?);
            fusions.push(Fusion::create(
                &mut schema.scope(format!("fusions.{index}")),
                fusion,
            )?);
        }

        let head = {
            let mut schema = schema.scope("head");
            [
                schema.conv2d("0", fusion, fusion / 2, 3, same_2d(3, 1), true)?,
                schema.conv2d("1", fusion / 2, config.head, 3, same_2d(3, 1), true)?,
                schema.conv2d("2", config.head, 1, 1, same_2d(1, 1), true)?,
            ]
        };
        Ok(Self {
            backbone,
            stages: config.stages.clone(),
            patch_size: config.backbone.patch_size,
            reassemble,
            convolutions,
            fusions,
            head,
            metric: config.metric,
            max_depth: config.max_depth,
        })
    }
}

impl Layer for DepthAnythingLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let grid = spatial_dimensions(input)?.map(|side| side / self.patch_size);
        let features = self.backbone.features(input, &self.stages, state)?;
        let maps = features
            .iter()
            .zip(&self.reassemble)
            .zip(&self.convolutions)
            .map(|((hidden, reassemble), convolution)| {
                if hidden.shape().rank() != 3 {
                    return Err(SingeError::invalid_argument("backbone features"));
                }

                let patches = hidden.narrow_with(1, 1_i64, hidden.dimension(1)? - 1)?;
                to_map(&patches, grid.clone())?
                    .apply(reassemble, state)?
                    .apply(convolution, state)
            })
            .collect::<SingeResult<Vec<_>>>()?;

        // Fused from the coarsest map, each fusion resized to the next finer map, and the last doubled.
        let mut fused: Option<Tensor> = None;
        for (index, (map, fusion)) in maps.iter().rev().zip(&self.fusions).enumerate() {
            let size = match maps.len().checked_sub(index + 2) {
                Some(next) => Some(spatial_dimensions(&maps[next])?),
                None => None,
            };
            fused = Some(match &fused {
                None => fusion.forward(map, None, size, state)?,
                Some(hidden) => fusion.forward(hidden, Some(map), size, state)?,
            });
        }

        let fused = fused.ok_or_else(|| SingeError::invalid_argument("no backbone features"))?;

        let [first, second, third] = &self.head;
        let depth = bilinear(
            &fused.apply(first, state)?,
            grid.map(|side| side * self.patch_size),
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
    loaders: &[HeadLoader::DepthEstimation(load_estimator)],
};

pub fn load_estimator(loader: &mut Loader<'_>) -> Result<DepthEstimator> {
    let config: DepthAnythingConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DepthAnythingLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(DepthEstimator::new(model))
}
