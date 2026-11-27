//! `ResNet`'s bottleneck networks as the DETR family's checkpoints store them, each batch norm frozen at its running statistics: timm's `resnet50`, a strided stem and stages of bottleneck blocks, and `ResNet`-vd, as RT-DETR's backbone has it.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, BatchNorm, Conv2d, Layer, ModelState},
    shape::spatial::Padding2d,
};

use crate::{
    assembly::{PYTORCH_NORM_EPS, same_2d},
    schema::Schema,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResNetConfig {
    /// The number of blocks in each stage.
    pub depths: Vec<i64>,
    pub stem: i64,
}

impl ResNetConfig {
    /// Returns the channels of stage `index`'s map, `4 · stem · 2^index`.
    #[must_use]
    pub const fn channels(&self, index: usize) -> i64 {
        (self.stem * 4) << index
    }
}

/// A convolution without bias, padded to keep the size at stride one, then a batch norm and any activation.
#[derive(Debug, Clone)]
pub struct ConvNorm {
    convolution: Conv2d,
    norm: BatchNorm,
    activation: Option<Activation>,
}

impl ConvNorm {
    /// Registers the convolution and the norm under their names.
    pub fn create(
        schema: &mut Schema<'_>,
        (convolution, norm): (&str, &str),
        input: i64,
        out: i64,
        kernel: i64,
        stride: i64,
    ) -> SingeResult<Self> {
        Ok(Self {
            convolution: schema.conv2d(
                convolution,
                input,
                out,
                kernel,
                same_2d(kernel, stride),
                false,
            )?,
            norm: schema.batch_norm(norm, out, PYTORCH_NORM_EPS)?,
            activation: None,
        })
    }

    #[must_use]
    pub const fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }
}

impl Layer for ConvNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let normed = input
            .apply(&self.convolution, state)?
            .apply(&self.norm, state)?;
        match &self.activation {
            Some(activation) => normed.apply(activation, state),
            None => Ok(normed),
        }
    }
}

/// A bottleneck: a 1 × 1 reduction, a 3 × 3 convolution carrying the stride, and a 1 × 1 expansion by four, added to the input, or to its projection when the shape or stride changes.
#[derive(Debug, Clone)]
struct Bottleneck {
    reduce: ConvNorm,
    spatial: ConvNorm,
    expand: ConvNorm,
    shortcut: Option<ConvNorm>,
}

impl Bottleneck {
    fn create(schema: &mut Schema<'_>, input: i64, width: i64, stride: i64) -> SingeResult<Self> {
        let out = width * 4;
        let reduce = ConvNorm::create(schema, ("conv1", "bn1"), input, width, 1, 1)?
            .with_activation(Activation::Relu);
        let spatial = ConvNorm::create(schema, ("conv2", "bn2"), width, width, 3, stride)?
            .with_activation(Activation::Relu);
        let expand = ConvNorm::create(schema, ("conv3", "bn3"), width, out, 1, 1)?;
        let shortcut = (input != out || stride != 1)
            .then(|| {
                ConvNorm::create(
                    schema,
                    ("downsample.0", "downsample.1"),
                    input,
                    out,
                    1,
                    stride,
                )
            })
            .transpose()?;
        Ok(Self {
            reduce,
            spatial,
            expand,
            shortcut,
        })
    }
}

impl Layer for Bottleneck {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = input
            .apply(&self.reduce, state)?
            .apply(&self.spatial, state)?
            .apply(&self.expand, state)?;
        let shortcut = match &self.shortcut {
            Some(shortcut) => input.apply(shortcut, state)?,
            None => input.clone(),
        };
        hidden.add(&shortcut)?.apply(&Activation::Relu, state)
    }
}

/// A network from images to the last stage's map, `[B, 32 · stem, H / 32, W / 32]` when it has four stages.
#[derive(Debug, Clone)]
pub struct ResNet {
    stem: ConvNorm,
    stages: Vec<Vec<Bottleneck>>,
}

impl ResNet {
    /// Registers the network under timm's names: `conv1`, `bn1`, and `layer{N}.{block}` with `conv1`-`conv3`, `bn1`-`bn3`, and `downsample.0` and `.1`.
    pub fn create(schema: &mut Schema<'_>, config: &ResNetConfig) -> SingeResult<Self> {
        let stem = ConvNorm::create(schema, ("conv1", "bn1"), 3, config.stem, 7, 2)?
            .with_activation(Activation::Relu);
        let mut input = config.stem;
        let mut stages = Vec::new();
        for (index, &depth) in config.depths.iter().enumerate() {
            let width = config.stem << index;
            let out = config.channels(index);
            let blocks = (0..depth)
                .map(|block| {
                    let mut schema = schema.scope(format!("layer{}.{block}", index + 1));
                    let stride = if block == 0 && index > 0 { 2 } else { 1 };
                    let block_input = if block == 0 { input } else { out };
                    Bottleneck::create(&mut schema, block_input, width, stride)
                })
                .collect::<SingeResult<Vec<_>>>()?;
            stages.push(blocks);
            input = out;
        }
        Ok(Self { stem, stages })
    }

    /// Returns each stage's map, the first at a quarter of the image's size and each after it at half the one before.
    pub fn stages(&self, input: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let mut hidden = input.apply(&self.stem, state)?.max_pool2d(
            (3, 3),
            (2, 2),
            Padding2d::symmetric(1, 1),
            (1, 1),
            false,
        )?;
        let mut maps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                hidden = hidden.apply(block, state)?;
            }

            maps.push(hidden.clone());
        }
        Ok(maps)
    }
}

impl Layer for ResNet {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.stages(input, state)?
            .pop()
            .ok_or_else(|| SingeError::invalid_argument("a ResNet without stages"))
    }
}

/// A `ResNet`-vd bottleneck: a 1 × 1 reduction, a 3 × 3 convolution carrying the stride, and a 1 × 1 expansion, added to the input, to its projection when the channels change, or to its projection after 2 × 2 average pooling when the block strides.
#[derive(Debug, Clone)]
struct VdBottleneck {
    reduce: ConvNorm,
    spatial: ConvNorm,
    expand: ConvNorm,
    shortcut: Option<ConvNorm>,
    pools: bool,
}

impl VdBottleneck {
    fn create(schema: &mut Schema<'_>, input: i64, out: i64, stride: i64) -> SingeResult<Self> {
        let width = out / 4;
        let conv = |schema: &mut Schema<'_>, scope: &str, input, out, kernel, stride| {
            let names = ("convolution", "norm");
            ConvNorm::create(&mut schema.scope(scope), names, input, out, kernel, stride)
        };
        let reduce = conv(schema, "reduce", input, width, 1, 1)?.with_activation(Activation::Relu);
        let spatial =
            conv(schema, "spatial", width, width, 3, stride)?.with_activation(Activation::Relu);
        let expand = conv(schema, "expand", width, out, 1, 1)?;
        let shortcut = (input != out || stride != 1)
            .then(|| conv(schema, "shortcut", input, out, 1, 1))
            .transpose()?;
        Ok(Self {
            reduce,
            spatial,
            expand,
            shortcut,
            pools: stride != 1,
        })
    }
}

impl Layer for VdBottleneck {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = input
            .apply(&self.reduce, state)?
            .apply(&self.spatial, state)?
            .apply(&self.expand, state)?;
        let pooled = if self.pools {
            input.avg_pool2d(
                (2, 2),
                (2, 2),
                Padding2d::symmetric(0, 0),
                (1, 1),
                true,
                true,
            )?
        } else {
            input.clone()
        };
        let shortcut = match &self.shortcut {
            Some(shortcut) => pooled.apply(shortcut, state)?,
            None => pooled,
        };
        hidden.add(&shortcut)?.apply(&Activation::Relu, state)
    }
}

/// `ResNet`-vd: a stem of three 3 × 3 convolutions, the first strided, and a strided max pool, then stages of [`VdBottleneck`]s, each but the first strided in its first block.
#[derive(Debug, Clone)]
pub struct ResNetVd {
    stem: Vec<ConvNorm>,
    stages: Vec<Vec<VdBottleneck>>,
}

impl ResNetVd {
    /// Registers the network as `stem.{N}` and `stages.{stage}.{block}`, whose blocks hold `reduce`, `spatial`, `expand`, and any `shortcut`, each a `convolution` and a `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &ResNetConfig) -> SingeResult<Self> {
        let half = config.stem / 2;
        let stem = [(3, half, 2), (half, half, 1), (half, config.stem, 1)]
            .into_iter()
            .enumerate()
            .map(|(index, (input, out, stride))| {
                Ok(ConvNorm::create(
                    &mut schema.scope(format!("stem.{index}")),
                    ("convolution", "norm"),
                    input,
                    out,
                    3,
                    stride,
                )?
                .with_activation(Activation::Relu))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut input = config.stem;
        let mut stages = Vec::new();
        for (index, &depth) in config.depths.iter().enumerate() {
            let out = config.channels(index);
            let blocks = (0..depth)
                .map(|block| {
                    let mut schema = schema.scope(format!("stages.{index}.{block}"));
                    let stride = if block == 0 && index > 0 { 2 } else { 1 };
                    let block_input = if block == 0 { input } else { out };
                    VdBottleneck::create(&mut schema, block_input, out, stride)
                })
                .collect::<SingeResult<Vec<_>>>()?;
            stages.push(blocks);
            input = out;
        }
        Ok(Self { stem, stages })
    }

    /// Returns each stage's map, the first at a quarter of the image's size and each after it at half the one before.
    pub fn stages(&self, input: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let mut hidden = input.clone();
        for layer in &self.stem {
            hidden = hidden.apply(layer, state)?;
        }

        hidden = hidden.max_pool2d((3, 3), (2, 2), Padding2d::symmetric(1, 1), (1, 1), false)?;
        let mut maps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                hidden = hidden.apply(block, state)?;
            }

            maps.push(hidden.clone());
        }
        Ok(maps)
    }
}
