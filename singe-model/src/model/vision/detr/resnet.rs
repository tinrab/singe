//! `ResNet`'s bottleneck network, as timm's `resnet50` lays it out and DETR's checkpoints store it: a strided stem and stages of bottleneck blocks, each batch norm frozen at its running statistics.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, BatchNorm, Conv2d, Layer, ModelState},
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

#[derive(Debug, Clone)]
struct ConvNorm {
    convolution: Conv2d,
    norm: BatchNorm,
}

impl ConvNorm {
    fn create(
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
        })
    }
}

impl Layer for ConvNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.convolution, state)?
            .apply(&self.norm, state)
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
        let reduce = ConvNorm::create(schema, ("conv1", "bn1"), input, width, 1, 1)?;
        let spatial = ConvNorm::create(schema, ("conv2", "bn2"), width, width, 3, stride)?;
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
            .apply(&Activation::Relu, state)?
            .apply(&self.spatial, state)?
            .apply(&Activation::Relu, state)?
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
        let stem = ConvNorm::create(schema, ("conv1", "bn1"), 3, config.stem, 7, 2)?;
        let mut input = config.stem;
        let mut stages = Vec::new();
        for (index, &depth) in config.depths.iter().enumerate() {
            let width = config.stem << index;
            let out = width * 4;
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
}

impl Layer for ResNet {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.stem, state)?
            .apply(&Activation::Relu, state)?
            .max_pool2d((3, 3), (2, 2), (1, 1, 1, 1), (1, 1), false)?;
        for block in self.stages.iter().flatten() {
            hidden = hidden.apply(block, state)?;
        }
        Ok(hidden)
    }
}
