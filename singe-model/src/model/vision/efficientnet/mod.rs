//! `EfficientNet`: a strided stem, then stages of mobile inverted bottlenecks, each expanding its channels, filtering each channel alone, reweighting the channels by squeeze and excitation, and projecting back, then a 1 × 1 convolution averaged into a classifier.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, BatchNorm, Conv2d, Conv2dConfig, Layer, Linear, ModelState},
    shape::spatial::{Padding, Padding2d, Spatial2d},
};

use crate::{
    error::Result,
    format::Loader,
    head::{Boxed, Classifier, ClassifierConfig, HeadLoader, ImageClassifier},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EfficientNetConfig {
    pub stem: i64,
    pub blocks: Vec<BlockConfig>,
    /// The channels of the last 1 × 1 convolution, which the classifier reads.
    pub last: i64,
    pub squeeze_ratio: f64,
    pub activation: Activation,
    pub eps: f32,
    pub classifier: ClassifierConfig,
}

/// One mobile inverted bottleneck's sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockConfig {
    pub input: i64,
    pub output: i64,
    pub expansion: i64,
    pub kernel: i64,
    pub stride: i64,
    /// Whether the output adds the input, as a stage's repeated blocks do.
    pub residual: bool,
}

/// A convolution after explicit zero padding, then a batch norm and any activation.
#[derive(Debug, Clone)]
struct ConvNorm {
    convolution: Conv2d,
    norm: BatchNorm,
    activation: Option<Activation>,
    /// The zeros before and after each spatial axis.
    padding: Padding,
}

impl ConvNorm {
    /// Registers `convolution` and `norm`, padding a stride-one convolution by half its kernel on each side, and a strided one by one less before than after, as `correct_pad` does.
    fn create(
        schema: &mut Schema<'_>,
        [input, out, kernel, stride, groups]: [i64; 5],
        eps: f32,
        activation: Option<Activation>,
    ) -> SingeResult<Self> {
        let half = kernel / 2;
        let padding = if stride == 1 {
            Padding::new(half, half)
        } else {
            Padding::new(half - 1, half)
        };
        Ok(Self {
            convolution: schema.conv2d(
                "convolution",
                input,
                out,
                kernel,
                Conv2dConfig {
                    stride: Spatial2d::square(stride),
                    padding: Padding2d::symmetric(0, 0),
                    dilation: Spatial2d::square(1),
                    groups,
                },
                false,
            )?,
            norm: schema.batch_norm("norm", out, eps)?,
            activation,
            padding,
        })
    }
}

impl Layer for ConvNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let padding = self.padding;
        let padded = if padding == Padding::ZERO {
            input.clone()
        } else {
            input.pad(&[Padding::ZERO, Padding::ZERO, padding, padding], 0.0)?
        };
        let normed = padded
            .apply(&self.convolution, state)?
            .apply(&self.norm, state)?;
        match &self.activation {
            Some(activation) => normed.apply(activation, state),
            None => Ok(normed),
        }
    }
}

/// A mobile inverted bottleneck, as `EfficientNetBlock` computes it.
#[derive(Debug, Clone)]
struct Block {
    expand: Option<ConvNorm>,
    depthwise: ConvNorm,
    squeeze: Conv2d,
    excite: Conv2d,
    project: ConvNorm,
    activation: Activation,
    residual: bool,
}

impl Block {
    fn create(
        schema: &mut Schema<'_>,
        config: &EfficientNetConfig,
        block: BlockConfig,
    ) -> SingeResult<Self> {
        let activation = Some(config.activation);
        let expanded = block.input * block.expansion;
        // The squeeze narrows to a share of the block's input, not of its expanded channels.
        let squeezed = ((block.input as f64 * config.squeeze_ratio) as i64).max(1);
        let pointwise = Conv2dConfig::default();
        Ok(Self {
            expand: (block.expansion != 1)
                .then(|| {
                    ConvNorm::create(
                        &mut schema.scope("expand"),
                        [block.input, expanded, 1, 1, 1],
                        config.eps,
                        activation,
                    )
                })
                .transpose()?,
            depthwise: ConvNorm::create(
                &mut schema.scope("depthwise"),
                [expanded, expanded, block.kernel, block.stride, expanded],
                config.eps,
                activation,
            )?,
            squeeze: schema.conv2d("squeeze", expanded, squeezed, 1, pointwise, true)?,
            excite: schema.conv2d("excite", squeezed, expanded, 1, pointwise, true)?,
            project: ConvNorm::create(
                &mut schema.scope("project"),
                [expanded, block.output, 1, 1, 1],
                config.eps,
                None,
            )?,
            activation: config.activation,
            residual: block.residual,
        })
    }
}

impl Layer for Block {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let expanded = match &self.expand {
            Some(expand) => input.apply(expand, state)?,
            None => input.clone(),
        };
        let filtered = expanded.apply(&self.depthwise, state)?;
        let weights = filtered
            .mean(&[2, 3], true)?
            .apply(&self.squeeze, state)?
            .apply(&self.activation, state)?
            .apply(&self.excite, state)?
            .apply(&Activation::Sigmoid, state)?;
        let output = filtered.mul(&weights)?.apply(&self.project, state)?;
        if self.residual {
            output.add(input)
        } else {
            Ok(output)
        }
    }
}

/// `EfficientNet` and its classifier: from images to each label's score, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct EfficientNetLayers {
    stem: ConvNorm,
    blocks: Vec<Block>,
    top: ConvNorm,
    classifier: Linear,
}

impl EfficientNetLayers {
    pub fn create(schema: &mut Schema<'_>, config: &EfficientNetConfig) -> SingeResult<Self> {
        let last_block = config
            .blocks
            .last()
            .map_or(config.stem, |block| block.output);
        // The stem pads one row and column after, which a strided 3 × 3 kernel's one less before than after gives.
        let stem = ConvNorm::create(
            &mut schema.scope("stem"),
            [3, config.stem, 3, 2, 1],
            config.eps,
            Some(config.activation),
        )?;
        let blocks = config
            .blocks
            .iter()
            .enumerate()
            .map(|(index, &block)| {
                Block::create(&mut schema.scope(format!("blocks.{index}")), config, block)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            stem,
            blocks,
            top: ConvNorm::create(
                &mut schema.scope("top"),
                [last_block, config.last, 1, 1, 1],
                config.eps,
                Some(config.activation),
            )?,
            classifier: schema.linear(
                "classifier",
                config.last,
                config.classifier.label_count()?,
                true,
            )?,
        })
    }
}

impl Layer for EfficientNetLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.apply(&self.stem, state)?;
        for block in &self.blocks {
            hidden = hidden.apply(block, state)?;
        }
        // `AvgPool2d(1280, ceil_mode=True)` over a smaller map is the mean of the whole map.
        hidden
            .apply(&self.top, state)?
            .mean(&[2, 3], false)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "efficientnet",
    formats: &[(Format::Transformers, &["efficientnet"])],
    loaders: &[HeadLoader::ImageClassification(load_classifier)],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<ImageClassifier> {
    let config: EfficientNetConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        EfficientNetLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ImageClassifier(Classifier::new(model, config.classifier)))
}
