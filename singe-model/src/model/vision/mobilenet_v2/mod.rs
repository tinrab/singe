//! `MobileNetV2`: a strided stem, then inverted residuals, each expanding its channels, filtering each channel alone, and projecting back, with every convolution padded as TensorFlow's "SAME" pads it.
//! The classifier averages a last 1 × 1 convolution's map; `DeepLabV3` keeps the map at an eighth of the image by dilating the later blocks instead of striding, then reads it beside its average.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, BatchNorm, Conv2d, Conv2dConfig, Layer, Linear, ModelState},
    shape::spatial::{Padding, Padding2d, Spatial2d},
};

use crate::{
    error::Result,
    format::Loader,
    head::{
        Boxed, Classifier, ClassifierConfig, HeadLoader, ImageClassifier, ImageSegmenter,
        SegmenterLayout,
    },
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The epsilon of `DeepLabV3`'s batch norms, which do not read the config's.
const HEAD_NORM_EPS: f32 = 1e-5;

/// The channels of `DeepLabV3`'s branches.
const HEAD_CHANNELS: i64 = 256;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MobileNetV2Config {
    /// The stem's output channels, then each inverted residual's.
    pub channels: Vec<i64>,
    /// Each inverted residual's stride, before any is traded for dilation.
    pub strides: Vec<i64>,
    /// The channels of the stem's first convolution.
    pub stem: i64,
    /// Whether the first block expands the stem's channels with a 1 × 1 convolution, rather than taking the stem's convolution as its expansion.
    pub first_expands: bool,
    /// The channels of the last 1 × 1 convolution, which the classifier reads.
    pub last: i64,
    pub expansion: f64,
    pub divisor: i64,
    pub min_depth: i64,
    /// The largest stride of the blocks' map, past which blocks dilate instead.
    pub output_stride: i64,
    pub activation: Activation,
    pub eps: f32,
    pub classifier: ClassifierConfig,
}

/// Rounds `value` to the nearest multiple of `divisor`, at least `min_value`, and up a multiple when rounding loses more than a tenth, as `make_divisible` does.
pub(crate) fn make_divisible(value: f64, divisor: i64, min_value: i64) -> i64 {
    let rounded = ((value + divisor as f64 / 2.0) as i64 / divisor * divisor).max(min_value);
    if (rounded as f64) < 0.9 * value {
        rounded + divisor
    } else {
        rounded
    }
}

/// A convolution padded as TensorFlow's "SAME" pads it, from each input's size, then any batch norm and activation, as `MobileNetV2ConvLayer` computes it with `tf_padding`.
#[derive(Debug, Clone)]
struct TfConv {
    convolution: Conv2d,
    norm: Option<BatchNorm>,
    activation: Option<Activation>,
    kernel: i64,
    stride: i64,
    dilation: i64,
}

impl TfConv {
    /// Registers `convolution` and, with `norm_eps`, `norm`.
    fn create(
        schema: &mut Schema<'_>,
        [input, out, kernel, stride, dilation, groups]: [i64; 6],
        norm_eps: Option<f32>,
        activation: Option<Activation>,
    ) -> SingeResult<Self> {
        Ok(Self {
            convolution: schema.conv2d(
                "convolution",
                input,
                out,
                kernel,
                Conv2dConfig {
                    stride: Spatial2d::square(stride),
                    padding: Padding2d::symmetric(0, 0),
                    dilation: Spatial2d::square(dilation),
                    groups,
                },
                norm_eps.is_none(),
            )?,
            norm: norm_eps
                .map(|eps| schema.batch_norm("norm", out, eps))
                .transpose()?,
            activation,
            kernel,
            stride,
            dilation,
        })
    }

    /// Returns the zeros before and after an axis of `extent`: what TensorFlow pads for the undilated kernel, the odd one after, times the dilation.
    fn padding(&self, extent: i64) -> Padding {
        let remainder = extent % self.stride;
        let total = if remainder == 0 {
            (self.kernel - self.stride).max(0)
        } else {
            (self.kernel - remainder).max(0)
        };
        let before = total / 2;
        Padding::new(before * self.dilation, (total - before) * self.dilation)
    }
}

impl Layer for TfConv {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, _, height, width] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "MobileNetV2 map, [B, C, H, W]",
            ));
        };
        let padded = input.pad(
            &[
                Padding::ZERO,
                Padding::ZERO,
                self.padding(height),
                self.padding(width),
            ],
            0.0,
        )?;
        let mut output = padded.apply(&self.convolution, state)?;
        if let Some(norm) = &self.norm {
            output = output.apply(norm, state)?;
        }
        match &self.activation {
            Some(activation) => output.apply(activation, state),
            None => Ok(output),
        }
    }
}

/// An inverted residual: a 1 × 1 expansion, a 3 × 3 depthwise convolution carrying the stride or the dilation, and a 1 × 1 projection without activation, added to the input when the shapes match.
#[derive(Debug, Clone)]
struct InvertedResidual {
    expand: Option<TfConv>,
    depthwise: TfConv,
    reduce: TfConv,
    residual: bool,
}

impl InvertedResidual {
    fn create(
        schema: &mut Schema<'_>,
        config: &MobileNetV2Config,
        [input, expanded, out]: [i64; 3],
        stride: i64,
        dilation: i64,
        expands: bool,
    ) -> SingeResult<Self> {
        let activation = Some(config.activation);
        let eps = Some(config.eps);
        Ok(Self {
            expand: expands
                .then(|| {
                    TfConv::create(
                        &mut schema.scope("expand"),
                        [input, expanded, 1, 1, 1, 1],
                        eps,
                        activation,
                    )
                })
                .transpose()?,
            depthwise: TfConv::create(
                &mut schema.scope("depthwise"),
                [expanded, expanded, 3, stride, dilation, expanded],
                eps,
                activation,
            )?,
            reduce: TfConv::create(
                &mut schema.scope("reduce"),
                [expanded, out, 1, 1, 1, 1],
                eps,
                None,
            )?,
            residual: stride == 1 && input == out,
        })
    }
}

impl Layer for InvertedResidual {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let expanded = match &self.expand {
            Some(expand) => input.apply(expand, state)?,
            None => input.clone(),
        };
        let output = expanded
            .apply(&self.depthwise, state)?
            .apply(&self.reduce, state)?;
        if self.residual {
            output.add(input)
        } else {
            Ok(output)
        }
    }
}

/// The network from images to the last inverted residual's map.
#[derive(Debug, Clone)]
struct Backbone {
    stem: TfConv,
    first: InvertedResidual,
    blocks: Vec<InvertedResidual>,
}

impl Backbone {
    /// Registers `stem`, the first block, which the stem's convolution expands, as `first`, and `blocks.N`.
    fn create(schema: &mut Schema<'_>, config: &MobileNetV2Config) -> SingeResult<Self> {
        let channels = &config.channels;
        let stem = TfConv::create(
            &mut schema.scope("stem"),
            [3, config.stem, 3, 2, 1, 1],
            Some(config.eps),
            Some(config.activation),
        )?;
        let first = InvertedResidual::create(
            &mut schema.scope("first"),
            config,
            [config.stem, config.stem, channels[0]],
            1,
            1,
            config.first_expands,
        )?;
        let mut stride = 2;
        let mut dilation = 1;
        let mut blocks = Vec::with_capacity(config.strides.len());
        for (index, &block_stride) in config.strides.iter().enumerate() {
            // Past the output stride, a block keeps the map's size and dilates instead; the next block takes the larger dilation.
            let (layer_stride, layer_dilation) = if stride == config.output_stride {
                let layer_dilation = dilation;
                dilation *= block_stride;
                (1, layer_dilation)
            } else {
                stride *= block_stride;
                (block_stride, 1)
            };
            let (input, out) = (channels[index], channels[index + 1]);
            let expanded = make_divisible(
                (input as f64 * config.expansion).round(),
                config.divisor,
                config.min_depth,
            );
            blocks.push(InvertedResidual::create(
                &mut schema.scope(format!("blocks.{index}")),
                config,
                [input, expanded, out],
                layer_stride,
                layer_dilation,
                true,
            )?);
        }
        Ok(Self {
            stem,
            first,
            blocks,
        })
    }
}

impl Layer for Backbone {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.apply(&self.stem, state)?.apply(&self.first, state)?;
        for block in &self.blocks {
            hidden = hidden.apply(block, state)?;
        }
        Ok(hidden)
    }
}

/// `MobileNetV2` and its classifier: from images to each label's score, `[B, labels]`.
#[derive(Debug, Clone)]
pub struct MobileNetV2ClassifierLayers {
    backbone: Backbone,
    output: TfConv,
    classifier: Linear,
}

impl MobileNetV2ClassifierLayers {
    pub fn create(schema: &mut Schema<'_>, config: &MobileNetV2Config) -> SingeResult<Self> {
        let last = config.channels.last().copied().unwrap_or_default();
        Ok(Self {
            backbone: Backbone::create(schema, config)?,
            output: TfConv::create(
                &mut schema.scope("output"),
                [last, config.last, 1, 1, 1, 1],
                Some(config.eps),
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

impl Layer for MobileNetV2ClassifierLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.backbone, state)?
            .apply(&self.output, state)?
            .mean(&[2, 3], false)?
            .apply(&self.classifier, state)
    }
}

/// `MobileNetV2` and `DeepLabV3`'s head: from images to each label's score at each point of a map an eighth of the image's size, `[B, labels, h, w]`.
#[derive(Debug, Clone)]
pub struct DeepLabV3Layers {
    backbone: Backbone,
    pool: TfConv,
    branch: TfConv,
    projection: TfConv,
    classifier: TfConv,
}

impl DeepLabV3Layers {
    pub fn create(schema: &mut Schema<'_>, config: &MobileNetV2Config) -> SingeResult<Self> {
        let last = config.channels.last().copied().unwrap_or_default();
        let branch = |schema: &mut Schema<'_>, name: &str, input: i64| {
            TfConv::create(
                &mut schema.scope(name),
                [input, HEAD_CHANNELS, 1, 1, 1, 1],
                Some(HEAD_NORM_EPS),
                Some(Activation::Relu),
            )
        };
        let backbone = Backbone::create(schema, config)?;
        let mut head = schema.scope("deeplab");
        Ok(Self {
            backbone,
            pool: branch(&mut head, "pool", last)?,
            branch: branch(&mut head, "branch", last)?,
            projection: branch(&mut head, "projection", 2 * HEAD_CHANNELS)?,
            classifier: TfConv::create(
                &mut head.scope("classifier"),
                [HEAD_CHANNELS, config.classifier.label_count()?, 1, 1, 1, 1],
                None,
                None,
            )?,
        })
    }
}

impl Layer for DeepLabV3Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let features = input.apply(&self.backbone, state)?;
        // The average's branch, resized from one point with aligned corners, is that point everywhere.
        let pooled = features
            .mean(&[2, 3], true)?
            .apply(&self.pool, state)?
            .expand_with(&[
                features.dimension(0)?,
                HEAD_CHANNELS.into(),
                features.dimension(2)?,
                features.dimension(3)?,
            ])?;
        Tensor::concat(&[pooled, features.apply(&self.branch, state)?], 1)?
            .apply(&self.projection, state)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "mobilenet_v2",
    formats: &[(Format::Transformers, &["mobilenet_v2"])],
    loaders: &[
        HeadLoader::ImageClassification(load_classifier),
        HeadLoader::ImageSegmentation(load_segmenter),
    ],
};

pub fn load_classifier(loader: &mut Loader<'_>) -> Result<ImageClassifier> {
    let config: MobileNetV2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        MobileNetV2ClassifierLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::classifier_rules())?;
    Ok(ImageClassifier(Classifier::new(model, config.classifier)))
}

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<ImageSegmenter> {
    let config: MobileNetV2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DeepLabV3Layers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::segmenter_rules())?;
    Ok(ImageSegmenter::new(
        model,
        config.classifier,
        SegmenterLayout::Semantic,
    ))
}
