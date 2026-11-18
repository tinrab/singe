// SegFormer: four stages of overlapping patch embeddings and blocks whose attention reads keys and values from a strided reduction of the map, each stage's map projected, resized to the first's, and fused into each pixel's label scores.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Activation, BatchNorm2d, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, Model, ModelState,
        StateBuilder,
    },
    tensor::InterpolateMode,
};

use crate::{
    assembly::{to_map, to_tokens},
    error::Result,
    format::Loader,
    head::{ClassifierConfig, HeadLoader, PixelLayers, SemanticSegmenter},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// `nn.LayerNorm`'s default, which SegFormer's norms keep whatever the config says.
const EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegformerStageConfig {
    pub width: i64,
    pub blocks: i64,
    pub heads: i64,
    pub patch_size: i64,
    pub stride: i64,
    // The side of the squares the keys and values are reduced over, or one for none.
    pub reduction: i64,
    pub inner: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegformerConfig {
    pub stages: Vec<SegformerStageConfig>,
    pub channels: i64,
    pub activation: Activation,
    pub decoder: i64,
    pub classifier: ClassifierConfig,
}

fn spatial(tensor: &Tensor) -> (i64, i64) {
    let dimensions = tensor.shape().dimensions();
    (dimensions[2], dimensions[3])
}

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    reduction: Option<(Conv2d, LayerNorm)>,
    heads: i64,
}

impl Attention {
    fn forward(
        &self,
        hidden: &Tensor,
        size: (i64, i64),
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, length, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention input"));
        };
        let context = match &self.reduction {
            Some((convolution, norm)) => {
                to_tokens(&to_map(hidden, size)?.apply(convolution, state)?)?.apply(norm, state)?
            }
            None => hidden.clone(),
        };
        let head = width / self.heads;
        let split = |tensor: Tensor| -> SingeResult<Tensor> {
            let length = tensor.shape().dimensions()[1];
            tensor
                .reshape([batch, length, self.heads, head])?
                .transpose(1, 2)
        };
        let query = split(hidden.apply(&self.query, state)?)?;
        let key = split(context.apply(&self.key, state)?)?;
        let value = split(context.apply(&self.value, state)?)?;
        query
            .scaled_dot_product_attention(&key, &value, None, false)?
            .transpose(1, 2)?
            .reshape([batch, length, width])?
            .apply(&self.output, state)
    }
}

// A feed-forward whose depthwise convolution over the map stands in for positions.
#[derive(Debug, Clone)]
struct MixFeedForward {
    up: Linear,
    depthwise: Conv2d,
    down: Linear,
    activation: Activation,
}

#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    attention: Attention,
    feed_forward_norm: LayerNorm,
    feed_forward: MixFeedForward,
}

impl Block {
    fn forward(
        &self,
        hidden: &Tensor,
        size: (i64, i64),
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self
            .attention
            .forward(&hidden.apply(&self.attention_norm, state)?, size, state)?
            .add(hidden)?;
        let feed_forward = &self.feed_forward;
        let inner = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&feed_forward.up, state)?;
        let inner = to_tokens(&to_map(&inner, size)?.apply(&feed_forward.depthwise, state)?)?
            .apply(&feed_forward.activation, state)?
            .apply(&feed_forward.down, state)?;
        inner.add(&hidden)
    }
}

#[derive(Debug, Clone)]
struct Stage {
    patches: Conv2d,
    patch_norm: LayerNorm,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl Stage {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let size = spatial(&patches);
        let mut hidden = to_tokens(&patches)?.apply(&self.patch_norm, state)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, size, state)?;
        }
        to_map(&hidden.apply(&self.norm, state)?, size)
    }
}

// SegFormer and its decode head: from images to each label's score at a quarter of the image's size, `[B, labels, H / 4, W / 4]`.
#[derive(Debug, Clone)]
pub struct SegformerLayers {
    stages: Vec<Stage>,
    projections: Vec<Linear>,
    fuse: Conv2d,
    norm: BatchNorm2d,
    classifier: Conv2d,
}

impl Layer for SegformerLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut maps = Vec::with_capacity(self.stages.len());
        let mut hidden = input.clone();
        for stage in &self.stages {
            hidden = stage.forward(&hidden, state)?;
            maps.push(hidden.clone());
        }

        let size = maps
            .first()
            .map(spatial)
            .ok_or_else(|| SingeError::invalid_argument("no stages"))?;
        // Concatenated coarsest first, as `SegformerDecodeHead` stacks them.
        let projected = maps
            .iter()
            .zip(&self.projections)
            .rev()
            .map(|(stage, projection)| {
                to_map(&to_tokens(stage)?.apply(projection, state)?, spatial(stage))?
                    .interpolate(size, InterpolateMode::Bilinear)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Tensor::concat(&projected, 1)?
            .apply(&self.fuse, state)?
            .apply(&self.norm, state)?
            .apply(&Activation::Relu, state)?
            .apply(&self.classifier, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "segformer",
    formats: &[(Format::Transformers, &["segformer"])],
    heads: &[HeadLoader::ImageSegmentation(load_segmenter)],
};

fn stage(
    schema: &mut Schema<'_>,
    config: &SegformerStageConfig,
    input: i64,
    activation: Activation,
) -> SingeResult<Stage> {
    let width = config.width;
    let half = config.patch_size / 2;
    let patches = schema.conv2d(
        "patches",
        input,
        width,
        config.patch_size,
        Conv2dConfig {
            stride: (config.stride, config.stride),
            padding: (half, half, half, half),
            ..Conv2dConfig::default()
        },
        true,
    )?;
    let patch_norm = schema.layer_norm("patch_norm", width, EPS)?;
    let blocks = (0..config.blocks)
        .map(|index| {
            let mut schema = schema.scope(format!("blocks.{index}"));
            let attention = {
                let mut schema = schema.scope("attention");
                let reduction = (config.reduction > 1)
                    .then(|| -> SingeResult<_> {
                        Ok((
                            schema.conv2d(
                                "reduction",
                                width,
                                width,
                                config.reduction,
                                Conv2dConfig {
                                    stride: (config.reduction, config.reduction),
                                    ..Conv2dConfig::default()
                                },
                                true,
                            )?,
                            schema.layer_norm("reduction_norm", width, EPS)?,
                        ))
                    })
                    .transpose()?;
                Attention {
                    query: schema.linear("query", width, width, true)?,
                    key: schema.linear("key", width, width, true)?,
                    value: schema.linear("value", width, width, true)?,
                    output: schema.linear("output", width, width, true)?,
                    reduction,
                    heads: config.heads,
                }
            };
            let feed_forward = {
                let mut schema = schema.scope("feed_forward");
                MixFeedForward {
                    up: schema.linear("up", width, config.inner, true)?,
                    depthwise: schema.conv2d(
                        "depthwise",
                        config.inner,
                        config.inner,
                        3,
                        Conv2dConfig {
                            padding: (1, 1, 1, 1),
                            groups: config.inner,
                            ..Conv2dConfig::default()
                        },
                        true,
                    )?,
                    down: schema.linear("down", config.inner, width, true)?,
                    activation,
                }
            };
            Ok(Block {
                attention_norm: schema.layer_norm("attention_norm", width, EPS)?,
                attention,
                feed_forward_norm: schema.layer_norm("feed_forward_norm", width, EPS)?,
                feed_forward,
            })
        })
        .collect::<SingeResult<Vec<_>>>()?;
    Ok(Stage {
        patches,
        patch_norm,
        blocks,
        norm: schema.layer_norm("norm", width, EPS)?,
    })
}

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<SemanticSegmenter> {
    let config: SegformerConfig = loader.config(transformers::config)?;
    let decoder = config.decoder;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let mut input = config.channels;
        let mut stages = Vec::new();
        let mut projections = Vec::new();
        for (index, stage_config) in config.stages.iter().enumerate() {
            stages.push(stage(
                &mut schema.scope(format!("stages.{index}")),
                stage_config,
                input,
                config.activation,
            )?);
            projections.push(schema.linear(
                &format!("projections.{index}"),
                stage_config.width,
                decoder,
                true,
            )?);
            input = stage_config.width;
        }

        let count =
            i64::try_from(config.stages.len()).map_err(|_| SingeError::out_of_range("stages"))?;
        let labels = i64::try_from(config.classifier.labels.len())
            .map_err(|_| SingeError::out_of_range("labels"))?;
        SegformerLayers {
            stages,
            projections,
            fuse: schema.conv2d(
                "fuse",
                decoder * count,
                decoder,
                1,
                Conv2dConfig::default(),
                false,
            )?,
            norm: schema.batch_norm2d("norm", decoder, EPS)?,
            classifier: schema.conv2d(
                "classifier",
                decoder,
                labels,
                1,
                Conv2dConfig::default(),
                true,
            )?,
        }
    };
    let schema = Model::new(PixelLayers(Box::new(layers)), builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(SemanticSegmenter::new(model, config.classifier))
}
