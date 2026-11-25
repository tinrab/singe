//! `SegFormer`: stages of overlapping patch embeddings and blocks whose attention reads keys and values from a strided reduction of the map, each stage's map projected, resized to the first's, and fused into each pixel's label scores.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, BatchNorm, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, ModelState},
    shape::spatial::{Spatial2d, Spatial2dI64},
    tensor::InterpolateMode,
};

use crate::{
    assembly::{Attention, AttentionSpec, PYTORCH_NORM_EPS, same_2d, spatial, to_map, to_tokens},
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, HeadLoader, SemanticSegmenter},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegformerStageConfig {
    pub width: i64,
    pub blocks: i64,
    pub heads: i64,
    pub patch_size: i64,
    pub stride: i64,
    /// The side of the squares the keys and values are reduced over, or one for none.
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

#[derive(Debug, Clone)]
struct ReducedAttention {
    attention: Attention,
    reduction: Option<(Conv2d, LayerNorm)>,
}

impl ReducedAttention {
    fn create(schema: &mut Schema<'_>, config: &SegformerStageConfig) -> SingeResult<Self> {
        let width = config.width;
        let reduction = if config.reduction > 1 {
            Some((
                schema.conv2d(
                    "reduction",
                    width,
                    width,
                    config.reduction,
                    Conv2dConfig {
                        stride: Spatial2d::square(config.reduction),
                        ..Conv2dConfig::default()
                    },
                    true,
                )?,
                schema.layer_norm("reduction_norm", width, PYTORCH_NORM_EPS)?,
            ))
        } else {
            None
        };
        Ok(Self {
            attention: Attention::create(schema, AttentionSpec::square(width, config.heads))?,
            reduction,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        size: Spatial2dI64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let context = match &self.reduction {
            Some((convolution, norm)) => {
                to_tokens(&to_map(hidden, size)?.apply(convolution, state)?)?.apply(norm, state)?
            }
            None => hidden.clone(),
        };
        self.attention.forward(hidden, &context, &context, state)
    }
}

/// A feed-forward whose depthwise convolution over the map stands in for positions.
#[derive(Debug, Clone)]
struct MixFeedForward {
    up: Linear,
    depthwise: Conv2d,
    down: Linear,
    activation: Activation,
}

impl MixFeedForward {
    fn create(
        schema: &mut Schema<'_>,
        config: &SegformerStageConfig,
        activation: Activation,
    ) -> SingeResult<Self> {
        let (width, inner) = (config.width, config.inner);
        Ok(Self {
            up: schema.linear("up", width, inner, true)?,
            depthwise: schema.conv2d(
                "depthwise",
                inner,
                inner,
                3,
                Conv2dConfig {
                    groups: inner,
                    ..same_2d(3, 1)
                },
                true,
            )?,
            down: schema.linear("down", inner, width, true)?,
            activation,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        size: Spatial2dI64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let inner = hidden.apply(&self.up, state)?;
        to_tokens(&to_map(&inner, size)?.apply(&self.depthwise, state)?)?
            .apply(&self.activation, state)?
            .apply(&self.down, state)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    attention: ReducedAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: MixFeedForward,
}

impl Block {
    fn create(
        schema: &mut Schema<'_>,
        config: &SegformerStageConfig,
        activation: Activation,
    ) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            attention: ReducedAttention::create(&mut schema.scope("attention"), config)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: MixFeedForward::create(
                &mut schema.scope("feed_forward"),
                config,
                activation,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        size: Spatial2dI64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self
            .attention
            .forward(&hidden.apply(&self.attention_norm, state)?, size, state)?
            .add(hidden)?;
        self.feed_forward
            .forward(&hidden.apply(&self.feed_forward_norm, state)?, size, state)?
            .add(&hidden)
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
    fn create(
        schema: &mut Schema<'_>,
        config: &SegformerStageConfig,
        input: i64,
        activation: Activation,
    ) -> SingeResult<Self> {
        let width = config.width;
        let patches = schema.conv2d(
            "patches",
            input,
            width,
            config.patch_size,
            same_2d(config.patch_size, config.stride),
            true,
        )?;
        let patch_norm = schema.layer_norm("patch_norm", width, PYTORCH_NORM_EPS)?;
        let blocks = (0..config.blocks)
            .map(|index| {
                Block::create(
                    &mut schema.scope(format!("blocks.{index}")),
                    config,
                    activation,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            patches,
            patch_norm,
            blocks,
            norm: schema.layer_norm("norm", width, PYTORCH_NORM_EPS)?,
        })
    }
}

impl Layer for Stage {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let size = spatial(&patches)?;
        let mut hidden = to_tokens(&patches)?.apply(&self.patch_norm, state)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, size, state)?;
        }
        to_map(&hidden.apply(&self.norm, state)?, size)
    }
}

/// `SegFormer` and its decode head: from images to each label's score at the first stage's resolution, `[B, labels, H / s, W / s]` for the first stage's stride `s`.
#[derive(Debug, Clone)]
pub struct SegformerLayers {
    stages: Vec<Stage>,
    projections: Vec<Linear>,
    fuse: Conv2d,
    norm: BatchNorm,
    classifier: Conv2d,
}

impl SegformerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &SegformerConfig) -> SingeResult<Self> {
        let decoder = config.decoder;
        let mut input = config.channels;
        let mut stages = Vec::new();
        let mut projections = Vec::new();
        for (index, stage) in config.stages.iter().enumerate() {
            stages.push(Stage::create(
                &mut schema.scope(format!("stages.{index}")),
                stage,
                input,
                config.activation,
            )?);
            projections.push(schema.linear(
                &format!("projections.{index}"),
                stage.width,
                decoder,
                true,
            )?);
            input = stage.width;
        }

        let count =
            i64::try_from(config.stages.len()).map_err(|_| SingeError::out_of_range("stages"))?;
        let labels = config.classifier.label_count()?;
        Ok(Self {
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
            norm: schema.batch_norm("norm", decoder, PYTORCH_NORM_EPS)?,
            classifier: schema.conv2d(
                "classifier",
                decoder,
                labels,
                1,
                Conv2dConfig::default(),
                true,
            )?,
        })
    }
}

impl Layer for SegformerLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut maps = Vec::with_capacity(self.stages.len());
        let mut hidden = input.clone();
        for stage in &self.stages {
            hidden = hidden.apply(stage, state)?;
            maps.push(hidden.clone());
        }

        let first = maps
            .first()
            .ok_or_else(|| SingeError::invalid_argument("no stages"))?;
        let size = spatial(first)?;
        // Concatenated coarsest first, as `SegformerDecodeHead` stacks them.
        let projected = maps
            .iter()
            .zip(&self.projections)
            .rev()
            .map(|(stage, projection)| {
                to_map(
                    &to_tokens(stage)?.apply(projection, state)?,
                    spatial(stage)?,
                )?
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
    loaders: &[HeadLoader::ImageSegmentation(load_segmenter)],
};

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<SemanticSegmenter> {
    let config: SegformerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        SegformerLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(SemanticSegmenter::new(model, config.classifier))
}
