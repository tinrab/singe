//! DETR: a `ResNet`'s last map, projected and flattened, through a post-norm transformer encoder with sine positions of each point, then a decoder of learned object queries, each read by a class head and a box head of three linear layers.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, LinearStack, PYTORCH_NORM_EPS, ResNet, ResNetConfig,
        SinePositions, sine_positions, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, DetectionScoring, HeadLoader, ObjectDetector},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetrConfig {
    pub backbone: ResNetConfig,
    pub width: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub encoder_heads: i64,
    pub decoder_heads: i64,
    pub encoder_inner: i64,
    pub decoder_inner: i64,
    pub activation: Activation,
    pub queries: i64,
    /// The labels, before the last score, which stands for no object.
    pub classifier: ClassifierConfig,
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl EncoderBlock {
    fn create(schema: &mut Schema<'_>, config: &DetrConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, config.encoder_heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.encoder_inner, width],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
        })
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    cross_attention: Attention,
    cross_attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl DecoderBlock {
    fn create(schema: &mut Schema<'_>, config: &DetrConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, config.decoder_heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            cross_attention: Attention::create(
                &mut schema.scope("cross_attention"),
                AttentionSpec::square(width, config.decoder_heads),
            )?,
            cross_attention_norm: schema.layer_norm(
                "cross_attention_norm",
                width,
                PYTORCH_NORM_EPS,
            )?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.decoder_inner, width],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
        })
    }
}

/// DETR and its heads: from images to each query's label scores and box, `[B, Q, labels + 1 + 4]`.
#[derive(Debug, Clone)]
pub struct DetrLayers {
    backbone: ResNet,
    projection: Conv2d,
    encoder: Vec<EncoderBlock>,
    decoder: Vec<DecoderBlock>,
    decoder_norm: LayerNorm,
    queries: StateKey,
    classes: Linear,
    boxes: LinearStack,
}

impl DetrLayers {
    pub fn create(schema: &mut Schema<'_>, config: &DetrConfig) -> SingeResult<Self> {
        let width = config.width;
        let backbone = ResNet::create(&mut schema.scope("backbone"), &config.backbone)?;
        let channels = config.backbone.channels(config.backbone.depths.len() - 1);
        let projection = schema.conv2d(
            "projection",
            channels,
            width,
            1,
            Conv2dConfig::default(),
            true,
        )?;
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                EncoderBlock::create(&mut schema.scope(format!("encoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                DecoderBlock::create(&mut schema.scope(format!("decoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let labels = config.classifier.label_count()?;
        let boxes = LinearStack::create(
            &mut schema.scope("boxes"),
            [width, width, 4],
            3,
            Activation::Relu,
        )?;
        Ok(Self {
            backbone,
            projection,
            encoder,
            decoder,
            decoder_norm: schema.layer_norm("decoder_norm", width, PYTORCH_NORM_EPS)?,
            queries: schema.parameter("queries", [config.queries, width])?,
            classes: schema.linear("classes", width, labels + 1, true)?,
            boxes,
        })
    }
}

impl Layer for DetrLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Detections are thresholded scores, which TF32 convolutions in the backbone move by about 1e-3 from `transformers`' own, as with SuperPoint's keypoints.
        let map = with_convolution_precision(ComputePrecision::Full, || {
            input
                .apply(&self.backbone, state)?
                .apply(&self.projection, state)
        })?;
        let &[_, width, _, _] = map.shape().dimensions() else {
            return Err(SingeError::invalid_argument("DETR feature map"));
        };
        let positions = sine_positions(&map, width, SinePositions::DETR)?.cast(map.data_type())?;
        let mut memory = to_tokens(&map)?;
        for block in &self.encoder {
            let keyed = memory.add(&positions)?;
            memory = block
                .attention
                .forward(&keyed, &keyed, &memory, state)?
                .add(&memory)?
                .apply(&block.attention_norm, state)?;
            memory = memory
                .apply(&block.feed_forward, state)?
                .add(&memory)?
                .apply(&block.feed_forward_norm, state)?;
        }

        let queries = state.tensor(&self.queries)?;
        let &[count, _] = queries.shape().dimensions() else {
            return Err(SingeError::invalid_argument("DETR queries"));
        };
        let queries =
            queries
                .unsqueeze(0)?
                .expand_with(&[map.dimension(0)?, count.into(), width.into()])?;
        let keys = memory.add(&positions)?;
        let mut hidden = queries.mul_scalar(0.0)?;
        for block in &self.decoder {
            let positioned = hidden.add(&queries)?;
            hidden = block
                .attention
                .forward(&positioned, &positioned, &hidden, state)?
                .add(&hidden)?
                .apply(&block.attention_norm, state)?;
            hidden = block
                .cross_attention
                .forward(&hidden.add(&queries)?, &keys, &memory, state)?
                .add(&hidden)?
                .apply(&block.cross_attention_norm, state)?;
            hidden = hidden
                .apply(&block.feed_forward, state)?
                .add(&hidden)?
                .apply(&block.feed_forward_norm, state)?;
        }

        let hidden = hidden.apply(&self.decoder_norm, state)?;

        let scores = hidden.apply(&self.classes, state)?;
        let boxes = hidden
            .apply(&self.boxes, state)?
            .apply(&Activation::Sigmoid, state)?;
        Tensor::concat(&[scores, boxes], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "detr",
    formats: &[(Format::Transformers, &["detr"])],
    loaders: &[HeadLoader::ObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<ObjectDetector> {
    let config: DetrConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        DetrLayers::create(schema, &config).map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ObjectDetector::new(
        model,
        config.classifier,
        DetectionScoring::NoObject,
    ))
}
