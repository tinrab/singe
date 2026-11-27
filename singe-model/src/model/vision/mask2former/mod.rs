//! `Mask2Former`: a Swin's four stage maps, the three coarsest projected through a post-norm encoder of deformable self-attention and the finest added to an upsample of the result, giving mask features at a quarter of the image's size.
//! Learned queries then pass through post-norm decoder layers that cycle over the three coarsest levels, each attending only where the query's mask from the layer before is likely, and each query's last state gives its label scores and, against the mask features, its mask.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    graph::{ComputePrecision, capture, with_convolution_precision},
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::Spatial2d,
    tensor::{AttentionOptions, InterpolateMode},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, ConvGroupNorm, DeformableAttentionSpec, DeformableEncoderLayer,
        FeaturePyramid, LinearStack, PYTORCH_NORM_EPS, SinePositionSpec, Swin, SwinConfig,
        sine_positions, to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ClassifierConfig, HeadLoader, ImageSegmenter, SegmenterLayout},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The levels the pixel decoder's encoder and the decoder layers read: the backbone's three coarsest.
const LEVELS: usize = 3;

/// The groups of the pixel decoder's norms.
const GROUPS: i64 = 32;

/// The sampling points of each head at each level in the pixel decoder's encoder.
const ENCODER_POINTS: i64 = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mask2FormerConfig {
    pub backbone: SwinConfig,
    pub width: i64,
    pub heads: i64,
    pub encoder_layers: i64,
    pub encoder_inner: i64,
    pub decoder_layers: i64,
    pub decoder_inner: i64,
    pub activation: Activation,
    pub queries: i64,
    /// How many times smaller than the image the mask features are.
    pub stride: i64,
    /// The labels, before the last score, which stands for no object.
    pub classifier: ClassifierConfig,
}

/// A post-norm decoder layer: cross-attention to one level under the mask of each query's likely pixels, self-attention of the queries, then a feed-forward network, as `Mask2FormerMaskedAttentionDecoderLayer` computes it without `pre_norm`.
#[derive(Debug, Clone)]
struct DecoderLayer {
    cross_attention: Attention,
    cross_attention_norm: LayerNorm,
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl DecoderLayer {
    fn create(schema: &mut Schema<'_>, config: &Mask2FormerConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            cross_attention: Attention::create(
                &mut schema.scope("cross_attention"),
                AttentionSpec::square(width, config.heads),
            )?,
            cross_attention_norm: schema.layer_norm(
                "cross_attention_norm",
                width,
                PYTORCH_NORM_EPS,
            )?,
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, config.heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.decoder_inner, width],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
        })
    }

    /// Runs the queries `hidden`, `[B, Q, W]`, with `positions`, against `memory`, `[B, S, W]`, whose keys carry `memory_positions`, under the additive `mask`, `[B, 1, Q, S]`.
    fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        memory: &Tensor,
        memory_positions: &Tensor,
        mask: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self
            .cross_attention
            .attend(
                &hidden.add(positions)?,
                &memory.add(memory_positions)?,
                memory,
                Some(mask),
                AttentionOptions::default(),
                state,
            )?
            .add(hidden)?
            .apply(&self.cross_attention_norm, state)?;
        let positioned = hidden.add(positions)?;
        let hidden = self
            .attention
            .forward(&positioned, &positioned, &hidden, state)?
            .add(&hidden)?
            .apply(&self.attention_norm, state)?;
        hidden
            .apply(&self.feed_forward, state)?
            .add(&hidden)?
            .apply(&self.feed_forward_norm, state)
    }
}

/// `Mask2Former` and its heads: from images to each query's label scores, the last for no object, and its mask's logits at a quarter of the image's size, `[B, Q, labels + 1 + h · w]`.
#[derive(Debug, Clone)]
pub struct Mask2FormerLayers {
    width: i64,
    backbone: Swin,
    stage_norms: Vec<LayerNorm>,
    input_projections: Vec<ConvGroupNorm>,
    level_embeddings: StateKey,
    encoder: Vec<DeformableEncoderLayer>,
    lateral: ConvGroupNorm,
    output: ConvGroupNorm,
    mask_projection: Conv2d,
    query_level_embeddings: StateKey,
    query_positions: StateKey,
    query_features: StateKey,
    decoder_norm: LayerNorm,
    mask_embedder: LinearStack,
    decoder: Vec<DecoderLayer>,
    classes: Linear,
}

impl Mask2FormerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Mask2FormerConfig) -> SingeResult<Self> {
        let width = config.width;
        let backbone = Swin::create(&mut schema.scope("backbone"), &config.backbone)?;
        let stages = config.backbone.stages.len();
        let channels = |stage: usize| config.backbone.width << stage;
        let stage_norms = (0..stages)
            .map(|stage| {
                schema.layer_norm(
                    &format!("stage_norms.{stage}"),
                    channels(stage),
                    PYTORCH_NORM_EPS,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        // From the coarsest map.
        let input_projections = (0..LEVELS)
            .map(|level| {
                ConvGroupNorm::create(
                    &mut schema.scope(format!("input_projections.{level}")),
                    [channels(stages - 1 - level), width, 1, 1, GROUPS],
                    true,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let levels = i64::try_from(LEVELS).unwrap_or(i64::MAX);
        let spec = DeformableAttentionSpec {
            width,
            heads: config.heads,
            levels,
            points: ENCODER_POINTS,
        };
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                DeformableEncoderLayer::create(
                    &mut schema.scope(format!("encoder.{index}")),
                    spec,
                    config.encoder_inner,
                    Activation::Relu,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let lateral = ConvGroupNorm::create(
            &mut schema.scope("lateral"),
            [channels(0), width, 1, 1, GROUPS],
            false,
        )?;
        let output = ConvGroupNorm::create(
            &mut schema.scope("output"),
            [width, width, 3, 1, GROUPS],
            false,
        )?
        .with_activation(Activation::Relu);
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                DecoderLayer::create(&mut schema.scope(format!("decoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let labels = config.classifier.label_count()?;
        Ok(Self {
            width,
            backbone,
            stage_norms,
            input_projections,
            level_embeddings: schema.parameter("level_embeddings", [levels, width])?,
            encoder,
            lateral,
            output,
            mask_projection: schema.conv2d(
                "mask_projection",
                width,
                width,
                1,
                Conv2dConfig::default(),
                true,
            )?,
            query_level_embeddings: schema.parameter("query_level_embeddings", [levels, width])?,
            query_positions: schema.parameter("query_positions", [config.queries, width])?,
            query_features: schema.parameter("query_features", [config.queries, width])?,
            decoder_norm: schema.layer_norm("decoder_norm", width, PYTORCH_NORM_EPS)?,
            mask_embedder: LinearStack::create(
                &mut schema.scope("mask_embedder"),
                [width, width, width],
                3,
                Activation::Relu,
            )?,
            decoder,
            classes: schema.linear("classes", width, labels + 1, true)?,
        })
    }

    /// Returns the three coarsest levels after the pixel decoder's encoder, from the coarsest, and the mask features, `[B, W, H / 4, W / 4]`.
    fn pixel_decoder(
        &self,
        input: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Vec<Tensor>, Tensor)> {
        let maps = self
            .backbone
            .stage_tokens(input, state)?
            .into_iter()
            .zip(&self.stage_norms)
            .map(|((tokens, size), norm)| to_map(&tokens.apply(norm, state)?, size))
            .collect::<SingeResult<Vec<_>>>()?;
        let level_embeddings = state.tensor(&self.level_embeddings)?;
        let mut tokens = Vec::with_capacity(LEVELS);
        let mut positions = Vec::with_capacity(LEVELS);
        let mut sizes = Vec::with_capacity(LEVELS);
        for (level, projection) in self.input_projections.iter().enumerate() {
            let map = &maps[maps.len() - 1 - level];
            let projected = map.apply(projection, state)?;
            let embedding = level_embeddings
                .narrow(0, i64::try_from(level).unwrap_or(i64::MAX), 1)?
                .unsqueeze(0)?;
            positions.push(
                sine_positions(map, self.width, SinePositionSpec::DETR)?
                    .cast(projected.data_type())?
                    .add(&embedding)?,
            );
            sizes.push([projected.dimension(2)?, projected.dimension(3)?]);
            tokens.push(to_tokens(&projected)?);
        }

        let positions = Tensor::concat(&positions, 1)?;
        let levels = FeaturePyramid::create(input.device(), sizes.clone())?;
        let references = levels.reference_points()?.cast(positions.data_type())?;
        let mut memory = Tensor::concat(&tokens, 1)?;
        for layer in &self.encoder {
            memory = layer.forward(&memory, &positions, &references, &levels, state)?;
        }

        let mut start = 0;
        let mut outputs = Vec::with_capacity(LEVELS);
        for (map, [height, width]) in maps.iter().rev().zip(sizes) {
            let &[_, _, rows, columns] = map.shape().dimensions() else {
                return Err(SingeError::invalid_argument(
                    "Mask2Former map, [B, C, H, W]",
                ));
            };
            let count = rows * columns;
            outputs.push(to_map(
                &memory.narrow(1, start, count)?,
                Spatial2d::new(height, width),
            )?);
            start += count;
        }

        let finest = maps[0].apply(&self.lateral, state)?;
        let size = (finest.dimension(2)?, finest.dimension(3)?);
        let upsampled = outputs[LEVELS - 1].interpolate(size, InterpolateMode::Bilinear)?;
        let features = finest
            .add(&upsampled)?
            .apply(&self.output, state)?
            .apply(&self.mask_projection, state)?;
        Ok((outputs, features))
    }

    /// Returns each query's mask logits against `features`, `[B, Q, h, w]`, and the additive mask its next cross-attention reads at a level of `size`, `[B, 1, Q, s]`: negative infinity where the mask resized to the level is less likely than not, unless that is everywhere.
    fn predict_masks(
        &self,
        normed: &Tensor,
        features: &Tensor,
        size: [i64; 2],
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let &[_, _, height, width] = features.shape().dimensions() else {
            return Err(SingeError::invalid_argument("mask features, [B, W, h, w]"));
        };
        let queries = normed.dimension(1)?;
        let masks = normed
            .apply(&self.mask_embedder, state)?
            .matmul(&features.flatten(2, 3)?)?
            .reshape_with(&[features.dimension(0)?, queries, height.into(), width.into()])?;
        let device = normed.device();
        let literal = |value: f64| capture::scalar(device, value as f32);
        let likely = masks
            .cast(DataType::F32)?
            .interpolate(Spatial2d::new(size[0], size[1]), InterpolateMode::Bilinear)?
            .sigmoid()?
            .flatten(2, 3)?;
        let masked = likely.less(&literal(0.5)?)?;
        let everywhere = masked
            .cast(DataType::F32)?
            .sum(&[2], true)?
            .equal(&literal((size[0] * size[1]) as f64)?)?;
        let masked = masked.logical_and(&everywhere.logical_not()?)?;
        let additive = Tensor::where_select(&masked, &literal(f64::NEG_INFINITY)?, &literal(0.0)?)?
            .cast(normed.data_type())?
            .unsqueeze(1)?;
        Ok((masks, additive))
    }
}

impl Layer for Mask2FormerLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        // Segments are thresholded scores, which TF32 convolutions move, as DETR's detections.
        let (levels, features) = with_convolution_precision(ComputePrecision::Full, || {
            self.pixel_decoder(input, state)
        })?;
        let query_level_embeddings = state.tensor(&self.query_level_embeddings)?;
        let mut memories = Vec::with_capacity(LEVELS);
        let mut memory_positions = Vec::with_capacity(LEVELS);
        let mut sizes = Vec::with_capacity(LEVELS);
        for (index, map) in levels.iter().enumerate() {
            let &[_, _, height, width] = map.shape().dimensions() else {
                return Err(SingeError::invalid_argument(
                    "Mask2Former level, [B, W, h, w]",
                ));
            };
            let embedding = query_level_embeddings
                .narrow(0, i64::try_from(index).unwrap_or(i64::MAX), 1)?
                .unsqueeze(0)?;
            memory_positions.push(
                sine_positions(map, self.width, SinePositionSpec::DETR)?.cast(map.data_type())?,
            );
            memories.push(to_tokens(map)?.add(&embedding)?);
            sizes.push([height, width]);
        }

        let positions = state.tensor(&self.query_positions)?.unsqueeze(0)?;
        let query_features = state.tensor(&self.query_features)?;
        let batched = [
            input.dimension(0)?,
            query_features.dimension(0)?,
            self.width.into(),
        ];
        let mut hidden = query_features.unsqueeze(0)?.expand_with(&batched)?;
        let mut normed = hidden.apply(&self.decoder_norm, state)?;
        let (mut masks, mut mask) = self.predict_masks(&normed, &features, sizes[0], state)?;
        for (index, layer) in self.decoder.iter().enumerate() {
            let level = index % LEVELS;
            hidden = layer.forward(
                &hidden,
                &positions,
                &memories[level],
                &memory_positions[level],
                &mask,
                state,
            )?;
            normed = hidden.apply(&self.decoder_norm, state)?;
            (masks, mask) =
                self.predict_masks(&normed, &features, sizes[(index + 1) % LEVELS], state)?;
        }

        let classes = normed.apply(&self.classes, state)?;
        Tensor::concat(&[classes, masks.flatten(2, 3)?], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "mask2former",
    formats: &[(Format::Transformers, &["mask2former"])],
    loaders: &[HeadLoader::ImageSegmentation(load_segmenter)],
};

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<ImageSegmenter> {
    let config: Mask2FormerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Mask2FormerLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(ImageSegmenter::new(
        model,
        config.classifier,
        SegmenterLayout::Universal {
            stride: config.stride,
        },
    ))
}
