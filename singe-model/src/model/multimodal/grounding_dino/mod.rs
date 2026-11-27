//! Grounding DINO: a Swin's last three maps and a fourth strided from the last, projected as Deformable DETR's levels, and BERT over the text with each phrase reading only itself.
//! Each encoder layer fuses the image and the text by attention both ways, refines the text by self-attention, and the image by deformable attention.
//! The image points whose best token scores highest become the decoder's box references; each decoder layer attends to the queries, the text, and the levels around the boxes, then moves them.
//! A query's score for each token is its last state against the text's.

mod transformers;

use std::{cmp::Ordering, f32::consts::PI};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{ComputePrecision, capture, with_convolution_precision},
    nn::{Activation, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, ConvGroupNorm, DeformableAttention, DeformableAttentionSpec,
        DeformableEncoderLayer, FeaturePyramid, LinearStack, PostNormEncoderConfig,
        SinePositionSpec, Swin, SwinConfig, anchor_proposals, gather_rows, logit, merge_heads,
        sine_positions, split_heads, to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, GroundingInput, HeadLoader, OpenVocabularyDetector},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The tokens that end a phrase: `[CLS]`, `[SEP]`, `.`, and `?` in BERT's vocabulary, as `transformers` fixes them.
const SPECIAL_TOKENS: &[i64] = &[101, 102, 1012, 1029];

/// The groups of each level's projection norm.
const PROJECTION_GROUPS: i64 = 32;

/// The scores the fused attention clamps to after subtracting their maximum.
const FUSED_SCORE_LIMIT: f32 = 50_000.0;

/// What `torch.special.logit` keeps references away from 0 and 1 by.
const LOGIT_EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundingDinoConfig {
    pub backbone: SwinConfig,
    /// The backbone's last stages read as levels; any levels beyond them are strided from the last.
    pub backbone_levels: usize,
    pub text: PostNormEncoderConfig,
    pub width: i64,
    pub levels: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub encoder_heads: i64,
    pub decoder_heads: i64,
    pub encoder_inner: i64,
    pub decoder_inner: i64,
    pub encoder_points: i64,
    pub decoder_points: i64,
    pub activation: Activation,
    pub queries: i64,
    pub temperature: f32,
    pub norm_eps: f32,
}

/// Returns each coordinate of `coordinates`, `[..., n]` in `F32`, as `features` interleaved sines and cosines of `2π · c / 10000^(2k / features)`, `[..., n · features]`, with the first two coordinates' features swapped, as `encode_sinusoidal_position_embedding` gives them.
fn coordinate_sines(coordinates: &Tensor, features: i64) -> SingeResult<Tensor> {
    let device = coordinates.device();
    let half = features / 2;
    let periods = capture::literal(device, |device| {
        let values = (0..half)
            .map(|pair| 10_000_f32.powf((2 * pair) as f32 / features as f32))
            .collect();
        Tensor::from_vec(device, Shape::create([half])?, values)
    })?;
    let angles = coordinates
        .unsqueeze(-1)?
        .mul_scalar(f64::from(2.0 * PI))?
        .div(&periods)?;
    let sines = Tensor::stack(&[angles.sin()?, angles.cos()?], -1)?.flatten(-2, -1)?;
    let count = coordinates
        .shape()
        .dimensions()
        .last()
        .copied()
        .unwrap_or(1);
    let sines = if count >= 2 {
        let mut parts = vec![sines.narrow(-2, 1, 1)?, sines.narrow(-2, 0, 1)?];
        if count > 2 {
            parts.push(sines.narrow(-2, 2, count - 2)?);
        }
        Tensor::concat(&parts, -2)?
    } else {
        sines
    };
    sines.flatten(-2, -1)
}

#[derive(Debug, Clone)]
struct TextBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

/// BERT's encoder as Grounding DINO's text backbone reads it: each token's position given, and attention under a mask of which tokens read which.
#[derive(Debug, Clone)]
struct TextEncoder {
    tokens: StateKey,
    positions: StateKey,
    token_types: StateKey,
    embedding_norm: LayerNorm,
    blocks: Vec<TextBlock>,
}

impl TextEncoder {
    /// Registers BERT under the names [`post_norm_encoder_rules`](crate::assembly::post_norm_encoder_rules) gives it.
    fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<Self> {
        let width = config.width;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                Ok(TextBlock {
                    attention: Attention::create(
                        &mut schema.scope("attention"),
                        AttentionSpec::square(width, config.heads),
                    )?,
                    attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
                    feed_forward: schema.mlp(
                        "feed_forward",
                        [width, config.inner, width],
                        config.activation,
                        true,
                    )?,
                    feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut embeddings = schema.scope("embeddings");
        Ok(Self {
            tokens: embeddings.parameter("tokens.weight", [config.vocab_size, width])?,
            positions: embeddings.parameter("positions.weight", [config.max_positions, width])?,
            token_types: embeddings
                .parameter("token_types.weight", [config.token_types.max(1), width])?,
            embedding_norm: embeddings.layer_norm("norm", width, config.eps)?,
            blocks,
        })
    }

    fn forward(&self, input: &GroundingInput, state: &ModelState) -> SingeResult<Tensor> {
        let embed = |table: &StateKey, rows: &Tensor| -> SingeResult<Tensor> {
            let table = state.tensor(table)?;
            let width = table.dimension(1)?;
            table.index_select(0, &rows.flatten(0, 1)?)?.reshape_with(&[
                rows.dimension(0)?,
                rows.dimension(1)?,
                width,
            ])
        };
        // Every token is of type zero.
        let token_type = state
            .tensor(&self.token_types)?
            .narrow(0, 0, 1)?
            .unsqueeze(0)?;
        let mut hidden = embed(&self.tokens, &input.tokens)?
            .add(&embed(&self.positions, &input.positions)?)?
            .add(&token_type)?
            .apply(&self.embedding_norm, state)?;
        let mask = input.mask.cast(hidden.data_type())?;
        for block in &self.blocks {
            hidden = block
                .attention
                .attend(
                    &hidden,
                    &hidden,
                    &hidden,
                    Some(&mask),
                    AttentionOptions::default(),
                    state,
                )?
                .add(&hidden)?
                .apply(&block.attention_norm, state)?;
            hidden = hidden
                .apply(&block.feed_forward, state)?
                .add(&hidden)?
                .apply(&block.feed_forward_norm, state)?;
        }
        Ok(hidden)
    }
}

/// Attention from the image to the text and from the text to the image on shared scores, as `GroundingDinoBiMultiHeadAttention` computes them: the scores less their maximum and clamped, then each direction's softmax, the text's after its rows' own maxima.
#[derive(Debug, Clone)]
struct FusedAttention {
    vision: Linear,
    text: Linear,
    vision_values: Linear,
    text_values: Linear,
    vision_output: Linear,
    text_output: Linear,
    heads: i64,
    scale: f64,
}

impl FusedAttention {
    fn create(schema: &mut Schema<'_>, width: i64, inner: i64, heads: i64) -> SingeResult<Self> {
        Ok(Self {
            vision: schema.linear("vision", width, inner, true)?,
            text: schema.linear("text", width, inner, true)?,
            vision_values: schema.linear("vision_values", width, inner, true)?,
            text_values: schema.linear("text_values", width, inner, true)?,
            vision_output: schema.linear("vision_output", inner, width, true)?,
            text_output: schema.linear("text_output", inner, width, true)?,
            heads,
            scale: ((inner / heads) as f64).powf(-0.5),
        })
    }

    /// Returns what the image reads of the text and what the text reads of the image.
    fn forward(
        &self,
        vision: &Tensor,
        text: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let heads =
            |input: &Tensor, linear: &Linear| split_heads(&input.apply(linear, state)?, self.heads);
        let queries = split_heads(
            &vision.apply(&self.vision, state)?.mul_scalar(self.scale)?,
            self.heads,
        )?;
        let keys = heads(text, &self.text)?;
        let scores = queries.matmul(&keys.transpose(2, 3)?)?;
        let scores = scores
            .sub(&scores.max_reduce(&[0, 1, 2, 3], true)?)?
            .clamp(-FUSED_SCORE_LIMIT, FUSED_SCORE_LIMIT)?;
        let transposed = scores.transpose(2, 3)?;
        let text_scores = transposed
            .sub(&transposed.max_reduce(&[3], true)?)?
            .clamp(-FUSED_SCORE_LIMIT, FUSED_SCORE_LIMIT)?;
        let read_by_vision = merge_heads(
            &scores
                .softmax(-1)?
                .matmul(&heads(text, &self.text_values)?)?,
        )?
        .apply(&self.vision_output, state)?;
        let read_by_text = merge_heads(
            &text_scores
                .softmax(-1)?
                .matmul(&heads(vision, &self.vision_values)?)?,
        )?
        .apply(&self.text_output, state)?;
        Ok((read_by_vision, read_by_text))
    }
}

/// One encoder layer: the fusion of the normed image and text, scaled per channel and added to them, the text's self-attention under its phrase mask, then the image's deformable self-attention.
#[derive(Debug, Clone)]
struct EncoderLayer {
    vision_norm: LayerNorm,
    text_norm: LayerNorm,
    fusion: FusedAttention,
    vision_scale: StateKey,
    text_scale: StateKey,
    text_attention: Attention,
    text_attention_norm: LayerNorm,
    text_feed_forward: Mlp,
    text_feed_forward_norm: LayerNorm,
    deformable: DeformableEncoderLayer,
}

impl EncoderLayer {
    fn create(schema: &mut Schema<'_>, config: &GroundingDinoConfig) -> SingeResult<Self> {
        let width = config.width;
        let eps = config.norm_eps;
        // The fusion and the text's layers take half the encoder's heads and half its inner width.
        let (heads, inner) = (config.encoder_heads / 2, config.encoder_inner / 2);
        Ok(Self {
            vision_norm: schema.layer_norm("fusion.vision_norm", width, eps)?,
            text_norm: schema.layer_norm("fusion.text_norm", width, eps)?,
            fusion: FusedAttention::create(&mut schema.scope("fusion"), width, inner, heads)?,
            vision_scale: schema.parameter("fusion.vision_scale", [width])?,
            text_scale: schema.parameter("fusion.text_scale", [width])?,
            text_attention: Attention::create(
                &mut schema.scope("text.attention"),
                AttentionSpec::square(width, heads),
            )?,
            text_attention_norm: schema.layer_norm("text.attention_norm", width, eps)?,
            text_feed_forward: schema.mlp(
                "text.feed_forward",
                [width, inner, width],
                config.activation,
                true,
            )?,
            text_feed_forward_norm: schema.layer_norm("text.feed_forward_norm", width, eps)?,
            deformable: DeformableEncoderLayer::create(
                &mut schema.scope("deformable"),
                DeformableAttentionSpec {
                    width,
                    heads: config.encoder_heads,
                    levels: config.levels,
                    points: config.encoder_points,
                },
                config.encoder_inner,
                config.activation,
            )?,
        })
    }

    fn forward(
        &self,
        vision: &Tensor,
        vision_positions: &Tensor,
        references: &Tensor,
        levels: &FeaturePyramid,
        text: &Tensor,
        text_positions: &Tensor,
        text_mask: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let vision = vision.apply(&self.vision_norm, state)?;
        let text = text.apply(&self.text_norm, state)?;
        let (read_by_vision, read_by_text) = self.fusion.forward(&vision, &text, state)?;
        let vision = vision.add(&read_by_vision.mul(state.tensor(&self.vision_scale)?)?)?;
        let text = text.add(&read_by_text.mul(state.tensor(&self.text_scale)?)?)?;

        let positioned = text.add(text_positions)?;
        let text = self
            .text_attention
            .attend(
                &positioned,
                &positioned,
                &text,
                Some(text_mask),
                AttentionOptions::default(),
                state,
            )?
            .add(&text)?
            .apply(&self.text_attention_norm, state)?;
        let text = text
            .apply(&self.text_feed_forward, state)?
            .add(&text)?
            .apply(&self.text_feed_forward_norm, state)?;

        let vision =
            self.deformable
                .forward(&vision, vision_positions, references, levels, state)?;
        Ok((vision, text))
    }
}

/// One decoder layer, post-norm: the queries' self-attention, attention to the text, deformable attention to the levels, then a feed-forward network; the queries' positions join their queries everywhere and their keys in the self-attention.
#[derive(Debug, Clone)]
struct DecoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    text_attention: Attention,
    text_attention_norm: LayerNorm,
    cross_attention: DeformableAttention,
    cross_attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl DecoderLayer {
    fn create(schema: &mut Schema<'_>, config: &GroundingDinoConfig) -> SingeResult<Self> {
        let width = config.width;
        let eps = config.norm_eps;
        let spec = AttentionSpec::square(width, config.decoder_heads);
        Ok(Self {
            attention: Attention::create(&mut schema.scope("attention"), spec)?,
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            text_attention: Attention::create(&mut schema.scope("text_attention"), spec)?,
            text_attention_norm: schema.layer_norm("text_attention_norm", width, eps)?,
            cross_attention: DeformableAttention::create(
                &mut schema.scope("cross_attention"),
                DeformableAttentionSpec {
                    width,
                    heads: config.decoder_heads,
                    levels: config.levels,
                    points: config.decoder_points,
                },
            )?,
            cross_attention_norm: schema.layer_norm("cross_attention_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.decoder_inner, width],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        references: &Tensor,
        memory: &Tensor,
        levels: &FeaturePyramid,
        text: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let positioned = hidden.add(positions)?;
        let hidden = self
            .attention
            .forward(&positioned, &positioned, hidden, state)?
            .add(hidden)?
            .apply(&self.attention_norm, state)?;
        let hidden = self
            .text_attention
            .forward(&hidden.add(positions)?, text, text, state)?
            .add(&hidden)?
            .apply(&self.text_attention_norm, state)?;
        let hidden = self
            .cross_attention
            .forward(&hidden.add(positions)?, references, memory, levels, state)?
            .add(&hidden)?
            .apply(&self.cross_attention_norm, state)?;
        hidden
            .apply(&self.feed_forward, state)?
            .add(&hidden)?
            .apply(&self.feed_forward_norm, state)
    }
}

/// Grounding DINO and its head: from images and one text to each query's logit for each token, then its box, `[B, Q, T + 4]`.
#[derive(Debug, Clone)]
pub struct GroundingDinoLayers {
    width: i64,
    temperature: f32,
    queries: i64,
    backbone: Swin,
    backbone_levels: usize,
    stage_norms: Vec<LayerNorm>,
    projections: Vec<ConvGroupNorm>,
    level_embeddings: StateKey,
    text: TextEncoder,
    text_projection: Linear,
    encoder: Vec<EncoderLayer>,
    proposal: Linear,
    proposal_norm: LayerNorm,
    proposal_boxes: LinearStack,
    targets: StateKey,
    query_positions: LinearStack,
    decoder: Vec<DecoderLayer>,
    decoder_norm: LayerNorm,
    boxes: LinearStack,
}

impl GroundingDinoLayers {
    pub fn create(schema: &mut Schema<'_>, config: &GroundingDinoConfig) -> SingeResult<Self> {
        let width = config.width;
        let backbone = Swin::create(&mut schema.scope("backbone"), &config.backbone)?;
        let stages = config.backbone.stages.len();
        let first = stages - config.backbone_levels;
        let channels = |stage: usize| config.backbone.width << stage;
        let stage_norms = (first..stages)
            .map(|stage| {
                schema.layer_norm(
                    &format!("stage_norms.{}", stage - first),
                    channels(stage),
                    config.norm_eps,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let projections = (0..usize::try_from(config.levels).unwrap_or(0))
            .map(|level| {
                let (input, kernel, stride) = match level.cmp(&config.backbone_levels) {
                    Ordering::Less => (channels(first + level), 1, 1),
                    Ordering::Equal => (channels(stages - 1), 3, 2),
                    Ordering::Greater => (width, 3, 2),
                };
                ConvGroupNorm::create(
                    &mut schema.scope(format!("projections.{level}")),
                    [input, width, kernel, stride, PROJECTION_GROUPS],
                    true,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                EncoderLayer::create(&mut schema.scope(format!("encoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                DecoderLayer::create(&mut schema.scope(format!("decoder.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let box_head = |schema: &mut Schema<'_>, scope: &str| {
            LinearStack::create(
                &mut schema.scope(scope),
                [width, width, 4],
                3,
                Activation::Relu,
            )
        };
        Ok(Self {
            width,
            temperature: config.temperature,
            queries: config.queries,
            backbone,
            backbone_levels: config.backbone_levels,
            stage_norms,
            projections,
            level_embeddings: schema.parameter("level_embeddings", [config.levels, width])?,
            text: TextEncoder::create(&mut schema.scope("text"), &config.text)?,
            text_projection: schema.linear("text_projection", config.text.width, width, true)?,
            encoder,
            proposal: schema.linear("proposal", width, width, true)?,
            proposal_norm: schema.layer_norm("proposal_norm", width, config.norm_eps)?,
            proposal_boxes: box_head(schema, "proposal_boxes")?,
            targets: schema.parameter("targets", [config.queries, width])?,
            query_positions: LinearStack::create(
                &mut schema.scope("query_positions"),
                [2 * width, width, width],
                2,
                Activation::Relu,
            )?,
            decoder,
            decoder_norm: schema.layer_norm("decoder_norm", width, config.norm_eps)?,
            boxes: box_head(schema, "boxes")?,
        })
    }

    /// Returns each level's map, `[B, W, H_l, W_l]`.
    fn levels(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let stages = self.backbone.stage_tokens(pixels, state)?;
        let first = stages.len() - self.backbone_levels;
        let maps = stages[first..]
            .iter()
            .zip(&self.stage_norms)
            .map(|((tokens, size), norm)| to_map(&tokens.apply(norm, state)?, *size))
            .collect::<SingeResult<Vec<_>>>()?;
        let mut levels = Vec::with_capacity(self.projections.len());
        for (index, projection) in self.projections.iter().enumerate() {
            let source = match index.checked_sub(self.backbone_levels) {
                None => &maps[index],
                Some(0) => &maps[maps.len() - 1],
                Some(_) => &levels[index - 1],
            };
            let level = source.apply(projection, state)?;
            levels.push(level);
        }
        Ok(levels)
    }
}

impl Layer<GroundingInput> for GroundingDinoLayers {
    fn forward(&self, input: &GroundingInput, state: &ModelState) -> SingeResult<Tensor> {
        let text = self
            .text
            .forward(input, state)?
            .apply(&self.text_projection, state)?;
        let data_type = text.data_type();
        let text_mask = input.mask.cast(data_type)?;
        let text_positions = coordinate_sines(
            &input.positions.cast(DataType::F32)?.unsqueeze(-1)?,
            self.width,
        )?
        .cast(data_type)?;

        // Detections are thresholded scores, which TF32 convolutions move, as in DETR.
        let maps = with_convolution_precision(ComputePrecision::Full, || {
            self.levels(&input.pixels, state)
        })?;
        let level_embeddings = state.tensor(&self.level_embeddings)?;
        let mut tokens = Vec::with_capacity(maps.len());
        let mut positions = Vec::with_capacity(maps.len());
        let mut sizes = Vec::with_capacity(maps.len());
        let mut extents = Vec::with_capacity(maps.len());
        for (index, map) in maps.iter().enumerate() {
            let &[_, _, height, width] = map.shape().dimensions() else {
                return Err(SingeError::invalid_argument(
                    "Grounding DINO level, [B, W, h, w]",
                ));
            };
            let embedding = level_embeddings
                .narrow(0, i64::try_from(index).unwrap_or(i64::MAX), 1)?
                .unsqueeze(0)?;
            let sines = sine_positions(
                map,
                self.width,
                SinePositionSpec {
                    offset: 0.0,
                    temperature: self.temperature,
                },
            )?;
            positions.push(sines.cast(data_type)?.add(&embedding)?);
            tokens.push(to_tokens(map)?);
            sizes.push([map.dimension(2)?, map.dimension(3)?]);
            extents.push([height, width]);
        }

        let positions = Tensor::concat(&positions, 1)?;
        let levels = FeaturePyramid::create(input.pixels.device(), sizes)?;
        let references = levels.reference_points()?.cast(data_type)?;
        let mut vision = Tensor::concat(&tokens, 1)?;
        let mut text = text;
        for layer in &self.encoder {
            (vision, text) = layer.forward(
                &vision,
                &positions,
                &references,
                &levels,
                &text,
                &text_positions,
                &text_mask,
                state,
            )?;
        }

        let (anchors, valid) = anchor_proposals(input.pixels.device(), &extents, f32::INFINITY)?;
        let proposals = vision
            .mul(&valid.cast(data_type)?)?
            .apply(&self.proposal, state)?
            .apply(&self.proposal_norm, state)?;
        let scores = proposals.matmul(&text.transpose(1, 2)?)?;
        let boxes = proposals
            .apply(&self.proposal_boxes, state)?
            .add(&anchors.cast(data_type)?)?;
        let (_, chosen) = scores
            .max_reduce(&[2], false)?
            .topk(self.queries, 1, true, true)?;
        let mut references = gather_rows(&boxes, &chosen)?.apply(&Activation::Sigmoid, state)?;
        let targets = state.tensor(&self.targets)?;
        let mut hidden = targets.unsqueeze(0)?.expand_with(&[
            input.pixels.dimension(0)?,
            targets.dimension(0)?,
            self.width.into(),
        ])?;

        let mut last_references = references.clone();
        for layer in &self.decoder {
            let positions = coordinate_sines(&references, self.width / 2)?
                .apply(&self.query_positions, state)?;
            hidden = layer.forward(
                &hidden,
                &positions,
                &references,
                &vision,
                &levels,
                &text,
                state,
            )?;
            last_references = references;
            references = hidden
                .apply(&self.boxes, state)?
                .add(&logit(&last_references, LOGIT_EPS)?)?
                .apply(&Activation::Sigmoid, state)?;
        }

        let normed = hidden.apply(&self.decoder_norm, state)?;
        let logits = normed.matmul(&text.transpose(1, 2)?)?;
        let boxes = normed
            .apply(&self.boxes, state)?
            .add(&logit(&last_references, LOGIT_EPS)?)?
            .apply(&Activation::Sigmoid, state)?;
        Tensor::concat(&[logits, boxes], 2)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "grounding_dino",
    formats: &[(Format::Transformers, &["grounding-dino"])],
    loaders: &[HeadLoader::ZeroShotObjectDetection(load_detector)],
};

pub fn load_detector(loader: &mut Loader<'_>) -> Result<OpenVocabularyDetector> {
    let config: GroundingDinoConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        GroundingDinoLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Layer<GroundingInput>>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config))?;
    Ok(OpenVocabularyDetector::Grounded {
        model,
        special: SPECIAL_TOKENS.to_vec(),
    })
}
