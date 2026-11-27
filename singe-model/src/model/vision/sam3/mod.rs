//! SAM 3 for images with text prompts, as `Sam3Model` finds a concept: a vision transformer whose layers attend within windows or globally under axial rotary positions, a pyramid of its map at four, two, and one times its grid, and CLIP's text tower projected to the detector's width.
//! A DETR encoder fuses the coarsest level with the text, and a decoder refines a presence token and a set of queries, each with a box whose relative positions bias its attention to the image; each query is scored against the pooled text and masked through a pixel decoder over the pyramid.
//! Box prompts, which `Sam3GeometryEncoder` reads, are not supported.

mod transformers;

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::capture,
    nn::{
        Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, GroupNorm, Layer,
        LayerNorm, Linear, Mlp, ModelState, StateKey, TokenBatch,
    },
    shape::spatial::{Padding2d, Spatial2d},
    tensor::{AttentionOptions, InterpolateMode},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, AxialRotary, LinearStack, SinePositions, inverse_sigmoid,
        merge_heads, merge_windows, partition_windows, sine_positions, split_heads, to_map,
        to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ConceptLayers, ConceptSegmenter, HeadLoader},
    model::{
        ModelEntry,
        multimodal::clip::{ClipText, ClipTextConfig},
    },
    schema::Schema,
    source::Format,
};

/// The epsilon of the layer norms `PyTorch`'s default gives the detector, its scoring, and its mask head.
const NORM_EPS: f32 = 1e-5;

/// The base of the sine positions' periods.
const TEMPERATURE: f32 = 10000.0;

/// The bound on the presence logit, as `clamp_presence_logit_max_val` sets it.
const PRESENCE_BOUND: f32 = 10.0;

/// The bound on the query logits, as `clamp_max_val` sets it.
const SCORE_BOUND: f32 = 12.0;

/// The smallest probability `inverse_sigmoid` reads, either way of a box coordinate.
const LEAST_PROBABILITY: f32 = 1e-3;

/// The groups of the pixel decoder's norms.
const PIXEL_GROUPS: i64 = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sam3VitConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub patch: i64,
    pub image_size: i64,
    /// The side of the grid the learned positions cover, tiled over the image's.
    pub pretrain_side: i64,
    pub window: i64,
    /// The layers that attend over the whole grid.
    pub global: Vec<usize>,
    pub theta: f32,
    pub channels: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sam3Config {
    pub vit: Sam3VitConfig,
    pub text: ClipTextConfig,
    /// The width of the pyramid, the detector, and the mask head.
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub encoder_layers: i64,
    pub encoder_activation: Activation,
    pub decoder_layers: i64,
    pub decoder_inner: i64,
    pub decoder_activation: Activation,
    pub queries: i64,
    pub mask_heads: i64,
}

/// A vision transformer layer: attention within windows, or over the grid, under axial rotary positions, and a feed-forward, each normed before and added back.
#[derive(Debug, Clone)]
struct VitLayer {
    attention_norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    window: Option<i64>,
}

impl VitLayer {
    /// Attends over `tokens`, `[N, T, C]`, turned by `rotary`.
    fn attend(
        &self,
        tokens: &Tensor,
        rotary: &AxialRotary,
        heads: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let head = |linear: &Linear| split_heads(&tokens.apply(linear, state)?, heads);
        let query = rotary.rotated(&head(&self.query)?)?;
        let key = rotary.rotated(&head(&self.key)?)?;
        let attended = query.attend(
            &key,
            &head(&self.value)?,
            None,
            None,
            AttentionOptions::default(),
        )?;
        merge_heads(&attended)?.apply(&self.output, state)
    }

    /// Runs `hidden`, `[1, G, G, C]`.
    fn forward(
        &self,
        hidden: &Tensor,
        rotary: [&AxialRotary; 2],
        heads: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, height, width, channels] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 3's grid, [B, G, G, C]"));
        };
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended = match self.window {
            Some(window) => merge_windows(
                &self.attend(
                    &partition_windows(&normed, window)?,
                    rotary[0],
                    heads,
                    state,
                )?,
                window,
                batch,
                (height, width),
            )?,
            None => self
                .attend(
                    &normed.reshape([batch, height * width, channels])?,
                    rotary[1],
                    heads,
                    state,
                )?
                .reshape([batch, height, width, channels])?,
        };
        let hidden = hidden.add(&attended)?;
        hidden.add(
            &hidden
                .apply(&self.feed_forward_norm, state)?
                .apply(&self.feed_forward, state)?,
        )
    }
}

/// One pyramid level: transposed convolutions up from the grid, with GELU between two, or none, then a 1 × 1 and a 3 × 3 convolution.
#[derive(Debug, Clone)]
struct FpnLevel {
    upsampling: Vec<ConvTranspose2d>,
    project: Conv2d,
    smooth: Conv2d,
}

impl Layer for FpnLevel {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for (index, upsampling) in self.upsampling.iter().enumerate() {
            if index > 0 {
                hidden = hidden.apply(&Activation::GeluExact, state)?;
            }

            hidden = hidden.apply(upsampling, state)?;
        }
        hidden
            .apply(&self.project, state)?
            .apply(&self.smooth, state)
    }
}

/// A DETR encoder layer: self-attention with the positions on the queries and keys, attention to the text, and a feed-forward, each normed before and added back.
#[derive(Debug, Clone)]
struct EncoderLayer {
    norms: [LayerNorm; 3],
    self_attention: Attention,
    cross_attention: Attention,
    feed_forward: Mlp,
}

/// A DETR decoder layer: self-attention, attention to the text, and attention to the image biased by each query's box, with the query positions on the queries, and a feed-forward, each added back and normed after.
#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attention: Attention,
    text_attention: Attention,
    vision_attention: Attention,
    feed_forward: Mlp,
    norms: [LayerNorm; 4],
}

#[derive(Debug, Clone)]
struct Decoder {
    layers: Vec<DecoderLayer>,
    output_norm: LayerNorm,
    box_head: LinearStack,
    queries: StateKey,
    reference_points: StateKey,
    presence_token: StateKey,
    presence_head: LinearStack,
    presence_norm: LayerNorm,
    reference_head: LinearStack,
    box_bias_x: LinearStack,
    box_bias_y: LinearStack,
}

#[derive(Debug, Clone)]
struct Scoring {
    text_mlp: LinearStack,
    text_norm: LayerNorm,
    text_projection: Linear,
    query_projection: Linear,
}

#[derive(Debug, Clone)]
struct MaskHead {
    prompt_attention: Attention,
    prompt_norm: LayerNorm,
    convolutions: Vec<Conv2d>,
    norms: Vec<GroupNorm>,
    instance_projection: Conv2d,
    embedder: LinearStack,
}

#[derive(Debug, Clone)]
pub struct Sam3Layers {
    patches: Conv2d,
    positions: StateKey,
    vit_norm: LayerNorm,
    vit_layers: Vec<VitLayer>,
    fpn: Vec<FpnLevel>,
    text: ClipText,
    text_projection: Linear,
    encoder: Vec<EncoderLayer>,
    decoder: Decoder,
    scoring: Scoring,
    mask: MaskHead,
    config: Sam3Config,
}

/// Returns a `[1, N]` constant of `values` on `like`'s device, captured as a literal.
fn literal(like: &Tensor, values: Vec<f32>) -> SingeResult<Tensor> {
    let count = i64::try_from(values.len()).map_err(|_| SingeError::out_of_range("constants"))?;
    capture::literal(like.device(), |device| {
        Tensor::from_vec(device, Shape::create([1, count])?, values)
    })
}

impl Sam3Layers {
    pub fn create(schema: &mut Schema<'_>, config: &Sam3Config) -> SingeResult<Self> {
        let vit = &config.vit;
        let width = config.width;
        let mut backbone = schema.scope("vision_encoder.backbone");
        let patches = backbone.conv2d(
            "patches",
            vit.channels,
            vit.width,
            vit.patch,
            Conv2dConfig {
                stride: Spatial2d::square(vit.patch),
                ..Conv2dConfig::default()
            },
            false,
        )?;
        let positions = backbone.parameter(
            "positions",
            [1, vit.pretrain_side * vit.pretrain_side, vit.width],
        )?;
        let vit_norm = backbone.layer_norm("layer_norm", vit.width, vit.eps)?;
        let vit_layers = (0..vit.layers)
            .map(|index| {
                let mut layer = backbone.scope(format!("layers.{index}"));
                let [query, key, value, output] = layer
                    .scope("attention")
                    .attention_projections(vit.width, [true; 4])?;
                Ok(VitLayer {
                    attention_norm: layer.layer_norm("layer_norm1", vit.width, vit.eps)?,
                    query,
                    key,
                    value,
                    output,
                    feed_forward_norm: layer.layer_norm("layer_norm2", vit.width, vit.eps)?,
                    feed_forward: layer.mlp(
                        "mlp",
                        [vit.width, vit.inner, vit.width],
                        vit.activation,
                        true,
                    )?,
                    window: (!vit.global.contains(&(index as usize))).then_some(vit.window),
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let doubling = ConvTranspose2dConfig {
            stride: Spatial2d::square(2),
            ..ConvTranspose2dConfig::default()
        };
        let fpn = (0..3_i64)
            .map(|level| {
                let mut scope = schema.scope(format!("vision_encoder.neck.fpn_layers.{level}"));
                // Four times the grid takes two doublings, the second at index 2 after the GELU.
                let doublings = 2 - level;
                let mut channels = vit.width;
                let mut upsampling = Vec::new();
                for step in 0..doublings {
                    upsampling.push(scope.conv_transpose2d(
                        &format!("scale_layers.{}", 2 * step),
                        channels,
                        channels / 2,
                        2,
                        doubling,
                        true,
                    )?);
                    channels /= 2;
                }
                Ok(FpnLevel {
                    upsampling,
                    project: scope.conv2d(
                        "proj1",
                        channels,
                        width,
                        1,
                        Conv2dConfig::default(),
                        true,
                    )?,
                    smooth: scope.conv2d(
                        "proj2",
                        width,
                        width,
                        3,
                        Conv2dConfig {
                            padding: Padding2d::symmetric(1, 1),
                            ..Conv2dConfig::default()
                        },
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let square = AttentionSpec::square(width, config.heads);
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                let mut layer = schema.scope(format!("detr_encoder.layers.{index}"));
                Ok(EncoderLayer {
                    norms: [
                        layer.layer_norm("layer_norm1", width, NORM_EPS)?,
                        layer.layer_norm("layer_norm2", width, NORM_EPS)?,
                        layer.layer_norm("layer_norm3", width, NORM_EPS)?,
                    ],
                    self_attention: Attention::create(&mut layer.scope("self_attn"), square)?,
                    cross_attention: Attention::create(&mut layer.scope("cross_attn"), square)?,
                    feed_forward: layer.mlp(
                        "mlp",
                        [width, config.inner, width],
                        config.encoder_activation,
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let mut scope = schema.scope("detr_decoder");
        let relu_stack = |scope: &mut Schema<'_>, name: &str, sizes: [i64; 3], depth: i64| {
            LinearStack::create(&mut scope.scope(name), sizes, depth, Activation::Relu)
        };
        let decoder = Decoder {
            layers: (0..config.decoder_layers)
                .map(|index| {
                    let mut layer = scope.scope(format!("layers.{index}"));
                    Ok(DecoderLayer {
                        self_attention: Attention::create(&mut layer.scope("self_attn"), square)?,
                        text_attention: Attention::create(
                            &mut layer.scope("text_cross_attn"),
                            square,
                        )?,
                        vision_attention: Attention::create(
                            &mut layer.scope("vision_cross_attn"),
                            square,
                        )?,
                        feed_forward: layer.mlp(
                            "mlp",
                            [width, config.decoder_inner, width],
                            config.decoder_activation,
                            true,
                        )?,
                        norms: [
                            layer.layer_norm("self_attn_layer_norm", width, NORM_EPS)?,
                            layer.layer_norm("text_cross_attn_layer_norm", width, NORM_EPS)?,
                            layer.layer_norm("vision_cross_attn_layer_norm", width, NORM_EPS)?,
                            layer.layer_norm("mlp_layer_norm", width, NORM_EPS)?,
                        ],
                    })
                })
                .collect::<SingeResult<Vec<_>>>()?,
            output_norm: scope.layer_norm("output_layer_norm", width, NORM_EPS)?,
            box_head: relu_stack(&mut scope, "box_head", [width, width, 4], 3)?,
            queries: scope.parameter("query_embed.weight", [config.queries, width])?,
            reference_points: scope.parameter("reference_points.weight", [config.queries, 4])?,
            presence_token: scope.parameter("presence_token.weight", [1, width])?,
            presence_head: relu_stack(&mut scope, "presence_head", [width, width, 1], 3)?,
            presence_norm: scope.layer_norm("presence_layer_norm", width, NORM_EPS)?,
            reference_head: relu_stack(&mut scope, "ref_point_head", [2 * width, width, width], 2)?,
            box_bias_x: relu_stack(&mut scope, "box_rpb_embed_x", [2, width, config.heads], 2)?,
            box_bias_y: relu_stack(&mut scope, "box_rpb_embed_y", [2, width, config.heads], 2)?,
        };
        let mut scope = schema.scope("dot_product_scoring");
        let scoring = Scoring {
            text_mlp: LinearStack::create(
                &mut scope.scope("text_mlp"),
                [width, config.decoder_inner, width],
                2,
                Activation::Relu,
            )?,
            text_norm: scope.layer_norm("text_mlp_out_norm", width, NORM_EPS)?,
            text_projection: scope.linear("text_proj", width, width, true)?,
            query_projection: scope.linear("query_proj", width, width, true)?,
        };
        let mut scope = schema.scope("mask_decoder");
        let same = Conv2dConfig {
            padding: Padding2d::symmetric(1, 1),
            ..Conv2dConfig::default()
        };
        let mask = MaskHead {
            prompt_attention: Attention::create(
                &mut scope.scope("prompt_cross_attn"),
                AttentionSpec::square(width, config.mask_heads),
            )?,
            prompt_norm: scope.layer_norm("prompt_cross_attn_norm", width, NORM_EPS)?,
            convolutions: (0..3)
                .map(|index| {
                    scope.conv2d(
                        &format!("pixel_decoder.conv_layers.{index}"),
                        width,
                        width,
                        3,
                        same,
                        true,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            norms: (0..3)
                .map(|index| {
                    scope.group_norm(
                        &format!("pixel_decoder.norms.{index}"),
                        width,
                        PIXEL_GROUPS,
                        NORM_EPS,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            instance_projection: scope.conv2d(
                "instance_projection",
                width,
                width,
                1,
                Conv2dConfig::default(),
                true,
            )?,
            embedder: LinearStack::create(
                &mut scope.scope("mask_embedder"),
                [width; 3],
                3,
                Activation::Relu,
            )?,
        };
        Ok(Self {
            patches,
            positions,
            vit_norm,
            vit_layers,
            fpn,
            text: ClipText::create(&mut schema.scope("text"), &config.text)?,
            text_projection: schema.linear(
                "text_projection",
                config.text.encoder.width,
                width,
                true,
            )?,
            encoder,
            decoder,
            scoring,
            mask,
            config: config.clone(),
        })
    }

    /// Returns the vision transformer's map, `[1, C, G, G]`.
    fn vit(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let vit = &self.config.vit;
        let map = pixels.apply(&self.patches, state)?.permute(&[0, 2, 3, 1])?;
        let &[batch, grid, _, channels] = map.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 3's patches"));
        };
        // The learned positions tiled over the grid, as `_tile_position_embeddings` repeats and cuts them.
        let side = vit.pretrain_side;
        let repeats = grid / side + 1;
        let tile = state
            .tensor(&self.positions)?
            .reshape([1, side, side, channels])?;
        let rows = Tensor::concat(&vec![tile; repeats as usize], 1)?;
        let tiled = Tensor::concat(&vec![rows; repeats as usize], 2)?
            .narrow(1, 0, grid)?
            .narrow(2, 0, grid)?;
        let mut hidden = map.add(&tiled)?.apply(&self.vit_norm, state)?;
        let head = vit.width / vit.heads;
        let windowed = AxialRotary::create(vit.window, head, vit.theta, 1.0, &hidden)?;
        let global = AxialRotary::create(
            grid,
            head,
            vit.theta,
            vit.window as f32 / grid as f32,
            &hidden,
        )?;
        for layer in &self.vit_layers {
            hidden = layer.forward(&hidden, [&windowed, &global], vit.heads, state)?;
        }
        hidden
            .reshape([batch, grid, grid, channels])?
            .permute(&[0, 3, 1, 2])
    }

    /// Returns `encode_boxes` of boxes, `[Q, 4]` as centre, width, and height: each coordinate's sine positions, the vertical centre's first, `[Q, 2W]`.
    fn box_positions(&self, boxes: &Tensor) -> SingeResult<Tensor> {
        let half = self.config.width / 2;
        let periods = (0..half / 2)
            .map(|pair| TEMPERATURE.powf((2 * pair) as f32 / half as f32))
            .collect();
        let count = boxes.shape().dimensions()[0];
        let angles = boxes
            .mul_scalar(2.0 * PI)?
            .unsqueeze(2)?
            .div(&literal(boxes, periods)?.unsqueeze(0)?)?;
        let encoded =
            Tensor::stack(&[angles.sin()?, angles.cos()?], -1)?.reshape([count, 4, half])?;
        let order = capture::literal(boxes.device(), |device| {
            Tensor::from_vec(device, Shape::create([4])?, vec![1_i64, 0, 2, 3])
        })?;
        encoded.index_select(1, &order)?.reshape([count, 4 * half])
    }

    /// Returns `_get_rpb_matrix`'s bias for each query's box over the `grid × grid` image, with a row of zeros for the presence token first, `[1, H, 1 + Q, G²]`.
    fn box_bias(&self, boxes: &Tensor, grid: i64, state: &ModelState) -> SingeResult<Tensor> {
        let decoder = &self.decoder;
        let count = boxes.shape().dimensions()[0];
        let (centres, sizes) = (
            boxes.narrow(1, 0, 2)?,
            boxes.narrow(1, 2, 2)?.mul_scalar(0.5)?,
        );
        let (low, high) = (centres.sub(&sizes)?, centres.add(&sizes)?);
        let coordinates = Tensor::range(boxes.device(), 0_i64, grid)?
            .cast(boxes.data_type())?
            .div_scalar(grid as f64)?
            .reshape([1, grid, 1])?;
        let log8 = 8.0_f64.ln();
        let offsets = |axis: i64, stack: &LinearStack| -> SingeResult<Tensor> {
            let edges = Tensor::concat(&[low.narrow(1, axis, 1)?, high.narrow(1, axis, 1)?], 1)?;
            let scaled = coordinates.sub(&edges.unsqueeze(1)?)?.mul_scalar(8.0)?;
            scaled
                .sign()?
                .mul(&scaled.abs()?.add_scalar(1.0)?.log()?.div_scalar(log8)?)?
                .apply(stack, state)
        };
        let horizontal = offsets(0, &decoder.box_bias_x)?;
        let vertical = offsets(1, &decoder.box_bias_y)?;
        let heads = self.config.heads;
        let bias = vertical
            .unsqueeze(2)?
            .add(&horizontal.unsqueeze(1)?)?
            .reshape([count, grid * grid, heads])?
            .permute(&[2, 0, 1])?;
        let presence = bias.narrow(1, 0, 1)?.mul_scalar(0.0)?;
        Tensor::concat(&[presence, bias], 1)?.unsqueeze(0)
    }

    /// Runs the decoder over the encoder's states, `[1, G², W]`, and returns the queries' normed states, `[1, Q, W]`, and the presence logit, `[1]`.
    fn decode(
        &self,
        image: &Tensor,
        positions: &Tensor,
        text: &Tensor,
        grid: i64,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let decoder = &self.decoder;
        let queries = state.tensor(&decoder.queries)?;
        let mut boxes = state.tensor(&decoder.reference_points)?.sigmoid()?;
        let mut hidden = Tensor::concat(
            &[
                state.tensor(&decoder.presence_token)?.clone(),
                queries.clone(),
            ],
            0,
        )?
        .unsqueeze(0)?;
        let keys = image.add(positions)?;
        let mut outputs = None;
        for layer in &decoder.layers {
            let query_positions = self
                .box_positions(&boxes)?
                .apply(&decoder.reference_head, state)?;
            let query_positions = Tensor::concat(
                &[
                    query_positions.narrow(0, 0, 1)?.mul_scalar(0.0)?,
                    query_positions,
                ],
                0,
            )?
            .unsqueeze(0)?;
            let bias = self.box_bias(&boxes, grid, state)?;
            let [self_norm, text_norm, vision_norm, feed_forward_norm] = &layer.norms;
            let positioned = hidden.add(&query_positions)?;
            hidden = hidden
                .add(
                    &layer
                        .self_attention
                        .forward(&positioned, &positioned, &hidden, state)?,
                )?
                .apply(self_norm, state)?;
            hidden = hidden
                .add(&layer.text_attention.forward(
                    &hidden.add(&query_positions)?,
                    text,
                    text,
                    state,
                )?)?
                .apply(text_norm, state)?;
            hidden = hidden
                .add(&layer.vision_attention.attend(
                    &hidden.add(&query_positions)?,
                    &keys,
                    image,
                    Some(&bias),
                    AttentionOptions::default(),
                    state,
                )?)?
                .apply(vision_norm, state)?;
            hidden = hidden
                .add(&hidden.apply(&layer.feed_forward, state)?)?
                .apply(feed_forward_norm, state)?;
            let normed = hidden
                .narrow(1, 1, self.config.queries)?
                .apply(&decoder.output_norm, state)?;
            boxes = normed
                .squeeze(0)?
                .apply(&decoder.box_head, state)?
                .add(&inverse_sigmoid(&boxes, LEAST_PROBABILITY)?)?
                .sigmoid()?;
            let presence = hidden
                .narrow(1, 0, 1)?
                .apply(&decoder.presence_norm, state)?
                .apply(&decoder.presence_head, state)?
                .reshape([1])?
                .clamp(-PRESENCE_BOUND, PRESENCE_BOUND)?;
            outputs = Some((normed, presence));
        }
        outputs.ok_or_else(|| SingeError::invalid_argument("a SAM 3 decoder without layers"))
    }

    /// Returns each query's logit against the pooled text, `[Q]`, as `Sam3DotProductScoring` gives it.
    fn score(&self, queries: &Tensor, text: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let scoring = &self.scoring;
        let pooled = text
            .apply(&scoring.text_mlp, state)?
            .add(text)?
            .apply(&scoring.text_norm, state)?
            .mean(&[1], false)?
            .apply(&scoring.text_projection, state)?;
        let scale = 1.0 / (self.config.width as f64).sqrt();
        queries
            .squeeze(0)?
            .apply(&scoring.query_projection, state)?
            .matmul(&pooled.transpose(0, 1)?)?
            .mul_scalar(scale)?
            .clamp(-SCORE_BOUND, SCORE_BOUND)?
            .squeeze(1)
    }

    /// Returns each query's mask logits at the finest level, `[Q, h, w]`, from the pyramid's two finer levels and the encoder's states, which attend to the text first.
    fn masks(
        &self,
        queries: &Tensor,
        levels: [&Tensor; 2],
        encoded: &Tensor,
        text: &Tensor,
        grid: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mask = &self.mask;
        let normed = encoded.apply(&mask.prompt_norm, state)?;
        let encoded = encoded.add(&mask.prompt_attention.forward(&normed, text, text, state)?)?;
        let mut hidden = to_map(&encoded, (grid, grid))?;
        for (index, finer) in levels.iter().rev().enumerate() {
            let &[_, _, height, width] = finer.shape().dimensions() else {
                return Err(SingeError::invalid_argument("a SAM 3 pyramid level"));
            };
            hidden = hidden
                .interpolate(Spatial2d::new(height, width), InterpolateMode::Nearest)?
                .add(finer)?
                .apply(&mask.convolutions[index], state)?
                .apply(&mask.norms[index], state)?
                .relu()?;
        }

        let instances = hidden.apply(&mask.instance_projection, state)?;
        let &[_, channels, height, width] = instances.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 3's instance embedding"));
        };
        queries
            .squeeze(0)?
            .apply(&mask.embedder, state)?
            .matmul(&instances.reshape([channels, height * width])?)?
            .reshape([self.config.queries, height, width])
    }
}

impl ConceptLayers for Sam3Layers {
    fn segment(
        &self,
        pixels: &Tensor,
        tokens: &TokenBatch,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor, Tensor)> {
        let map = self.vit(pixels, state)?;
        let levels = self
            .fpn
            .iter()
            .map(|level| map.apply(level, state))
            .collect::<SingeResult<Vec<_>>>()?;
        let [finest, finer, coarsest] = levels.as_slice() else {
            return Err(SingeError::invalid_argument("SAM 3's three pyramid levels"));
        };
        let grid = coarsest.shape().dimensions()[2];
        let positions = sine_positions(coarsest, self.config.width, SinePositions::DETR)?
            .cast(coarsest.data_type())?;
        let text = self
            .text
            .forward(tokens, state)?
            .apply(&self.text_projection, state)?;
        let mut encoded = to_tokens(coarsest)?;
        for layer in &self.encoder {
            let [first, second, third] = &layer.norms;
            let normed = encoded.apply(first, state)?;
            let positioned = normed.add(&positions)?;
            encoded = encoded.add(&layer.self_attention.forward(
                &positioned,
                &positioned,
                &normed,
                state,
            )?)?;
            let normed = encoded.apply(second, state)?;
            encoded = encoded.add(
                &layer
                    .cross_attention
                    .forward(&normed, &text, &text, state)?,
            )?;
            encoded = encoded.add(
                &encoded
                    .apply(third, state)?
                    .apply(&layer.feed_forward, state)?,
            )?;
        }

        let (queries, presence) = self.decode(&encoded, &positions, &text, grid, state)?;
        let logits = self.score(&queries, &text, state)?;
        let masks = self.masks(&queries, [finest, finer], &encoded, &text, grid, state)?;
        Ok((logits, presence, masks))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "sam3",
    formats: &[(Format::Transformers, &["sam3"])],
    loaders: &[HeadLoader::ConceptSegmentation(load_segmenter)],
};

pub fn load_segmenter(loader: &mut Loader<'_>) -> Result<ConceptSegmenter> {
    let config: Sam3Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Sam3Layers::create(schema, &config)
            .map(|layers| Boxed::<dyn ConceptLayers>(Box::new(layers)))
    })?;
    Ok(ConceptSegmenter::new(
        loader.load_whole(&model, &transformers::rules())?,
    ))
}
