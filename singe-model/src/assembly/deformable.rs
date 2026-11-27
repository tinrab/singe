//! Multi-scale deformable attention and what the DETR family builds around it: the feature levels it samples, their reference points, and the post-norm encoder and decoder layers.

use std::slice;

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{DimensionExpr, capture},
    nn::{Activation, LayerNorm, Linear, Mlp, ModelState},
};

use crate::{
    assembly::{Attention, AttentionSpec, PYTORCH_NORM_EPS, attention_rules},
    format::{Rules, join},
    schema::Schema,
};

/// The sizes of a [`DeformableAttention`]: `width` features in `heads` heads, each sampling `points` points in each of `levels` levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeformableAttentionSpec {
    pub width: i64,
    pub heads: i64,
    pub levels: i64,
    pub points: i64,
}

/// The feature levels deformable attention samples, whose maps are flattened and concatenated into one sequence.
/// Their sizes are a program's dimensions, so a program with run-time axes follows each image's.
#[derive(Debug, Clone)]
pub struct FeatureLevels {
    sizes: Vec<[DimensionExpr; 2]>,
    /// Each level's height and width, `[L, 2]` in `I64`.
    shapes: Tensor,
    /// Where each level starts in the sequence, `[L]` in `I64`.
    starts: Tensor,
    /// Each level's width and height, `[L, 1, 2]` in `F32`, which divide the sampling offsets.
    normalizer: Tensor,
}

impl FeatureLevels {
    /// Describes levels of `sizes`, each its height and width.
    pub fn create(device: Device, sizes: Vec<[DimensionExpr; 2]>) -> SingeResult<Self> {
        let count = i64::try_from(sizes.len()).map_err(|_| SingeError::out_of_range("levels"))?;
        let shapes =
            Tensor::from_dimensions(device, sizes.concat().as_slice())?.reshape([count, 2])?;
        let mut starts = vec![DimensionExpr::from(0)];
        for [height, width] in &sizes[..sizes.len().saturating_sub(1)] {
            let last = starts[starts.len() - 1].clone();
            starts.push(last + height.clone() * width.clone());
        }

        let starts = Tensor::from_dimensions(device, &starts)?;
        let flipped = sizes
            .iter()
            .flat_map(|[height, width]| [width.clone(), height.clone()])
            .collect::<Vec<_>>();
        let normalizer = Tensor::from_dimensions(device, &flipped)?
            .cast(DataType::F32)?
            .reshape([count, 1, 2])?;
        Ok(Self {
            sizes,
            shapes,
            starts,
            normalizer,
        })
    }

    /// Returns the centre of each level's every point, normalized `(x, y)`, as `[1, S, 2]` in `F32`, as `get_reference_points` places the encoder's with every valid ratio one.
    pub fn reference_points(&self) -> SingeResult<Tensor> {
        let device = self.shapes.device();
        let centres = |extent: &DimensionExpr| -> SingeResult<Tensor> {
            let length =
                Tensor::from_dimensions(device, slice::from_ref(extent))?.cast(DataType::F32)?;
            Tensor::range(device, 0_i64, extent.clone())?
                .cast(DataType::F32)?
                .add_scalar(0.5)?
                .div(&length)
        };
        let levels = self
            .sizes
            .iter()
            .map(|[height, width]| {
                let grid = [height.clone(), width.clone()];
                let x = centres(width)?.unsqueeze(0)?.expand_with(&grid)?;
                let y = centres(height)?.unsqueeze(1)?.expand_with(&grid)?;
                Tensor::stack(&[x, y], -1)?.flatten(0, 1)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Tensor::concat(&levels, 0)?.unsqueeze(0)
    }
}

/// Multi-scale deformable attention, as `DeformableDetrMultiscaleDeformableAttention` computes it: each query predicts, per head, offsets around its reference point at each level and a softmax of weights over every level's points, and takes the weighted sum of the projected memory sampled bilinearly there.
#[derive(Debug, Clone)]
pub struct DeformableAttention {
    spec: DeformableAttentionSpec,
    offsets: Linear,
    weights: Linear,
    value: Linear,
    output: Linear,
}

impl DeformableAttention {
    /// Registers the projections as `offsets`, `weights`, `value`, and `output`.
    pub fn create(schema: &mut Schema<'_>, spec: DeformableAttentionSpec) -> SingeResult<Self> {
        let samples = spec.heads * spec.levels * spec.points;
        Ok(Self {
            spec,
            offsets: schema.linear("offsets", spec.width, samples * 2, true)?,
            weights: schema.linear("weights", spec.width, samples, true)?,
            value: schema.linear("value", spec.width, spec.width, true)?,
            output: schema.linear("output", spec.width, spec.width, true)?,
        })
    }

    /// Attends from `query`, `[B, Q, W]` with any positions added, around `references`, the same at every level, to `memory`, `[B, S, W]`, the flattened `levels`.
    /// The references are `[B or 1, Q, 2]` normalized points `(x, y)`, which the offsets move in each level's pixels, or `[B, Q, 4]` boxes `(cx, cy, w, h)`, which they move by `offset / points` times half the box, as `transformers` scales them.
    pub fn forward(
        &self,
        query: &Tensor,
        references: &Tensor,
        memory: &Tensor,
        levels: &FeatureLevels,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let DeformableAttentionSpec {
            width,
            heads,
            levels: count,
            points,
        } = self.spec;
        let (batch, queries) = (query.dimension(0)?, query.dimension(1)?);
        let value = memory.apply(&self.value, state)?.reshape_with(&[
            memory.dimension(0)?,
            memory.dimension(1)?,
            heads.into(),
            (width / heads).into(),
        ])?;
        let sampled = [
            batch.clone(),
            queries.clone(),
            heads.into(),
            count.into(),
            points.into(),
        ];
        let offsets = query
            .apply(&self.offsets, state)?
            .reshape_with(&[sampled.as_slice(), &[2.into()]].concat())?;
        let weights = query
            .apply(&self.weights, state)?
            .reshape_with(&[batch, queries, heads.into(), (count * points).into()])?
            .softmax(-1)?
            .reshape_with(&sampled)?;
        let references = references.unsqueeze(2)?.unsqueeze(2)?.unsqueeze(2)?;
        let locations = match references.shape().dimensions().last() {
            Some(2) => {
                references.add(&offsets.div(&levels.normalizer.cast(offsets.data_type())?)?)?
            }
            Some(4) => references.narrow(-1, 0, 2)?.add(
                &offsets
                    .div_scalar(points as f64)?
                    .mul(&references.narrow(-1, 2, 2)?)?
                    .mul_scalar(0.5)?,
            )?,
            _ => {
                return Err(SingeError::invalid_argument(
                    "deformable attention references, (x, y) or (cx, cy, w, h)",
                ));
            }
        };
        value
            .deformable_attention(&levels.shapes, &levels.starts, &locations, &weights)?
            .apply(&self.output, state)
    }
}

/// Renames a checkpoint's deformable attention under `from` to [`DeformableAttention`]'s names under `to`.
#[must_use]
pub fn deformable_attention_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("sampling_offsets", "offsets"),
            ("attention_weights", "weights"),
            ("value_proj", "value"),
            ("output_proj", "output"),
        ],
    )
}

/// A post-norm encoder layer of deformable self-attention over the levels, then a feed-forward network, as `DeformableDetrEncoderLayer` is.
#[derive(Debug, Clone)]
pub struct DeformableEncoderLayer {
    attention: DeformableAttention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl DeformableEncoderLayer {
    /// Registers the layer as `attention`, `attention_norm`, `feed_forward` of `inner` hidden features, and `feed_forward_norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        spec: DeformableAttentionSpec,
        inner: i64,
        activation: Activation,
    ) -> SingeResult<Self> {
        let width = spec.width;
        Ok(Self {
            attention: DeformableAttention::create(&mut schema.scope("attention"), spec)?,
            attention_norm: schema.layer_norm("attention_norm", width, PYTORCH_NORM_EPS)?,
            feed_forward: schema.mlp("feed_forward", [width, inner, width], activation, true)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, PYTORCH_NORM_EPS)?,
        })
    }

    /// Runs the layer over `hidden`, `[B, S, W]`, whose queries carry `positions` and sample around `references`.
    pub fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        references: &Tensor,
        levels: &FeatureLevels,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self
            .attention
            .forward(&hidden.add(positions)?, references, hidden, levels, state)?
            .add(hidden)?
            .apply(&self.attention_norm, state)?;
        hidden
            .apply(&self.feed_forward, state)?
            .add(&hidden)?
            .apply(&self.feed_forward_norm, state)
    }
}

/// Renames a checkpoint's `DeformableDetrEncoderLayer`s under `from` to [`DeformableEncoderLayer`]'s names under `to`, both ending in the layer index's `*`.
#[must_use]
pub fn deformable_encoder_rules(from: &str, to: &str) -> Rules {
    deformable_attention_rules(&join(from, "self_attn"), &join(to, "attention")).rename_each(
        from,
        to,
        &[
            ("self_attn_layer_norm", "attention_norm"),
            ("fc1", "feed_forward.up"),
            ("fc2", "feed_forward.down"),
            ("final_layer_norm", "feed_forward_norm"),
        ],
    )
}

/// A post-norm decoder layer, as `DeformableDetrDecoderLayer` is: self-attention of the queries, whose positions join their queries and keys, deformable attention from the queries to the levels, then a feed-forward network.
#[derive(Debug, Clone)]
pub struct DeformableDecoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    cross_attention: DeformableAttention,
    cross_attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

impl DeformableDecoderLayer {
    /// Registers the layer as `attention`, `attention_norm`, `cross_attention`, `cross_attention_norm`, `feed_forward` of `inner` hidden features, and `feed_forward_norm`, each norm with `eps`.
    pub fn create(
        schema: &mut Schema<'_>,
        spec: DeformableAttentionSpec,
        inner: i64,
        activation: Activation,
        eps: f32,
    ) -> SingeResult<Self> {
        let width = spec.width;
        Ok(Self {
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, spec.heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            cross_attention: DeformableAttention::create(
                &mut schema.scope("cross_attention"),
                spec,
            )?,
            cross_attention_norm: schema.layer_norm("cross_attention_norm", width, eps)?,
            feed_forward: schema.mlp("feed_forward", [width, inner, width], activation, true)?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
        })
    }

    /// Runs the layer over the queries `hidden`, `[B, Q, W]`, with their `positions`, sampling `memory` around `references`.
    pub fn forward(
        &self,
        hidden: &Tensor,
        positions: &Tensor,
        references: &Tensor,
        memory: &Tensor,
        levels: &FeatureLevels,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let positioned = hidden.add(positions)?;
        let hidden = self
            .attention
            .forward(&positioned, &positioned, hidden, state)?
            .add(hidden)?
            .apply(&self.attention_norm, state)?;
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

/// Renames a checkpoint's DETR-family decoder layers under `from` to [`DeformableDecoderLayer`]'s names under `to`, both ending in the layer index's `*`.
#[must_use]
pub fn deformable_decoder_rules(from: &str, to: &str) -> Rules {
    attention_rules(&join(from, "self_attn"), &join(to, "attention"))
        .then(deformable_attention_rules(
            &join(from, "encoder_attn"),
            &join(to, "cross_attention"),
        ))
        .rename_each(
            from,
            to,
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
}

/// Returns the logit of `x` after clamping it to `[0, 1]`, with each side of the ratio held at least `eps`, as `inverse_sigmoid` computes it.
pub fn inverse_sigmoid(x: &Tensor, eps: f32) -> SingeResult<Tensor> {
    let x = x.clamp(0.0, 1.0)?;
    let numerator = x.clamp(eps, 1.0)?;
    let denominator = x.rsub_scalar(1.0)?.clamp(eps, 1.0)?;
    numerator.div(&denominator)?.log()
}

/// The width and height of the first level's anchors, as a share of the image, doubling at each level.
const ANCHOR_SIZE: f32 = 0.05;

/// How far inside the image an anchor's every coordinate must lie to be valid.
const ANCHOR_MARGIN: f32 = 0.01;

/// Returns an anchor box at each point of levels of `sizes`, centred on the point and as wide and tall as 0.05 of the image doubled at each level, as logits `[1, S, 4]`, and whether each lies inside the image, `[1, S, 1]` as ones and zeros, as RT-DETR's `_cached_generate_anchors` and the two-stage DETRs' `gen_encoder_output_proposals` compute them in `F32`.
/// An anchor outside holds `invalid` in each coordinate: `F32`'s largest value for RT-DETR, infinity for Grounding DINO.
pub fn anchor_proposals(
    device: Device,
    sizes: &[[i64; 2]],
    invalid: f32,
) -> SingeResult<(Tensor, Tensor)> {
    let mut logits = Vec::new();
    let mut valid = Vec::new();
    for (level, &[height, width]) in sizes.iter().enumerate() {
        let side = ANCHOR_SIZE * 2_f32.powi(i32::try_from(level).unwrap_or(i32::MAX));
        for row in 0..height {
            for column in 0..width {
                let anchor = [
                    (column as f32 + 0.5) / width as f32,
                    (row as f32 + 0.5) / height as f32,
                    side,
                    side,
                ];
                let inside = anchor
                    .iter()
                    .all(|&value| value > ANCHOR_MARGIN && value < 1.0 - ANCHOR_MARGIN);
                logits.extend(anchor.map(|value| {
                    if inside {
                        (value / (1.0 - value)).ln()
                    } else {
                        invalid
                    }
                }));
                valid.push(if inside { 1.0 } else { 0.0 });
            }
        }
    }

    let count = i64::try_from(valid.len()).map_err(|_| SingeError::out_of_range("anchors"))?;
    let logits = capture::literal(device, |device| {
        Tensor::from_vec(device, Shape::create([1, count, 4])?, logits)
    })?;
    let valid = capture::literal(device, |device| {
        Tensor::from_vec(device, Shape::create([1, count, 1])?, valid)
    })?;
    Ok((logits, valid))
}

/// Returns the logit of `x` clamped to `[eps, 1 - eps]`, as `torch.special.logit` computes it.
pub fn logit(x: &Tensor, eps: f32) -> SingeResult<Tensor> {
    let x = x.clamp(eps, 1.0 - eps)?;
    x.div(&x.rsub_scalar(1.0)?)?.log()
}

/// Returns the rows of `tensor`, `[B, S, C]`, that `indices`, `[B, K]`, pick in each batch, `[B, K, C]`.
pub fn gather_rows(tensor: &Tensor, indices: &Tensor) -> SingeResult<Tensor> {
    let (batch, sequence, channels) = (
        tensor.dimension(0)?,
        tensor.dimension(1)?,
        tensor.dimension(2)?,
    );
    let count = indices.dimension(1)?;
    let device = tensor.device();
    let offsets = Tensor::range(device, 0_i64, batch.clone())?
        .mul(&Tensor::from_dimensions(device, &[sequence])?)?
        .unsqueeze(1)?;
    let rows = indices.cast(DataType::I64)?.add(&offsets)?.flatten(0, 1)?;
    tensor
        .flatten(0, 1)?
        .index_select(0, &rows)?
        .reshape_with(&[batch, count, channels])
}
