//! Swin, as Donut's encoder and CLAP's HTSAT have it: patches embedded and normed, then stages of pre-norm blocks attending within windows, shifted every other block, with a learned bias of relative positions, each stage but the last merging two by two patches.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::capture,
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::{Padding, Spatial2d, Spatial2dI64},
};

use crate::{
    assembly::{
        PYTORCH_NORM_EPS, merge_heads, merge_windows, partition_windows, split_heads, to_tokens,
    },
    format::{Rules, join},
    schema::Schema,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwinConfig {
    pub width: i64,
    /// Each stage's blocks and heads; widths double from stage to stage.
    pub stages: Vec<(i64, i64)>,
    pub window: i64,
    pub patch_size: i64,
    pub channels: i64,
    pub mlp_ratio: f64,
    pub activation: Activation,
    pub eps: f32,
}

impl SwinConfig {
    #[must_use]
    pub fn out_width(&self) -> i64 {
        self.width << self.stages.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    bias_table: StateKey,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    heads: i64,
    shift: i64,
}

impl Block {
    fn create(
        schema: &mut Schema<'_>,
        config: &SwinConfig,
        width: i64,
        heads: i64,
        shift: i64,
    ) -> SingeResult<Self> {
        let inner = (width as f64 * config.mlp_ratio) as i64;
        let side = 2 * config.window - 1;
        let (query, key, value, output, bias_table) = {
            let mut schema = schema.scope("attention");
            let projections = (
                schema.linear("query", width, width, true)?,
                schema.linear("key", width, width, true)?,
                schema.linear("value", width, width, true)?,
                schema.linear("output", width, width, true)?,
                schema.parameter("bias_table", [side * side, heads])?,
            );
            schema.join_rows(&[&projections.0, &projections.1, &projections.2])?;
            projections
        };
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
            query,
            key,
            value,
            output,
            bias_table,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, inner, width],
                config.activation,
                true,
            )?,
            heads,
            shift,
        })
    }
}

#[derive(Debug, Clone)]
struct Merge {
    norm: LayerNorm,
    reduction: Linear,
}

impl Merge {
    fn create(schema: &mut Schema<'_>, width: i64) -> SingeResult<Self> {
        let mut schema = schema.scope("merge");
        Ok(Self {
            norm: schema.layer_norm("norm", 4 * width, PYTORCH_NORM_EPS)?,
            reduction: schema.linear("reduction", 4 * width, 2 * width, false)?,
        })
    }
}

#[derive(Debug, Clone)]
struct Stage {
    blocks: Vec<Block>,
    merge: Option<Merge>,
}

#[derive(Debug, Clone)]
pub struct Swin {
    patch: i64,
    patches: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
    window: i64,
}

/// Returns each pair of a window's points' index into the relative position table, `[w⁴]`.
#[must_use]
pub fn relative_index(window: i64) -> Vec<i64> {
    let points = (0..window)
        .flat_map(|row| (0..window).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    points
        .iter()
        .flat_map(|&(row, column)| {
            points.iter().map(move |&(other_row, other_column)| {
                (row - other_row + window - 1) * (2 * window - 1)
                    + (column - other_column + window - 1)
            })
        })
        .collect()
}

/// Returns the mask of a shifted window's scores, `[windows, w², w²]`: `masked` between points of regions the roll brought together, and zero elsewhere.
#[must_use]
pub fn shift_mask(height: i64, width: i64, window: i64, shift: i64, masked: f32) -> Vec<f32> {
    let region = |index: i64, side: i64| {
        i64::from(index >= side - window) + i64::from(index >= side - shift)
    };
    let mut mask = Vec::new();
    for window_row in 0..height / window {
        for window_column in 0..width / window {
            let labels = (0..window)
                .flat_map(|row| (0..window).map(move |column| (row, column)))
                .map(|(row, column)| {
                    region(window_row * window + row, height) * 3
                        + region(window_column * window + column, width)
                })
                .collect::<Vec<_>>();
            for &label in &labels {
                for &other in &labels {
                    mask.push(if label == other { 0.0 } else { masked });
                }
            }
        }
    }
    mask
}

impl Swin {
    /// Registers the encoder: `embeddings.patches` and `embeddings.norm`, then `stages.N`, each with `blocks.M` and, in every stage but the last, `merge`.
    pub fn create(schema: &mut Schema<'_>, config: &SwinConfig) -> SingeResult<Self> {
        let width = config.width;
        let patches = schema.conv2d(
            "embeddings.patches",
            config.channels,
            width,
            config.patch_size,
            Conv2dConfig {
                stride: Spatial2d::square(config.patch_size),
                ..Conv2dConfig::default()
            },
            true,
        )?;
        // The patch norm and the merges' norms keep `nn.LayerNorm`'s default epsilon instead of `layer_norm_eps`, in Donut's Swin and in HTSAT.
        let patch_norm = schema.layer_norm("embeddings.norm", width, PYTORCH_NORM_EPS)?;
        let count = config.stages.len();
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(index, &(depth, heads))| {
                let width = width << index;
                let mut schema = schema.scope(format!("stages.{index}"));
                let blocks = (0..depth)
                    .map(|block| {
                        let shift = if block % 2 == 0 { 0 } else { config.window / 2 };
                        Block::create(
                            &mut schema.scope(format!("blocks.{block}")),
                            config,
                            width,
                            heads,
                            shift,
                        )
                    })
                    .collect::<SingeResult<Vec<_>>>()?;
                let merge = (index + 1 < count)
                    .then(|| Merge::create(&mut schema, width))
                    .transpose()?;
                Ok(Stage { blocks, merge })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            patch: config.patch_size,
            patches,
            patch_norm,
            stages,
            window: config.window,
        })
    }

    fn block(
        &self,
        block: &Block,
        hidden: &Tensor,
        Spatial2d { height, width }: Spatial2dI64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let channels = hidden.shape().dimensions()[2];
        let batch = hidden.dimension(0)?;
        // A map no larger than a window attends in one unshifted window.
        let (window, shift) = if height.min(width) <= self.window {
            (height.min(width), 0)
        } else {
            (self.window, block.shift)
        };
        let padded_height = (height + window - 1) / window * window;
        let padded_width = (width + window - 1) / window * window;
        let normed = hidden
            .apply(&block.attention_norm, state)?
            .reshape_with(&[batch.clone(), height.into(), width.into(), channels.into()])?
            .pad(
                &[
                    Padding::ZERO,
                    Padding::new(0, padded_height - height),
                    Padding::new(0, padded_width - width),
                    Padding::ZERO,
                ],
                0.0,
            )?;
        let (rows, columns) = (padded_height / window, padded_width / window);
        let area = window * window;
        let windows = partition_windows(&normed.roll(&[-shift, -shift], &[1, 2])?, window)?;
        let count = batch.clone() * (rows * columns);
        let heads = block.heads;
        let head = channels / heads;
        let query = split_heads(&windows.apply(&block.query, state)?, heads)?;
        let key = split_heads(&windows.apply(&block.key, state)?, heads)?;
        let value = split_heads(&windows.apply(&block.value, state)?, heads)?;
        let device = hidden.device();
        let index = capture::literal(device, |device| {
            Tensor::from_vec(
                device,
                Shape::create([area * area])?,
                relative_index(window),
            )
        })?;
        let bias = state
            .tensor(&block.bias_table)?
            .index_select(0, &index)?
            .reshape([area, area, heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?;
        let scale = (head as f64).powf(-0.5);
        let mut scores = query
            .mul_scalar(scale)?
            .matmul(&key.transpose(2, 3)?)?
            .add(&bias)?;
        if shift > 0 {
            let mask = capture::literal(device, |device| {
                Tensor::from_vec(
                    device,
                    Shape::create([1, rows * columns, 1, area, area])?,
                    shift_mask(padded_height, padded_width, window, shift, -100.0),
                )
            })?
            .cast(hidden.data_type())?;
            scores = scores
                .reshape_with(&[
                    batch.clone(),
                    (rows * columns).into(),
                    heads.into(),
                    area.into(),
                    area.into(),
                ])?
                .add(&mask)?
                .reshape_with(&[count, heads.into(), area.into(), area.into()])?;
        }

        let attended =
            merge_heads(&scores.softmax(-1)?.matmul(&value)?)?.apply(&block.output, state)?;
        let attended = merge_windows(
            &attended,
            window,
            batch.clone(),
            (padded_height, padded_width),
        )?
        .roll(&[shift, shift], &[1, 2])?
        .narrow(1, 0, height)?
        .narrow(2, 0, width)?
        .reshape_with(&[batch, (height * width).into(), channels.into()])?;
        let hidden = hidden.add(&attended)?;
        hidden
            .apply(&block.feed_forward_norm, state)?
            .apply(&block.feed_forward, state)?
            .add(&hidden)
    }

    /// Returns each stage's tokens before its merge, `[B, h · w, C]`, with the height and width of their map.
    pub fn stage_tokens(
        &self,
        input: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Vec<(Tensor, Spatial2dI64)>> {
        // Zeros below and to the right make whole patches, as `SwinPatchEmbeddings.maybe_pad` adds them.
        let &[_, _, rows, columns] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("Swin images, [B, C, H, W]"));
        };
        let excess = |extent: i64| (self.patch - extent % self.patch) % self.patch;
        let input = if excess(rows) + excess(columns) > 0 {
            input.pad(
                &[
                    Padding::ZERO,
                    Padding::ZERO,
                    Padding::new(0, excess(rows)),
                    Padding::new(0, excess(columns)),
                ],
                0.0,
            )?
        } else {
            input.clone()
        };
        let patches = input.apply(&self.patches, state)?;
        let dimensions = patches.shape().dimensions();
        let (mut height, mut width) = (dimensions[2], dimensions[3]);
        let mut hidden = to_tokens(&patches)?.apply(&self.patch_norm, state)?;
        let mut stages = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in &stage.blocks {
                hidden = self.block(block, &hidden, Spatial2d::new(height, width), state)?;
            }

            stages.push((hidden.clone(), Spatial2d::new(height, width)));
            if let Some(merge) = &stage.merge {
                let channels = hidden.shape().dimensions()[2];
                let batch = hidden.dimension(0)?;
                let grid = hidden
                    .reshape_with(&[batch.clone(), height.into(), width.into(), channels.into()])?
                    .pad(
                        &[
                            Padding::ZERO,
                            Padding::new(0, height % 2),
                            Padding::new(0, width % 2),
                            Padding::ZERO,
                        ],
                        0.0,
                    )?;
                let (half_height, half_width) = ((height + 1) / 2, (width + 1) / 2);
                // Each 2 × 2 square's points in `transformers`' order: top left, bottom left, top right, bottom right.
                let squares = grid.reshape_with(&[
                    batch.clone(),
                    half_height.into(),
                    2.into(),
                    half_width.into(),
                    2.into(),
                    channels.into(),
                ])?;
                let corner = |row: i64, column: i64| -> SingeResult<Tensor> {
                    squares
                        .narrow(2, row, 1)?
                        .narrow(4, column, 1)?
                        .reshape_with(&[
                            batch.clone(),
                            (half_height * half_width).into(),
                            channels.into(),
                        ])
                };
                hidden = Tensor::concat(
                    &[corner(0, 0)?, corner(1, 0)?, corner(0, 1)?, corner(1, 1)?],
                    2,
                )?
                .apply(&merge.norm, state)?
                .apply(&merge.reduction, state)?;
                height = half_height;
                width = half_width;
            }
        }
        Ok(stages)
    }
}

impl Layer for Swin {
    /// Encodes images `[B, C, H, W]` into their last stage's tokens, `[B, h · w, W]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.stage_tokens(input, state)?
            .pop()
            .map(|(tokens, _)| tokens)
            .ok_or_else(|| SingeError::invalid_argument("a Swin without stages"))
    }
}

/// Imports a Swin stored as `transformers` names it, its stages under `{from}.layers` and its patch projection and norm at `patches` and `patch_norm`, to the schema's names under `to`.
#[must_use]
pub fn swin_rules(from: &str, patches: &str, patch_norm: &str, to: &str) -> Rules {
    let blocks = join(from, "layers.*.blocks.*");
    Rules::new()
        .rename(patches, &join(to, "embeddings.patches"))
        .rename(patch_norm, &join(to, "embeddings.norm"))
        .rename_each(
            &blocks,
            &join(to, "stages.*.blocks.*"),
            &[
                ("layernorm_before", "attention_norm"),
                ("attention.self.query", "attention.query"),
                ("attention.self.key", "attention.key"),
                ("attention.self.value", "attention.value"),
                (
                    "attention.self.relative_position_bias_table",
                    "attention.bias_table",
                ),
                ("attention.output.dense", "attention.output"),
                ("layernorm_after", "feed_forward_norm"),
                ("intermediate.dense", "feed_forward.up"),
                ("output.dense", "feed_forward.down"),
            ],
        )
        // The table's index, which the encoder computes.
        .skip(&join(&blocks, "attention.self.relative_position_index"))
        .rename(
            &join(from, "layers.*.downsample"),
            &join(to, "stages.*.merge"),
        )
}
