// Swin, as Donut's encoder and CLAP's HTSAT have it: patches embedded and normed, then stages of pre-norm blocks attending within windows, shifted every other block, with a learned bias of relative positions, each stage but the last merging two by two patches.
// The rolls, windows, shift masks, and token reshapes it is built from serve Swin2SR and SegFormer too.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, ModelState, StateKey},
};

use crate::{format::Rules, schema::Schema};

// `nn.LayerNorm`'s default epsilon.
const DEFAULT_EPS: f32 = 1e-5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwinConfig {
    pub width: i64,
    // Each stage's blocks and heads; widths double from stage to stage.
    pub stages: Vec<(i64, i64)>,
    pub window: i64,
    pub patch_size: i64,
    pub channels: i64,
    pub mlp_ratio: f64,
    pub activation: Activation,
    pub eps: f32,
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
    up: Linear,
    down: Linear,
    heads: i64,
    shift: i64,
}

#[derive(Debug, Clone)]
struct Merge {
    norm: LayerNorm,
    reduction: Linear,
}

#[derive(Debug, Clone)]
struct Stage {
    blocks: Vec<Block>,
    merge: Option<Merge>,
}

#[derive(Debug, Clone)]
pub struct Swin {
    patches: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
    window: i64,
    activation: Activation,
}

// Rolls `[B, H, W, C]` by `shift` along height and width, as `torch.roll` does.
pub fn roll(input: &Tensor, shift: i64) -> SingeResult<Tensor> {
    let mut rolled = input.clone();
    for (axis, index) in [(1_i64, 1_usize), (2, 2)] {
        let length = rolled.shape().dimensions()[index];
        let split = shift.rem_euclid(length);
        if split == 0 {
            continue;
        }

        let at = length - split;
        rolled = Tensor::concat(
            &[rolled.narrow(axis, at, split)?, rolled.narrow(axis, 0, at)?],
            axis,
        )?;
    }
    Ok(rolled)
}

// `[B, H, W, C]` as windows of `window × window` points, `[B · windows, window², C]`, row by row.
pub fn partition_windows(input: &Tensor, window: i64) -> SingeResult<Tensor> {
    let &[batch, height, width, channels] = input.shape().dimensions() else {
        return Err(singe::Error::invalid_argument("window partition input"));
    };
    input
        .reshape([
            batch,
            height / window,
            window,
            width / window,
            window,
            channels,
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape([
            batch * (height / window) * (width / window),
            window * window,
            channels,
        ])
}

// Windows, `[B · windows, window², C]`, back into `[B, H, W, C]`.
pub fn merge_windows(
    windows: &Tensor,
    window: i64,
    (batch, height, width): (i64, i64, i64),
) -> SingeResult<Tensor> {
    let channels = windows.shape().dimensions()[2];
    windows
        .reshape([
            batch,
            height / window,
            width / window,
            window,
            window,
            channels,
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape([batch, height, width, channels])
}

// A map `[B, C, H, W]` as tokens `[B, H · W, C]`.
pub fn to_tokens(map: &Tensor) -> SingeResult<Tensor> {
    map.flatten(2, 3)?.transpose(1, 2)
}

// Tokens `[B, H · W, C]` as a map `[B, C, H, W]`.
pub fn to_map(tokens: &Tensor, (height, width): (i64, i64)) -> SingeResult<Tensor> {
    let &[batch, _, channels] = tokens.shape().dimensions() else {
        return Err(singe::Error::invalid_argument("tokens"));
    };
    tokens
        .transpose(1, 2)?
        .reshape([batch, channels, height, width])
}

// Each pair of a window's points' index into the relative position table, `[w⁴]`.
fn relative_index(window: i64) -> Vec<i64> {
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

// The mask of a shifted window's scores, `[windows, w², w²]`: `masked` between points of regions the roll brought together, and zero elsewhere.
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
    // Registers the encoder under `transformers`' Swin names: `embeddings`, then `encoder.layers.N` of `blocks.M` and `downsample`.
    pub fn create(schema: &mut Schema<'_>, config: &SwinConfig) -> SingeResult<Self> {
        let width = config.width;
        let patches = schema.conv2d(
            "embeddings.patches",
            config.channels,
            width,
            config.patch_size,
            Conv2dConfig {
                stride: (config.patch_size, config.patch_size),
                ..Conv2dConfig::default()
            },
            true,
        )?;
        // The patch norm and the merges' norms keep `nn.LayerNorm`'s default epsilon instead of `layer_norm_eps`, in Donut's Swin and in HTSAT.
        let patch_norm = schema.layer_norm("embeddings.norm", width, DEFAULT_EPS)?;
        let count = config.stages.len();
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(index, &(depth, heads))| {
                let width = width << index;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    reason = "widths are small"
                )]
                let inner = (width as f64 * config.mlp_ratio) as i64;
                let mut schema = schema.scope(format!("stages.{index}"));
                let side = 2 * config.window - 1;
                let blocks = (0..depth)
                    .map(|block| {
                        let mut schema = schema.scope(format!("blocks.{block}"));
                        Ok(Block {
                            attention_norm: schema.layer_norm(
                                "attention_norm",
                                width,
                                config.eps,
                            )?,
                            query: schema.linear("query", width, width, true)?,
                            key: schema.linear("key", width, width, true)?,
                            value: schema.linear("value", width, width, true)?,
                            output: schema.linear("output", width, width, true)?,
                            bias_table: schema.parameter("bias_table", [side * side, heads])?,
                            feed_forward_norm: schema.layer_norm(
                                "feed_forward_norm",
                                width,
                                config.eps,
                            )?,
                            up: schema.linear("up", width, inner, true)?,
                            down: schema.linear("down", inner, width, true)?,
                            heads,
                            shift: if block % 2 == 0 { 0 } else { config.window / 2 },
                        })
                    })
                    .collect::<SingeResult<Vec<_>>>()?;
                let merge = (index + 1 < count)
                    .then(|| -> SingeResult<Merge> {
                        Ok(Merge {
                            norm: schema.layer_norm("merge.norm", 4 * width, DEFAULT_EPS)?,
                            reduction: schema.linear(
                                "merge.reduction",
                                4 * width,
                                2 * width,
                                false,
                            )?,
                        })
                    })
                    .transpose()?;
                Ok(Stage { blocks, merge })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            patches,
            patch_norm,
            stages,
            window: config.window,
            activation: config.activation,
        })
    }

    fn block(
        &self,
        block: &Block,
        hidden: &Tensor,
        (height, width): (i64, i64),
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (batch, channels) = (dimensions[0], dimensions[2]);
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
            .reshape([batch, height, width, channels])?
            .pad(
                &[
                    (0, 0),
                    (0, padded_height - height),
                    (0, padded_width - width),
                    (0, 0),
                ],
                0.0,
            )?;
        let (rows, columns) = (padded_height / window, padded_width / window);
        let area = window * window;
        let windows = partition_windows(&roll(&normed, -shift)?, window)?;
        let count = batch * rows * columns;
        let heads = block.heads;
        let head = channels / heads;
        let split = |tensor: Tensor| tensor.reshape([count, area, heads, head])?.transpose(1, 2);
        let query = split(windows.apply(&block.query, state)?)?;
        let key = split(windows.apply(&block.key, state)?)?;
        let value = split(windows.apply(&block.value, state)?)?;
        let device = hidden.device();
        let index = Tensor::from_vec(
            device,
            Shape::create([area * area, 1])?,
            relative_index(window),
        )?
        .expand(Shape::create([area * area, heads])?)?;
        let bias = state
            .tensor(&block.bias_table)?
            .gather(0, &index)?
            .reshape([area, area, heads])?
            .permute(&[2, 0, 1])?
            .unsqueeze(0)?;
        #[expect(clippy::cast_precision_loss, reason = "head widths are small")]
        let scale = (head as f64).powf(-0.5);
        let mut scores = query
            .mul_scalar(scale)?
            .matmul(&key.transpose(2, 3)?)?
            .add(&bias)?;
        if shift > 0 {
            let mask = Tensor::from_vec(
                device,
                Shape::create([1, rows * columns, 1, area, area])?,
                shift_mask(padded_height, padded_width, window, shift, -100.0),
            )?
            .cast(hidden.data_type())?;
            scores = scores
                .reshape([batch, rows * columns, heads, area, area])?
                .add(&mask)?
                .reshape([count, heads, area, area])?;
        }

        let attended = scores
            .softmax(-1)?
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape([count, area, channels])?
            .apply(&block.output, state)?;
        let attended = roll(
            &merge_windows(&attended, window, (batch, padded_height, padded_width))?,
            shift,
        )?
        .narrow(1, 0, height)?
        .narrow(2, 0, width)?
        .reshape([batch, height * width, channels])?;
        let hidden = hidden.add(&attended)?;
        hidden
            .apply(&block.feed_forward_norm, state)?
            .apply(&block.up, state)?
            .apply(&self.activation, state)?
            .apply(&block.down, state)?
            .add(&hidden)
    }
}

impl Layer for Swin {
    // Images `[B, C, H, W]` to their last stage's tokens, `[B, h · w, W]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let patches = input.apply(&self.patches, state)?;
        let dimensions = patches.shape().dimensions();
        let (mut height, mut width) = (dimensions[2], dimensions[3]);
        let mut hidden = to_tokens(&patches)?.apply(&self.patch_norm, state)?;
        for stage in &self.stages {
            for block in &stage.blocks {
                hidden = self.block(block, &hidden, (height, width), state)?;
            }
            if let Some(merge) = &stage.merge {
                let dimensions = hidden.shape().dimensions();
                let (batch, channels) = (dimensions[0], dimensions[2]);
                let grid = hidden
                    .reshape([batch, height, width, channels])?
                    .pad(&[(0, 0), (0, height % 2), (0, width % 2), (0, 0)], 0.0)?;
                let (half_height, half_width) = ((height + 1) / 2, (width + 1) / 2);
                // Each 2 × 2 square's points in `transformers`' order: top left, bottom left, top right, bottom right.
                let squares = grid.reshape([batch, half_height, 2, half_width, 2, channels])?;
                let corner = |row: i64, column: i64| -> SingeResult<Tensor> {
                    squares.narrow(2, row, 1)?.narrow(4, column, 1)?.reshape([
                        batch,
                        half_height * half_width,
                        channels,
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
        Ok(hidden)
    }
}

// Imports a Swin stored as `transformers` names it, its stages under `{from}layers` and its patch projection and norm at `patches` and `patch_norm`, to the schema's names under `to`.
#[must_use]
pub fn swin_rules(from: &str, patches: &str, patch_norm: &str, to: &str) -> Rules {
    let block = |name: &str| format!("{from}layers.*.blocks.*.{name}");
    let target = |name: &str| format!("{to}stages.*.blocks.*.{name}");
    Rules::new()
        .rename(patches, &format!("{to}embeddings.patches"))
        .rename(patch_norm, &format!("{to}embeddings.norm"))
        .rename(&block("layernorm_before"), &target("attention_norm"))
        .rename(&block("attention.self.query"), &target("query"))
        .rename(&block("attention.self.key"), &target("key"))
        .rename(&block("attention.self.value"), &target("value"))
        .rename(
            &block("attention.self.relative_position_bias_table"),
            &target("bias_table"),
        )
        // The table's index, which the encoder computes.
        .skip(&block("attention.self.relative_position_index"))
        .rename(&block("attention.output.dense"), &target("output"))
        .rename(&block("layernorm_after"), &target("feed_forward_norm"))
        .rename(&block("intermediate.dense"), &target("up"))
        .rename(&block("output.dense"), &target("down"))
        .rename(
            &format!("{from}layers.*.downsample"),
            &format!("{to}stages.*.merge"),
        )
}
