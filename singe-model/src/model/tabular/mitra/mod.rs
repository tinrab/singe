// Mitra: AutoGluon's Tab2D, a table model that reads labeled rows as context; each cell is a token, each feature first replaced by its quantile bucket among the context rows and normalized over them, with each row's target as one more token, and its blocks attend across the rows of each column, a row to classify reading the context rows, and then across the cells of each row.
//
// Its state is `cells`, `targets` (a table of classes or a linear map of a target), `unknown`, the target token of a row to predict, `blocks.N` (`rows_norm`, `rows` with `query`, `key`, `value`, and `output`, `rows_feed_forward_norm`, `rows_feed_forward` with `up` and `down`, and the same four for `cells`), `norm`, and `output`.

mod autogluon;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Embedding, LayerNorm, Linear, Model, ModelState, StateBuilder},
};

use crate::{
    assembly::{Attention, AttentionSpec},
    error::{Error, Result},
    format::Loader,
    head::{HeadLoader, TabularClassifier, TabularRegressor},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// The quantiles each feature is bucketed by: `1 / 1000` to `999 / 1000`.
const QUANTILES: usize = 999;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MitraTask {
    Classification,
    Regression,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MitraConfig {
    pub width: i64,
    // The classes a classifier scores, or one for a regressor.
    pub outputs: i64,
    pub layers: i64,
    pub heads: i64,
    pub task: MitraTask,
}

#[derive(Debug, Clone)]
struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn create(schema: &mut Schema<'_>, width: i64) -> SingeResult<Self> {
        Ok(Self {
            up: schema.linear("up", width, 4 * width, true)?,
            down: schema.linear("down", 4 * width, width, true)?,
        })
    }

    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.up, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.down, state)
    }
}

#[derive(Debug, Clone)]
struct MitraBlock {
    rows_norm: LayerNorm,
    rows: Attention,
    rows_feed_forward_norm: LayerNorm,
    rows_feed_forward: FeedForward,
    cells_norm: LayerNorm,
    cells: Attention,
    cells_feed_forward_norm: LayerNorm,
    cells_feed_forward: FeedForward,
}

impl MitraBlock {
    // The context's cells, `[S, C, W]`, and the cells to predict, `[Q, C, W]`, after the block.
    fn forward(
        &self,
        context: &Tensor,
        queries: &Tensor,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        // Down each column: the context's rows attend to each other, and each row to predict to the context's.
        let columns = |cells: &Tensor| cells.apply(&self.rows_norm, state)?.permute(&[1, 0, 2]);
        let (context_columns, query_columns) = (columns(context)?, columns(queries)?);
        let context = context.add(
            &self
                .rows
                .forward(&context_columns, &context_columns, &context_columns, state)?
                .permute(&[1, 0, 2])?,
        )?;
        let queries = queries.add(
            &self
                .rows
                .forward(&query_columns, &context_columns, &context_columns, state)?
                .permute(&[1, 0, 2])?,
        )?;
        let fed =
            |cells: Tensor, norm: &LayerNorm, feed_forward: &FeedForward| -> SingeResult<Tensor> {
                let update = feed_forward.forward(&cells.apply(norm, state)?, state)?;
                cells.add(&update)
            };
        let context = fed(
            context,
            &self.rows_feed_forward_norm,
            &self.rows_feed_forward,
        )?;
        let queries = fed(
            queries,
            &self.rows_feed_forward_norm,
            &self.rows_feed_forward,
        )?;
        // Along each row: its cells attend to each other.
        let across = |cells: Tensor| -> SingeResult<Tensor> {
            let normed = cells.apply(&self.cells_norm, state)?;
            let update = self.cells.forward(&normed, &normed, &normed, state)?;
            cells.add(&update)
        };
        let context = across(context)?;
        let queries = across(queries)?;
        Ok((
            fed(
                context,
                &self.cells_feed_forward_norm,
                &self.cells_feed_forward,
            )?,
            fed(
                queries,
                &self.cells_feed_forward_norm,
                &self.cells_feed_forward,
            )?,
        ))
    }
}

#[derive(Debug, Clone)]
enum TargetEmbedding {
    Classes(Embedding),
    Values(Linear),
}

#[derive(Debug, Clone)]
pub struct MitraLayers {
    config: MitraConfig,
    cells: Linear,
    targets: TargetEmbedding,
    unknown: Embedding,
    blocks: Vec<MitraBlock>,
    norm: LayerNorm,
    output: Linear,
}

impl MitraLayers {
    pub fn create(schema: &mut Schema<'_>, config: &MitraConfig) -> SingeResult<Self> {
        let width = config.width;
        let eps = 1e-5;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut block = schema.scope(format!("blocks.{index}"));
                Ok(MitraBlock {
                    rows_norm: block.layer_norm("rows_norm", width, eps)?,
                    rows: Attention::create(
                        &mut block.scope("rows"),
                        AttentionSpec::square(width, config.heads),
                    )?,
                    rows_feed_forward_norm: block.layer_norm(
                        "rows_feed_forward_norm",
                        width,
                        eps,
                    )?,
                    rows_feed_forward: FeedForward::create(
                        &mut block.scope("rows_feed_forward"),
                        width,
                    )?,
                    cells_norm: block.layer_norm("cells_norm", width, eps)?,
                    cells: Attention::create(
                        &mut block.scope("cells"),
                        AttentionSpec::square(width, config.heads),
                    )?,
                    cells_feed_forward_norm: block.layer_norm(
                        "cells_feed_forward_norm",
                        width,
                        eps,
                    )?,
                    cells_feed_forward: FeedForward::create(
                        &mut block.scope("cells_feed_forward"),
                        width,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            cells: schema.linear("cells", 1, width, true)?,
            targets: match config.task {
                MitraTask::Classification => TargetEmbedding::Classes(Embedding::new(
                    schema.parameter("targets.weight", [config.outputs, width])?,
                )),
                MitraTask::Regression => {
                    TargetEmbedding::Values(schema.linear("targets", 1, width, true)?)
                }
            },
            unknown: Embedding::new(schema.parameter("unknown.weight", [1, width])?),
            blocks,
            norm: schema.layer_norm("norm", width, eps)?,
            output: schema.linear("output", width, config.outputs, true)?,
        })
    }

    // Each row to predict's outputs, `[Q, outputs]`, from the context's normalized features, `[S, F]`, and targets, `[S]` (`I64` classes or `F32` values), and the normalized features of the rows to predict, `[Q, F]`.
    fn predict(
        &self,
        features: &Tensor,
        targets: &Tensor,
        queries: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let data_type = state.floating_type()?;
        let cells = |features: &Tensor| {
            features
                .cast(data_type)?
                .unsqueeze(2)?
                .apply(&self.cells, state)
        };
        let target_tokens = match &self.targets {
            TargetEmbedding::Classes(table) => targets.apply(table, state)?,
            TargetEmbedding::Values(linear) => targets
                .cast(data_type)?
                .unsqueeze(1)?
                .apply(linear, state)?,
        }
        .unsqueeze(1)?;
        let count = queries.shape().dimensions()[0];
        let unknown = Tensor::from_vec(
            queries.device(),
            Shape::create([count])?,
            vec![0_i64; usize::try_from(count).unwrap_or(0)],
        )?
        .apply(&self.unknown, state)?
        .unsqueeze(1)?;
        let mut context = Tensor::concat(&[target_tokens, cells(features)?], 1)?;
        let mut rows = Tensor::concat(&[unknown, cells(queries)?], 1)?;
        for block in &self.blocks {
            (context, rows) = block.forward(&context, &rows, state)?;
        }
        rows.select(1, 0)?
            .apply(&self.norm, state)?
            .apply(&self.output, state)
    }
}

pub type Mitra = Model<MitraLayers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "mitra",
    formats: &[(Format::AutoGluon, &["tab2d"])],
    heads: &[
        HeadLoader::TabularClassification(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::TabularRegression(|loader| Ok(Box::new(load(loader)?))),
    ],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Mitra> {
    let config: MitraConfig = loader.config(autogluon::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        MitraLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "model",
        &Model::new(layers, builder.finish()?),
        &autogluon::rules(),
    )?;
    loader.finish()?;
    Ok(model)
}

// `torch.lerp` in `F32`, as CUDA's kernel computes it with fused multiply-adds.
fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    let difference = end - start;
    if weight.abs() < 0.5 {
        weight.mul_add(difference, start)
    } else {
        (-difference).mul_add(1.0 - weight, end)
    }
}

// Each context row's and each row to predict's features, `[S · F]` and `[Q · F]` row-major, as Tab2D's quantile embedding gives them: a feature's bucket among the context's quantiles of it, over the context's rows, then centered and scaled by the context's mean and deviation of those, or zero where the context's are all one.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "row counts are far below where `F32` and `F64` lose whole numbers"
)]
fn bucketed(rows: &[Vec<f32>], queries: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let (count, features) = (rows.len(), rows.first().map_or(0, Vec::len));
    let mut context = vec![0.0; count * features];
    let mut predicted = vec![0.0; queries.len() * features];
    for feature in 0..features {
        let mut sorted = rows.iter().map(|row| row[feature]).collect::<Vec<f32>>();
        sorted.sort_by(f32::total_cmp);
        // `torch.quantile`'s linear interpolation: ranks in `F64`, weights and values in `F32`.
        let boundaries = (1..=QUANTILES)
            .map(|index| {
                let quantile = index as f32 / 1000.0;
                let rank = f64::from(quantile) * (count - 1) as f64;
                let below = rank as usize;
                let weight = (rank - below as f64) as f32;
                lerp(sorted[below], sorted[rank.ceil() as usize], weight)
            })
            .collect::<Vec<f32>>();
        // `torch.bucketize`: how many boundaries lie below the value.
        let bucket = |value: f32| {
            boundaries.partition_point(|&boundary| boundary < value) as f32 / count as f32
        };
        let column = rows
            .iter()
            .map(|row| bucket(row[feature]))
            .collect::<Vec<f32>>();
        let mean = column.iter().sum::<f32>() / count as f32;
        let variance = column
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / count as f32;
        let scale = |value: f32| {
            if variance == 0.0 {
                0.0
            } else {
                (value - mean) / variance.sqrt()
            }
        };
        for (row, value) in column.iter().enumerate() {
            context[row * features + feature] = scale(*value);
        }
        for (row, query) in queries.iter().enumerate() {
            predicted[row * features + feature] = scale(bucket(query[feature]));
        }
    }
    (context, predicted)
}

impl MitraLayers {
    fn run(
        &self,
        rows: &[Vec<f32>],
        targets: &Tensor,
        queries: &[Vec<f32>],
        state: &ModelState,
    ) -> Result<Vec<Vec<f32>>> {
        let features = rows.first().map_or(0, Vec::len);
        if features == 0 {
            return Err(SingeError::invalid_argument("rows without features").into());
        }

        for (name, table) in [("rows", rows), ("queries", queries)] {
            if let Some((index, row)) = table
                .iter()
                .enumerate()
                .find(|(_, row)| row.len() != features)
            {
                return Err(SingeError::length_mismatch(
                    format!("{name}[{index}]"),
                    features,
                    row.len(),
                )
                .into());
            }
        }

        let (context, predicted) = bucketed(rows, queries);
        let device = state.device()?;
        let shape = |count: usize| -> Result<Shape> {
            let count = i64::try_from(count).map_err(|_| SingeError::out_of_range("rows"))?;
            let features =
                i64::try_from(features).map_err(|_| SingeError::out_of_range("features"))?;
            Ok(Shape::create([count, features])?)
        };
        let outputs = self.predict(
            &Tensor::from_vec(device, shape(rows.len())?, context)?,
            targets,
            &Tensor::from_vec(device, shape(queries.len())?, predicted)?,
            state,
        )?;
        let width =
            usize::try_from(self.config.outputs).map_err(|_| Error::unsupported("outputs"))?;
        Ok(outputs
            .cast(DataType::F32)?
            .to_vec::<f32>()?
            .chunks(width)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl TabularClassifier for Mitra {
    // The softmax of each row's scores for the classes the labels hold, as `MitraClassifier.predict_proba` gives it without fine-tuning.
    fn classify(
        &self,
        rows: &[Vec<f32>],
        labels: &[i64],
        queries: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let layers = self.layer();
        let classes = labels
            .iter()
            .copied()
            .max()
            .map_or(0, |largest| largest + 1);
        if labels.len() != rows.len() {
            return Err(SingeError::length_mismatch("labels", rows.len(), labels.len()).into());
        }

        if labels.iter().any(|&label| label < 0) || classes > layers.config.outputs {
            return Err(SingeError::invalid_argument(
                "labels other than 0 to below the model's classes",
            )
            .into());
        }

        let count = i64::try_from(labels.len()).map_err(|_| SingeError::out_of_range("labels"))?;
        let device = self.state().device()?;
        let targets = Tensor::from_vec(device, Shape::create([count])?, labels.to_vec())?;
        let classes = usize::try_from(classes).map_err(|_| SingeError::out_of_range("labels"))?;
        Ok(layers
            .run(rows, &targets, queries, self.state())?
            .into_iter()
            .map(|scores| {
                let scores = &scores[..classes];
                // Less the largest score, which leaves the softmax as it is and keeps every exponent finite.
                let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponents = scores
                    .iter()
                    .map(|score| (score - largest).exp())
                    .collect::<Vec<f32>>();
                let total = exponents.iter().sum::<f32>();
                exponents.iter().map(|value| value / total).collect()
            })
            .collect())
    }
}

impl TabularRegressor for Mitra {
    // The targets scaled to `[0, 1]` by the context's range, and the predictions scaled back, as `MitraRegressor.predict` gives them without fine-tuning or mirroring.
    #[expect(
        clippy::suboptimal_flops,
        reason = "rounds each product as the reference does, which the verifications compare with"
    )]
    fn regress(
        &self,
        rows: &[Vec<f32>],
        targets: &[f32],
        queries: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if targets.len() != rows.len() {
            return Err(SingeError::length_mismatch("targets", rows.len(), targets.len()).into());
        }

        let smallest = targets.iter().copied().fold(f32::INFINITY, f32::min);
        let largest = targets.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if smallest >= largest {
            return Err(SingeError::invalid_argument("targets that do not vary").into());
        }

        let range = largest - smallest;
        let scaled = targets
            .iter()
            .map(|target| (target - smallest) / range)
            .collect::<Vec<f32>>();
        let count =
            i64::try_from(targets.len()).map_err(|_| SingeError::out_of_range("targets"))?;
        let targets = Tensor::from_vec(self.state().device()?, Shape::create([count])?, scaled)?;
        Ok(self
            .layer()
            .run(rows, &targets, queries, self.state())?
            .into_iter()
            .map(|outputs| outputs[0] * range + smallest)
            .collect())
    }
}
