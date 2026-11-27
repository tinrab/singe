//! DeBERTa-v2 and v3: a post-norm encoder without absolute positions, whose attention adds to each content score the query's score against the key's relative position and the key's against the query's, from one normed table of relative embeddings that every layer projects with its own query and key projections.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    graph::{DimensionExpr, capture},
    nn::{Activation, Layer, LayerNorm, Linear, Mlp, ModelState, StateKey, TokenBatch},
};

use crate::{
    assembly::{merge_heads, split_heads},
    head::{EncoderBody, EncoderSizes, encoder_heads},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, Copy)]
pub struct DebertaV2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebertaV2Config {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    /// Half the relative table's rows: the span of relative positions on each side.
    pub buckets: i64,
    /// The distance from which buckets grow logarithmically up to it.
    pub max_distance: i64,
}

#[derive(Debug, Clone)]
struct Block {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

/// DeBERTa-v2's encoder, from tokens to each token's last hidden state, `[B, T, W]`.
#[derive(Debug, Clone)]
pub struct DebertaV2Encoder {
    tokens: StateKey,
    embedding_norm: LayerNorm,
    relative: StateKey,
    relative_norm: LayerNorm,
    blocks: Vec<Block>,
    config: DebertaV2Config,
}

impl DebertaV2Encoder {
    pub fn create(schema: &mut Schema<'_>, config: &DebertaV2Config) -> SingeResult<Self> {
        let width = config.width;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                Ok(Block {
                    query: schema.linear("attention.query", width, width, true)?,
                    key: schema.linear("attention.key", width, width, true)?,
                    value: schema.linear("attention.value", width, width, true)?,
                    output: schema.linear("attention.output", width, width, true)?,
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
        Ok(Self {
            tokens: schema.parameter("embeddings.tokens.weight", [config.vocab_size, width])?,
            embedding_norm: schema.layer_norm("embeddings.norm", width, config.eps)?,
            relative: schema.parameter("relative.weight", [2 * config.buckets, width])?,
            relative_norm: schema.layer_norm("relative_norm", width, config.eps)?,
            blocks,
            config: config.clone(),
        })
    }

    /// Returns the rows of the two position tables each score reads, `[L · L]` in `I64`, as offsets into the scores' flattened last two axes, `[L, 2 · buckets]`: for the query's score against the key's position, `q · 2b + clamp(bucket(q − k) + b)`, and for the key's against the query's, `k · 2b + clamp(b − bucket(k − q))` at `[k, q]`.
    fn positions(&self, device: Device, count: DimensionExpr) -> SingeResult<[Tensor; 2]> {
        let DebertaV2Config {
            buckets,
            max_distance,
            ..
        } = self.config;
        let indices = Tensor::range(device, 0_i64, count)?;
        let distances = indices
            .unsqueeze(1)?
            .sub(&indices.unsqueeze(0)?)?
            .cast(DataType::F32)?;
        // `make_log_bucket_position` in `F32`: distances within half the buckets keep their value, longer ones grow logarithmically to the largest distance.
        let middle = (buckets / 2) as f32;
        let magnitude = distances.abs()?;
        let near = magnitude.less(&capture::scalar(device, middle)?)?;
        let magnitude =
            Tensor::where_select(&near, &capture::scalar(device, middle - 1.0)?, &magnitude)?;
        let ratio = ((max_distance - 1) as f64 / f64::from(middle)) as f32;
        let logarithmic = magnitude
            .div_scalar(f64::from(middle))?
            .log()?
            .div(&capture::scalar(device, ratio.ln())?)?
            .mul_scalar(f64::from(middle - 1.0))?
            .ceil()?
            .add_scalar(f64::from(middle))?;
        let negative = distances.less(&capture::scalar(device, 0.0_f32)?)?;
        let signed = Tensor::where_select(&negative, &logarithmic.neg()?, &logarithmic)?;
        let bucket = Tensor::where_select(
            &magnitude.less_equal(&capture::scalar(device, middle)?)?,
            &distances,
            &signed,
        )?;
        let rows = 2 * buckets;
        // `F32` holds every offset exactly, as they stay below `L · 2b`.
        let offsets = indices
            .cast(DataType::F32)?
            .unsqueeze(1)?
            .mul_scalar(rows as f64)?;
        let row = |bucket: &Tensor| -> SingeResult<Tensor> {
            bucket
                .add_scalar(buckets as f64)?
                .clamp(0.0, (rows - 1) as f32)?
                .add(&offsets)?
                .cast(DataType::I64)?
                .flatten(0, 1)
        };
        // The bucket function is odd, so `−bucket(k − q)` at `[k, q]` is the buckets transposed.
        Ok([row(&bucket)?, row(&bucket.transpose(0, 1)?)?])
    }
}

/// Returns `scores`, `[B, H, L, 2b]`, gathered at `rows` of their flattened last two axes, `[B, H, L, L]`.
fn gathered(scores: &Tensor, rows: &Tensor) -> SingeResult<Tensor> {
    let shape = [
        scores.dimension(0)?,
        scores.dimension(1)?,
        scores.dimension(2)?,
        scores.dimension(2)?,
    ];
    scores
        .flatten(2, 3)?
        .index_select(2, rows)?
        .reshape_with(&shape)
}

impl Layer<TokenBatch> for DebertaV2Encoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let tokens = input.tokens();
        let device = tokens.device();
        let real = input.mask()?;
        let table = state.tensor(&self.tokens)?;
        let width = table.dimension(1)?;
        let mut hidden = table
            .index_select(0, &tokens.flatten(0, 1)?)?
            .reshape_with(&[tokens.dimension(0)?, tokens.dimension(1)?, width])?
            .apply(&self.embedding_norm, state)?;
        hidden = hidden.mul(&real.unsqueeze(2)?.cast(hidden.data_type())?)?;

        // A score between two real tokens stays; any other becomes the lowest value, as `masked_fill` with `finfo.min` makes it.
        let pairs = real
            .unsqueeze(2)?
            .logical_and(&real.unsqueeze(1)?)?
            .unsqueeze(1)?;
        let lowest = capture::scalar(device, f32::MIN)?.cast(hidden.data_type())?;
        let relative = state
            .tensor(&self.relative)?
            .apply(&self.relative_norm, state)?
            .unsqueeze(0)?;
        let [query_rows, key_rows] = self.positions(device, tokens.dimension(1)?)?;
        let heads = self.config.heads;
        let head = self.config.width / heads;
        // `scaled_size_sqrt` with content, content-to-position, and position-to-content terms.
        let scale = f64::from(((head * 3) as f32).sqrt());
        for block in &self.blocks {
            let project =
                |input: &Tensor, linear: &Linear| split_heads(&input.apply(linear, state)?, heads);
            let query = project(&hidden, &block.query)?;
            let key = project(&hidden, &block.key)?;
            let value = project(&hidden, &block.value)?;
            let position_query = project(&relative, &block.query)?;
            let position_key = project(&relative, &block.key)?;
            let content = query.matmul(&key.transpose(2, 3)?.div_scalar(scale)?)?;
            let to_position =
                gathered(&query.matmul(&position_key.transpose(2, 3)?)?, &query_rows)?;
            let from_position =
                gathered(&key.matmul(&position_query.transpose(2, 3)?)?, &key_rows)?
                    .transpose(2, 3)?;
            let scores = content.add(
                &to_position
                    .div_scalar(scale)?
                    .add(&from_position.div_scalar(scale)?)?,
            )?;
            let scores = Tensor::where_select(&pairs, &scores, &lowest)?;
            let attended = merge_heads(&scores.softmax(-1)?.matmul(&value)?)?;
            hidden = attended
                .apply(&block.output, state)?
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

/// Creates DeBERTa-v2's encoder as the body heads sit on.
pub fn body(schema: &mut Schema<'_>, config: &DebertaV2Config) -> SingeResult<EncoderBody> {
    let encoder = DebertaV2Encoder::create(schema, config)?;
    Ok(EncoderBody {
        sizes: EncoderSizes {
            width: config.width,
            vocab_size: config.vocab_size,
            eps: config.eps,
            activation: config.activation,
        },
        tokens: encoder.tokens.clone(),
        layers: Box::new(encoder),
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "deberta_v2",
    formats: &[(Format::Transformers, &["deberta-v2"])],
    loaders: &encoder_heads::<DebertaV2>(),
};
