//! The text encoder BERT, `DistilBERT`, and `RoBERTa` share: token embeddings plus learned positions and, where the model has them, token types, normed; then blocks that norm after each residual.
//! Its state is `embeddings` (`tokens`, `positions`, `token_types`, `norm`) and each block's `attention` (`query`, `key`, `value`, `output`), `attention_norm`, `feed_forward` (`up`, `down`), and `feed_forward_norm`.

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Shape, Tensor,
    graph::capture,
    nn::{
        Activation, DenseAttentionInput, Embedding, EncoderBlock, Layer, ModelState, SelfAttention,
        StateKey, TokenBatch, TokenEmbeddings,
    },
};

use crate::{
    format::{RuleSet, join},
    schema::Schema,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNormEncoderConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    /// The rows of the position table, from which a sequence reads its positions starting at `first_position`.
    pub max_positions: i64,
    pub first_position: i64,
    /// The token types the model embeds; 0 for a model without token type embeddings.
    pub token_types: i64,
    pub eps: f32,
    /// Which keys each query reads; `None` reads every key.
    #[serde(default)]
    pub keys: Option<KeyPattern>,
}

/// The keys each query reads, short of every key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyPattern {
    /// Keys at most this many positions from their query on either side, as Longformer's local attention without global tokens reads them.
    Band(i64),
    /// `BigBird`'s block-sparse attention as `transformers` runs it in evaluation on a sequence of more than `(5 + 2 · random) · size` tokens; a shorter sequence reads every key.
    ///
    /// The sequence splits into blocks of `size`, the last one short.
    /// The first and last blocks' queries read every key; the others read the first and last blocks and their own block's neighbours.
    /// Evaluation draws every random block as the first block, so it counts `1 + random` times in their softmax, a bias of `ln(1 + random)`.
    Blocks { size: i64, random: i64 },
}

impl PostNormEncoderConfig {
    #[must_use]
    pub const fn block(&self) -> PostNormBlockConfig {
        PostNormBlockConfig {
            width: self.width,
            heads: self.heads,
            inner: self.inner,
            activation: self.activation,
            eps: self.eps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PostNormBlockConfig {
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
}

/// The encoder's layers, from a batch of token sequences to each token's last hidden state, `[B, T, W]`, of which padded rows hold what attending to the real tokens gives.
#[derive(Debug, Clone)]
pub struct PostNormEncoder {
    embeddings: TokenEmbeddings,
    tokens: StateKey,
    blocks: Vec<EncoderBlock>,
    config: PostNormEncoderConfig,
}

impl PostNormEncoder {
    /// Creates the encoder, registering its state in `schema`.
    pub fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<Self> {
        let width = config.width;
        let (embeddings, tokens) = {
            let mut schema = schema.scope("embeddings");
            let tokens = schema.parameter("tokens.weight", [config.vocab_size, width])?;
            let positions = schema.parameter("positions.weight", [config.max_positions, width])?;
            let mut embeddings = TokenEmbeddings::new(Embedding::new(tokens.clone()))
                .with_positions(positions)
                .with_first_position(config.first_position);
            if config.token_types > 0 {
                let token_types =
                    schema.parameter("token_types.weight", [config.token_types, width])?;
                embeddings = embeddings.with_token_types(Embedding::new(token_types));
            }
            (
                embeddings.with_norm(schema.layer_norm("norm", width, config.eps)?),
                tokens,
            )
        };
        let block = config.block();
        let blocks = (0..config.layers)
            .map(|layer| {
                post_norm_encoder_block(&mut schema.scope(format!("blocks.{layer}")), &block)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            tokens,
            blocks,
            config: config.clone(),
        })
    }

    #[must_use]
    pub const fn config(&self) -> &PostNormEncoderConfig {
        &self.config
    }

    /// Returns the key of the token embedding table, `[vocab, width]`, which a masked language model's output may share.
    #[must_use]
    pub const fn tokens(&self) -> &StateKey {
        &self.tokens
    }
}

/// Registers one post-norm block, BERT's: `attention` (`query`, `key`, `value`, `output`), `attention_norm`, `feed_forward` (`up`, `down`), and `feed_forward_norm`.
pub fn post_norm_encoder_block(
    schema: &mut Schema<'_>,
    config: &PostNormBlockConfig,
) -> SingeResult<EncoderBlock> {
    let &PostNormBlockConfig {
        width,
        heads,
        inner,
        activation,
        eps,
    } = config;
    let [query, key, value, output] = schema
        .scope("attention")
        .attention_projections(width, [true; 4])?;
    let attention = SelfAttention::create(query, key, value, output, heads, width)?;
    let attention_norm = schema.layer_norm("attention_norm", width, eps)?;
    let (up, down) = {
        let mut schema = schema.scope("feed_forward");
        (
            schema.linear("up", width, inner, true)?,
            schema.linear("down", inner, width, true)?,
        )
    };
    let feed_forward_norm = schema.layer_norm("feed_forward_norm", width, eps)?;
    Ok(
        EncoderBlock::new(attention, attention_norm, up, down, feed_forward_norm)
            .with_activation(activation),
    )
}

impl Layer<TokenBatch> for PostNormEncoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = self.embeddings.forward(input, state)?;
        let key_lengths = input.key_lengths()?;
        let mask = match self.config.keys {
            None => None,
            Some(KeyPattern::Band(band)) => Some(band_mask(input, band, hidden.data_type())?),
            Some(KeyPattern::Blocks { size, random }) => {
                block_mask(input, size, random, hidden.data_type())?
            }
        };
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: mask.is_none().then_some(&key_lengths),
                    mask: mask.as_ref(),
                },
                state,
            )?;
        }
        Ok(hidden)
    }
}

/// Returns the additive mask of keys at most `band` positions from their query and within their sequence, `[B, 1, T, T]`: zero where a query reads a key, `F32`'s lowest value elsewhere, as Longformer adds `finfo.min` outside its window.
fn band_mask(input: &TokenBatch, band: i64, data_type: DataType) -> SingeResult<Tensor> {
    let tokens = input.tokens();
    let device = tokens.device();
    let positions = Tensor::range(device, 0_i64, tokens.dimension(1)?)?;
    let distance = positions
        .unsqueeze(1)?
        .sub(&positions.unsqueeze(0)?)?
        .cast(DataType::F32)?
        .abs()?;
    let reads = distance
        .less_equal(&capture::scalar(device, band as f32)?)?
        .unsqueeze(0)?
        .logical_and(&input.mask()?.unsqueeze(1)?)?
        .unsqueeze(1)?;
    Tensor::where_select(
        &reads,
        &capture::scalar(device, 0.0_f32)?,
        &capture::scalar(device, f32::MIN)?,
    )?
    .cast(data_type)
}

/// Returns `BigBird`'s additive block-sparse mask, `[B, 1, T, T]`, as [`KeyPattern::Blocks`] describes it, each sequence's blocks counted over its own tokens as `transformers` counts a sequence run alone, or `None` for a batch short enough that every sequence reads every key.
fn block_mask(
    input: &TokenBatch,
    size: i64,
    random: i64,
    data_type: DataType,
) -> SingeResult<Option<Tensor>> {
    let tokens = input.tokens();
    let length = tokens
        .shape()
        .dimensions()
        .get(1)
        .copied()
        .unwrap_or_default();
    let limit = (5 + 2 * random) * size;
    if length <= limit {
        return Ok(None);
    }

    let device = tokens.device();
    let count = (length + size - 1) / size;
    let blocks = Tensor::range(device, 0_i64, count)?
        .unsqueeze(1)?
        .expand(Shape::create([count, size])?)?
        .reshape([count * size])?
        .narrow(0, 0, length)?
        .cast(DataType::F32)?;
    let lengths = input.lengths();
    // Each sequence's last block, the block of its last token, `[B, 1, 1]`.
    let last = blocks
        .index_select(0, &lengths.sub_scalar(1)?)?
        .unsqueeze(1)?
        .unsqueeze(2)?;
    let short = lengths
        .cast(DataType::F32)?
        .less_equal(&capture::scalar(device, limit as f32)?)?
        .unsqueeze(1)?
        .unsqueeze(2)?;
    let first = capture::scalar(device, 0.0_f32)?;
    let is_end = |blocks: &Tensor| blocks.equal(&first)?.logical_or(&blocks.equal(&last)?);
    let (queries, keys) = (
        blocks.unsqueeze(1)?.unsqueeze(0)?,
        blocks.unsqueeze(0)?.unsqueeze(0)?,
    );
    let global_queries = is_end(&queries)?;
    let near = queries
        .sub(&keys)?
        .abs()?
        .less_equal(&capture::scalar(device, 1.0_f32)?)?;
    let reads = short
        .logical_or(&global_queries)?
        .logical_or(&is_end(&keys)?)?
        .logical_or(&near)?
        .logical_and(&input.mask()?.unsqueeze(1)?)?
        .unsqueeze(1)?;
    let repeated = keys
        .equal(&first)?
        .logical_and(&global_queries.logical_or(&short)?.logical_not()?)?;
    let bias = Tensor::where_select(
        &repeated,
        &capture::scalar(device, ((1 + random) as f32).ln())?,
        &first,
    )?;
    Ok(Some(
        Tensor::where_select(
            &reads,
            &bias.unsqueeze(1)?,
            &capture::scalar(device, f32::MIN)?,
        )?
        .cast(data_type)?,
    ))
}

/// Imports the encoder `transformers`' BERT layout stores under `from` (such as `bert`, `roberta`, or nothing): `embeddings` and `encoder.layer.N`, to the schema's names under `to`.
#[must_use]
pub fn post_norm_encoder_rules(from: &str, to: &str) -> RuleSet {
    let embeddings = join(from, "embeddings");
    RuleSet::new()
        .rename_each(
            &embeddings,
            &join(to, "embeddings"),
            &[
                ("word_embeddings", "tokens"),
                ("position_embeddings", "positions"),
                ("token_type_embeddings", "token_types"),
                ("LayerNorm", "norm"),
            ],
        )
        // Buffers of the positions and token types an input leaves out, which the embeddings compute.
        .skip(&join(&embeddings, "position_ids"))
        .skip(&join(&embeddings, "token_type_ids"))
        .rename_each(
            &join(from, "encoder.layer.*"),
            &join(to, "blocks.*"),
            &[
                ("attention.self.query", "attention.query"),
                ("attention.self.key", "attention.key"),
                ("attention.self.value", "attention.value"),
                ("attention.output.dense", "attention.output"),
                ("attention.output.LayerNorm", "attention_norm"),
                ("intermediate.dense", "feed_forward.up"),
                ("output.dense", "feed_forward.down"),
                ("output.LayerNorm", "feed_forward_norm"),
            ],
        )
        // The pooler serves only a classifier, whose rules read it first.
        .skip(&join(from, "pooler"))
}
