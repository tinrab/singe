//! The text encoder BERT, `DistilBERT`, and `RoBERTa` share: token embeddings plus learned positions and, where the model has them, token types, normed; then blocks that norm after each residual.
//! Its state is `embeddings` (`tokens`, `positions`, `token_types`, `norm`) and each block's `attention` (`query`, `key`, `value`, `output`), `attention_norm`, `feed_forward` (`up`, `down`), and `feed_forward_norm`.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, DenseAttentionInput, Embedding, EncoderBlock, Layer, ModelState, SelfAttention,
        StateKey, TokenBatch, TokenEmbeddings,
    },
};

use crate::{
    format::{Rules, join},
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
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: Some(&key_lengths),
                },
                state,
            )?;
        }
        Ok(hidden)
    }
}

/// Imports the encoder `transformers`' BERT layout stores under `from` (such as `bert`, `roberta`, or nothing): `embeddings` and `encoder.layer.N`, to the schema's names under `to`.
#[must_use]
pub fn post_norm_encoder_rules(from: &str, to: &str) -> Rules {
    let embeddings = join(from, "embeddings");
    Rules::new()
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
