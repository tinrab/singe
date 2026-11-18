// The text encoder BERT, DistilBERT, and RoBERTa share: token embeddings plus learned positions and, where the model has them, token types, normed; then blocks that norm after each residual.
// Its state is `embeddings` (`tokens`, `positions`, `token_types`, `norm`) and each block's `attention` (`query`, `key`, `value`, `output`), `attention_norm`, `feed_forward` (`up`, `down`), and `feed_forward_norm`.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, DenseAttentionInput, Embedding, EncoderBlock, Layer, ModelState, SelfAttention,
        StateKey, TokenBatch, TokenEmbeddings,
    },
};

use crate::{format::Rules, schema::Schema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNormEncoderConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    // The rows of the position table, from which a sequence reads its positions starting at `first_position`.
    pub max_positions: i64,
    pub first_position: i64,
    // The token types the model embeds; none for a model without token type embeddings.
    pub token_types: i64,
    pub eps: f32,
}

// The encoder's layers, from a batch of token sequences to each token's last hidden state, `[B, T, W]`, of which padded rows hold what attending to the real tokens gives.
#[derive(Debug, Clone)]
pub struct PostNormEncoder {
    embeddings: TokenEmbeddings,
    tokens: StateKey,
    blocks: Vec<EncoderBlock>,
    config: PostNormEncoderConfig,
}

impl PostNormEncoder {
    // Registers the encoder's state in `schema`.
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
        let blocks = (0..config.layers)
            .map(|layer| block(&mut schema.scope(format!("blocks.{layer}")), config))
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

    // The token embedding table, `[vocab, width]`, which a masked language model's output may share.
    #[must_use]
    pub const fn tokens(&self) -> &StateKey {
        &self.tokens
    }
}

fn block(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<EncoderBlock> {
    encoder_block(
        schema,
        config.width,
        config.heads,
        config.inner,
        config.activation,
        config.eps,
    )
}

// Registers one post-norm block, BERT's: `attention` (`query`, `key`, `value`, `output`), `attention_norm`, `feed_forward` (`up`, `down`), and `feed_forward_norm`.
pub fn encoder_block(
    schema: &mut Schema<'_>,
    width: i64,
    heads: i64,
    inner: i64,
    activation: Activation,
    eps: f32,
) -> SingeResult<EncoderBlock> {
    let attention = {
        let mut schema = schema.scope("attention");
        SelfAttention::create(
            schema.linear("query", width, width, true)?,
            schema.linear("key", width, width, true)?,
            schema.linear("value", width, width, true)?,
            schema.linear("output", width, width, true)?,
            heads,
            width,
        )?
    };
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

// Renames a LayerNorm, whose parameters older checkpoints name `gamma` and `beta`.
fn norm(rules: Rules, from: &str, to: &str) -> Rules {
    rules
        .rename(&format!("{from}.gamma"), &format!("{to}.weight"))
        .rename(&format!("{from}.beta"), &format!("{to}.bias"))
        .rename(from, to)
}

// Imports the encoder `transformers`' BERT layout stores after `prefix` (such as `bert.`, `roberta.`, or nothing): `embeddings` and `encoder.layer.N`, to the schema's names under `to`.
#[must_use]
pub fn post_norm_encoder_rules(prefix: &str, to: &str) -> Rules {
    let target = |name: &str| format!("{to}{name}");
    let embeddings = format!("{prefix}embeddings");
    let layer = |name: &str| format!("{prefix}encoder.layer.*.{name}");
    let rules = Rules::new()
        .rename(
            &format!("{embeddings}.word_embeddings"),
            &target("embeddings.tokens"),
        )
        .rename(
            &format!("{embeddings}.position_embeddings"),
            &target("embeddings.positions"),
        )
        .rename(
            &format!("{embeddings}.token_type_embeddings"),
            &target("embeddings.token_types"),
        )
        // Buffers of the positions and token types an input leaves out, which the embeddings compute.
        .skip(&format!("{embeddings}.position_ids"))
        .skip(&format!("{embeddings}.token_type_ids"))
        .rename(
            &layer("attention.self.query"),
            &target("blocks.*.attention.query"),
        )
        .rename(
            &layer("attention.self.key"),
            &target("blocks.*.attention.key"),
        )
        .rename(
            &layer("attention.self.value"),
            &target("blocks.*.attention.value"),
        )
        .rename(
            &layer("attention.output.dense"),
            &target("blocks.*.attention.output"),
        )
        .rename(
            &layer("intermediate.dense"),
            &target("blocks.*.feed_forward.up"),
        )
        .rename(
            &layer("output.dense"),
            &target("blocks.*.feed_forward.down"),
        )
        // The pooler serves only a classifier, whose rules read it first.
        .skip(&format!("{prefix}pooler"));
    [
        (
            format!("{embeddings}.LayerNorm"),
            &target("embeddings.norm"),
        ),
        (
            layer("attention.output.LayerNorm"),
            &target("blocks.*.attention_norm"),
        ),
        (
            layer("output.LayerNorm"),
            &target("blocks.*.feed_forward_norm"),
        ),
    ]
    .into_iter()
    .fold(rules, |rules, (from, to)| norm(rules, &from, to))
}
