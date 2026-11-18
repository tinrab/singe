// BART's encoder and decoder stacks, which TAPEX, MBart's decoder in Donut, and TrOCR's decoder share: token embeddings, scaled or not, and learned positions offset by two, normed, then blocks that norm after each residual, or before as MBart's do with a norm after the last.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Activation, Embedding, LayerNorm, Linear, ModelState, StateKey},
    tensor::AttentionOptions,
};

use crate::{format::Rules, schema::Schema};

// The positions BART's learned position table skips before the first, as its padding once took them.
const POSITION_OFFSET: i64 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BartStackConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    // The positions the learned table covers.
    pub positions: i64,
    // Whether token embeddings are multiplied by `sqrt(width)`.
    pub scale_embedding: bool,
    pub embedding_norm: bool,
    // Whether blocks norm before each part, with a norm after the last block, as MBart's do, rather than after each residual.
    pub pre_norm: bool,
    // The width of the encoder states a decoder's cross-attention reads.
    pub cross_width: i64,
    pub eps: f32,
}

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl Attention {
    fn create(schema: &mut Schema<'_>, width: i64, context: i64, heads: i64) -> SingeResult<Self> {
        Ok(Self {
            query: schema.linear("query", width, width, true)?,
            key: schema.linear("key", context, width, true)?,
            value: schema.linear("value", context, width, true)?,
            output: schema.linear("output", width, width, true)?,
            heads,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        causal: bool,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let split = |tensor: Tensor| -> SingeResult<Tensor> {
            let dimensions = tensor.shape().dimensions();
            let (batch, length, width) = (dimensions[0], dimensions[1], dimensions[2]);
            tensor
                .reshape([batch, length, self.heads, width / self.heads])?
                .transpose(1, 2)
        };
        let attended = split(hidden.apply(&self.query, state)?)?.attend(
            &split(context.apply(&self.key, state)?)?,
            &split(context.apply(&self.value, state)?)?,
            None,
            None,
            AttentionOptions {
                causal,
                ..AttentionOptions::default()
            },
        )?;
        let dimensions = attended.shape().dimensions();
        attended
            .transpose(1, 2)?
            .reshape([dimensions[0], dimensions[2], dimensions[1] * dimensions[3]])?
            .apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct Block {
    attention: Attention,
    attention_norm: LayerNorm,
    cross: Option<(Attention, LayerNorm)>,
    up: Linear,
    down: Linear,
    feed_forward_norm: LayerNorm,
}

// One encoder or decoder stack.
#[derive(Debug, Clone)]
pub struct BartStack {
    tokens: Embedding,
    // What token embeddings are multiplied by.
    embed_scale: f64,
    positions: StateKey,
    embedding_norm: Option<LayerNorm>,
    blocks: Vec<Block>,
    norm: Option<LayerNorm>,
    config: BartStackConfig,
}

impl BartStack {
    // Registers a stack in `schema`: `tokens.weight`, `positions.weight`, `embedding_norm`, `blocks.N` with `attention`, `attention_norm`, a decoder's `cross_attention` and `cross_norm`, `up`, `down`, and `feed_forward_norm`, and a pre-norm stack's `norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        decoder: bool,
        tokens: Option<&StateKey>,
    ) -> SingeResult<Self> {
        let width = config.width;
        #[expect(clippy::cast_precision_loss, reason = "model widths are small")]
        let embed_scale = if config.scale_embedding {
            (width as f64).sqrt()
        } else {
            1.0
        };
        let tokens = match tokens {
            Some(shared) => schema.tie("tokens.weight", shared)?,
            None => schema.parameter("tokens.weight", [config.vocab_size, width])?,
        };
        let positions = schema.parameter(
            "positions.weight",
            [config.positions + POSITION_OFFSET, width],
        )?;
        let embedding_norm = config
            .embedding_norm
            .then(|| schema.layer_norm("embedding_norm", width, config.eps))
            .transpose()?;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                let cross = decoder
                    .then(|| -> SingeResult<_> {
                        Ok((
                            Attention::create(
                                &mut schema.scope("cross_attention"),
                                width,
                                config.cross_width,
                                config.heads,
                            )?,
                            schema.layer_norm("cross_norm", width, config.eps)?,
                        ))
                    })
                    .transpose()?;
                Ok(Block {
                    attention: Attention::create(
                        &mut schema.scope("attention"),
                        width,
                        width,
                        config.heads,
                    )?,
                    attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
                    cross,
                    up: schema.linear("up", width, config.inner, true)?,
                    down: schema.linear("down", config.inner, width, true)?,
                    feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = config
            .pre_norm
            .then(|| schema.layer_norm("norm", width, config.eps))
            .transpose()?;
        Ok(Self {
            tokens: Embedding::new(tokens),
            embed_scale,
            positions,
            embedding_norm,
            blocks,
            norm,
            config: config.clone(),
        })
    }

    // Runs the stack over `tokens`, `[1, T]`, against `encoded` for a decoder, returning each token's final state, `[1, T, W]`.
    pub fn forward(
        &self,
        tokens: &Tensor,
        encoded: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = tokens.shape().dimensions()[1];
        let positions = Tensor::from_vec(
            tokens.device(),
            Shape::create([length])?,
            (POSITION_OFFSET..length + POSITION_OFFSET).collect::<Vec<_>>(),
        )?;
        let table = state.tensor(&self.positions)?;
        let mut hidden = tokens
            .apply(&self.tokens, state)?
            .mul_scalar(self.embed_scale)?
            .add(&table.index_select(0, &positions)?.unsqueeze(0)?)?;
        if let Some(norm) = &self.embedding_norm {
            hidden = hidden.apply(norm, state)?;
        }

        let pre_norm = self.config.pre_norm;
        // A part with its residual, normed before it or after the sum.
        let part = |hidden: &Tensor,
                    norm: &LayerNorm,
                    run: &dyn Fn(&Tensor) -> SingeResult<Tensor>|
         -> SingeResult<Tensor> {
            if pre_norm {
                hidden.add(&run(&hidden.apply(norm, state)?)?)
            } else {
                hidden.add(&run(hidden)?)?.apply(norm, state)
            }
        };
        let decoder = encoded.is_some();
        for block in &self.blocks {
            hidden = part(&hidden, &block.attention_norm, &|input| {
                block.attention.forward(input, input, decoder, state)
            })?;
            if let (Some((attention, norm)), Some(encoded)) = (&block.cross, encoded) {
                hidden = part(&hidden, norm, &|input| {
                    attention.forward(input, encoded, false, state)
                })?;
            }

            hidden = part(&hidden, &block.feed_forward_norm, &|input| {
                input
                    .apply(&block.up, state)?
                    .apply(&self.config.activation, state)?
                    .apply(&block.down, state)
            })?;
        }
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }

    // Runs a decoder over `tokens` against `encoded`, returning the last token's final state, `[W]`.
    pub fn last_state(
        &self,
        encoded: &Tensor,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let count =
            i64::try_from(tokens.len()).map_err(|_| singe::Error::out_of_range("tokens"))?;
        let tokens = Tensor::from_vec(
            encoded.device(),
            Shape::create([1, count])?,
            tokens.to_vec(),
        )?;
        self.forward(&tokens, Some(encoded), state)?
            .narrow(1, count - 1, 1)?
            .squeeze(1)?
            .squeeze(0)
    }
}

// Imports a stack stored under `from`, as `transformers`' BART-family models name their parts, to the schema's names under `to`.
#[must_use]
pub fn stack_rules(from: &str, to: &str) -> Rules {
    let source = |name: &str| format!("{from}{name}");
    let target = |name: &str| format!("{to}{name}");
    let layer = |name: &str| format!("{from}layers.*.{name}");
    let block = |name: &str| format!("{to}blocks.*.{name}");
    let mut rules = Rules::new()
        .rename(&source("embed_tokens"), &target("tokens"))
        .rename(&source("embed_positions"), &target("positions"))
        .rename(&source("layernorm_embedding"), &target("embedding_norm"))
        .rename(&source("layer_norm"), &target("norm"));
    for (from, to) in [
        ("self_attn", "attention"),
        ("encoder_attn", "cross_attention"),
    ] {
        for (projection, name) in [
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            ("out_proj", "output"),
        ] {
            rules = rules.rename(
                &layer(&format!("{from}.{projection}")),
                &block(&format!("{to}.{name}")),
            );
        }
    }
    rules
        .rename(&layer("self_attn_layer_norm"), &block("attention_norm"))
        .rename(&layer("encoder_attn_layer_norm"), &block("cross_norm"))
        .rename(&layer("fc1"), &block("up"))
        .rename(&layer("fc2"), &block("down"))
        .rename(&layer("final_layer_norm"), &block("feed_forward_norm"))
}
