// BART's encoder and decoder stacks, which TAPEX, MBart's decoder in Donut, and TrOCR's decoder share: token embeddings, scaled or not, and learned positions offset by two, normed, then blocks that norm after each residual, or before as MBart's do with a norm after the last.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, DecodeStep, Embedding, LayerNorm, Linear, Mlp, ModelState, StateKey},
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        Attention, AttentionSpec, Seq2SeqCache, Seq2SeqCacheSpec, Seq2SeqTables, StepRows,
        attention_rules,
    },
    format::{Rules, join},
    head::Seq2SeqDecoding,
    schema::Schema,
};

// The positions BART's learned position table skips before the first, as its padding once took them.
const POSITION_OFFSET: i64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackRole {
    Encoder,
    Decoder,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BartStackConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    // The positions the learned table covers.
    pub max_positions: i64,
    // What token embeddings are multiplied by.
    pub embedding_scale: f64,
    pub embedding_norm: bool,
    // Whether blocks norm before each part, with a norm after the last block, as MBart's do, rather than after each residual.
    pub pre_norm: bool,
    // The width of the encoder states a decoder's cross-attention reads.
    pub cross_width: i64,
    pub eps: f32,
}

#[must_use]
pub fn embedding_scale(scale_embedding: bool, width: i64) -> f64 {
    if scale_embedding {
        (width as f64).sqrt()
    } else {
        1.0
    }
}

#[derive(Debug, Clone)]
struct BartBlock {
    attention_norm: LayerNorm,
    attention: Attention,
    cross: Option<(LayerNorm, Attention)>,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

impl BartBlock {
    fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        role: StackRole,
    ) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let cross = match role {
            StackRole::Encoder => None,
            StackRole::Decoder => Some((
                schema.layer_norm("cross_attention_norm", width, eps)?,
                Attention::create(
                    &mut schema.scope("cross_attention"),
                    AttentionSpec {
                        context: config.cross_width,
                        ..AttentionSpec::square(width, config.heads)
                    },
                )?,
            )),
        };
        Ok(Self {
            attention_norm: schema.layer_norm("attention_norm", width, eps)?,
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, config.heads),
            )?,
            cross,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                true,
            )?,
        })
    }
}

// One encoder or decoder stack.
#[derive(Debug, Clone)]
pub struct BartStack {
    tokens: Embedding,
    positions: StateKey,
    embedding_norm: Option<LayerNorm>,
    blocks: Vec<BartBlock>,
    norm: Option<LayerNorm>,
    config: BartStackConfig,
}

impl BartStack {
    // Registers a stack in `schema`: `tokens.weight`, `positions.weight`, `embedding_norm`, `blocks.N` with `attention_norm`, `attention`, a decoder's `cross_attention_norm` and `cross_attention`, `feed_forward_norm`, and `feed_forward`, and a pre-norm stack's `norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        config: &BartStackConfig,
        role: StackRole,
        tokens: Option<&StateKey>,
    ) -> SingeResult<Self> {
        let width = config.width;
        let tokens = match tokens {
            Some(shared) => schema.tie("tokens.weight", shared)?,
            None => schema.parameter("tokens.weight", [config.vocab_size, width])?,
        };
        let positions = schema.parameter(
            "positions.weight",
            [config.max_positions + POSITION_OFFSET, width],
        )?;
        let embedding_norm = config
            .embedding_norm
            .then(|| schema.layer_norm("embedding_norm", width, config.eps))
            .transpose()?;
        let blocks = (0..config.layers)
            .map(|index| {
                BartBlock::create(&mut schema.scope(format!("blocks.{index}")), config, role)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = config
            .pre_norm
            .then(|| schema.layer_norm("norm", width, config.eps))
            .transpose()?;
        Ok(Self {
            tokens: Embedding::new(tokens),
            positions,
            embedding_norm,
            blocks,
            norm,
            config: config.clone(),
        })
    }

    // Runs an encoder over `tokens`, `[1, T]`, returning each token's final state, `[1, T, W]`.
    pub fn forward(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, length] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a stack's tokens"));
        };
        let positions = Tensor::from_vec(
            tokens.device(),
            Shape::create([length])?,
            (0..length).collect::<Vec<_>>(),
        )?;
        self.run(&self.embed(tokens, &positions, state)?, None, state)
    }

    // Runs a decoder over a decode step's tokens, reading and extending `caches`, and returns each row's final state, `[T, W]`.
    pub fn decode(
        &self,
        step: &DecodeStep,
        rows: &StepRows,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let hidden = self.embed(step.tokens(), step.positions(), state)?;
        self.run(&hidden, Some((rows, caches)), state)
    }

    // Each token's embedding, scaled, with its position's added and normed when the stack norms its embeddings.
    fn embed(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let table =
            state
                .tensor(&self.positions)?
                .narrow(0, POSITION_OFFSET, self.config.max_positions)?;
        let hidden = tokens
            .apply(&self.tokens, state)?
            .mul_scalar(self.config.embedding_scale)?
            .add(&table.index_select(0, positions)?)?;
        match &self.embedding_norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }

    // Runs the blocks over `hidden`, an encoder's whole input, or a decode step's rows, which read and extend each block's cache.
    fn run(
        &self,
        hidden: &Tensor,
        step: Option<(&StepRows, &[Seq2SeqCache])>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        // A part's input, normed before it when blocks norm first.
        let pre_norm = self.config.pre_norm;
        let input = |hidden: &Tensor, norm: &LayerNorm| {
            if pre_norm {
                hidden.apply(norm, state)
            } else {
                Ok(hidden.clone())
            }
        };
        // A part's output added to its residual, normed after the sum when blocks norm last.
        let residual = |hidden: &Tensor, output: &Tensor, norm: &LayerNorm| {
            let sum = hidden.add(output)?;
            if pre_norm {
                Ok(sum)
            } else {
                sum.apply(norm, state)
            }
        };
        let mut hidden = hidden.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            let cached = step
                .map(|(rows, caches)| {
                    caches
                        .get(index)
                        .map(|cache| (rows, cache))
                        .ok_or_else(|| SingeError::missing_input("a decoder block's cache"))
                })
                .transpose()?;
            let normed = input(&hidden, &block.attention_norm)?;
            let attended = match cached {
                Some((rows, cache)) => block.attention.decode(
                    &normed,
                    rows,
                    cache,
                    rows.mask(),
                    AttentionOptions::default(),
                    state,
                )?,
                None => block.attention.forward(&normed, &normed, &normed, state)?,
            };
            hidden = residual(&hidden, &attended, &block.attention_norm)?;
            if let (Some((norm, attention)), Some((rows, cache))) = (&block.cross, cached) {
                let normed = input(&hidden, norm)?;
                let attended = attention.attend_sources(
                    &normed,
                    rows,
                    cache,
                    AttentionOptions::default(),
                    state,
                )?;
                hidden = residual(&hidden, &attended, norm)?;
            }

            let normed = input(&hidden, &block.feed_forward_norm)?;
            let output = normed.apply(&block.feed_forward, state)?;
            hidden = residual(&hidden, &output, &block.feed_forward_norm)?;
        }
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

// A BART-family decoder with its output: a decoder stack, the head over its last states, and the bias BART adds to the logits, `[1, V]`.
#[derive(Debug, Clone)]
pub struct BartDecoder {
    stack: BartStack,
    head: Linear,
    logits_bias: Option<StateKey>,
}

impl BartDecoder {
    #[must_use]
    pub const fn new(stack: BartStack, head: Linear, logits_bias: Option<StateKey>) -> Self {
        Self {
            stack,
            head,
            logits_bias,
        }
    }
}

impl Seq2SeqDecoding for BartDecoder {
    fn cross_attentions(&self) -> Vec<&Attention> {
        self.stack
            .blocks
            .iter()
            .filter_map(|block| block.cross.as_ref().map(|(_, attention)| attention))
            .collect()
    }

    fn cache_spec(&self, sources: i64) -> Seq2SeqCacheSpec {
        let config = &self.stack.config;
        Seq2SeqCacheSpec {
            heads: config.heads,
            head_dim: config.width / config.heads,
            sources,
        }
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Seq2SeqTables,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let rows = StepRows::create(step, tables)?;
        let logits = self
            .stack
            .decode(step, &rows, caches, state)?
            .index_select(0, step.last_rows())?
            .apply(&self.head, state)?;
        match &self.logits_bias {
            Some(bias) => logits.add(state.tensor(bias)?),
            None => Ok(logits),
        }
    }
}

// Imports a stack stored under `from`, as `transformers`' BART-family models name their parts, to the schema's names under `to`.
#[must_use]
pub fn bart_stack_rules(from: &str, to: &str) -> Rules {
    let (layers, blocks) = (join(from, "layers.*"), join(to, "blocks.*"));
    Rules::new()
        .rename_each(
            from,
            to,
            &[
                ("embed_tokens", "tokens"),
                ("embed_positions", "positions"),
                ("layernorm_embedding", "embedding_norm"),
                ("layer_norm", "norm"),
            ],
        )
        .then(attention_rules(
            &join(&layers, "self_attn"),
            &join(&blocks, "attention"),
        ))
        .then(attention_rules(
            &join(&layers, "encoder_attn"),
            &join(&blocks, "cross_attention"),
        ))
        .rename_each(
            &layers,
            &blocks,
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
}
