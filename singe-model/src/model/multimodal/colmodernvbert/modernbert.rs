// ModernBERT: an encoder of pre-norm blocks without biases, rotary positions, and a gated feed-forward layer, whose blocks attend to every token or, in its local layers, only to tokens within a window on either side.
//
// Its state is `tokens`, `embedding_norm`, `blocks.N` (`attention_norm` but in the first, `attention` with `query`, `key`, `value`, and `output`, `feed_forward_norm`, and `feed_forward` with `gate`, `up`, and `down`), and `norm`.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Embedding, GatedMlp, GroupedQueryLayers, HeadProjection, LayerNorm, ModelState,
        RotaryAngles, RotaryFrequencies, RotaryLayout, RotarySpec, RotaryTable,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{GroupedAttentionSpec, grouped_attention_layers},
    schema::Schema,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub inner: i64,
    pub heads: i64,
    pub eps: f32,
    pub activation: Activation,
    // Whether each block attends locally, in order.
    pub local: Vec<bool>,
    // How far a local block's tokens see on either side.
    pub window: i64,
    pub global_theta: f64,
    pub local_theta: f64,
}

#[derive(Debug, Clone)]
struct ModernBertBlock {
    attention_norm: Option<LayerNorm>,
    attention: GroupedQueryLayers,
    feed_forward_norm: LayerNorm,
    feed_forward: GatedMlp,
    local: bool,
}

impl ModernBertBlock {
    fn create(
        schema: &mut Schema<'_>,
        config: &ModernBertConfig,
        index: usize,
        local: bool,
    ) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        // The first block reads the normed embeddings as they are.
        let attention_norm = (index > 0)
            .then(|| schema.unbiased_layer_norm("attention_norm", width, eps))
            .transpose()?;
        let attention = grouped_attention_layers(
            &mut schema.scope("attention"),
            &GroupedAttentionSpec::multi_head(width, config.heads, false),
        )?;
        Ok(Self {
            attention_norm,
            attention,
            feed_forward_norm: schema.unbiased_layer_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.gated_mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                false,
            )?,
            local,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        angles: &RotaryAngles,
        mask: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let normed = match &self.attention_norm {
            Some(norm) => hidden.apply(norm, state)?,
            None => hidden.clone(),
        };
        let heads = |projection: &HeadProjection, rotate: bool| -> SingeResult<Tensor> {
            let heads = normed.apply(projection, state)?;
            let heads = if rotate {
                angles.rotate(&heads)?
            } else {
                heads
            };
            heads.permute(&[0, 2, 1, 3])
        };
        let &[batch, length, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a ModernBERT block's input"));
        };
        let attention = &self.attention;
        let attended = heads(&attention.query, true)?
            .attend(
                &heads(&attention.key, true)?,
                &heads(&attention.value, false)?,
                mask,
                None,
                AttentionOptions::default(),
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([batch, length, width])?
            .apply(&attention.output, state)?;
        let hidden = hidden.add(&attended)?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?;
        hidden.add(&fed)
    }
}

#[derive(Debug, Clone)]
pub struct ModernBert {
    config: ModernBertConfig,
    tokens: Embedding,
    embedding_norm: LayerNorm,
    blocks: Vec<ModernBertBlock>,
    norm: LayerNorm,
}

impl ModernBert {
    pub fn create(schema: &mut Schema<'_>, config: &ModernBertConfig) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        let blocks = config
            .local
            .iter()
            .enumerate()
            .map(|(index, &local)| {
                ModernBertBlock::create(
                    &mut schema.scope(format!("blocks.{index}")),
                    config,
                    index,
                    local,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            tokens: Embedding::new(schema.parameter("tokens.weight", [config.vocab_size, width])?),
            embedding_norm: schema.unbiased_layer_norm("embedding_norm", width, eps)?,
            blocks,
            norm: schema.unbiased_layer_norm("norm", width, eps)?,
        })
    }

    // Each token's embedding, `[T, W]`, before the norm `encode` applies.
    pub fn embed(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        tokens.apply(&self.tokens, state)
    }

    // Encodes one sequence's embeddings, `[T, W]`, into its tokens' final states, `[T, W]`.
    pub fn encode(&self, embeddings: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let config = &self.config;
        let device = embeddings.device();
        let data_type = embeddings.data_type();
        let &[length, _] = embeddings.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a sequence's embeddings"));
        };
        let positions = Tensor::from_vec(
            device,
            Shape::create([length])?,
            (0..length).collect::<Vec<i64>>(),
        )?;
        let angles = |theta: f64| -> SingeResult<RotaryAngles> {
            RotaryTable::create(
                device,
                data_type,
                RotarySpec {
                    head_dim: config.width / config.heads,
                    frequencies: RotaryFrequencies::new(theta),
                    layout: RotaryLayout::HalfSplit,
                },
                length,
            )?
            .angles(&positions)
        };
        let (global, local) = (angles(config.global_theta)?, angles(config.local_theta)?);
        // A local block's tokens see those at most `window` places away, as `transformers`' bidirectional sliding mask allows.
        let size = usize::try_from(length).map_err(|_| SingeError::out_of_range("tokens"))?;
        let window = config.window;
        let mask = Tensor::from_vec(
            device,
            Shape::create([1, 1, length, length])?,
            (0..size * size)
                .map(|index| {
                    let (row, column) = (index / size, index % size);
                    if i64::try_from(row.abs_diff(column)).is_ok_and(|distance| distance <= window)
                    {
                        0.0_f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let mut hidden = embeddings
            .apply(&self.embedding_norm, state)?
            .unsqueeze(0)?;
        for block in &self.blocks {
            hidden = if block.local {
                block.forward(&hidden, &local, Some(&mask), state)?
            } else {
                block.forward(&hidden, &global, None, state)?
            };
        }
        hidden.squeeze(0)?.apply(&self.norm, state)
    }
}
