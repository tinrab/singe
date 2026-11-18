// ModernBERT: an encoder of pre-norm blocks without biases, rotary positions, and a gated feed-forward layer, whose blocks attend to every token or, in its local layers, only to tokens within a window on either side.
//
// Its state is `tokens`, `embedding_norm`, `blocks.N` (`attention_norm` but in the first, `attention` with `query`, `key`, `value`, and `output`, `feed_forward_norm`, and `feed_forward` with `gate`, `up`, and `down`), and `norm`.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Embedding, GatedMlp, HeadProjection, LayerNorm, Linear, ModelState,
        RotaryAngles, RotaryFrequencies, RotaryLayout, RotarySpec, RotaryTable,
    },
    tensor::AttentionOptions,
};

use crate::schema::Schema;

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
    query: HeadProjection,
    key: HeadProjection,
    value: HeadProjection,
    output: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward: GatedMlp,
    local: bool,
}

impl ModernBertBlock {
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
        let dimensions = hidden.shape().dimensions();
        let attended = heads(&self.query, true)?
            .attend(
                &heads(&self.key, true)?,
                &heads(&self.value, false)?,
                mask,
                None,
                AttentionOptions::default(),
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([dimensions[0], dimensions[1], dimensions[2]])?
            .apply(&self.output, state)?;
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
        let head_dim = width / config.heads;
        let blocks = config
            .local
            .iter()
            .enumerate()
            .map(|(index, &local)| {
                let mut block = schema.scope(format!("blocks.{index}"));
                // The first block reads the normed embeddings as they are.
                let attention_norm = (index > 0)
                    .then(|| block.unbiased_layer_norm("attention_norm", width, eps))
                    .transpose()?;
                let mut attention = block.scope("attention");
                let mut head = |name: &str| -> SingeResult<HeadProjection> {
                    Ok(HeadProjection::new(
                        attention.linear(name, width, width, false)?,
                        config.heads,
                        head_dim,
                    ))
                };
                let (query, key, value) = (head("query")?, head("key")?, head("value")?);
                let output = attention.linear("output", width, width, false)?;
                let feed_forward_norm =
                    block.unbiased_layer_norm("feed_forward_norm", width, eps)?;
                let mut feed_forward = block.scope("feed_forward");
                let feed_forward = GatedMlp::new(
                    feed_forward.linear("gate", width, config.inner, false)?,
                    feed_forward.linear("up", width, config.inner, false)?,
                    feed_forward.linear("down", config.inner, width, false)?,
                    config.activation,
                );
                Ok(ModernBertBlock {
                    attention_norm,
                    query,
                    key,
                    value,
                    output,
                    feed_forward_norm,
                    feed_forward,
                    local,
                })
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
        let length = embeddings.shape().dimensions()[0];
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
        let size = usize::try_from(length).map_err(|_| singe::Error::out_of_range("tokens"))?;
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
