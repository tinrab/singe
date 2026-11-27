//! Janus's VQ decoder, as `JanusVQVAE.decode` runs it: each image token's codebook entry, normalized to unit length, a 1 × 1 projection, then a taming-VQGAN decoder of group-normed residual blocks, single-head attention at the coarsest level, and nearest upsampling between levels.

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, GroupNorm, ModelState, StateKey},
    tensor::NormOrder,
};

use crate::{
    assembly::{GroupNormSpec, MiddleBlock, ResnetBlock, SpatialAttention, UpsampleBlock, same_2d},
    schema::Schema,
};

/// The norms of every block, as `JanusVQVAEResnetBlock` and `JanusVQVAEAttnBlock` make them.
const NORMS: GroupNormSpec = GroupNormSpec {
    groups: 32,
    eps: 1e-6,
};

/// The smallest norm a codebook entry is divided by, `F.normalize`'s `eps`.
const LEAST_NORM: f32 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JanusVqConfig {
    /// The codebook's entries.
    pub codebook: i64,
    /// The width of each codebook entry.
    pub entry: i64,
    pub latent: i64,
    pub base: i64,
    /// Each level's channels as multiples of `base`, from the finest.
    pub multipliers: Vec<i64>,
    pub blocks: i64,
    /// The side of the grid of image tokens.
    pub side: i64,
    pub channels: i64,
}

/// One level of the decoder, from the coarsest: residual blocks, attention after each at the coarsest, and an upsampling into the next.
#[derive(Debug, Clone)]
struct Level {
    blocks: Vec<ResnetBlock>,
    attentions: Vec<SpatialAttention>,
    upsample: Option<UpsampleBlock>,
}

#[derive(Debug, Clone)]
pub struct VqDecoder {
    codebook: StateKey,
    post_quant: Conv2d,
    conv_in: Conv2d,
    middle: MiddleBlock<SpatialAttention>,
    levels: Vec<Level>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    side: i64,
}

impl VqDecoder {
    pub fn create(schema: &mut Schema<'_>, config: &JanusVqConfig) -> SingeResult<Self> {
        let count = config.multipliers.len();
        let channels_at = |level: usize| config.base * config.multipliers[level];
        let mut input = channels_at(count - 1);
        let conv_in = schema.conv2d("conv_in", config.latent, input, 3, same_2d(3, 1), true)?;
        let middle = MiddleBlock::create(
            &mut schema.scope("middle"),
            input,
            None,
            NORMS,
            1.0,
            |schema| SpatialAttention::create(schema, input, 1, NORMS, 1.0),
        )?;
        let mut levels = Vec::with_capacity(count);
        for index in 0..count {
            let output = channels_at(count - 1 - index);
            let coarsest = index == 0;
            let mut blocks = Vec::new();
            let mut attentions = Vec::new();
            for block in 0..=config.blocks {
                blocks.push(ResnetBlock::create(
                    &mut schema.scope(format!("levels.{index}.blocks.{block}")),
                    [input, output],
                    None,
                    NORMS,
                    1.0,
                )?);
                input = output;
                if coarsest {
                    attentions.push(SpatialAttention::create(
                        &mut schema.scope(format!("levels.{index}.attentions.{block}")),
                        input,
                        1,
                        NORMS,
                        1.0,
                    )?);
                }
            }

            let upsample = (index + 1 < count)
                .then(|| UpsampleBlock::create(schema, &format!("levels.{index}.upsample"), input))
                .transpose()?;
            levels.push(Level {
                blocks,
                attentions,
                upsample,
            });
        }
        Ok(Self {
            codebook: schema.parameter("codebook", [config.codebook, config.entry])?,
            post_quant: schema.conv2d(
                "post_quant",
                config.entry,
                config.latent,
                1,
                Conv2dConfig::default(),
                true,
            )?,
            conv_in,
            middle,
            levels,
            norm_out: schema.group_norm("norm_out", input, NORMS.groups, NORMS.eps)?,
            conv_out: schema.conv2d("conv_out", input, config.channels, 3, same_2d(3, 1), true)?,
            side: config.side,
        })
    }

    /// Decodes image tokens, `[B, side²]`, to pixels in `[-1, 1]`, `[B, C, H, W]`.
    pub fn decode(&self, tokens: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, count] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image tokens, [B, N]"));
        };
        if count != self.side * self.side {
            return Err(SingeError::invalid_argument(format!(
                "{count} image tokens for a grid of {} × {}",
                self.side, self.side
            )));
        }

        let entries = state
            .tensor(&self.codebook)?
            .index_select(0, &tokens.reshape([batch * count])?)?;
        let norms = entries
            .vector_norm(NormOrder::Finite(2.0), &[1], true)?
            .clamp(LEAST_NORM, f32::MAX)?;
        let entries = entries.div(&norms)?;
        let width = entries.shape().dimensions()[1];
        let mut hidden = entries
            .reshape([batch, self.side, self.side, width])?
            .permute(&[0, 3, 1, 2])?
            .contiguous()?
            .apply(&self.post_quant, state)?
            .apply(&self.conv_in, state)?
            .apply(&self.middle, state)?;
        for level in &self.levels {
            for (index, block) in level.blocks.iter().enumerate() {
                hidden = block.forward(&hidden, None, state)?;
                if let Some(attention) = level.attentions.get(index) {
                    hidden = hidden.apply(attention, state)?;
                }
            }
            if let Some(upsample) = &level.upsample {
                hidden = hidden.apply(upsample, state)?;
            }
        }
        hidden
            .apply(&self.norm_out, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.conv_out, state)
    }
}
