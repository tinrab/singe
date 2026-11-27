mod cross;
mod delta_net;
mod grouped;
mod latent;
mod projection;
mod relative;
mod self_attention;
mod state_space;

pub use self::{
    cross::{CrossAttention, CrossAttentionInput, SourcePages, pages_for},
    delta_net::{GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec},
    grouped::{GroupedQueryAttention, GroupedQueryLayers},
    latent::{LatentQuery, MultiHeadLatentAttention, MultiHeadLatentLayers},
    projection::{HeadNorm, HeadProjection},
    relative::{
        LocalAttention, RelativeAttention, RelativeAttentionInput, RelativeAttentionLayers,
    },
    self_attention::SelfAttention,
    state_space::{
        GatedNorm, Mamba2Layers, Mamba2Mixer, Mamba2Spec, MambaLayers, MambaMixer, MambaSpec,
        RgLru, RgLruLayers, RgLruSpec,
    },
};
use crate::{
    nn::{DecodeStep, RotaryAngles},
    tensor::Tensor,
};

// What attention without a cache reads: `hidden`, `[B, T, W]`, its rotation if any, and `key_lengths` if a batch's sequences differ in length, `[B]` of `I32`, each sequence's count of real tokens, which come first, so no query attends to a padded key.
// A `mask`, added to the scores and broadcast to `[B, H, T, T]`, takes the place of `key_lengths` and must mask padded keys itself.
#[derive(Debug, Clone, Copy)]
pub struct DenseAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub angles: Option<&'a RotaryAngles>,
    pub key_lengths: Option<&'a Tensor>,
    pub mask: Option<&'a Tensor>,
}

// What a decoding attention layer reads: `hidden`, `[T, W]`, the step's paging, its rotation if any, and a cache it extends.
#[derive(Debug)]
pub struct AttentionInput<'a, C> {
    pub hidden: &'a Tensor,
    pub step: &'a DecodeStep,
    pub angles: Option<&'a RotaryAngles>,
    pub cache: &'a C,
}

// By hand, since deriving these would demand them of the cache, which is only borrowed.
impl<C> Clone for AttentionInput<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for AttentionInput<'_, C> {}
