mod cross;
mod delta_net;
mod grouped;
mod latent;
mod projection;
mod self_attention;

pub use self::{
    cross::{CrossAttention, CrossAttentionInput, SourcePages, pages_for},
    delta_net::{GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec},
    grouped::{GroupedQueryAttention, GroupedQueryLayers},
    latent::{LatentQuery, MultiHeadLatentAttention, MultiHeadLatentLayers},
    projection::{HeadNorm, HeadProjection},
    self_attention::SelfAttention,
};
use crate::{
    nn::{DecodeStep, RotaryAngles},
    tensor::Tensor,
};

// What attention without a cache reads: `hidden`, `[B, T, W]`, its rotation if any, and `key_lengths` if a batch's sequences differ in length, `[B]` of `I32`, each sequence's count of real tokens, which come first, so no query attends to a padded key.
#[derive(Debug, Clone, Copy)]
pub struct DenseAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub angles: Option<&'a RotaryAngles>,
    pub key_lengths: Option<&'a Tensor>,
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
