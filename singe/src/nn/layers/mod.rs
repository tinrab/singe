// Reusable layers, sorted by what each computes, never by the model or task that uses it.

mod activation;
mod attention;
mod blocks;
mod convolution;
mod embedding;
mod feed_forward;
mod heads;
mod linear;
mod networks;
mod normalization;
mod positional;
mod recurrent;
mod resampling;

pub use self::{
    activation::Activation,
    attention::{
        AttentionInput, CrossAttention, CrossAttentionInput, DenseAttentionInput, GatedDeltaNet,
        GatedDeltaNetLayers, GatedDeltaNetSpec, GroupedQueryAttention, GroupedQueryLayers,
        HeadNorm, HeadProjection, LatentQuery, LocalAttention, MultiHeadLatentAttention,
        MultiHeadLatentLayers, RelativeAttention, RelativeAttentionInput, RelativeAttentionLayers,
        SelfAttention, SourcePages, pages_for,
    },
    blocks::{EncoderBlock, EncoderDecoderBlock, EncoderDecoderParts, HybridLayer, PreNormBlock},
    convolution::{
        Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose1d,
        ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig, ConvTransposeConfig,
        PatchEmbedding,
    },
    embedding::{Embedding, PatchTokens, TokenEmbeddings},
    feed_forward::{DecoderFeedForward, GatedMlp, MixtureOfExperts, Mlp, MoeLayers, MoeRouting},
    heads::{Pooling, PoolingInput, PoolingMode},
    linear::{Flatten, Linear},
    networks::{CausalLm, CausalLmParts, Decoder, MultiTokenPrediction, SourcedCausalLm},
    normalization::{BatchNorm, GroupNorm, LayerNorm, LayerScale, OffsetRmsNorm, RmsNorm},
    positional::{
        RotaryAngles, RotaryFrequencies, RotaryLayout, RotarySpec, RotaryTable, YarnFrequencies,
    },
    recurrent::{
        Recurrent, RecurrentCell, RecurrentConfig, RecurrentDirection, RecurrentLayer,
        RecurrentState, cell_step, step_masks,
    },
    resampling::{Pool2d, Pool2dConfig, Pool2dKind, Upsample},
};
