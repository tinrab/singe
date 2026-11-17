mod backward_config;
mod config;
mod dropout;
mod mask;
mod policy;
mod primitive;
mod remap;
mod score;
mod sdpa;

pub(crate) use crate::frontend::operation::attention::sdpa::{
    SdpaAuxOutputRequest, SdpaQuantizedAuxOutputRequest, SdpaSoftmaxAuxOutputRequest,
};
pub use crate::frontend::operation::attention::{
    backward_config::{AttentionBackwardAuxGradients, AttentionBackwardConfig},
    config::{
        AttentionConfig, AttentionImplementation, AttentionPagedCache, AttentionPagedKvCache,
    },
    dropout::{
        AttentionDropoutConfig, AttentionDropoutMask, AttentionDropoutMode, DirectSdpaDropout,
    },
    mask::{AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode},
    primitive::AttentionPrimitiveOperation,
    score::{AlibiBias, AttentionScoreConfig, SdpaScoreModifiers, SdpaScoreSubgraph},
    sdpa::{
        AttentionMaxTotalSequenceLengths, DirectSdpaBackwardConfig, DirectSdpaForwardConfig,
        SdpaConfig, SdpaOperation,
    },
};
