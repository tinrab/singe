//! What model definitions share: parts such as Llama-family decoders, BERT-style post-norm encoders, Whisper-style pre-norm encoder blocks, Conformers, transducers, and `diffusers`' `UNet` and VAE blocks under Singe's names, and the rules that import other formats into them.

mod attention;
mod conformer;
mod convnext;
mod convolution;
mod decoder;
mod deformable;
mod encoder_decoder;
mod expert;
mod feature_map;
mod feed_forward;
mod hifigan;
mod normalization;
mod post_norm_encoder;
mod pre_norm_encoder;
mod recurrent;
mod resnet;
mod seanet;
mod sinusoid;
mod state_space;
mod swin;
mod transducer;
mod unet;
mod vision_tower;

pub use self::{
    attention::{
        Attention, AttentionSpec, AxialRotary, attention_rules, attention_rules_with_output,
        merge_heads, paged_cross_attention, paged_self_attention, split_heads, swapped_pairs,
    },
    conformer::{
        ConformerBlock, ConformerConfig, ConformerEncoder, ConformerEncoderConfig,
        ConformerFeedForward, ConvSubsampling, ConvolutionModule, SubsamplingConfig,
        SubsamplingKind, conformer_rules,
    },
    convnext::{ConvNext, ConvNextBlock, ConvNextConfig, ConvNextScaling, convnext_rules},
    convolution::{ConvGroupNorm, same_1d, same_2d},
    decoder::{
        CausalLmSpec, DecoderConfig, DraftLayer, GroupedAttentionSpec, GroupedDecoder, HeadNorms,
        causal, causal_lm, gated_mlp, grouped_attention, grouped_attention_layers, grouped_decoder,
        kv_caches, llama_block_rules, llama_rules, pre_norm_block, pre_norm_block_with,
        rename_scaled,
    },
    deformable::{
        DeformableAttention, DeformableAttentionSpec, DeformableDecoderLayer,
        DeformableEncoderLayer, FeatureLevels, anchor_proposals, deformable_attention_rules,
        deformable_decoder_rules, deformable_encoder_rules, gather_rows, inverse_sigmoid, logit,
    },
    encoder_decoder::{
        BartDecoder, BartEncoder, BartStackConfig, bart_stack_rules, embedding_scale,
    },
    expert::{ExpertRouting, ExpertsConfig, expert_rules, mixture_of_experts},
    feature_map::{
        merge_windows, partition_windows, spatial, spatial_dimensions, to_map, to_tokens,
    },
    feed_forward::LinearStack,
    hifigan::{HifiGan, HifiGanConfig, hifigan_rules},
    normalization::{channel_norm, unit_vectors},
    post_norm_encoder::{
        KeyPattern, PostNormBlockConfig, PostNormEncoder, PostNormEncoderConfig,
        post_norm_encoder_block, post_norm_encoder_rules,
    },
    pre_norm_encoder::{
        FinalNorm, PreNormEncoder, PreNormEncoderBlock, PreNormEncoderConfig, clip_layer_rules,
        pre_norm_encoder_block, pre_norm_encoder_rules, vit_layer_rules,
    },
    recurrent::recurrent_rules,
    resnet::{ConvNorm, ResNet, ResNetConfig, ResNetVd},
    seanet::{PadMode, PaddedConv, Seanet, SeanetConfig, TrimmedTranspose, seanet_rules},
    sinusoid::{
        SinePositions, SinusoidArithmetic, SinusoidOrder, relative_sinusoids, sine_positions,
        sinusoids,
    },
    state_space::{Mamba2Layout, mamba2_mixer},
    swin::{Swin, SwinConfig, relative_index, shift_mask, swin_rules},
    transducer::{
        JointNetwork, PredictionNetwork, TransducerConfig, TransducerDecoding, TransducerLayers,
        TransducerStep, greedy_decode, transducer_rules,
    },
    unet::{
        MiddleBlock, Norms, ResnetBlock, SpatialAttention, Unet, UnetConfig, UnetSite,
        UpsampleBlock, downsample, resnet_stack, timestep_input, unet_block_rules,
    },
    vision_tower::{VisionTowerConfig, patch_embedding},
};

pub const PYTORCH_NORM_EPS: f32 = 1e-5;
