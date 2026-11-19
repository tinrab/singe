// What model definitions share: the parts of Llama-family decoders, BERT-style post-norm encoders, Whisper-style pre-norm encoder blocks, Swin's windowed encoder, attention over separate queries, keys, and values, stacks of linear layers, and `diffusers`' UNet and VAE blocks under Singe's names, and the rules that import other formats into them.

mod attention;
mod convolution;
mod decoder;
mod decoder_step;
mod encoder_decoder;
mod expert;
mod feature_map;
mod feed_forward;
mod normalization;
mod post_norm_encoder;
mod pre_norm_encoder;
mod sinusoid;
mod swin;
mod unet;
mod vision_tower;

pub use self::{
    attention::{Attention, AttentionSpec, attention_rules, merge_heads, split_heads},
    convolution::{same_1d, same_2d},
    decoder::{
        CausalLmSpec, DecoderConfig, DraftLayer, GroupedAttentionSpec, GroupedDecoder, HeadNorms,
        causal, causal_lm, gated_mlp, grouped_attention, grouped_attention_layers, grouped_decoder,
        hidden_states, kv_caches, llama_block_rules, llama_rules, pre_norm_block,
        pre_norm_block_with,
    },
    decoder_step::{Seq2SeqCache, Seq2SeqCacheSpec, Seq2SeqTables, StepRows},
    encoder_decoder::{
        BartDecoder, BartStack, BartStackConfig, StackRole, bart_stack_rules, embedding_scale,
    },
    expert::{ExpertRouting, ExpertsConfig, expert_rules, mixture_of_experts},
    feature_map::{merge_windows, partition_windows, spatial, to_map, to_tokens},
    feed_forward::LinearStack,
    normalization::unit_vectors,
    post_norm_encoder::{
        PostNormBlockConfig, PostNormEncoder, PostNormEncoderConfig, post_norm_encoder_block,
        post_norm_encoder_rules,
    },
    pre_norm_encoder::{
        FinalNorm, PreNormEncoder, PreNormEncoderBlock, PreNormEncoderConfig, clip_layer_rules,
        pre_norm_encoder_block, pre_norm_encoder_rules, vit_layer_rules,
    },
    sinusoid::{SinusoidArithmetic, SinusoidOrder, sinusoids},
    swin::{Swin, SwinConfig, relative_index, shift_mask, swin_rules},
    unet::{
        MiddleBlock, Norms, ResnetBlock, SpatialAttention, Unet, UnetConfig, UnetSite,
        UpsampleBlock, downsample, resnet_stack, timestep_input, unet_block_rules,
    },
    vision_tower::{VisionTowerConfig, patch_embedding},
};

pub const PYTORCH_NORM_EPS: f32 = 1e-5;
