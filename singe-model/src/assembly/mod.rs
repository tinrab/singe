// What model definitions share: the parts of Llama-family decoders, BERT-style post-norm encoders, Whisper-style pre-norm encoder blocks, Swin's windowed encoder, attention over separate queries, keys, and values, ReLU MLPs, and `diffusers`' UNet and VAE blocks under Singe's names, and the rules that import other formats into them.

mod attention;
mod decoder;
mod expert;
mod feed_forward;
mod post_norm_encoder;
mod pre_norm_encoder;
mod swin;
mod unet;

pub use self::{
    attention::{Attention, AttentionCache, AttentionSpec, causal_mask},
    decoder::{
        CausalLmSpec, DecoderConfig, DraftLayer, GroupedDecoder, causal, causal_lm, decoder,
        gated_mlp, grouped_attention, grouped_decoder, grouped_decoder_parts, hidden_states,
        kv_caches, llama_rules, pre_norm_block, pre_norm_block_with,
    },
    expert::expert_rules,
    feed_forward::Mlp,
    post_norm_encoder::{
        PostNormEncoder, PostNormEncoderConfig, encoder_block, post_norm_encoder_rules,
    },
    pre_norm_encoder::{
        PreNormEncoder, PreNormEncoderBlock, PreNormEncoderSpec, clip_layer_rules,
        pre_norm_encoder_block, pre_norm_encoder_rules, vit_layer_rules,
    },
    swin::{
        Swin, SwinConfig, merge_windows, partition_windows, roll, shift_mask, swin_rules, to_map,
        to_tokens,
    },
    unet::{
        Norms, ResnetBlock, SpatialAttention, TimeEmbedding, UnetLayout, UnetSite, UnetStages,
        UpsampleBlock, attends, conv3x3, downsample, timestep_input, unet_block_rules,
    },
};
