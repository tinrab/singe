// Reads a `transformers` VITS config and imports its checkpoint, folding its WaveNets' weight norms and leaving the parts only training reads.

use serde::Deserialize;

use super::VitsConfig;
use crate::{
    error::{Error, Result},
    format::Rules,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    ffn_dim: i64,
    ffn_kernel_size: i64,
    window_size: i64,
    layer_norm_eps: f32,
    flow_size: i64,
    duration_predictor_num_flows: i64,
    duration_predictor_flow_bins: i64,
    duration_predictor_tail_bound: f32,
    depth_separable_num_layers: i64,
    duration_predictor_kernel_size: i64,
    depth_separable_channels: i64,
    prior_encoder_num_flows: i64,
    prior_encoder_num_wavenet_layers: i64,
    wavenet_kernel_size: i64,
    wavenet_dilation_rate: i64,
    upsample_initial_channel: i64,
    upsample_rates: Vec<i64>,
    upsample_kernel_sizes: Vec<i64>,
    resblock_kernel_sizes: Vec<i64>,
    resblock_dilation_sizes: Vec<Vec<i64>>,
    leaky_relu_slope: f64,
    sampling_rate: i64,
    noise_scale: f32,
    noise_scale_duration: f32,
    speaking_rate: f32,
    use_stochastic_duration_prediction: bool,
    speaker_embedding_size: i64,
    hidden_act: String,
}

pub(crate) fn config(source: &Source) -> Result<VitsConfig> {
    let fields: Fields = source.json("config.json")?;
    // One speaker, the stochastic duration predictor, ReLU feed-forwards, and undilated WaveNets of two channels' flows.
    if !fields.use_stochastic_duration_prediction
        || fields.speaker_embedding_size != 0
        || fields.hidden_act != "relu"
        || fields.wavenet_dilation_rate != 1
        || fields.depth_separable_channels != 2
    {
        return Err(Error::unsupported("this VITS config"));
    }

    Ok(VitsConfig {
        vocab_size: fields.vocab_size,
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        inner: fields.ffn_dim,
        ffn_kernel: fields.ffn_kernel_size,
        window: fields.window_size,
        eps: fields.layer_norm_eps,
        flow_size: fields.flow_size,
        duration_flows: fields.duration_predictor_num_flows,
        duration_bins: fields.duration_predictor_flow_bins,
        tail_bound: fields.duration_predictor_tail_bound,
        dds_layers: fields.depth_separable_num_layers,
        dds_kernel: fields.duration_predictor_kernel_size,
        prior_flows: fields.prior_encoder_num_flows,
        wavenet_layers: fields.prior_encoder_num_wavenet_layers,
        wavenet_kernel: fields.wavenet_kernel_size,
        upsample_initial: fields.upsample_initial_channel,
        upsample_rates: fields.upsample_rates,
        upsample_kernels: fields.upsample_kernel_sizes,
        resblock_kernels: fields.resblock_kernel_sizes,
        resblock_dilations: fields.resblock_dilation_sizes,
        leaky_slope: fields.leaky_relu_slope,
        sample_rate: fields.sampling_rate,
        noise_scale: fields.noise_scale,
        noise_scale_duration: fields.noise_scale_duration,
        speaking_rate: fields.speaking_rate,
    })
}

fn depth_separable(rules: Rules, from: &str, to: &str) -> Rules {
    rules
        .rename(&format!("{from}.convs_dilated"), &format!("{to}.dilated"))
        .rename(
            &format!("{from}.convs_pointwise"),
            &format!("{to}.pointwise"),
        )
        .rename(&format!("{from}.norms_1"), &format!("{to}.first_norms"))
        .rename(&format!("{from}.norms_2"), &format!("{to}.second_norms"))
}

pub(crate) fn rules() -> Rules {
    let layer = |name: &str| format!("text_encoder.encoder.layers.*.{name}");
    let mut rules = Rules::new()
        // The posterior encoder and the duration predictor's posterior flows serve only training.
        .skip("posterior_encoder")
        .skip("duration_predictor.post_conv_pre")
        .skip("duration_predictor.post_conv_dds")
        .skip("duration_predictor.post_conv_proj")
        .skip("duration_predictor.post_flows")
        .rename("text_encoder.embed_tokens", "tokens")
        .rename(&layer("attention.q_proj"), "layers.*.query")
        .rename(&layer("attention.k_proj"), "layers.*.key")
        .rename(&layer("attention.v_proj"), "layers.*.value")
        .rename(&layer("attention.out_proj"), "layers.*.output")
        .rename(&layer("attention.emb_rel_k"), "layers.*.relative_keys")
        .rename(&layer("attention.emb_rel_v"), "layers.*.relative_values")
        .rename(&layer("layer_norm"), "layers.*.attention_norm")
        .rename(&layer("feed_forward.conv_1"), "layers.*.up")
        .rename(&layer("feed_forward.conv_2"), "layers.*.down")
        .rename(&layer("final_layer_norm"), "layers.*.feed_forward_norm")
        .rename("text_encoder.project", "prior")
        .rename("duration_predictor.conv_pre", "duration.pre")
        .rename("duration_predictor.conv_proj", "duration.projection")
        .rename(
            "duration_predictor.flows.0.translate",
            "duration.affine.shift",
        )
        .rename(
            "duration_predictor.flows.0.log_scale",
            "duration.affine.log_scale",
        );
    rules = depth_separable(rules, "duration_predictor.conv_dds", "duration.dds");
    // The convolutional flows after the affine one, numbered from one.
    for flow in 1..=8 {
        let from = format!("duration_predictor.flows.{flow}");
        let to = format!("duration.flows.{}", flow - 1);
        rules = rules
            .rename(&format!("{from}.conv_pre"), &format!("{to}.pre"))
            .rename(&format!("{from}.conv_proj"), &format!("{to}.projection"));
        rules = depth_separable(rules, &format!("{from}.conv_dds"), &format!("{to}.dds"));
    }
    rules
        .weight_norm(
            "flow.flows.*.wavenet.in_layers.*",
            "couplings.*.wavenet.inputs.*.weight",
            0,
        )
        .rename(
            "flow.flows.*.wavenet.in_layers.*.bias",
            "couplings.*.wavenet.inputs.*.bias",
        )
        .weight_norm(
            "flow.flows.*.wavenet.res_skip_layers.*",
            "couplings.*.wavenet.residual_skips.*.weight",
            0,
        )
        .rename(
            "flow.flows.*.wavenet.res_skip_layers.*.bias",
            "couplings.*.wavenet.residual_skips.*.bias",
        )
        .rename("flow.flows.*.conv_pre", "couplings.*.pre")
        .rename("flow.flows.*.conv_post", "couplings.*.post")
        .rename("decoder.conv_pre", "decoder.pre")
        .rename("decoder.conv_post", "decoder.post")
        .rename("decoder.upsampler", "decoder.upsamplers")
        .rename("decoder.resblocks.*.convs1", "decoder.resblocks.*.first")
        .rename("decoder.resblocks.*.convs2", "decoder.resblocks.*.second")
}
