//! Reads a `transformers` VITS config and imports its checkpoint, folding its `WaveNets`' weight norms and leaving the parts only training reads.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{HifiGanConfig, attention_rules, hifigan_rules},
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::audio::vits::VitsConfig,
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
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
}

pub(crate) fn config(source: &Source) -> Result<VitsConfig> {
    let fields: Fields = source.json("config.json")?;
    if !fields.use_stochastic_duration_prediction {
        return Err(Error::unsupported(
            "a VITS without `use_stochastic_duration_prediction`",
        ));
    }

    if fields.speaker_embedding_size != 0 {
        return Err(Error::unsupported(
            "a VITS `speaker_embedding_size` other than zero",
        ));
    }

    if fields.hidden_act != Activation::Relu {
        return Err(Error::unsupported("a VITS `hidden_act` other than `relu`"));
    }

    if fields.wavenet_dilation_rate != 1 {
        return Err(Error::unsupported(
            "a VITS `wavenet_dilation_rate` other than one",
        ));
    }

    if fields.depth_separable_channels != 2 {
        return Err(Error::unsupported(
            "a VITS `depth_separable_channels` other than two",
        ));
    }

    if fields.upsample_kernel_sizes.len() != fields.upsample_rates.len() {
        return Err(Error::unsupported(
            "VITS `upsample_rates` and `upsample_kernel_sizes` of different lengths",
        ));
    }

    if fields.resblock_kernel_sizes.is_empty()
        || fields.resblock_dilation_sizes.len() != fields.resblock_kernel_sizes.len()
    {
        return Err(Error::unsupported(
            "VITS `resblock_kernel_sizes` and `resblock_dilation_sizes` of other than one nonzero length",
        ));
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
        vocoder: HifiGanConfig {
            input: fields.flow_size,
            initial: fields.upsample_initial_channel,
            upsample_rates: fields.upsample_rates,
            upsample_kernels: fields.upsample_kernel_sizes,
            resblock_kernels: fields.resblock_kernel_sizes,
            resblock_dilations: fields.resblock_dilation_sizes,
            leaky_slope: fields.leaky_relu_slope,
            post_bias: false,
        },
        sample_rate: fields.sampling_rate,
        noise_scale: fields.noise_scale,
        noise_scale_duration: fields.noise_scale_duration,
        speaking_rate: fields.speaking_rate,
    })
}

fn conv_flow_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
            ("conv_pre", "pre"),
            ("conv_proj", "projection"),
            ("conv_dds.convs_dilated", "dds.dilated"),
            ("conv_dds.convs_pointwise", "dds.pointwise"),
            ("conv_dds.norms_1", "dds.first_norms"),
            ("conv_dds.norms_2", "dds.second_norms"),
        ],
    )
}

pub(crate) fn rules(config: &VitsConfig) -> Rules {
    let layers = "text_encoder.encoder.layers.*";
    let mut rules = Rules::new()
        // The posterior encoder and the duration predictor's posterior flows serve only training.
        .skip("posterior_encoder")
        .skip("duration_predictor.post_conv_pre")
        .skip("duration_predictor.post_conv_dds")
        .skip("duration_predictor.post_conv_proj")
        .skip("duration_predictor.post_flows")
        .rename("text_encoder.embed_tokens", "tokens")
        .then(attention_rules(
            &format!("{layers}.attention"),
            "blocks.*.attention",
        ))
        .rename_each(
            layers,
            "blocks.*",
            &[
                ("attention.emb_rel_k", "attention.relative_keys"),
                ("attention.emb_rel_v", "attention.relative_values"),
                ("layer_norm", "attention_norm"),
                ("feed_forward.conv_1", "feed_forward.up"),
                ("feed_forward.conv_2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
        .rename("text_encoder.project", "prior")
        .then(conv_flow_rules("duration_predictor", "duration"))
        .rename_each(
            "duration_predictor.flows.0",
            "duration.affine",
            &[("translate", "shift"), ("log_scale", "log_scale")],
        );
    // The convolutional flows after the affine one, numbered from one.
    for flow in 1..=config.duration_flows {
        rules = rules.then(conv_flow_rules(
            &format!("duration_predictor.flows.{flow}"),
            &format!("duration.flows.{}", flow - 1),
        ));
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
        .then(hifigan_rules("decoder", "decoder"))
}
