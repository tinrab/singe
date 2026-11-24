//! Reads a `transformers` `Swin2SR` config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::vision::swin2sr::Swin2SrConfig,
    source::Source,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum Upsampler {
    #[serde(rename = "pixelshuffle")]
    PixelShuffle,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum ResidualConnection {
    #[serde(rename = "1conv")]
    OneConvolution,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    embed_dim: i64,
    depths: Vec<i64>,
    num_heads: Vec<i64>,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    window_size: i64,
    num_channels: i64,
    upscale: i64,
    upsampler: Upsampler,
    resi_connection: ResidualConnection,
    img_range: f64,
    patch_size: i64,
    #[serde(default)]
    use_absolute_embeddings: bool,
    image_size: i64,
}

/// Refuses a config other than the classical models': three channels, the pixel-shuffle upsampler, one convolution closing each stage, patches of one point, no absolute embeddings, an `img_range` of one, as many `num_heads` as `depths`, an `upscale` that is a power of two, and an `image_size` larger than one window.
fn check_supported(fields: &Fields) -> Result<()> {
    if fields.num_channels != 3 {
        return Err(Error::unsupported(format!(
            "a Swin2SR of {} channels, not three",
            fields.num_channels
        )));
    }

    if fields.upsampler != Upsampler::PixelShuffle {
        return Err(Error::unsupported(
            "a Swin2SR upsampler other than `pixelshuffle`",
        ));
    }

    if fields.resi_connection != ResidualConnection::OneConvolution {
        return Err(Error::unsupported(
            "a Swin2SR `resi_connection` other than `1conv`",
        ));
    }

    if fields.patch_size != 1 {
        return Err(Error::unsupported(format!(
            "a Swin2SR `patch_size` of {}, not one",
            fields.patch_size
        )));
    }

    if fields.use_absolute_embeddings {
        return Err(Error::unsupported("Swin2SR's absolute embeddings"));
    }

    if (fields.img_range - 1.0).abs() > f64::EPSILON {
        return Err(Error::unsupported(format!(
            "a Swin2SR `img_range` of {}, not one",
            fields.img_range
        )));
    }

    if fields.depths.len() != fields.num_heads.len() {
        return Err(Error::unsupported(format!(
            "Swin2SR `depths` of {} stages beside `num_heads` of {}",
            fields.depths.len(),
            fields.num_heads.len()
        )));
    }

    if fields.upscale <= 0 || fields.upscale & (fields.upscale - 1) != 0 {
        return Err(Error::unsupported(format!(
            "a Swin2SR `upscale` of {}, not a power of two",
            fields.upscale
        )));
    }

    if fields.image_size <= fields.window_size {
        return Err(Error::unsupported(format!(
            "a Swin2SR `image_size` of {} within one window of {}",
            fields.image_size, fields.window_size
        )));
    }
    Ok(())
}

pub(crate) fn config(source: &Source) -> Result<Swin2SrConfig> {
    let fields: Fields = source.json("config.json")?;
    check_supported(&fields)?;

    let inner = (fields.mlp_ratio * fields.embed_dim as f64) as i64;
    Ok(Swin2SrConfig {
        width: fields.embed_dim,
        stages: fields.depths.into_iter().zip(fields.num_heads).collect(),
        inner,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        window: fields.window_size,
        channels: fields.num_channels,
        upscale: fields.upscale,
        upsampler: 64,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        // One convolution before each doubling, numbered in its name.
        .rename_each(
            "upsample.upsample",
            "upsample",
            &[
                ("convolution_0", "0"),
                ("convolution_1", "1"),
                ("convolution_2", "2"),
                ("convolution_3", "3"),
            ],
        )
        .rename("swin2sr.first_convolution", "first")
        .rename_each(
            "swin2sr.embeddings.patch_embeddings",
            "",
            &[
                ("projection", "projection"),
                ("layernorm", "projection_norm"),
            ],
        )
        .rename_each(
            "swin2sr.encoder.stages.*.layers.*",
            "stages.*.blocks.*",
            &[
                ("attention.self.query", "attention.query"),
                ("attention.self.key", "attention.key"),
                ("attention.self.value", "attention.value"),
                ("attention.self.logit_scale", "attention.logit_scale"),
                (
                    "attention.self.continuous_position_bias_mlp.0",
                    "attention.bias_hidden",
                ),
                (
                    "attention.self.continuous_position_bias_mlp.2",
                    "attention.bias_out",
                ),
                ("attention.output.dense", "attention.output"),
                ("layernorm_before", "attention_norm"),
                ("intermediate.dense", "feed_forward.up"),
                ("output.dense", "feed_forward.down"),
                ("layernorm_after", "feed_forward_norm"),
            ],
        )
        .rename_each(
            "swin2sr.encoder.stages.*",
            "stages.*",
            &[
                ("conv", "convolution"),
                ("patch_embed.projection", "projection"),
            ],
        )
        .rename("swin2sr.layernorm", "norm")
        .rename("swin2sr.conv_after_body", "after_body")
        .rename_each(
            "upsample",
            "",
            &[
                ("conv_before_upsample", "before_upsample"),
                ("final_convolution", "last"),
            ],
        )
}
