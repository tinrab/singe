//! Reads a `diffusers` `UNet2DModel`: its config and its tensors' names.

use serde::Deserialize;

use crate::{
    assembly::unet_block_rules,
    error::{Error, Result},
    format::{
        RuleSet,
        diffusers::{AttentionHeadDim, UnetFields, default_true},
    },
    model::diffusion::unet_2d::Unet2dConfig,
    source::Source,
};

const CLASS: &str = "UNet2DModel";

fn conv() -> String {
    "conv".to_owned()
}

#[derive(Debug, Deserialize)]
struct Fields {
    #[serde(flatten)]
    unet: UnetFields,
    #[serde(default = "conv")]
    downsample_type: String,
    #[serde(default = "conv")]
    upsample_type: String,
    #[serde(default)]
    attn_norm_num_groups: Option<i64>,
    #[serde(default = "default_true")]
    add_attention: bool,
}

impl Fields {
    /// Refuses the options `diffusers` offers that the model does not build.
    fn validate(&self) -> Result<()> {
        self.unet.validate(CLASS, "UNetMidBlock2D")?;
        if self.downsample_type != "conv" || self.upsample_type != "conv" {
            return Err(Error::unsupported(format!(
                "{CLASS} downsample_type or upsample_type other than conv"
            )));
        }

        if !self.add_attention || self.attn_norm_num_groups.is_some() {
            return Err(Error::unsupported(format!(
                "{CLASS} add_attention off or attn_norm_num_groups set"
            )));
        }
        Ok(())
    }
}

pub(crate) fn config(source: &Source) -> Result<Unet2dConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    let head_width = match &fields.unet.attention_head_dim {
        None => None,
        Some(AttentionHeadDim::One(width)) => Some(*width),
        Some(AttentionHeadDim::Each(_)) => {
            return Err(Error::unsupported(format!(
                "{CLASS} attention_head_dim per block"
            )));
        }
    };
    Ok(Unet2dConfig {
        unet: fields.unet.config(["AttnDownBlock2D", "AttnUpBlock2D"])?,
        head_width,
    })
}

pub(crate) fn rules() -> RuleSet {
    RuleSet::new()
        .rename("conv_in", "input")
        .rename("time_embedding.linear_1", "time.up")
        .rename("time_embedding.linear_2", "time.down")
        .then(unet_block_rules("down_blocks.*", "down.*"))
        .then(unet_block_rules("mid_block", "middle"))
        .then(unet_block_rules("up_blocks.*", "up.*"))
        .rename("conv_norm_out", "norm")
        .rename("conv_out", "output")
}
