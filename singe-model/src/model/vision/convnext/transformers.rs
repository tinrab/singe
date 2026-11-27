//! Reads a `transformers` `ConvNeXt` or `ConvNeXt` V2 config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{ConvNextConfig, ConvNextScaling, convnext_rules},
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{activation_field, classifier_config},
    },
    model::vision::convnext::ConvNextClassifierConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct ConvNextFields {
    pub model_type: String,
    pub depths: Vec<i64>,
    pub hidden_sizes: Vec<i64>,
    pub patch_size: i64,
    #[serde(deserialize_with = "activation_field")]
    pub hidden_act: Activation,
    pub layer_norm_eps: f32,
    #[serde(default = "default_channels")]
    pub num_channels: i64,
    pub layer_scale_init_value: Option<f64>,
}

const fn default_channels() -> i64 {
    3
}

impl ConvNextFields {
    /// Returns the backbone these fields describe: `ConvNeXt` V2 with its response norm, `ConvNeXt` with its layer scale.
    pub(crate) fn backbone(&self) -> Result<ConvNextConfig> {
        let scaling = match self.model_type.as_str() {
            "convnextv2" => ConvNextScaling::ResponseNorm,
            "convnext" if self.layer_scale_init_value.is_some_and(|value| value > 0.0) => {
                ConvNextScaling::LayerScale
            }
            other => {
                return Err(Error::unsupported(format!(
                    "ConvNeXt `{other}`, other than ConvNeXt with a layer scale or ConvNeXt V2"
                )));
            }
        };
        if self.num_channels != 3
            || self.depths.is_empty()
            || self.depths.len() != self.hidden_sizes.len()
        {
            return Err(Error::unsupported(
                "a ConvNeXt of other than three input channels, without stages, or with other than one width per stage",
            ));
        }

        Ok(ConvNextConfig {
            depths: self.depths.clone(),
            widths: self.hidden_sizes.clone(),
            patch: self.patch_size,
            activation: self.hidden_act,
            scaling,
        })
    }
}

pub(crate) fn config(source: &Source) -> Result<ConvNextClassifierConfig> {
    let fields: ConvNextFields = source.json("config.json")?;
    Ok(ConvNextClassifierConfig {
        backbone: fields.backbone()?,
        eps: fields.layer_norm_eps,
        classifier: classifier_config(source)?,
    })
}

/// Returns the rules of a classifier whose backbone `transformers` stores as `convnext`, or as `convnextv2` with response norms.
pub(crate) fn rules(scaling: ConvNextScaling) -> RuleSet {
    let prefix = match scaling {
        ConvNextScaling::LayerScale => "convnext",
        ConvNextScaling::ResponseNorm => "convnextv2",
    };
    convnext_rules(prefix, "backbone")
        .rename(&format!("{prefix}.layernorm"), "norm")
        .rename("classifier", "classifier")
}
