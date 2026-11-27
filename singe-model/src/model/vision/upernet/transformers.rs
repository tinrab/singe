//! Reads a `transformers` `UperNet` config, whose backbone is a `ConvNeXt` giving every stage, and imports its checkpoint without the auxiliary head, which only trains.

use serde::Deserialize;

use crate::{
    assembly::convnext_rules,
    error::{Error, Result},
    format::{RuleSet, transformers::classifier_config},
    model::vision::{convnext::transformers::ConvNextFields, upernet::UperNetConfig},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BackboneFields {
    #[serde(flatten)]
    convnext: ConvNextFields,
    out_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: BackboneFields,
    pool_scales: Vec<i64>,
    hidden_size: i64,
}

pub(crate) fn config(source: &Source) -> Result<UperNetConfig> {
    let fields: Fields = source.json("config.json")?;
    let backbone = &fields.backbone_config;
    let all_stages = (1..=backbone.convnext.depths.len())
        .map(|stage| format!("stage{stage}"))
        .collect::<Vec<_>>();
    if backbone.out_features != all_stages {
        return Err(Error::unsupported(
            "an UperNet whose ConvNeXt does not give every stage",
        ));
    }

    Ok(UperNetConfig {
        backbone: backbone.convnext.backbone()?,
        pool_scales: fields.pool_scales,
        width: fields.hidden_size,
        classifier: classifier_config(source)?,
    })
}

/// Renames an `UperNetConvModule` under `from`, `conv` and `batch_norm`, to a convolution and norm under `to`.
fn conv_module(rules: RuleSet, from: &str, to: &str) -> RuleSet {
    rules
        .skip(&format!("{from}.batch_norm.num_batches_tracked"))
        .rename_each(from, to, &[("conv", "convolution"), ("batch_norm", "norm")])
}

/// Returns the rules of an `UperNet` whose backbone has `stages` stages.
pub(crate) fn rules(stages: usize) -> RuleSet {
    let mut rules = RuleSet::new()
        // The auxiliary head only feeds the training loss.
        .skip("auxiliary_head")
        .then(convnext_rules("backbone", "backbone"));
    for stage in 0..stages {
        rules = rules.rename(
            &format!("backbone.hidden_states_norms.stage{}", stage + 1),
            &format!("stage_norms.{stage}"),
        );
    }

    for (from, to) in [
        ("psp_modules.*.1", "pyramid.*"),
        ("bottleneck", "bottleneck"),
        ("lateral_convs.*", "laterals.*"),
        ("fpn_convs.*", "outputs.*"),
        ("fpn_bottleneck", "fusion"),
    ] {
        rules = conv_module(rules, &format!("decode_head.{from}"), &format!("head.{to}"));
    }
    rules.rename("decode_head.classifier", "head.classifier")
}
