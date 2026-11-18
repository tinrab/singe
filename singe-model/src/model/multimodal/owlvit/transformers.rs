// Reads a `transformers` OWL-ViT config, CLIP's with its towers under `owlvit.`, and imports its checkpoint.

use super::OwlVitConfig;
use crate::{
    error::Result,
    format::Rules,
    model::multimodal::clip::transformers::{self as clip, clip_rules},
    source::Source,
};

pub(crate) fn config(source: &Source) -> Result<OwlVitConfig> {
    Ok(OwlVitConfig {
        clip: clip::config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("layer_norm", "norm")
        .rename("box_head.dense0", "boxes.0")
        .rename("box_head.dense1", "boxes.1")
        .rename("box_head.dense2", "boxes.2")
        .rename("class_head.dense0", "classes.embedding")
        .rename("class_head.logit_shift", "classes.shift")
        .rename("class_head.logit_scale", "classes.scale")
        // The detector reads the patches, not the pooled image embedding, and its own logit scales.
        .skip("owlvit.visual_projection")
        .skip("owlvit.logit_scale")
        .then(clip_rules("owlvit."))
}
