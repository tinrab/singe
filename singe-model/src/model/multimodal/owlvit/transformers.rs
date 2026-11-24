//! Imports a `transformers` OWL-ViT checkpoint, CLIP's with its towers under `owlvit`, whose config is CLIP's.

use crate::{format::Rules, model::multimodal::clip::transformers::clip_rules};

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("layer_norm", "norm")
        .rename_each(
            "box_head",
            "boxes",
            &[("dense0", "0"), ("dense1", "1"), ("dense2", "2")],
        )
        .rename_each(
            "class_head",
            "classes",
            &[
                ("dense0", "embedding"),
                ("logit_shift", "shift"),
                ("logit_scale", "scale"),
            ],
        )
        // The detector reads the patches, not the pooled image embedding, and its own logit scales.
        .skip("owlvit.visual_projection")
        .skip("owlvit.logit_scale")
        .then(clip_rules("owlvit"))
}
