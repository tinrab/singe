// Reads a `transformers` VideoMAE config and imports its checkpoint, whose attention stores the query and value biases apart from their projections and has no key bias.

use serde::Deserialize;

use crate::{
    assembly::vit_layer_rules,
    error::{Error, Result},
    format::{Rules, transformers::classifier_config},
    model::{video::videomae::VideoMaeConfig, vision::vit::transformers::VitFields},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    vit: VitFields,
    image_size: i64,
    num_frames: i64,
    tubelet_size: i64,
    #[serde(default)]
    use_mean_pooling: bool,
}

pub(crate) fn config(source: &Source) -> Result<VideoMaeConfig> {
    let fields: Fields = source.json("config.json")?;
    // Without mean pooling the classifier reads a final norm's class token, which these blocks lack.
    if !fields.use_mean_pooling {
        return Err(Error::unsupported("VideoMAE without mean pooling"));
    }

    let mut encoder = fields.vit.encoder()?;
    encoder.projection_biases = [true, false, true, true];
    Ok(VideoMaeConfig {
        encoder,
        layers: fields.vit.num_hidden_layers,
        image_size: fields.image_size,
        patch_size: fields.vit.patch_size,
        frames: fields.num_frames,
        tubelet: fields.tubelet_size,
        channels: fields.vit.num_channels,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename(
            "videomae.embeddings.patch_embeddings.projection",
            "embeddings.tubelets",
        )
        .rename_each(
            "videomae.encoder.layer.*.attention.attention",
            "blocks.*.attention",
            &[("q_bias", "query.bias"), ("v_bias", "value.bias")],
        )
        .then(vit_layer_rules("videomae.encoder.layer", "blocks"))
        .rename("fc_norm", "norm")
        .rename("classifier", "classifier.output")
}
