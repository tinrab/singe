//! Reads a `transformers` DAC config and imports its checkpoint, whose weight norms are folded already.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::Rules,
    model::audio::dac::{DILATIONS, DacConfig},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    encoder_hidden_size: i64,
    decoder_hidden_size: i64,
    hidden_size: i64,
    downsampling_ratios: Vec<i64>,
    upsampling_ratios: Vec<i64>,
    n_codebooks: i64,
    codebook_size: i64,
    codebook_dim: i64,
    sampling_rate: i64,
}

pub(crate) fn config(source: &Source) -> Result<DacConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.n_codebooks < 1 {
        return Err(Error::unsupported("a DAC `n_codebooks` below one"));
    }

    Ok(DacConfig {
        encoder_width: fields.encoder_hidden_size,
        decoder_width: fields.decoder_hidden_size,
        latent: fields.hidden_size,
        downsampling: fields.downsampling_ratios,
        upsampling: fields.upsampling_ratios,
        codebooks: fields.n_codebooks,
        codebook_size: fields.codebook_size,
        codebook_dim: fields.codebook_dim,
        sample_rate: fields.sampling_rate,
    })
}

pub(crate) fn rules() -> Rules {
    let mut rules = Rules::new()
        .rename("encoder.conv1", "encoder_in")
        .rename("encoder.snake1.alpha", "encoder_snake")
        .rename("encoder.conv2", "encoder_out")
        .rename("encoder.block.*.snake1.alpha", "encoder.*.snake")
        .rename("encoder.block.*.conv1", "encoder.*.resample")
        .rename_each(
            "quantizer.quantizers.*",
            "codebooks.*",
            &[
                ("in_proj", "project_in"),
                ("codebook.weight", "entries"),
                ("out_proj", "project_out"),
            ],
        )
        .rename("decoder.conv1", "decoder_in")
        .rename("decoder.snake1.alpha", "decoder_snake")
        .rename("decoder.conv2", "decoder_out")
        .rename("decoder.block.*.snake1.alpha", "decoder.*.snake")
        .rename("decoder.block.*.conv_t1", "decoder.*.resample");
    // Each block's residual units, numbered from one.
    for part in ["encoder", "decoder"] {
        for unit in 0..DILATIONS.len() {
            rules = rules.rename_each(
                &format!("{part}.block.*.res_unit{}", unit + 1),
                &format!("{part}.*.units.{unit}"),
                &[
                    ("snake1.alpha", "first_snake"),
                    ("conv1", "first"),
                    ("snake2.alpha", "second_snake"),
                    ("conv2", "second"),
                ],
            );
        }
    }
    rules
}
