//! Reads a `transformers` `MusicGen` config, which nests its text encoder's, decoder's, and codec's, and imports its checkpoint.

use serde::Deserialize;

use crate::{
    assembly::attention_rules,
    error::{Error, Result},
    format::{Rules, transformers::activation},
    head::{MusicGeneration, SamplingFields},
    model::{
        audio::{
            encodec::transformers as encodec,
            musicgen::{MusicgenConfig, MusicgenDecoderConfig},
        },
        text::t5::transformers as t5,
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct DecoderFields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    ffn_dim: i64,
    activation_function: String,
    num_codebooks: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    #[serde(default = "mono")]
    audio_channels: i64,
}

const fn mono() -> i64 {
    1
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    text_encoder: t5::Fields,
    audio_encoder: encodec::Fields,
    decoder: DecoderFields,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct GenerationFields {
    decoder_start_token_id: i64,
    #[serde(default)]
    guidance_scale: Option<f32>,
    max_length: usize,
    #[serde(flatten)]
    sampling: SamplingFields,
}

pub(crate) fn config(source: &Source) -> Result<MusicgenConfig> {
    let fields: Fields = source.json("config.json")?;
    let generation: GenerationFields = source.json("generation_config.json")?;
    let decoder = fields.decoder;
    if decoder.audio_channels != 1 {
        return Err(Error::unsupported("stereo MusicGen"));
    }

    let codec = encodec::fields_config(fields.audio_encoder)?;
    if codec.codebooks != decoder.num_codebooks {
        return Err(Error::unsupported(
            "a MusicGen whose codec's bandwidth uses other codebooks than its decoder",
        ));
    }

    // Each code and the pad fit in `bits` bits, and every codebook's in one token.
    let bits = u64::try_from(decoder.vocab_size)
        .map_err(|_| Error::unsupported("a negative MusicGen vocabulary"))?;
    let bits = bits.bit_width();
    if i64::from(bits) * decoder.num_codebooks > 62 {
        return Err(Error::unsupported(
            "more MusicGen codes than fit in one token",
        ));
    }
    Ok(MusicgenConfig {
        text: t5::fields_config(&fields.text_encoder)?,
        decoder: MusicgenDecoderConfig {
            width: decoder.hidden_size,
            layers: decoder.num_hidden_layers,
            heads: decoder.num_attention_heads,
            inner: decoder.ffn_dim,
            activation: activation(&decoder.activation_function)?,
            codebooks: decoder.num_codebooks,
            vocab_size: decoder.vocab_size,
            max_positions: decoder.max_position_embeddings,
            eps: 1e-5,
        },
        generation: MusicGeneration {
            codebooks: usize::try_from(decoder.num_codebooks)
                .map_err(|_| Error::unsupported("a negative codebook count"))?,
            pad: generation.decoder_start_token_id,
            bits,
            guidance_scale: generation.guidance_scale.unwrap_or(1.0),
            max_length: generation.max_length,
            sampling: generation.sampling.sampling(),
            sample_rate: codec.sample_rate,
        },
        codec,
    })
}

pub(crate) fn rules(config: &MusicgenConfig) -> Rules {
    let (from, to) = ("decoder.model.decoder.layers.*", "decoder.blocks.*");
    Rules::new()
        .rename("enc_to_dec_proj", "enc_to_dec")
        .rename("decoder.model.decoder.embed_tokens", "decoder.tokens")
        .rename(
            "decoder.model.decoder.embed_positions.weights",
            "decoder.positions",
        )
        .rename("decoder.model.decoder.layer_norm", "decoder.norm")
        .rename("decoder.lm_heads", "decoder.heads")
        .then(attention_rules(
            &format!("{from}.self_attn"),
            &format!("{to}.attention"),
        ))
        .then(attention_rules(
            &format!("{from}.encoder_attn"),
            &format!("{to}.cross_attention"),
        ))
        .rename_each(
            from,
            to,
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
        .then(t5::encoder_rules("text_encoder"))
        .then(encodec::rules(&config.codec, "audio_encoder"))
}
