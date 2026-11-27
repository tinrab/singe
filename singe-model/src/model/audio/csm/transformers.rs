//! Reads a `transformers` CSM snapshot: the backbone's, the depth decoder's, and the codec's configs, and its tensors' names.

use serde::Deserialize;
use singe::{DataType, Result as SingeResult, Tensor};

use crate::{
    assembly::llama_block_rules,
    error::Result,
    format::{
        RuleSet,
        transformers::{DecoderDefaults, DecoderFields},
    },
    head::SamplingFields,
    model::audio::{
        csm::CsmConfig,
        mimi::transformers::{Fields as MimiFields, fields_config},
    },
    source::Source,
};

#[derive(Debug, Deserialize)]
struct Fields {
    #[serde(flatten)]
    backbone: DecoderFields,
    depth_decoder_config: DecoderFields,
    codec_config: MimiFields,
    num_codebooks: i64,
    codebook_eos_token_id: i64,
    audio_token_id: i64,
    text_vocab_size: i64,
}

#[derive(Debug, Deserialize)]
struct GenerationFields {
    /// `CsmForConditionalGeneration`'s default frames when the generation config names none.
    #[serde(default = "frames")]
    max_new_tokens: usize,
    #[serde(flatten)]
    sampling: SamplingFields,
    #[serde(default)]
    depth_decoder_do_sample: bool,
    #[serde(default)]
    depth_decoder_temperature: Option<f32>,
    #[serde(default)]
    depth_decoder_top_k: Option<usize>,
}

impl Default for GenerationFields {
    fn default() -> Self {
        Self {
            max_new_tokens: frames(),
            sampling: SamplingFields::default(),
            depth_decoder_do_sample: false,
            depth_decoder_temperature: None,
            depth_decoder_top_k: None,
        }
    }
}

const fn frames() -> usize {
    125
}

pub(crate) fn config(source: &Source) -> Result<CsmConfig> {
    let fields: Fields = source.json("config.json")?;
    let defaults = DecoderDefaults {
        tied: false,
        head_dim: None,
    };
    fields.backbone.refuse_sliding_layers()?;
    let generation = source
        .optional_json::<GenerationFields>("generation_config.json")?
        .unwrap_or_default();
    let depth_sampling = SamplingFields {
        do_sample: generation.depth_decoder_do_sample,
        temperature: generation.depth_decoder_temperature,
        top_k: generation.depth_decoder_top_k,
    }
    .sampling();
    Ok(CsmConfig {
        backbone: fields.backbone.config(defaults)?,
        depth: fields.depth_decoder_config.config(defaults)?,
        codec: fields_config(fields.codec_config)?,
        codebooks: fields.num_codebooks,
        end_code: fields.codebook_eos_token_id,
        audio_token_id: fields.audio_token_id,
        text_vocabulary: fields.text_vocab_size,
        frames: generation.max_new_tokens,
        sampling: generation.sampling.sampling(),
        depth_sampling,
    })
}

/// Returns `lm_head`'s weight, `[V, W]`, with a `[W, W]` identity under it, so the backbone's head gives its last state beside its logits.
fn with_identity(weight: &Tensor) -> SingeResult<Tensor> {
    let width = weight.shape().dimensions()[1];
    let range = Tensor::range(weight.device(), 0_i64, width)?;
    let identity = range
        .unsqueeze(1)?
        .equal(&range.unsqueeze(0)?)?
        .cast(DataType::F32)?;
    Tensor::concat(&[weight.cast(DataType::F32)?, identity], 0)
}

/// The backbone part, skipping the depth decoder, whose part reads the audio embeddings, and the codec.
pub(crate) fn backbone_rules() -> RuleSet {
    RuleSet::new()
        .skip("depth_decoder")
        .skip("codec_model")
        .rename("embed_text_tokens", "embeddings.tokens")
        .then(llama_block_rules("backbone_model.layers.*", "blocks.*"))
        .rename("backbone_model.norm", "norm")
        .derive("lm_head.weight", "head.weight", with_identity)
}

/// The depth part: the audio embeddings, stored as the depth decoder's, which the backbone's frames tie to, its projector, blocks, norm, and per-codebook head.
pub(crate) fn depth_rules() -> RuleSet {
    RuleSet::new()
        .rename("depth_decoder.model.embed_tokens.weight", "audio")
        .rename("depth_decoder.model.inputs_embeds_projector", "projector")
        .then(llama_block_rules(
            "depth_decoder.model.layers.*",
            "blocks.*",
        ))
        .rename("depth_decoder.model.norm", "norm")
        .rename("depth_decoder.codebooks_head.weight", "head")
}
