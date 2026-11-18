// Llama: multi-head attention over rotary positions and a gated feed-forward layer.

mod transformers;

use singe::nn::Model;

use crate::{
    assembly::{DecoderConfig, GroupedDecoder, causal, grouped_decoder, llama_rules},
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    source::Format,
};

pub type LlamaLayers = GroupedDecoder;

pub type Llama = Model<LlamaLayers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "llama",
    formats: &[(Format::Transformers, &["llama"])],
    heads: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Llama> {
    let config: DecoderConfig = loader.config(transformers::config)?;
    let schema = grouped_decoder(&config, loader.options().data_type, causal())?;
    let model = loader.load("model", &schema, &llama_rules("model"))?;
    loader.finish()?;
    Ok(model)
}
