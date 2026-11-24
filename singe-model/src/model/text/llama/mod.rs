//! Llama: grouped-query attention over rotary positions and a gated feed-forward layer.

mod transformers;

use singe::nn::Model;

use crate::{
    assembly::{DecoderConfig, GroupedDecoder, causal, grouped_decoder, llama_rules},
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

pub type Llama = Model<GroupedDecoder>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "llama",
    formats: &[(Format::Transformers, &["llama"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Llama> {
    let config: DecoderConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        grouped_decoder(schema, &config, causal())
    })?;
    loader.load_whole(&model, &llama_rules("model"))
}
