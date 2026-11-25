//! Qwen3: grouped-query attention with normed queries and keys, and a gated feed-forward layer; as Qwen3-Embedding, its final states embed texts.

mod transformers;

use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState, TokenBatch},
};

use crate::{
    assembly::{DecoderConfig, GroupedDecoder, causal, grouped_decoder, llama_rules},
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

pub type Qwen3 = Model<GroupedDecoder>;

/// Qwen3 as a text encoder: each token's final normed state, `[B, T, W]`, from causal attention over the tokens up to it, in one pass over the padded batch, with padding rows zero.
#[derive(Debug)]
pub struct Qwen3Encoder(Qwen3);

impl Layer<TokenBatch> for Qwen3Encoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .0
            .layer()
            .encode(input.tokens(), &input.key_lengths()?, state)?;
        let real = input.padding().cast(hidden.data_type())?.unsqueeze(2)?;
        hidden.mul(&real)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3",
    formats: &[(Format::Transformers, &["qwen3"])],
    loaders: &[
        HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::TextEncoder(|loader| {
            let model = load(loader)?;
            let state = model.state().clone();
            Ok(Model::new(
                Boxed::<dyn Layer<TokenBatch>>(Box::new(Qwen3Encoder(model))),
                state,
            ))
        }),
    ],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Qwen3> {
    let config: DecoderConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        grouped_decoder(schema, &config, causal())
    })?;
    // Qwen3-Embedding stores the bare `Qwen3Model`, without the language model's prefix.
    let rules = llama_rules("model").then(llama_rules(""));
    loader.load_whole(&model, &rules)
}
