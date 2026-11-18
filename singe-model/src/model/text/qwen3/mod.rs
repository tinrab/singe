// Qwen3: grouped-query attention with normed queries and keys, and a gated feed-forward layer; as Qwen3-Embedding, its final states embed texts.

mod transformers;

use singe::{
    DataType, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Model, ModelState, TokenBatch},
};

use crate::{
    assembly::{
        DecoderConfig, GroupedDecoder, causal, grouped_decoder, hidden_states, llama_rules,
    },
    error::Result,
    format::Loader,
    head::{HeadLoader, TextEncoderLayers},
    model::ModelEntry,
    source::Format,
};

pub type Qwen3Layers = GroupedDecoder;

pub type Qwen3 = Model<Qwen3Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "qwen3",
    formats: &[(Format::Transformers, &["qwen3"])],
    heads: &[
        HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?))),
        HeadLoader::TextEncoder(|loader| {
            let model = load(loader)?;
            let state = model.state().clone();
            Ok(Model::new(
                TextEncoderLayers(Box::new(Qwen3Encoder(model))),
                state,
            ))
        }),
    ],
};

// Qwen3 as a text encoder: each token's final normed state, `[B, T, W]`, from causal attention over the tokens before it, padding rows zero.
#[derive(Debug)]
pub struct Qwen3Encoder(Qwen3);

impl Layer<TokenBatch> for Qwen3Encoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = input.tokens().shape().dimensions();
        let columns = dimensions[1];
        let tokens = input.tokens().contiguous()?.to_vec::<i64>()?;
        let lengths = input.key_lengths()?.cast(DataType::I64)?.to_vec::<i64>()?;
        let width = usize::try_from(columns).map_err(|_| singe::Error::out_of_range("columns"))?;
        let sequences = lengths
            .iter()
            .enumerate()
            .map(|(row, &length)| {
                let length =
                    usize::try_from(length).map_err(|_| singe::Error::out_of_range("lengths"))?;
                Ok(&tokens[row * width..row * width + length])
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let hidden = hidden_states(self.0.layer(), &sequences, state)?;
        let features = hidden.shape().dimensions()[1];
        // Each sequence's rows, then zeros to the batch's length.
        let mut start = 0;
        let mut padded = Vec::with_capacity(sequences.len());
        for &length in &lengths {
            let mut parts = vec![hidden.narrow(0, start, length)?];
            if length < columns {
                parts.push(Tensor::zeroed_on(
                    hidden.device(),
                    hidden.data_type(),
                    Shape::create([columns - length, features])?,
                )?);
            }

            padded.push(Tensor::concat(&parts, 0)?.unsqueeze(0)?);
            start += length;
        }
        Tensor::concat(&padded, 0)
    }
}

pub fn load(loader: &mut Loader<'_>) -> Result<Qwen3> {
    let config: DecoderConfig = loader.config(transformers::config)?;
    let schema = grouped_decoder(&config, loader.options().data_type, causal())?;
    // Qwen3-Embedding stores the bare `Qwen3Model`, without the language model's prefix.
    let rules = llama_rules("model").then(llama_rules(""));
    let model = loader.load("model", &schema, &rules)?;
    loader.finish()?;
    Ok(model)
}
