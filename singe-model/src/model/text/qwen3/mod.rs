// Qwen3: grouped-query attention with normed queries and keys, and a gated feed-forward layer; as Qwen3-Embedding, its final states embed texts.

mod transformers;

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Layer, Model, ModelState, TokenBatch},
};

use crate::{
    assembly::{
        DecoderConfig, GroupedDecoder, causal, grouped_decoder, hidden_states, llama_rules,
    },
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

pub type Qwen3 = Model<GroupedDecoder>;

// Qwen3 as a text encoder: each token's final normed state, `[B, T, W]`, from causal attention over the tokens before it, padding rows zero.
#[derive(Debug)]
pub struct Qwen3Encoder(Qwen3);

impl Layer<TokenBatch> for Qwen3Encoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, columns] = input.tokens().shape().dimensions() else {
            return Err(SingeError::invalid_argument("a token batch"));
        };
        let tokens = input.tokens().contiguous()?.to_vec::<i64>()?;
        let lengths = input.key_lengths()?.cast(DataType::I64)?.to_vec::<i64>()?;
        let width = usize::try_from(columns).map_err(|_| SingeError::out_of_range("columns"))?;
        let sequences = lengths
            .iter()
            .enumerate()
            .map(|(row, &length)| {
                let length =
                    usize::try_from(length).map_err(|_| SingeError::out_of_range("lengths"))?;
                Ok(&tokens[row * width..row * width + length])
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let hidden = hidden_states(self.0.layer(), &sequences, state)?;
        let &[_, features] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a decoder's hidden states"));
        };
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
