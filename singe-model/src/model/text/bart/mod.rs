//! BART: an encoder and a decoder of blocks that norm after each residual, sharing one token embedding with the output, which adds a bias of its own; TAPEX is BART trained on tables.

mod transformers;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{Linear, ModelState},
};

use crate::{
    assembly::{BartDecoder, BartEncoder, BartStackConfig},
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{Boxed, HeadLoader, Seq2SeqDecoding, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BartConfig {
    pub encoder: BartStackConfig,
    pub decoder: BartStackConfig,
}

#[derive(Debug, Clone)]
pub struct BartLayers {
    encoder: BartEncoder,
    decoder: Arc<dyn Seq2SeqDecoding>,
}

impl BartLayers {
    pub fn create(schema: &mut Schema<'_>, config: &BartConfig) -> SingeResult<Self> {
        let (vocabulary, width) = (config.decoder.vocab_size, config.decoder.width);
        let tokens = schema.parameter("tokens.weight", [vocabulary, width])?;
        let head = Linear::new(schema.tie("head.weight", &tokens)?, None);
        let logits_bias = schema.parameter("logits_bias", [1, vocabulary])?;
        Ok(Self {
            encoder: BartEncoder::create(
                &mut schema.scope("encoder"),
                &config.encoder,
                Some(&tokens),
            )?,
            decoder: Arc::new(BartDecoder::create(
                &mut schema.scope("decoder"),
                &config.decoder,
                Some(&tokens),
                head,
                Some(logits_bias),
            )?),
        })
    }
}

impl Seq2SeqLayers for BartLayers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Tokens(batch) = input else {
            return Err(SingeError::invalid_argument("BART reads tokens"));
        };
        self.encoder.forward(batch.tokens(), state)
    }

    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding> {
        Arc::clone(&self.decoder)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "bart",
    formats: &[(Format::Transformers, &["bart"])],
    loaders: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: BartConfig = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let model = Schema::model(loader.options().data_type, |schema| {
        BartLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn Seq2SeqLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(Seq2SeqLm::new(model, generation))
}
