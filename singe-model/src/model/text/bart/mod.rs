// BART: an encoder and a decoder of blocks that norm after each residual, sharing one token embedding with the output, which adds a bias of its own; TAPEX is BART trained on tables.

mod stack;
mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Linear, Model, ModelState, StateBuilder, StateKey},
};

pub use self::stack::{BartStack, BartStackConfig, stack_rules};
use crate::{
    error::Result,
    format::{Loader, transformers::seq2seq_generation},
    head::{DecoderCache, HeadLoader, Seq2SeqInput, Seq2SeqLayers, Seq2SeqLm},
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
    encoder: BartStack,
    decoder: BartStack,
    head: Linear,
    logits_bias: StateKey,
}

impl Seq2SeqLayers for BartLayers {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor> {
        let Seq2SeqInput::Tokens(batch) = input else {
            return Err(singe::Error::invalid_argument("BART reads tokens"));
        };
        self.encoder.forward(batch.tokens(), state)
    }

    fn start(&self, encoded: &Tensor, state: &ModelState) -> SingeResult<DecoderCache> {
        self.decoder.start(encoded, state)
    }

    fn next_scores(
        &self,
        cache: &mut DecoderCache,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.decoder
            .last_state(cache, tokens, state)?
            .apply(&self.head, state)?
            .add(&state.tensor(&self.logits_bias)?.squeeze(0)?)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "bart",
    formats: &[(Format::Transformers, &["bart"])],
    heads: &[HeadLoader::Seq2SeqLm(load_seq2seq)],
};

pub fn load_seq2seq(loader: &mut Loader<'_>) -> Result<Seq2SeqLm> {
    let config: BartConfig = loader.config(transformers::config)?;
    let generation = seq2seq_generation(loader.source())?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let (vocabulary, width) = (config.decoder.vocab_size, config.decoder.width);
        let tokens = schema.parameter("tokens.weight", [vocabulary, width])?;
        BartLayers {
            encoder: BartStack::create(
                &mut schema.scope("encoder"),
                &config.encoder,
                false,
                Some(&tokens),
            )?,
            decoder: BartStack::create(
                &mut schema.scope("decoder"),
                &config.decoder,
                true,
                Some(&tokens),
            )?,
            head: Linear::new(schema.tie("head.weight", &tokens)?, None),
            logits_bias: schema.parameter("logits_bias", [1, vocabulary])?,
        }
    };
    let layers: Box<dyn Seq2SeqLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(Seq2SeqLm::new(model, generation))
}
