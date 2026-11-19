// An encoder–decoder speech model, as Whisper and Qwen3-ASR are: its config, its encoder and its decoder, each with state of its own, `encoder` and `decoder`, and how its snapshot generates.

use singe::{
    generation::{CachedDecoder, GenerationOptions},
    nn::{DecoderModel, Model},
};

use crate::{
    decoder_cache::{DecoderKey, Decoders},
    error::Result,
    format::{Loader, Rules, transformers::generation_options},
};

#[derive(Debug, Clone)]
pub struct SpeechModel<C, E, D: DecoderModel> {
    config: C,
    encoder: Model<E>,
    decoder: Model<D>,
    // The snapshot's stop and suppressed tokens, generating nothing until a caller sets `max_new_tokens`.
    generation: GenerationOptions,
    decoders: Decoders<DecoderKey, CachedDecoder<D>>,
}

impl<C, E, D: DecoderModel> SpeechModel<C, E, D> {
    pub(crate) fn load(
        loader: &mut Loader<'_>,
        config: C,
        (encoder, encoder_rules): (&Model<E>, &Rules),
        (decoder, decoder_rules): (&Model<D>, &Rules),
    ) -> Result<Self> {
        let generation = generation_options(loader.source())?;
        let encoder = loader.load("encoder", encoder, encoder_rules)?;
        let decoder = loader.load("decoder", decoder, decoder_rules)?;
        loader.finish()?;
        Ok(Self {
            config,
            encoder,
            decoder,
            generation,
            decoders: Decoders::default(),
        })
    }

    #[must_use]
    pub const fn config(&self) -> &C {
        &self.config
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<E> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<D> {
        &self.decoder
    }

    // How `transformers`' `generate` continues a prompt for this snapshot, greedily, up to `max_new_tokens`: the generation config's end of text stops a sequence, and its suppressed tokens are not chosen.
    #[must_use]
    pub fn generation_options(&self, max_new_tokens: usize) -> GenerationOptions {
        GenerationOptions {
            max_new_tokens,
            ..self.generation.clone()
        }
    }

    pub(crate) const fn generation(&self) -> &GenerationOptions {
        &self.generation
    }

    pub(crate) const fn decoders(&self) -> &Decoders<DecoderKey, CachedDecoder<D>> {
        &self.decoders
    }
}
