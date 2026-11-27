//! An encoder–decoder speech model, as Whisper and Qwen3-ASR are: its config, its encoder and its decoder, each with state of its own, `encoder` and `decoder`, and how its snapshot generates.

use singe::{
    generation::{CachedDecoder, GenerationOptions},
    nn::{DecoderModel, Model},
};

use crate::{
    decoder_cache::{DecoderCache, DecoderKey},
    error::Result,
    format::{Loader, RuleSet, transformers::generation_options},
};

#[derive(Debug, Clone)]
pub struct SpeechModel<C, E, D: DecoderModel> {
    config: C,
    encoder: Model<E>,
    decoder: Model<D>,
    /// The snapshot's stop tokens, suppressed tokens and first suppressed tokens, with a `max_new_tokens` of zero until a caller sets one.
    generation: GenerationOptions,
    decoders: DecoderCache<DecoderKey, CachedDecoder<D>>,
}

impl<C, E, D: DecoderModel> SpeechModel<C, E, D> {
    pub(crate) fn load(
        loader: &mut Loader<'_>,
        config: C,
        (encoder, encoder_rules): (&Model<E>, &RuleSet),
        (decoder, decoder_rules): (&Model<D>, &RuleSet),
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
            decoders: DecoderCache::default(),
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

    /// Returns the options with which `transformers`' `generate` continues a prompt for this snapshot greedily, up to `max_new_tokens`.
    /// The snapshot's stop tokens end a sequence, the `suppress_tokens` of `generation_config.json` are never chosen, and its `begin_suppress_tokens` are never chosen first.
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

    pub(crate) const fn decoders(&self) -> &DecoderCache<DecoderKey, CachedDecoder<D>> {
        &self.decoders
    }
}
