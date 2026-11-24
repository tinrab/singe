use std::result::Result as StdResult;

use singe::{
    Error as SingeError,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions},
    nn::{DecoderModel, Model},
};

use crate::decoder_cache::{DecoderKey, Decoders, bucketed_capacity, cached_decoder};

#[derive(Debug, Clone)]
pub struct BatchOptions {
    pub generation: GenerationOptions,
    pub rows: usize,
    pub compile: bool,
}

impl BatchOptions {
    #[must_use]
    pub fn new(max_new_tokens: usize, compile: bool) -> Self {
        Self {
            generation: GenerationOptions {
                max_new_tokens,
                ..GenerationOptions::default()
            },
            rows: 1,
            compile,
        }
    }

    /// Returns a copy of these options with `generation`'s stop, suppressed, and first-step suppressed tokens added to their own, such as those a model's snapshot names.
    #[must_use]
    pub fn including_tokens_of(&self, generation: &GenerationOptions) -> Self {
        let join = |tokens: &mut Vec<i64>, more: &[i64]| {
            for &token in more {
                if !tokens.contains(&token) {
                    tokens.push(token);
                }
            }
        };
        let mut options = self.clone();
        let own = &mut options.generation;
        join(&mut own.stop_tokens, &generation.stop_tokens);
        join(&mut own.suppressed_tokens, &generation.suppressed_tokens);
        join(
            &mut own.first_suppressed_tokens,
            &generation.first_suppressed_tokens,
        );
        options
    }

    #[must_use]
    pub fn decoder_options(&self, prompts: impl Iterator<Item = usize>) -> DecoderOptions {
        let capacity =
            bucketed_capacity(prompts.max().unwrap_or(0) + self.generation.max_new_tokens);
        DecoderOptions::new(self.rows, capacity)
    }

    pub(crate) fn run_decoder<L: DecoderModel, T, E: From<SingeError>>(
        &self,
        decoders: &Decoders<DecoderKey, CachedDecoder<L>>,
        model: &Model<L>,
        options: DecoderOptions,
        run: impl FnOnce(&mut CachedDecoder<L>) -> StdResult<T, E>,
    ) -> StdResult<T, E> {
        decoders.run(
            (options.sequences, options.capacity, self.compile),
            || Ok(cached_decoder(model, options, self.compile)?),
            run,
        )
    }
}
