use std::fmt::Debug;

use singe::{
    generation::{CachedDecoder, Decoder, DecoderOptions, GenerationOptions},
    nn::{DecoderModel, Model},
};

use crate::{
    decoder_cache::cached_decoder,
    error::Result,
    head::{LoopLimits, LoopStepSet},
};

/// A model that generates text a token at a time.
pub trait CausalLm: Debug {
    /// Returns a decoder over the model, compiled if `compile` is true.
    fn decoder(&self, options: DecoderOptions, compile: bool) -> Result<Box<dyn Decoder>>;

    /// Returns the steps an exported generation loop runs within `limits`, each choosing tokens as `generation` asks.
    fn loop_steps(
        &self,
        limits: &LoopLimits,
        generation: &GenerationOptions,
    ) -> Result<LoopStepSet>;
}

impl<L: DecoderModel> CausalLm for Model<L> {
    fn decoder(&self, options: DecoderOptions, compile: bool) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(cached_decoder(self, options, compile)?))
    }

    fn loop_steps(
        &self,
        limits: &LoopLimits,
        generation: &GenerationOptions,
    ) -> Result<LoopStepSet> {
        let decoder =
            CachedDecoder::create(self, DecoderOptions::new(limits.rows, limits.capacity))?;
        LoopStepSet::create(&decoder, None, generation, Some(limits.prompt_rows))
    }
}
