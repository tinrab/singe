use std::fmt::Debug;

use singe::{
    generation::{CachedDecoder, Decoder, DecoderOptions, GenerationOptions},
    nn::{DecoderModel, Model},
};

use super::{LoopLimits, LoopSteps};
use crate::{assembly, error::Result};

// A decoder that generates text a token at a time.
pub trait CausalLm: Debug {
    // Returns a decoder over the model, compiled if asked.
    fn decoder(&self, options: DecoderOptions, compile: bool) -> Result<Box<dyn Decoder>>;

    // The steps an exported generation loop runs within `limits`, each choosing as `generation` asks.
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps>;
}

impl<L: DecoderModel> CausalLm for Model<L> {
    fn decoder(&self, options: DecoderOptions, compile: bool) -> Result<Box<dyn Decoder>> {
        assembly::decoder(self, options, compile)
    }

    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps> {
        let decoder =
            CachedDecoder::create(self, DecoderOptions::new(limits.rows, limits.capacity))?;
        LoopSteps::create(&decoder, None, generation, Some(limits.prompt_rows))
    }
}
