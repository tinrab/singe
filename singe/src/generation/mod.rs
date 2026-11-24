// Token generation over any decoder that turns tokens into logits, and the decoder machinery a cached model shares.
// A step reads the tokens the step before chose on the device, and the host reads them while it runs, for stop checks and tokenizers.

mod batch;
mod cached;
mod choosing;
mod compiled;
mod decoder;
mod reading;
mod speculative;

pub use self::{
    batch::{GeneratedToken, RunningBatch},
    cached::{CachedDecoder, DecoderOptions, DecoderStepGraph, PromptChunks},
    choosing::choose,
    decoder::{Decoder, GenerationOptions, SequenceOptions, generate, generate_batch},
    speculative::{DraftingModel, Speculation, SpeculativeDecoder},
};
