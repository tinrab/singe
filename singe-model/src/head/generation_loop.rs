// What an exported generation loop runs, which the generating heads give.

use singe::{
    generation::{CachedDecoder, DecoderStepGraph, GenerationOptions},
    graph::Graph,
    nn::DecoderModel,
    random::Generator,
};

use crate::error::Result;

// The sizes an exported generation loop fixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopLimits {
    // The tokens a sequence holds, prompt included.
    pub capacity: i64,
    // The rows of the prompt step, beside the one-token step.
    pub prompt_rows: usize,
    // The sequences the loop runs at once, one per row.
    pub rows: usize,
    // The seconds of audio an encode step reads.
    pub audio_seconds: i64,
    // The most patches of an image an encode step reads.
    pub image_patches: Option<i64>,
}

// What an exported generation loop runs: an encode step that writes a prompt's sources, the one-token step, a prompt step, and how they choose tokens.
#[derive(Debug)]
pub struct LoopSteps {
    pub encode: Option<Graph>,
    pub step: DecoderStepGraph,
    pub prompt: Option<DecoderStepGraph>,
    pub generation: GenerationOptions,
}

impl LoopSteps {
    // Captures `decoder`'s one-token step and, with `prompt_rows`, a step that reads that many more rows, each choosing as `generation` asks.
    pub fn create<L: DecoderModel>(
        decoder: &CachedDecoder<L>,
        encode: Option<Graph>,
        generation: &GenerationOptions,
        prompt_rows: Option<usize>,
    ) -> Result<Self> {
        // The loop seeds the draws, so the generator's own seed is never read.
        let generator = Generator::new(0);
        let step = decoder.choosing_step_graph(0, generation, Some(&generator))?;
        let prompt = prompt_rows
            .map(|rows| {
                decoder.choosing_step_graph(
                    rows.saturating_sub(1).max(1),
                    generation,
                    Some(&generator),
                )
            })
            .transpose()?;
        Ok(Self {
            encode,
            step,
            prompt,
            generation: generation.clone(),
        })
    }
}
