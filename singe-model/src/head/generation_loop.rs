//! What an exported generation loop runs, which the generating heads give.

use singe::{
    generation::{CachedDecoder, DecoderOptions, DecoderStepGraph, GenerationOptions},
    graph::Graph,
    nn::{DecoderModel, Model},
};

use crate::error::Result;

/// The sizes an exported generation loop fixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopLimits {
    /// The tokens a sequence holds, prompt included.
    pub capacity: i64,
    /// The rows the prompt step reads at once, beside the one-token step.
    pub prompt_rows: usize,
    /// The sequences the loop runs at once, one per row.
    pub rows: usize,
    /// The seconds of audio an encode step reads.
    pub audio_seconds: i64,
    /// The most patches of an image an encode step reads.
    pub image_patches: Option<i64>,
}

/// What an exported generation loop runs: an encode step that writes a prompt's sources, the one-token step, a prompt step, and how they choose tokens.
#[derive(Debug)]
pub struct LoopStepSet {
    pub encode: Option<Graph>,
    pub step: DecoderStepGraph,
    pub prompt: Option<DecoderStepGraph>,
    pub generation: GenerationOptions,
}

impl LoopStepSet {
    /// Captures `decoder`'s one-token step and, with `prompt_rows`, a step that reads that many rows at once, at least two, each choosing tokens as `generation` asks.
    pub fn create<L: DecoderModel>(
        decoder: &CachedDecoder<L>,
        encode: Option<Graph>,
        generation: &GenerationOptions,
        prompt_rows: Option<usize>,
    ) -> Result<Self> {
        let step = decoder.choosing_step_graph(0, generation)?;
        let prompt = prompt_rows
            .map(|rows| decoder.choosing_step_graph(rows.saturating_sub(1).max(1), generation))
            .transpose()?;
        Ok(Self {
            encode,
            step,
            prompt,
            generation: generation.clone(),
        })
    }

    pub fn one_sequence<L: DecoderModel>(
        model: &Model<L>,
        limits: &LoopLimits,
        generation: &GenerationOptions,
        encode: impl FnOnce(&CachedDecoder<L>) -> Result<Option<Graph>>,
    ) -> Result<Self> {
        let decoder = CachedDecoder::create(model, DecoderOptions::new(1, limits.capacity))?;
        let encode = encode(&decoder)?;
        Self::create(&decoder, encode, generation, Some(limits.prompt_rows))
    }
}
