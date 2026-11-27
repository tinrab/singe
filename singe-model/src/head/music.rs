//! Text to music, as `transformers`' `MusicgenForConditionalGeneration.generate` runs it: the text encoded once, then each frame's codes chosen, greedily or sampled, after the frames before it, one code per codebook, with each codebook delayed a step behind the one before it.
//! Classifier-free guidance runs a second decoder row that reads no text, and each code is chosen from the unguided row's logits moved `guidance_scale` times their distance toward the guided row's. A frame's codes reach the decoder packed into one token, `bits` bits per codebook, which the decoder unpacks.

use std::{fmt::Debug, slice, sync::Arc};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, DecoderOptions, PromptChunking},
    nn::{Model, ModelState, TokenBatch},
};

use crate::{
    decoder_cache::{DecoderCache, bucketed_capacity, cached_decoder},
    error::{Error, Result},
    head::{
        Boxed, Sampler, Sampling, SamplingRequest, Seq2SeqDecoding,
        seq2seq::{self, Seq2SeqDecoder},
    },
};

/// A music generator's layers.
pub trait MusicLayers: Debug + Send + Sync {
    /// Returns the text encoder's states as the decoder's cross-attention reads them, `[1, S, D]`.
    fn encode(&self, tokens: &TokenBatch, state: &ModelState) -> SingeResult<Tensor>;

    /// Returns the decoder, whose logits are each codebook's in turn, `[S, K · V]`.
    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding>;

    /// Returns the samples, `[1, N]`, of a clip's codes, `[1, K, T]`.
    fn decode_codes(&self, codes: &Tensor, state: &ModelState) -> SingeResult<Tensor>;
}

/// How a music generator generates, from its config and generation config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicGeneration {
    pub codebooks: usize,
    /// The code each codebook starts from and holds where its delay keeps it from playing.
    pub pad: i64,
    /// The bits of a packed token each codebook's code takes.
    pub bits: u32,
    pub guidance_scale: f32,
    pub max_length: usize,
    /// How the generation config samples each code.
    pub sampling: Sampling,
    pub sample_rate: i64,
}

impl MusicGeneration {
    /// Returns whether codebook `codebook` holds the pad at `position` of a sequence `max_length` long: before its delay has passed, or once the codebooks before it have finished.
    const fn is_padded(&self, codebook: usize, position: usize, max_length: usize) -> bool {
        position <= codebook || position + self.codebooks >= max_length + 1 + codebook
    }

    /// Returns the token of the codes at `position`, each codebook's in its own `bits` bits.
    fn packed(&self, codes: &[Vec<i64>], position: usize) -> i64 {
        codes
            .iter()
            .enumerate()
            .map(|(codebook, codes)| codes[position] << (self.bits as usize * codebook))
            .sum()
    }
}

/// The key of a kept decoder: its rows, its capacity, and the encoder states each row reads at most.
type MusicKey = (usize, i64, i64);

#[derive(Debug)]
pub struct TextToMusic {
    model: Model<Boxed<dyn MusicLayers>>,
    generation: MusicGeneration,
    decoders: DecoderCache<MusicKey, CachedDecoder<Seq2SeqDecoder>>,
}

impl TextToMusic {
    #[must_use]
    pub fn new(model: Model<Boxed<dyn MusicLayers>>, generation: MusicGeneration) -> Self {
        Self {
            model,
            generation,
            decoders: DecoderCache::default(),
        }
    }

    #[must_use]
    pub const fn sample_rate(&self) -> i64 {
        self.generation.sample_rate
    }

    /// Generates a clip from `tokens`, a tokenized text, `[1, T]`, choosing codes as `sampling` asks over the generation config's sampling, and returns its samples.
    /// It runs `max_new_tokens` decoder steps when given and the generation config's maximum length otherwise; a clip has that many frames less the codebooks' delays.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder's states are not `[1, S, D]` or if a decoder step or the codec fails.
    pub fn generate(
        &self,
        tokens: &TokenBatch,
        max_new_tokens: Option<usize>,
        sampling: &SamplingRequest,
    ) -> Result<Vec<f32>> {
        let generation = &self.generation;
        let state = self.model.state();
        let layers = self.model.layer();
        let encoded = layers.encode(tokens, state)?.squeeze(0)?;
        let &[sources, width] = encoded.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a music encoder's states").into());
        };
        let max_length = max_new_tokens.map_or(generation.max_length, |new| new + 1);
        let guided = generation.guidance_scale > 1.0;
        let rows = if guided { 2 } else { 1 };
        let held = seq2seq::held_sources(sources);
        let capacity = bucketed_capacity(max_length);
        let codes = self.decoders.run(
            (rows, capacity, held),
            || {
                let decoder = Seq2SeqDecoder::new(layers.decoder(), held);
                let options =
                    DecoderOptions::new(rows, capacity).with_prompt_chunks(PromptChunking::NONE);
                Ok(cached_decoder(
                    &Model::new(decoder, state.clone()),
                    options,
                    false,
                )?)
            },
            |decoder| {
                let layer = decoder.model().layer();
                layer.write_sources(&encoded, 0, decoder.caches(), state)?;
                if guided {
                    // The unguided row reads one state of zeros, which gives its cross-attention the zeros a fully masked text gives `transformers`.
                    let unguided = Tensor::zeroed_on(
                        encoded.device(),
                        encoded.data_type(),
                        Shape::create([1, width])?,
                    )?;
                    layer.write_sources(&unguided, 1, decoder.caches(), state)?;
                }
                self.choose(
                    decoder,
                    rows,
                    max_length,
                    &mut sampling.sampler(generation.sampling),
                )
            },
        )?;

        let frames = max_length.saturating_sub(generation.codebooks);
        if frames == 0 {
            return Ok(Vec::new());
        }

        // Each codebook's frames start after its delay.
        let played = codes
            .iter()
            .enumerate()
            .flat_map(|(codebook, codes)| codes[codebook + 1..codebook + 1 + frames].to_vec())
            .collect::<Vec<_>>();
        let shape = Shape::create([
            1,
            i64::try_from(generation.codebooks)
                .map_err(|_| SingeError::out_of_range("codebooks"))?,
            i64::try_from(frames).map_err(|_| SingeError::out_of_range("frames"))?,
        ])?;
        let codes = Tensor::from_vec(state.device()?, shape, played)?;
        Ok(layers
            .decode_codes(&codes, state)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?)
    }

    /// Returns each codebook's codes at every position of a sequence `max_length` long, chosen by `sampler` from the decoder's rows.
    fn choose(
        &self,
        decoder: &mut CachedDecoder<Seq2SeqDecoder>,
        rows: usize,
        max_length: usize,
        sampler: &mut Sampler,
    ) -> Result<Vec<Vec<i64>>> {
        let generation = &self.generation;
        let mut codes = vec![vec![generation.pad; max_length]; generation.codebooks];
        let start = generation.packed(&codes, 0);
        let mut scores = seq2seq::step(decoder, &vec![slice::from_ref(&start); rows])?;
        for position in 1..max_length {
            for (codebook, codes) in codes.iter_mut().enumerate() {
                codes[position] = if generation.is_padded(codebook, position, max_length) {
                    generation.pad
                } else {
                    self.chosen(&scores, codebook, sampler)?
                };
            }

            if position + 1 == max_length {
                break;
            }

            let token = generation.packed(&codes, position);
            scores = seq2seq::step(decoder, &vec![slice::from_ref(&token); rows])?;
        }
        Ok(codes)
    }

    /// Returns codebook `codebook`'s code chosen by `sampler` from the guided logits, `ClassifierFreeGuidanceLogitsProcessor`'s `unguided + (guided − unguided) · scale`.
    fn chosen(&self, scores: &[Vec<f32>], codebook: usize, sampler: &mut Sampler) -> Result<i64> {
        let generation = &self.generation;
        let guided = scores
            .first()
            .ok_or_else(|| SingeError::missing_output("decoder scores"))?;
        let vocabulary = guided.len() / generation.codebooks;
        let range = codebook * vocabulary..(codebook + 1) * vocabulary;
        let guided = &guided[range.clone()];
        let logits = scores.get(1).map_or_else(
            || guided.to_vec(),
            |unguided| {
                unguided[range]
                    .iter()
                    .zip(guided)
                    .map(|(&unguided, &guided)| {
                        unguided + (guided - unguided) * generation.guidance_scale
                    })
                    .collect()
            },
        );
        i64::try_from(sampler.choose(&logits)?).map_err(|_| Error::unsupported("a code past `i64`"))
    }
}
