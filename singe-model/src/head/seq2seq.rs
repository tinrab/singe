//! Encoder-decoder generation, as `transformers`' `generate` runs it for T5, BART, `TrOCR`, and Donut: the input encoded once, then each token chosen after the tokens before it, greedily or by beam search, under the snapshot's generation config.
//! The decoder is a cached decoder, compiled or eager, whose rows each read an encoding written to their cross-attention caches first; beam search runs a row per beam, each step copying its parents' cached keys and values into their rows' pages.

use std::{collections::HashSet, fmt::Debug, slice, sync::Arc};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Tensor,
    generation::{CachedDecoder, Decoder, DecoderOptions, PromptChunking},
    nn::{
        CacheSize, CrossAttention, DecodeCache, DecodeStep, DecoderModel, EncoderDecoderCache,
        EncoderDecoderCacheSpec, KvCacheSpec, Model, ModelState, TokenBatch,
    },
};

use crate::{
    decoder_cache::{DecoderCache, bucketed_capacity, cached_decoder},
    error::{Error, Result},
    head::{Boxed, argmax},
    host,
    processor::text::TextTokenizer,
};

const UNREACHABLE: f32 = -1.0e9;

/// The multiple a kept decoder's encoder states per sequence are rounded up to, so inputs of nearby lengths share one decoder.
const SOURCE_BUCKET: i64 = 256;

/// The slots of a cross-attention cache page, each of which runs as a sequence of its own.
const SOURCE_PAGE: i64 = 64;

/// The key of a kept decoder: its rows, its capacity, the encoder states each row reads at most, and whether it compiles.
type Seq2SeqKey = (usize, i64, i64, bool);

/// What an encoder-decoder reads: tokens, images' pixels, or an image with a prompt whose placeholder tokens its features take the place of, as Florence-2 reads them.
#[derive(Debug, Clone, Copy)]
pub enum Seq2SeqInput<'a> {
    Tokens(&'a TokenBatch),
    Pixels(&'a Tensor),
    Prompted {
        tokens: &'a Tensor,
        pixels: &'a Tensor,
    },
}

/// An encoder-decoder's layers.
pub trait Seq2SeqLayers: Debug + Send + Sync {
    /// Returns the encoder's states for one input, `[1, S, D]`.
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor>;

    /// Returns the decoder that reads the encoder's states.
    fn decoder(&self) -> Arc<dyn Seq2SeqDecoding>;
}

/// An encoder-decoder's decoder, as a cached decoder runs it.
pub trait Seq2SeqDecoding: Debug + Send + Sync {
    /// Returns each block's cross-attention, which reads the encoder's states.
    fn cross_attentions(&self) -> Vec<&CrossAttention>;

    /// Returns the heads of its self-attention's cache and their width.
    fn cache_spec(&self) -> KvCacheSpec;

    /// Runs one step and returns the logits of each sequence's last token, `[S, V]`.
    fn forward(
        &self,
        step: &DecodeStep,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor>;
}

/// A decoder whose sequences read up to `sources` encoder states each.
#[derive(Debug, Clone)]
pub(crate) struct Seq2SeqDecoder {
    layers: Arc<dyn Seq2SeqDecoding>,
    sources: i64,
}

impl Seq2SeqDecoder {
    pub(crate) fn new(layers: Arc<dyn Seq2SeqDecoding>, sources: i64) -> Self {
        Self { layers, sources }
    }

    /// Writes sequence `sequence`'s encoder states, `[S, D]`, to each block's cross-attention cache.
    pub(crate) fn write_sources(
        &self,
        encoded: &Tensor,
        sequence: usize,
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<()> {
        for (attention, cache) in self.layers.cross_attentions().into_iter().zip(caches) {
            let (keys, values) = attention.sources(encoded, state)?;
            cache.write_cross(sequence, &keys, &values)?;
        }
        Ok(())
    }
}

impl DecoderModel for Seq2SeqDecoder {
    type Cache = EncoderDecoderCache;

    /// No tables: what a decoder looks up by position, it looks up in its state.
    type Tables = ();

    fn tables(&self, _: DataType, _: Device, _: i64) -> SingeResult<()> {
        Ok(())
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<EncoderDecoderCache>> {
        let KvCacheSpec { kv_heads, head_dim } = self.layers.cache_spec();
        let spec = EncoderDecoderCacheSpec {
            heads: kv_heads,
            head_dim,
            sources: self.sources,
            source_page: SOURCE_PAGE,
        };
        self.layers
            .cross_attentions()
            .iter()
            .map(|_| EncoderDecoderCache::create(device, data_type, size, spec))
            .collect()
    }

    fn forward(
        &self,
        step: &DecodeStep,
        (): &(),
        caches: &[EncoderDecoderCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.layers.forward(step, caches, state)
    }
}

/// How a snapshot generates, from its `generation_config.json`, or from its `config.json` when it has none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Seq2SeqGeneration {
    /// The token the decoder starts from.
    pub decoder_start: Option<i64>,
    /// The tokens that end a sequence.
    pub stop_tokens: Vec<i64>,
    /// The token forced right after the start token.
    pub forced_first: Option<i64>,
    /// The token forced last, when a sequence reaches its maximum length.
    pub forced_last: Option<i64>,
    /// The most tokens a sequence holds, counted with the start token.
    pub max_length: usize,
    /// The fewest tokens a sequence holds before a stop token may end it, counted with the start token.
    pub min_length: usize,
    /// The size of the n-grams that may not repeat, or 0 for none.
    pub no_repeat_ngram: usize,
    /// The size of the encoder's input's n-grams that the answer may not repeat, or 0 for none.
    pub encoder_no_repeat_ngram: usize,
    /// What divides the positive scores of tokens already in the sequence and multiplies their negative ones, 1 for none.
    pub repetition_penalty: f32,
    /// The tokens never chosen.
    pub banned: Vec<i64>,
    /// The beams searched, 1 for greedy generation.
    pub beams: usize,
    /// The power of a finished beam's length that its score is divided by.
    pub length_penalty: f32,
    /// Whether search stops once every beam has finished.
    pub early_stopping: bool,
}

/// A beam: its tokens and its summed log-probability.
#[derive(Debug, Clone)]
struct Beam {
    tokens: Vec<i64>,
    score: f32,
    /// The decoder row whose state, after all but the last token, a running beam continues from.
    parent: usize,
}

fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let most = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = scores
        .iter()
        .map(|score| (score - most).exp())
        .sum::<f32>()
        .ln();
    scores.iter().map(|score| score - most - total).collect()
}

/// Returns the indices of `values`' `count` greatest, greatest first, the earlier of equals first.
fn top(values: &[f32], count: usize) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    host::best_first(&mut order, count, |&index| values[index]);
    order
}

#[derive(Debug)]
pub struct Seq2SeqLm {
    model: Model<Boxed<dyn Seq2SeqLayers>>,
    generation: Seq2SeqGeneration,
    decoders: DecoderCache<Seq2SeqKey, CachedDecoder<Seq2SeqDecoder>>,
}

/// Returns the tokens that would complete an n-gram of `size` already in `tokens`, as `transformers`' `NoRepeatNGramLogitsProcessor` bans them.
fn repeated_ngram_tokens(tokens: &[i64], size: usize) -> HashSet<i64> {
    if size == 0 || tokens.len() + 1 < size {
        return HashSet::new();
    }

    let prefix = &tokens[tokens.len() + 1 - size..];
    tokens
        .windows(size)
        .filter(|window| &window[..size - 1] == prefix)
        .map(|window| window[size - 1])
        .collect()
}

/// Returns the tokens that follow the last `size − 1` of `tokens` wherever those appear in `sources`, as `EncoderNoRepeatNGramLogitsProcessor` bans them.
fn encoder_ngram_tokens(sources: &[i64], tokens: &[i64], size: usize) -> HashSet<i64> {
    if size == 0 || tokens.len() + 1 < size {
        return HashSet::new();
    }

    let prefix = &tokens[tokens.len() + 1 - size..];
    sources
        .windows(size)
        .filter(|window| &window[..size - 1] == prefix)
        .map(|window| window[size - 1])
        .collect()
}

impl Seq2SeqGeneration {
    /// Applies `repetition_penalty` to `scores` for each token in `tokens`, as `RepetitionPenaltyLogitsProcessor` does.
    fn penalize(&self, scores: &mut [f32], tokens: &[i64]) {
        if (self.repetition_penalty - 1.0).abs() < f32::EPSILON {
            return;
        }

        for &token in &tokens.iter().copied().collect::<HashSet<_>>() {
            if let Some(score) = usize::try_from(token)
                .ok()
                .and_then(|token| scores.get_mut(token))
            {
                *score = if *score < 0.0 {
                    *score * self.repetition_penalty
                } else {
                    *score / self.repetition_penalty
                };
            }
        }
    }

    /// Returns what `generate`'s logits processors allow after `tokens`, in a sequence at most `max_length` long, from an encoder that read `sources`.
    /// That is the one token forced first or last, if any, and the tokens kept from being chosen: banned tokens, repeated n-grams, the encoder's input's n-grams, and stop tokens while `tokens` is shorter than `min_length`.
    fn constraints(
        &self,
        tokens: &[i64],
        sources: &[i64],
        max_length: usize,
    ) -> (Option<i64>, HashSet<i64>) {
        let forced = if tokens.len() == 1 {
            self.forced_first
        } else if tokens.len() + 1 == max_length {
            self.forced_last
        } else {
            None
        };
        let mut banned = self.banned.iter().copied().collect::<HashSet<_>>();
        banned.extend(repeated_ngram_tokens(tokens, self.no_repeat_ngram));
        banned.extend(encoder_ngram_tokens(
            sources,
            tokens,
            self.encoder_no_repeat_ngram,
        ));
        if tokens.len() < self.min_length {
            banned.extend(&self.stop_tokens);
        }
        (forced, banned)
    }

    /// Returns the log-probabilities of the token after `tokens` from the decoder's `scores`, penalized for repetition, with the tokens the constraints rule out at negative infinity and a forced token at zero, as beam search processes log-probabilities.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "vocabularies are far smaller than `i64`"
    )]
    fn log_probabilities(
        &self,
        scores: &[f32],
        tokens: &[i64],
        sources: &[i64],
        max_length: usize,
    ) -> Vec<f32> {
        let mut log_probabilities = log_softmax(scores);
        self.penalize(&mut log_probabilities, tokens);
        let (forced, banned) = self.constraints(tokens, sources, max_length);
        for (token, value) in log_probabilities.iter_mut().enumerate() {
            let token = token as i64;
            let allowed = forced.map_or_else(|| !banned.contains(&token), |forced| forced == token);
            if !allowed {
                *value = f32::NEG_INFINITY;
            } else if forced.is_some() {
                *value = 0.0;
            }
        }
        log_probabilities
    }

    fn ends(&self, beam: &Beam, max_length: usize) -> bool {
        beam.tokens
            .last()
            .is_some_and(|token| self.stop_tokens.contains(token))
            || beam.tokens.len() >= max_length
    }

    fn running(&self, candidates: &[Beam], ends: &[bool]) -> Vec<Beam> {
        let continuing = candidates
            .iter()
            .zip(ends)
            .map(|(beam, &end)| beam.score + if end { UNREACHABLE } else { 0.0 })
            .collect::<Vec<_>>();
        top(&continuing, self.beams)
            .into_iter()
            .map(|index| Beam {
                tokens: candidates[index].tokens.clone(),
                score: continuing[index],
                parent: candidates[index].parent,
            })
            .collect()
    }

    fn finished(
        &self,
        finished: &[(Beam, bool)],
        candidates: &[Beam],
        ends: &[bool],
        normalizer: f32,
        improvable: bool,
    ) -> Vec<(Beam, bool)> {
        let all_finished = finished.iter().all(|(_, done)| *done);
        let mut merged = finished.to_vec();
        for (index, (beam, &end)) in candidates.iter().zip(ends).enumerate() {
            let just_finished = end && index < self.beams;
            let mut score = beam.score / normalizer;
            if all_finished && self.early_stopping {
                score += UNREACHABLE;
            }

            if !improvable {
                score += UNREACHABLE;
            }

            if !just_finished {
                score += UNREACHABLE;
            }

            merged.push((
                Beam {
                    tokens: beam.tokens.clone(),
                    score,
                    parent: 0,
                },
                just_finished,
            ));
        }

        let scores = merged
            .iter()
            .map(|(beam, _)| beam.score)
            .collect::<Vec<_>>();
        top(&scores, self.beams)
            .into_iter()
            .map(|index| merged[index].clone())
            .collect()
    }

    /// Returns the best `max(2, 1 + stop tokens) · beams` continuations of the running beams, each the next token after a beam, from each beam's decoder `scores`.
    fn continuations(
        &self,
        running: &[Beam],
        scores: &[Vec<f32>],
        sources: &[i64],
        max_length: usize,
    ) -> Vec<Beam> {
        let keep = (1 + self.stop_tokens.len()).max(2) * self.beams;
        let mut accumulated = Vec::new();
        let mut vocabulary = 0;
        for (beam, scores) in running.iter().zip(scores) {
            let log_probabilities =
                self.log_probabilities(scores, &beam.tokens, sources, max_length);
            vocabulary = log_probabilities.len();
            accumulated.extend(
                log_probabilities
                    .into_iter()
                    .map(|value| value + beam.score),
            );
        }

        top(&accumulated, keep)
            .into_iter()
            .map(|index| {
                let parent = index / vocabulary;
                let mut tokens = running[parent].tokens.clone();
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "vocabularies are far smaller than `i64`"
                )]
                tokens.push((index % vocabulary) as i64);
                Beam {
                    tokens,
                    score: accumulated[index],
                    parent,
                }
            })
            .collect()
    }
}

impl Seq2SeqLm {
    #[must_use]
    pub fn new(model: Model<Boxed<dyn Seq2SeqLayers>>, generation: Seq2SeqGeneration) -> Self {
        Self {
            model,
            generation,
            decoders: DecoderCache::default(),
        }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn Seq2SeqLayers>> {
        &self.model
    }

    #[must_use]
    pub const fn generation(&self) -> &Seq2SeqGeneration {
        &self.generation
    }

    /// Returns the token a decoder starts from, which a pipeline that gives no prompt of its own needs.
    ///
    /// # Errors
    ///
    /// Returns an error if the generation config names no decoder start token.
    pub fn decoder_start(&self) -> Result<i64> {
        self.generation
            .decoder_start
            .ok_or_else(|| Error::unsupported("a generation config without a decoder start token"))
    }

    pub fn encode(&self, input: Seq2SeqInput<'_>) -> Result<Tensor> {
        Ok(self.model.layer().encode(input, self.model.state())?)
    }

    pub fn generate_from(
        &self,
        tokenizer: &TextTokenizer,
        input: Seq2SeqInput<'_>,
        max_new_tokens: Option<usize>,
        compile: bool,
    ) -> Result<String> {
        let sources = match input {
            Seq2SeqInput::Tokens(batch) => batch.tokens().contiguous()?.to_vec::<i64>()?,
            Seq2SeqInput::Prompted { tokens, .. } => tokens.contiguous()?.to_vec::<i64>()?,
            Seq2SeqInput::Pixels(_) => Vec::new(),
        };
        let encoded = self.encode(input)?;
        let generated = self.generate(
            &encoded,
            &sources,
            &[self.decoder_start()?],
            max_new_tokens,
            compile,
        )?;
        tokenizer.decode(&generated, true)
    }

    pub fn generate_from_text(
        &self,
        tokenizer: &TextTokenizer,
        text: &str,
        max_new_tokens: Option<usize>,
        compile: bool,
    ) -> Result<String> {
        let batch = tokenizer.encode_batch(&[text], None, self.model.state().device()?)?;
        self.generate_from(
            tokenizer,
            Seq2SeqInput::Tokens(&batch),
            max_new_tokens,
            compile,
        )
    }

    /// Generates after `prefix`, the decoder's start token and any prompt after it, from the encoder's states, `[1, S, D]`, which read the tokens `sources`, empty for an encoder of pixels, and returns the new tokens.
    /// Generation ends at a stop token, or after `max_new_tokens` new tokens when given and at the config's maximum length otherwise.
    /// It runs greedily or by the config's beam search, with the decoder's steps compiled when `compile` is set.
    ///
    /// # Errors
    ///
    /// Returns an error if `encoded` is not `[1, S, D]`, if `prefix` is empty, if greedy generation finds every token banned, or if a decoder step fails.
    pub fn generate(
        &self,
        encoded: &Tensor,
        sources: &[i64],
        prefix: &[i64],
        max_new_tokens: Option<usize>,
        compile: bool,
    ) -> Result<Vec<i64>> {
        let max_length =
            max_new_tokens.map_or(self.generation.max_length, |new| prefix.len() + new);
        if prefix.len() >= max_length {
            return Ok(Vec::new());
        }

        let encoded = encoded.squeeze(0)?;
        let &[states, _] = encoded.shape().dimensions() else {
            return Err(SingeError::invalid_argument("an encoder's states").into());
        };
        let held = held_sources(states);
        let rows = self.generation.beams.max(1);
        let capacity = bucketed_capacity(max_length);
        let state = self.model.state();
        self.decoders.run(
            (rows, capacity, held, compile),
            || {
                let decoder = Seq2SeqDecoder {
                    layers: self.model.layer().decoder(),
                    sources: held,
                };
                let options =
                    DecoderOptions::new(rows, capacity).with_prompt_chunks(PromptChunking::NONE);
                Ok(cached_decoder(
                    &Model::new(decoder, state.clone()),
                    options,
                    compile,
                )?)
            },
            |decoder| {
                for row in 0..rows {
                    decoder.model().layer().write_sources(
                        &encoded,
                        row,
                        decoder.caches(),
                        state,
                    )?;
                }

                if rows > 1 {
                    self.beam_search(decoder, sources, prefix, max_length)
                } else {
                    self.greedy(decoder, sources, prefix, max_length)
                }
            },
        )
    }

    fn greedy(
        &self,
        decoder: &mut CachedDecoder<Seq2SeqDecoder>,
        sources: &[i64],
        prefix: &[i64],
        max_length: usize,
    ) -> Result<Vec<i64>> {
        let config = &self.generation;
        let mut tokens = prefix.to_vec();
        let mut generated = Vec::new();
        let mut scores = read_prompt(decoder, prefix)?;
        loop {
            let mut row = scores
                .first()
                .ok_or_else(|| SingeError::missing_output("decoder scores"))?
                .clone();
            // Greedy search processes the logits themselves.
            config.penalize(&mut row, &tokens);
            let (forced, banned) = config.constraints(&tokens, sources, max_length);
            let token = match forced {
                Some(token) => token,
                None => argmax(row.iter().copied().enumerate().filter(|&(token, _)| {
                    i64::try_from(token).is_ok_and(|token| !banned.contains(&token))
                }))
                .and_then(|(token, _)| i64::try_from(token).ok())
                .ok_or_else(|| Error::unsupported("generation with every token banned"))?,
            };
            tokens.push(token);
            generated.push(token);
            if config.stop_tokens.contains(&token) || tokens.len() >= max_length {
                return Ok(generated);
            }

            scores = step(decoder, &[slice::from_ref(&token)])?;
        }
    }

    /// Runs beam search as `transformers`' `_beam_search` runs it for one input, with a decoder row per beam.
    /// Each step takes the best `max(2, 1 + stop tokens) · beams` continuations of the running beams.
    /// Those among the best `beams` that end join the finished beams, scored by their summed log-probability over their length to the config's power, and the best `beams` that do not end go on running, each row taking its parent's state.
    /// Search stops once no running beam could beat the worst finished one, as `early_stopping=False` judges it from the current length, once every candidate ends, or, with `early_stopping`, once every beam has finished.
    fn beam_search(
        &self,
        decoder: &mut CachedDecoder<Seq2SeqDecoder>,
        sources: &[i64],
        prefix: &[i64],
        max_length: usize,
    ) -> Result<Vec<i64>> {
        let config = &self.generation;
        let beams = config.beams;
        let prompt = prefix.len();
        let mut scores = read_prompt(decoder, prefix)?;
        let mut running = (0..beams)
            .map(|row| Beam {
                tokens: prefix.to_vec(),
                score: if row == 0 { 0.0 } else { UNREACHABLE },
                parent: row,
            })
            .collect::<Vec<_>>();
        let mut finished = vec![
            (
                Beam {
                    tokens: prefix.to_vec(),
                    score: UNREACHABLE,
                    parent: 0,
                },
                false
            );
            beams
        ];
        let mut improvable = true;
        let mut length = prompt;
        loop {
            let candidates = config.continuations(&running, &scores, sources, max_length);
            let ends = candidates
                .iter()
                .map(|beam| config.ends(beam, max_length))
                .collect::<Vec<_>>();
            let next = config.running(&candidates, &ends);
            let normalizer = ((length + 1 - prompt) as f32).powf(config.length_penalty);
            finished = config.finished(&finished, &candidates, &ends, normalizer, improvable);
            running = next;
            length += 1;

            let best_running =
                running[0].score / ((length - prompt) as f32).powf(config.length_penalty);
            let worst_finished = finished
                .iter()
                .map(|(beam, _)| beam.score)
                .fold(f32::INFINITY, f32::min);
            improvable &= finished
                .iter()
                .any(|(_, done)| best_running > if *done { worst_finished } else { UNREACHABLE });
            let open = !(finished.iter().all(|(_, done)| *done) && config.early_stopping);
            let continues = !ends.iter().all(|&end| end);
            if !(improvable && open && continues) {
                break;
            }

            reorder(decoder, &running)?;
            let last = running
                .iter()
                .map(|beam| {
                    beam.tokens
                        .last()
                        .map(slice::from_ref)
                        .ok_or_else(|| SingeError::missing_input("a running beam's last token"))
                })
                .collect::<SingeResult<Vec<_>>>()?;
            scores = step(decoder, &last)?;
        }
        Ok(finished[0].0.tokens[prompt..].to_vec())
    }
}

/// Returns the encoder states a kept decoder holds per sequence for `states` of them, rounded up to a multiple of `SOURCE_BUCKET`.
pub(crate) const fn held_sources(states: i64) -> i64 {
    (states + SOURCE_BUCKET - 1) / SOURCE_BUCKET * SOURCE_BUCKET
}

/// Reads `prompt` in each of the decoder's rows, a token a step when its steps are compiled, and returns each row's scores for the token after it.
fn read_prompt(
    decoder: &mut CachedDecoder<Seq2SeqDecoder>,
    prompt: &[i64],
) -> Result<Vec<Vec<f32>>> {
    let rows = decoder.sequences();
    let width = decoder
        .largest_chunk()
        .map_or(prompt.len(), |extra| 1 + extra / rows)
        .max(1);
    let mut scores = None;
    for piece in prompt.chunks(width) {
        scores = Some(step(decoder, &vec![piece; rows])?);
    }
    scores.ok_or_else(|| SingeError::invalid_argument("an empty decoder prompt").into())
}

/// Runs one step of `decoder` over each row's `tokens` and returns each row's scores for the token after them.
pub(crate) fn step(
    decoder: &mut CachedDecoder<Seq2SeqDecoder>,
    tokens: &[&[i64]],
) -> Result<Vec<Vec<f32>>> {
    let logits = decoder.step(tokens, None)?;
    let &[_, vocabulary] = logits.shape().dimensions() else {
        return Err(SingeError::invalid_argument("decoder logits").into());
    };
    let vocabulary =
        usize::try_from(vocabulary).map_err(|_| SingeError::out_of_range("vocabulary"))?;
    Ok(host::rows(
        &logits.cast(DataType::F32)?.to_vec::<f32>()?,
        vocabulary,
    ))
}

/// Gives each of the decoder's rows the state of the parent of the running beam it now holds.
fn reorder(decoder: &CachedDecoder<Seq2SeqDecoder>, running: &[Beam]) -> Result<()> {
    let parents = running.iter().map(|beam| beam.parent).collect::<Vec<_>>();
    Ok(decoder.reorder(&parents)?)
}
