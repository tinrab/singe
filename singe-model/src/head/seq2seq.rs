//! Encoder-decoder generation, as `transformers`' `generate` runs it for T5, BART, `TrOCR`, and Donut: the input encoded once, then each token chosen after the tokens before it, greedily or by beam search, under the snapshot's generation config.
//! The decoder is a cached decoder, compiled or eager, whose rows each read an encoding written to their cross-attention caches first; beam search runs a row per beam, each step taking its parents' rows.

use std::{collections::HashSet, fmt::Debug, slice, sync::Arc};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, Decoder, DecoderOptions, PromptChunks},
    nn::{CacheSize, DecodeCache, DecodeStep, DecoderModel, Model, ModelState, TokenBatch},
};

use crate::{
    assembly::{Attention, Seq2SeqCache, Seq2SeqCacheSpec, Seq2SeqTables},
    decoder_cache::{Decoders, bucketed_capacity, cached_decoder},
    error::{Error, Result},
    head::{Boxed, argmax},
    host,
    processor::text::TextTokenizer,
};

const UNREACHABLE: f32 = -1.0e9;

/// The multiple a kept decoder's encoder states per sequence are rounded up to, so inputs of nearby lengths share one decoder.
const SOURCE_BUCKET: i64 = 256;

/// The key of a kept decoder: its rows, its capacity, the encoder states each row reads at most, and whether it compiles.
type Seq2SeqKey = (usize, i64, i64, bool);

/// What an encoder-decoder reads: tokens, or images' pixels.
#[derive(Debug, Clone, Copy)]
pub enum Seq2SeqInput<'a> {
    Tokens(&'a TokenBatch),
    Pixels(&'a Tensor),
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
    fn cross_attentions(&self) -> Vec<&Attention>;

    /// Returns its caches' sizes for sequences of up to `sources` encoder states.
    fn cache_spec(&self, sources: i64) -> Seq2SeqCacheSpec;

    /// Returns the tables it looks up by position.
    fn tables(
        &self,
        data_type: DataType,
        device: Device,
        capacity: i64,
    ) -> SingeResult<Seq2SeqTables> {
        Seq2SeqTables::create(data_type, device, capacity, None)
    }

    /// Runs one step and returns the logits of each sequence's last token, `[S, V]`.
    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Seq2SeqTables,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<Tensor>;
}

/// A decoder whose sequences read up to `sources` encoder states each.
#[derive(Debug, Clone)]
struct Seq2SeqDecoder {
    layers: Arc<dyn Seq2SeqDecoding>,
    sources: i64,
}

impl Seq2SeqDecoder {
    /// Writes sequence `sequence`'s encoder states, `[S, D]`, to each block's cross-attention cache.
    fn write_sources(
        &self,
        encoded: &Tensor,
        sequence: usize,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<()> {
        for (attention, cache) in self.layers.cross_attentions().into_iter().zip(caches) {
            let (keys, values) = attention.project_sources(encoded, state)?;
            cache.write_sources(sequence, &keys, &values)?;
        }
        Ok(())
    }
}

impl DecoderModel for Seq2SeqDecoder {
    type Cache = Seq2SeqCache;

    type Tables = Seq2SeqTables;

    fn tables(
        &self,
        data_type: DataType,
        device: Device,
        capacity: i64,
    ) -> SingeResult<Seq2SeqTables> {
        self.layers.tables(data_type, device, capacity)
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<Seq2SeqCache>> {
        let spec = self.layers.cache_spec(self.sources);
        self.layers
            .cross_attentions()
            .iter()
            .map(|_| Seq2SeqCache::create(device, data_type, size, spec))
            .collect()
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &Seq2SeqTables,
        caches: &[Seq2SeqCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        self.layers.forward(step, tables, caches, state)
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
    decoders: Decoders<Seq2SeqKey, CachedDecoder<Seq2SeqDecoder>>,
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

impl Seq2SeqGeneration {
    /// Returns what `generate`'s logits processors allow after `tokens`, in a sequence at most `max_length` long.
    /// That is the one token forced first or last, if any, and the tokens kept from being chosen: banned tokens, repeated n-grams, and stop tokens while `tokens` is shorter than `min_length`.
    fn constraints(&self, tokens: &[i64], max_length: usize) -> (Option<i64>, HashSet<i64>) {
        let forced = if tokens.len() == 1 {
            self.forced_first
        } else if tokens.len() + 1 == max_length {
            self.forced_last
        } else {
            None
        };
        let mut banned = self.banned.iter().copied().collect::<HashSet<_>>();
        banned.extend(repeated_ngram_tokens(tokens, self.no_repeat_ngram));
        if tokens.len() < self.min_length {
            banned.extend(&self.stop_tokens);
        }
        (forced, banned)
    }

    /// Returns the log-probabilities of the token after `tokens` from the decoder's `scores`, with the tokens the constraints rule out at negative infinity and a forced token at zero.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "vocabularies are far smaller than `i64`"
    )]
    fn log_probabilities(&self, scores: &[f32], tokens: &[i64], max_length: usize) -> Vec<f32> {
        let mut log_probabilities = log_softmax(scores);
        let (forced, banned) = self.constraints(tokens, max_length);
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
    fn continuations(&self, running: &[Beam], scores: &[Vec<f32>], max_length: usize) -> Vec<Beam> {
        let keep = (1 + self.stop_tokens.len()).max(2) * self.beams;
        let mut accumulated = Vec::new();
        let mut vocabulary = 0;
        for (beam, scores) in running.iter().zip(scores) {
            let log_probabilities = self.log_probabilities(scores, &beam.tokens, max_length);
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
            decoders: Decoders::default(),
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
        let encoded = self.encode(input)?;
        let generated =
            self.generate(&encoded, &[self.decoder_start()?], max_new_tokens, compile)?;
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

    /// Generates after `prefix`, the decoder's start token and any prompt after it, from the encoder's states, `[1, S, D]`, and returns the new tokens.
    /// Generation ends at a stop token, or after `max_new_tokens` new tokens when given and at the config's maximum length otherwise.
    /// It runs greedily or by the config's beam search, with the decoder's steps compiled when `compile` is set.
    ///
    /// # Errors
    ///
    /// Returns an error if `encoded` is not `[1, S, D]`, if `prefix` is empty, if greedy generation finds every token banned, or if a decoder step fails.
    pub fn generate(
        &self,
        encoded: &Tensor,
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
        let &[sources, _] = encoded.shape().dimensions() else {
            return Err(SingeError::invalid_argument("an encoder's states").into());
        };
        let held = (sources + SOURCE_BUCKET - 1) / SOURCE_BUCKET * SOURCE_BUCKET;
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
                    DecoderOptions::new(rows, capacity).with_prompt_chunks(PromptChunks::NONE);
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
                    self.beam_search(decoder, prefix, max_length)
                } else {
                    self.greedy(decoder, prefix, max_length)
                }
            },
        )
    }

    fn greedy(
        &self,
        decoder: &mut CachedDecoder<Seq2SeqDecoder>,
        prefix: &[i64],
        max_length: usize,
    ) -> Result<Vec<i64>> {
        let config = &self.generation;
        let mut tokens = prefix.to_vec();
        let mut generated = Vec::new();
        let mut scores = read_prompt(decoder, prefix)?;
        loop {
            let row = scores
                .first()
                .ok_or_else(|| SingeError::missing_output("decoder scores"))?;
            let (forced, banned) = config.constraints(&tokens, max_length);
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
            let candidates = config.continuations(&running, &scores, max_length);
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
fn step(decoder: &mut CachedDecoder<Seq2SeqDecoder>, tokens: &[&[i64]]) -> Result<Vec<Vec<f32>>> {
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
    if running
        .iter()
        .enumerate()
        .all(|(row, beam)| beam.parent == row)
    {
        return Ok(());
    }

    let parents = running
        .iter()
        .map(|beam| i64::try_from(beam.parent).map_err(|_| SingeError::out_of_range("beams")))
        .collect::<SingeResult<Vec<_>>>()?;
    let count = i64::try_from(parents.len()).map_err(|_| SingeError::out_of_range("beams"))?;
    let parents = Tensor::from_vec(
        decoder.model().state().device()?,
        Shape::create([count])?,
        parents,
    )?;
    for cache in decoder.caches() {
        cache.reorder(&parents)?;
    }
    Ok(())
}
