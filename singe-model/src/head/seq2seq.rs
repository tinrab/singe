// Encoder-decoder generation, as `transformers`' `generate` runs it greedily for T5, BART, TrOCR, and Donut: the input encoded once, then each token the decoder's best after the tokens before it, under the snapshot's generation config.
// The decoder keeps each block's keys and values between steps, so a step reads only its new token; beam search keeps them per beam.

use std::{collections::HashSet, fmt::Debug};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{Model, ModelState, TokenBatch},
};

use crate::{
    assembly::AttentionCache,
    error::{Error, Result},
};

// What an encoder-decoder reads: tokens, or images' pixels.
#[derive(Debug, Clone, Copy)]
pub enum Seq2SeqInput<'a> {
    Tokens(&'a TokenBatch),
    Pixels(&'a Tensor),
}

// What an encoder-decoder's layers give: the encoder's states for one input, `[1, S, D]`, the decoder's state before its first token against those states, and its scores for the token after `tokens`, `[V]`.
pub trait Seq2SeqLayers: Debug + Send + Sync {
    fn encode(&self, input: Seq2SeqInput<'_>, state: &ModelState) -> SingeResult<Tensor>;

    // Each block's cross-attention keys and values over `encoded`, with no token read yet.
    fn start(&self, encoded: &Tensor, state: &ModelState) -> SingeResult<DecoderCache>;

    // The decoder reads the tokens of `tokens` past those `cache` has read, adding them to it.
    fn next_scores(
        &self,
        cache: &mut DecoderCache,
        tokens: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor>;
}

// A decoder's state between steps.
#[derive(Debug, Clone)]
pub struct DecoderCache {
    // Each block's self-attention keys and values over the tokens read, and its cross-attention's over the encoder's states.
    pub blocks: Vec<(Option<AttentionCache>, AttentionCache)>,
    // The tokens read.
    pub read: usize,
}

// How a snapshot generates, from its `generation_config.json`: the token the decoder starts from, those that end a sequence, tokens forced first and last, the bounds of a sequence's length, counted with the start token, the size of n-grams that may not repeat, tokens never chosen, and the beams searched, one for greedy generation, with the power of a finished beam's length its score is divided by and whether search stops once every beam has finished.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Seq2SeqGeneration {
    pub decoder_start: Option<i64>,
    pub stop_tokens: Vec<i64>,
    pub forced_first: Option<i64>,
    pub forced_last: Option<i64>,
    pub max_length: usize,
    pub min_length: usize,
    pub no_repeat_ngram: usize,
    pub banned: Vec<i64>,
    pub beams: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

// A beam: its tokens and its summed log-probability.
#[derive(Debug, Clone)]
struct Beam {
    tokens: Vec<i64>,
    score: f32,
    // The decoder's state after all but the last token, which a running beam reads next, and a finished one drops.
    cache: Option<DecoderCache>,
}

// The log-softmax of `scores`.
fn log_softmax(scores: &[f32]) -> Vec<f32> {
    let most = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = scores
        .iter()
        .map(|score| (score - most).exp())
        .sum::<f32>()
        .ln();
    scores.iter().map(|score| score - most - total).collect()
}

// The indices of `values`' `count` greatest, greatest first, the earlier of equals first.
fn top(values: &[f32], count: usize) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    order.truncate(count);
    order
}

#[derive(Debug)]
pub struct Seq2SeqLm {
    model: Model<Box<dyn Seq2SeqLayers>>,
    generation: Seq2SeqGeneration,
}

// The tokens that would complete an n-gram already in `tokens`, as `NoRepeatNGramLogitsProcessor` bans them.
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
    // The tokens `generate`'s logits processors keep from being chosen after `tokens`, of a sequence at most `max_length` long: banned tokens, repeated n-grams, and stop tokens before the least length; or the one token forced first or last.
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

    // Log-probabilities of the token after `tokens` from the decoder's `scores`, with the constraints' tokens at negative infinity.
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
}

impl Seq2SeqLm {
    #[must_use]
    pub const fn new(model: Model<Box<dyn Seq2SeqLayers>>, generation: Seq2SeqGeneration) -> Self {
        Self { model, generation }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Box<dyn Seq2SeqLayers>> {
        &self.model
    }

    #[must_use]
    pub const fn generation(&self) -> &Seq2SeqGeneration {
        &self.generation
    }

    // The token a decoder starts from, which a pipeline that gives no prompt of its own needs.
    pub fn decoder_start(&self) -> Result<i64> {
        self.generation
            .decoder_start
            .ok_or_else(|| Error::unsupported("a generation config without a decoder start token"))
    }

    pub fn encode(&self, input: Seq2SeqInput<'_>) -> Result<Tensor> {
        Ok(self.model.layer().encode(input, self.model.state())?)
    }

    // Generates after `prefix`, the decoder's start token and any prompt after it, until a stop token, `max_new_tokens` new tokens when given, or the config's maximum length, greedily or by the config's beam search, and returns the new tokens.
    pub fn generate(
        &self,
        encoded: &Tensor,
        prefix: &[i64],
        max_new_tokens: Option<usize>,
    ) -> Result<Vec<i64>> {
        let max_length =
            max_new_tokens.map_or(self.generation.max_length, |new| prefix.len() + new);
        if self.generation.beams > 1 {
            return self.beam_search(encoded, prefix, max_length);
        }

        let config = &self.generation;
        let mut tokens = prefix.to_vec();
        let mut generated = Vec::new();
        let mut cache = self.start(encoded)?;
        while tokens.len() < max_length {
            let scores = self.scores(&mut cache, &tokens)?;
            let (forced, banned) = config.constraints(&tokens, max_length);
            let token = match forced {
                Some(token) => token,
                None => scores
                    .iter()
                    .enumerate()
                    .filter(|&(token, _)| {
                        i64::try_from(token).is_ok_and(|token| !banned.contains(&token))
                    })
                    .fold(
                        None,
                        |best: Option<(usize, f32)>, (token, &score)| match best {
                            Some((_, most)) if most >= score => best,
                            _ => Some((token, score)),
                        },
                    )
                    .and_then(|(token, _)| i64::try_from(token).ok())
                    .ok_or_else(|| Error::unsupported("generation with every token banned"))?,
            };
            tokens.push(token);
            generated.push(token);
            if config.stop_tokens.contains(&token) {
                break;
            }
        }
        Ok(generated)
    }

    fn start(&self, encoded: &Tensor) -> Result<DecoderCache> {
        Ok(self.model.layer().start(encoded, self.model.state())?)
    }

    fn scores(&self, cache: &mut DecoderCache, tokens: &[i64]) -> Result<Vec<f32>> {
        Ok(self
            .model
            .layer()
            .next_scores(cache, tokens, self.model.state())?
            .cast(DataType::F32)?
            .to_vec::<f32>()?)
    }

    // Beam search as `transformers`' `_beam_search` runs it for one input: each step, the best `2 · beams` continuations of the running beams, those that end joining the finished, scored by their summed log-probability over their length to the config's power, and the best `beams` that go on running, until no running beam could beat the worst finished one, as `early_stopping=False` judges it from the current length.
    #[expect(clippy::cast_precision_loss, reason = "sequence lengths are small")]
    fn beam_search(&self, encoded: &Tensor, prefix: &[i64], max_length: usize) -> Result<Vec<i64>> {
        const UNREACHABLE: f32 = -1.0e9;
        let config = &self.generation;
        let beams = config.beams;
        let keep = (1 + config.stop_tokens.len()).max(2) * beams;
        let prompt = prefix.len();
        let start = self.start(encoded)?;
        let mut running = (0..beams)
            .map(|index| Beam {
                tokens: prefix.to_vec(),
                score: if index == 0 { 0.0 } else { UNREACHABLE },
                cache: Some(start.clone()),
            })
            .collect::<Vec<_>>();
        let mut finished = vec![
            (
                Beam {
                    tokens: prefix.to_vec(),
                    score: UNREACHABLE,
                    cache: None,
                },
                false
            );
            beams
        ];
        let mut improvable = true;
        let mut length = prompt;
        loop {
            let mut accumulated = Vec::new();
            let mut vocabulary = 0;
            for beam in &mut running {
                let cache = beam
                    .cache
                    .as_mut()
                    .ok_or_else(|| Error::unsupported("a running beam without a decoder state"))?;
                let log_probabilities = config.log_probabilities(
                    &self.scores(cache, &beam.tokens)?,
                    &beam.tokens,
                    max_length,
                );
                vocabulary = log_probabilities.len();
                accumulated.extend(
                    log_probabilities
                        .into_iter()
                        .map(|value| value + beam.score),
                );
            }

            let candidates = top(&accumulated, keep)
                .into_iter()
                .map(|index| {
                    let mut tokens = running[index / vocabulary].tokens.clone();
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "vocabularies are far smaller than `i64`"
                    )]
                    tokens.push((index % vocabulary) as i64);
                    Beam {
                        tokens,
                        score: accumulated[index],
                        cache: running[index / vocabulary].cache.clone(),
                    }
                })
                .collect::<Vec<_>>();
            let ends = candidates
                .iter()
                .map(|beam| {
                    beam.tokens
                        .last()
                        .is_some_and(|token| config.stop_tokens.contains(token))
                        || beam.tokens.len() >= max_length
                })
                .collect::<Vec<_>>();

            let continuing = candidates
                .iter()
                .zip(&ends)
                .map(|(beam, &end)| beam.score + if end { UNREACHABLE } else { 0.0 })
                .collect::<Vec<_>>();
            let next = top(&continuing, beams)
                .into_iter()
                .map(|index| Beam {
                    tokens: candidates[index].tokens.clone(),
                    score: continuing[index],
                    cache: candidates[index].cache.clone(),
                })
                .collect::<Vec<_>>();

            // Only the best `beams` candidates may finish, and none once no running beam could improve on the finished.
            let all_finished = finished.iter().all(|(_, done)| *done);
            let normalizer = ((length + 1 - prompt) as f32).powf(config.length_penalty);
            let mut merged = finished.clone();
            for (index, (beam, &end)) in candidates.iter().zip(&ends).enumerate() {
                let just_finished = end && index < beams;
                let mut score = beam.score / normalizer;
                if all_finished && config.early_stopping {
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
                        cache: None,
                    },
                    just_finished,
                ));
            }

            let scores = merged
                .iter()
                .map(|(beam, _)| beam.score)
                .collect::<Vec<_>>();
            finished = top(&scores, beams)
                .into_iter()
                .map(|index| merged[index].clone())
                .collect();
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
        }
        Ok(finished[0].0.tokens[prompt..].to_vec())
    }
}
