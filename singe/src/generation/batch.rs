use std::iter;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    generation::{
        Bias, Choice, Chosen, ChosenLogprobs, Decoder, GenerationOptions, PenaltyCounts,
        SequenceOptions, choose, reading::HostReader,
    },
    random::{self, Generator},
    shape::Shape,
    tensor::{DrawSource, SamplingOptions, Tensor},
};

// One token a running batch chose.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedToken {
    // Which sequence, by the order they joined.
    pub sequence: usize,
    pub token: i64,
    // Whether this is the sequence's last, so its row is free again.
    pub finished: bool,
    pub logprobs: Option<TokenLogprobs>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub alternatives: Vec<(i64, f32)>,
}

struct RunningSequence {
    id: usize,
    // What its next step reads from the host: its prompt, or the token last chosen when no step read it on the device.
    pending: Vec<i64>,
    generated: usize,
    options: SequenceOptions,
    draws: u64,
    uncounted_prompt: Vec<i64>,
}

#[derive(Clone)]
struct PenaltyState {
    generated: Tensor,
    prompt: Tensor,
}

pub struct RunningBatch<'a, D: Decoder + ?Sized> {
    decoder: &'a mut D,
    options: GenerationOptions,
    rows: Vec<Option<RunningSequence>>,
    joined: usize,
    unread: Option<UnreadStep>,
    penalties: Option<PenaltyState>,
    token_reader: HostReader,
    logprob_token_reader: HostReader,
    logprob_value_reader: HostReader,
}

struct UnreadStep {
    chosen: Tensor,
    logprobs: Option<ChosenLogprobs>,
    choosing: Vec<Option<usize>>,
}

impl<'a, D: Decoder + ?Sized> RunningBatch<'a, D> {
    // Starts an empty batch, forgetting whatever `decoder` held.
    pub fn create(decoder: &'a mut D, options: GenerationOptions) -> Result<Self> {
        if options.max_new_tokens == 0 {
            return Err(Error::invalid_argument(
                "a running batch generates at least one token per sequence",
            ));
        }

        decoder.reset();
        let rows = (0..decoder.sequences()).map(|_| None).collect();
        Ok(Self {
            decoder,
            options,
            rows,
            joined: 0,
            unread: None,
            penalties: None,
            token_reader: HostReader::default(),
            logprob_token_reader: HostReader::default(),
            logprob_value_reader: HostReader::default(),
        })
    }

    pub fn join(&mut self, prompt: &[i64]) -> Result<Option<usize>> {
        let options = self.options.sequence();
        self.join_with_options(prompt, options, |_, _| Ok(()))
    }

    // Starts a sequence as `join` does, after `prepare` has run with the decoder and the row it takes, as an encoder-decoder writes the row's sources there before its first step.
    pub fn join_with(
        &mut self,
        prompt: &[i64],
        prepare: impl FnOnce(&D, usize) -> Result<()>,
    ) -> Result<Option<usize>> {
        let options = self.options.sequence();
        self.join_with_options(prompt, options, prepare)
    }

    pub fn join_with_options(
        &mut self,
        prompt: &[i64],
        options: SequenceOptions,
        prepare: impl FnOnce(&D, usize) -> Result<()>,
    ) -> Result<Option<usize>> {
        if prompt.is_empty() {
            return Err(Error::invalid_argument("generation prompt is empty"));
        }

        options.validate()?;
        let Some(index) = self.rows.iter().position(Option::is_none) else {
            return Ok(None);
        };
        prepare(self.decoder, index)?;
        let row = &mut self.rows[index];
        let id = self.joined;
        let uncounted_prompt = if options.penalties.is_neutral() {
            Vec::new()
        } else {
            prompt.to_vec()
        };
        *row = Some(RunningSequence {
            id,
            pending: prompt.to_vec(),
            generated: 0,
            options,
            draws: 0,
            uncounted_prompt,
        });
        self.joined += 1;
        Ok(Some(id))
    }

    pub fn leave(&mut self, id: usize) -> Result<bool> {
        let Some(index) = self
            .rows
            .iter()
            .position(|row| row.as_ref().is_some_and(|row| row.id == id))
        else {
            return Ok(false);
        };
        self.rows[index] = None;
        self.decoder.release(index)?;
        Ok(true)
    }

    #[must_use]
    pub fn running(&self) -> usize {
        self.rows.iter().flatten().count()
    }

    // Returns whether no sequence is running.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(Option::is_none)
    }

    // Enqueues a step, and returns the tokens the step before it chose for each running sequence, in row order, which is none for a sequence still reading its prompt.
    // A step that fails, as one that overruns a shared cache does, changes nothing, so the caller can wait for a sequence to finish, or stop.
    pub fn advance(&mut self, generator: Option<&Generator>) -> Result<Vec<GeneratedToken>> {
        if self.is_empty() {
            self.unread = None;
            return Ok(Vec::new());
        }

        let previous = match self.unread.take() {
            Some(previous) => previous,
            None => match self.enqueue(None, generator)? {
                Some(previous) => previous,
                None => return Ok(Vec::new()),
            },
        };
        // The next step reads the tokens the previous one chose on the device, and runs while the host reads them; a sequence that stopped leaves its row of it unread.
        // A next step that fails changes nothing, and runs again after these tokens, from the host, which reports what fails then.
        let reading = self.token_reader.start(&previous.chosen)?;
        let logprob_reading = match &previous.logprobs {
            Some(logprobs) => Some((
                self.logprob_token_reader.start(&logprobs.tokens)?,
                self.logprob_value_reader.start(&logprobs.values)?,
            )),
            None => None,
        };
        let next = self.enqueue(Some(&previous), generator).ok().flatten();
        let chosen = self.token_reader.finish::<i64>(reading)?;
        if chosen.len() != self.rows.len() {
            return Err(Error::missing_output("generated tokens"));
        }

        let logprobs = match logprob_reading {
            Some((tokens, values)) => {
                let tokens = self.logprob_token_reader.finish::<i64>(tokens)?;
                let values = self.logprob_value_reader.finish::<f32>(values)?;
                Some(ReadLogprobs {
                    width: tokens.len() / chosen.len(),
                    tokens,
                    values,
                })
            }
            None => None,
        };

        let mut generated = Vec::new();
        for (index, (row, &token)) in self.rows.iter_mut().zip(&chosen).enumerate() {
            let Some(running) = row else {
                continue;
            };
            if !previous.chooses_for(index, running) {
                continue;
            }

            running.generated += 1;
            let finished = running.generated == running.options.max_new_tokens
                || running.options.stop_tokens.contains(&token);
            let logprobs = logprobs
                .as_ref()
                .zip(running.options.logprobs)
                .map(|(logprobs, alternatives)| logprobs.row(index, alternatives));
            generated.push(GeneratedToken {
                sequence: running.id,
                token,
                finished,
                logprobs,
            });
            if finished {
                *row = None;
                self.decoder.release(index)?;
            } else if !next
                .as_ref()
                .is_some_and(|next| next.chooses_for(index, running))
            {
                running.pending.push(token);
            }
        }

        self.unread = next;
        Ok(generated)
    }

    fn enqueue(
        &mut self,
        previous: Option<&UnreadStep>,
        generator: Option<&Generator>,
    ) -> Result<Option<UnreadStep>> {
        let reads = row_reads(&self.rows, previous);
        if reads.iter().all(|tokens| tokens.is_empty()) {
            return Ok(None);
        }

        let lengths = reads.iter().map(|tokens| tokens.len()).collect::<Vec<_>>();
        let portions = step_portions(&lengths, self.decoder.largest_chunk());
        let tokens = reads
            .iter()
            .zip(&portions)
            .map(|(tokens, &portion)| &tokens[..portion])
            .collect::<Vec<_>>();
        let logits = self
            .decoder
            .step(&tokens, previous.map(|previous| &previous.chosen))?;
        // A row chooses for its sequence once it has read the rest of what it was given.
        let choosing = self
            .rows
            .iter()
            .zip(&portions)
            .zip(&lengths)
            .map(|((row, &portion), &length)| {
                row.as_ref()
                    .filter(|_| portion > 0 && portion == length)
                    .map(|row| row.id)
            })
            .collect::<Vec<_>>();
        // The index of the token each row chooses: the next after those read, and after the one `previous` chose for it.
        let positions = self
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.as_ref().map_or(0, |row| {
                    row.generated
                        + usize::from(
                            previous.is_some_and(|previous| previous.chooses_for(index, row)),
                        )
                })
            })
            .collect::<Vec<_>>();
        let chosen = self.choose_rows(&logits, &choosing, &positions, generator)?;
        for (row, &portion) in self.rows.iter_mut().zip(&portions) {
            if let Some(row) = row {
                let read = portion.min(row.pending.len());
                row.pending.drain(..read);
            }
        }
        Ok(Some(UnreadStep {
            chosen: chosen.tokens,
            logprobs: chosen.logprobs,
            choosing,
        }))
    }

    fn choose_rows(
        &mut self,
        logits: &Tensor,
        choosing: &[Option<usize>],
        positions: &[usize],
        generator: Option<&Generator>,
    ) -> Result<Chosen> {
        let device = logits.device();
        let rows = self.row_count()?;
        let first = choosing
            .iter()
            .zip(positions)
            .map(|(choosing, &position)| i64::from(choosing.is_some() && position == 0))
            .collect::<Vec<_>>();
        let first = Tensor::from_vec(device, Shape::create([rows])?, first)?;
        let choosers = self
            .rows
            .iter()
            .zip(choosing)
            .map(|(row, choosing)| row.as_ref().filter(|_| choosing.is_some()))
            .collect::<Vec<_>>();
        let sampling = choosers
            .iter()
            .map(|row| row.and_then(|row| row.options.sampling))
            .collect::<Vec<_>>();
        let logprobs = choosers
            .iter()
            .flatten()
            .filter_map(|row| row.options.logprobs)
            .max()
            .map(|alternatives| i64::try_from(alternatives).unwrap_or(i64::MAX));
        let penalized = choosers
            .iter()
            .map(|row| row.is_some_and(|row| !row.options.penalties.is_neutral()))
            .collect::<Vec<_>>();
        let streams = if sampling.iter().all(Option::is_none) {
            None
        } else {
            Some(self.streams(&sampling, device, generator)?)
        };
        let bias = self.bias(choosing, positions, logits)?;
        let penalty_settings = self.prepare_penalties(&penalized, logits)?;

        let draws = streams
            .as_ref()
            .map_or(DrawSource::Generator(None), DrawSource::Streams);
        let mut choice = Choice::new(&first, &self.options, &sampling, draws);
        choice.bias = bias
            .as_ref()
            .map(|(tokens, values)| Bias { tokens, values });
        choice.penalties =
            penalty_settings
                .as_ref()
                .zip(self.penalties.as_ref())
                .map(|(settings, state)| PenaltyCounts {
                    settings,
                    generated: &state.generated,
                    prompt: &state.prompt,
                });
        choice.logprobs = logprobs;
        let chosen = choose(logits, &choice)?;

        for (row, sampling) in self.rows.iter_mut().zip(&sampling) {
            if let Some(row) = row.as_mut().filter(|_| sampling.is_some()) {
                row.draws += 1;
            }
        }

        if penalty_settings.is_some() {
            self.count_chosen(&penalized, &chosen.tokens)?;
        }
        Ok(chosen)
    }

    fn row_count(&self) -> Result<i64> {
        i64::try_from(self.rows.len()).map_err(|_| Error::out_of_range("generation rows"))
    }

    fn streams(
        &mut self,
        sampling: &[Option<SamplingOptions>],
        device: Device,
        generator: Option<&Generator>,
    ) -> Result<Tensor> {
        let generator = match generator {
            Some(generator) => generator.clone(),
            None => Generator::default_for(device)?,
        };
        let mut states = Vec::with_capacity(2 * self.rows.len());
        for (row, sampling) in self.rows.iter_mut().zip(sampling) {
            let stream = match row {
                Some(row) if sampling.is_some() => {
                    let seed = match row.options.seed {
                        Some(seed) => seed,
                        None => fresh_seed(&generator)?,
                    };
                    row.options.seed = Some(seed);
                    [seed, row.draws]
                }
                _ => [0, 0],
            };
            states.extend(stream.map(u64::cast_signed));
        }
        Tensor::from_vec(device, Shape::create([self.row_count()?, 2])?, states)
    }

    fn bias(
        &self,
        choosing: &[Option<usize>],
        positions: &[usize],
        logits: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let vocabulary = vocabulary(logits);
        let entries = self
            .rows
            .iter()
            .zip(choosing)
            .zip(positions)
            .map(|((row, choosing), &position)| {
                let Some(row) = row.as_ref().filter(|_| choosing.is_some()) else {
                    return Vec::new();
                };
                let mut entries = row.options.logit_bias.clone();
                if position < row.options.min_tokens {
                    entries.extend(
                        row.options
                            .stop_tokens
                            .iter()
                            .map(|&token| (token, f32::NEG_INFINITY)),
                    );
                }
                entries
            })
            .collect::<Vec<_>>();
        let width = entries.iter().map(Vec::len).max().unwrap_or(0);
        if width == 0 {
            return Ok(None);
        }

        if let Some((token, _)) = entries
            .iter()
            .flatten()
            .find(|(token, _)| !(0..vocabulary).contains(token))
        {
            return Err(Error::invalid_argument(format!(
                "biased token {token} is outside a vocabulary of {vocabulary}"
            )));
        }

        let mut tokens = Vec::with_capacity(entries.len() * width);
        let mut values = Vec::with_capacity(entries.len() * width);
        for row in &entries {
            tokens.extend(row.iter().map(|&(token, _)| token));
            tokens.extend(iter::repeat_n(0, width - row.len()));
            values.extend(row.iter().map(|&(_, value)| value));
            values.extend(iter::repeat_n(0.0_f32, width - row.len()));
        }

        let width = i64::try_from(width).map_err(|_| Error::out_of_range("logit bias"))?;
        let shape = Shape::create([self.row_count()?, width])?;
        Ok(Some((
            Tensor::from_vec(logits.device(), shape.clone(), tokens)?,
            Tensor::from_vec(logits.device(), shape, values)?,
        )))
    }

    fn prepare_penalties(&mut self, penalized: &[bool], logits: &Tensor) -> Result<Option<Tensor>> {
        if self
            .rows
            .iter()
            .flatten()
            .all(|row| row.options.penalties.is_neutral())
        {
            self.penalties = None;
            return Ok(None);
        }

        if !penalized.contains(&true) {
            return Ok(None);
        }

        let device = logits.device();
        let rows = self.row_count()?;
        let vocabulary = vocabulary(logits);
        let state = if let Some(state) = &self.penalties {
            state.clone()
        } else {
            let shape = Shape::create([rows, vocabulary])?;
            PenaltyState {
                generated: Tensor::zeroed_on(device, DataType::F32, shape.clone())?,
                prompt: Tensor::zeroed_on(device, DataType::F32, shape)?,
            }
        };
        let state = self.count_prompts(state, penalized, vocabulary)?;
        let settings = self
            .rows
            .iter()
            .flat_map(|row| {
                let penalties = row.as_ref().map_or_default(|row| row.options.penalties);
                [
                    penalties.repetition,
                    penalties.frequency,
                    penalties.presence,
                ]
            })
            .collect::<Vec<_>>();
        let settings = Tensor::from_vec(device, Shape::create([rows, 3])?, settings)?;
        self.penalties = Some(state);
        for (row, &penalized) in self.rows.iter_mut().zip(penalized) {
            if let Some(row) = row.as_mut().filter(|_| penalized) {
                row.uncounted_prompt.clear();
            }
        }
        Ok(Some(settings))
    }

    fn count_prompts(
        &self,
        mut state: PenaltyState,
        penalized: &[bool],
        vocabulary: i64,
    ) -> Result<PenaltyState> {
        let prompts = self
            .rows
            .iter()
            .zip(penalized)
            .map(|(row, &penalized)| match row {
                Some(row) if penalized => &row.uncounted_prompt[..],
                _ => &[][..],
            })
            .collect::<Vec<_>>();
        let width = prompts.iter().map(|prompt| prompt.len()).max().unwrap_or(0);
        if width == 0 {
            return Ok(state);
        }

        if let Some(token) = prompts
            .iter()
            .copied()
            .flatten()
            .find(|token| !(0..vocabulary).contains(*token))
        {
            return Err(Error::invalid_argument(format!(
                "prompt token {token} is outside a vocabulary of {vocabulary}"
            )));
        }

        let keep = prompts
            .iter()
            .map(|prompt| if prompt.is_empty() { 1.0_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let mut tokens = Vec::with_capacity(prompts.len() * width);
        let mut ones = Vec::with_capacity(prompts.len() * width);
        for prompt in &prompts {
            tokens.extend_from_slice(prompt);
            tokens.extend(iter::repeat_n(0, width - prompt.len()));
            ones.extend(iter::repeat_n(1.0_f32, prompt.len()));
            ones.extend(iter::repeat_n(0.0_f32, width - prompt.len()));
        }

        let device = state.prompt.device();
        let rows = self.row_count()?;
        let width = i64::try_from(width).map_err(|_| Error::out_of_range("prompt"))?;
        let keep = Tensor::from_vec(device, Shape::create([rows, 1])?, keep)?;
        let tokens = Tensor::from_vec(device, Shape::create([rows, width])?, tokens)?;
        let ones = Tensor::from_vec(device, Shape::create([rows, width])?, ones)?;
        state.generated = state.generated.mul(&keep)?;
        state.prompt = state.prompt.mul(&keep)?.scatter_add(1, &tokens, &ones)?;
        Ok(state)
    }

    fn count_chosen(&mut self, penalized: &[bool], tokens: &Tensor) -> Result<()> {
        let rows = self.row_count()?;
        let Some(state) = self.penalties.as_mut() else {
            return Ok(());
        };

        let increments = penalized
            .iter()
            .map(|&penalized| if penalized { 1.0_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let increments = Tensor::from_vec(tokens.device(), Shape::create([rows, 1])?, increments)?;
        state.generated = state
            .generated
            .scatter_add(1, &tokens.unsqueeze(1)?, &increments)?;
        Ok(())
    }
}

struct ReadLogprobs {
    tokens: Vec<i64>,
    values: Vec<f32>,
    width: usize,
}

impl ReadLogprobs {
    fn row(&self, index: usize, alternatives: usize) -> TokenLogprobs {
        let row = index * self.width..(index + 1) * self.width;
        let (tokens, values) = (&self.tokens[row.clone()], &self.values[row]);
        TokenLogprobs {
            logprob: values[0],
            alternatives: iter::zip(&tokens[1..], &values[1..])
                .map(|(&token, &value)| (token, value))
                .take(alternatives)
                .collect(),
        }
    }
}

fn vocabulary(logits: &Tensor) -> i64 {
    logits.shape().dimensions().last().copied().unwrap_or(0)
}

const PREVIOUS_CHOICE: &[i64] = &[-1];

fn fresh_seed(generator: &Generator) -> Result<u64> {
    let state = generator.reserve(1)?;
    let [low, high, ..] = random::philox(state.seed, state.offset);
    Ok(u64::from(low) | (u64::from(high) << 32))
}

fn row_reads<'a>(
    rows: &'a [Option<RunningSequence>],
    previous: Option<&UnreadStep>,
) -> Vec<&'a [i64]> {
    // A row reads the rest of its prompt, or, if its sequence may take another token, the one `previous` chooses for it, which the step reads on the device as `DecodeStep::follow` reads a negative token.
    rows.iter()
        .enumerate()
        .map(|(index, row)| match row {
            Some(row) if !row.pending.is_empty() => &row.pending[..],
            Some(row)
                if previous.is_some_and(|previous| previous.chooses_for(index, row))
                    && row.generated + 1 < row.options.max_new_tokens =>
            {
                PREVIOUS_CHOICE
            }
            _ => &[][..],
        })
        .collect()
}

impl UnreadStep {
    fn chooses_for(&self, row: usize, sequence: &RunningSequence) -> bool {
        self.choosing[row] == Some(sequence.id)
    }
}

// How many of its `pending` tokens each row reads in a step, so the step fits what the decoder compiled: a token per running row, and up to `largest` more, given to the first rows that have them.
fn step_portions(pending: &[usize], largest: Option<usize>) -> Vec<usize> {
    let Some(largest) = largest else {
        return pending.to_vec();
    };

    let spare: usize = pending.iter().map(|tokens| tokens.saturating_sub(1)).sum();
    let mut extra = spare.min(largest);
    pending
        .iter()
        .map(|&tokens| {
            let taken = tokens.saturating_sub(1).min(extra);
            extra -= taken;
            tokens.min(1) + taken
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{PenaltySet, generate, generate_batch};

    // Answers each step with logits favouring a token one above each sequence's last.
    struct Counting {
        sequences: usize,
        steps: Vec<Vec<Vec<i64>>>,
        resets: usize,
        released: Vec<usize>,
        largest: Option<usize>,
    }

    impl Counting {
        fn new(sequences: usize) -> Self {
            Self {
                sequences,
                steps: Vec::new(),
                resets: 0,
                released: Vec::new(),
                largest: None,
            }
        }
    }

    impl Decoder for Counting {
        fn sequences(&self) -> usize {
            self.sequences
        }

        fn step(&mut self, tokens: &[&[i64]], previous: Option<&Tensor>) -> Result<Tensor> {
            let previous = previous.map(|previous| previous.to_vec::<i64>().unwrap());
            let tokens = tokens
                .iter()
                .enumerate()
                .map(|(row, tokens)| {
                    tokens
                        .iter()
                        .map(|&token| {
                            if token < 0 {
                                previous.as_ref().unwrap()[row]
                            } else {
                                token
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            self.steps.push(tokens.clone());
            let mut logits = vec![0.0_f32; tokens.len() * 8];
            for (row, tokens) in tokens.iter().enumerate() {
                let best = tokens.last().copied().unwrap_or_default() + 1;
                logits[row * 8 + usize::try_from(best).unwrap() % 8] = 1.0;
            }
            Tensor::from_vec(
                Device::Cpu,
                Shape::create([i64::try_from(tokens.len()).unwrap(), 8]).unwrap(),
                logits,
            )
        }

        fn largest_chunk(&self) -> Option<usize> {
            self.largest
        }

        fn reset(&mut self) {
            self.resets += 1;
        }

        fn release(&mut self, sequence: usize) -> Result<()> {
            self.released.push(sequence);
            Ok(())
        }
    }

    fn run(prompt: &[i64], max_new_tokens: usize, stop_tokens: Vec<i64>) -> (Vec<i64>, Counting) {
        let mut decoder = Counting::new(1);
        let options = GenerationOptions {
            max_new_tokens,
            stop_tokens,
            ..GenerationOptions::default()
        };
        let mut seen = Vec::new();
        let tokens = generate(&mut decoder, prompt, &options, None, |token| {
            seen.push(token);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, tokens);
        (tokens, decoder)
    }

    // Whatever a prompt's length, no step reads more than the largest chunk past a token per row, and the next token follows the prompt's last.
    #[test]
    fn test_a_prompt_of_any_length_reads_in_steps_up_to_the_largest_chunk() {
        for length in 1..=20 {
            let prompt = (0..length).collect::<Vec<i64>>();
            let mut decoder = Counting::new(2);
            decoder.largest = Some(5);
            let options = GenerationOptions {
                max_new_tokens: 2,
                ..GenerationOptions::default()
            };
            let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
            batch.join(&prompt).unwrap();
            let mut generated = Vec::new();
            while !batch.is_empty() {
                generated.extend(batch.advance(None).unwrap());
            }
            assert_eq!(generated.len(), 2);
            assert_eq!(generated[0].token, length % 8);
            let mut read = Vec::new();
            for step in &decoder.steps {
                let total = step.iter().map(|tokens| tokens.len().max(1)).sum::<usize>();
                assert!(total <= 7, "{step:?}");
                read.extend_from_slice(&step[0]);
            }
            assert_eq!(read[..prompt.len()], prompt);
            // A prompt reads in as few steps as the largest chunk allows.
            assert_eq!(
                decoder.steps.len(),
                usize::try_from(length).unwrap().div_ceil(6) + 1
            );
        }
    }

    #[test]
    fn test_generation_runs_the_prompt_then_one_token_at_a_time() {
        let (tokens, decoder) = run(&[0, 1], 3, Vec::new());
        assert_eq!(tokens, [2, 3, 4]);
        assert_eq!(
            decoder.steps,
            [vec![vec![0, 1]], vec![vec![2]], vec![vec![3]]]
        );
        assert_eq!(decoder.resets, 1);
    }

    // A suppressed token is never chosen, and a first-suppressed one only after the first token, as Whisper's generation config asks.
    #[test]
    fn test_suppressed_tokens_are_not_chosen() {
        let mut decoder = Counting::new(1);
        let options = GenerationOptions {
            max_new_tokens: 3,
            suppressed_tokens: vec![2],
            first_suppressed_tokens: vec![1],
            ..GenerationOptions::default()
        };
        let tokens = generate(&mut decoder, &[0], &options, None, |_| Ok(())).unwrap();
        // The favoured token is one above the last; with it suppressed, the first token left wins.
        assert_eq!(tokens, [0, 1, 0]);
        let options = GenerationOptions {
            max_new_tokens: 1,
            suppressed_tokens: vec![8],
            ..GenerationOptions::default()
        };
        let error = generate(&mut decoder, &[0], &options, None, |_| Ok(())).unwrap_err();
        assert!(
            error.to_string().contains("outside a vocabulary"),
            "{error}"
        );
    }

    #[test]
    fn test_generation_returns_the_stop_token_and_drops_the_step_after() {
        let (tokens, decoder) = run(&[0], 5, vec![2]);
        assert_eq!(tokens, [1, 2]);
        // The step after the stop token runs before the host reads it, and its token is dropped.
        assert_eq!(decoder.steps, [vec![vec![0]], vec![vec![1]], vec![vec![2]]]);
    }

    // `usize::MAX` means until a stop token, so it must not reserve anything up front.
    #[test]
    fn test_generation_reserves_nothing_for_its_limit() {
        let (tokens, _) = run(&[0], usize::MAX, vec![2]);
        assert_eq!(tokens, [1, 2]);
    }

    #[test]
    fn test_generating_nothing_runs_nothing() {
        let (tokens, decoder) = run(&[0], 0, Vec::new());
        assert!(tokens.is_empty());
        assert!(decoder.steps.is_empty());
    }

    #[test]
    fn test_generation_refuses_an_empty_prompt() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 1,
            ..GenerationOptions::default()
        };
        assert!(
            generate_batch(
                &mut decoder,
                &[&[1], &[]],
                &options,
                None,
                |_, _, _| Ok(()),
                |_, _| Ok(()),
            )
            .is_err()
        );
    }

    // A finished sequence holds for the rest of the batch's steps and keeps nothing more.
    #[test]
    fn test_batches_finish_each_sequence_on_its_own_and_keep_stepping() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 3,
            stop_tokens: vec![2],
            sampling: None,
            ..GenerationOptions::default()
        };
        let mut seen = Vec::new();
        let tokens = generate_batch(
            &mut decoder,
            &[&[1], &[3, 4]],
            &options,
            None,
            |_, _, _| Ok(()),
            |sequence, token| {
                seen.push((sequence, token));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(tokens, [vec![2], vec![5, 6, 7]]);
        assert_eq!(seen[0], (0, 2));
        assert_eq!(
            seen.iter().filter(|(sequence, _)| *sequence == 0).count(),
            1
        );
        assert_eq!(decoder.steps[1], [vec![2], vec![5]]);
        assert_eq!(decoder.steps[2], [vec![], vec![6]]);
    }

    #[test]
    fn test_a_sequence_joins_the_row_another_left_while_the_rest_run() {
        let mut decoder = Counting::new(2);
        let options = GenerationOptions {
            max_new_tokens: 2,
            ..GenerationOptions::default()
        };
        let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
        assert_eq!(batch.join(&[0]).unwrap(), Some(0));
        batch.advance(None).unwrap();
        assert_eq!(batch.join(&[4, 5]).unwrap(), Some(1));
        assert_eq!(batch.join(&[6]).unwrap(), None);
        // The first takes its last token, and leaves its row to a third.
        let tokens = batch.advance(None).unwrap();
        assert_eq!(tokens[0].sequence, 0);
        assert!(tokens[0].finished);
        assert_eq!(batch.join(&[6]).unwrap(), Some(2));
        let tokens = batch.advance(None).unwrap();
        assert_eq!((tokens[0].sequence, tokens[0].token), (1, 6));
        let tokens = batch.advance(None).unwrap();
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.sequence)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert!(tokens[1].finished);
        let tokens = batch.advance(None).unwrap();
        assert_eq!(tokens[0].sequence, 2);
        assert!(batch.is_empty());
        assert!(batch.advance(None).unwrap().is_empty());
        assert_eq!(decoder.released, [0, 1, 0]);
        // Each step was enqueued before the host read the one before it, so a sequence that joins starts a step later than it would have otherwise.
        assert_eq!(
            decoder.steps,
            [
                vec![vec![0], vec![]],
                vec![vec![1], vec![]],
                vec![vec![], vec![4, 5]],
                vec![vec![6], vec![6]],
                vec![vec![7], vec![]],
            ]
        );

        // A seeded sequence draws the same tokens alone as beside sequences that choose otherwise.
        let seeded = SequenceOptions {
            max_new_tokens: 6,
            sampling: Some(SamplingOptions::new()),
            seed: Some(5),
            ..SequenceOptions::default()
        };
        let generator = Generator::new(1);
        let draw = |neighbours: &[SequenceOptions]| {
            let mut decoder = Counting::new(3);
            let options = GenerationOptions {
                max_new_tokens: 1,
                ..GenerationOptions::default()
            };
            let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
            for neighbour in neighbours {
                batch
                    .join_with_options(&[3], neighbour.clone(), |_, _| Ok(()))
                    .unwrap();
            }
            let id = batch
                .join_with_options(&[0], seeded.clone(), |_, _| Ok(()))
                .unwrap()
                .unwrap();
            let mut tokens = Vec::new();
            while !batch.is_empty() {
                let generated = batch.advance(Some(&generator)).unwrap();
                tokens.extend(
                    generated
                        .iter()
                        .filter(|token| token.sequence == id)
                        .map(|token| token.token),
                );
            }
            tokens
        };
        let neighbours = [
            SequenceOptions {
                max_new_tokens: 4,
                ..SequenceOptions::default()
            },
            SequenceOptions {
                max_new_tokens: 8,
                sampling: Some(
                    SamplingOptions::new()
                        .with_temperature(0.5)
                        .with_top_k(Some(2)),
                ),
                ..SequenceOptions::default()
            },
        ];
        let alone = draw(&[]);
        assert_eq!(alone.len(), 6);
        assert_eq!(draw(&neighbours), alone);

        // Bias, penalties and `min_tokens` change only their own sequence's tokens, and log probabilities come from the logits as the decoder gave them.
        let limited = |max_new_tokens| SequenceOptions {
            max_new_tokens,
            ..SequenceOptions::default()
        };
        // Token 6 beats the decoder's choice until two of it cost more than the bias.
        let frequent = SequenceOptions {
            logit_bias: vec![(6, 2.0)],
            penalties: PenaltySet {
                frequency: 0.6,
                ..PenaltySet::default()
            },
            logprobs: Some(2),
            ..limited(4)
        };
        // The decoder's choice, 4, is in the prompt, so the penalty halves it below token 6's bias.
        let repeated = SequenceOptions {
            logit_bias: vec![(6, 0.8)],
            penalties: PenaltySet {
                repetition: 2.0,
                ..PenaltySet::default()
            },
            ..limited(1)
        };
        let held = SequenceOptions {
            stop_tokens: vec![4],
            min_tokens: 1,
            ..limited(9)
        };
        let mut decoder = Counting::new(3);
        let options = GenerationOptions {
            max_new_tokens: 1,
            ..GenerationOptions::default()
        };
        let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
        let ids = [
            (&[0][..], frequent),
            (&[4, 3][..], repeated),
            (&[3][..], held),
        ]
        .into_iter()
        .map(|(prompt, options)| {
            batch
                .join_with_options(prompt, options, |_, _| Ok(()))
                .unwrap()
                .unwrap()
        })
        .collect::<Vec<_>>();
        let mut generated = vec![Vec::new(); ids.len()];
        let mut logprobs = Vec::new();
        while !batch.is_empty() {
            for token in batch.advance(None).unwrap() {
                let index = ids.iter().position(|&id| id == token.sequence).unwrap();
                generated[index].push(token.token);
                logprobs.extend(token.logprobs);
            }
        }
        assert_eq!(generated[0], [6, 6, 7, 0]);
        assert_eq!(generated[1], [6]);
        assert_ne!(generated[2][0], 4);
        assert_eq!(generated[2].last(), Some(&4));
        // Every row of the decoder's logits holds one 1 and seven 0s.
        let unlikely = -(1.0_f32.exp() + 7.0).ln();
        assert_eq!(logprobs.len(), 4);
        assert!((logprobs[0].logprob - unlikely).abs() < 1e-5);
        assert_eq!(logprobs[0].alternatives.len(), 2);
        assert_eq!(logprobs[0].alternatives[0].0, 1);
        assert!((logprobs[0].alternatives[0].1 - (1.0 + unlikely)).abs() < 1e-5);
    }
}
