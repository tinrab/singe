use crate::{
    device::Device,
    error::{Error, Result},
    generation::{Decoder, GenerationOptions, SequenceOptions, choose, reading::TokenReader},
    random::Generator,
    shape::Shape,
    tensor::{SamplingOptions, Tensor},
};

// One token a running batch chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedToken {
    // Which sequence, by the order they joined.
    pub sequence: usize,
    pub token: i64,
    // Whether this is the sequence's last, so its row is free again.
    pub finished: bool,
}

struct RunningSequence {
    id: usize,
    // What its next step reads from the host: its prompt, or the token last chosen when no step read it on the device.
    pending: Vec<i64>,
    generated: usize,
    options: SequenceOptions,
}

pub struct RunningBatch<'a, D: Decoder + ?Sized> {
    decoder: &'a mut D,
    options: GenerationOptions,
    rows: Vec<Option<RunningSequence>>,
    joined: usize,
    unread: Option<UnreadStep>,
    reader: TokenReader,
}

struct UnreadStep {
    chosen: Tensor,
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
            reader: TokenReader::default(),
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

        if options.max_new_tokens == 0 {
            return Err(Error::invalid_argument(
                "a running batch generates at least one token per sequence",
            ));
        }

        let Some(index) = self.rows.iter().position(Option::is_none) else {
            return Ok(None);
        };
        prepare(self.decoder, index)?;
        let row = &mut self.rows[index];
        let id = self.joined;
        *row = Some(RunningSequence {
            id,
            pending: prompt.to_vec(),
            generated: 0,
            options,
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
        let reading = self.reader.start(&previous.chosen)?;
        let next = self.enqueue(Some(&previous), generator).ok().flatten();
        let chosen = self.reader.finish(reading)?;
        if chosen.len() != self.rows.len() {
            return Err(Error::missing_output("generated tokens"));
        }

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
            generated.push(GeneratedToken {
                sequence: running.id,
                token,
                finished,
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
        let first = self.first_tokens(previous, &choosing, logits.device())?;
        let chosen = self.choose_rows(&logits, &first, &choosing, generator)?;
        for (row, &portion) in self.rows.iter_mut().zip(&portions) {
            if let Some(row) = row {
                let read = portion.min(row.pending.len());
                row.pending.drain(..read);
            }
        }
        Ok(Some(UnreadStep { chosen, choosing }))
    }

    fn choose_rows(
        &self,
        logits: &Tensor,
        first: &Tensor,
        choosing: &[Option<usize>],
        generator: Option<&Generator>,
    ) -> Result<Tensor> {
        let samplings = self
            .rows
            .iter()
            .zip(choosing)
            .map(|(row, choosing)| {
                row.as_ref()
                    .filter(|_| choosing.is_some())
                    .map(|row| row.options.sampling)
            })
            .collect::<Vec<_>>();
        let mut kinds = samplings.iter().flatten();
        let shared = kinds.next().copied().unwrap_or(None);
        let options = |sampling: Option<SamplingOptions>| GenerationOptions {
            sampling,
            ..self.options.clone()
        };
        if kinds.all(|sampling| *sampling == shared) {
            return choose(logits, first, &options(shared), generator);
        }

        let greedy = choose(logits, first, &options(None), generator)?;
        let rows = samplings
            .iter()
            .enumerate()
            .map(|(index, sampling)| {
                let index =
                    i64::try_from(index).map_err(|_| Error::out_of_range("generation row"))?;
                match sampling {
                    Some(Some(sampling)) => choose(
                        &logits.narrow(0, index, 1)?,
                        &first.narrow(0, index, 1)?,
                        &options(Some(*sampling)),
                        generator,
                    ),
                    _ => greedy.narrow(0, index, 1),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::concat(&rows, 0)
    }

    fn first_tokens(
        &self,
        previous: Option<&UnreadStep>,
        choosing: &[Option<usize>],
        device: Device,
    ) -> Result<Tensor> {
        let first = self
            .rows
            .iter()
            .zip(choosing)
            .enumerate()
            .map(|(index, (row, choosing))| {
                i64::from(row.as_ref().is_some_and(|row| {
                    let chosen = row.generated
                        + usize::from(
                            previous.is_some_and(|previous| previous.chooses_for(index, row)),
                        );
                    choosing.is_some() && chosen == 0
                }))
            })
            .collect::<Vec<_>>();
        let rows =
            i64::try_from(first.len()).map_err(|_| Error::out_of_range("generation rows"))?;
        Tensor::from_vec(device, Shape::create([rows])?, first)
    }
}

const PREVIOUS_CHOICE: [i64; 1] = [-1];

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
                &PREVIOUS_CHOICE[..]
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
    use crate::generation::{generate, generate_batch};

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
    }
}
