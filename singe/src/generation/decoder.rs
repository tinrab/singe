use crate::{
    error::{Error, Result},
    generation::RunningBatch,
    random::Generator,
    tensor::{SamplingOptions, Tensor},
};

// A decoder that keeps its own cache between steps.
pub trait Decoder {
    // Returns how many sequences each step takes.
    fn sequences(&self) -> usize;

    // Extends each sequence with its `tokens`, and returns the logits of each sequence's last token, `[S, V]`.
    //
    // An empty slice holds a finished sequence, which then costs no capacity, and whose row of logits means nothing.
    // A negative token stands for the one `previous`, `[S]` of `I64`, holds for its sequence, as the step before chose it on the device.
    // At least one sequence must have tokens, and a step that fails changes nothing.
    fn step(&mut self, tokens: &[&[i64]], previous: Option<&Tensor>) -> Result<Tensor>;

    // Forgets every sequence, so the next step starts at position zero.
    fn reset(&mut self);

    // Forgets one sequence and frees what it holds, so its row can start another while the others run.
    fn release(&mut self, sequence: usize) -> Result<()>;

    // The most extra tokens a step may read, past one token per row, or `None` when a step takes whatever it is given, and a whole prompt reads in one step.
    fn largest_chunk(&self) -> Option<usize> {
        None
    }

    // Summarizes what a compiled decoder prepared, if it did.
    fn report(&self) -> Result<Option<String>> {
        Ok(None)
    }
}

// What to generate and when to stop.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    // The most tokens to generate for each sequence.
    pub max_new_tokens: usize,
    // Tokens that end a sequence, and are returned as its last token.
    pub stop_tokens: Vec<i64>,
    // How to choose each token, or the most likely one for `None`.
    pub sampling: Option<SamplingOptions>,
    // Tokens never chosen, as Whisper's generation config lists in `suppress_tokens`.
    pub suppressed_tokens: Vec<i64>,
    // Tokens not chosen as a sequence's first, as Whisper's `begin_suppress_tokens`.
    pub first_suppressed_tokens: Vec<i64>,
}

// Generates tokens after `prompt`, calling `on_token` with each as it is chosen, and returns them.
pub fn generate(
    decoder: &mut (impl Decoder + ?Sized),
    prompt: &[i64],
    options: &GenerationOptions,
    generator: Option<&Generator>,
    mut on_token: impl FnMut(i64) -> Result<()>,
) -> Result<Vec<i64>> {
    let mut sequences = generate_batch(
        decoder,
        &[prompt],
        options,
        generator,
        |_, _, _| Ok(()),
        |_, token| on_token(token),
    )?;
    sequences
        .pop()
        .ok_or_else(|| Error::missing_output("generated sequence"))
}

// Generates tokens after each of `prompts`, which may differ in length, calling `on_token` with a sequence's index and each token as it is chosen, and returns each sequence's tokens.
//
// Prompts beyond the decoder's rows wait, and each takes the first row a finished sequence frees, before the next step.
// `prepare` runs with the decoder, the row a prompt takes, and the prompt's index before it joins, as an encoder-decoder writes the row's sources there.
pub fn generate_batch<D: Decoder + ?Sized>(
    decoder: &mut D,
    prompts: &[&[i64]],
    options: &GenerationOptions,
    generator: Option<&Generator>,
    mut prepare: impl FnMut(&D, usize, usize) -> Result<()>,
    mut on_token: impl FnMut(usize, i64) -> Result<()>,
) -> Result<Vec<Vec<i64>>> {
    if prompts.is_empty() || prompts.iter().any(|prompt| prompt.is_empty()) {
        return Err(Error::invalid_argument("generation prompt is empty"));
    }

    let mut tokens = vec![Vec::new(); prompts.len()];
    if options.max_new_tokens == 0 {
        decoder.reset();
        return Ok(tokens);
    }

    let mut batch = RunningBatch::create(decoder, options.clone())?;
    let mut next = 0;
    loop {
        while let Some(prompt) = prompts.get(next) {
            if batch
                .join_with(prompt, |decoder, row| prepare(decoder, row, next))?
                .is_none()
            {
                break;
            }

            next += 1;
        }

        if batch.is_empty() {
            break;
        }

        for generated in batch.advance(generator)? {
            tokens[generated.sequence].push(generated.token);
            on_token(generated.sequence, generated.token)?;
        }
    }
    Ok(tokens)
}
