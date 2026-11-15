use crate::{
    error::{Error, Result},
    generation::GenerationOptions,
    graph::capture,
    random::Generator,
    shape::Shape,
    tensor::Tensor,
};

// Chooses a token for each row of `logits`, `[S, V]`, as `options` asks: never a suppressed token, nor a first-suppressed one in a row whose `first`, `[S]` of `I64`, is one, then the most likely token or a draw from `generator`.
// Returns `[S]` of `I64`; a captured step chooses through this too, so a program chooses as a running batch does.
pub fn choose(
    logits: &Tensor,
    first: &Tensor,
    options: &GenerationOptions,
    generator: Option<&Generator>,
) -> Result<Tensor> {
    let rows = logits.dimension(0)?;
    let logits =
        if options.suppressed_tokens.is_empty() && options.first_suppressed_tokens.is_empty() {
            logits.clone()
        } else {
            let vocabulary = logits.shape().dimensions()[1];
            let width = usize::try_from(vocabulary)
                .map_err(|_| Error::out_of_range("generation vocabulary"))?;
            // What each suppression adds to the logits, `[2, V]`: the first row for later tokens, the second for a sequence's first.
            let mut table = vec![0.0_f32; 2 * width];
            let mut mask = |token: i64, rows: &[usize]| {
                let index = usize::try_from(token)
                    .ok()
                    .filter(|&index| index < width)
                    .ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "suppressed token {token} is outside a vocabulary of {width}"
                        ))
                    })?;
                for row in rows {
                    table[row * width + index] = f32::NEG_INFINITY;
                }
                Ok::<_, Error>(())
            };
            for &token in &options.suppressed_tokens {
                mask(token, &[0, 1])?;
            }

            for &token in &options.first_suppressed_tokens {
                mask(token, &[1])?;
            }

            let table = capture::literal(logits.device(), |device| {
                Tensor::from_vec(device, Shape::create([2, vocabulary])?, table)
            })?
            .cast(logits.data_type())?;
            logits.add(&table.index_select(0, first)?)?
        };
    options
        .sampling
        .as_ref()
        .map_or_else(
            || logits.argmax(-1),
            |sampling| logits.sample(sampling, generator),
        )?
        .reshape_with(&[rows])
}
