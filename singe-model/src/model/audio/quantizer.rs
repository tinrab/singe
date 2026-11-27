//! What the residual vector quantizers of `EnCodec` and Mimi share: Euclidean codebooks, each choosing the entry nearest the residual the codebooks before it leave, and codes decoded as the sum of their entries.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{ModelState, StateKey},
};

/// Returns the index of the entry of `entries`, `[V, D]`, nearest each row of `residual`, `[N, D]`, as `[N, 1]`.
/// The distance is the reference's, `−(‖x‖² − 2 x · e + ‖e‖²)`, whose constant term can still flip a near tie.
pub(crate) fn nearest(residual: &Tensor, entries: &Tensor) -> SingeResult<Tensor> {
    let distance = residual
        .mul(residual)?
        .sum(&[1], true)?
        .sub(
            &residual
                .matmul(&entries.transpose(0, 1)?)?
                .mul_scalar(2.0)?,
        )?
        .add(&entries.mul(entries)?.sum(&[1], false)?.unsqueeze(0)?)?
        .neg()?;
    let (_, indices) = distance.topk(1, -1, true, false)?;
    Ok(indices)
}

/// Returns the sum of each codebook's entries at its codes, `[B, D, T]`, of codes, `[B, K, T]`, for the first `K` of `codebooks`.
pub(crate) fn summed_entries(
    codebooks: &[StateKey],
    codes: &Tensor,
    state: &ModelState,
) -> SingeResult<Tensor> {
    let &[batch, count, frames] = codes.shape().dimensions() else {
        return Err(SingeError::invalid_argument("codec codes, [B, K, T]"));
    };
    if usize::try_from(count).map_or(true, |count| count > codebooks.len()) {
        return Err(SingeError::invalid_argument(
            "codec codes for more codebooks than it holds",
        ));
    }

    let mut summed: Option<Tensor> = None;
    for (codebook, index) in codebooks.iter().zip(0..count) {
        let entries = state
            .tensor(codebook)?
            .index_select(0, &codes.select(1, index)?.reshape([batch * frames])?)?;
        summed = Some(match summed {
            Some(summed) => summed.add(&entries)?,
            None => entries,
        });
    }

    let summed =
        summed.ok_or_else(|| SingeError::invalid_argument("codec codes of no codebooks"))?;
    let width = summed.shape().dimensions()[1];
    summed.reshape([batch, frames, width])?.transpose(1, 2)
}
