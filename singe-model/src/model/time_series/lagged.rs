//! What the time series transformer, Informer, and Autoformer share: a series scaled by its mean absolute value over the context, each step read at a set of lags beside the series' static features and the step's time features, and three projections mapped to a Student-T distribution per step.

use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Embedding, LayerNorm, Linear, ModelState, StateKey},
};

use crate::{
    error::{Error, Result},
    schema::Schema,
};

/// A stack's input embedding: each step's features projected, its position from the stored table added, and normed.
#[derive(Debug, Clone)]
pub(crate) struct StepEmbedding {
    value: Linear,
    positions: StateKey,
    norm: LayerNorm,
}

impl StepEmbedding {
    pub(crate) fn create(
        schema: &mut Schema<'_>,
        features: i64,
        width: i64,
        positions: i64,
    ) -> SingeResult<Self> {
        Ok(Self {
            value: schema.linear("value", features, width, false)?,
            positions: schema.parameter("positions", [positions, width])?,
            norm: schema.layer_norm("embedding_norm", width, 1e-5)?,
        })
    }

    /// Embeds `input`, `[B, L, F]`, whose steps sit at positions from `first`.
    pub(crate) fn forward(
        &self,
        input: &Tensor,
        first: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let length = input.shape().dimensions()[1];
        input
            .apply(&self.value, state)?
            .add(
                &state
                    .tensor(&self.positions)?
                    .narrow(0, first, length)?
                    .unsqueeze(0)?,
            )?
            .apply(&self.norm, state)
    }
}

/// Creates the embeddings of the static categories, `categories.{index}`, one per cardinality.
pub(crate) fn category_embeddings(
    schema: &mut Schema<'_>,
    cardinalities: &[i64],
    dims: &[i64],
) -> SingeResult<Vec<Embedding>> {
    cardinalities
        .iter()
        .zip(dims)
        .enumerate()
        .map(|(index, (&cardinality, &dims))| {
            schema
                .parameter(&format!("categories.{index}.weight"), [cardinality, dims])
                .map(Embedding::new)
        })
        .collect()
}

/// Returns the static features, `[1, 1, S]`: each category's embedding, the static reals, `log1p |loc|` of the scaler's zero location, and the log of the scale, in `like`'s type.
pub(crate) fn statics(
    embeddings: &[Embedding],
    categories: &[i64],
    reals: &[f32],
    log_scale: f32,
    like: &Tensor,
    state: &ModelState,
) -> SingeResult<Tensor> {
    let device = like.device();
    let mut statics = Vec::new();
    for (embedding, &category) in embeddings.iter().zip(categories) {
        let index = Tensor::from_vec(device, Shape::create([1])?, vec![category])?;
        statics.push(index.apply(embedding, state)?);
    }

    let scalars = [reals, &[0.0, log_scale]].concat();
    let count =
        i64::try_from(scalars.len()).map_err(|_| SingeError::out_of_range("static features"))?;
    statics.push(
        Tensor::from_vec(device, Shape::create([1, count])?, scalars)?.cast(like.data_type())?,
    );
    Tensor::concat(&statics, 1)?.unsqueeze(0)
}

/// Maps three projections to a Student-T's degrees of freedom, location, and scale, as `StudentTOutput.domain_map` does: degrees of freedom above 2 and a positive scale, both through squareplus.
pub(crate) fn student_t([df, loc, scale]: [Tensor; 3]) -> SingeResult<[Tensor; 3]> {
    let squareplus = |x: Tensor| -> SingeResult<Tensor> {
        x.add(&x.mul(&x)?.add_scalar(4.0)?.sqrt()?)?.div_scalar(2.0)
    };
    Ok([
        squareplus(df)?.add_scalar(2.0)?,
        loc,
        squareplus(scale)?.clamp(f32::EPSILON, f32::MAX)?,
    ])
}

/// Returns `input`, `[B, C, L]`, with its last step before its first and its first after its last, as a circular padding of one pads a convolution's input.
pub(crate) fn circular_pad(input: &Tensor) -> SingeResult<Tensor> {
    let length = input.shape().dimensions()[2];
    Tensor::concat(
        &[
            input.narrow(2, length - 1, 1)?,
            input.clone(),
            input.narrow(2, 0, 1)?,
        ],
        2,
    )
}

/// Returns the scale of a series' `context`, as `TimeSeriesMeanScaler` and `AutoformerMeanScaler` give it with no location: its mean absolute value, at least `minimum`.
pub(crate) fn mean_scale(context: &[f32], minimum: f32) -> f32 {
    (context.iter().map(|value| value.abs()).sum::<f32>() / context.len() as f32).max(minimum)
}

/// Returns steps `first..first + count` of `scaled`, each read at every lag, `[count · lags]`.
pub(crate) fn at_lags(scaled: &[f32], lags: &[usize], first: usize, count: usize) -> Vec<f32> {
    (first..first + count)
        .flat_map(|step| lags.iter().map(move |&lag| scaled[step - lag]))
        .collect()
}

/// Returns `value` as a count, or an error naming `what` if it is negative.
pub(crate) fn size(value: i64, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid(format!("a negative {what}")))
}

/// Returns the last `count` of `values`, or an error naming `what` if there are fewer.
pub(crate) fn last<'a, T>(values: &'a [T], count: usize, what: &str) -> Result<&'a [T]> {
    values
        .len()
        .checked_sub(count)
        .map(|start| &values[start..])
        .ok_or_else(|| Error::invalid(format!("{what} shorter than {count} steps")))
}

/// Returns `rows` of time features one after another, or an error if a row has other than `width` values.
pub(crate) fn time_rows(rows: &[Vec<f32>], width: i64) -> Result<Vec<f32>> {
    if rows
        .iter()
        .any(|row| i64::try_from(row.len()).ok() != Some(width))
    {
        return Err(Error::invalid(format!(
            "time features of other than {width} values"
        )));
    }
    Ok(rows.iter().flatten().copied().collect())
}
