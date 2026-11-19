// Affine projections, and flattening what a projection reads.

use crate::{
    backend::registry::{FlattenAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, ModelState, StateKey, joined_rows},
    tensor::Tensor,
};

/// Affine projection with checkpoint weights in `[out_features, in_features]` order.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: StateKey,
    bias: Option<StateKey>,
}

impl Linear {
    /// Stores affine weight and optional bias keys without resolving their tensors.
    ///
    /// Weights are interpreted in `[out_features, in_features]` order when the layer runs.
    #[must_use]
    pub fn new(weight: StateKey, bias: Option<StateKey>) -> Self {
        Self { weight, bias }
    }

    #[must_use]
    pub const fn weight(&self) -> &StateKey {
        &self.weight
    }

    #[must_use]
    pub const fn bias(&self) -> Option<&StateKey> {
        self.bias.as_ref()
    }

    pub fn forward_with(
        &self,
        input: &Tensor,
        weight: &Tensor,
        state: &ModelState,
    ) -> Result<Tensor> {
        let output = input.matmul(&weight.transpose(0, 1)?)?;
        match &self.bias {
            Some(bias) => output.add(state.tensor(bias)?),
            None => Ok(output),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        self.forward_with(input, state.tensor(&self.weight)?, state)
    }
}

// Runs `linears`, which read one `input`, as one matmul when their weights, and biases if they all have one, are adjacent rows of one tensor, as `ModelState::join_projection_rows` binds them, and returns each one's output as a view of the joined product.
// A capture records the linears apart, `None` then, as when the weights are not joined; compiling merges them again (`merge_sibling_projections`) after it has seen the whole graph.
pub(crate) fn joined_forward<const N: usize>(
    linears: &[&Linear; N],
    input: &Tensor,
    state: &ModelState,
) -> Result<Option<[Tensor; N]>> {
    if capture::is_active() {
        return Ok(None);
    }

    let weights = linears
        .iter()
        .map(|linear| state.tensor(&linear.weight))
        .collect::<Result<Vec<_>>>()?;
    let Some(weight) = joined_rows(&weights) else {
        return Ok(None);
    };
    let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
        None
    } else {
        let Some(biases) = linears
            .iter()
            .map(|linear| linear.bias.as_ref().map(|bias| state.tensor(bias)))
            .collect::<Option<Result<Vec<_>>>>()
            .transpose()?
        else {
            return Ok(None);
        };
        let Some(bias) = joined_rows(&biases) else {
            return Ok(None);
        };
        Some(bias)
    };

    let mut product = input.matmul(&weight.transpose(0, 1)?)?;
    if let Some(bias) = bias {
        product = product.add(&bias)?;
    }

    let mut start = 0;
    let mut outputs = Vec::with_capacity(N);
    for weight in weights {
        let &[rows, ..] = weight.shape().dimensions() else {
            return Err(Error::invalid_argument("joined projection weight"));
        };
        outputs.push(product.narrow(-1, start, rows)?);
        start += rows;
    }

    let outputs = <[Tensor; N]>::try_from(outputs)
        .map_err(|_| Error::missing_output("joined projections"))?;
    Ok(Some(outputs))
}

/// A stateless layer combining an inclusive consecutive range of tensor axes.
#[derive(Debug, Clone, Copy)]
pub struct Flatten {
    start_dim: i64,
    end_dim: i64,
}

impl Flatten {
    /// Stores inclusive start and end dimensions for flattening.
    ///
    /// Negative axes count from the end.
    /// Validation occurs when the layer runs.
    #[must_use]
    pub fn new(start_dim: i64, end_dim: i64) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        call_one(
            Operation::Flatten,
            OperationAttributes::Flatten(FlattenAttributes {
                start_dim: self.start_dim,
                end_dim: self.end_dim,
            }),
            &[input],
        )
    }
}
