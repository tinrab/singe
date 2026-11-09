// Affine projections, and flattening what a projection reads.

use crate::{
    backend::registry::{FlattenAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::Result,
    nn::{Layer, ModelState, StateKey},
    tensor::Tensor,
};

/// Affine projection with checkpoint weights in `[out_features, in_features]` order.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: StateKey,
    bias: Option<StateKey>,
    input_first: bool,
    outputs: Option<(i64, i64)>,
}

impl Linear {
    /// Stores affine weight and optional bias keys without resolving their tensors.
    ///
    /// Weights are interpreted in `[out_features, in_features]` order when the layer runs.
    #[must_use]
    pub fn new(weight: StateKey, bias: Option<StateKey>) -> Self {
        Self {
            weight,
            bias,
            input_first: false,
            outputs: None,
        }
    }

    // Reads a weight stored `[in_features, out_features]`, as GPT-2's `Conv1D` stores it.
    #[must_use]
    pub const fn with_input_first(mut self) -> Self {
        self.input_first = true;
        self
    }

    // Computes only `count` outputs from `first`, for a checkpoint that stores several projections in one weight.
    #[must_use]
    pub const fn with_outputs(mut self, first: i64, count: i64) -> Self {
        self.outputs = Some((first, count));
        self
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let weight = state.tensor(&self.weight)?;
        let mut weight = if self.input_first {
            weight.clone()
        } else {
            weight.transpose(0, 1)?
        };
        if let Some((first, count)) = self.outputs {
            weight = weight.narrow(1, first, count)?;
        }

        let output = input.matmul(&weight)?;
        let Some(bias) = &self.bias else {
            return Ok(output);
        };
        let bias = state.tensor(bias)?;
        match self.outputs {
            Some((first, count)) => output.add(&bias.narrow(0, first, count)?),
            None => output.add(bias),
        }
    }
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
