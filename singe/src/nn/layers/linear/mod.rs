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
