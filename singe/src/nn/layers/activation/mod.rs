// Element-wise activations a configuration names.

use serde::{Deserialize, Serialize};

use crate::{
    backend::registry::{Operation, OperationAttributes},
    dispatch::call_one,
    error::Result,
    nn::{Layer, ModelState},
    tensor::Tensor,
};

/// A stateless elementwise activation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// Rectified linear activation, `max(x, 0)`.
    Relu,
    /// Gaussian error linear activation using the tanh approximation.
    Gelu,
    /// Gaussian error linear activation using the error function.
    GeluExact,
    /// Sigmoid-weighted linear activation, `x * sigmoid(x)`.
    Silu,
    /// Logistic activation, `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    // GELU as CLIP approximates it, `x * sigmoid(1.702 x)`.
    QuickGelu,
}

impl Layer for Activation {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let operation = match self {
            Self::Relu => Operation::Relu,
            Self::Gelu => Operation::Gelu,
            Self::GeluExact => Operation::GeluExact,
            Self::Silu => Operation::Silu,
            Self::Sigmoid => Operation::Sigmoid,
            Self::Tanh => Operation::Tanh,
            Self::QuickGelu => {
                let gate = call_one(
                    Operation::Sigmoid,
                    OperationAttributes::None,
                    &[&input.mul_scalar(1.702)?],
                )?;
                return input.mul(&gate);
            }
        };
        call_one(operation, OperationAttributes::None, &[input])
    }
}
