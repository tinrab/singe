// Element-wise activations a configuration names.

use std::result::Result as StdResult;

use serde::{Deserialize, Deserializer, de::Error as DeError};

use crate::{
    backend::registry::{Operation, OperationAttributes},
    dispatch::call_one,
    error::Result,
    nn::{Layer, ModelState},
    tensor::Tensor,
};

/// A stateless elementwise activation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

// Deserializes from the name a Hugging Face config uses.
impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::from_config_name(&name)
            .ok_or_else(|| DeError::custom(format!("unknown activation `{name}`")))
    }
}

impl Activation {
    // Returns the activation a Hugging Face config names, as `transformers` maps these names.
    #[must_use]
    pub fn from_config_name(name: &str) -> Option<Self> {
        match name {
            "relu" => Some(Self::Relu),
            "gelu" => Some(Self::GeluExact),
            "gelu_new" | "gelu_pytorch_tanh" => Some(Self::Gelu),
            "silu" | "swish" => Some(Self::Silu),
            "sigmoid" => Some(Self::Sigmoid),
            "tanh" => Some(Self::Tanh),
            _ => None,
        }
    }
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
        };
        call_one(operation, OperationAttributes::None, &[input])
    }
}
