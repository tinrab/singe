mod core;
mod matmul;

pub use self::matmul::{grouped_matmul, matmul};

use crate::{
    backend::{cpu::utility::floating_binary, registry::Operation},
    error::Result,
    tensor::Tensor,
};

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Add)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Sub)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Mul)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Div)
}

pub fn equal(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Equal)
}

pub fn not_equal(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::NotEqual)
}

pub fn less(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Less)
}

pub fn less_equal(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::LessEqual)
}

pub fn greater(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::Greater)
}

pub fn greater_equal(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::GreaterEqual)
}

/// Defines binary operations on floating data types whose expression is the same for `f32` and `f64`.
macro_rules! floating_binary_operations {
    ($($name:ident($x:ident, $y:ident) => $expression:expr),+ $(,)?) => {
        $(pub fn $name(a: &Tensor, b: &Tensor) -> Result<Tensor> {
            floating_binary(a, b, |$x: f32, $y: f32| $expression, |$x: f64, $y: f64| $expression)
        })+
    };
}

floating_binary_operations!(
    // `modulo` and `remainder` take the sign of the divisor, as Python's `%` does: the truncated remainder, plus the divisor where their signs differ, as `PyTorch` computes it.
    modulo(x, y) => {
        let truncated = x % y;
        if truncated != 0.0 && (truncated < 0.0) != (y < 0.0) { truncated + y } else { truncated }
    },
    remainder(x, y) => {
        let truncated = x % y;
        if truncated != 0.0 && (truncated < 0.0) != (y < 0.0) { truncated + y } else { truncated }
    },
    atan2(x, y) => x.atan2(y),
    maximum(x, y) => if x.is_nan() || y.is_nan() { x + y } else if x < y { y } else { x },
    minimum(x, y) => if x.is_nan() || y.is_nan() { x + y } else if y < x { y } else { x },
);

pub fn logical_and(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::LogicalAnd)
}

pub fn logical_or(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    core::binary(a, b, Operation::LogicalOr)
}
