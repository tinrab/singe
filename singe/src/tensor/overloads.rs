//! Arithmetic operators for tensors, scalars, and fallible tensor results.
//!
//! Every operator returns `Result<Tensor>`, and a `Result<Tensor>` operand passes its error through, so a chain such as `(&x * &w + &b)?` reports its first error at the end.
//! Rust's orphan rules forbid an operator between `Result<Tensor>` and a plain scalar, so such a chain needs `?` before the scalar operation.

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::{error::Result, tensor::Tensor};

/// Implements a binary operator for every pairing of `&Tensor`, `Tensor`, and `Result<Tensor>` operands.
macro_rules! tensor_operators {
    ($($trait:ident::$function:ident => $method:ident),+ $(,)?) => {
        $(
            impl $trait<&Tensor> for &Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: &Tensor) -> Result<Tensor> {
                    Tensor::$method(&self, rhs)
                }
            }

            impl $trait<Tensor> for &Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: Tensor) -> Result<Tensor> {
                    Tensor::$method(&self, &rhs)
                }
            }

            impl $trait<Result<Tensor>> for &Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: Result<Tensor>) -> Result<Tensor> {
                    Tensor::$method(&self, &rhs?)
                }
            }

            impl $trait<&Tensor> for Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: &Tensor) -> Result<Tensor> {
                    Tensor::$method(&self, rhs)
                }
            }

            impl $trait<Tensor> for Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: Tensor) -> Result<Tensor> {
                    Tensor::$method(&self, &rhs)
                }
            }

            impl $trait<Result<Tensor>> for Tensor {
                type Output = Result<Tensor>;

                fn $function(self, rhs: Result<Tensor>) -> Result<Tensor> {
                    Tensor::$method(&self, &rhs?)
                }
            }

            impl $trait<&Tensor> for Result<Tensor> {
                type Output = Result<Tensor>;

                fn $function(self, rhs: &Tensor) -> Result<Tensor> {
                    Tensor::$method(&self?, rhs)
                }
            }

            impl $trait<Tensor> for Result<Tensor> {
                type Output = Result<Tensor>;

                fn $function(self, rhs: Tensor) -> Result<Tensor> {
                    Tensor::$method(&self?, &rhs)
                }
            }
        )+
    };
}

/// Implements a binary operator between tensor operands and each scalar type, on both sides.
macro_rules! scalar_operators {
    ($($scalar:ty),+ $(,)?) => {
        $(
            scalar_operators!(@tensor $scalar, Add::add => add_scalar, add_scalar);
            scalar_operators!(@tensor $scalar, Sub::sub => sub_scalar, rsub_scalar);
            scalar_operators!(@tensor $scalar, Mul::mul => mul_scalar, mul_scalar);
            scalar_operators!(@tensor $scalar, Div::div => div_scalar, rdiv_scalar);
        )+
    };
    (@tensor $scalar:ty, $trait:ident::$function:ident => $right:ident, $left:ident) => {
        impl $trait<$scalar> for &Tensor {
            type Output = Result<Tensor>;

            fn $function(self, rhs: $scalar) -> Result<Tensor> {
                Tensor::$right(&self, rhs)
            }
        }

        impl $trait<$scalar> for Tensor {
            type Output = Result<Tensor>;

            fn $function(self, rhs: $scalar) -> Result<Tensor> {
                Tensor::$right(&self, rhs)
            }
        }

        impl $trait<&Tensor> for $scalar {
            type Output = Result<Tensor>;

            fn $function(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$left(&rhs, self)
            }
        }

        impl $trait<Tensor> for $scalar {
            type Output = Result<Tensor>;

            fn $function(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$left(&rhs, self)
            }
        }
    };
}

tensor_operators!(
    Add::add => add,
    Sub::sub => sub,
    Mul::mul => mul,
    Div::div => div,
    Rem::rem => remainder,
);

scalar_operators!(f32, f64, i32, i64);

impl Neg for &Tensor {
    type Output = Result<Tensor>;

    fn neg(self) -> Result<Tensor> {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Result<Self>;

    fn neg(self) -> Result<Self> {
        Self::neg(&self)
    }
}
