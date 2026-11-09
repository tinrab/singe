use libm::{erf as erf_f64, erff};

use crate::{
    backend::cpu::{
        iteration::{ElementwisePlan, typed_values},
        utility::floating_unary,
    },
    device::Device,
    error::Result,
    tensor::Tensor,
};

pub fn erf(input: &Tensor) -> Result<Tensor> {
    floating_unary(input, erff, erf_f64)
}

macro_rules! floating_unary_operations {
    ($($name:ident($value:ident) => $expression:expr),+ $(,)?) => {
        $(pub fn $name(input: &Tensor) -> Result<Tensor> {
            floating_unary(input, |$value: f32| $expression, |$value: f64| $expression)
        })+
    };
}

floating_unary_operations!(
    neg(x) => -x,
    abs(x) => x.abs(),
    sqrt(x) => x.sqrt(),
    exp(x) => x.exp(),
    log(x) => x.ln(),
    sin(x) => x.sin(),
    cos(x) => x.cos(),
    tan(x) => x.tan(),
    sinh(x) => x.sinh(),
    cosh(x) => x.cosh(),
    tanh(x) => x.tanh(),
    asin(x) => x.asin(),
    acos(x) => x.acos(),
    atan(x) => x.atan(),
    ceil(x) => x.ceil(),
    floor(x) => x.floor(),
    truncate(x) => x.trunc(),
    reciprocal(x) => 1.0 / x,
    square(x) => x * x,
    rsqrt(x) => 1.0 / x.sqrt(),
    asinh(x) => x.asinh(),
    acosh(x) => x.acosh(),
    atanh(x) => x.atanh(),
);

pub fn logical_not(input: &Tensor) -> Result<Tensor> {
    let plan = ElementwisePlan::create(&[input])?;
    let values = typed_values::<bool>(input)?;
    Tensor::from_vec(
        Device::Cpu,
        plan.shape(),
        plan.collect(|linear| !values[plan.offset(0, linear)]),
    )
}
