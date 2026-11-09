use crate::{
    backend::{
        cpu::iteration::{ElementwisePlan, typed_values},
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    scalar::Scalar,
    tensor::Tensor,
};

fn scalar_operation(input: &Tensor, scalar: Scalar, operation: Operation) -> Result<Tensor> {
    let input = input.scalar_operand(scalar, operation == Operation::RDivScalar)?;
    let plan = ElementwisePlan::create(&[&input])?;
    macro_rules! floating {
        ($ty:ty, $coefficient:expr) => {{
            let values = typed_values::<$ty>(&input)?;
            let scalar = $coefficient;
            Tensor::from_vec(
                Device::Cpu,
                plan.shape(),
                plan.collect(|linear| {
                    let value = values[plan.offset(0, linear)];
                    match operation {
                        Operation::AddScalar => value + scalar,
                        Operation::SubScalar => value - scalar,
                        Operation::RSubScalar => scalar - value,
                        Operation::Scale => value * scalar,
                        Operation::RDivScalar => scalar / value,
                        _ => unreachable!(),
                    }
                }),
            )
        }};
    }
    match input.data_type() {
        DataType::F32 => floating!(f32, scalar.single()),
        DataType::F64 => floating!(f64, scalar.float()),
        DataType::I64 => {
            let values = typed_values::<i64>(&input)?;
            let scalar = scalar.signed();
            let output = plan
                .collect(|linear| {
                    let value = values[plan.offset(0, linear)];
                    match operation {
                        Operation::AddScalar => value.checked_add(scalar),
                        Operation::SubScalar => value.checked_sub(scalar),
                        Operation::RSubScalar => scalar.checked_sub(value),
                        Operation::Scale => value.checked_mul(scalar),
                        _ => None,
                    }
                    .ok_or_else(|| Error::out_of_range("integer arithmetic overflow"))
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            Tensor::from_vec(Device::Cpu, plan.shape(), output)
        }
        _ => Err(Error::unsupported_data_type("cpu scalar arithmetic")),
    }
}

pub fn add_scalar(a: &Tensor, scalar: Scalar) -> Result<Tensor> {
    scalar_operation(a, scalar, Operation::AddScalar)
}

pub fn sub_scalar(a: &Tensor, scalar: Scalar) -> Result<Tensor> {
    scalar_operation(a, scalar, Operation::SubScalar)
}

pub fn rsub_scalar(scalar: Scalar, a: &Tensor) -> Result<Tensor> {
    scalar_operation(a, scalar, Operation::RSubScalar)
}

pub fn scale(a: &Tensor, scalar: Scalar) -> Result<Tensor> {
    scalar_operation(a, scalar, Operation::Scale)
}

pub fn rdiv_scalar(scalar: Scalar, a: &Tensor) -> Result<Tensor> {
    scalar_operation(a, scalar, Operation::RDivScalar)
}
