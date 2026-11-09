use crate::{
    backend::{
        cpu::iteration::{ElementwisePlan, typed_values},
        registry::{Operation, operation_schema},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    storage::cpu::CpuSlice,
    tensor::Tensor,
};

enum Values<'a> {
    F32(CpuSlice<'a, f32>),
    F64(CpuSlice<'a, f64>),
    I64(CpuSlice<'a, i64>),
    Bool(CpuSlice<'a, bool>),
}

impl<'a> Values<'a> {
    fn create(input: &'a Tensor) -> Result<Self> {
        Ok(match input.data_type() {
            DataType::F32 => Self::F32(typed_values(input)?),
            DataType::F64 => Self::F64(typed_values(input)?),
            DataType::I64 => Self::I64(typed_values(input)?),
            DataType::Bool => Self::Bool(typed_values(input)?),
            _ => return Err(Error::unsupported_data_type("cpu core arithmetic")),
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "tensor promotion rounds integer operands to f64 precision"
    )]
    fn float(&self, index: usize) -> f64 {
        match self {
            Self::F32(values) => f64::from(values[index]),
            Self::F64(values) => values[index],
            Self::I64(values) => values[index] as f64,
            Self::Bool(values) => f64::from(u8::from(values[index])),
        }
    }

    fn integer(&self, index: usize) -> i64 {
        match self {
            Self::I64(values) => values[index],
            Self::Bool(values) => i64::from(values[index]),
            _ => unreachable!("integer promotion excludes floating operands"),
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "tensor promotion rounds operands to f32 precision"
    )]
    fn single(&self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::F64(values) => values[index] as f32,
            Self::I64(values) => values[index] as f32,
            Self::Bool(values) => f32::from(u8::from(values[index])),
        }
    }
}

pub fn binary(lhs: &Tensor, rhs: &Tensor, operation: Operation) -> Result<Tensor> {
    let inference = operation_schema(operation)
        .binary_request(lhs.value_spec(), rhs.value_spec())
        .ok_or_else(|| Error::unsupported_operation("cpu core binary operation"))?
        .infer()?;
    let plan = ElementwisePlan::create(&[lhs, rhs])?;
    if matches!(operation, Operation::LogicalAnd | Operation::LogicalOr) {
        let a = typed_values::<bool>(lhs)?;
        let b = typed_values::<bool>(rhs)?;
        return Tensor::from_vec(
            Device::Cpu,
            plan.shape(),
            plan.collect(|linear| {
                let a = a[plan.offset(0, linear)];
                let b = b[plan.offset(1, linear)];
                if operation == Operation::LogicalAnd {
                    a && b
                } else {
                    a || b
                }
            }),
        );
    }

    let data_type = inference.computation_data_type();
    let lhs = Values::create(lhs)?;
    let rhs = Values::create(rhs)?;
    let comparison = matches!(
        operation,
        Operation::Equal
            | Operation::NotEqual
            | Operation::Less
            | Operation::LessEqual
            | Operation::Greater
            | Operation::GreaterEqual
    );
    if comparison {
        let output = plan.collect(|linear| {
            let (a, b) = (plan.offset(0, linear), plan.offset(1, linear));
            if matches!(data_type, DataType::I64 | DataType::Bool) {
                compare(operation, lhs.integer(a), rhs.integer(b))
            } else if data_type == DataType::F32 {
                compare(operation, lhs.single(a), rhs.single(b))
            } else {
                compare(operation, lhs.float(a), rhs.float(b))
            }
        });
        return Tensor::from_vec(Device::Cpu, plan.shape(), output);
    }
    match data_type {
        DataType::F32 => Tensor::from_vec(
            Device::Cpu,
            plan.shape(),
            plan.collect(|linear| {
                let (a, b) = (plan.offset(0, linear), plan.offset(1, linear));
                let a = lhs.single(a);
                let b = rhs.single(b);
                match operation {
                    Operation::Add => a + b,
                    Operation::Sub => a - b,
                    Operation::Mul => a * b,
                    Operation::Div => a / b,
                    _ => unreachable!(),
                }
            }),
        ),
        DataType::F64 => Tensor::from_vec(
            Device::Cpu,
            plan.shape(),
            plan.collect(|linear| {
                let (a, b) = (plan.offset(0, linear), plan.offset(1, linear));
                let a = lhs.float(a);
                let b = rhs.float(b);
                match operation {
                    Operation::Add => a + b,
                    Operation::Sub => a - b,
                    Operation::Mul => a * b,
                    Operation::Div => a / b,
                    _ => unreachable!(),
                }
            }),
        ),
        DataType::I64 => {
            let output = plan
                .collect(|linear| {
                    let (a, b) = (plan.offset(0, linear), plan.offset(1, linear));
                    let a = lhs.integer(a);
                    let b = rhs.integer(b);
                    match operation {
                        Operation::Add => a.checked_add(b),
                        Operation::Sub => a.checked_sub(b),
                        Operation::Mul => a.checked_mul(b),
                        _ => None,
                    }
                    .ok_or_else(|| Error::out_of_range("integer arithmetic overflow"))
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            Tensor::from_vec(Device::Cpu, plan.shape(), output)
        }
        _ => Err(Error::unsupported_data_type("cpu core arithmetic")),
    }
}

fn compare<T: Copy + PartialEq + PartialOrd>(operation: Operation, a: T, b: T) -> bool {
    match operation {
        Operation::Equal => a == b,
        Operation::NotEqual => a != b,
        Operation::Less => a < b,
        Operation::LessEqual => a <= b,
        Operation::Greater => a > b,
        Operation::GreaterEqual => a >= b,
        _ => unreachable!("comparison opcode checked by caller"),
    }
}
