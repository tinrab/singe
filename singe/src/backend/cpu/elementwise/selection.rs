use crate::{
    backend::cpu::iteration::{ElementwisePlan, typed_values},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    tensor::{Tensor, TensorDataType, tensor_data_type_cases},
};

pub fn where_op(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let plan = ElementwisePlan::create(&[condition, x, y])?;
    if condition.data_type() != DataType::Bool {
        return Err(Error::unsupported_data_type("where condition"));
    }

    if x.data_type() != y.data_type() {
        return Err(Error::data_type_mismatch(
            "where values",
            x.data_type(),
            y.data_type(),
        ));
    }

    let conditions = typed_values::<bool>(condition)?;
    macro_rules! select_type {
        ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
            match x.data_type() {
                $(DataType::$variant => select::<$ty>(&plan, &conditions, x, y),)+
                _ => Err(Error::unsupported_data_type("cpu where values")),
            }
        };
    }
    tensor_data_type_cases!(select_type)
}

fn select<T: TensorDataType + 'static>(
    plan: &ElementwisePlan,
    condition: &[bool],
    x: &Tensor,
    y: &Tensor,
) -> Result<Tensor> {
    let x = typed_values::<T>(x)?;
    let y = typed_values::<T>(y)?;
    let output = plan.collect(|linear| {
        if condition[plan.offset(0, linear)] {
            x[plan.offset(1, linear)]
        } else {
            y[plan.offset(2, linear)]
        }
    });
    Tensor::from_vec(Device::Cpu, plan.shape(), output)
}
