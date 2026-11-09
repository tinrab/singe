use crate::{
    backend::registry::{Operation, OperationAttributes},
    dispatch::call_one,
    error::Result,
    graph::capture,
    scalar::Scalar,
    tensor::Tensor,
};

macro_rules! unary_operator {
    ($function:ident => $operation:ident) => {
        pub fn $function(input: &Tensor) -> Result<Tensor> {
            call_one(Operation::$operation, OperationAttributes::None, &[input])
        }
    };
}

macro_rules! binary_operator {
    ($function:ident => $operation:ident) => {
        pub fn $function(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            let (lhs, rhs) = lhs.binary_operands(rhs, Operation::$operation)?;
            call_one(
                Operation::$operation,
                OperationAttributes::None,
                &[&lhs, &rhs],
            )
        }
    };
}

macro_rules! right_scalar_operator {
    ($function:ident => $operation:ident) => {
        pub fn $function(lhs: &Tensor, rhs: Scalar) -> Result<Tensor> {
            let lhs = lhs.scalar_operand(rhs, false)?;
            call_one(
                Operation::$operation,
                OperationAttributes::Scalar(rhs),
                &[&lhs],
            )
        }
    };
}

macro_rules! left_scalar_operator {
    ($function:ident => $operation:ident) => {
        pub fn $function(rhs: &Tensor, lhs: Scalar) -> Result<Tensor> {
            let rhs = rhs.scalar_operand(lhs, Operation::$operation == Operation::RDivScalar)?;
            call_one(
                Operation::$operation,
                OperationAttributes::Scalar(lhs),
                &[&rhs],
            )
        }
    };
}

unary_operator!(neg_tensor => Neg);

binary_operator!(add_tensors => Add);

binary_operator!(sub_tensors => Sub);

binary_operator!(mul_tensors => Mul);

binary_operator!(div_tensors => Div);

right_scalar_operator!(add_tensor_scalar => AddScalar);

right_scalar_operator!(sub_tensor_scalar => SubScalar);

left_scalar_operator!(rsub_tensor_scalar => RSubScalar);

right_scalar_operator!(mul_tensor_scalar => Scale);

right_scalar_operator!(pow_tensor_scalar => PowScalar);

pub fn div_tensor_scalar(lhs: &Tensor, rhs: Scalar) -> Result<Tensor> {
    let lhs = lhs.scalar_operand(rhs, true)?;
    let divisor = capture::literal(lhs.device(), |device| rhs.tensor(device, lhs.data_type()))?;
    lhs.div(&divisor)
}

left_scalar_operator!(rdiv_tensor_scalar => RDivScalar);
