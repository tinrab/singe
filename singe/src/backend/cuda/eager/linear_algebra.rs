use crate::{
    backend::{
        cuda::{
            self,
            eager::unsupported_eager_operation,
            gemm::GemmLayout,
            kernel_plan,
            provider_selection::{KernelChoice, Operand, matmul_kernel},
            stream::ensure_not_stream_capturing,
            utility::ensure_same_device,
        },
        registry::{Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    tensor::Tensor,
};

pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_same_device(lhs, rhs)?;
    if lhs.data_type() != rhs.data_type() {
        return Err(Error::data_type_mismatch(
            "cuda matmul rhs",
            lhs.data_type(),
            rhs.data_type(),
        ));
    }

    let problem = spec::matmul_problem(lhs.shape(), rhs.shape(), "cuda matmul")?;
    if matches!(lhs.data_type(), DataType::Bool | DataType::F4E2M1) {
        return unsupported_eager_operation(
            Operation::MatMul,
            lhs.data_type(),
            &problem.output_shape,
            &[lhs.data_type(), rhs.data_type()],
        );
    }

    if problem.is_zero_work() {
        ensure_not_stream_capturing(lhs.device(), "cuda eager matmul during stream capture")?;
        return cuda::execution_context(lhs.device())?
            .zeroed_tensor(lhs.data_type(), &problem.output_shape);
    }

    let operands = [Operand::of(lhs), Operand::of(rhs)];
    if matmul_kernel(operands[0], operands[1]) == KernelChoice::Library {
        return cublas_matmul(lhs, rhs);
    }
    kernel_plan::execute(Operation::MatMul, &OperationAttributes::None, &[lhs, rhs])
}

/// Multiplies `lhs` by `rhs` with one cuBLAS strided batched GEMM that reads both operands in place, accumulating in `F32`.
fn cublas_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    ensure_not_stream_capturing(lhs.device(), "cuda eager matmul during stream capture")?;
    let gemm = GemmLayout::create(lhs.layout(), rhs.layout())
        .ok_or_else(|| Error::unsupported_operation("cuda cuBLAS matmul operand layouts"))?;
    let context = cuda::execution_context(lhs.device())?;
    context.bind()?;
    let output = context.allocate_output(lhs.data_type(), &gemm.output_shape)?;
    context.sync_scope(|| {
        context.with_cublas(|cublas| gemm.launch(cublas, lhs, rhs, output.region()))
    })?;
    // SAFETY: beta is zero and the completed GEMM writes every contiguous output element.
    Ok(unsafe { output.assume_init() })
}
