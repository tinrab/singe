use std::ptr;

pub use crate::backend::cuda::eager::linear_algebra::matmul;
use crate::{
    backend::{
        cuda::{
            self,
            eager::ensure_cuda_eager_same_device_inputs,
            gemm::{ScaledMatmulPointers, enqueue_scaled_matmul},
            kernel_plan,
            stream::{CudaStream, ensure_not_stream_capturing},
            tensor_region::CudaTensorSource,
        },
        decomposition,
        registry::{Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    utility::Locked,
};

// `torch._scaled_mm` on FP8 tensor cores through cuBLASLt when it takes the operands, and as its decomposition, a dequantized `F32` product, otherwise.
pub fn scaled_matmul(
    lhs: &Tensor,
    rhs: &Tensor,
    lhs_scale: &Tensor,
    rhs_scale: &Tensor,
    output: DataType,
) -> Result<Tensor> {
    ensure_cuda_eager_same_device_inputs(lhs, &[rhs, lhs_scale, rhs_scale])?;
    let problem = spec::scaled_matmul_problem(
        operand(lhs),
        operand(rhs),
        operand(lhs_scale),
        operand(rhs_scale),
        "cuda scaled_mm",
    )?;
    if spec::takes_scaled_matmul(lhs.layout(), rhs.layout(), &problem) {
        // Row-wise scales are vectors cuBLASLt reads contiguously, as outer-vector scales where the target has them.
        let scales_read = problem.scaling == spec::MatmulScaling::Tensorwise
            || (lhs_scale.is_contiguous() && rhs_scale.is_contiguous());
        if scales_read
            && let Ok(product) =
                cublaslt_scaled_matmul(lhs, rhs, Some((lhs_scale, rhs_scale)), output, &problem)
        {
            return Ok(product);
        }
        // Elsewhere the unscaled FP8 product, in `F32`, takes the outer product of the scales and rounds once.
        if problem.scaling == spec::MatmulScaling::Rowwise {
            let unscaled = spec::ScaledMatmulProblem {
                scaling: spec::MatmulScaling::Tensorwise,
                ..problem
            };
            if let Ok(product) = cublaslt_scaled_matmul(lhs, rhs, None, DataType::F32, &unscaled) {
                return product.mul(lhs_scale)?.mul(rhs_scale)?.cast(output);
            }
        }
    }
    decomposition::decompose(
        Operation::ScaledMatMul,
        &OperationAttributes::Cast(output),
        &[lhs, rhs, lhs_scale, rhs_scale],
    )?
    .pop()
    .ok_or_else(|| Error::missing_output("scaled matmul"))
}

fn operand(tensor: &Tensor) -> (&Shape, DataType) {
    (tensor.shape(), tensor.data_type())
}

// The FP8 product of `lhs` and `rhs` by cuBLASLt, scaled by `scales`, or by one without them.
fn cublaslt_scaled_matmul(
    lhs: &Tensor,
    rhs: &Tensor,
    scales: Option<(&Tensor, &Tensor)>,
    output: DataType,
    problem: &spec::ScaledMatmulProblem,
) -> Result<Tensor> {
    ensure_not_stream_capturing(lhs.device(), "cuda eager scaled_mm during stream capture")?;
    let context = cuda::execution_context(lhs.device())?;
    context.bind()?;
    let product = context.allocate_output(output, &problem.output_shape)?;
    let pointer = |tensor: &Tensor| -> Result<_> {
        Ok(tensor
            .tensor_region()?
            .region()
            .as_ptr()
            .cast_const()
            .cast())
    };
    let pointers = ScaledMatmulPointers {
        lhs: pointer(lhs)?,
        rhs: pointer(rhs)?,
        lhs_scale: scales.map_or(Ok(ptr::null()), |(lhs_scale, _)| pointer(lhs_scale))?,
        rhs_scale: scales.map_or(Ok(ptr::null()), |(_, rhs_scale)| pointer(rhs_scale))?,
        output: product.region().region().as_ptr().cast(),
    };
    context.sync_scope(|| {
        let cublaslt = context.runtime().cublaslt().locked("cuda runtime cache")?;
        // SAFETY: the operands, scales, and output live through the synchronized scope.
        unsafe {
            enqueue_scaled_matmul(
                &cublaslt,
                problem,
                [lhs.data_type(), rhs.data_type(), output],
                &pointers,
                context.stream().map(CudaStream::raw),
            )
        }
    })?;
    // SAFETY: cuBLASLt wrote every element of the product.
    Ok(unsafe { product.assume_init() })
}

pub fn grouped_matmul(input: &Tensor, weights: &Tensor, offsets: &Tensor) -> Result<Tensor> {
    ensure_cuda_eager_same_device_inputs(input, &[weights, offsets])?;
    kernel_plan::execute(
        Operation::GroupedMatMul,
        &OperationAttributes::None,
        &[input, weights, offsets],
    )
}
