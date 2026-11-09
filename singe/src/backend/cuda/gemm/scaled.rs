// FP8 products on tensor cores through cuBLASLt, which takes FP8 operands in one layout only, "TN": the left operand transposed and both column-major.
// `torch._scaled_mm`'s row-major `lhs [M, K]` and column-major `rhs [K, N]` are that layout read the other way round: cuBLASLt's `A` is `rhs`, `K × N` with leading dimension `K`, transposed, its `B` is `lhs`, `K × M` with leading dimension `K`, and its column-major `N × M` result is the row-major `[M, N]` output.

use std::ffi::c_void;

use singe_cublas::{
    lt::{
        context::Context as CublasLtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul_algorithm_heuristics, matmul_raw},
        types::{MatmulDescriptorAttribute, MatrixScale},
    },
    types::{ComputeType, Operation as CublasOperation},
};
use singe_cuda::{data_type::DataType as CudaDataType, stream::Stream};

use crate::{
    backend::spec::{MatmulScaling, ScaledMatmulProblem},
    data_type::DataType,
    error::{Error, Result},
    utility::{to_i64, to_u64},
};

// Device pointers of one FP8 product: its operands and their `F32` scales, and its output.
pub struct ScaledMatmulPointers {
    pub lhs: *const c_void,
    pub rhs: *const c_void,
    pub lhs_scale: *const c_void,
    pub rhs_scale: *const c_void,
    pub output: *mut c_void,
}

fn cuda_data_type(data_type: DataType) -> Result<CudaDataType> {
    Ok(match data_type {
        DataType::F8E4M3 => CudaDataType::F8E4M3,
        DataType::F8E5M2 => CudaDataType::F8E5M2,
        DataType::F16 => CudaDataType::F16,
        DataType::Bf16 => CudaDataType::Bf16,
        DataType::F32 => CudaDataType::F32,
        _ => return Err(Error::unsupported_data_type("cuda scaled_mm")),
    })
}

fn extent(value: i64) -> Result<u64> {
    to_u64(value, "cuda scaled_mm extent")
}

// Enqueues the product that `spec::takes_scaled_matmul` accepted on `stream`, choosing cuBLASLt's first algorithm that needs no workspace.
//
// # Safety
//
// The pointers address live device memory of the problem's operands, scales, and output on the context's device, which stays alive until the stream's work completes.
pub unsafe fn enqueue_scaled_matmul(
    cublaslt: &CublasLtContext,
    problem: &ScaledMatmulProblem,
    types: [DataType; 3],
    pointers: &ScaledMatmulPointers,
    stream: Option<&Stream>,
) -> Result<()> {
    let [lhs_type, rhs_type, output_type] = types;
    let (rows, depth, columns) = (
        extent(problem.rows)?,
        extent(problem.depth)?,
        extent(problem.columns)?,
    );
    let leading = |value: u64| to_i64(value, "cuda scaled_mm leading dimension");
    let mut descriptor = MatmulDescriptor::create(ComputeType::F32, CudaDataType::F32)?;
    descriptor.set_transpose_a(CublasOperation::Transpose)?;
    descriptor.set_transpose_b(CublasOperation::NonTranspose)?;
    // SAFETY: the scale attributes hold device pointers to one `F32` each, which the caller keeps alive.
    unsafe {
        descriptor.set_attribute(
            MatmulDescriptorAttribute::AScalePointer,
            &pointers.rhs_scale,
        )?;
        descriptor.set_attribute(
            MatmulDescriptorAttribute::BScalePointer,
            &pointers.lhs_scale,
        )?;
    }

    // Row-wise scales are cuBLASLt's outer vectors: one per column of its `A`, `rhs`, and one per column of its `B`, `lhs`'s rows.
    if problem.scaling == MatmulScaling::Rowwise {
        descriptor.set_a_scale_mode(MatrixScale::OuterVector32F)?;
        descriptor.set_b_scale_mode(MatrixScale::OuterVector32F)?;
    }

    let a_layout =
        MatrixLayout::create(cuda_data_type(rhs_type)?, depth, columns, leading(depth)?)?;
    let b_layout = MatrixLayout::create(cuda_data_type(lhs_type)?, depth, rows, leading(depth)?)?;
    let output_layout = MatrixLayout::create(
        cuda_data_type(output_type)?,
        columns,
        rows,
        leading(columns)?,
    )?;
    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(0)?;
    let heuristics = matmul_algorithm_heuristics(
        cublaslt,
        &descriptor,
        &a_layout,
        &b_layout,
        &output_layout,
        &output_layout,
        &preference,
        1,
    )?;
    let algorithm = heuristics.first().ok_or_else(|| {
        Error::library_refused(
            "cuBLASLt",
            Error::unsupported_operation("cuda scaled_mm without a cuBLASLt algorithm"),
        )
    })?;
    let (alpha, beta) = (1.0_f32, 0.0_f32);
    // SAFETY: the layouts describe the caller's buffers; with beta zero cuBLASLt reads nothing of C, which aliases D.
    unsafe {
        matmul_raw(
            cublaslt,
            &descriptor,
            (&raw const alpha).cast(),
            pointers.rhs,
            &a_layout,
            pointers.lhs,
            &b_layout,
            (&raw const beta).cast(),
            pointers.output.cast_const(),
            &output_layout,
            pointers.output,
            &output_layout,
            Some(&algorithm.algorithm),
            None,
            stream,
        )
    }?;
    Ok(())
}
