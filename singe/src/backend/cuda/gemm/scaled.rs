// FP8 products on tensor cores through cuBLASLt, which takes FP8 operands in one layout only, "TN": the left operand transposed and both column-major.
// `torch._scaled_mm`'s row-major `lhs [M, K]` and column-major `rhs [K, N]` are that layout read the other way round: cuBLASLt's `A` is `rhs`, `K × N` with leading dimension `K`, transposed, its `B` is `lhs`, `K × M` with leading dimension `K`, and its column-major `N × M` result is the row-major `[M, N]` output.

use std::{ffi::c_void, ptr};

use singe_cublas::{
    lt::{
        context::Context as CublasLtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul_algorithm_heuristics, matmul_async},
        types::{MatmulDescriptorAttribute, MatrixScale},
    },
    scalar::Scalar,
    types::{ComputeType, Operation as CublasOperation},
};
use singe_cuda::{
    data_type::DataType as CudaDataType,
    stream::Stream,
    view::{DeviceInputView, DeviceOutputView, DeviceRegion},
};

use crate::{
    backend::spec::{MatmulScaling, ScaledMatmulProblem},
    data_type::DataType,
    error::{Error, Result},
    utility::{to_i64, to_u64},
};

// Device storage of one FP8 product: its operands and their `F32` scales, and its output.
pub struct ScaledMatmulOperands<'a> {
    pub lhs: DeviceRegion<'a>,
    pub rhs: DeviceRegion<'a>,
    pub lhs_scale: Option<DeviceRegion<'a>>,
    pub rhs_scale: Option<DeviceRegion<'a>>,
    pub output: DeviceRegion<'a>,
}

fn scale_pointer(scale: Option<DeviceRegion<'_>>) -> *const c_void {
    scale.map_or(ptr::null(), |scale| scale.as_ptr().cast_const().cast())
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
// The regions hold the problem's operands, scales, and output on the context's device, which stay alive until the stream's work completes, and nothing else accesses the output until then.
pub unsafe fn enqueue_scaled_matmul(
    cublaslt: &CublasLtContext,
    problem: &ScaledMatmulProblem,
    types: [DataType; 3],
    operands: &ScaledMatmulOperands<'_>,
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
            &scale_pointer(operands.rhs_scale),
        )?;
        descriptor.set_attribute(
            MatmulDescriptorAttribute::BScalePointer,
            &scale_pointer(operands.lhs_scale),
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
    let rhs = DeviceInputView::<u8>::create(operands.rhs)?;
    let lhs = DeviceInputView::<u8>::create(operands.lhs)?;
    // SAFETY: the caller gives the output region exclusively to this product until the stream's work completes.
    let mut output = unsafe {
        DeviceOutputView::<u8>::from_raw_parts(operands.output, operands.output.byte_len())?
    };
    // SAFETY: the layouts describe the caller's buffers; with beta zero cuBLASLt reads nothing of C, which aliases D.
    unsafe {
        matmul_async(
            cublaslt,
            &descriptor,
            Scalar::host(&alpha),
            &rhs,
            &a_layout,
            &lhs,
            &b_layout,
            Scalar::host(&beta),
            None,
            &output_layout,
            &mut output,
            &output_layout,
            Some(&algorithm.algorithm),
            None,
            stream,
        )
    }?;
    Ok(())
}
