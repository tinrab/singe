use std::mem::MaybeUninit;

use singe_cublas::{
    lt::{
        matmul::{MatmulAlgorithm, matmul_raw},
        types::Epilogue,
    },
    types::ComputeType,
};
use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceInputView, DeviceOutput, DeviceOutputView},
};

use crate::{
    backend::cuda::{
        compiled::plan::cublas::{
            algorithm::{
                CublasLtMatmulDTypePlan, validate_algorithm_numerics, validate_cublaslt_bias,
                with_cublaslt,
            },
            descriptor::CublasLtMatmulDescriptors,
            layout::{CublasLtMatmulOutputLayout, CublasLtMatrixLayout},
        },
        device_views::tensor_output_view_with_len,
        gemm::GemmLayout,
        graph_value::CudaGraphValue,
        runtime::CudaRuntime,
        stream::{CudaStream, current_stream, with_cublas_stream},
        tensor_region::CudaTensorSource,
        utility::{ensure_same_data_type, ensure_same_device},
    },
    error::{Error, Result},
    utility::{checked_mul, to_i64},
};

/// Captures `a @ b` into `output` as one cuBLAS strided batched GEMM that reads both operands in place.
pub fn launch_cublas_matmul(
    runtime: &CudaRuntime,
    a: &CudaGraphValue,
    b: &CudaGraphValue,
    output: &CudaGraphValue,
) -> Result<()> {
    let gemm = validate_cublas_matmul_launch(a, b, output)?;
    with_cublas_stream(runtime, a.device(), |cublas| {
        gemm.launch(cublas, a, b, output.region()?)
    })
}

/// Returns the GEMM that writes `a @ b` into the contiguous `output`.
///
/// # Errors
///
/// Returns an error if the tensors are not on one CUDA device with one data type, cuBLAS cannot read the operands in place, or `output` is not the contiguous product.
pub fn validate_cublas_matmul_launch(
    a: &(impl CudaTensorSource + ?Sized),
    b: &(impl CudaTensorSource + ?Sized),
    output: &(impl CudaTensorSource + ?Sized),
) -> Result<GemmLayout> {
    ensure_same_device(a, b)?;
    ensure_same_device(a, output)?;
    ensure_same_data_type(a, b)?;
    ensure_same_data_type(a, output)?;
    let gemm = GemmLayout::create(a.layout(), b.layout())
        .ok_or_else(|| Error::unsupported_operation("cuda graph cuBLAS matmul operand layouts"))?;
    if output.shape() != &gemm.output_shape || !output.is_contiguous() {
        return Err(Error::shape_mismatch(
            "cuda graph cuBLAS matmul output shape",
            gemm.output_shape.dimensions().to_vec(),
            output.shape().dimensions().to_vec(),
        ));
    }
    Ok(gemm)
}

// A cuBLASLt matrix multiplication to launch.
pub struct CublasLtMatmulLaunch<'a> {
    pub runtime: &'a CudaRuntime,
    pub a: &'a CudaGraphValue,
    pub b: &'a CudaGraphValue,
    pub output: &'a CudaGraphValue,
    pub a_layout: &'a CublasLtMatrixLayout,
    pub b_layout: &'a CublasLtMatrixLayout,
    pub algorithm: &'a MatmulAlgorithm,
    pub compute_type: ComputeType,
    pub epilogue: Epilogue,
    pub bias: Option<&'a CudaGraphValue>,
    pub addend: Option<&'a CudaGraphValue>,
    pub workspace: Option<&'a mut DeviceMemory<MaybeUninit<u8>>>,
    pub stream: Option<&'a CudaStream>,
}

pub fn launch_cublaslt_matmul(launch: CublasLtMatmulLaunch<'_>) -> Result<()> {
    let CublasLtMatmulLaunch {
        runtime,
        a,
        b,
        output,
        a_layout,
        b_layout,
        algorithm,
        compute_type,
        epilogue,
        bias,
        addend,
        workspace,
        stream,
    } = launch;
    validate_algorithm_numerics(algorithm, output.data_type(), compute_type)?;
    // An addend is cuBLASLt's `C`, read with the output's layout, so it must have it.
    let addend = addend
        .map(|addend| {
            if addend.data_type() != output.data_type() || addend.layout() != output.layout() {
                return Err(Error::invalid_argument(
                    "cuda graph cuBLASLt addend layout differs from the output's",
                ));
            }
            Ok(addend.region()?.region().as_ptr())
        })
        .transpose()?;
    let current = current_stream(output.device())?;
    let stream = stream.map(CudaStream::raw).or(current.as_deref());
    let plan = CublasLtMatmulLaunchPlan::create(a, b, output, a_layout, b_layout, epilogue, bias)?;
    let mut memory = CublasLtMatmulMemoryViews::create(
        a,
        b,
        output,
        a_layout,
        b_layout,
        plan.output_layout(),
        plan.dtype_plan(),
    )?;
    with_cublaslt(runtime, |cublaslt| {
        let descriptors = CublasLtMatmulDescriptors::create(
            plan.dtype_plan(),
            compute_type,
            a_layout,
            b_layout,
            epilogue,
            bias,
            plan.output_layout(),
            plan.rows(),
            plan.cols(),
        )?;
        let alpha = 1.0_f32;
        let beta = if addend.is_some() { 1.0_f32 } else { 0.0_f32 };
        let output_pointer = memory.output.as_mut_ptr();
        let c = addend.map_or_else(
            || output_pointer.cast_const().cast(),
            |addend| addend.cast_const().cast(),
        );
        // SAFETY: the checked layouts describe A, B, C, and D.
        // With beta zero, cuBLASLt does not read C's values, so the output needs only write access; an addend has the output's layout and lives as long as the compiled graph holds its value.
        unsafe {
            matmul_raw(
                cublaslt,
                &descriptors.descriptor,
                (&raw const alpha).cast(),
                memory.a.as_ptr().cast(),
                &descriptors.a_layout,
                memory.b.as_ptr().cast(),
                &descriptors.b_layout,
                (&raw const beta).cast(),
                c,
                &descriptors.output_layout,
                output_pointer.cast(),
                &descriptors.output_layout,
                Some(algorithm),
                workspace.map(|workspace| workspace as &mut dyn DeviceOutput<u8>),
                stream,
            )
        }?;
        Ok(())
    })
}

pub struct CublasLtMatmulMemoryViews<'a> {
    pub(crate) a: DeviceInputView<'a, u8>,
    pub(crate) b: DeviceInputView<'a, u8>,
    pub(crate) output: DeviceOutputView<'a, u8>,
}

impl<'a> CublasLtMatmulMemoryViews<'a> {
    pub(crate) fn create(
        a: &'a CudaGraphValue,
        b: &'a CudaGraphValue,
        output: &'a CudaGraphValue,
        a_layout: &CublasLtMatrixLayout,
        b_layout: &CublasLtMatrixLayout,
        output_layout: &CublasLtMatmulOutputLayout,
        dtype_plan: CublasLtMatmulDTypePlan,
    ) -> Result<Self> {
        let a_bytes = checked_mul(
            a_layout.element_count,
            dtype_plan.data_type.size_bytes(),
            "cuBLASLt lhs bytes",
        )?;
        let b_bytes = checked_mul(
            b_layout.element_count,
            dtype_plan.data_type.size_bytes(),
            "cuBLASLt rhs bytes",
        )?;
        let a = DeviceInputView::create(a.region()?.region().slice(..a_bytes)?)?;
        let b = DeviceInputView::create(b.region()?.region().slice(..b_bytes)?)?;
        let output = tensor_output_view_with_len::<u8>(
            output.region()?,
            dtype_plan.data_type,
            checked_mul(
                output_layout.element_count,
                dtype_plan.data_type.size_bytes(),
                "cuBLASLt output bytes",
            )?,
            "cuBLASLt output",
        )?;
        Ok(Self { a, b, output })
    }
}

pub struct CublasLtMatmulLaunchPlan {
    dtype_plan: CublasLtMatmulDTypePlan,
    output_layout: CublasLtMatmulOutputLayout,
    rows: u64,
    cols: u64,
}

impl CublasLtMatmulLaunchPlan {
    pub(crate) fn create(
        a: &CudaGraphValue,
        b: &CudaGraphValue,
        output: &CudaGraphValue,
        a_layout: &CublasLtMatrixLayout,
        b_layout: &CublasLtMatrixLayout,
        epilogue: Epilogue,
        bias: Option<&CudaGraphValue>,
    ) -> Result<Self> {
        ensure_same_device(a, b)?;
        ensure_same_device(a, output)?;
        let dtype_plan = CublasLtMatmulDTypePlan::create(output.data_type(), epilogue)?;
        if a.data_type() != dtype_plan.data_type
            || b.data_type() != dtype_plan.data_type
            || output.data_type() != dtype_plan.data_type
        {
            return Err(Error::unsupported_data_type("cuda graph cuBLASLt matmul"));
        }

        if a_layout.logical_cols != b_layout.logical_rows {
            return Err(Error::shape_mismatch(
                "cublaslt matmul inner dimension",
                vec![to_i64(
                    a_layout.logical_cols,
                    "cuda graph cuBLASLt lhs inner dimension",
                )?],
                vec![to_i64(
                    b_layout.logical_rows,
                    "cuda graph cuBLASLt rhs inner dimension",
                )?],
            ));
        }

        let rows = a_layout.logical_rows;
        let cols = b_layout.logical_cols;
        if a_layout.batch_count != b_layout.batch_count {
            return Err(Error::length_mismatch(
                "cublaslt matmul batch count",
                a_layout.batch_count,
                b_layout.batch_count,
            ));
        }

        let output_layout =
            CublasLtMatmulOutputLayout::create(output.layout(), rows, cols, a_layout.batch_count)?;
        if let Some(bias) = bias {
            validate_cublaslt_bias(output, bias)?;
        }

        Ok(Self {
            dtype_plan,
            output_layout,
            rows,
            cols,
        })
    }

    pub(crate) fn dtype_plan(&self) -> CublasLtMatmulDTypePlan {
        self.dtype_plan
    }

    pub(crate) fn output_layout(&self) -> &CublasLtMatmulOutputLayout {
        &self.output_layout
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    pub(crate) fn cols(&self) -> u64 {
        self.cols
    }
}
