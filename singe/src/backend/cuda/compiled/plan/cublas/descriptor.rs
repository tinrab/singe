use singe_cublas::{
    lt::{
        descriptor::MatrixLayout,
        matmul::MatmulDescriptor,
        types::{Epilogue, Order},
    },
    types::ComputeType,
};

use singe_cuda::{data_type::DataType as CudaDataType, view::DeviceInputView};

use crate::{
    backend::cuda::{
        compiled::plan::cublas::{
            algorithm::CublasLtMatmulDTypePlan,
            layout::{CublasLtMatmulOutputLayout, CublasLtMatrixLayout},
        },
        device_views::tensor_input_view,
        graph_value::CudaGraphValue,
    },
    data_type::DataType,
    error::Result,
    utility::to_i64,
};

pub struct CublasLtMatmulDescriptors<'a> {
    pub(crate) descriptor: MatmulDescriptor,
    pub(crate) a_layout: MatrixLayout,
    pub(crate) b_layout: MatrixLayout,
    pub(crate) output_layout: MatrixLayout,
    _bias_memory: Option<DeviceInputView<'a, f32>>,
}

impl<'a> CublasLtMatmulDescriptors<'a> {
    pub(crate) fn create(
        dtype_plan: CublasLtMatmulDTypePlan,
        compute_type: ComputeType,
        a_layout: &CublasLtMatrixLayout,
        b_layout: &CublasLtMatrixLayout,
        epilogue: Epilogue,
        bias: Option<&'a CudaGraphValue>,
        output_layout: &CublasLtMatmulOutputLayout,
        rows: u64,
        cols: u64,
    ) -> Result<Self> {
        let mut descriptor = MatmulDescriptor::create(compute_type, dtype_plan.scale_type)?;
        descriptor.set_transpose_a(a_layout.operation)?;
        descriptor.set_transpose_b(b_layout.operation)?;
        descriptor.set_epilogue(epilogue)?;
        let bias_memory = if let Some(bias) = bias {
            let memory = tensor_input_view::<f32>(bias, DataType::F32, "cuBLASLt bias")?;
            descriptor.set_bias_data_type(CudaDataType::F32)?;
            descriptor.set_bias_pointer(&memory)?;
            Some(memory)
        } else {
            None
        };

        let mut a_descriptor = MatrixLayout::create(
            dtype_plan.cuda_data_type,
            a_layout.descriptor_rows,
            a_layout.descriptor_cols,
            a_layout.leading_dimension,
        )?;
        a_descriptor.set_order(Order::RowMajor)?;
        a_layout.configure_descriptor(&mut a_descriptor)?;

        let mut b_descriptor = MatrixLayout::create(
            dtype_plan.cuda_data_type,
            b_layout.descriptor_rows,
            b_layout.descriptor_cols,
            b_layout.leading_dimension,
        )?;
        b_descriptor.set_order(Order::RowMajor)?;
        b_layout.configure_descriptor(&mut b_descriptor)?;

        let mut output_descriptor = MatrixLayout::create(
            dtype_plan.cuda_data_type,
            rows,
            cols,
            to_i64(cols, "cuda graph cuBLASLt output leading dimension")?,
        )?;
        output_descriptor.set_order(Order::RowMajor)?;
        output_layout.configure_descriptor(&mut output_descriptor)?;

        Ok(Self {
            descriptor,
            a_layout: a_descriptor,
            b_layout: b_descriptor,
            output_layout: output_descriptor,
            _bias_memory: bias_memory,
        })
    }
}
