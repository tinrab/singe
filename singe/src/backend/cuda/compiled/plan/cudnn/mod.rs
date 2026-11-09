mod attention;
mod conv2d;
mod matmul;
mod patterns;

use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

pub use self::{
    attention::{CudnnPagedValues, build_cudnn_attention_graph, build_cudnn_paged_attention_graph},
    conv2d::{
        CudnnConv2dGeometry, build_cudnn_conv2d_bias_relu_graph, build_cudnn_convolution_graph,
    },
    matmul::{
        build_cudnn_matmul_add_graph, build_cudnn_matmul_add_relu_graph, build_cudnn_matmul_graph,
    },
    patterns::{
        CudnnConv2dBiasReluPattern, CudnnMatmulAddPattern, CudnnMatmulAddReluPattern,
        cudnn_conv2d_bias_relu_pattern, cudnn_matmul_activation_pattern, cudnn_matmul_add_pattern,
        cudnn_matmul_add_relu_pattern,
    },
};
use crate::{
    backend::{
        cuda::{
            compiled::plan::lowering::LoweringProvider, cudnn_backend::CudnnBackendTensorMetadata,
            graph_binding::validate_graph_node_value_operand,
        },
        registry::{CudaGraphProvider, Operation, operation_support},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    storage::cuda::map_cuda_allocation_error,
    utility::to_i64,
};

/// CUDA graph lowering provider for cuDNN graph templates and supported library primitives.
#[derive(Debug, Clone, Copy)]
pub struct CudnnBackendProvider;

impl LoweringProvider for CudnnBackendProvider {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::CudnnBackend
    }

    fn cost(&self) -> u32 {
        10
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::CudnnBackend)
    }
}

pub fn ensure_cudnn_value_shape(actual: &[i64], expected: &[i64], name: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::shape_mismatch(
            name.to_string(),
            expected.to_vec(),
            actual.to_vec(),
        ));
    }
    Ok(())
}

pub fn validate_cudnn_same_shape_unary(
    ir: &Graph,
    id: ValueId,
    input: ValueId,
    name: &str,
) -> Result<()> {
    validate_graph_node_value_operand(ir, id, input)?;
    let output_value = id;
    let input_value = input;
    let output_shape = ir.required_value_shape(output_value)?;
    let input_shape = ir.required_value_shape(input_value)?;
    ensure_cudnn_value_shape(output_shape.dimensions(), input_shape.dimensions(), name)?;
    Ok(())
}

pub fn cudnn_tensor_metadata_for_value(
    ir: &Graph,
    value: ValueId,
) -> Result<CudnnBackendTensorMetadata> {
    Ok(CudnnBackendTensorMetadata::new(
        ir.required_value_data_type(value)?,
        ir.required_value_shape(value)?,
        ir.required_value_layout(value)?,
    ))
}

pub fn allocate_cudnn_workspace(
    context: &Arc<CudaContext>,
    workspace_size: usize,
) -> Result<DeviceMemory<MaybeUninit<u8>>> {
    let shape = vec![to_i64(workspace_size, "cudnn workspace byte size")?];
    let device = Device::Cuda(context.device().id());
    DeviceMemory::<u8>::create_uninit(context, workspace_size).map_err(|error| {
        map_cuda_allocation_error(error, device, workspace_size, DataType::U8, Some(shape))
    })
}
