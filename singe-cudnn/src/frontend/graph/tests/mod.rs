use singe_cuda::{
    memory::DeviceMemory,
    types::{DevicePtr, f8e4m3, f8ue8m0},
};

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    behavior::BackendBehaviorNote,
    context::Context,
    convolution::ConvolutionMode,
    data_type::{DataType, DataTypeLike, f16},
    device::DeviceProperties,
    engine::EngineIndex,
    error::{Error, Result, Status},
    execution::{
        EngineConfig, KernelCache,
        advanced::MoeGroupedMatmulMode,
        resample::{PaddingMode, ResampleMode},
    },
    frontend::{
        bindings::Bindings,
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaInputs, SdpaOutputTensors,
        },
        graph::{
            BlockScaleMatmulInputs, DataTypePolicy, Graph, MatmulFp8Inputs,
            MatmulFp8OperationTensors, MatmulFp8QuantizeTensors, MatmulFp8Tensors,
            MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors,
            MoeGroupedMatmulInputs, MoeGroupedMatmulTensors, PreparedGraph,
        },
        infer::{infer_layer_norm_scale_bias_shape, infer_layer_norm_stats_shape},
        lower::LoweredGraph,
        operation::*,
        plan::{AutotuneConfig, BuildPlanPolicy, PlanCandidates},
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    scalar::ScalarValue,
    tensor::{Shape, TensorSpec},
    testing::setup_context,
    utility::to_i64,
    version,
};

fn is_expected_compile_status(error: &Error) -> bool {
    match error {
        Error::NoAvailableEngines => true,
        Error::Cudnn { code, .. } => {
            *code == Status::NotSupported
                || *code == Status::NotSupportedGraphPattern
                || *code == Status::NotSupportedRuntimePrerequisiteMissing
                || *code == Status::InternalErrorUnexpectedValue
                || *code == Status::BadParamAttributeType
        }
        Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }) => {
            *code == singe_cuda::error::Status::StubLibrary
                || *code == singe_cuda::error::Status::NoDevice
        }
        _ => false,
    }
}

fn is_expected_prepare_status(error: &Error) -> bool {
    is_expected_compile_status(error)
        || matches!(error, Error::FrontendKernelCacheRequiresDynamicShape)
        || matches!(error, Error::Cudnn { code, .. } if *code == Status::BadParam)
}

fn prepare_or_skip(graph: &Graph, ctx: &Context) -> Result<Option<PreparedGraph>> {
    match graph.prepare(ctx) {
        Ok(prepared) => Ok(Some(prepared)),
        Err(error) if is_expected_prepare_status(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

fn operation_descriptors_or_skip(
    prepared: &PreparedGraph,
) -> Result<Option<Vec<singe_cudnn_sys::cudnnBackendDescriptor_t>>> {
    match prepared
        .operation_graph
        .descriptor()
        .attribute_descriptor_slice(
            BackendAttributeName::OperationGraphOps,
            BackendAttributeType::BackendDescriptor,
        ) {
        Ok(operation_descriptors) => Ok(Some(operation_descriptors)),
        Err(error) if is_expected_prepare_status(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

fn is_expected_engine_query_status(error: &Error) -> bool {
    is_expected_compile_status(error)
        || matches!(
            error,
            Error::Cudnn { code, .. } if *code == Status::BadParamAttributeType
        )
}

mod elementwise_reduction_resample;
mod matmul_normalization;
mod operation_families;
mod planning_softmax;
mod sdpa_prepare_tensor;
mod shape_views;
