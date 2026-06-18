use crate::{
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaFp8BackwardGradientTensors,
            SdpaFp8BackwardInputs, SdpaFp8Inputs, SdpaFp8OutputTensors, SdpaInputs,
            SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs, SdpaMxfp8Inputs,
            SdpaMxfp8OutputTensors, SdpaOutputTensors,
        },
        graph::Graph,
        infer::reduce_last_axis,
        operation::{
            AttentionBackwardConfig, AttentionConfig, AttentionDropoutConfig,
            AttentionImplementation, BlockScaleDequantizeConfig, DirectSdpaForwardConfig,
            HeuristicMode, Operation, PointwiseOperation, SdpaConfig, SdpaScoreModifiers,
            SdpaScoreSubgraph, SoftmaxConfig,
        },
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    scalar::ScalarValue,
    tensor::{Shape, TensorSpec},
    testing::setup_context,
    version,
};

fn is_expected_compile_status(error: &Error) -> bool {
    match error {
        Error::NoAvailableEngines => true,
        Error::Cudnn { code, .. } => {
            *code == Status::NotSupported
                || *code == Status::NotSupportedRuntimePrerequisiteMissing
                || *code == Status::InternalErrorUnexpectedValue
                || *code == Status::BadParamAttributeType
        }
        _ => false,
    }
}

mod backward;
mod forward;
mod masks_support;
mod quantized;
