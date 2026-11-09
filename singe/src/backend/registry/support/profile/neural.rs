use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::FLOAT_TYPES,
            providers::{
                CUDNN_GRAPH_PROVIDER, CUFFT_GRAPH_PROVIDER, KERNEL_CUDNN_GRAPH_PROVIDERS,
                KERNEL_GRAPH_PROVIDER,
            },
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        OperationSupportProfile::ConvolutionCudnn
        | OperationSupportProfile::Normalization
        | OperationSupportProfile::Pooling => OperationSupport {
            operation,
            // Normalizations interpret their row kernels on the CPU for every floating type; convolutions and pooling compute `F32` and `F64`, widening half precision.
            cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: match profile {
                // Pooling is a generated window reduction, and so are depthwise convolutions; cuDNN runs the other convolutions.
                OperationSupportProfile::ConvolutionCudnn => KERNEL_CUDNN_GRAPH_PROVIDERS,
                _ => KERNEL_GRAPH_PROVIDER,
            },
        },
        // CUDA runs one cuDNN convolution, in eager calls and compiled graphs.
        OperationSupportProfile::Convolution3d => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: CUDNN_GRAPH_PROVIDER,
        },
        // CUDA runs attention as its matmuls and softmax: eager calls compose them, and graph passes decompose the operation before lowering.
        OperationSupportProfile::Attention => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            layout: LayoutSupport::RequiresContiguous,
            // Compiled graphs run half-precision attention through cuDNN when it is allowed and takes the options, else as one fused generated kernel, and decompose the rest before lowering.
            cuda_graph_providers: KERNEL_CUDNN_GRAPH_PROVIDERS,
        },
        // Both backends compute every floating type; CUDA uses tensor cores for half precision of the shapes they suit, and one general kernel otherwise.
        OperationSupportProfile::PagedAttention => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_CUDNN_GRAPH_PROVIDERS,
        },
        // Both backends run one kernel for every floating type, on tensor cores for half precision on CUDA.
        // cuDNN has no form of latent attention, whose scores sum two dot products.
        OperationSupportProfile::PagedMlaAttention | OperationSupportProfile::GroupedMatMul => {
            OperationSupport {
                operation,
                cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
                cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
                compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
                layout: LayoutSupport::AnyStrided,
                cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
            }
        }
        // Compiled graphs run long `F32` and `F64` transforms as captured cuFFT plans, and decompose the rest into a matmul with the basis.
        OperationSupportProfile::Fourier => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            compiled: Some(OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: CUFFT_GRAPH_PROVIDER,
        },
        _ => unreachable!("non-neural support profile routed to neural support"),
    }
}
