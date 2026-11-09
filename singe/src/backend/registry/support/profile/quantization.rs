use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::{
                ELEMENT_TYPES, F32_TYPES, FP8_TYPES, QUANTIZED_I8_U8_TYPES,
                QUANTIZED_MATMUL_INPUT_TYPES, SCALED_MATMUL_OUTPUT_TYPES,
            },
            providers::{CUBLASLT_GRAPH_PROVIDER, KERNEL_GRAPH_PROVIDER, NO_CUDA_GRAPH_PROVIDER},
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        OperationSupportProfile::Cast => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::Quantize => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(F32_TYPES, QUANTIZED_I8_U8_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(F32_TYPES, QUANTIZED_I8_U8_TYPES)),
            compiled: Some(OperationDataTypes::new(F32_TYPES, QUANTIZED_I8_U8_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::Dequantize => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(QUANTIZED_I8_U8_TYPES, F32_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(QUANTIZED_I8_U8_TYPES, F32_TYPES)),
            compiled: Some(OperationDataTypes::new(QUANTIZED_I8_U8_TYPES, F32_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::QuantizedMatMul => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(
                QUANTIZED_MATMUL_INPUT_TYPES,
                F32_TYPES,
            )),
            cuda_eager: Some(OperationDataTypes::new(
                QUANTIZED_MATMUL_INPUT_TYPES,
                F32_TYPES,
            )),
            // Compiled graphs run its decomposition, a dequantization of each operand and their product.
            compiled: Some(OperationDataTypes::new(
                QUANTIZED_MATMUL_INPUT_TYPES,
                F32_TYPES,
            )),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: NO_CUDA_GRAPH_PROVIDER,
        },
        // CUDA multiplies on FP8 tensor cores through cuBLASLt, and both paths run the decomposition, a dequantized `F32` product, for what cuBLASLt cannot take.
        OperationSupportProfile::ScaledMatMul => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(
                FP8_TYPES,
                SCALED_MATMUL_OUTPUT_TYPES,
            )),
            cuda_eager: Some(OperationDataTypes::new(
                FP8_TYPES,
                SCALED_MATMUL_OUTPUT_TYPES,
            )),
            compiled: Some(OperationDataTypes::new(
                FP8_TYPES,
                SCALED_MATMUL_OUTPUT_TYPES,
            )),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: CUBLASLT_GRAPH_PROVIDER,
        },
        _ => unreachable!("non-quantization support profile routed to quantization support"),
    }
}
