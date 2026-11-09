use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::{ELEMENT_TYPES, FLOAT_TYPES},
            providers::{CUTENSOR_GRAPH_PROVIDER, KERNEL_CUDNN_CUBLAS_GRAPH_PROVIDERS},
            support,
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        OperationSupportProfile::MatMul => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_CUDNN_CUBLAS_GRAPH_PROVIDERS,
        },
        // Native contractions compute `F32`, and the other types run the contraction's decomposition.
        OperationSupportProfile::Contraction => support(
            operation,
            FLOAT_TYPES,
            FLOAT_TYPES,
            LayoutSupport::AnyStrided,
            CUTENSOR_GRAPH_PROVIDER,
        ),
        _ => unreachable!("non-algebra support profile routed to algebra support"),
    }
}
