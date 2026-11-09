use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::{BOOL_TYPES, ELEMENT_TYPES, FLOAT_TYPES},
            providers::{KERNEL_GRAPH_PROVIDER, NO_CUDA_GRAPH_PROVIDER},
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        // Padding is its decomposition everywhere, which holds every type; rotary embeddings run `F32` kernels, widening half precision, and their decomposition otherwise; compiled graphs lower both decompositions.
        OperationSupportProfile::DecomposedInGraphs => {
            let types = if operation == Operation::Pad {
                OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)
            } else {
                OperationDataTypes::new(FLOAT_TYPES, FLOAT_TYPES)
            };
            OperationSupport {
                operation,
                cpu_eager: Some(types),
                cuda_eager: Some(types),
                compiled: Some(types),
                layout: LayoutSupport::AnyStrided,
                cuda_graph_providers: NO_CUDA_GRAPH_PROVIDER,
            }
        }
        // Every element-wise operation is one kernel IR expression, which CPU interprets and CUDA emits, so the three executions accept the same types; inference rejects the types an operation has no meaning for.
        OperationSupportProfile::ElementWise => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::Comparison => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, BOOL_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, BOOL_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, BOOL_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        // Inference holds empty tensors of every type, which their decomposition returns unchanged, and rejects other values that are not floating.
        OperationSupportProfile::Softmax => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::RequiresContiguous,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        _ => unreachable!("non-built-in support profile routed to built-in support"),
    }
}
