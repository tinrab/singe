use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::{ELEMENT_TYPES, I64_TYPES},
            providers::{KERNEL_GRAPH_PROVIDER, NO_CUDA_GRAPH_PROVIDER},
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        // Reductions are kernel IR row kernels, which the CPU interprets for the types its native kernels do not compute, and every backend ranks values as top-k's decomposition does, while compiled graphs select few values in a kernel.
        OperationSupportProfile::Reduction | OperationSupportProfile::TopKValues => {
            OperationSupport {
                operation,
                cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
                cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
                compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
                layout: LayoutSupport::AnyStrided,
                cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
            }
        }
        OperationSupportProfile::ArgReduceMaterialized => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            layout: LayoutSupport::MaterializesToContiguous,
            // Compiled graphs run top-k as its decomposition.
            cuda_graph_providers: if operation == Operation::TopKIndices {
                NO_CUDA_GRAPH_PROVIDER
            } else {
                KERNEL_GRAPH_PROVIDER
            },
        },
        OperationSupportProfile::Sort => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, I64_TYPES)),
            layout: LayoutSupport::MaterializesToContiguous,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::Scan => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::MaterializesToContiguous,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        _ => unreachable!("non-reduction support profile routed to reduction support"),
    }
}
