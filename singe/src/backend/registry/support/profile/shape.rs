use crate::backend::registry::{
    LayoutSupport, Operation, OperationSupport,
    support::{
        OperationDataTypes,
        profile::{
            OperationSupportProfile,
            data_types::{ALL_DTYPE_TYPES, CORE_TYPES, ELEMENT_TYPES, F32_F64_TYPES},
            providers::{
                ALIAS_GRAPH_PROVIDER, ALIAS_KERNEL_GRAPH_PROVIDERS, KERNEL_GRAPH_PROVIDER,
                NO_CUDA_GRAPH_PROVIDER,
            },
            support,
        },
    },
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        OperationSupportProfile::ContiguousAlias => support(
            operation,
            ALL_DTYPE_TYPES,
            ALL_DTYPE_TYPES,
            LayoutSupport::AnyStrided,
            if operation == Operation::Copy {
                KERNEL_GRAPH_PROVIDER
            } else {
                ALIAS_KERNEL_GRAPH_PROVIDERS
            },
        ),
        OperationSupportProfile::Materialization => OperationSupport {
            operation,
            // Head repeats of types other than `F32` compose from copies on the CPU.
            cpu_eager: Some(OperationDataTypes::new(
                if operation == Operation::Concat {
                    ALL_DTYPE_TYPES
                } else {
                    ELEMENT_TYPES
                },
                if operation == Operation::Concat {
                    ALL_DTYPE_TYPES
                } else {
                    ELEMENT_TYPES
                },
            )),
            // The generated copy kernel moves bytes, so every data type is supported on CUDA.
            cuda_eager: Some(OperationDataTypes::new(ALL_DTYPE_TYPES, ALL_DTYPE_TYPES)),
            compiled: Some(OperationDataTypes::new(ALL_DTYPE_TYPES, ALL_DTYPE_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::Random => OperationSupport {
            operation,
            // Sampling returns `I64` indices or a `Bool` mask, so outputs cover both.
            cpu_eager: Some(OperationDataTypes::new(F32_F64_TYPES, CORE_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(F32_F64_TYPES, CORE_TYPES)),
            compiled: Some(OperationDataTypes::new(F32_F64_TYPES, CORE_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        OperationSupportProfile::AllDataTypes => support(
            operation,
            ALL_DTYPE_TYPES,
            ALL_DTYPE_TYPES,
            LayoutSupport::AnyStrided,
            KERNEL_GRAPH_PROVIDER,
        ),
        OperationSupportProfile::AllDataTypesEagerOnly => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ALL_DTYPE_TYPES, ALL_DTYPE_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ALL_DTYPE_TYPES, ALL_DTYPE_TYPES)),
            compiled: None,
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: NO_CUDA_GRAPH_PROVIDER,
        },
        OperationSupportProfile::AliasAnyStrided => support(
            operation,
            ALL_DTYPE_TYPES,
            ALL_DTYPE_TYPES,
            LayoutSupport::AnyStrided,
            ALIAS_GRAPH_PROVIDER,
        ),
        // Indexing kernels copy elements' bits, so they hold every type kernels do.
        OperationSupportProfile::Indexing => OperationSupport {
            operation,
            cpu_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            cuda_eager: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            compiled: Some(OperationDataTypes::new(ELEMENT_TYPES, ELEMENT_TYPES)),
            layout: LayoutSupport::AnyStrided,
            cuda_graph_providers: KERNEL_GRAPH_PROVIDER,
        },
        _ => unreachable!("non-shape support profile routed to shape support"),
    }
}
