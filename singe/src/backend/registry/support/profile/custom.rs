use crate::{
    backend::registry::{
        LayoutSupport, Operation, OperationSupport,
        support::{OperationDataTypes, profile::providers::NO_CUDA_GRAPH_PROVIDER},
    },
    dispatch::{DispatchKey, has_backend_kernel},
};

/// Reports the declared dtypes of a custom operation on each backend with a registered kernel.
pub fn operation_support(operation: Operation) -> OperationSupport {
    let Operation::Custom(custom) = operation else {
        unreachable!("built-in operation routed to custom support")
    };
    let data_types = OperationDataTypes::new(custom.data_types(), custom.data_types());
    let on = |key| has_backend_kernel(operation, key).then_some(data_types);
    OperationSupport {
        operation,
        cpu_eager: on(DispatchKey::Cpu),
        cuda_eager: on(DispatchKey::Cuda),
        compiled: None,
        layout: LayoutSupport::AnyStrided,
        cuda_graph_providers: NO_CUDA_GRAPH_PROVIDER,
    }
}
