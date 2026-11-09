use crate::backend::{
    cuda,
    registry::{BackendExecution, Operation, OperationExecutionFamily},
};

#[test]
fn test_cuda_execution_family_facade_matches_registry_boundaries() {
    assert_eq!(
        cuda::eager_execution_family(Operation::Transpose),
        OperationExecutionFamily::MetadataOnly
    );
    assert_eq!(
        cuda::eager_execution_family(Operation::Contiguous),
        OperationExecutionFamily::LayoutMaterialization
    );
    assert_eq!(
        cuda::eager_execution_family(Operation::CopyTo),
        OperationExecutionFamily::StorageCopy
    );
    assert_eq!(
        cuda::eager_execution_family(Operation::MatMul),
        OperationExecutionFamily::BackendCompute
    );
    assert_eq!(
        cuda::eager_execution_family(Operation::Relu),
        OperationExecutionFamily::BackendCompute
    );
    assert_eq!(
        cuda::operation_execution_family(Operation::Relu, BackendExecution::CudaGraph),
        OperationExecutionFamily::BackendCompute
    );
}
