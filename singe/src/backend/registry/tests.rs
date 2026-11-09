use crate::{
    backend::registry::{
        AliasBehavior, BackendExecution, CudaGraphProvider, OperandRole, Operation,
        OperationAttributeKind, OperationExecutionFamily, operation_schema, operation_support,
        registered_operations,
    },
    data_type::DataType,
};

#[test]
fn test_elementwise_helpers_come_from_operation_metadata() {
    assert!(Operation::Neg.is_unary_elementwise());
    assert!(Operation::Relu.is_unary_elementwise());

    assert!(Operation::Sub.is_binary_elementwise());
    assert!(Operation::Add.is_binary_elementwise());

    assert!(Operation::Scale.is_scalar_elementwise());
    assert!(!Operation::Clamp.is_scalar_elementwise());
    assert!(!Operation::MatMul.is_binary_elementwise());
    assert!(!Operation::Softmax.is_unary_elementwise());
}

#[test]
fn test_attribute_kind_comes_from_operation_metadata() {
    assert_eq!(
        Operation::Scale.attribute_kind(),
        OperationAttributeKind::Scalar
    );
    assert_eq!(
        Operation::Softmax.attribute_kind(),
        OperationAttributeKind::Dimension
    );
    assert_eq!(
        Operation::MaxPool2d.attribute_kind(),
        OperationAttributeKind::Pool2d
    );
    assert_eq!(
        Operation::BatchNorm.attribute_kind(),
        OperationAttributeKind::BatchNormalization
    );
    assert_eq!(
        Operation::QuantizedMatMul.attribute_kind(),
        OperationAttributeKind::None
    );
    assert_eq!(
        Operation::Add.attribute_kind(),
        OperationAttributeKind::None
    );
    assert_eq!(
        Operation::Embedding.attribute_kind(),
        OperationAttributeKind::None
    );
}

#[test]
fn test_operand_schema_comes_from_operation_metadata() {
    let add = operation_schema(Operation::Add);
    assert!(add.operands().accepts_count(2));
    assert_eq!(
        add.operands().roles_for_count(2),
        vec![OperandRole::Lhs, OperandRole::Rhs]
    );

    let concat = operation_schema(Operation::Concat);
    assert!(!concat.operands().accepts_count(0));
    assert!(concat.operands().accepts_count(3));
    assert_eq!(
        concat.operands().roles_for_count(3),
        vec![OperandRole::Input, OperandRole::Input, OperandRole::Input]
    );

    assert_eq!(
        operation_schema(Operation::Where)
            .operands()
            .roles_for_count(3),
        vec![OperandRole::Condition, OperandRole::Lhs, OperandRole::Rhs]
    );
    assert_eq!(
        operation_schema(Operation::Embedding)
            .operands()
            .roles_for_count(2),
        vec![OperandRole::Weight, OperandRole::Indices]
    );
    assert_eq!(
        operation_schema(Operation::ScaledDotProductAttention)
            .operands()
            .roles_for_count(3),
        vec![OperandRole::Input, OperandRole::Key, OperandRole::Value]
    );
}

#[test]
fn test_output_and_alias_schema_come_from_operation_metadata() {
    let argmax = operation_schema(Operation::ArgMax);
    assert_eq!(argmax.output_count(), 1);
    assert_eq!(
        argmax
            .single_output_role()
            .unwrap()
            .data_type(DataType::F32),
        DataType::I64
    );

    let add = operation_schema(Operation::Add);
    assert_eq!(add.output_count(), 1);
    assert_eq!(
        add.single_output_role().unwrap().data_type(DataType::F32),
        DataType::F32
    );
    assert_eq!(add.alias_behavior(), AliasBehavior::None);

    let reshape = operation_schema(Operation::Reshape);
    assert_eq!(reshape.alias_behavior(), AliasBehavior::View);
    assert_eq!(
        operation_schema(Operation::Contiguous).alias_behavior(),
        AliasBehavior::MayAliasInput
    );
    assert_eq!(
        operation_schema(Operation::Expand).alias_behavior(),
        AliasBehavior::View
    );
}

#[test]
fn test_cuda_eager_support_matches_implemented_eager_paths() {
    // Every registered operation has a CUDA eager path; attention composes its matmuls and softmax.
    for &operation in registered_operations() {
        assert!(
            operation_support(operation).supports(BackendExecution::CudaEager),
            "{operation:?} cuda eager support metadata is not truthful"
        );
    }
}

#[test]
fn test_cuda_graph_providers_are_real_lowering_providers() {
    for operation in [Operation::Relu, Operation::Add, Operation::Cast] {
        let support = operation_support(operation);
        assert!(support.supports(BackendExecution::CudaGraph));
        assert_eq!(
            support.cuda_graph_providers(),
            &[CudaGraphProvider::Kernel],
            "{operation:?}"
        );
    }
    let pooling = operation_support(Operation::MaxPool2d);
    assert_eq!(pooling.cuda_graph_providers(), &[CudaGraphProvider::Kernel]);
}

#[test]
fn test_execution_family_identifies_backend_ownership() {
    assert_eq!(
        operation_support(Operation::Transpose).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::MetadataOnly
    );
    assert_eq!(
        operation_support(Operation::Contiguous).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::LayoutMaterialization
    );
    assert_eq!(
        operation_support(Operation::CopyTo).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::StorageCopy
    );
    assert_eq!(
        operation_support(Operation::MatMul).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::BackendCompute
    );
    assert_eq!(
        operation_support(Operation::Add).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::BackendCompute
    );
    assert_eq!(
        operation_support(Operation::Relu).execution_family(BackendExecution::CudaEager),
        OperationExecutionFamily::BackendCompute
    );
    assert_eq!(
        operation_support(Operation::Relu).execution_family(BackendExecution::CudaGraph),
        OperationExecutionFamily::BackendCompute
    );
}
