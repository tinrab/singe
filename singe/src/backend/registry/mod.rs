mod attributes;
mod infer;
mod operation_schema;
mod request;
mod schema;
mod support;
mod table;

use self::schema::{
    ASSIGN_OPERANDS, ATTENTION_LSE_OUTPUTS, BATCH_NORM_OPERANDS, BINARY_OPERANDS,
    CAUSAL_CONV1D_OPERANDS, CAUSAL_CONV1D_STATE_OPERANDS, CONTRACTION_OPERANDS, CONV2D_OPERANDS,
    EMBEDDING_OPERANDS, GATED_DELTA_RULE_OPERANDS, GROUPED_MATMUL_OPERANDS, INDEX_LEADING_OPERANDS,
    INDEX_OUTPUT, INDEX_PUT_LEADING_OPERANDS, INDEX_SELECT_OPERANDS, INDEX_WRITE_LEADING_OPERANDS,
    LAYER_NORM_OPERANDS, LERP_OPERANDS, LSE_OUTPUT, MASK_OUTPUT, MATMUL_OPERANDS, NO_OUTPUTS,
    PAGED_ATTENTION_OPERANDS, PAGED_MLA_ATTENTION_OPERANDS, QUANTIZE_OPERANDS,
    QUANTIZED_MATMUL_OPERANDS, RANDOM_OPERANDS, ROTARY_EMBEDDING_OPERANDS, SAMPLING_OPERANDS,
    SCALED_DOT_PRODUCT_ATTENTION_OPERANDS, SCALED_MATMUL_OPERANDS, SCATTER_OPERANDS,
    TERNARY_WHERE_OPERANDS, TOPK_OUTPUTS, UNARY_OPERANDS, VALUE_OUTPUT,
};

pub use self::{
    attributes::{
        BatchNormalizationAttributes, BitHash, ClampAttributes, ContractionAttributes,
        Conv2dAttributes, Conv3dAttributes, FlattenAttributes, IndexPutAttributes,
        LayerNormalizationAttributes, OperationAttributeKind, OperationAttributes, PackedFormat,
        PadAttributes, Pool2dAttributes, QuantizationAttributes, RandomAttributes,
        ReductionAttributes, ScaledDotProductAttentionAttributes, SliceAttributes, SortAttributes,
        StridedAttributes, TopKAttributes, TransposeAttributes,
    },
    operation_schema::{OperationSchema, operation_schema, operation_schema_for_attributes},
    request::{
        BinaryOperationInference, BinaryOperationRequest, CheckedOperationRequest,
        OperationInference, OperationRequest, ResultAlias,
    },
    schema::{
        AliasBehavior, DeclaredOutputSchema, OperandRole, OperandSchema, OperationExecutorFamily,
        OutputRole,
    },
    support::{
        BackendExecution, CublasFeature, CublasLtFeature, CudaGraphProvider, CudaGraphProviderSet,
        CudnnFeature, CufftFeature, CutensorFeature, LayoutSupport, LibraryFeature,
        OperationExecutionFamily, OperationSupport, operation_support, operation_support_table,
    },
    table::{Operation, operation_name},
};

/// Returns every operation known to the backend registry.
#[must_use]
pub fn registered_operations() -> &'static [Operation] {
    table::REGISTERED_OPERATIONS
}

#[cfg(test)]
mod tests;
