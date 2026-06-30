use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    frontend::operation::{
        AttentionPrimitiveOperation, BlockScaleOperation, ConvolutionOperation, GenStatsOperation,
        MatmulOperation, MoeGroupedMatmulOperation, NormalizationOperation, PagedCacheLoadSpec,
        PointwiseOperation, RandomNumberGeneratorOperation, ReductionOperation, ResampleOperation,
        SdpaOperation, TensorOperation,
    },
    frontend::{
        lower::{LoweringContext, LoweringOutput},
        operation::FrontendOperationTensors,
    },
    tensor::TensorId,
};

macro_rules! frontend_operation_variants {
    ($macro:ident) => {
        $macro! {
            Matmul(MatmulOperation) => output,
            Convolution(ConvolutionOperation) => operation,
            Pointwise(PointwiseOperation) => operation,
            Reduction(ReductionOperation) => operation,
            Tensor(TensorOperation) => output,
            Resample(ResampleOperation) => operation,
            RandomNumberGenerator(RandomNumberGeneratorOperation) => operation,
            GenStats(GenStatsOperation) => operation,
            BlockScale(BlockScaleOperation) => operation,
            Normalization(NormalizationOperation) => operation,
            PagedCacheLoad(PagedCacheLoadSpec) => operation,
            AttentionPrimitive(AttentionPrimitiveOperation) => operation,
            Sdpa(SdpaOperation) => sdpa,
            MoeGroupedMatmul(MoeGroupedMatmulOperation) => operation,
        }
    };
}

macro_rules! define_operation_enum {
    ($($variant:ident($operation:ty) => $lowering:ident,)*) => {
        /// Serializable frontend operation node.
        ///
        /// Each variant records tensor IDs and operation-specific attributes for one
        /// node in the declarative cuDNN frontend graph. The graph is later lowered into
        /// backend descriptors and planned with cuDNN engines.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(tag = "kind", content = "operation")]
        #[non_exhaustive]
        pub enum Operation {
            $($variant($operation),)*
        }
    };
}

macro_rules! impl_operation_tensors {
    ($($variant:ident($operation:ty) => $lowering:ident,)*) => {
        impl FrontendOperationTensors for Operation {
            fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
                match self {
                    $(Operation::$variant(op) => op.append_tensor_ids(tensors),)*
                }
            }
        }
    };
}

macro_rules! lower_operation {
    (output, $op:expr, $context:expr) => {
        $op.lower($context)
    };
    (operation, $op:expr, $context:expr) => {
        Ok(LoweringOutput::Operation($op.lower($context)?))
    };
    (sdpa, $op:expr, $context:expr) => {
        Ok(LoweringOutput::Operation($op.lower($context, None)?))
    };
}

macro_rules! impl_operation_lowering {
    ($($variant:ident($operation:ty) => $lowering:ident,)*) => {
        impl Operation {
            pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweringOutput> {
                match self {
                    $(Operation::$variant(op) => lower_operation!($lowering, op, context),)*
                }
            }
        }
    };
}

frontend_operation_variants!(define_operation_enum);
frontend_operation_variants!(impl_operation_tensors);
frontend_operation_variants!(impl_operation_lowering);
