mod algebra;
mod built_in;
mod custom;
mod data_types;
mod neural;
mod providers;
mod quantization;
mod reduction;
mod shape;

use crate::{
    backend::registry::{
        CudaGraphProvider, LayoutSupport, Operation,
        support::{OperationDataTypes, OperationSupport},
    },
    data_type::DataType,
};

pub fn operation_support(
    operation: Operation,
    profile: OperationSupportProfile,
) -> OperationSupport {
    match profile {
        OperationSupportProfile::DecomposedInGraphs
        | OperationSupportProfile::ElementWise
        | OperationSupportProfile::Comparison
        | OperationSupportProfile::Softmax => built_in::operation_support(operation, profile),
        OperationSupportProfile::MatMul | OperationSupportProfile::Contraction => {
            algebra::operation_support(operation, profile)
        }
        OperationSupportProfile::Cast
        | OperationSupportProfile::Quantize
        | OperationSupportProfile::Dequantize
        | OperationSupportProfile::QuantizedMatMul
        | OperationSupportProfile::ScaledMatMul => {
            quantization::operation_support(operation, profile)
        }
        OperationSupportProfile::AliasAnyStrided
        | OperationSupportProfile::AllDataTypesEagerOnly
        | OperationSupportProfile::Random
        | OperationSupportProfile::AllDataTypes
        | OperationSupportProfile::Indexing
        | OperationSupportProfile::ContiguousAlias
        | OperationSupportProfile::Materialization => shape::operation_support(operation, profile),
        OperationSupportProfile::Reduction
        | OperationSupportProfile::ArgReduceMaterialized
        | OperationSupportProfile::Sort
        | OperationSupportProfile::Scan
        | OperationSupportProfile::TopKValues => reduction::operation_support(operation, profile),
        OperationSupportProfile::ConvolutionCudnn
        | OperationSupportProfile::Convolution3d
        | OperationSupportProfile::Normalization
        | OperationSupportProfile::Pooling
        | OperationSupportProfile::Attention
        | OperationSupportProfile::PagedAttention
        | OperationSupportProfile::PagedMlaAttention
        | OperationSupportProfile::GroupedMatMul
        | OperationSupportProfile::StateRecurrence
        | OperationSupportProfile::Fourier => neural::operation_support(operation, profile),
        OperationSupportProfile::Custom => custom::operation_support(operation),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationSupportProfile {
    Random,
    AllDataTypesEagerOnly,
    AllDataTypes,
    DecomposedInGraphs,
    ElementWise,
    Comparison,
    MatMul,
    Softmax,
    Reduction,
    Contraction,
    Cast,
    Quantize,
    Dequantize,
    QuantizedMatMul,
    ScaledMatMul,
    AliasAnyStrided,
    ContiguousAlias,
    Materialization,
    Indexing,
    ArgReduceMaterialized,
    TopKValues,
    Sort,
    Scan,
    ConvolutionCudnn,
    Convolution3d,
    Normalization,
    Pooling,
    Attention,
    PagedAttention,
    PagedMlaAttention,
    GroupedMatMul,
    StateRecurrence,
    Fourier,
    Custom,
}

pub fn support(
    operation: Operation,
    input_data_types: &'static [DataType],
    data_types: &'static [DataType],
    layout: LayoutSupport,
    cuda_graph_providers: &'static [CudaGraphProvider],
) -> OperationSupport {
    OperationSupport {
        operation,
        cpu_eager: Some(OperationDataTypes::new(input_data_types, data_types)),
        cuda_eager: Some(OperationDataTypes::new(input_data_types, data_types)),
        compiled: Some(OperationDataTypes::new(input_data_types, data_types)),
        layout,
        cuda_graph_providers,
    }
}
