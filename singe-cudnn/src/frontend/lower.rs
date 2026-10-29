use std::collections::{BTreeMap, HashSet};

use crate::{
    backend::{
        descriptor::BackendDescriptor,
        execution::{
            attention::{
                DiagonalBandMaskOperation, SdpaBackwardOperation, SdpaForwardOperation,
                SdpaSubgraph, SoftmaxOperation,
            },
            block_scale::{BlockScaleDequantizeOperation, BlockScaleQuantizeOperation},
            convolution::{
                ConvolutionBackwardDataOperation, ConvolutionBackwardFilterOperation,
                ConvolutionForwardOperation,
            },
            matmul::MatmulOperation,
            moe::{MoeGroupedMatmulBackwardOperation, MoeGroupedMatmulOperation},
            normalization::{
                BatchNormalizationBackwardWeightsOperation,
                BatchNormalizationFinalizeStatsOperation, GenStatsOperation,
                NormalizationBackwardOperation, NormalizationForwardOperation,
            },
            paged_cache::PagedCacheLoadOperation,
            pointwise::PointwiseOperation as BackendPointwiseOperation,
            reduction::ReductionOperation as BackendReductionOperation,
            resample::{ResampleBackwardOperation, ResampleForwardOperation},
            rng::RngOperation,
            rope::{RopeBackwardOperation, RopeForwardOperation},
            tensor_ops::{ConcatOperation, ReshapeOperation, SliceOperation, TransposeOperation},
        },
        tensor::{Tensor, TensorId},
    },
    context::Context,
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphLayout, PreparedGraph, TensorRecord, ValidatedGraph},
    },
};

#[derive(Debug)]
pub struct LoweredGraph {
    pub(crate) layout: GraphLayout,
    pub operations: Vec<LoweredOperation>,
    pub backend_tensors: BTreeMap<TensorId, Tensor>,
    pub retained_subgraphs: Vec<PreparedGraph>,
}

impl LoweredGraph {
    pub(crate) fn lower(
        graph: &Graph,
        layout: GraphLayout,
        subgraphs: &BTreeMap<usize, SdpaSubgraph<'_>>,
        mut reducers: BTreeMap<usize, ValidatedGraph>,
        cudnn_context: Option<&Context>,
    ) -> Result<Self> {
        let mut backend_tensors = BTreeMap::new();
        let mut active_tensors = HashSet::new();
        for &tensor_id in &layout.native_tensor_ids {
            lower_tensor_at(
                tensor_id,
                &layout.tensors,
                &mut backend_tensors,
                &mut active_tensors,
            )?;
        }

        let context = LoweringContext {
            backend_tensors: &backend_tensors,
            data_type_policy: graph.data_type_policy(),
            cudnn_context,
        };
        let mut operations = Vec::with_capacity(graph.operations.len());
        for (index, operation) in graph.operations.iter().enumerate() {
            operations.push(operation.lower(
                &context,
                subgraphs.get(&index).copied(),
                reducers.remove(&index),
            )?);
        }

        Ok(Self {
            layout,
            backend_tensors,
            operations,
            retained_subgraphs: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoweringContext<'a> {
    backend_tensors: &'a BTreeMap<TensorId, Tensor>,
    data_type_policy: DataTypePolicy,
    cudnn_context: Option<&'a Context>,
}

impl LoweringContext<'_> {
    pub(crate) fn backend_tensors(&self) -> &BTreeMap<TensorId, Tensor> {
        self.backend_tensors
    }

    pub(crate) fn compute_data_type(&self, requested: Option<DataType>) -> DataType {
        self.compute_data_type_with_fallback(requested, DataType::F32)
    }

    pub(crate) fn compute_data_type_with_fallback(
        &self,
        requested: Option<DataType>,
        fallback: DataType,
    ) -> DataType {
        requested
            .or_else(|| self.data_type_policy.compute())
            .unwrap_or(fallback)
    }

    pub(crate) fn cudnn_context(&self) -> Option<&Context> {
        self.cudnn_context
    }
}

#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "lowered descriptors are built once per graph; boxing variants would only complicate matching"
)]
pub enum LoweredOperation {
    Matmul(MatmulOperation),
    ConvolutionForward(ConvolutionForwardOperation),
    ConvolutionBackwardData(ConvolutionBackwardDataOperation),
    ConvolutionBackwardFilter(ConvolutionBackwardFilterOperation),
    Pointwise(BackendPointwiseOperation),
    Reduction(BackendReductionOperation),
    CustomReduction {
        operation: BackendReductionOperation,
        /// Keeps the reducer body's native descriptors alive with the reduction descriptor.
        _body: Box<PreparedGraph>,
    },
    Reshape(ReshapeOperation),
    Slice(SliceOperation),
    Transpose(TransposeOperation),
    RopeForward(RopeForwardOperation),
    RopeBackward(RopeBackwardOperation),
    Concat(ConcatOperation),
    Resample(ResampleForwardOperation),
    ResampleBackward(ResampleBackwardOperation),
    Rng(RngOperation),
    GenStats(GenStatsOperation),
    BlockScaleQuantize(BlockScaleQuantizeOperation),
    BlockScaleDequantize(BlockScaleDequantizeOperation),
    LayerNormalization(NormalizationForwardOperation),
    RmsNormalization(NormalizationForwardOperation),
    LayerNormalizationBackward(NormalizationBackwardOperation),
    RmsNormalizationBackward(NormalizationBackwardOperation),
    InstanceNormalization(NormalizationForwardOperation),
    InstanceNormalizationBackward(NormalizationBackwardOperation),
    BatchNormalization(NormalizationForwardOperation),
    BatchNormalizationInference(NormalizationForwardOperation),
    BatchNormalizationFinalize(BatchNormalizationFinalizeStatsOperation),
    BatchNormalizationBackward(NormalizationBackwardOperation),
    BatchNormalizationBackwardWeights(BatchNormalizationBackwardWeightsOperation),
    AdaptiveLayerNormalization(NormalizationForwardOperation),
    AdaptiveLayerNormalizationBackward(NormalizationBackwardOperation),
    PagedCacheLoad(PagedCacheLoadOperation),
    DiagonalBandMask(DiagonalBandMaskOperation),
    Softmax(SoftmaxOperation),
    SdpaForward(SdpaForwardOperation),
    SdpaBackward(SdpaBackwardOperation),
    MoeGroupedMatmul(MoeGroupedMatmulOperation),
    MoeGroupedMatmulBackward(MoeGroupedMatmulBackwardOperation),
}

impl LoweredOperation {
    pub fn descriptor(&self) -> &BackendDescriptor {
        match self {
            Self::Matmul(op) => op.descriptor(),
            Self::ConvolutionForward(op) => op.descriptor(),
            Self::ConvolutionBackwardData(op) => op.descriptor(),
            Self::ConvolutionBackwardFilter(op) => op.descriptor(),
            Self::Pointwise(op) => op.descriptor(),
            Self::Reduction(op) => op.descriptor(),
            Self::CustomReduction { operation, .. } => operation.descriptor(),
            Self::Reshape(op) => op.descriptor(),
            Self::Slice(op) => op.descriptor(),
            Self::Transpose(op) => op.descriptor(),
            Self::RopeForward(op) => op.descriptor(),
            Self::RopeBackward(op) => op.descriptor(),
            Self::Concat(op) => op.descriptor(),
            Self::Resample(op) => op.descriptor(),
            Self::ResampleBackward(op) => op.descriptor(),
            Self::Rng(op) => op.descriptor(),
            Self::GenStats(op) => op.descriptor(),
            Self::BlockScaleQuantize(op) => op.descriptor(),
            Self::BlockScaleDequantize(op) => op.descriptor(),
            Self::LayerNormalization(op)
            | Self::RmsNormalization(op)
            | Self::InstanceNormalization(op)
            | Self::BatchNormalization(op)
            | Self::BatchNormalizationInference(op)
            | Self::AdaptiveLayerNormalization(op) => op.descriptor(),
            Self::LayerNormalizationBackward(op)
            | Self::RmsNormalizationBackward(op)
            | Self::InstanceNormalizationBackward(op)
            | Self::BatchNormalizationBackward(op)
            | Self::AdaptiveLayerNormalizationBackward(op) => op.descriptor(),
            Self::BatchNormalizationFinalize(op) => op.descriptor(),
            Self::BatchNormalizationBackwardWeights(op) => op.descriptor(),
            Self::PagedCacheLoad(op) => op.descriptor(),
            Self::DiagonalBandMask(op) => op.descriptor(),
            Self::Softmax(op) => op.descriptor(),
            Self::SdpaForward(op) => op.descriptor(),
            Self::SdpaBackward(op) => op.descriptor(),
            Self::MoeGroupedMatmul(op) => op.descriptor(),
            Self::MoeGroupedMatmulBackward(op) => op.descriptor(),
        }
    }
}

fn lower_tensor_at(
    tensor_id: TensorId,
    tensors: &BTreeMap<TensorId, TensorRecord>,
    lowered_tensors: &mut BTreeMap<TensorId, Tensor>,
    active_tensors: &mut HashSet<TensorId>,
) -> Result<()> {
    if lowered_tensors.contains_key(&tensor_id) {
        return Ok(());
    }
    if !active_tensors.insert(tensor_id) {
        return Err(Error::DescriptorMismatch {
            name: "tensor ragged offset cycle".into(),
        });
    }

    let record = tensors
        .get(&tensor_id)
        .ok_or(FrontendError::TensorNotFound(tensor_id))?;
    let tensor = &record.tensor;
    if let Some(ragged_offset) = tensor.ragged_offset {
        lower_tensor_at(ragged_offset, tensors, lowered_tensors, active_tensors)?;
    }

    let ragged_offset = tensor
        .ragged_offset
        .map(|id| tensor_at(lowered_tensors, id))
        .transpose()?;
    let lowered = Tensor::create_with_spec(record.id.into(), tensor, ragged_offset)?;
    lowered_tensors.insert(tensor_id, lowered);
    active_tensors.remove(&tensor_id);
    Ok(())
}

pub fn tensor_at(tensors: &BTreeMap<TensorId, Tensor>, tensor: TensorId) -> Result<&Tensor> {
    tensors
        .get(&tensor)
        .ok_or(FrontendError::TensorNotFound(tensor).into())
}

pub fn optional_tensor_at(
    tensors: &BTreeMap<TensorId, Tensor>,
    tensor: Option<TensorId>,
) -> Result<Option<&Tensor>> {
    tensor.map(|tensor| tensor_at(tensors, tensor)).transpose()
}

pub fn tensor_slice_at<'a>(
    tensors: &'a BTreeMap<TensorId, Tensor>,
    tensor_ids: &[TensorId],
) -> Result<Vec<&'a Tensor>> {
    tensor_ids
        .iter()
        .map(|tensor_id| tensor_at(tensors, *tensor_id))
        .collect()
}
