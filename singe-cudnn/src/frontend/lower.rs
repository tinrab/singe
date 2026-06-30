use std::collections::{BTreeMap, HashSet};

use crate::{
    descriptor::BackendDescriptor,
    error::{Error, Result},
    execution::{
        OperationGraph,
        advanced::{
            BlockScaleDequantizeOperation, BlockScaleQuantizeOperation, DiagonalBandMaskOperation,
            MatmulOperation, MoeGroupedMatmulOperation, PagedCacheLoadOperation,
            SdpaBackwardOperation, SdpaForwardOperation, SoftmaxOperation,
        },
        convolution::{
            ConvolutionBackwardDataOperation, ConvolutionBackwardFilterOperation,
            ConvolutionForwardOperation,
        },
        normalization::{
            BatchNormalizationBackwardWeightsOperation, BatchNormalizationFinalizeStatsOperation,
            GenStatsOperation, NormalizationBackwardOperation, NormalizationForwardOperation,
        },
        pointwise::PointwiseOperation as BackendPointwiseOperation,
        reduction::ReductionOperation as BackendReductionOperation,
        resample::{ResampleBackwardOperation, ResampleForwardOperation},
        rng::RngOperation,
        tensor_ops::{ConcatOperation, ReshapeOperation},
    },
    frontend::{
        graph::{Graph, TensorRecord},
        operation::{FrontendOperationTensors, Operation},
        plan::BindingReplacement,
    },
    tensor::{Tensor, TensorId, TensorSpec},
};

#[derive(Debug)]
pub struct LoweredGraph {
    pub tensors: BTreeMap<TensorId, TensorRecord>,
    pub backend_tensors: BTreeMap<TensorId, Tensor>,
    pub binding_template_ids: Vec<TensorId>,
    pub binding_replacements: Vec<BindingReplacement>,
    pub operations: Vec<LoweredOperation>,
}

impl LoweredGraph {
    pub fn lower(graph: &Graph) -> Result<Self> {
        let mut seen_ids = HashSet::new();
        let tensors = graph
            .tensors
            .iter()
            .map(|(&tensor_ref, tensor)| {
                let id = tensor.id.ok_or(Error::DescriptorMismatch {
                    name: "frontend tensor id".into(),
                })?;
                if !seen_ids.insert(id) {
                    return Err(Error::FrontendTensorIdConflict(id));
                }
                Ok((
                    tensor_ref,
                    TensorRecord {
                        tensor: tensor.clone(),
                        id,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        let mut backend_tensors = BTreeMap::new();
        let mut active_tensors = HashSet::new();
        for &tensor_ref in tensors.keys() {
            lower_tensor_at(
                tensor_ref,
                &tensors,
                &mut backend_tensors,
                &mut active_tensors,
            )?;
        }

        let mut used_tensors = graph
            .operations
            .iter()
            .flat_map(OperationSpec::tensor_ids)
            .chain(graph.alias_bindings.iter().map(|binding| binding.source))
            .collect::<HashSet<_>>();
        let seed_tensors = used_tensors.iter().copied().collect::<Vec<_>>();
        for tensor_ref in seed_tensors {
            collect_used_tensors(graph, tensor_ref, &mut used_tensors)?;
        }

        let binding_template_ids = tensors
            .iter()
            .filter(|(tensor_ref, record)| {
                !record.tensor.is_virtual && used_tensors.contains(tensor_ref)
            })
            .map(|(_, record)| record.id)
            .collect();

        let context = LoweringContext {
            tensor_records: &tensors,
            backend_tensors: &backend_tensors,
        };
        let mut binding_replacements = Vec::new();
        let mut operations = Vec::with_capacity(graph.operations.len());
        for operation in &graph.operations {
            match operation.lower(&context)? {
                LoweringOutput::Operation(operation) => operations.push(operation),
                LoweringOutput::BindingReplacement(replacement) => {
                    binding_replacements.push(replacement);
                }
                LoweringOutput::Expanded => {}
            }
        }

        Ok(Self {
            tensors,
            backend_tensors,
            binding_template_ids,
            binding_replacements,
            operations,
        })
    }

    pub(crate) fn tensor_record(&self, tensor: TensorId) -> Result<&TensorRecord> {
        self.tensors
            .get(&tensor)
            .ok_or(Error::FrontendTensorNotFound(tensor))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoweringContext<'a> {
    tensor_records: &'a BTreeMap<TensorId, TensorRecord>,
    backend_tensors: &'a BTreeMap<TensorId, Tensor>,
}

impl LoweringContext<'_> {
    pub(crate) fn backend_id(&self, tensor: TensorId) -> Result<TensorId> {
        Ok(self
            .tensor_records
            .get(&tensor)
            .ok_or(Error::FrontendTensorNotFound(tensor))?
            .id)
    }

    pub(crate) fn backend_tensors(&self) -> &BTreeMap<TensorId, Tensor> {
        self.backend_tensors
    }
}

#[derive(Debug)]
pub(crate) enum LoweringOutput {
    Operation(LoweredOperation),
    BindingReplacement(BindingReplacement),
    Expanded,
}

pub(crate) trait OperationSpec {
    fn tensor_ids(&self) -> Vec<TensorId>;
}

#[derive(Debug)]
pub(crate) enum LoweredOperation {
    Matmul(MatmulOperation),
    ConvolutionForward(ConvolutionForwardOperation),
    ConvolutionBackwardData(ConvolutionBackwardDataOperation),
    ConvolutionBackwardFilter(ConvolutionBackwardFilterOperation),
    Pointwise(BackendPointwiseOperation),
    Reduction(BackendReductionOperation),
    Reshape(ReshapeOperation),
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
    DbnWeight(BatchNormalizationBackwardWeightsOperation),
    AdaptiveLayerNormalization(NormalizationForwardOperation),
    AdaptiveLayerNormalizationBackward(NormalizationBackwardOperation),
    PagedCacheLoad(PagedCacheLoadOperation),
    DiagonalBandMask(DiagonalBandMaskOperation),
    Softmax(SoftmaxOperation),
    SdpaForward(SdpaForwardOperation),
    SdpaBackward(SdpaBackwardOperation),
    MoeGroupedMatmul(MoeGroupedMatmulOperation),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoweredSdpaSubgraph<'a> {
    pub(crate) operation_graph: &'a OperationGraph,
    pub(crate) input_id: TensorId,
    pub(crate) output_id: TensorId,
}

impl OperationSpec for Operation {
    fn tensor_ids(&self) -> Vec<TensorId> {
        let mut tensors = Vec::new();
        self.append_tensor_ids(&mut tensors);
        tensors
    }
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
            Self::Reshape(op) => op.descriptor(),
            Self::Concat(op) => op.descriptor(),
            Self::Resample(op) => op.descriptor(),
            Self::ResampleBackward(op) => op.descriptor(),
            Self::Rng(op) => op.descriptor(),
            Self::GenStats(op) => op.descriptor(),
            Self::BlockScaleQuantize(op) => op.descriptor(),
            Self::BlockScaleDequantize(op) => op.descriptor(),
            Self::LayerNormalization(op) => op.descriptor(),
            Self::RmsNormalization(op) => op.descriptor(),
            Self::LayerNormalizationBackward(op) => op.descriptor(),
            Self::RmsNormalizationBackward(op) => op.descriptor(),
            Self::InstanceNormalization(op) => op.descriptor(),
            Self::InstanceNormalizationBackward(op) => op.descriptor(),
            Self::BatchNormalization(op) => op.descriptor(),
            Self::BatchNormalizationInference(op) => op.descriptor(),
            Self::BatchNormalizationFinalize(op) => op.descriptor(),
            Self::BatchNormalizationBackward(op) => op.descriptor(),
            Self::DbnWeight(op) => op.descriptor(),
            Self::AdaptiveLayerNormalization(op) => op.descriptor(),
            Self::AdaptiveLayerNormalizationBackward(op) => op.descriptor(),
            Self::PagedCacheLoad(op) => op.descriptor(),
            Self::DiagonalBandMask(op) => op.descriptor(),
            Self::Softmax(op) => op.descriptor(),
            Self::SdpaForward(op) => op.descriptor(),
            Self::SdpaBackward(op) => op.descriptor(),
            Self::MoeGroupedMatmul(op) => op.descriptor(),
        }
    }
}

fn lower_tensor_at(
    tensor_ref: TensorId,
    tensors: &BTreeMap<TensorId, TensorRecord>,
    lowered_tensors: &mut BTreeMap<TensorId, Tensor>,
    active_tensors: &mut HashSet<TensorId>,
) -> Result<()> {
    if lowered_tensors.contains_key(&tensor_ref) {
        return Ok(());
    }
    if !active_tensors.insert(tensor_ref) {
        return Err(Error::DescriptorMismatch {
            name: "tensor ragged offset cycle".into(),
        });
    }

    let record = tensors
        .get(&tensor_ref)
        .ok_or(Error::FrontendTensorNotFound(tensor_ref))?;
    let tensor = &record.tensor;
    if let Some(ragged_offset) = tensor.ragged_offset {
        lower_tensor_at(ragged_offset, tensors, lowered_tensors, active_tensors)?;
    }

    let lowered = lower_tensor(record.id, tensor)?;
    lowered_tensors.insert(tensor_ref, lowered);
    active_tensors.remove(&tensor_ref);
    Ok(())
}

fn collect_used_tensors(
    graph: &Graph,
    tensor_ref: TensorId,
    used_tensors: &mut HashSet<TensorId>,
) -> Result<()> {
    let Some(tensor) = graph.tensors.get(&tensor_ref) else {
        return Err(Error::FrontendTensorNotFound(tensor_ref));
    };
    if let Some(ragged_offset) = tensor.ragged_offset
        && used_tensors.insert(ragged_offset)
    {
        collect_used_tensors(graph, ragged_offset, used_tensors)?;
    }
    Ok(())
}

fn lower_tensor(id: TensorId, tensor: &TensorSpec) -> Result<Tensor> {
    if tensor.vector_count.is_some() || tensor.vectorized_dimension.is_some() {
        let vectorized_dimension =
            tensor
                .vectorized_dimension
                .ok_or(Error::DescriptorMismatch {
                    name: "frontend tensor vectorization".into(),
                })?;
        let vectorized_dimension = if vectorized_dimension < 0 {
            tensor.shape.rank() + vectorized_dimension
        } else {
            vectorized_dimension
        };
        let vector_count = tensor.vector_count.ok_or(Error::DescriptorMismatch {
            name: "frontend tensor vectorization".into(),
        })?;
        Tensor::create_with_resolved_spec(
            id,
            tensor,
            Some(vectorized_dimension),
            Some(vector_count),
        )
    } else {
        Tensor::create_with_resolved_spec(id, tensor, None, None)
    }
}

pub(crate) fn tensor_at(tensors: &BTreeMap<TensorId, Tensor>, tensor: TensorId) -> Result<&Tensor> {
    tensors
        .get(&tensor)
        .ok_or(Error::FrontendTensorNotFound(tensor))
}

pub(crate) fn optional_tensor_at(
    tensors: &BTreeMap<TensorId, Tensor>,
    tensor: Option<TensorId>,
) -> Result<Option<&Tensor>> {
    tensor.map(|tensor| tensor_at(tensors, tensor)).transpose()
}

pub(crate) fn tensor_slice_at<'a>(
    tensors: &'a BTreeMap<TensorId, Tensor>,
    tensor_ids: &[TensorId],
) -> Result<Vec<&'a Tensor>> {
    tensor_ids
        .iter()
        .map(|tensor_id| tensor_at(tensors, *tensor_id))
        .collect()
}
