mod block_scale;
mod compile;
mod config;
mod convolution;
mod genstats;
mod index;
mod matmul;
mod moe;
mod normalization;
mod paged_cache;
mod pointwise;
mod reduction;
mod resample;
mod reshape;
mod rng;
mod serialization;
mod state;

pub use block_scale::BlockScaleQuantizeOutputs;
pub use genstats::GenStatsOutputs;
pub use matmul::{
    BlockScaleMatmulInputs, MatmulFp8Inputs, MatmulFp8OperationTensors, MatmulFp8QuantizeOutputs,
    MatmulFp8QuantizeTensors, MatmulFp8Tensors,
};
pub use moe::{
    MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors, MoeGroupedMatmulInputs,
    MoeGroupedMatmulTensors,
};
pub use normalization::{
    AdaptiveLayerNormalizationBackwardOutputs, AdaptiveLayerNormalizationOutputs,
    BatchNormalizationBackwardOutputs, BatchNormalizationFinalizeOutputs,
    BatchNormalizationOutputs, DbnWeightOutputs, InstanceNormalizationBackwardOutputs,
    InstanceNormalizationOutputs, LayerNormalizationBackwardOutputs, LayerNormalizationOutputs,
    RmsNormalizationBackwardOutputs, RmsNormalizationOutputs,
};
pub use resample::ResampleOutputs;
pub use state::{
    AliasBinding, DataTypePolicy, Graph, GraphConfig, PreparedGraph, RuntimeShapeConstraints,
};

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    data_type::DataType,
    device::DeviceProperties,
    error::{Error, Result},
    execution::KernelCache,
    tensor::{Shape, TensorId, TensorSpec},
};
use state::GraphMutationCheckpoint;
pub(crate) use state::TensorRecord;
impl Graph {
    fn allocate_tensor_id(&self) -> TensorId {
        let mut id = TensorId::generate();
        while self.tensors.contains_key(&id) {
            id = TensorId::generate();
        }
        id
    }

    fn insert_tensor_with_id(&mut self, mut tensor: TensorSpec, id: TensorId) -> TensorId {
        tensor.id = Some(id);
        self.tensors.insert(id, tensor);
        id
    }

    /// Inserts a tensor and returns the logical tensor ID.
    ///
    /// If the config has no ID, the graph assigns one. If the config has an
    /// explicit ID, that ID is preserved exactly.
    ///
    /// This panics if an explicit ID is already in use. Use
    /// [`Graph::insert_tensor`] or [`Graph::try_tensor`] when duplicate
    /// explicit IDs should be reported as recoverable errors, and use
    /// [`Graph::tensor_with_fresh_id`] when a fresh generated ID should be
    /// assigned regardless of the tensor spec.
    pub fn tensor(&mut self, tensor: TensorSpec) -> TensorId {
        let id = match tensor.id {
            Some(id) => {
                assert!(
                    !self.tensors.contains_key(&id),
                    "frontend tensor id {id:?} already exists"
                );
                id
            }
            None => self.allocate_tensor_id(),
        };
        self.insert_tensor_with_id(tensor, id)
    }

    /// Inserts a tensor using exactly the ID stored in the tensor spec.
    ///
    /// This is the checked insertion API for caller-provided tensor IDs. It
    /// returns [`Error::FrontendTensorIdConflict`] if the ID is already present.
    pub fn insert_tensor(&mut self, tensor: TensorSpec) -> Result<TensorId> {
        let id = tensor.id.ok_or_else(|| Error::FrontendTensorIdMissing {
            operation: "frontend tensor id".into(),
        })?;
        self.check_id(id)?;
        Ok(self.insert_tensor_with_id(tensor, id))
    }

    /// Inserts a tensor after assigning a fresh generated ID.
    ///
    /// This intentionally ignores any ID already present in the config and is
    /// useful for internally created virtual/intermediate tensors.
    pub fn tensor_with_fresh_id(&mut self, tensor: TensorSpec) -> TensorId {
        let id = self.allocate_tensor_id();
        self.insert_tensor_with_id(tensor, id)
    }

    pub fn try_tensor(&mut self, tensor: TensorSpec) -> Result<TensorId> {
        self.insert_tensor(tensor)
    }

    pub(crate) fn alias_tensor(
        &mut self,
        source: TensorId,
        tensor: TensorSpec,
        byte_offset: i64,
    ) -> Result<TensorId> {
        let source_tensor = self.tensor_config(source)?;
        if source_tensor.is_virtual {
            return Err(Error::FrontendAliasSourceNotBindable {
                source_id: source,
                reason: "source tensor is virtual".into(),
            });
        }
        if source_tensor.is_by_value {
            return Err(Error::FrontendAliasSourceNotBindable {
                source_id: source,
                reason: "source tensor is by value".into(),
            });
        }
        let target_id = tensor.id.unwrap_or_else(|| self.allocate_tensor_id());
        let element_size = tensor.data_type.size();
        if byte_offset % element_size != 0 {
            return Err(Error::FrontendAliasByteOffsetUnaligned {
                source_id: source,
                target_id,
                byte_offset,
                element_size,
            });
        }
        if source_tensor.data_type != tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: target_id,
                operation: "alias target".into(),
                expected: source_tensor.data_type,
                actual: tensor.data_type,
            });
        }
        validate_alias_byte_range(source, source_tensor, target_id, &tensor, byte_offset)?;
        let target = self.try_tensor(tensor.with_id(target_id))?;
        self.alias_bindings.push(AliasBinding {
            source,
            target,
            byte_offset,
        });
        Ok(target)
    }

    pub fn tensor_config(&self, id: TensorId) -> Result<&TensorSpec> {
        self.tensors
            .get(&id)
            .ok_or(Error::FrontendTensorNotFound(id))
    }

    pub fn tensor_config_mut(&mut self, id: TensorId) -> Result<&mut TensorSpec> {
        self.tensors
            .get_mut(&id)
            .ok_or(Error::FrontendTensorNotFound(id))
    }

    pub(crate) fn validate_tensor_data_type(
        &self,
        tensor: TensorId,
        expected: DataType,
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.data_type;
        if actual != expected {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensors_data_type<const N: usize>(
        &self,
        tensors: [(TensorId, &str); N],
        expected: DataType,
    ) -> Result<()> {
        for (tensor, operation) in tensors {
            self.validate_tensor_data_type(tensor, expected, operation)?;
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_data_type_and_dimensions(
        &self,
        tensor: TensorId,
        expected_data_type: DataType,
        expected_dimensions: &[i64],
        operation: &str,
    ) -> Result<()> {
        self.validate_tensor_data_type(tensor, expected_data_type, operation)?;
        self.validate_tensor_dimensions(tensor, expected_dimensions, operation)
    }

    pub(crate) fn validate_tensor_data_type_supported(
        &self,
        tensor: TensorId,
        supported: &[DataType],
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.data_type;
        if !supported.contains(&actual) {
            return Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id: tensor,
                operation: operation.into(),
                supported: supported.to_vec(),
                actual,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_dimensions(
        &self,
        tensor: TensorId,
        expected: &[i64],
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.dimensions();
        if actual != expected {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensors_dimensions<const N: usize>(
        &self,
        tensors: [(TensorId, &str); N],
        expected: &[i64],
    ) -> Result<()> {
        for (tensor, operation) in tensors {
            self.validate_tensor_dimensions(tensor, expected, operation)?;
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_strides(
        &self,
        tensor: TensorId,
        expected: &[i64],
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.strides();
        if actual != expected {
            return Err(Error::FrontendTensorStridesMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_rank(
        &self,
        tensor: TensorId,
        expected: usize,
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.dimensions().len();
        if actual != expected {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_min_rank(
        &self,
        tensor: TensorId,
        expected: usize,
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.dimensions().len();
        if actual < expected {
            return Err(Error::FrontendTensorRankMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: format!(">= {expected}"),
                actual,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_element_count(
        &self,
        tensor: TensorId,
        expected: usize,
        operation: &str,
    ) -> Result<()> {
        let actual = self.tensor_config(tensor)?.shape.element_count()?;
        if actual != expected {
            return Err(Error::FrontendTensorElementCountMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    pub fn modify_tensor(
        &mut self,
        id: TensorId,
        modify: impl FnOnce(TensorSpec) -> TensorSpec,
    ) -> Result<()> {
        let updated = modify(self.tensor_config(id)?.clone());
        self.replace_tensor(id, updated)
    }

    pub fn mark_as_virtual(&mut self, id: TensorId) -> Result<()> {
        self.modify_tensor(id, TensorSpec::virtual_tensor)
    }

    pub fn mark_as_output(&mut self, id: TensorId) -> Result<()> {
        self.modify_tensor(id, TensorSpec::output_tensor)
    }

    pub fn tensor_like(&mut self, id: TensorId) -> Result<TensorId> {
        self.tensor_like_with_name(id, None::<String>)
    }

    pub fn tensor_like_named(&mut self, id: TensorId, name: impl Into<String>) -> Result<TensorId> {
        self.tensor_like_with_name(id, Some(name))
    }

    pub fn tensor_like_with_name(
        &mut self,
        id: TensorId,
        name: Option<impl Into<String>>,
    ) -> Result<TensorId> {
        let mut cloned = self.tensor_config(id)?.clone();
        cloned.id = None;
        cloned.clear_name();
        if let Some(name) = name {
            cloned.set_name(name);
        }
        Ok(self.tensor(cloned))
    }

    /// Sets a ragged offset tensor for a tensor using packed variable-length layout.
    ///
    /// In SDPA THD/ragged layouts, the offset tensor contains cumulative token
    /// offsets in elements and lets cuDNN represent variable-length sequences
    /// without padding.
    pub fn set_ragged_offset(
        &mut self,
        tensor_ref: TensorId,
        ragged_offset: TensorId,
    ) -> Result<()> {
        self.tensor_config(ragged_offset)?;
        if tensor_ref == ragged_offset || self.ragged_offset_reaches(ragged_offset, tensor_ref)? {
            return Err(Error::FrontendRaggedOffsetCycle {
                tensor_id: tensor_ref,
                ragged_offset,
            });
        }
        let updated = self
            .tensor_config(tensor_ref)?
            .clone()
            .with_ragged_offset(ragged_offset);
        *self.tensor_config_mut(tensor_ref)? = updated;
        Ok(())
    }

    /// Clears the ragged offset tensor associated with a tensor.
    pub fn clear_ragged_offset(&mut self, tensor_ref: TensorId) -> Result<()> {
        let updated = self
            .tensor_config(tensor_ref)?
            .clone()
            .clear_ragged_offset();
        *self.tensor_config_mut(tensor_ref)? = updated;
        Ok(())
    }

    pub fn replace_tensor(&mut self, id: TensorId, mut new_tensor: TensorSpec) -> Result<()> {
        if let Some(constraints) = self.dynamic_shape_constraints.get(&id)
            && constraints.min_dimensions.len() != new_tensor.shape.dimensions().len()
        {
            return Err(Error::LengthMismatch {
                name: "dynamic_shape_constraints_rank".into(),
                expected: constraints.min_dimensions.len(),
                actual: new_tensor.shape.dimensions().len(),
            });
        }
        let slot = self
            .tensors
            .get_mut(&id)
            .ok_or(Error::FrontendTensorNotFound(id))?;
        new_tensor.id = Some(id);
        *slot = new_tensor;
        Ok(())
    }

    fn ragged_offset_reaches(&self, start: TensorId, target: TensorId) -> Result<bool> {
        let mut current = Some(start);
        let mut visited = HashSet::new();
        while let Some(tensor_id) = current {
            if !visited.insert(tensor_id) {
                return Ok(false);
            }
            if tensor_id == target {
                return Ok(true);
            }
            current = self.tensor_config(tensor_id)?.ragged_offset;
        }
        Ok(false)
    }

    pub fn shape(&self, id: TensorId) -> Result<&Shape> {
        Ok(&self.tensor_config(id)?.shape)
    }

    pub(crate) fn resolved_device_properties(&self) -> Result<Option<Arc<DeviceProperties>>> {
        if let Some(device_properties) = self.runtime.device_properties.as_ref() {
            return Ok(Some(Arc::clone(device_properties)));
        }

        self.device_properties_json()
            .map(DeviceProperties::from_json_representation)
            .transpose()
            .map(|device_properties| device_properties.map(Arc::new))
    }

    pub(crate) fn runtime_kernel_cache(&self) -> Option<&Arc<Mutex<KernelCache>>> {
        self.runtime.kernel_cache.as_ref()
    }

    fn check_id(&mut self, id: TensorId) -> Result<()> {
        if self.tensors.contains_key(&id) {
            return Err(Error::FrontendTensorIdConflict(id));
        }
        Ok(())
    }

    pub(crate) fn mutation_checkpoint(&self) -> GraphMutationCheckpoint {
        GraphMutationCheckpoint {
            tensor_ids: self.tensors.keys().copied().collect(),
            operation_count: self.operations.len(),
            alias_binding_count: self.alias_bindings.len(),
        }
    }

    pub(crate) fn rollback_to(&mut self, checkpoint: GraphMutationCheckpoint) {
        self.tensors
            .retain(|id, _| checkpoint.tensor_ids.contains(id));
        self.operations.truncate(checkpoint.operation_count);
        self.alias_bindings.truncate(checkpoint.alias_binding_count);
    }
}

fn validate_alias_byte_range(
    source_id: TensorId,
    source: &TensorSpec,
    target_id: TensorId,
    target: &TensorSpec,
    byte_offset: i64,
) -> Result<()> {
    let source_byte_len = tensor_byte_len(source)?;
    let target_byte_len = tensor_byte_len(target)?;
    let start =
        usize::try_from(byte_offset).map_err(|_| Error::FrontendAliasByteRangeOutOfBounds {
            source_id,
            target_id,
            start: 0,
            end: target_byte_len,
            source_byte_len,
        })?;
    let end = start.checked_add(target_byte_len).ok_or({
        Error::FrontendAliasByteRangeOutOfBounds {
            source_id,
            target_id,
            start,
            end: usize::MAX,
            source_byte_len,
        }
    })?;
    if end > source_byte_len {
        return Err(Error::FrontendAliasByteRangeOutOfBounds {
            source_id,
            target_id,
            start,
            end,
            source_byte_len,
        });
    }
    Ok(())
}

fn tensor_byte_len(tensor: &TensorSpec) -> Result<usize> {
    let element_size = usize::try_from(tensor.data_type.size()).map_err(|_| Error::OutOfRange {
        name: "tensor element size".into(),
    })?;
    tensor
        .shape
        .element_count()?
        .checked_mul(element_size)
        .ok_or(Error::OutOfRange {
            name: "tensor byte length".into(),
        })
}
