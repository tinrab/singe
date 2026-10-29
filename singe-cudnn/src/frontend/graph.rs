mod attention_stats;
mod block_scale;
mod captures;
mod compile;
mod config;
mod convolution;
mod data_type_policy;
mod engine;
mod genstats;
mod index;
mod initial_reads;
mod key;
mod layout;
mod matmul;
mod moe;
mod normalization;
mod output;
mod paged_cache;
mod pointwise;
mod prepared;
mod reduction;
mod requirements;
mod resample;
mod reshape;
mod rng;
mod rope;
mod serialization;
mod shape_constraints;
mod softmax;
mod state;
mod subgraphs;
mod validation;

use std::{
    collections::{HashSet, btree_map::Entry},
    mem,
    sync::{Arc, Mutex},
};

pub use crate::frontend::graph::{
    block_scale::{BlockScaleDequantizeInputs, BlockScaleQuantizeOutputs},
    config::GraphConfig,
    data_type_policy::DataTypePolicy,
    genstats::GenStatsOutputs,
    matmul::{
        BlockScaleMatmulInputs, MatmulFp8Inputs, MatmulFp8QuantizeOutputs,
        MatmulFp8QuantizeTensors, MatmulFp8Tensors,
    },
    moe::{
        MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors, MoeGroupedMatmulInputs,
        MoeGroupedMatmulTensors,
    },
    output::TensorOutputConfig,
    prepared::PreparedGraph,
    resample::{ResampleIndexOutput, ResampleOutputs},
    serialization::GraphKeyShapePolicy,
    shape_constraints::RuntimeShapeConstraints,
    softmax::SoftmaxOutputs,
    state::{Graph, StorageView},
    validation::ValidatedGraph,
};
pub(crate) use crate::frontend::graph::{
    captures::TensorCaptureTarget,
    layout::GraphLayout,
    normalization::{
        infer_adaptive_layer_normalization_stats_dimensions,
        infer_instance_normalization_stats_dimensions, infer_normalization_channel_dimensions,
        validate_normalization_broadcast_shape,
    },
    paged_cache::{infer_paged_cache_dimensions, infer_paged_cache_output},
    pointwise::infer_binary_pointwise_dimensions,
    softmax::reduce_last_axis,
    state::TensorRecord,
};
use crate::{
    backend::{
        device::DeviceProperties,
        execution::KernelCache,
        tensor::{Shape, TensorId, TensorPlacement, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::{Error as FrontendError, RankExpectation},
        graph::state::GraphMutationCheckpoint,
        operation::Operation,
    },
    utility::to_i64,
};

impl Graph {
    fn allocate_tensor_id(&mut self) -> TensorId {
        self.reserve_score_tensor_ids();
        let mut id = TensorId::generate();
        while self.tensors.contains_key(&id) || self.runtime.reserved_tensor_ids.contains(&id) {
            id = TensorId::generate();
        }
        id
    }

    /// Reserves newly attached score graphs, including recipes restored without runtime state.
    fn reserve_score_tensor_ids(&mut self) {
        for operation in &self.operations[self.runtime.indexed_operation_count..] {
            for subgraph in operation.score_subgraphs() {
                subgraph
                    .graph()
                    .collect_scope_tensor_ids(&mut self.runtime.reserved_tensor_ids);
            }
        }
        self.runtime.indexed_operation_count = self.operations.len();
    }

    /// Retains nested identities while operations are expanded outside the graph's list.
    pub(crate) fn take_operations(&mut self) -> Vec<Operation> {
        self.reserve_score_tensor_ids();
        self.runtime.indexed_operation_count = 0;
        mem::take(&mut self.operations)
    }

    pub(crate) fn replace_operations(&mut self, operations: Vec<Operation>) {
        self.operations = operations;
        self.runtime.indexed_operation_count = 0;
    }

    fn collect_scope_tensor_ids(&self, ids: &mut HashSet<TensorId>) {
        ids.extend(self.tensors.keys().copied());
        for operation in &self.operations {
            for subgraph in operation.score_subgraphs() {
                subgraph.graph().collect_scope_tensor_ids(ids);
            }
        }
    }

    /// Keeps generated subgraph identities distinct from their enclosing graph and its ancestors.
    pub(crate) fn new_subgraph(&self) -> Self {
        let mut subgraph = Self::new();
        self.collect_scope_tensor_ids(&mut subgraph.runtime.reserved_tensor_ids);
        subgraph
            .runtime
            .reserved_tensor_ids
            .extend(self.runtime.reserved_tensor_ids.iter().copied());
        subgraph
    }

    /// Inserts a tensor with a fresh graph identity.
    pub fn tensor(&mut self, tensor: TensorSpec) -> TensorId {
        let id = self.allocate_tensor_id();
        self.tensors.insert(id, tensor);
        self.runtime.tensor_insertions.push(id);
        id
    }

    /// Inserts a tensor with a caller-supplied identity, rejecting duplicates.
    /// Use [`TensorId::try_new`] to parse runtime integers before inserting them.
    ///
    /// # Errors
    /// Returns an error if this graph already contains the identity.
    pub fn insert_tensor(&mut self, id: TensorId, tensor: TensorSpec) -> Result<TensorId> {
        let Entry::Vacant(entry) = self.tensors.entry(id) else {
            return Err(FrontendError::TensorIdConflict(id).into());
        };
        entry.insert(tensor);
        self.runtime.tensor_insertions.push(id);
        Ok(id)
    }

    /// Creates an explicit byte-offset storage view without inserting a native operation.
    /// The source must have caller-supplied storage and cover the view's physical span.
    /// Ragged endpoints require [`TensorSpec::with_ragged_byte_capacity`] because their
    /// nominal dimensions do not determine the physical range.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte offset cannot fit `i64`, the source is unknown or lacks caller-supplied device storage, either declaration is invalid, the element encodings differ, the offset is not aligned to the scalar byte width, a required ragged capacity is absent, storage-size arithmetic fails, or the view exceeds the source byte or packed-bit range.
    pub fn storage_view(
        &mut self,
        source: TensorId,
        tensor: TensorSpec,
        byte_offset: u64,
    ) -> Result<TensorId> {
        let byte_offset = to_i64(byte_offset, "storage view byte offset")?;
        let target_id = self.allocate_tensor_id();
        let source_tensor = self.tensor_spec(source)?;
        if source_tensor.is_virtual() {
            return Err(FrontendError::StorageViewSourceNotBindable {
                source_id: source,
                reason: "source tensor is virtual",
            }
            .into());
        }
        if source_tensor.is_by_value() || source_tensor.is_workspace() {
            return Err(FrontendError::StorageViewSourceNotBindable {
                source_id: source,
                reason: "source tensor has no caller-supplied device storage",
            }
            .into());
        }
        validate_storage_view_byte_range(source, source_tensor, target_id, &tensor, byte_offset)?;
        let target = self.insert_tensor(target_id, tensor)?;
        self.storage_views.push(StorageView {
            source,
            target,
            byte_offset,
        });
        Ok(target)
    }

    /// Borrows the tensor declaration with the supplied graph identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor identity is absent from this graph.
    pub fn tensor_spec(&self, id: TensorId) -> Result<&TensorSpec> {
        self.tensors
            .get(&id)
            .ok_or(FrontendError::TensorNotFound(id).into())
    }

    /// Declared tensors in graph identity order, including unused declarations and virtual intermediates.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (TensorId, &TensorSpec)> {
        self.tensors.iter().map(|(&id, tensor)| (id, tensor))
    }

    /// Finds every tensor with this name; names need not be unique.
    pub fn tensors_named<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = (TensorId, &'a TensorSpec)> + 'a {
        self.tensors()
            .filter(move |(_, tensor)| tensor.name.as_deref() == Some(name))
    }

    /// Borrows declared operations in insertion order, before target-specific preparation and composite expansion.
    #[must_use]
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Explicit byte-offset storage views, separate from native reshape/slice/transpose operations.
    #[must_use]
    pub fn storage_views(&self) -> &[StorageView] {
        &self.storage_views
    }

    /// Mutably borrows a tensor declaration without checking replacement constraints.
    ///
    /// Use [`Self::replace_tensor`] or [`Self::modify_tensor`] to check recorded runtime shape bounds and ragged-offset relations before replacing a declaration.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor identity is absent from this graph.
    pub fn tensor_spec_mut(&mut self, id: TensorId) -> Result<&mut TensorSpec> {
        self.tensors
            .get_mut(&id)
            .ok_or(FrontendError::TensorNotFound(id).into())
    }

    pub(crate) fn validate_tensor_data_type(
        &self,
        tensor: TensorId,
        expected: DataType,
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.data_type;
        if actual != expected {
            return Err(FrontendError::TensorDataTypeMismatch {
                tensor_id: tensor,
                operation,
                expected,
                actual,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_data_type_and_dimensions(
        &self,
        tensor: TensorId,
        expected_data_type: DataType,
        expected_dimensions: &[i64],
        operation: &'static str,
    ) -> Result<()> {
        self.validate_tensor_data_type(tensor, expected_data_type, operation)?;
        self.validate_tensor_dimensions(tensor, expected_dimensions, operation)
    }

    pub(crate) fn validate_tensor_data_type_supported(
        &self,
        tensor: TensorId,
        supported: &[DataType],
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.data_type;
        if !supported.contains(&actual) {
            return Err(FrontendError::TensorDataTypeUnsupported {
                tensor_id: tensor,
                operation,
                supported: supported.to_vec(),
                actual,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_dimensions(
        &self,
        tensor: TensorId,
        expected: &[i64],
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.shape.dimensions();
        if actual != expected {
            return Err(FrontendError::TensorDimensionsMismatch {
                tensor_id: tensor,
                operation,
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensors_dimensions<const N: usize>(
        &self,
        tensors: [(TensorId, &'static str); N],
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
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.shape.strides();
        if actual != expected {
            return Err(FrontendError::TensorStridesMismatch {
                tensor_id: tensor,
                operation,
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_rank(
        &self,
        tensor: TensorId,
        expected: usize,
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.shape.rank();
        if actual != expected {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: tensor,
                operation,
                expected: RankExpectation::Exact(expected),
                actual,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_min_rank(
        &self,
        tensor: TensorId,
        expected: usize,
        operation: &'static str,
    ) -> Result<()> {
        let actual = self.tensor_spec(tensor)?.shape.rank();
        if actual < expected {
            return Err(FrontendError::TensorRankMismatch {
                tensor_id: tensor,
                operation,
                expected: RankExpectation::AtLeast(expected),
                actual,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn validate_tensor_scalar_shape(
        &self,
        tensor: TensorId,
        operation: &'static str,
    ) -> Result<()> {
        if !self.tensor_spec(tensor)?.has_scalar_shape()? {
            return Err(FrontendError::TensorScalarShapeMismatch {
                tensor_id: tensor,
                operation,
            }
            .into());
        }
        Ok(())
    }

    /// Replaces a tensor declaration using a closure applied to a cloned specification.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is unknown, its recorded runtime shape bounds are incompatible with the replacement, or a ragged-offset reference is unknown or creates a cycle. The original declaration remains unchanged on error.
    pub fn modify_tensor(
        &mut self,
        id: TensorId,
        modify: impl FnOnce(TensorSpec) -> TensorSpec,
    ) -> Result<()> {
        let updated = modify(self.tensor_spec(id)?.clone());
        self.replace_tensor(id, updated)
    }

    /// Changes the tensor placement to a cuDNN-managed virtual intermediate.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is unknown, its recorded runtime shape bounds are incompatible with the replacement, or a ragged-offset reference is unknown or creates a cycle. The original declaration remains unchanged on error.
    pub fn mark_as_virtual(&mut self, id: TensorId) -> Result<()> {
        self.modify_tensor(id, TensorSpec::virtual_tensor)
    }

    /// Makes a virtual or workspace tensor bindable as a device output and clears its workspace initializer.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is unknown, its recorded runtime shape bounds are incompatible with the replacement, or a ragged-offset reference is unknown or creates a cycle. The original declaration remains unchanged on error.
    pub fn mark_as_output(&mut self, id: TensorId) -> Result<()> {
        self.modify_tensor(id, TensorSpec::output_tensor)
    }

    /// Copies a tensor declaration under a fresh identity, matching C++ `tensor_like`.
    /// Replaces its name with `name`, or clears the name when omitted.
    /// Retains placement, layout, encoding, scalar defaults, workspace initializers,
    /// ragged-offset references and recorded runtime shape bounds.
    /// This does not alias device storage or copy the source's producer operations.
    ///
    /// # Errors
    /// Returns an error if the source is unknown or its runtime shape bounds no
    /// longer match its declaration.
    pub fn tensor_like(&mut self, id: TensorId, name: Option<&str>) -> Result<TensorId> {
        let mut cloned = self.tensor_spec(id)?.clone();
        cloned.clear_name();
        if let Some(name) = name {
            cloned.set_name(name);
        }
        let constraints = self.runtime_shape_constraints.get(&id).cloned();
        if let Some(constraints) = &constraints {
            constraints.validate_for(&cloned)?;
        }
        let cloned_id = self.tensor(cloned);
        if let Some(constraints) = constraints {
            self.runtime_shape_constraints
                .insert(cloned_id, constraints);
        }
        Ok(cloned_id)
    }

    /// Sets a ragged offset tensor for a tensor using packed variable-length layout.
    ///
    /// In SDPA THD/ragged layouts, the offset tensor contains cumulative token
    /// offsets in elements and lets cuDNN represent variable-length sequences
    /// without padding. A nonunit [`TensorSpec::with_ragged_offset_multiplier`]
    /// changes the number of elements represented by each offset unit.
    ///
    /// # Errors
    ///
    /// Returns an error if either identity or a referenced offset tensor is unknown, or the new relation reaches a ragged-offset cycle. The existing relation remains unchanged on error.
    pub fn set_ragged_offset(
        &mut self,
        tensor_id: TensorId,
        ragged_offset: TensorId,
    ) -> Result<()> {
        self.validate_ragged_offset_relation(tensor_id, ragged_offset)?;
        self.tensor_spec_mut(tensor_id)?.ragged_offset = Some(ragged_offset);
        Ok(())
    }

    /// Restores dense storage by clearing ragged offsets, physical capacity and the offset-unit multiplier.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor identity is absent from this graph.
    pub fn clear_ragged_offset(&mut self, tensor_id: TensorId) -> Result<()> {
        self.tensor_spec_mut(tensor_id)?.clear_ragged_offset();
        Ok(())
    }

    /// Replaces a declaration while checking recorded runtime shape bounds and ragged-offset relations.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is unknown, its recorded runtime shape bounds are incompatible with the replacement, or a ragged-offset reference is unknown or creates a cycle. The original declaration remains unchanged on error.
    pub fn replace_tensor(&mut self, id: TensorId, new_tensor: TensorSpec) -> Result<()> {
        if let Some(constraints) = self.runtime_shape_constraints.get(&id) {
            constraints.validate_for(&new_tensor)?;
        }
        if let Some(offset) = new_tensor.ragged_offset {
            self.validate_ragged_offset_relation(id, offset)?;
        }
        let slot = self
            .tensors
            .get_mut(&id)
            .ok_or(FrontendError::TensorNotFound(id))?;
        *slot = new_tensor;
        Ok(())
    }

    fn validate_ragged_offset_relation(
        &self,
        tensor: TensorId,
        ragged_offset: TensorId,
    ) -> Result<()> {
        self.tensor_spec(tensor)?;
        let mut current = Some(ragged_offset);
        let mut visited = HashSet::new();
        while let Some(tensor_id) = current {
            if tensor_id == tensor {
                return Err(FrontendError::RaggedOffsetCycle { tensor_id: tensor }.into());
            }
            if !visited.insert(tensor_id) {
                return Err(FrontendError::RaggedOffsetCycle { tensor_id }.into());
            }
            current = self.tensor_spec(tensor_id)?.ragged_offset;
        }
        Ok(())
    }

    /// Borrows the tensor's checked dimensions and element strides.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor identity is absent from this graph.
    pub fn shape(&self, id: TensorId) -> Result<&Shape> {
        Ok(&self.tensor_spec(id)?.shape)
    }

    pub(crate) fn resolved_device_properties(&self) -> Result<Option<Arc<DeviceProperties>>> {
        if let Some(device_properties) = self.runtime.device_properties.as_ref() {
            return Ok(Some(Arc::clone(device_properties)));
        }

        self.device_properties_json()
            .map(DeviceProperties::deserialize)
            .transpose()
            .map(|device_properties| device_properties.map(Arc::new))
    }

    pub(crate) fn runtime_kernel_cache(&self) -> Option<&Arc<Mutex<KernelCache>>> {
        self.runtime.kernel_cache.as_ref()
    }

    pub(crate) fn mutation_checkpoint(&self) -> GraphMutationCheckpoint {
        GraphMutationCheckpoint {
            tensor_insertions: self.runtime.tensor_insertions.len(),
            operations: self.operations.len(),
            storage_views: self.storage_views.len(),
        }
    }

    pub(crate) fn rollback_to(&mut self, checkpoint: GraphMutationCheckpoint) {
        for id in self
            .runtime
            .tensor_insertions
            .drain(checkpoint.tensor_insertions..)
        {
            self.tensors.remove(&id);
            self.runtime_shape_constraints.remove(&id);
            self.vector_aligned_tensors.remove(&id);
        }
        self.operations.truncate(checkpoint.operations);
        self.runtime.indexed_operation_count = self
            .runtime
            .indexed_operation_count
            .min(self.operations.len());
        self.storage_views.truncate(checkpoint.storage_views);
    }
}

pub(crate) fn validate_storage_view_byte_range(
    source_id: TensorId,
    source: &TensorSpec,
    target_id: TensorId,
    target: &TensorSpec,
    byte_offset: i64,
) -> Result<()> {
    source.validate()?;
    target.validate()?;
    if !matches!(source.placement, TensorPlacement::Device)
        || !matches!(target.placement, TensorPlacement::Device)
    {
        return Err(FrontendError::StorageViewPlacementMismatch {
            source_id,
            target_id,
        }
        .into());
    }
    if source.data_type != target.data_type {
        return Err(FrontendError::TensorDataTypeMismatch {
            tensor_id: target_id,
            operation: "storage view target",
            expected: source.data_type,
            actual: target.data_type,
        }
        .into());
    }
    let element_size = target.data_type.size();
    if byte_offset % element_size != 0 {
        return Err(FrontendError::StorageViewByteOffsetUnaligned {
            source_id,
            target_id,
            byte_offset,
            element_size,
        }
        .into());
    }
    let source_byte_len = source.storage_byte_len(None)?;
    let target_byte_len = target.storage_byte_len(None)?;
    let start = usize::try_from(byte_offset).map_err(|_| {
        FrontendError::StorageViewByteRangeOutOfBounds {
            source_id,
            target_id,
            start: 0,
            end: target_byte_len,
            source_byte_len,
        }
    })?;
    let end = start.checked_add(target_byte_len).ok_or(
        FrontendError::StorageViewByteRangeOutOfBounds {
            source_id,
            target_id,
            start,
            end: usize::MAX,
            source_byte_len,
        },
    )?;
    if end > source_byte_len {
        return Err(FrontendError::StorageViewByteRangeOutOfBounds {
            source_id,
            target_id,
            start,
            end,
            source_byte_len,
        }
        .into());
    }
    let source_bits = source.storage_bit_len()?;
    let start_bits = start as u128 * 8;
    let end_bits = start_bits
        .checked_add(target.storage_bit_len()?)
        .ok_or_else(|| Error::out_of_range("storage view bits"))?;
    if end_bits > source_bits {
        return Err(FrontendError::StorageViewBitRangeOutOfBounds {
            source_id,
            target_id,
            start_bits,
            end_bits,
            source_bits,
        }
        .into());
    }
    Ok(())
}
